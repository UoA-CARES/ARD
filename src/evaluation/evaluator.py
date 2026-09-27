"""
Local evaluation orchestrator.

``RewardEvaluator`` is the high-level entry point ARD's refinement loop calls.
Its sole responsibility is **dispatch + capture** — running candidates and
collecting their output. For a batch of :class:`~src.reward_history.RewardRecord`
(each carrying a proposed ``compute_reward`` method) it walks them one at a time:

1. Builds the candidate's job codebase (pristine ard-isaaclab-tasks repo + the
   proposed reward spliced in) — :class:`WorkspaceManager`.
2. Builds + runs it on the local machine, blocking until it finishes —
   :class:`LocalRunner`.
3. Reads the run's logs in place (from its work dir) and captures its run paths +
   scalar summary — :class:`ResultProcessor`.

It writes job status and captured artifact paths back onto each record but does
**not** read fitness or pick a winner — that judgement is
:class:`~src.evaluation.scorer.FitnessScorer`'s job. This keeps the evaluator a
pure executor and leaves scoring a separate, swappable step.
"""

import os
import time
import shlex
import shutil
import pickle
import logging
import zipfile
import collections
from typing import Dict, List, Optional

from .local_runner import LocalRunner
from .hpc_runner import HPCRunner, HPCRunnerError, HPCJob
from .workspace_manager import WorkspaceManager
from .reward_injection import RewardInjectionError
from .result_processor import ResultProcessor, CHECKPOINT_SEL_BEST
from . import config
from src.reward_history import (
    RewardRecord,
    STATUS_GEN_FAILED,
    STATUS_BUILD_FAILED,
    STATUS_SUBMITTED,
    STATUS_NO_METRICS,
)

logger = logging.getLogger(__name__)


class WarmStartError(RuntimeError):
    """A warm-start checkpoint could not be used as specified."""


# Classes the checkpoint reader below will reconstruct for real. Everything else
# in the pickle — torch tensors, storages, custom classes — becomes `_Opaque`.
# The containers have to be genuine: the unpickler fills them with SETITEM /
# APPEND opcodes, which a stub can't accept ("does not support item assignment").
_SAFE_CLASSES = {
    ("collections", "OrderedDict"): collections.OrderedDict,
    ("builtins", "dict"): dict,
    ("builtins", "list"): list,
    ("builtins", "tuple"): tuple,
    ("builtins", "set"): set,
    ("builtins", "frozenset"): frozenset,
    ("builtins", "int"): int,
    ("builtins", "float"): float,
    ("builtins", "complex"): complex,
    ("builtins", "bool"): bool,
    ("builtins", "str"): str,
    ("builtins", "bytes"): bytes,
    ("builtins", "object"): object,
}


class _Opaque:
    """Stand-in for any object the checkpoint reader declines to build."""

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        pass


class _StubUnpickler(pickle.Unpickler):
    """Unpickler that recovers a checkpoint's plain scalars and nothing else.

    ``find_class`` never imports anything, so no code from the checkpoint can
    execute — strictly safer than the ``torch.load(weights_only=False)`` this
    replaced. ``persistent_load`` is how torch routes tensor storages; returning
    a placeholder means the archive's multi-megabyte data entries are never read.
    """

    def find_class(self, module, name):
        return _SAFE_CLASSES.get((module, name), _Opaque)

    def persistent_load(self, pid):
        return _Opaque()


class RewardEvaluator:
    """
    Orchestrates local evaluation of reward candidates.

    Args:
        tasks_repo: Path to the ard-isaaclab-tasks checkout.
        env_file_rel: Task env file (relative to ``tasks_repo``) to inject into.
        task: Registered task ID, e.g. ``Isaac-ARD-Cartpole-v0``.
        runner: Dict of backend settings. ``backend`` selects the execution path:
            * ``local`` (default): use_gpu, timeout_seconds, image, env (extra
              container env passed to every job), build_args, and an optional
              command_template override. Jobs build + ``docker run`` one at a time.
            * ``hpc``: an ``hpc`` sub-dict (registry, upi, nas_outputs,
              max_runtime_hours, datasets, poll_seconds, job_name_prefix,
              extra_args). Each candidate is built + pushed as its own image and
              submitted to the CARES HPC Scheduler; the batch trains concurrently.
              ``env`` MAX_ITERATIONS/NUM_ENVS are translated to command flags
              because the scheduler drops the job env block.
        output_dir: Where each candidate's job runs and its logs land
            (``<output_dir>/<tag>/``). For the hpc backend this is where NAS
            artifacts are recycled to.
        build_root: Optional staging dir for codebase tarballs.
        checkpoint_sel_mode: Which of a finished run's ``nn/*.pth`` files is
            recorded as its checkpoint, and so is carried into the next
            iteration when warm-starting. See
            :meth:`ResultProcessor.find_checkpoint`.
    """

    def __init__(
        self,
        tasks_repo: str,
        env_file_rel: str,
        task: str,
        runner: Dict,
        output_dir: str,
        build_root: Optional[str] = None,
        warm_start: Optional[Dict] = None,
        checkpoint_sel_mode: str = CHECKPOINT_SEL_BEST,
    ):
        self.task = task
        self.output_dir = os.path.abspath(os.path.expanduser(output_dir))
        os.makedirs(self.output_dir, exist_ok=True)

        # Per-job parameters. Each job builds the project's Dockerfile (no prebuilt
        # image tag); the task image's entrypoint is driven by the job `env`, so
        # the task/seed are passed there rather than as a command.
        self.use_gpu = bool(runner.get("use_gpu", config.DEFAULT_USE_GPU))
        self.timeout_seconds = int(
            runner.get("timeout_seconds", config.DEFAULT_TRAINING_TIMEOUT)
        )
        # Extra container env applied to every job (e.g. MAX_ITERATIONS, NUM_ENVS,
        # WANDB_*), and optional docker build args.
        self.env_extra = dict(runner.get("env", {}))
        self.build_args = dict(runner.get("build_args", {}))
        # Optional override of the image CMD. Default None -> the image's own
        # entrypoint runs, configured entirely through `env`.
        self.command_template = runner.get("command_template")

        self.backend = str(runner.get("backend", config.DEFAULT_BACKEND)).lower()

        self.workspace = WorkspaceManager(
            tasks_repo=tasks_repo,
            env_file_rel=env_file_rel,
            build_root=build_root,
        )
        self.processor = ResultProcessor(checkpoint_sel_mode=checkpoint_sel_mode)
        # Memoised checkpoint -> epoch reads. One warm-start checkpoint is shared
        # by every candidate in a batch, so this saves reopening the archive once
        # per record.
        self._epoch_cache: Dict[str, int] = {}

        # refineconfig's `warm_start:` block. Its `enabled` key is not used
        # here - main.py already acted on it by deciding whether to pass a
        # checkpoint at all, and a job only sees these settings when it is
        # given one. Translated into scripts/train.py flags by
        # _warm_start_flags below.
        self.warm_start_cfg = dict(warm_start or {})
        self._validate_critic_warmup(self.warm_start_cfg)
        self._validate_plasticity_injection()

        if self.backend == "hpc":
            # Build + push each candidate's image, submit to the CARES scheduler,
            # then recycle NAS artifacts. Jobs train concurrently on the cluster.
            hpc = dict(runner.get("hpc", {}))
            self.runner = HPCRunner(
                registry=hpc.get("registry", config.DEFAULT_HPC_REGISTRY),
                # Per-user image repo. `upi` (or $ARD_UPI) is the supported knob;
                # `image_repo` stays as a raw override for a non-standard name.
                image_repo=hpc.get("image_repo")
                or config.hpc_image_repo(str(hpc.get("upi", "") or "")),
                nas_outputs=hpc.get("nas_outputs", config.DEFAULT_HPC_NAS_OUTPUTS),
                max_active_jobs=int(
                    hpc.get("max_active_jobs", config.DEFAULT_HPC_MAX_ACTIVE_JOBS)
                ),
            )
            self.max_runtime_hours = float(
                hpc.get("max_runtime_hours", config.DEFAULT_HPC_MAX_RUNTIME_HOURS)
            )
            self.datasets = list(hpc.get("datasets", []))
            self.poll_seconds = float(
                hpc.get("poll_seconds", config.DEFAULT_HPC_POLL_SECONDS)
            )
            self.job_name_prefix = hpc.get(
                "job_name_prefix", config.DEFAULT_HPC_JOB_NAME_PREFIX
            )
            self.hpc_extra_args = str(hpc.get("extra_args", "") or "")
            # A job that ends without writing results back never judged its
            # reward, so it is re-run rather than scored as a failure.
            self.result_grace_seconds = float(
                hpc.get("result_grace_seconds", config.DEFAULT_HPC_RESULT_GRACE_SECONDS)
            )
            self.max_retries = int(
                hpc.get("max_retries", config.DEFAULT_HPC_MAX_RETRIES)
            )
            health = "reachable" if self.runner.healthz() else "UNREACHABLE"
            logger.info(
                f"RewardEvaluator ready: task={task} backend=hpc "
                f"({self.runner.registry}/{self.runner.image_repo}) "
                f"max_runtime={self.max_runtime_hours}h scheduler={health} "
                f"(build+push per candidate, submit-all, monitor concurrently)"
            )
        else:
            # Build + run each job's Dockerfile on the local machine, one at a time.
            self.runner = LocalRunner(
                image=runner.get("image", "ard-local"),
                use_gpu=self.use_gpu,
            )
            if not self.runner.healthz():
                logger.warning(
                    "local docker did not pass health check; runs may fail."
                )
            logger.info(
                f"RewardEvaluator ready: task={task} backend=local docker "
                f"gpu={'on' if self.use_gpu else 'off'} timeout={self.timeout_seconds}s "
                f"(deploy-by-Dockerfile, env-driven)"
            )

    # ------------------------------------------------------------------ prompt
    def get_reward_template(self) -> str:
        """Pristine ``compute_reward`` source, for seeding the LLM prompt."""
        return self.workspace.get_reward_template()

    def get_env_source(self) -> str:
        """Full pristine task env source, for LLM task context."""
        return self.workspace.get_env_source()

    # ---------------------------------------------------------------- evaluate
    @staticmethod
    def _checkpoint_epoch(checkpoint_path: str) -> int:
        """Read the epoch count rl_games saved inside a checkpoint.

        rl_games checkpoints bundle their own epoch counter alongside the
        weights (``a2c_common.py``: ``state['epoch'] = self.epoch_num`` on
        save, ``self.epoch_num = weights['epoch']`` on load) — loading a
        checkpoint resumes the epoch count, not just the network.

        Deliberately avoids ``torch.load``. torch is not an ARD dependency —
        training runs inside the per-job Isaac Lab image, see requirements.txt —
        and the earlier version of this that imported it lazily raised
        ImportError on every single call, silently disabling the warm-start
        budget extension below for a whole 5-iteration run. A ``torch.save``
        file is just a zip archive whose ``*/data.pkl`` entry holds the state
        dict's structure, with tensor payloads in separate entries reached via
        ``persistent_load``, so the scalars are readable with the stdlib alone
        and without materialising a single weight.

        Raises:
            WarmStartError: if the epoch can't be read. Never returns a
                fallback — quietly submitting an unextended budget under-trains
                every warm-started job in the batch, which is precisely the
                failure this method exists to prevent.
        """
        try:
            if not os.path.isfile(checkpoint_path):
                raise WarmStartError(f"{checkpoint_path} does not exist")
            if not zipfile.is_zipfile(checkpoint_path):
                raise WarmStartError(
                    f"{checkpoint_path} is not a torch checkpoint archive"
                )
            with zipfile.ZipFile(checkpoint_path) as archive:
                entries = [n for n in archive.namelist() if n.endswith("data.pkl")]
                if not entries:
                    raise WarmStartError(f"{checkpoint_path} contains no data.pkl")
                with archive.open(entries[0]) as handle:
                    state = _StubUnpickler(handle).load()
        except WarmStartError:
            raise
        except Exception as e:  # noqa: BLE001 - any read/unpickle failure
            raise WarmStartError(f"could not read {checkpoint_path}: {e}") from e

        # Insist on the key rather than defaulting to 0: a missing 'epoch' would
        # extend the ceiling by nothing and reproduce the original bug silently.
        if not isinstance(state, dict) or "epoch" not in state:
            found = sorted(state)[:10] if isinstance(state, dict) else type(state).__name__
            raise WarmStartError(
                f"{checkpoint_path} has no 'epoch' key (found: {found})"
            )
        return int(state["epoch"])

    def _checkpoint_epoch_cached(self, checkpoint_path: str) -> int:
        """``_checkpoint_epoch`` memoised for the life of this evaluator."""
        if checkpoint_path not in self._epoch_cache:
            self._epoch_cache[checkpoint_path] = self._checkpoint_epoch(checkpoint_path)
        return self._epoch_cache[checkpoint_path]

    def _effective_max_iterations(self, checkpoint_path: Optional[str]) -> Optional[str]:
        """The ``--max_iterations`` value for one job, accounting for warm-start.

        Still correct under fine-grained warm start: a transfer deliberately
        inherits the checkpoint's epoch counter (only the optimizer, the lr
        schedule and the normalizers are selectable), so the arithmetic below
        is unchanged.

        Without this, warm-starting from a checkpoint that already reached the
        configured ``MAX_ITERATIONS`` immediately hits that same ceiling on
        load (rl_games restores the checkpoint's own epoch count) and stops
        after ~0 additional epochs. Extending the ceiling by the checkpoint's
        inherited epoch count guarantees every candidate still gets the full
        configured budget of *new* training on top of what it resumed from.
        Returns None if no ``MAX_ITERATIONS`` is configured at all (the task's
        own ``max_epochs`` default applies, unaffected by this).

        Raises:
            WarmStartError: propagated from :meth:`_checkpoint_epoch` when a
                warm-start checkpoint's epoch can't be read. ``evaluate`` calls
                this once up front so that surfaces before anything is built.
        """
        configured = self.env_extra.get("MAX_ITERATIONS")
        if configured is None:
            return None
        if not checkpoint_path:
            return str(configured)
        return str(self._checkpoint_epoch_cached(checkpoint_path) + int(configured))

    def _plasticity_enabled(self) -> bool:
        """Whether ``--plasticity`` should be appended to this job's training command.

        Read from ``runner.env`` (either ``plasticity`` or ``PLASTICITY``, so a
        plain ``plasticity: true`` in settings.yaml just works). This can't ride
        as a plain passthrough env var like ``MAX_ITERATIONS``/``NUM_ENVS`` — the
        image entrypoint (``pcs_entrypoint.sh``) doesn't read a dedicated
        variable for it, so it has to go through the same ``EXTRA_ARGS``/command
        translation as ``--checkpoint``.
        """
        return bool(self.env_extra.get("plasticity", self.env_extra.get("PLASTICITY", False)))

    # runner.env keys that are translated into train.py flags rather than passed
    # through to the container env verbatim.
    _PLASTICITY_ENV_KEYS = (
        "plasticity",
        "PLASTICITY",
        "plasticity_injection_strategy",
        "PLASTICITY_INJECTION_STRATEGY",
    )

    def _plasticity_injection_strategy(self) -> Optional[str]:
        """The ``--plasticity_injection_strategy`` value for this job, or None.

        Read from ``runner.env`` like :meth:`_plasticity_enabled`. Any value
        turns on rl_games neuron replacement with that strategy (train.py sets
        ``plasticity.replacement_enabled`` / ``replacement_strategy``); empty or
        null leaves replacement off. ``plasticity`` itself only enables the
        diagnostics, never replacement.
        """
        value = self.env_extra.get(
            "plasticity_injection_strategy",
            self.env_extra.get("PLASTICITY_INJECTION_STRATEGY"),
        )
        return str(value) if value else None

    def _validate_plasticity_injection(self) -> None:
        """Reject an injection strategy without plasticity before any job is built.

        rl_games only replaces units inside an enabled plasticity manager, and
        train.py refuses the combination - but only once the job is running in
        its container, after an image build and a scheduler round trip.
        """
        if self._plasticity_injection_strategy() and not self._plasticity_enabled():
            raise ValueError(
                "runner.env.plasticity_injection_strategy "
                f"{self._plasticity_injection_strategy()!r} requires runner.env.plasticity: true"
            )

    # The refineconfig warm-start keys that map 1:1 onto scripts/train.py flags.
    # `enabled` is not among them: it is expressed by --warm_start itself.
    _WARM_START_BOOL_KEYS = (
        "reset_optimizer",
        "reset_lr_schedule",
        "reset_obs_normalizer",
        "reset_value_normalizer",
    )

    @staticmethod
    def _validate_critic_warmup(cfg: dict) -> None:
        """Reject a bad critic_warmup_epoch_count before any job is built.

        rl_games validates this too, but only once the job is running inside its
        container - which surfaces to ARD as a generic job failure, after an
        image build and a scheduler round trip. Checking it at construction
        fails the whole run immediately, naming the key.
        """
        value = cfg.get("critic_warmup_epoch_count")
        if value is None:
            return
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                "refineconfig warm_start.critic_warmup_epoch_count must be an integer, "
                f"got {value!r}"
            )
        if value < 0:
            raise ValueError(
                "refineconfig warm_start.critic_warmup_epoch_count must be >= 0, "
                f"got {value}"
            )

    def _warm_start_flags(self) -> List[str]:
        """``scripts/train.py`` flags for the configured warm-start settings.

        One builder for both backends so the local (`EXTRA_ARGS`) and hpc (argv)
        paths cannot drift apart. Every configured value is emitted explicitly
        rather than relying on train.py's or rl_games' defaults, so what a job
        actually does is decided by refineconfig.yaml and is visible in the
        job's own command line.

        Callers only invoke this when a checkpoint is being delivered, so it
        never emits --warm_start for a cold-start job.
        """
        flags = ["--warm_start"]
        for key in self._WARM_START_BOOL_KEYS:
            value = self.warm_start_cfg.get(key)
            if value is not None:
                flags.append(f"--warm_start_{key} {'true' if value else 'false'}")
        critic_warmup = self.warm_start_cfg.get("critic_warmup_epoch_count")
        if critic_warmup is not None:
            flags.append(f"--critic_warmup_epoch_count {int(critic_warmup)}")
        return flags

    def _build_env(
        self, seed: Optional[int], checkpoint_path: Optional[str] = None
    ) -> Dict[str, str]:
        """Container env for one job: the task, its seed, and any configured extras.

        The ard-isaaclab-tasks image entrypoint (``scripts/pcs_entrypoint.sh``)
        reads ``TASK`` and ``SEED`` (plus optional ``MAX_ITERATIONS``/``NUM_ENVS``/
        ``WANDB_*``) from the env, so the per-eval seed is now honoured (the old
        quickstart command path ignored it). It has no dedicated checkpoint env
        var, so warm-start and plasticity both ride its ``EXTRA_ARGS`` catch-all
        instead (verbatim extra flags appended to ``train.py``) as
        ``--checkpoint <path>`` and ``--plasticity`` respectively. The
        checkpoint flag points at the fixed in-image path the checkpoint was
        baked to by ``WorkspaceManager.build_codebase``
        (``config.WARM_START_CHECKPOINT_REL`` under ``config.IMAGE_REPO_ROOT``)
        — the host-side ``checkpoint_path`` here only decides *whether* one was
        baked in, not the path used. A checkpoint additionally brings the
        ``--warm_start`` flags from :meth:`_warm_start_flags`, which tell
        rl_games to apply it as a transfer rather than as a resume. All are
        appended after any user-configured ``EXTRA_ARGS`` from ``runner.env``,
        as is ``--plasticity_injection_strategy`` when one is configured.
        """
        env = {"TASK": self.task}
        if seed is not None:
            env["SEED"] = str(seed)
        env.update({
            k: str(v) for k, v in self.env_extra.items()
            if k not in self._PLASTICITY_ENV_KEYS
        })
        max_iterations = self._effective_max_iterations(checkpoint_path)
        if max_iterations is not None:
            env["MAX_ITERATIONS"] = max_iterations

        extra_flags = []
        if checkpoint_path:
            extra_flags.append(f"--checkpoint {config.IMAGE_REPO_ROOT}/{config.WARM_START_CHECKPOINT_REL}")
            extra_flags.extend(self._warm_start_flags())
        if self._plasticity_enabled():
            extra_flags.append("--plasticity")
        strategy = self._plasticity_injection_strategy()
        if strategy:
            extra_flags.append(f"--plasticity_injection_strategy {strategy}")
        if extra_flags:
            env["EXTRA_ARGS"] = f"{env.get('EXTRA_ARGS', '')} {' '.join(extra_flags)}".strip()
        return env

    def _build_command(self, seed: Optional[int]) -> Optional[str]:
        """Optional CMD override; None means run the image's own entrypoint."""
        if not self.command_template:
            return None
        return self.command_template.format(
            task=self.task,
            seed="" if seed is None else seed,
        )

    def _build_hpc_command(
        self, seed: Optional[int], checkpoint_path: Optional[str] = None
    ) -> str:
        """Build the ``hpc_entrypoint.sh`` command for one job.

        The CARES scheduler drops the job ``env`` block, so task/seed/tunables
        must ride the ``command`` (unlike the local backend, which passes them as
        env). ``MAX_ITERATIONS`` / ``NUM_ENVS`` from ``runner.env`` are therefore
        translated into ``--max_iterations`` / ``--num_envs`` flags. (WANDB_* are
        intentionally not forwarded: they cannot reach the container via command
        without exposing the key in ``hpc-client jobs``.) ``checkpoint_path``
        (warm-start) rides the same way, as ``--checkpoint`` plus the
        ``--warm_start`` flags from :meth:`_warm_start_flags`, matching
        ``scripts/train.py``'s own flags — pointed at the fixed in-image path
        the checkpoint was baked to (see ``_build_env``), not the host path.
        ``plasticity`` from ``runner.env`` is likewise appended as ``--plasticity``,
        and ``plasticity_injection_strategy`` as ``--plasticity_injection_strategy``.
        """
        flags = ["--task", self.task]
        if seed is not None:
            flags += ["--seed", str(seed)]
        if checkpoint_path:
            flags += ["--checkpoint", f"{config.IMAGE_REPO_ROOT}/{config.WARM_START_CHECKPOINT_REL}"]
            # Same flags as the local path, re-split because this backend passes
            # argv rather than one EXTRA_ARGS string.
            flags += [part for flag in self._warm_start_flags() for part in flag.split()]
        max_iterations = self._effective_max_iterations(checkpoint_path)
        if max_iterations is not None:
            flags += ["--max_iterations", max_iterations]
        if self.env_extra.get("NUM_ENVS") is not None:
            flags += ["--num_envs", str(self.env_extra["NUM_ENVS"])]
        if self._plasticity_enabled():
            flags += ["--plasticity"]
        strategy = self._plasticity_injection_strategy()
        if strategy:
            flags += ["--plasticity_injection_strategy", strategy]

        extra = self.hpc_extra_args
        # Vision tasks instantiate a TiledCamera, which train.py only allows with
        # --enable_cameras; add it here so the job doesn't fail minutes in on a
        # worker (mirrors ard-isaaclab-tasks scripts/hpc_submit.py).
        if "Vision" in self.task and "--enable_cameras" not in extra:
            extra = f"{extra} --enable_cameras".strip()

        command = config.HPC_ENTRYPOINT + " " + " ".join(shlex.quote(f) for f in flags)
        if extra:
            command += " " + extra
        return command

    def evaluate(
        self,
        records: List[RewardRecord],
        checkpoint_path: Optional[str] = None,
    ) -> List[RewardRecord]:
        """
        Train a batch of candidate records and capture their output.

        Dispatches to the configured backend: the ``local`` backend builds +
        ``docker run``s each candidate one at a time; the ``hpc`` backend builds +
        pushes each candidate's image, submits the whole batch to the CARES
        scheduler, then recycles NAS artifacts as jobs finish. Either way this
        mutates each record in place: it sets ``status``, ``eval_error`` and (on
        success) the captured ``log_path`` / ``tb_path`` / ``summary_path`` /
        ``checkpoint_path``. Fitness and best-selection are left to
        :class:`FitnessScorer`. Training length is controlled by each task's
        ``max_epochs`` in its ``rl_games_ppo_cfg.yaml``.

        Args:
                records: Candidate records. Each must carry ``reward_method`` (records
                whose generation failed are skipped) and provides ``tag`` / ``seed``.
                checkpoint_path: Warm-start checkpoint (e.g. the previous iteration's
                best candidate) to resume every job in this batch from. None
                means every job trains from scratch.

        Returns:
            The same ``records`` list, mutated in place.
        """
        if not records:
            logger.error("No records provided for evaluation")
            return records
        if not self.workspace.validate():
            logger.error("Workspace validation failed")
            for record in records:
                if record.has_method:
                    record.status = STATUS_BUILD_FAILED
                    record.eval_error = "workspace validation failed"
            return records

        # Resolve the warm-start-adjusted epoch ceiling once, before anything is
        # staged, built or pushed. `checkpoint_path` is batch-wide, so failing to
        # read it is a batch-level condition, not a per-candidate one — and
        # discovering it from inside _build_env/_build_hpc_command would strand
        # images already pushed for earlier records. The log line is the
        # observability this previously lacked: an unextended ceiling on a
        # warm-started batch is now visible at submit time rather than only in a
        # training log that stops after a handful of epochs.
        try:
            max_iterations = self._effective_max_iterations(checkpoint_path)
        except WarmStartError as e:
            logger.error(f"Warm start failed: {e}")
            for record in records:
                if record.has_method:
                    record.status = STATUS_BUILD_FAILED
                    record.eval_error = f"warm start: {e}"
            return records
        if checkpoint_path and max_iterations is not None:
            logger.info(
                f"Warm start: checkpoint at epoch "
                f"{self._checkpoint_epoch_cached(checkpoint_path)} -> this batch "
                f"trains to {max_iterations} "
                f"(+{self.env_extra['MAX_ITERATIONS']})"
            )

        try:
            if self.backend == "hpc":
                return self._evaluate_hpc(records, checkpoint_path)
            return self._evaluate_local(records, checkpoint_path)
        except KeyboardInterrupt:
            # Manual kill (Ctrl-C): stop whatever the backend left running so we
            # don't strand a training container (local) or cluster jobs (hpc),
            # then re-raise so the interrupt still tears the run down.
            logger.warning("evaluation interrupted; terminating running jobs")
            self.runner.terminate()
            raise

    # ----------------------------------------------------------- local backend
    def _evaluate_local(
        self, records: List[RewardRecord], checkpoint_path: Optional[str] = None
    ) -> List[RewardRecord]:
        """Build -> run -> capture each candidate in turn on the local machine."""
        pending = [r for r in records if r.has_method]
        logger.info(f"Running {len(pending)} candidate(s), one at a time")

        # No queue, no artifacts tarball: the job writes its logs into
        # <output_dir>/<tag>/ and we read them there.
        for record in records:
            if not record.has_method:
                record.status = STATUS_GEN_FAILED
                continue
            tag = record.tag
            try:
                tarball = self.workspace.build_codebase(
                    record.reward_method, tag, checkpoint_path
                )
            except RewardInjectionError as e:
                logger.error(f"[{tag}] reward injection failed: {e}")
                record.status = STATUS_BUILD_FAILED
                record.eval_error = f"injection: {e}"
                continue

            result = self.runner.run(
                tarball_path=tarball,
                work_dir=os.path.join(self.output_dir, tag),
                env=self._build_env(record.seed, checkpoint_path),
                command=self._build_command(record.seed),
                build_args=self.build_args,
                timeout_seconds=self.timeout_seconds,
            )
            record.status = result.status
            if result.status != "succeeded":
                record.eval_error = result.error or result.status
                continue

            captured = self.processor.capture(result.work_dir)
            if captured is None:
                record.status = STATUS_NO_METRICS
                record.eval_error = "no usable TensorBoard logs"
                continue
            record.log_path = captured.log_path
            record.tb_path = captured.tb_path
            record.summary_path = captured.summary_path
            record.checkpoint_path = captured.checkpoint_path

        return records

    # -------------------------------------------------------- hpc primitives
    # submit_record / poll_record / harvest_record are the non-blocking halves of
    # _evaluate_hpc below, which is written in terms of them. They are public
    # because a driver that schedules across several *batches* at once cannot use
    # evaluate(): it submits a whole batch and blocks until the last job lands, so
    # a straggler in one batch idles the cluster for every other. scripts/
    # transfer_ablation.py drives these three directly to keep one global window
    # of in-flight jobs filled from whichever batch is unblocked. Everything that
    # touches the scheduler or the record's lifecycle lives here, so the two
    # callers cannot drift apart.

    def submit_record(
        self, record: RewardRecord, checkpoint_path: Optional[str] = None
    ) -> Optional[HPCJob]:
        """Stage + inject + build + push + submit one candidate. Non-blocking.

        Mutates ``record``: on success ``job_id`` and ``status`` (submitted); on
        failure ``status`` (gen_failed / build_failed) and ``eval_error``.

        Returns:
            The job handle to poll, or None if the record never reached the
            scheduler — in which case the record already carries the reason.
        """
        if not record.has_method:
            record.status = STATUS_GEN_FAILED
            return None

        tag = record.tag
        try:
            tarball = self.workspace.build_codebase(
                record.reward_method, tag, checkpoint_path
            )
        except RewardInjectionError as e:
            logger.error(f"[{tag}] reward injection failed: {e}")
            record.status = STATUS_BUILD_FAILED
            record.eval_error = f"injection: {e}"
            return None

        try:
            # _build_hpc_command resolves the warm-start-adjusted epoch ceiling,
            # which reads the checkpoint and can raise. evaluate() catches that
            # batch-wide before anything is staged; a caller driving records one
            # at a time has no such batch, so it is caught per record here.
            command = self._build_hpc_command(record.seed, checkpoint_path)
        except WarmStartError as e:
            logger.error(f"[{tag}] warm start failed: {e}")
            record.status = STATUS_BUILD_FAILED
            record.eval_error = f"warm start: {e}"
            return None

        try:
            job = self.runner.submit(
                tarball_path=tarball,
                tag=tag,
                command=command,
                max_runtime_hours=self.max_runtime_hours,
                datasets=self.datasets,
                job_name=f"{self.job_name_prefix}_{tag}",
            )
        except HPCRunnerError as e:
            logger.error(f"[{tag}] submit failed: {e}")
            record.status = STATUS_BUILD_FAILED
            record.eval_error = f"submit: {e}"
            return None

        record.job_id = job.job_id
        record.status = STATUS_SUBMITTED
        return job

    def poll_record(self, job: HPCJob) -> str:
        """This job's current scheduler status. One request, never blocks."""
        return self.runner.poll(job.job_id)

    def harvest_record(
        self, record: RewardRecord, job: HPCJob, status: str
    ) -> Optional[HPCJob]:
        """Recycle a terminal job's artifacts onto its record.

        Copies the NAS output into ``<output_dir>/<tag>/`` and reads it exactly as
        the local backend does, setting ``log_path`` / ``tb_path`` /
        ``summary_path`` / ``checkpoint_path`` / ``status`` on the record. Fitness
        is still :class:`FitnessScorer`'s job.

        The recycle runs *before* the status check on purpose — however the job
        ended, what matters is whether it wrote results back. A reward bad enough
        to fail training still does; a job the cluster killed never gets the
        chance, so an empty result means nothing was measured and it is re-run
        instead of scored.

        Returns:
            A replacement job handle when the job came back empty and was
            resubmitted (the caller should keep polling that one instead), else
            None — the record is finished, successfully or not.
        """
        work_dir = os.path.join(self.output_dir, record.tag)
        if os.path.exists(work_dir):
            shutil.rmtree(work_dir, ignore_errors=True)
        collected = self.runner.collect(
            job.job_id, work_dir, self.result_grace_seconds
        )
        if collected is None:
            return self._retry_no_results(record, job, status)

        if status != "completed":
            logger.warning(f"[{record.tag}] job {job.job_id} {status}")
            record.status = "failed"
            record.eval_error = f"hpc job {status}"
            return None

        captured = self.processor.capture(work_dir)
        if captured is None:
            record.status = STATUS_NO_METRICS
            record.eval_error = "no usable TensorBoard logs"
            return None

        record.log_path = captured.log_path
        record.tb_path = captured.tb_path
        record.summary_path = captured.summary_path
        record.checkpoint_path = captured.checkpoint_path
        record.status = "succeeded"
        logger.info(f"[{record.tag}] completed and captured")
        return None

    # ------------------------------------------------------------- hpc backend
    def _evaluate_hpc(
        self, records: List[RewardRecord], checkpoint_path: Optional[str] = None
    ) -> List[RewardRecord]:
        """Submit the whole batch to the CARES scheduler, then recycle as they finish.

        Two phases so jobs train *concurrently* on the cluster:
          A. Submit — stage + inject + build + push + submit each candidate.
          B. Monitor + recycle — poll every outstanding job; as each reaches a
             terminal status, copy its NAS artifacts into <output_dir>/<tag>/ and
             capture them in place (same ResultProcessor as local).
        """
        pending = [r for r in records if r.has_method]

        # Respect the scheduler's active-job cap up front (mirrors hpc_submit.py):
        # a mid-batch rejection would strand already-submitted jobs.
        active = self.runner.active_job_count()
        if active + len(pending) > self.runner.max_active_jobs:
            msg = (
                f"{active} active job(s) + {len(pending)} new would exceed the "
                f"{self.runner.max_active_jobs}-job scheduler cap; submit a "
                f"smaller batch or wait for running jobs to finish"
            )
            logger.error(msg)
            for record in pending:
                record.status = STATUS_BUILD_FAILED
                record.eval_error = msg
            return records

        # --- Phase A: submit every candidate ---------------------------------
        handles: Dict[str, RewardRecord] = {}   # job_id -> record
        jobs: Dict[str, HPCJob] = {}            # job_id -> handle (for resubmit)
        for record in records:
            job = self.submit_record(record, checkpoint_path)
            if job is None:
                continue  # submit_record already recorded why
            handles[job.job_id] = record
            jobs[job.job_id] = job

        if not handles:
            logger.error("No candidates were submitted to the HPC scheduler")
            return records

        # --- Phase B: monitor + recycle --------------------------------------
        logger.info(
            f"Monitoring {len(handles)} HPC job(s); polling every "
            f"{self.poll_seconds:.0f}s"
        )
        outstanding = set(handles)
        while outstanding:
            for job_id in list(outstanding):
                job = jobs[job_id]
                status = self.poll_record(job)
                if not self.runner.is_terminal(status):
                    continue
                outstanding.discard(job_id)
                record = handles[job_id]
                retry = self.harvest_record(record, job, status)
                if retry is not None:
                    # Came back empty and was resubmitted; follow the new job id.
                    handles[retry.job_id] = record
                    jobs[retry.job_id] = retry
                    outstanding.add(retry.job_id)
            if outstanding:
                time.sleep(self.poll_seconds)

        return records

    def _retry_no_results(
        self, record: RewardRecord, job: HPCJob, status: str
    ) -> Optional[HPCJob]:
        """Resubmit a job that came back empty; None once the budget is spent.

        On None the record is marked failed with the attempt count, so the
        history shows it was never measured rather than judged.
        """
        if job.attempt > self.max_retries:
            logger.error(
                f"[{record.tag}] {status} with no results after {job.attempt} "
                f"attempt(s); giving up — this candidate goes UNMEASURED"
            )
            record.status = "failed"
            record.eval_error = f"no results after {job.attempt} attempt(s)"
            return None

        logger.warning(
            f"[{record.tag}] {status} with no results; "
            f"retrying {job.attempt}/{self.max_retries}"
        )
        try:
            new_job = self.runner.resubmit(job)
        except HPCRunnerError as e:
            logger.error(f"[{record.tag}] resubmit failed: {e}")
            record.status = "failed"
            record.eval_error = f"no results, resubmit failed: {e}"
            return None

        record.job_id = new_job.job_id
        record.status = STATUS_SUBMITTED
        return new_job
