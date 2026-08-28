"""
Local evaluation orchestrator.

``RewardEvaluator`` is the high-level entry point ARD's refinement loop calls.
Its sole responsibility is **dispatch + capture** — running candidates and
collecting their output. For a batch of :class:`~src.reward_history.RewardRecord`
(each carrying a proposed ``_get_rewards`` method) it walks them one at a time:

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
import logging
from typing import Dict, List, Optional

from .local_runner import LocalRunner
from .hpc_runner import HPCRunner, HPCRunnerError, HPCJob
from .workspace_manager import WorkspaceManager
from .reward_injection import RewardInjectionError
from .result_processor import ResultProcessor
from . import config
from src.reward_history import (
    RewardRecord,
    STATUS_GEN_FAILED,
    STATUS_BUILD_FAILED,
    STATUS_SUBMITTED,
    STATUS_NO_METRICS,
)

logger = logging.getLogger(__name__)


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
            * ``hpc``: an ``hpc`` sub-dict (registry, image_repo, nas_outputs,
              max_runtime_hours, datasets, poll_seconds, job_name_prefix,
              extra_args). Each candidate is built + pushed as its own image and
              submitted to the CARES HPC Scheduler; the batch trains concurrently.
              ``env`` MAX_ITERATIONS/NUM_ENVS are translated to command flags
              because the scheduler drops the job env block.
        output_dir: Where each candidate's job runs and its logs land
            (``<output_dir>/<tag>/``). For the hpc backend this is where NAS
            artifacts are recycled to.
        build_root: Optional staging dir for codebase tarballs.
    """

    def __init__(
        self,
        tasks_repo: str,
        env_file_rel: str,
        task: str,
        runner: Dict,
        output_dir: str,
        build_root: Optional[str] = None,
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
        self.processor = ResultProcessor()

        if self.backend == "hpc":
            # Build + push each candidate's image, submit to the CARES scheduler,
            # then recycle NAS artifacts. Jobs train concurrently on the cluster.
            hpc = dict(runner.get("hpc", {}))
            self.runner = HPCRunner(
                registry=hpc.get("registry", config.DEFAULT_HPC_REGISTRY),
                image_repo=hpc.get("image_repo", config.DEFAULT_HPC_IMAGE_REPO),
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
        """Pristine ``_get_rewards`` source, for seeding the LLM prompt."""
        return self.workspace.get_reward_template()

    def get_env_source(self) -> str:
        """Full pristine task env source, for LLM task context."""
        return self.workspace.get_env_source()

    # ---------------------------------------------------------------- evaluate
    @staticmethod
    def _checkpoint_epoch(checkpoint_path: str) -> Optional[int]:
        """Read the epoch count rl_games saved inside a checkpoint.

        rl_games checkpoints bundle their own epoch counter alongside the
        weights (``a2c_common.py``: ``state['epoch'] = self.epoch_num`` on
        save, ``self.epoch_num = weights['epoch']`` on load) — loading a
        checkpoint resumes the epoch count, not just the network. Returns
        None (rather than raising) if the file can't be read, e.g. torch
        isn't installed in this process — callers fall back to the
        configured budget as-is.
        """
        try:
            import torch
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            return int(state.get("epoch", 0))
        except Exception as e:  # noqa: BLE001 - any read/unpickle failure
            logger.warning(f"Could not read epoch count from {checkpoint_path}: {e}")
            return None

    def _effective_max_iterations(self, checkpoint_path: Optional[str]) -> Optional[str]:
        """The ``--max_iterations`` value for one job, accounting for warm-start.

        Without this, warm-starting from a checkpoint that already reached the
        configured ``MAX_ITERATIONS`` immediately hits that same ceiling on
        load (rl_games restores the checkpoint's own epoch count) and stops
        after ~0 additional epochs. Extending the ceiling by the checkpoint's
        inherited epoch count guarantees every candidate still gets the full
        configured budget of *new* training on top of what it resumed from.
        Returns None if no ``MAX_ITERATIONS`` is configured at all (the task's
        own ``max_epochs`` default applies, unaffected by this).
        """
        configured = self.env_extra.get("MAX_ITERATIONS")
        if configured is None:
            return None
        if not checkpoint_path:
            return str(configured)
        epoch = self._checkpoint_epoch(checkpoint_path)
        if epoch is None:
            return str(configured)
        return str(epoch + int(configured))

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
        baked in, not the path used. Both are appended after any
        user-configured ``EXTRA_ARGS`` from ``runner.env``.
        """
        env = {"TASK": self.task}
        if seed is not None:
            env["SEED"] = str(seed)
        env.update({
            k: str(v) for k, v in self.env_extra.items()
            if k not in ("plasticity", "PLASTICITY")
        })
        max_iterations = self._effective_max_iterations(checkpoint_path)
        if max_iterations is not None:
            env["MAX_ITERATIONS"] = max_iterations

        extra_flags = []
        if checkpoint_path:
            extra_flags.append(f"--checkpoint {config.IMAGE_REPO_ROOT}/{config.WARM_START_CHECKPOINT_REL}")
        if self._plasticity_enabled():
            extra_flags.append("--plasticity")
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
        (warm-start) rides the same way, as ``--checkpoint``, matching
        ``scripts/train.py``'s own flag — pointed at the fixed in-image path
        the checkpoint was baked to (see ``_build_env``), not the host path.
        ``plasticity`` from ``runner.env`` is likewise appended as ``--plasticity``.
        """
        flags = ["--task", self.task]
        if seed is not None:
            flags += ["--seed", str(seed)]
        if checkpoint_path:
            flags += ["--checkpoint", f"{config.IMAGE_REPO_ROOT}/{config.WARM_START_CHECKPOINT_REL}"]
        max_iterations = self._effective_max_iterations(checkpoint_path)
        if max_iterations is not None:
            flags += ["--max_iterations", max_iterations]
        if self.env_extra.get("NUM_ENVS") is not None:
            flags += ["--num_envs", str(self.env_extra["NUM_ENVS"])]
        if self._plasticity_enabled():
            flags += ["--plasticity"]

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
            try:
                job = self.runner.submit(
                    tarball_path=tarball,
                    tag=tag,
                    command=self._build_hpc_command(record.seed, checkpoint_path),
                    max_runtime_hours=self.max_runtime_hours,
                    datasets=self.datasets,
                    job_name=f"{self.job_name_prefix}_{tag}",
                )
            except HPCRunnerError as e:
                logger.error(f"[{tag}] submit failed: {e}")
                record.status = STATUS_BUILD_FAILED
                record.eval_error = f"submit: {e}"
                continue
            record.job_id = job.job_id
            record.status = STATUS_SUBMITTED
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
                status = self.runner.poll(job_id)
                if not self.runner.is_terminal(status):
                    continue
                outstanding.discard(job_id)
                record = handles[job_id]

                # Recycle: copy the NAS artifacts into the job's work dir, then
                # read them exactly as the local backend does. This runs before
                # the status check on purpose — however the job ended, what
                # matters is whether it wrote results back. A reward bad enough
                # to fail training still does; a job the cluster killed never
                # gets the chance, so an empty result means nothing was
                # measured and it is re-run instead of scored.
                work_dir = os.path.join(self.output_dir, record.tag)
                if os.path.exists(work_dir):
                    shutil.rmtree(work_dir, ignore_errors=True)
                collected = self.runner.collect(
                    job_id, work_dir, self.result_grace_seconds
                )
                if collected is None:
                    retry = self._retry_no_results(record, jobs[job_id], status)
                    if retry is not None:
                        handles[retry.job_id] = record
                        jobs[retry.job_id] = retry
                        outstanding.add(retry.job_id)
                    continue

                if status != "completed":
                    logger.warning(f"[{record.tag}] job {job_id} {status}")
                    record.status = "failed"
                    record.eval_error = f"hpc job {status}"
                    continue
                captured = self.processor.capture(work_dir)
                if captured is None:
                    record.status = STATUS_NO_METRICS
                    record.eval_error = "no usable TensorBoard logs"
                    continue
                record.log_path = captured.log_path
                record.tb_path = captured.tb_path
                record.summary_path = captured.summary_path
                record.checkpoint_path = captured.checkpoint_path
                record.status = "succeeded"
                logger.info(f"[{record.tag}] completed and captured")
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
