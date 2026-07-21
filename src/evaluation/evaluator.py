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
import logging
from typing import Dict, List, Optional

from .local_runner import LocalRunner
from .workspace_manager import WorkspaceManager
from .reward_injection import RewardInjectionError
from .result_processor import ResultProcessor
from . import config
from src.reward_history import (
    RewardRecord,
    STATUS_GEN_FAILED,
    STATUS_BUILD_FAILED,
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
        runner: Dict with local-runner settings: use_gpu, timeout_seconds, image,
            env (extra container env passed to every job), build_args, and an
            optional command_template override.
        output_dir: Where each candidate's job runs and its logs land
            (``<output_dir>/<tag>/``).
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

        # Build + run each job's Dockerfile on the local machine, one at a time.
        self.client = LocalRunner(
            image=runner.get("image", "ard-local"),
            use_gpu=self.use_gpu,
        )
        self.workspace = WorkspaceManager(
            tasks_repo=tasks_repo,
            env_file_rel=env_file_rel,
            build_root=build_root,
        )
        self.processor = ResultProcessor()

        if not self.client.healthz():
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
    def _build_env(self, seed: Optional[int]) -> Dict[str, str]:
        """Container env for one job: the task, its seed, and any configured extras.

        The ard-isaaclab-tasks image entrypoint reads ``TASK`` and ``SEED`` (plus
        optional ``MAX_ITERATIONS``/``NUM_ENVS``/``WANDB_*``) from the env, so the
        per-eval seed is now honoured (the old quickstart command path ignored it).
        """
        env = {"TASK": self.task}
        if seed is not None:
            env["SEED"] = str(seed)
        env.update({k: str(v) for k, v in self.env_extra.items()})
        return env

    def _build_command(self, seed: Optional[int]) -> Optional[str]:
        """Optional CMD override; None means run the image's own entrypoint."""
        if not self.command_template:
            return None
        return self.command_template.format(
            task=self.task,
            seed="" if seed is None else seed,
        )

    def evaluate(
        self,
        records: List[RewardRecord],
    ) -> List[RewardRecord]:
        """
        Train a batch of candidate records one at a time and capture their output.

        Mutates each record in place: sets ``status``, ``eval_error`` and (on
        success) the captured ``log_path`` / ``tb_path`` / ``summary_path``.
        Fitness and best-selection are left to :class:`FitnessScorer`.
        Training length is controlled by each task's ``max_epochs`` in its
        ``rl_games_ppo_cfg.yaml``.

        Args:
            records: Candidate records. Each must carry ``reward_method`` (records
                whose generation failed are skipped) and provides ``tag`` / ``seed``.

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

        pending = [r for r in records if r.has_method]
        logger.info(f"Running {len(pending)} candidate(s), one at a time")

        # Build -> run -> capture each candidate in turn. No queue, no artifacts
        # tarball: the job writes its logs into <output_dir>/<tag>/ and we read
        # them there.
        for record in records:
            if not record.has_method:
                record.status = STATUS_GEN_FAILED
                continue
            tag = record.tag
            try:
                tarball = self.workspace.build_codebase(record.reward_method, tag)
            except RewardInjectionError as e:
                logger.error(f"[{tag}] reward injection failed: {e}")
                record.status = STATUS_BUILD_FAILED
                record.eval_error = f"injection: {e}"
                continue

            result = self.client.run(
                tarball_path=tarball,
                work_dir=os.path.join(self.output_dir, tag),
                env=self._build_env(record.seed),
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

        return records
