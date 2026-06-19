"""
Coordinator-driven evaluation orchestrator.

``RewardEvaluator`` is the high-level entry point ARD's refinement loop calls.
Its sole responsibility is **dispatch + capture** — running candidates and
collecting their output. For a batch of :class:`~src.reward_history.RewardRecord`
(each carrying a proposed ``_get_rewards`` method) it:

1. Builds one job codebase per candidate (pristine ard-isaaclab-tasks repo + the
   proposed reward spliced in) — :class:`WorkspaceManager`.
2. Submits every candidate as a job to the PCS coordinator — :class:`CoordinatorClient`.
   The coordinator runs them concurrently across its registered GPU workers.
3. Waits for all jobs to terminate, downloads each succeeded job's artifacts, and
   captures its run paths + scalar summary — :class:`ResultProcessor`.

It writes job status and captured artifact paths back onto each record but does
**not** read fitness or pick a winner — that judgement is
:class:`~src.evaluation.scorer.FitnessScorer`'s job. This keeps the evaluator a
pure executor and leaves scoring a separate, swappable step.

This replaces the old SSH machine-pool + ``run_remote_pipeline.sh`` executor: ARD
is now purely a coordinator client.
"""

import os
import logging
from typing import Dict, List, Optional

from .coordinator_client import CoordinatorClient, CoordinatorError
from .local_runner import LocalRunner
from .workspace_manager import WorkspaceManager
from .reward_injection import RewardInjectionError
from .result_processor import ResultProcessor
from . import config
from src.reward_history import (
    RewardRecord,
    STATUS_GEN_FAILED,
    STATUS_BUILD_FAILED,
    STATUS_SUBMIT_FAILED,
    STATUS_SUBMITTED,
    STATUS_NO_ARTIFACTS,
    STATUS_NO_METRICS,
)

logger = logging.getLogger(__name__)


class RewardEvaluator:
    """
    Orchestrates coordinator-dispatched evaluation of reward candidates.

    Args:
        tasks_repo: Path to the ard-isaaclab-tasks checkout.
        env_file_rel: Task env file (relative to ``tasks_repo``) to inject into.
        task: Registered task ID, e.g. ``Isaac-ARD-Cartpole-v0``.
        coordinator: Dict with coordinator settings:
            base_url (required), token / token_env, gpus, timeout_seconds,
            poll_interval, output_paths, env (extra container env passed to every
            job), build_args, and an optional command_template override.
        output_dir: Where artifacts are downloaded and extracted.
        build_root: Optional staging dir for codebase tarballs.
    """

    def __init__(
        self,
        tasks_repo: str,
        env_file_rel: str,
        task: str,
        coordinator: Dict,
        output_dir: str,
        build_root: Optional[str] = None,
    ):
        self.task = task
        self.output_dir = os.path.abspath(os.path.expanduser(output_dir))
        os.makedirs(self.output_dir, exist_ok=True)

        # Coordinator job parameters. PCS builds the project's Dockerfile per job
        # (no prebuilt image tag); the task image's entrypoint is driven by the
        # job `env`, so the task/seed are passed there rather than as a command.
        self.gpus = float(coordinator.get("gpus", config.DEFAULT_GPUS))
        self.timeout_seconds = int(
            coordinator.get("timeout_seconds", config.DEFAULT_TRAINING_TIMEOUT)
        )
        self.output_paths = coordinator.get("output_paths", config.DEFAULT_OUTPUT_PATHS)
        # Extra container env applied to every job (e.g. MAX_ITERATIONS, NUM_ENVS,
        # WANDB_*), and optional docker build args.
        self.env_extra = dict(coordinator.get("env", {}))
        self.build_args = dict(coordinator.get("build_args", {}))
        # Optional override of the image CMD. Default None -> the image's own
        # entrypoint runs, configured entirely through `env`.
        self.command_template = coordinator.get("command_template")

        # Single-machine mode: replicate one PCS worker locally (build + run the
        # job's Dockerfile via docker) instead of dispatching to a coordinator.
        # LocalRunner duck-types the client methods evaluate() uses, so nothing
        # else below changes. Select it with `coordinator.mode: local`.
        self.local_mode = str(coordinator.get("mode", "coordinator")).lower() == "local"
        if self.local_mode:
            self.client = LocalRunner(
                image=coordinator.get("image", "ard-local"),
                gpus=self.gpus,
                work_root=coordinator.get("work_root"),
                max_concurrent=coordinator.get("max_concurrent"),
            )
        else:
            self.client = CoordinatorClient(
                base_url=coordinator["base_url"],
                token=coordinator.get("token"),
                token_env=coordinator.get("token_env", config.DEFAULT_TOKEN_ENV),
                poll_interval=float(
                    coordinator.get("poll_interval", config.DEFAULT_POLL_INTERVAL)
                ),
            )
        self.workspace = WorkspaceManager(
            tasks_repo=tasks_repo,
            env_file_rel=env_file_rel,
            build_root=build_root,
        )
        self.processor = ResultProcessor()

        backend = "local docker" if self.local_mode else coordinator.get("base_url")
        if not self.client.healthz():
            logger.warning(
                f"{backend} did not pass health check; submissions may fail."
            )
        logger.info(
            f"RewardEvaluator ready: task={task} backend={backend} gpus={self.gpus} "
            f"timeout={self.timeout_seconds}s (deploy-by-Dockerfile, env-driven)"
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
        Dispatch a batch of candidate records for training and capture their output.

        Mutates each record in place: sets ``job_id``, ``status``, ``eval_error``
        and (on success) the captured ``log_path`` / ``tb_path`` / ``summary_path``.
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

        # 1) Build + submit a job per candidate that has a reward method.
        job_meta: Dict[str, RewardRecord] = {}     # job_id -> record
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
            try:
                job_id = self.client.submit_job(
                    tarball_path=tarball,
                    output_paths=self.output_paths,
                    env=self._build_env(record.seed),
                    command=self._build_command(record.seed),
                    build_args=self.build_args,
                    gpus=self.gpus,
                    timeout_seconds=self.timeout_seconds,
                )
            except CoordinatorError as e:
                logger.error(f"[{tag}] job submission failed: {e}")
                record.status = STATUS_SUBMIT_FAILED
                record.eval_error = f"submit: {e}"
                continue
            record.job_id = job_id
            record.status = STATUS_SUBMITTED
            job_meta[job_id] = record

        if not job_meta:
            logger.error("No jobs were submitted successfully")
            return records

        # 2) Wait for all submitted jobs to finish.
        logger.info(f"Submitted {len(job_meta)} job(s); waiting for completion")
        finished = self.client.wait_for_all(list(job_meta.keys()))

        # 3) Capture artifacts for each finished job.
        for job_id, job in finished.items():
            record = job_meta[job_id]
            record.status = job["status"]
            if job["status"] != "succeeded":
                record.eval_error = job.get("error") or job["status"]
                logger.warning(f"[{record.tag}] job {job_id} {job['status']}")
                continue

            artifacts_tar = os.path.join(self.output_dir, f"{record.tag}.tar.gz")
            extract_dir = os.path.join(self.output_dir, record.tag)
            if not self.client.download_artifacts(job_id, artifacts_tar):
                record.status = STATUS_NO_ARTIFACTS
                record.eval_error = "job produced no artifacts"
                continue
            captured = self.processor.capture(artifacts_tar, extract_dir)
            if captured is None:
                record.status = STATUS_NO_METRICS
                record.eval_error = "no usable TensorBoard logs"
                continue
            record.log_path = captured.log_path
            record.tb_path = captured.tb_path
            record.summary_path = captured.summary_path

        return records
