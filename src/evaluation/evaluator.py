"""
Coordinator-driven evaluation orchestrator.

``RewardEvaluator`` is the high-level entry point ARD's refinement loop calls. For
a batch of LLM-proposed ``_get_rewards`` methods it:

1. Builds one job codebase per candidate (pristine ard-isaaclab-tasks repo + the
   proposed reward spliced in) — :class:`WorkspaceManager`.
2. Submits every candidate as a job to the PCS coordinator — :class:`CoordinatorClient`.
   The coordinator runs them concurrently across its registered GPU workers.
3. Waits for all jobs to terminate, downloads each succeeded job's artifacts, and
   reads its fitness + scalar summary — :class:`ResultProcessor`.
4. Returns the best candidate by fitness, plus per-candidate logs for LLM feedback.

This replaces the old SSH machine-pool + ``run_remote_pipeline.sh`` executor: ARD
is now purely a coordinator client.
"""

import os
import logging
from typing import Dict, List, Optional, Sequence

from .coordinator_client import CoordinatorClient, CoordinatorError
from .workspace_manager import WorkspaceManager
from .reward_injection import RewardInjectionError
from .result_processor import ResultProcessor
from . import config

logger = logging.getLogger(__name__)


class RewardEvaluator:
    """
    Orchestrates coordinator-dispatched evaluation of reward candidates.

    Args:
        tasks_repo: Path to the ard-isaaclab-tasks checkout.
        env_file_rel: Task env file (relative to ``tasks_repo``) to inject into.
        task: Registered task ID, e.g. ``Isaac-ARD-Cartpole-v0``.
        coordinator: Dict with coordinator settings:
            base_url (required), token / token_env, docker_image, gpus,
            timeout_seconds, poll_interval, output_paths, command_template.
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

        # Coordinator job parameters.
        self.docker_image = coordinator.get("docker_image", config.DEFAULT_DOCKER_IMAGE)
        self.gpus = int(coordinator.get("gpus", config.DEFAULT_GPUS))
        self.timeout_seconds = int(
            coordinator.get("timeout_seconds", config.DEFAULT_TRAINING_TIMEOUT)
        )
        self.output_paths = coordinator.get("output_paths", config.DEFAULT_OUTPUT_PATHS)
        self.command_template = coordinator.get(
            "command_template",
            "MAX_ITERATIONS={max_iterations} bash quickstart.sh {task}",
        )

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

        if not self.client.healthz():
            logger.warning(
                f"Coordinator at {coordinator['base_url']} did not pass health check; "
                "submissions may fail."
            )
        logger.info(
            f"RewardEvaluator ready: task={task} image={self.docker_image} "
            f"gpus={self.gpus} timeout={self.timeout_seconds}s"
        )

    # ------------------------------------------------------------------ prompt
    def get_reward_template(self) -> str:
        """Pristine ``_get_rewards`` source, for seeding the LLM prompt."""
        return self.workspace.get_reward_template()

    def get_env_source(self) -> str:
        """Full pristine task env source, for LLM task context."""
        return self.workspace.get_env_source()

    # ---------------------------------------------------------------- evaluate
    def _build_command(self, max_iterations: int, seed: Optional[int]) -> str:
        return self.command_template.format(
            task=self.task,
            max_iterations=max_iterations,
            seed="" if seed is None else seed,
        )

    def evaluate(
        self,
        reward_methods: List[str],
        max_iterations: int,
        tag_prefix: str = "run",
        seeds: Optional[Sequence[Optional[int]]] = None,
    ):
        """
        Evaluate a batch of candidate ``_get_rewards`` methods.

        Args:
            reward_methods: Candidate method sources (one job each).
            max_iterations: Training iterations per job.
            tag_prefix: Label prefix for this batch (e.g. "iter1_run").
            seeds: Optional per-candidate seeds (len == reward_methods).

        Returns:
            (best, logs):
              best — dict for the top candidate or None if all failed.
              logs — list of per-candidate dicts (idx, job_id, status, fitness,
                     log_path, summary_path, error).
        """
        logs: List[Dict] = []
        if not reward_methods:
            logger.error("No reward methods provided for evaluation")
            return None, logs
        if not self.workspace.validate():
            logger.error("Workspace validation failed")
            return None, logs
        if seeds is None:
            seeds = [None] * len(reward_methods)

        # 1) Build + submit a job per candidate.
        job_meta: Dict[str, Dict] = {}     # job_id -> {idx, tag}
        for idx, method in enumerate(reward_methods):
            tag = f"{tag_prefix}_{idx}"
            entry = {"idx": idx, "tag": tag, "job_id": None, "status": "build_failed",
                     "fitness": float("-inf"), "log_path": None,
                     "summary_path": None, "error": None}
            try:
                tarball = self.workspace.build_codebase(method, tag)
            except RewardInjectionError as e:
                logger.error(f"[{tag}] reward injection failed: {e}")
                entry["error"] = f"injection: {e}"
                logs.append(entry)
                continue
            try:
                job_id = self.client.submit_job(
                    tarball_path=tarball,
                    command=self._build_command(max_iterations, seeds[idx]),
                    docker_image=self.docker_image,
                    output_paths=self.output_paths,
                    gpus=self.gpus,
                    timeout_seconds=self.timeout_seconds,
                )
            except CoordinatorError as e:
                logger.error(f"[{tag}] job submission failed: {e}")
                entry["status"] = "submit_failed"
                entry["error"] = f"submit: {e}"
                logs.append(entry)
                continue
            entry["job_id"] = job_id
            entry["status"] = "submitted"
            job_meta[job_id] = entry
            logs.append(entry)

        if not job_meta:
            logger.error("No jobs were submitted successfully")
            return None, logs

        # 2) Wait for all submitted jobs to finish.
        logger.info(f"Submitted {len(job_meta)} job(s); waiting for completion")
        finished = self.client.wait_for_all(list(job_meta.keys()))

        # 3) Process each finished job.
        results = []
        for job_id, job in finished.items():
            entry = job_meta[job_id]
            entry["status"] = job["status"]
            if job["status"] != "succeeded":
                entry["error"] = job.get("error_message") or job["status"]
                logger.warning(f"[{entry['tag']}] job {job_id} {job['status']}")
                continue

            artifacts_tar = os.path.join(self.output_dir, f"{entry['tag']}.tar.gz")
            extract_dir = os.path.join(self.output_dir, entry["tag"])
            if not self.client.download_artifacts(job_id, artifacts_tar):
                entry["status"] = "no_artifacts"
                entry["error"] = "job produced no artifacts"
                continue
            result = self.processor.process_artifacts(
                artifacts_tar, extract_dir, idx=entry["idx"], job_id=job_id
            )
            if result is None:
                entry["status"] = "no_metrics"
                entry["error"] = "no usable TensorBoard logs"
                continue
            entry["fitness"] = result.fitness
            entry["log_path"] = result.log_path
            entry["summary_path"] = result.summary_path
            results.append(result)

        # 4) Select the best candidate by fitness.
        if not results:
            logger.error("No successful evaluations in this batch")
            return None, logs
        best = ResultProcessor.select_best_result(results)
        best_dict = {
            "idx": best.idx,
            "fitness": best.fitness,
            "log_path": best.log_path,
            "tb_path": best.tb_path,
            "summary_path": best.summary_path,
            "job_id": best.job_id,
        }
        logger.info(f"Best candidate: idx={best.idx} fitness={best.fitness:.4f}")
        return best_dict, logs
