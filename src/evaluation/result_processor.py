"""
Result processing for coordinator-dispatched training.

Each finished job is downloaded as an artifacts tarball containing the
``logs/`` tree produced by ``scripts/train.py`` (rl_games), i.e.
``logs/rl_games/<config>/<run>/summaries/events.out.tfevents.*`` plus params and
checkpoints. This module unpacks that tarball, locates the TensorBoard event
file, and reads the **fitness** metric the tasks log via
``self.extras["log"]["fitness_function"]`` — the fixed evaluation signal that
replaces the old ``consecutive_successes``. It also writes a human-readable
scalar summary used as LLM feedback.
"""

import os
import glob
import logging
import tarfile
from typing import Dict, List, Optional
from dataclasses import dataclass

import numpy as np
from tensorboard.backend.event_processing import event_accumulator

from . import config

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Container for a single finished training run."""
    log_path: str            # extracted run directory (holds summaries/, params/, nn/)
    fitness: float           # max fitness_function over training
    tb_path: str             # path to the TensorBoard event file
    summary_path: str        # path to the generated training_summary.txt
    idx: int
    job_id: Optional[str] = None

    def __repr__(self):
        return (f"EvaluationResult(idx={self.idx}, fitness={self.fitness:.4f}, "
                f"job_id={self.job_id})")


class ResultProcessor:
    """Unpacks job artifacts and extracts the fitness metric + scalar summary."""

    def __init__(self, fitness_tag: str = config.FITNESS_METRIC):
        # We match the tag by suffix so whatever scope rl_games/IsaacAlgoObserver
        # prefixes it with (e.g. "Episode/fitness_function") still resolves.
        self.fitness_tag = fitness_tag

    # ---------------------------------------------------------------- unpack
    @staticmethod
    def extract_artifacts(tarball_path: str, dest_dir: str) -> str:
        """Extract an artifacts tarball into ``dest_dir`` and return that dir."""
        os.makedirs(dest_dir, exist_ok=True)
        with tarfile.open(tarball_path, "r:gz") as tar:
            tar.extractall(dest_dir)
        return dest_dir

    @staticmethod
    def find_event_file(root: str) -> Optional[str]:
        """Find the TensorBoard event file under ``root`` (prefers a summaries/ dir)."""
        candidates = glob.glob(
            os.path.join(root, "**", "events.out.tfevents.*"), recursive=True
        )
        if not candidates:
            return None
        # Prefer files inside a 'summaries' directory, then the largest file.
        candidates.sort(
            key=lambda p: ("summaries" in p.split(os.sep), os.path.getsize(p)),
            reverse=True,
        )
        return candidates[0]

    # --------------------------------------------------------------- process
    def process_artifacts(
        self, tarball_path: str, dest_dir: str, idx: int, job_id: Optional[str] = None
    ) -> Optional[EvaluationResult]:
        """
        Unpack ``tarball_path`` and build an EvaluationResult.

        Returns None if no usable TensorBoard logs are found.
        """
        try:
            self.extract_artifacts(tarball_path, dest_dir)
        except (tarfile.TarError, OSError) as e:
            logger.error(f"Failed to extract artifacts for idx {idx}: {e}")
            return None

        tb_path = self.find_event_file(dest_dir)
        if not tb_path:
            logger.error(f"No TensorBoard event file in artifacts for idx {idx}")
            return None

        run_dir = os.path.dirname(os.path.dirname(tb_path)) \
            if os.path.basename(os.path.dirname(tb_path)) == "summaries" \
            else os.path.dirname(tb_path)

        record_dir = os.path.join(run_dir, config.TRAINING_RECORD_DIR)
        os.makedirs(record_dir, exist_ok=True)
        summary_path = os.path.join(record_dir, config.TRAINING_SUMMARY_FILE)
        self.summarize_tensorboard(tb_path, summary_path)

        fitness = self.read_fitness(tb_path)

        result = EvaluationResult(
            log_path=run_dir,
            fitness=fitness,
            tb_path=tb_path,
            summary_path=summary_path,
            idx=idx,
            job_id=job_id,
        )
        logger.info(f"Processed result: {result}")
        return result

    # ------------------------------------------------------------- TB reading
    def _load_accumulator(self, tb_file: str):
        ea = event_accumulator.EventAccumulator(
            tb_file, size_guidance=config.TENSORBOARD_SIZE_GUIDANCE
        )
        ea.Reload()
        return ea

    def _resolve_tag(self, ea, wanted: str) -> Optional[str]:
        """Resolve ``wanted`` to an actual scalar tag, matching by exact or suffix."""
        keys = ea.scalars.Keys()
        if wanted in keys:
            return wanted
        suffix = wanted.split("/")[-1]
        matches = [k for k in keys if k.split("/")[-1] == suffix or k.endswith(suffix)]
        if matches:
            if len(matches) > 1:
                logger.warning(f"Multiple tags match {wanted!r}: {matches}; using {matches[0]}")
            return matches[0]
        return None

    def read_fitness(self, tb_file: str) -> float:
        """Return the max value of the fitness metric over training (0.0 if absent)."""
        if not os.path.exists(tb_file):
            logger.error(f"TensorBoard file not found: {tb_file}")
            return 0.0
        try:
            ea = self._load_accumulator(tb_file)
            tag = self._resolve_tag(ea, self.fitness_tag)
            if tag is None:
                logger.warning(
                    f"Fitness tag {self.fitness_tag!r} not found. "
                    f"Available: {ea.scalars.Keys()}"
                )
                return 0.0
            events = ea.Scalars(tag)
            if not events:
                return 0.0
            return float(max(e.value for e in events))
        except Exception as e:  # noqa: BLE001 - TB parsing surfaces many error types
            logger.error(f"Error reading fitness from {tb_file}: {e}")
            return 0.0

    def summarize_tensorboard(self, event_file_path: str, output_txt_path: str):
        """Write a human-readable summary of all scalar metrics for LLM feedback."""
        try:
            acc = self._load_accumulator(event_file_path)
            scalar_tags = acc.Tags()["scalars"]

            lines = [
                "## Reinforcement Learning Model Performance Summary\n",
                f"Source File: {os.path.basename(event_file_path)}\n",
                "-" * 40 + "\n",
            ]
            for tag in scalar_tags:
                values = np.array([e.value for e in acc.Scalars(tag)])
                if len(values) == 0:
                    continue
                initial_idx = max(int(len(values) * 0.1), 1)
                mid_idx = int(len(values) * 0.5)
                initial_perf = np.mean(values[:initial_idx])
                mid_perf = values[mid_idx]
                final_perf = np.mean(values[-initial_idx:])

                lines.append(f"## Metric: {tag}\n")
                lines.append("- **Overall Statistics:**")
                lines.append(f"  - Mean: {np.mean(values):.4f}")
                lines.append(f"  - Std Dev: {np.std(values):.4f} (Measures stability/variance)")
                lines.append(f"  - Max Value: {np.max(values):.4f}")
                lines.append(f"  - Min Value: {np.min(values):.4f}\n")
                lines.append("- **Performance Trend:**")
                lines.append(f"  - Initial Performance (first 10%): ~{initial_perf:.4f}")
                lines.append(f"  - Mid-Training Performance (at 50%): ~{mid_perf:.4f}")
                lines.append(f"  - Final Performance (last 10%): ~{final_perf:.4f}\n")
                lines.append("-" * 40 + "\n")

            with open(output_txt_path, "w") as f:
                f.write("\n".join(lines))
            logger.info(f"Summary written to {output_txt_path}")
        except Exception as e:  # noqa: BLE001
            logger.error(f"Error summarizing TensorBoard file: {e}")

    # ---------------------------------------------------------------- ranking
    @staticmethod
    def select_best_result(results: List[EvaluationResult]) -> Optional[EvaluationResult]:
        """Return the highest-fitness result, or None if the list is empty."""
        if not results:
            logger.warning("No results to select from")
            return None
        best = max(results, key=lambda r: r.fitness)
        logger.info(f"Best result: {best}")
        return best
