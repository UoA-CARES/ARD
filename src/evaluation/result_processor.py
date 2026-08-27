"""
Artifact capture for local training jobs.

Each finished job leaves its ``logs/`` tree in its own ``work_dir`` (the ``/work``
mount), produced by ``scripts/train.py`` (rl_games), i.e.
``logs/rl_games/<config>/<run>/summaries/events.out.tfevents.*`` plus params and
checkpoints. This module's job is strictly **capture**: locate the TensorBoard
event file under that dir and write a human-readable scalar summary (used as LLM
feedback). Nothing is packed or unpacked — the logs are read where they landed.

It deliberately does **not** read the fitness metric or pick a winner — that
judgement lives in :mod:`src.evaluation.scorer`. Keeping capture and judgement
apart lets the evaluator be responsible only for "run it and collect the
output", while scoring is a separate, swappable step.
"""

import os
import glob
import logging
from typing import List, Optional
from dataclasses import dataclass

import numpy as np
from tensorboard.backend.event_processing import event_accumulator

from . import config

logger = logging.getLogger(__name__)


def load_accumulator(tb_file: str):
    """Load a TensorBoard event file into an EventAccumulator (shared helper)."""
    ea = event_accumulator.EventAccumulator(
        tb_file, size_guidance=config.TENSORBOARD_SIZE_GUIDANCE
    )
    ea.Reload()
    return ea


@dataclass
class CapturedArtifacts:
    """Paths captured from a finished job's logs (no metric judgement)."""
    log_path: str            # rl_games run directory (holds summaries/, params/, nn/)
    tb_path: str             # path to the TensorBoard event file
    summary_path: str        # path to the generated training_summary.txt
    checkpoint_path: Optional[str] = None  # latest rl_games policy checkpoint (nn/*.pth)


class ResultProcessor:
    """Reads a job's in-place logs and writes the scalar summary used for feedback."""

    # ---------------------------------------------------------------- locate
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

    @staticmethod
    def find_checkpoint(run_dir: str) -> Optional[str]:
        """Find the checkpoint under ``run_dir/nn`` to warm-start from.

        rl_games writes two kinds of file here: periodic ``last_<name>_ep_<N>_
        rew_<R>.pth`` snapshots (including a final one when training ends), and
        a single plain ``<name>.pth`` that it only overwrites when a *new* best
        reward is reached (once training has run past ``save_best_after`` in the
        task's ``rl_games_ppo_cfg.yaml``). Training reward isn't monotonic, so
        the final periodic snapshot can score worse than an earlier peak — the
        plain best-reward file is what warm-starting should resume from. A run
        too short to ever clear ``save_best_after`` won't have one, so fall back
        to the newest periodic snapshot in that case.
        """
        candidates = glob.glob(os.path.join(run_dir, "nn", "*.pth"))
        if not candidates:
            return None
        best = [c for c in candidates if not os.path.basename(c).startswith("last_")]
        if best:
            return max(best, key=os.path.getmtime)
        return max(candidates, key=os.path.getmtime)

    @staticmethod
    def find_videos(run_dir: str) -> List[str]:
        """Find the videos output by play.py under ``run_dir/videos/play`` for the VLM.

        ard-isaaclab-tasks play.py writes videos to the same directory as the checkpoint, but 
        in a subdirectory called ``videos/play``. This method returns a list of all video files 
        found in that directory.
        """
        videos = glob.glob(os.path.join(run_dir, "videos", "play", "*.mp4"))
        if not videos:
            logger.warning(f"No videos found under {os.path.join(run_dir, 'videos', 'play')}")
        return videos

    # --------------------------------------------------------------- capture
    def capture(self, work_dir: str) -> Optional[CapturedArtifacts]:
        """
        Locate the job's logs under ``work_dir`` and write their scalar summary.

        Returns the captured paths, or None if no usable TensorBoard logs are
        found. Does not read fitness — see :mod:`src.evaluation.scorer`.
        """
        tb_path = self.find_event_file(work_dir)
        if not tb_path:
            logger.error(f"No TensorBoard event file under {work_dir}")
            return None

        run_dir = os.path.dirname(os.path.dirname(tb_path)) \
            if os.path.basename(os.path.dirname(tb_path)) == "summaries" \
            else os.path.dirname(tb_path)

        record_dir = os.path.join(run_dir, config.TRAINING_RECORD_DIR)
        os.makedirs(record_dir, exist_ok=True)
        summary_path = os.path.join(record_dir, config.TRAINING_SUMMARY_FILE)
        self.summarise_tensorboard(tb_path, summary_path)

        captured = CapturedArtifacts(
            log_path=run_dir, tb_path=tb_path, summary_path=summary_path,
            checkpoint_path=self.find_checkpoint(run_dir),
        )
        logger.info(f"Captured artifacts: {run_dir}")
        return captured

    # ------------------------------------------------------------- TB summary
    def summarise_tensorboard(self, event_file_path: str, output_txt_path: str):
        """Write a human-readable summary of all scalar metrics for LLM feedback."""
        try:
            acc = load_accumulator(event_file_path)
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
