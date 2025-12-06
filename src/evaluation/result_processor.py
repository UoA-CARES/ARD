"""
Result processing module for training evaluation.

This module handles processing of training results including:
- TensorBoard log parsing and metric extraction
- Training log directory management
- Result aggregation and ranking
- Summary generation for LLM feedback
"""

import os
import glob
import shutil
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
import numpy as np
from tensorboard.backend.event_processing import event_accumulator

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Container for evaluation results from a single training run."""
    log_path: str
    max_consecutive_successes: float
    tb_path: str
    idx: int
    machine: Optional[str] = None

    def __repr__(self):
        return f"EvaluationResult(idx={self.idx}, max_con_successes={self.max_consecutive_successes:.2f}, machine={self.machine})"


class ResultProcessor:
    """
    Processes training results and TensorBoard logs.

    Responsibilities:
    - TensorBoard metric extraction
    - Log directory management (renaming, organization)
    - Summary generation for LLM analysis
    - Best result selection from multiple evaluations

    Args:
        task_config: Dictionary containing task configuration with 'logs_path' key

    Example:
        >>> processor = ResultProcessor(task_config)
        >>> result = processor.process_training_result(log_name="eval_0", idx=0, machine="gpu1")
        >>> print(f"Max consecutive successes: {result.max_consecutive_successes}")
    """

    def __init__(self, task_config: Dict):
        self.task_config = task_config
        self.logs_path = task_config.get("logs_path")

        if not self.logs_path:
            raise ValueError("task_config must contain 'logs_path' key")

    def process_training_result(self, log_name: str, idx: int, task_params: Dict, machine: Optional[str] = None) -> Optional[EvaluationResult]:
        """
        Process a single training result.

        Args:
            log_name: Name to assign to the log directory
            idx: Index of this evaluation run
            machine: Optional machine identifier where training ran

        Returns:
            EvaluationResult object if successful, None otherwise
        """
        try:

            # Create training record directory
            # self.logs_path = '/home/lee/code/isaactasks/ant/logs/rl_games/ant_direct'
            # 
            # log_name = eval_0
            log_path = os.path.join(
                task_params["local_workspace"], 
                task_params["task_folder"], 
                task_params["logs_folder"], 
                log_name, 
                "rl_games", 
                f"{task_params['task_folder']}_direct"
                )
            # Navigate to the single subfolder under log_path
            if os.path.exists(log_path):
                subfolders = [f for f in os.listdir(log_path) if os.path.isdir(os.path.join(log_path, f))]
                if len(subfolders) == 1:
                    log_path = os.path.join(log_path, subfolders[0])
                    logger.debug(f"Updated log_path to subfolder: {log_path}")
                elif len(subfolders) > 1:
                    logger.warning(f"Multiple subfolders found in {log_path}, using first: {subfolders[0]}")
                    log_path = os.path.join(log_path, subfolders[0])
            os.makedirs(os.path.join(log_path, "training_record"), exist_ok=True)

            # Find TensorBoard path
            summaries_dir = os.path.join(log_path, "summaries")
            if not os.path.exists(summaries_dir):
                logger.error(f"Summaries directory not found: {summaries_dir}")
                return None

            summary_files = os.listdir(summaries_dir)
            if not summary_files:
                logger.error(f"No summary files found in: {summaries_dir}")
                return None

            tb_path = os.path.join(summaries_dir, summary_files[0])

            # Summarize tensorboard
            summary_output = os.path.join(log_path, "training_record", "training_summary.txt")
            self.summarize_tensorboard(tb_path, summary_output)

            # Get consecutive successes metric
            consecutive_successes_events = self.read_tensorboard_scalar(tb_path, 'Episode/consecutive_successes')

            if not consecutive_successes_events:
                logger.warning(f"No consecutive_successes events found for {log_name}")
                max_con_successes = 0.0
            else:
                max_con_successes = max(event.value for event in consecutive_successes_events)

            result = EvaluationResult(
                log_path=log_path,
                max_consecutive_successes=max_con_successes,
                tb_path=tb_path,
                idx=idx,
                machine=machine
            )

            logger.info(f"Processed result: {result}")
            return result

        except Exception as e:
            logger.error(f"Error processing training result {idx}: {e}")
            return None

    def read_tensorboard_scalar(self, tb_file: str, tag: str) -> Optional[List]:
        """
        Read a scalar metric from a TensorBoard event file.

        Args:
            tb_file: Path to the TensorBoard event file
            tag: Metric tag to extract (e.g., 'Episode/consecutive_successes')

        Returns:
            List of scalar events if found, None otherwise
        """
        if not os.path.exists(tb_file):
            logger.error(f"TensorBoard file not found: {tb_file}")
            return None

        try:
            logger.debug(f"Reading metric '{tag}' from: {tb_file}")

            # Load the event file
            ea = event_accumulator.EventAccumulator(tb_file)
            ea.Reload()

            # Extract the metric if available
            if tag in ea.scalars.Keys():
                events = ea.Scalars(tag)
                if events:
                    logger.debug(f"Found {len(events)} events for tag '{tag}'")
                    return events
                else:
                    logger.warning(f"Tag '{tag}' exists but has no events")
                    return None
            else:
                logger.warning(f"Metric '{tag}' not found in TensorBoard logs")
                available_tags = ea.scalars.Keys()
                logger.debug(f"Available tags: {available_tags}")
                return None

        except Exception as e:
            logger.error(f"Error reading TensorBoard file: {e}")
            return None

    def summarize_tensorboard(self, event_file_path: str, output_txt_path: str):
        """
        Read a TensorBoard event file, summarize scalar data, and write
        a human-readable summary for LLM analysis.

        Args:
            event_file_path: Path to the .tfevents file
            output_txt_path: Path where the output .txt summary will be saved
        """
        try:
            # Initialize EventAccumulator
            size_guidance = {
                event_accumulator.COMPRESSED_HISTOGRAMS: 0,
                event_accumulator.IMAGES: 0,
                event_accumulator.AUDIO: 0,
                event_accumulator.SCALARS: 0,
                event_accumulator.HISTOGRAMS: 0,
            }

            acc = event_accumulator.EventAccumulator(event_file_path, size_guidance=size_guidance)
            acc.Reload()

            # Get all scalar tags
            scalar_tags = acc.Tags()['scalars']

            summary_lines = []
            summary_lines.append("## Reinforcement Learning Model Performance Summary\n")
            summary_lines.append(f"Source File: {os.path.basename(event_file_path)}\n")
            summary_lines.append("-" * 40 + "\n")

            for tag in scalar_tags:
                # Get all scalar events for the current tag
                events = acc.Scalars(tag)

                # Extract values into numpy array
                values = np.array([e.value for e in events])

                if len(values) == 0:
                    continue

                # Calculate summary statistics
                mean_val = np.mean(values)
                std_val = np.std(values)
                max_val = np.max(values)
                min_val = np.min(values)

                # Calculate trend data
                initial_idx = max(int(len(values) * 0.1), 1)
                mid_idx = int(len(values) * 0.5)

                initial_perf = np.mean(values[:initial_idx]) if initial_idx > 0 else values[0]
                mid_perf = values[mid_idx]
                final_perf = np.mean(values[-initial_idx:]) if initial_idx > 0 else values[-1]

                # Format the output
                summary_lines.append(f"## Metric: {tag}\n")
                summary_lines.append(f"- **Overall Statistics:**")
                summary_lines.append(f"  - Mean: {mean_val:.4f}")
                summary_lines.append(f"  - Std Dev: {std_val:.4f} (Measures stability/variance)")
                summary_lines.append(f"  - Max Value: {max_val:.4f}")
                summary_lines.append(f"  - Min Value: {min_val:.4f}\n")

                summary_lines.append(f"- **Performance Trend:**")
                summary_lines.append(f"  - Initial Performance (first 10%): ~{initial_perf:.4f}")
                summary_lines.append(f"  - Mid-Training Performance (at 50%): ~{mid_perf:.4f}")
                summary_lines.append(f"  - Final Performance (last 10%): ~{final_perf:.4f}\n")
                summary_lines.append("-" * 40 + "\n")

            # Write to file
            with open(output_txt_path, 'w') as f:
                f.write("\n".join(summary_lines))

            logger.info(f"Summary written to {output_txt_path}")

        except Exception as e:
            logger.error(f"Error summarizing TensorBoard file: {e}")

    @staticmethod
    def select_best_result(results: List[EvaluationResult], metric: str = 'max_consecutive_successes') -> Optional[EvaluationResult]:
        """
        Select the best result from a list of evaluation results.

        Args:
            results: List of EvaluationResult objects
            metric: Metric to use for selection (default: 'max_consecutive_successes')

        Returns:
            Best EvaluationResult, or None if results list is empty
        """
        if not results:
            logger.warning("No results to select from")
            return None

        if metric == 'max_consecutive_successes':
            best = max(results, key=lambda x: x.max_consecutive_successes)
        else:
            logger.warning(f"Unknown metric '{metric}', using first result")
            best = results[0]

        logger.info(f"Best result: {best}")
        return best


# Maintain backward compatibility - standalone functions
def read_tb(tb_file: str, tag: str) -> Optional[List]:
    """
    Legacy function for reading TensorBoard scalars.

    DEPRECATED: Use ResultProcessor.read_tensorboard_scalar() instead.
    """
    processor = ResultProcessor({'logs_path': os.path.dirname(tb_file)})
    return processor.read_tensorboard_scalar(tb_file, tag)


def summarize_tensorboard(event_file_path: str, output_txt_path: str):
    """
    Legacy function for summarizing TensorBoard logs.

    DEPRECATED: Use ResultProcessor.summarize_tensorboard() instead.
    """
    processor = ResultProcessor({'logs_path': os.path.dirname(event_file_path)})
    processor.summarize_tensorboard(event_file_path, output_txt_path)
