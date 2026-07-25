"""
Evaluation module for local reward-function evaluation.

ARD proposes reward functions, injects each into the ard-isaaclab-tasks substrate,
and trains each one on the local machine (one job at a time) via docker. This
package owns codebase preparation, job execution, and result processing.

The evaluator supports two execution backends: ``local`` builds + runs each
candidate's Dockerfile on this machine one at a time; ``hpc`` builds + pushes each
candidate's image and submits the batch to the CARES HPC Scheduler, training them
concurrently and recycling artifacts from the NAS.

Main classes:
- RewardEvaluator:  dispatch + capture orchestrator (runs jobs, collects output)
- FitnessScorer:    reads the fitness metric and selects the batch winner
- LocalRunner:      builds + runs each candidate's Dockerfile locally
- HPCRunner:        builds + pushes each candidate's image, drives the CARES scheduler
- WorkspaceManager: builds per-candidate job codebases (AST reward injection)
- ResultProcessor:  unpacks artifacts and writes the scalar summary

Example:
    >>> from src.evaluation import RewardEvaluator, FitnessScorer
    >>> from src.reward_history import RewardHistory
    >>> evaluator = RewardEvaluator(
    ...     tasks_repo="/home/lee/code/ard-isaaclab-tasks",
    ...     env_file_rel="source/ard_tasks/ard_tasks/tasks/direct/cartpole/cartpole_env.py",
    ...     task="Isaac-ARD-Cartpole-v0",
    ...     runner={"use_gpu": True},
    ...     output_dir="./runs/cartpole",
    ... )
    >>> evaluator.evaluate(records)   # dispatch + capture
    >>> best = FitnessScorer().score_all(records) and FitnessScorer().select_best(records)
"""

from .evaluator import RewardEvaluator
from .scorer import FitnessScorer
from .local_runner import LocalRunner, LocalRunnerError
from .hpc_runner import HPCRunner, HPCRunnerError, HPCJob
from .workspace_manager import WorkspaceManager
from .reward_injection import inject_reward, extract_method_source, RewardInjectionError
from .result_processor import ResultProcessor, CapturedArtifacts
from . import config

__all__ = [
    "RewardEvaluator",
    "FitnessScorer",
    "LocalRunner",
    "LocalRunnerError",
    "HPCRunner",
    "HPCRunnerError",
    "HPCJob",
    "WorkspaceManager",
    "ResultProcessor",
    "CapturedArtifacts",
    "inject_reward",
    "extract_method_source",
    "RewardInjectionError",
    "config",
]

__version__ = "0.2.0"
