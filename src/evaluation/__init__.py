"""
Evaluation module for coordinator-dispatched reward-function evaluation.

ARD proposes reward functions, injects each into the ard-isaaclab-tasks substrate,
and dispatches training to a Parallel Coordination System (PCS) coordinator, which
runs jobs across registered GPU workers. This package owns codebase preparation,
job dispatch, and result processing.

Main classes:
- RewardEvaluator:  dispatch + capture orchestrator (runs jobs, collects output)
- FitnessScorer:    reads the fitness metric and selects the batch winner
- CoordinatorClient: HTTP client for the PCS coordinator
- WorkspaceManager: builds per-candidate job codebases (AST reward injection)
- ResultProcessor:  unpacks artifacts and writes the scalar summary

Example:
    >>> from src.evaluation import RewardEvaluator, FitnessScorer
    >>> from src.reward_history import RewardHistory
    >>> evaluator = RewardEvaluator(
    ...     tasks_repo="/home/lee/code/ard-isaaclab-tasks",
    ...     env_file_rel="source/ard_tasks/ard_tasks/tasks/direct/cartpole/cartpole_env.py",
    ...     task="Isaac-ARD-Cartpole-v0",
    ...     coordinator={"base_url": "http://localhost:8000", "token_env": "TOKEN"},
    ...     output_dir="./runs/cartpole",
    ... )
    >>> evaluator.evaluate(records)   # dispatch + capture
    >>> best = FitnessScorer().score_all(records) and FitnessScorer().select_best(records)
"""

from .evaluator import RewardEvaluator
from .scorer import FitnessScorer
from .coordinator_client import CoordinatorClient, CoordinatorError
from .workspace_manager import WorkspaceManager
from .reward_injection import inject_reward, extract_method_source, RewardInjectionError
from .result_processor import ResultProcessor, CapturedArtifacts
from . import config

__all__ = [
    "RewardEvaluator",
    "FitnessScorer",
    "CoordinatorClient",
    "CoordinatorError",
    "WorkspaceManager",
    "ResultProcessor",
    "CapturedArtifacts",
    "inject_reward",
    "extract_method_source",
    "RewardInjectionError",
    "config",
]

__version__ = "0.2.0"
