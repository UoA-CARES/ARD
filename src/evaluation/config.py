"""
Configuration and constants for the evaluation module.

This module contains default values, patterns, and configuration
used throughout the evaluation pipeline.
"""

# Default regex pattern for matching reward functions in environment files
# Matches: @torch.jit.script def compute_rewards(...) ... return total_reward, reward_components
REWARD_FUNCTION_PATTERN = r'@torch\.jit\.script\s*\n*def\s+compute_rewards\s*\([^)]*\).*?return\s+total_reward, reward_components'

# Default log name template for evaluation runs
# LOG_NAME_TEMPLATE = "eval_{idx}"

# Default TensorBoard metric for evaluation
PRIMARY_METRIC = "Episode/consecutive_successes"

# Default aggregation method for selecting best result
METRIC_AGGREGATION = "max"

# Default timeout for training processes (in seconds)
# 1 hour = 3600 seconds
DEFAULT_TRAINING_TIMEOUT = 3600

# Default number of retry attempts for failed training runs
DEFAULT_MAX_RETRIES = 0

# Git operations timeout (in seconds)
GIT_OPERATION_TIMEOUT = 30

# Remote pipeline script name
REMOTE_PIPELINE_SCRIPT = "run_remote_pipeline.sh"

# TensorBoard summary size guidance
TENSORBOARD_SIZE_GUIDANCE = {
    'compressed_histograms': 0,
    'images': 0,
    'audio': 0,
    'scalars': 0,
    'histograms': 0,
}

# Training record subdirectory name
TRAINING_RECORD_DIR = "training_record"

# Training summary filename
TRAINING_SUMMARY_FILE = "training_summary.txt"

# TensorBoard summaries subdirectory
TENSORBOARD_SUMMARIES_DIR = "summaries"
