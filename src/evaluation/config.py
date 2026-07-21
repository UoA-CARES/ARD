"""
Configuration and constants for the evaluation module.

Defaults for local evaluation of LLM-proposed reward functions against the
ard-isaaclab-tasks substrate.
"""

# Name of the method ARD rewrites in each task env file (the "sole edit target").
REWARD_METHOD_NAME = "_get_rewards"

# The fixed evaluation metric the tasks log via
# ``self.extras["log"]["fitness_function"]``. Matched by suffix against the
# TensorBoard scalar tags, so any scope prefix the rl_games observer adds still
# resolves (e.g. "Episode/fitness_function").
FITNESS_METRIC = "fitness_function"

# Default per-job wall-clock timeout for a local training run (seconds).
DEFAULT_TRAINING_TIMEOUT = 36000

# Whether a job attaches the local GPU by default (docker run --gpus all).
# A single candidate runs at a time; there is no fractional-GPU request.
DEFAULT_USE_GPU = True

# TensorBoard summary size guidance (load all scalars, no histograms/images).
from tensorboard.backend.event_processing import event_accumulator as _ea  # noqa: E402

TENSORBOARD_SIZE_GUIDANCE = {
    _ea.COMPRESSED_HISTOGRAMS: 0,
    _ea.IMAGES: 0,
    _ea.AUDIO: 0,
    _ea.SCALARS: 0,
    _ea.HISTOGRAMS: 0,
}

# Per-run record subdirectory and summary filename.
TRAINING_RECORD_DIR = "training_record"
TRAINING_SUMMARY_FILE = "training_summary.txt"
