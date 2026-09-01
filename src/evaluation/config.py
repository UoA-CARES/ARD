"""
Configuration and constants for the evaluation module.

Defaults for local evaluation of LLM-proposed reward functions against the
ard-isaaclab-tasks substrate.
"""

import os

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
VLM_FEEDBACK_FILE = "vlm_feedback.txt"

# --------------------------------------------------------------------------- #
# Warm-start checkpoint delivery                                              #
# --------------------------------------------------------------------------- #
# A warm-start checkpoint is baked into the candidate's job codebase tarball
# (the same tarball WorkspaceManager already builds as the docker build
# context for both backends), at this path relative to the repo root, under a
# fixed filename so the in-image path never depends on the source checkpoint's
# original name. Must live under `scripts/` (or `source/`) — the Dockerfile
# only `COPY`s those two directories into the image, not the whole build
# context, so anything staged outside them (e.g. a top-level `warm_start/`)
# silently never reaches the running container.
WARM_START_CHECKPOINT_REL = "scripts/warm_start/checkpoint.pth"

# Where the tarball's contents land inside the built image (Dockerfile COPYs
# the build context here). Matches HPC_ENTRYPOINT below, which both backends'
# images share since they're built from the same tarball.
IMAGE_REPO_ROOT = "/opt/ard-isaaclab-tasks"

# --------------------------------------------------------------------------- #
# HPC backend (CARES HPC Scheduler)                                            #
# --------------------------------------------------------------------------- #
# Which execution backend the evaluator drives. "local" builds + `docker run`s
# each candidate on this machine one at a time; "hpc" builds + pushes a
# per-candidate image and submits it to the CARES HPC Scheduler, so the whole
# batch trains concurrently on the cluster.
DEFAULT_BACKEND = "local"

# CARES container registry + image repository. Each candidate is pushed as
# ``<registry>/<image_repo>:<tag>`` (the tag is the candidate's job tag), which
# is the image the submitted job pulls. Matches scripts/hpc_push.sh in
# ard-isaaclab-tasks.
DEFAULT_HPC_REGISTRY = "130.216.238.2:5500"

# The image repository is per-user: ``<upi>_ard-isaaclab``. A UPI is a personal
# account id, so it is NEVER hardcoded here — supply your own via
# ``runner.hpc.upi`` in configs/settings.yaml, or by exporting $ARD_UPI.
HPC_UPI_ENV_VAR = "ARD_UPI"
HPC_IMAGE_REPO_SUFFIX = "ard-isaaclab"


def hpc_image_repo(upi: str = "") -> str:
    """Return the per-user image repo ``<upi>_ard-isaaclab``.

    Args:
        upi: The UPI to namespace the repository with. Falls back to $ARD_UPI.

    Raises:
        ValueError: If no UPI was configured, with the exact fix.
    """
    upi = (upi or os.environ.get(HPC_UPI_ENV_VAR, "")).strip()
    if not upi:
        raise ValueError(
            "the hpc backend needs your UPI to name the container image "
            "repository (<upi>_" + HPC_IMAGE_REPO_SUFFIX + "). Set "
            "`runner.hpc.upi` in configs/settings.yaml, or export "
            f"{HPC_UPI_ENV_VAR}=<upi>."
        )
    return f"{upi}_{HPC_IMAGE_REPO_SUFFIX}"

# The in-image entrypoint every HPC job runs; ARD appends --task/--seed/… as
# argv because the scheduler does NOT inject the job `env` block (see
# scripts/hpc_entrypoint.sh in ard-isaaclab-tasks).
HPC_ENTRYPOINT = "bash /opt/ard-isaaclab-tasks/scripts/hpc_entrypoint.sh"

# Where the CARES NAS is mounted locally. The scheduler preserves each job's
# /workspace/output to /cares-nas/hpc/outputs/<upi>/<job_id>, exposed here as
# ``<nas_outputs>/<job_id>``; the recycle step copies that tree into ./runs.
DEFAULT_HPC_NAS_OUTPUTS = "~/hpc_outputs"

# Default wall-clock cap requested per HPC job (hours); the scheduler kills a job
# past this. Overestimate rather than under.
DEFAULT_HPC_MAX_RUNTIME_HOURS = 2.0

# How often the monitor loop polls each outstanding job's status (seconds).
DEFAULT_HPC_POLL_SECONDS = 30

# The scheduler rejects submissions once a user has this many active jobs.
DEFAULT_HPC_MAX_ACTIVE_JOBS = 50

# Prefix for submitted job names (the scheduler appends a timestamp + short hash
# to form the unique job_id / NAS folder name), e.g. ard_iter1_run_0.
DEFAULT_HPC_JOB_NAME_PREFIX = "ard"

# How long a terminated job has to produce its NAS output before ARD gives up on
# it. A reward bad enough to fail training still gets its results written back
# (the container ran, so the scheduler preserved /workspace/output), whereas a
# technical crash — driver fault, node power-off, cancellation — dies before
# anything is saved. So an empty result means nothing was measured, and the job
# is re-run rather than scored. Must exceed the scheduler's NAS-copy lag.
DEFAULT_HPC_RESULT_GRACE_SECONDS = 60

# How many times such a job may be resubmitted before ARD gives up (0 disables).
# A retry reuses the image already pushed for that candidate — cluster time only.
DEFAULT_HPC_MAX_RETRIES = 2
