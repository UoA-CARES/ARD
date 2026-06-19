#!/usr/bin/env bash
# CARES HPC launcher for ARD's single-machine loop.
#
# Wraps the *mandatory* CARES docker flags around an ARD run so containers are
# tracked and files stay yours:
#   -u $(id -u):$(id -g)                       # files owned by you, not root
#   --name <UPI>_ard  --label student_id=<UPI> # admin-trackable (untracked = killed)
#   -v student_data/<UPI>:/workspace           # the ONLY mount CARES permits
#   -e HOME=/workspace  -w /workspace          # Isaac Kit cache + logs land here
#   --gpus all                                 # attach the GPU
#
# This needs a COMBINED image (ARD + Isaac Lab + ard_tasks) that trains via
# in-container subprocesses — the plain client image cannot train on CARES, since
# its `local` backend spawns sibling docker containers (docker-out-of-docker),
# which CARES forbids. Build that image as <UPI>_ard:latest (override with
# $ARD_IMAGE).
#
# Usage:
#   export OPENROUTER_API_KEY=...
#   scripts/cares_run.sh <UPI> [main.py args...]
#   scripts/cares_run.sh hwil292 --refine --task cartpole
#
# Runs detached (-d), as policy recommends for long jobs. Follow it with:
#   docker logs -f <UPI>_ard
# and when your booking ends, clean up:
#   docker stop <UPI>_ard && docker rm <UPI>_ard
set -euo pipefail

UPI="${1:?usage: cares_run.sh <UPI> [ard args...]}"; shift || true
: "${OPENROUTER_API_KEY:?export OPENROUTER_API_KEY first}"

IMAGE="${ARD_IMAGE:-${UPI}_ard:latest}"
CONTAINER_PATH="/workspace"
STUDENT_DATA="/home/myuser1/student_data/${UPI}"

# Default to a settings file living under student_data (editable without rebuild).
if [[ $# -eq 0 ]]; then
    set -- --refine --task cartpole --settings "${CONTAINER_PATH}/configs/settings.yaml"
fi

mkdir -p "${STUDENT_DATA}"

exec docker run -d \
    -u "$(id -u):$(id -g)" \
    --name "${UPI}_ard" \
    --label "student_id=${UPI}" \
    -v "${STUDENT_DATA}:${CONTAINER_PATH}" \
    -e HOME="${CONTAINER_PATH}" \
    -w "${CONTAINER_PATH}" \
    --gpus all \
    -e OPENROUTER_API_KEY \
    "${IMAGE}" \
    "$@"
