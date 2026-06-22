#!/usr/bin/env bash
# Run ARD inside its Docker image, wiring the mounts each backend needs.
#
# ARD has two execution backends (configs/settings.yaml -> coordinator.mode);
# this script reads that mode (and tasks_repo) straight from your settings file
# and sets up the right container accordingly:
#
#   coordinator  ARD is a pure HTTP client of a PCS coordinator. The container
#                needs only the LLM key, the coordinator token, and a read-only
#                mount of the ard-isaaclab-tasks checkout it stages codebases from.
#                No docker socket, no GPU.
#
#   local        ARD builds + runs each candidate's ard-isaaclab-tasks Dockerfile
#                itself — one GPU task container per evaluation. The container
#                therefore drives the *host* docker daemon (docker-out-of-docker):
#                we mount the docker socket AND bind the repo + tasks repo at their
#                identical host paths. That last part is load-bearing: the per-job
#                `work_dir` ARD passes to `docker run -v` is resolved by the HOST
#                daemon, so it must name a real host path, not a path that only
#                exists inside this container.
#
# Usage:
#   export OPENROUTER_API_KEY=...           # always
#   export PCS_TOKEN=pcs_...                # coordinator mode only
#   scripts/docker_run.sh [--build] [--detach|-d] [--] [main.py args...]
#
#   scripts/docker_run.sh --build -- --refine --task cartpole
#   scripts/docker_run.sh --refine                 # default args: --refine
#   scripts/docker_run.sh -d --refine              # background; docker logs -f ard-run
#
# Env knobs:
#   ARD_IMAGE   image tag to build / run   (default: ard:latest)
#   SETTINGS    settings file to read      (default: configs/settings.yaml)
#   ARD_NAME    container name when -d      (default: ard-run)
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$PWD"

ARD_IMAGE="${ARD_IMAGE:-ard:latest}"
SETTINGS="${SETTINGS:-configs/settings.yaml}"
ARD_NAME="${ARD_NAME:-ard-run}"          # container name when detached (-d)

# --- parse script flags vs. forwarded main.py args --------------------------
# --build / --detach may appear in any order; the first non-flag (or `--`)
# starts the main.py args.
DO_BUILD=0
DETACH=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --build)     DO_BUILD=1; shift ;;
        -d|--detach) DETACH=1;   shift ;;
        --)          shift; break ;;
        *)           break ;;
    esac
done
ARD_ARGS=("$@")
[[ ${#ARD_ARGS[@]} -eq 0 ]] && ARD_ARGS=(--refine)

# --- build the image on request, or if it doesn't exist yet -----------------
if [[ $DO_BUILD -eq 1 ]] || ! docker image inspect "$ARD_IMAGE" >/dev/null 2>&1; then
    echo ">> docker build -t $ARD_IMAGE"
    docker build -t "$ARD_IMAGE" "$REPO"
fi

# --- read mode + tasks_repo from the settings file --------------------------
read -r MODE TASKS_REPO < <(python3 - "$SETTINGS" <<'PY'
import os, sys, yaml
cfg = yaml.safe_load(open(sys.argv[1])) or {}
mode = str(cfg.get("coordinator", {}).get("mode", "coordinator")).lower()
print(mode, os.path.abspath(os.path.expanduser(cfg["tasks_repo"])))
PY
)
[[ -d "$TASKS_REPO" ]] || { echo "tasks_repo not found: $TASKS_REPO" >&2; exit 1; }

: "${OPENROUTER_API_KEY:?export OPENROUTER_API_KEY first}"

# Detached (-d): background under a known --name so you can follow/stop it.
# Attached: keep stdin (+TTY when present) so tqdm bars stay tidy. --rm either
# way — on a shared box, don't leak stopped containers.
common=(--rm -e OPENROUTER_API_KEY -e PCS_TOKEN)
if [[ $DETACH -eq 1 ]]; then
    common+=(-d --name "$ARD_NAME")
else
    common+=(-i)
    [[ -t 1 ]] && common+=(-t)
fi

# Attached: hand the terminal to docker (exec). Detached: `docker run -d` just
# prints the id and returns, so run it and then print how to follow/stop.
launch() {
    if [[ $DETACH -eq 1 ]]; then
        docker run "$@"
        echo ">> detached as '$ARD_NAME'"
        echo "   follow:  docker logs -f $ARD_NAME"
        echo "   stop:    docker stop $ARD_NAME"
    else
        exec docker run "$@"
    fi
}

if [[ "$MODE" == "local" ]]; then
    echo ">> local (docker-out-of-docker) mode | tasks_repo=$TASKS_REPO"
    sock=/var/run/docker.sock
    [[ -S "$sock" ]] || { echo "no docker socket at $sock" >&2; exit 1; }
    DOCKER_GID="$(getent group docker | cut -d: -f3 || true)"

    # Run as the host user so the files ARD writes (and the task containers it
    # launches with `-u $(id -u)`) are owned by you, not root. --group-add gives
    # that uid access to the docker socket. The repo + tasks repo are bound at
    # their own host paths so every `docker run -v <path>` ARD emits resolves on
    # the host; HOME points into the mounted repo for any cache writes.
    launch "${common[@]}" \
        -u "$(id -u):$(id -g)" \
        ${DOCKER_GID:+--group-add "$DOCKER_GID"} \
        -e HOME="$REPO" \
        -v "$sock:$sock" \
        -v "$REPO:$REPO" \
        -v "$TASKS_REPO:$TASKS_REPO:ro" \
        -w "$REPO" \
        --entrypoint python \
        "$ARD_IMAGE" main.py "${ARD_ARGS[@]}"
else
    echo ">> coordinator mode | tasks_repo=$TASKS_REPO"
    # Pure HTTP client: persist runs/, keep configs editable, mount the tasks
    # repo read-only at the path settings.yaml names. The baked /app code runs.
    launch "${common[@]}" \
        -v "$TASKS_REPO:$TASKS_REPO:ro" \
        -v "$REPO/runs:/app/runs" \
        -v "$REPO/configs:/app/configs:ro" \
        "$ARD_IMAGE" "${ARD_ARGS[@]}"
fi
