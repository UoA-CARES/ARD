# syntax=docker/dockerfile:1

# ---- ARD (Autonomous RL Designer) orchestrator image -----------------------
# ARD is a thin, CPU-only Python orchestrator. It never trains locally; it either
#   (a) submits jobs to a PCS coordinator over HTTP        (coordinator mode), or
#   (b) builds + runs each candidate's ard-isaaclab-tasks Dockerfile itself,
#       i.e. one GPU task container per evaluation          (local mode).
#
# Because of (b) this image ships the docker CLI: in local mode you mount the
# host docker socket and ARD launches *sibling* task containers on the host
# daemon (docker-out-of-docker). The ARD process itself needs no GPU and no
# Isaac Lab / rl_games stack — those live in the task image the worker builds.
#
# See scripts/docker_run.sh (recommended launcher) and docker-compose.yml for the
# run-time wiring of sockets, env and host-path mounts.

FROM python:3.11-slim-bookworm

# docker CLI only — the daemon stays on the host, reached via the mounted socket.
# The client binary in the official image is a static Go binary, so it runs fine
# on this Debian-slim base. Pinned for reproducibility; an older client happily
# negotiates the API version down to whatever the host daemon serves.
COPY --from=docker:28-cli /usr/local/bin/docker /usr/local/bin/docker

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies first, so edits to the source don't bust the pip cache layer.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Application source (coordinator mode runs this baked copy; local mode bind-
# mounts the working tree over its own host path instead — see docker_run.sh).
COPY main.py ./
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts

# main.py is the entry point; flags pass straight through, e.g.
#   docker run ... ard:latest --refine --task cartpole
# Drop the entrypoint for a shell: docker run ... --entrypoint bash ard:latest
ENTRYPOINT ["python", "main.py"]
CMD ["--help"]
