# Dockerfile — ARD (Autonomous RL Designer) orchestrator, the client side.
#
# This packages the ARD refinement loop itself (the thing that runs `main.py
# --refine`): the LLM reward generator, the evaluator, and the result/scoring
# code. It deliberately does NOT contain Isaac Lab / rl_games / torch — training
# runs elsewhere, inside the ard-isaaclab-tasks image:
#   * coordinator mode — jobs are dispatched to the PCS coordinator;
#   * local mode       — ARD builds+runs the ard-isaaclab-tasks Dockerfile via
#                        the host docker daemon (see src/evaluation/local_runner.py).
#
# So this image stays tiny (a few hundred MB) and only needs Python + the docker
# CLI (the latter only used by local mode, to talk to a mounted host socket).
#
# Build (name images with your UPI per CARES policy):
#   docker build -t <UPI>_ard:latest .
#
# Run — coordinator mode (lightweight client; run on YOUR OWN workstation):
#   This image only dispatches to the PCS coordinator, so it needs no GPU and is
#   not meant for the shared GPU boxes. Run it on your laptop:
#     docker run --rm -e OPENROUTER_API_KEY -e PCS_TOKEN \
#       -v "$PWD/configs:/app/configs" -v "$PWD/runs:/app/runs" \
#       <UPI>_ard:latest --refine --task cartpole
#
# ── CARES shared GPU machines ────────────────────────────────────────────────
# IMPORTANT: ARD's *local* backend (src/evaluation/local_runner.py) shells out to
# `docker build`/`docker run` on the host daemon. That is docker-out-of-docker and
# is NOT permitted on the CARES machines: policy allows mounting only
# student_data/<UPI> (no docker socket, no arbitrary paths) and all work must run
# inside one tracked container. So the local *docker* backend is for your own
# workstation only.
#
# To run the single-machine loop ON a CARES box you need a combined image (ARD +
# Isaac Lab + ard_tasks) that trains via in-container subprocesses — a `native`
# backend — writing all outputs into the mounted student_data/<UPI>. Launch it
# with the mandatory CARES flags (see scripts/cares_run.sh):
#   docker run -d -u "$(id -u):$(id -g)" \
#     --name <UPI>_ard --label student_id=<UPI> \
#     -v /home/myuser1/student_data/<UPI>:/workspace \
#     -e HOME=/workspace -w /workspace --gpus all \
#     -e OPENROUTER_API_KEY \
#     <UPI>_ard:latest \
#     --refine --task cartpole --settings /workspace/configs/settings.yaml
#   (Put settings.yaml under student_data so you can edit it without rebuilding;
#    set output_dir + work_root under /workspace and the local backend to native.)
# The combined image + native backend are not in this client image — see the note
# returned with this file for how to add them.

FROM python:3.11-slim

# docker CLI (client only — no engine) so local mode can drive a mounted host
# daemon. Pulled as the official static binary so the image carries just `docker`.
# Coordinator-mode users can ignore it (it adds ~50 MB).
ARG DOCKER_CLI_VERSION=27.3.1
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl ca-certificates \
 && curl -fsSL "https://download.docker.com/linux/static/stable/x86_64/docker-${DOCKER_CLI_VERSION}.tgz" \
      | tar -xz -C /usr/local/bin --strip-components=1 docker/docker \
 && apt-get purge -y curl \
 && apt-get autoremove -y \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first so edits to source don't bust the dependency layer.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Application source (configs/ included as defaults; mount over them to override).
COPY . .

# Sensible runtime defaults; secrets/keys come in via `-e` at run time.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# `main.py` is the entrypoint; pass refinement flags as `docker run` args, e.g.
#   docker run ard:latest --refine --task cartpole
# With no args it prints usage.
ENTRYPOINT ["python", "main.py"]
CMD ["--help"]
