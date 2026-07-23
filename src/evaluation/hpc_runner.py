"""
HPC job runner — build+push one reward candidate's image and submit it to the
CARES HPC Scheduler, then reap its artifacts from the NAS.

Where :class:`~src.evaluation.local_runner.LocalRunner` builds the candidate's
image and *blocks* on ``docker run``, the HPC backend splits dispatch from
capture so the whole batch trains **concurrently** on the cluster:

    submit(tarball, tag, command, ...) -> HPCJob   build+push image, POST /jobs
    poll(job_id)                       -> status    non-blocking single check
    collect(job_id, dest)              -> work_dir   copy NAS/<job_id> -> dest

Reward delivery (per-candidate image). The scheduler *pulls a prebuilt image*,
so each candidate's injected ``_get_rewards`` must be baked into its own image
tag. We reuse the exact ``.tar.gz`` :class:`WorkspaceManager` already builds for
the local backend as the ``docker build`` context, tag it
``<registry>/<repo>:<tag>``, push it, and submit a job that pulls it. This is the
path the prototype ``ard_iter1_run_1`` NAS job used
(``image: …/cli797_ard-isaaclab:iter1_run_1``).

Config travels in the job ``command``, not ``env``. The CARES scheduler drops the
job ``env`` block, so task/seed/tunables are packed into the
``hpc_entrypoint.sh`` flags by the evaluator and handed here as ``command``.

Outputs. The scheduler preserves ``/workspace/output`` to the NAS at
``/cares-nas/hpc/outputs/<upi>/<job_id>``, mounted locally at ``nas_outputs``
(``~/hpc_outputs``) as ``<nas_outputs>/<job_id>``. Once a job reaches a terminal
status, :meth:`collect` copies that tree into the job's ``<output_dir>/<tag>/`` so
:class:`ResultProcessor` reads it exactly as in local mode.

Authentication reuses the client's stored session (``~/.hpc/config.json``). If it
has expired and ``$HPC_PASSWORD`` is set, the runner re-logs in automatically;
otherwise it raises with the exact ``hpc-client login`` fix.
"""

import os
import re
import time
import shutil
import logging
import subprocess
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from . import config

logger = logging.getLogger(__name__)

try:
    from hpc_client import HPCClient
    from hpc_client.config import HPCConfig
    from hpc_client.client import TERMINAL_STATUSES
    from hpc_client.errors import HPCAuthenticationError, HPCClientError
except ImportError:  # pragma: no cover - surfaced lazily in __init__
    HPCClient = None
    HPCConfig = None
    TERMINAL_STATUSES = {"completed", "failed", "cancelled", "deleted"}

    class HPCClientError(RuntimeError):
        pass

    class HPCAuthenticationError(HPCClientError):
        pass


class HPCRunnerError(RuntimeError):
    """Raised for infrastructure failures (docker/registry/scheduler/auth).

    A candidate's own training failure is not this — it surfaces as a terminal
    job status the monitor loop reads, mirroring LocalRunner's RunResult.
    """


@dataclass
class HPCJob:
    """Handle for one submitted HPC job."""
    job_id: str
    tag: str            # the candidate's job tag (e.g. iter1_run_0)
    image: str          # full registry ref pushed + pulled by the job
    job_name: str       # name shown in `hpc-client jobs`
    status: str = "submitted"


def job_id_of(response: Any) -> str:
    """Pull the job id out of a submit response (str or dict, depending on server)."""
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        for key in ("job_id", "id"):
            if key in response:
                return str(response[key])
        job = response.get("job")
        if isinstance(job, dict) and "job_id" in job:
            return str(job["job_id"])
    return str(response)


class HPCRunner:
    """
    Build + push each candidate's image and drive it on the CARES HPC Scheduler.

    Args:
        registry: CARES container registry host, e.g. ``130.216.238.2:5500``.
        image_repo: Image repository under the registry, e.g. ``cli797_ard-isaaclab``.
        nas_outputs: Local mount of the CARES NAS outputs share (``~/hpc_outputs``),
            under which each job's artifacts appear as ``<nas_outputs>/<job_id>``.
        max_active_jobs: Guard against the scheduler's active-job cap (default 50).
    """

    def __init__(
        self,
        registry: str = config.DEFAULT_HPC_REGISTRY,
        image_repo: str = config.DEFAULT_HPC_IMAGE_REPO,
        nas_outputs: str = config.DEFAULT_HPC_NAS_OUTPUTS,
        max_active_jobs: int = config.DEFAULT_HPC_MAX_ACTIVE_JOBS,
    ):
        if HPCClient is None:
            raise HPCRunnerError(
                "hpc_client is not installed. Install the HPC client into this "
                "interpreter, e.g.:\n    pip install -e /home/lee/hpc-client"
            )
        self.registry = registry.rstrip("/")
        self.image_repo = image_repo
        self.nas_outputs = os.path.abspath(os.path.expanduser(nas_outputs))
        self.max_active_jobs = int(max_active_jobs)

        self._docker = shutil.which("docker")
        if not self._docker:
            raise HPCRunnerError(
                "docker not found on PATH; the hpc backend needs docker to "
                "build + push each candidate's image"
            )

        self.client = self._connect()
        logger.info(
            f"HPCRunner ready: registry={self.registry} repo={self.image_repo} "
            f"nas={self.nas_outputs} -> submit-all, monitor concurrently"
        )

    # --------------------------------------------------------------- auth/health
    def _connect(self) -> "HPCClient":
        """Load the stored session, re-logging in with $HPC_PASSWORD if expired."""
        try:
            client = HPCClient.from_config()
        except RuntimeError as error:
            raise HPCRunnerError(
                f"{error}\nConfigure the client first:\n"
                "    hpc-client configure --scheduler-url http://<scheduler-host>:8080\n"
                "    hpc-client login <upi>"
            )
        try:
            client.me()
            return client
        except HPCAuthenticationError:
            return self._relogin(client)

    def _relogin(self, client: "HPCClient") -> "HPCClient":
        """Re-establish an expired session from $HPC_PASSWORD, or fail with the fix."""
        cfg = HPCConfig.load()
        password = os.environ.get("HPC_PASSWORD")
        if not (cfg.username and password):
            raise HPCRunnerError(
                "HPC session expired. Run `hpc-client login <upi>`, or set "
                "$HPC_PASSWORD to have ARD re-login automatically."
            )
        logger.info(f"[hpc] session expired; re-logging in as {cfg.username}")
        client.login(username=cfg.username, password=password)
        return client

    def healthz(self) -> bool:
        """True if docker answers and the HPC session is valid."""
        try:
            proc = subprocess.run(
                [self._docker, "version", "--format", "{{.Server.Version}}"],
                capture_output=True, text=True, timeout=30,
            )
            if proc.returncode != 0:
                logger.error("docker daemon did not answer")
                return False
        except (OSError, subprocess.SubprocessError) as e:
            logger.error(f"docker health check failed: {e}")
            return False
        try:
            self.client.me()
            return True
        except HPCClientError as e:
            logger.error(f"HPC scheduler health check failed: {e}")
            return False

    def active_job_count(self) -> int:
        """How many jobs the user currently has active (queued/running/…).

        ``jobs()`` returns the user's whole history (completed/failed included),
        but only non-terminal jobs count against the scheduler's active-job cap,
        so terminal ones are filtered out here.
        """
        try:
            jobs = self.client.jobs()
        except HPCAuthenticationError:
            self._relogin(self.client)
            jobs = self.client.jobs()
        except HPCClientError as e:
            logger.warning(f"could not read active job count: {e}")
            return 0
        return sum(1 for j in jobs if j.get("status") not in TERMINAL_STATUSES)

    # ------------------------------------------------------------------- utils
    def _image_tag(self, tag: str) -> str:
        """Docker-tag-safe label for a candidate (lowercase, restricted charset)."""
        safe = re.sub(r"[^a-z0-9_.-]", "-", tag.lower()).strip("-.")
        return safe or "job"

    # ------------------------------------------------------------------ submit
    def submit(
        self,
        tarball_path: str,
        tag: str,
        command: str,
        max_runtime_hours: float = config.DEFAULT_HPC_MAX_RUNTIME_HOURS,
        datasets: Optional[List[str]] = None,
        job_name: Optional[str] = None,
    ) -> HPCJob:
        """
        Build the candidate's image from ``tarball_path``, push it, and submit a job.

        Raises :class:`HPCRunnerError` for any infrastructure failure (build,
        push, or submit); the evaluator marks that record build-failed and moves
        on, exactly as the local backend does for a failed ``docker build``.
        """
        image = f"{self.registry}/{self.image_repo}:{self._image_tag(tag)}"
        job_name = job_name or f"{config.DEFAULT_HPC_JOB_NAME_PREFIX}_{tag}"

        # 1) Build the per-candidate image from the tarball (Dockerfile at its
        #    root), streamed on stdin exactly like LocalRunner. This bakes the
        #    injected reward into an image the scheduler can pull.
        logger.info(f"[{tag}] docker build -> {image}")
        try:
            with open(tarball_path, "rb") as ctx:
                proc = subprocess.run(
                    [self._docker, "build", "-t", image, "-"],
                    stdin=ctx, capture_output=True, text=True,
                )
        except OSError as e:
            raise HPCRunnerError(f"docker build invocation failed: {e}")
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "")[-1500:]
            logger.error(f"[{tag}] image build failed:\n{tail}")
            raise HPCRunnerError(f"docker build failed: {tail[-500:]}")

        # 2) Push it to the CARES registry so a worker can pull it. The rebuild is
        #    incremental: only the ard_tasks source + editable-install layers
        #    change per reward, so a re-push uploads a few MB, not the base.
        logger.info(f"[{tag}] docker push {image}")
        try:
            proc = subprocess.run(
                [self._docker, "push", image], capture_output=True, text=True,
            )
        except OSError as e:
            raise HPCRunnerError(f"docker push invocation failed: {e}")
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "")[-1500:]
            logger.error(f"[{tag}] image push failed:\n{tail}")
            raise HPCRunnerError(
                f"docker push failed (is '{self.registry}' in Docker's "
                f"insecure-registries?): {tail[-500:]}"
            )

        # 3) Submit the job. env is left empty on purpose — the scheduler drops
        #    it; all configuration is already in `command`.
        spec = {
            "job_name": job_name,
            "image": image,
            "command": command,
            "max_runtime_hours": float(max_runtime_hours),
            "required_datasets": list(datasets or []),
            "required_worker_ids": [],
            "env": {},
        }
        try:
            response = self.client.submit(spec)
        except HPCAuthenticationError:
            self._relogin(self.client)
            response = self.client.submit(spec)
        except HPCClientError as e:
            raise HPCRunnerError(f"submit failed for {job_name}: {e}")

        job_id = job_id_of(response)
        logger.info(f"[{tag}] submitted {job_name}: {job_id}")
        return HPCJob(job_id=job_id, tag=tag, image=image, job_name=job_name)

    # -------------------------------------------------------------------- poll
    def poll(self, job_id: str) -> str:
        """Return the job's current status (one non-blocking scheduler read)."""
        try:
            data = self.client.job(job_id)
        except HPCAuthenticationError:
            self._relogin(self.client)
            data = self.client.job(job_id)
        except HPCClientError as e:
            logger.warning(f"[{job_id}] status poll failed: {e}")
            return "unknown"
        job = data.get("job", data) if isinstance(data, dict) else {}
        return job.get("status", "unknown")

    @staticmethod
    def is_terminal(status: str) -> bool:
        """True once a status will no longer change (job left the scheduler)."""
        return status in TERMINAL_STATUSES

    # ----------------------------------------------------------------- collect
    def collect(self, job_id: str, dest_dir: str) -> Optional[str]:
        """
        Copy a finished job's NAS artifacts (``<nas_outputs>/<job_id>``) into ``dest_dir``.

        Returns ``dest_dir`` on success, or ``None`` if the NAS folder never
        appeared (e.g. the job died before the scheduler preserved any output).
        The copied tree keeps the scheduler's layout (``logs/rl_games/…`` +
        ``outputs/``), which :class:`ResultProcessor` reads unchanged.
        """
        src = os.path.join(self.nas_outputs, job_id)
        # The scheduler's NAS copy can lag a terminal status slightly; give it a
        # few seconds to appear before giving up.
        for _ in range(6):
            if os.path.isdir(src):
                break
            time.sleep(5)
        else:
            logger.error(f"[{job_id}] no NAS output at {src}")
            return None

        os.makedirs(dest_dir, exist_ok=True)
        logger.info(f"[{job_id}] recycling artifacts {src} -> {dest_dir}")
        shutil.copytree(src, dest_dir, dirs_exist_ok=True)
        return dest_dir
