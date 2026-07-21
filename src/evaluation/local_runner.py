"""
Single-machine job runner — build + run one reward candidate on this machine.

ARD's execution model is deliberately boring: for each candidate it builds the
submitted project's ``Dockerfile`` and ``docker run``s the image in a per-job
working directory, then reads the ``logs/`` the container wrote there. The
ard-isaaclab-tasks codebase is already packaged for exactly this (Dockerfile at
the repo root, env-driven ``scripts/pcs_entrypoint.sh``).

``LocalRunner`` is the only execution backend, and it exposes a single blocking
call the evaluator drives in a plain loop:

    healthz()      -> docker reachable?
    run(...)       -> build the image, run the container, return a RunResult

There is no job queue, no submit/wait split, and no artifacts tarball: the run
mounts the job's ``work_dir`` at ``/work``, the container writes ``logs/`` there,
and the evaluator reads those logs in place (:class:`ResultProcessor`). Codebase
staging (:class:`WorkspaceManager`) and scoring stay independent of how jobs run.

Container contract:
  * the project tarball (Dockerfile at root) is the ``docker build`` context, so
    the candidate's injected ``_get_rewards`` is baked into the image per job;
  * the container runs non-root (``-u $(id -u):$(id -g)``) with ``HOME`` and the
    working dir pointed at the per-job mount, into which ``logs/`` is written;
  * task/seed/tunables are passed through the job ``env`` (the image entrypoint
    reads ``TASK``/``SEED``/…), not a baked-in command.

``use_gpu`` attaches the whole local GPU (``--gpus all``); otherwise the job runs
CPU-only. There is no fractional-GPU request — one candidate runs at a time.
"""

import os
import re
import shutil
import logging
import subprocess
from typing import Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class LocalRunnerError(RuntimeError):
    """Raised when docker is unavailable (a job's own failure is a RunResult)."""


@dataclass
class RunResult:
    """Outcome of one local job.

    ``work_dir`` is the per-job ``/work`` mount; the container's ``logs/`` tree
    lives there and the evaluator reads it in place.
    """
    status: str                       # "succeeded" | "failed" | "timed_out"
    work_dir: str
    exit_code: Optional[int] = None
    error: Optional[str] = None
    log_path: Optional[str] = None    # container stdout/stderr capture


class LocalRunner:
    """
    Build + run one reward candidate on the local machine via docker.

    Args:
        image: Docker image *repository* name. Each job is built and run under its
            own ``<repo>:<job-tag>`` tag, then that tag is removed once the job
            finishes.
        use_gpu: Attach the whole local GPU (``--gpus all``). ``False`` runs
            CPU-only. Jobs run one at a time regardless.
        run_as_user: Run the container as the current uid:gid so the files each
            job writes stay owned by you, not root. Set False to run as the
            image's default user.
    """

    def __init__(
        self,
        image: str = "ard-local",
        use_gpu: bool = True,
        run_as_user: bool = True,
    ):
        # Keep only the repository part; the per-job tag is appended at build time.
        self.image_repo = image.split(":", 1)[0]
        self.use_gpu = bool(use_gpu)
        self.run_as_user = run_as_user
        self._docker = shutil.which("docker")
        if not self._docker:
            raise LocalRunnerError(
                "docker not found on PATH; single-machine mode needs docker"
            )
        logger.info(
            f"LocalRunner ready: image={self.image_repo} "
            f"gpu={'on' if self.use_gpu else 'off'} -> one candidate at a time"
        )

    # ------------------------------------------------------------------ utils
    def _run(self, args: List[str], **kwargs) -> subprocess.CompletedProcess:
        return subprocess.run([self._docker, *args], **kwargs)

    def _tag(self, work_dir: str) -> str:
        """Docker-tag-safe label derived from the job's work-dir name."""
        base = os.path.basename(work_dir.rstrip("/")) or "job"
        return re.sub(r"[^a-z0-9_.-]", "-", base.lower()).strip("-.") or "job"

    def healthz(self) -> bool:
        """True if the docker daemon answers."""
        try:
            proc = self._run(
                ["version", "--format", "{{.Server.Version}}"],
                capture_output=True, text=True, timeout=30,
            )
            return proc.returncode == 0
        except (OSError, subprocess.SubprocessError) as e:
            logger.error(f"Local docker health check failed: {e}")
            return False

    # ------------------------------------------------------------------- run
    def run(
        self,
        tarball_path: str,
        work_dir: str,
        env: Optional[Dict[str, str]] = None,
        command: Optional[str] = None,
        build_args: Optional[Dict[str, str]] = None,
        timeout_seconds: int = 3600,
    ) -> RunResult:
        """
        Build the candidate's image and run it, blocking until it finishes.

        Uses a per-job ``<repo>:<tag>`` image (removed at the end so disk doesn't
        grow across iterations) and mounts ``work_dir`` at ``/work``; the
        container writes ``logs/`` there. Never raises for a job-level failure —
        the outcome is carried in the returned :class:`RunResult`.
        """
        tag = self._tag(work_dir)
        image = f"{self.image_repo}:{tag}"
        name = re.sub(r"[^a-zA-Z0-9_.-]", "-", f"ard_{tag}")
        work_dir = os.path.abspath(work_dir)
        if os.path.exists(work_dir):
            shutil.rmtree(work_dir, ignore_errors=True)
        os.makedirs(work_dir, exist_ok=True)
        log_path = os.path.join(work_dir, "local_run.log")

        try:
            # 1) Build the image from the candidate tarball (Dockerfile at its
            #    root). The build context is the gzipped tar piped on stdin; docker
            #    auto-detects the compression. This bakes the injected reward in.
            args = []
            for k, v in (build_args or {}).items():
                args += ["--build-arg", f"{k}={v}"]
            logger.info(f"[{tag}] docker build -> {image}")
            try:
                with open(tarball_path, "rb") as ctx:
                    proc = self._run(
                        ["build", "-t", image, *args, "-"],
                        stdin=ctx, capture_output=True, text=True,
                    )
            except OSError as e:
                return RunResult(status="failed", work_dir=work_dir,
                                 error=f"docker build invocation failed: {e}")
            if proc.returncode != 0:
                tail = (proc.stderr or proc.stdout or "")[-1500:]
                logger.error(f"[{tag}] image build failed:\n{tail}")
                return RunResult(status="failed", work_dir=work_dir,
                                 exit_code=proc.returncode,
                                 error=f"docker build failed: {tail[-500:]}")

            # 2) Run the container: non-root, HOME + workdir on the per-job mount,
            #    env-driven entrypoint. logs/ lands in work_dir.
            run_args = ["run", "--rm", "--name", name]
            if self.use_gpu:
                run_args += ["--gpus", "all"]
            if self.run_as_user:
                run_args += ["-u", f"{os.getuid()}:{os.getgid()}"]
                # A non-root host uid often has no /etc/passwd entry inside the
                # task image, so getpass.getuser() -> pwd.getpwuid() raises
                # KeyError (torch inductor hits this naming its cache dir).
                # USER/LOGNAME are checked first by getuser(), so setting them
                # skips the pwd lookup.
                run_args += ["-e", "USER=ard", "-e", "LOGNAME=ard"]
            run_args += ["-e", "HOME=/work", "-w", "/work", "-v", f"{work_dir}:/work"]
            for k, v in (env or {}).items():
                run_args += ["-e", f"{k}={v}"]
            run_args.append(image)
            if command:                            # optional CMD override
                run_args += ["sh", "-c", command]

            logger.info(f"[{tag}] docker run (logs -> {log_path})")
            try:
                with open(log_path, "w") as log_fh:
                    proc = self._run(
                        run_args, stdout=log_fh, stderr=subprocess.STDOUT,
                        timeout=timeout_seconds,
                    )
                status = "succeeded" if proc.returncode == 0 else "failed"
                error = None if proc.returncode == 0 else \
                    f"container exited {proc.returncode} (see {log_path})"
                result = RunResult(status=status, work_dir=work_dir,
                                   exit_code=proc.returncode, error=error,
                                   log_path=log_path)
            except subprocess.TimeoutExpired:
                logger.warning(f"[{tag}] timed out after {timeout_seconds}s; killing")
                self._run(["kill", name], capture_output=True)
                result = RunResult(status="timed_out", work_dir=work_dir,
                                   error=f"exceeded timeout {timeout_seconds}s",
                                   log_path=log_path)
            except OSError as e:
                result = RunResult(status="failed", work_dir=work_dir,
                                   error=f"docker run invocation failed: {e}",
                                   log_path=log_path)
        finally:
            # Drop the per-job tag; shared base/cache layers are reference-counted
            # so this only frees this job's thin layers.
            self._run(["rmi", "-f", image], capture_output=True)

        if result.status != "succeeded":
            logger.warning(f"[{tag}] {result.status}: {result.error}")
        return result
