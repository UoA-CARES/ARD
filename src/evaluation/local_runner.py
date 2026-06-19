"""
Single-machine job runner — a local stand-in for the PCS coordinator.

The Parallel Coordination System (PCS) is overkill for routine local evaluation:
it brings a coordinator (FastAPI + scheduler + SQLite), SSH-to-worker dispatch,
GPU-slot scheduling and HTTP artifact transfer. But each PCS *worker* ultimately
does something simple — build the submitted project's ``Dockerfile`` and
``docker run`` the image, then collect its ``output_paths``. The ard-isaaclab-tasks
codebase is already packaged for exactly this (Dockerfile at the repo root,
env-driven ``scripts/pcs_entrypoint.sh``).

``LocalRunner`` replicates that one worker on this machine, with no coordinator in
the loop. It is a **drop-in duck-type** for the subset of
:class:`~src.evaluation.coordinator_client.CoordinatorClient` that
:class:`~src.evaluation.evaluator.RewardEvaluator` uses:

    healthz()            -> docker reachable?
    submit_job(...)      -> queue a candidate (returns a job id)
    wait_for_all(ids)    -> build + run each queued job sequentially (one GPU)
    download_artifacts() -> pack the job's collected output_paths into a tarball

So the evaluator only has to choose which client to construct; everything else —
codebase staging (:class:`WorkspaceManager`), artifact capture
(:class:`ResultProcessor`), scoring — is unchanged.

Faithful to the PCS contract / the repo's README "Direct run":
  * the project tarball (Dockerfile at root) is the ``docker build`` context, so
    the candidate's injected ``_get_rewards`` is baked into the image per job;
  * the container runs non-root (``-u $(id -u):$(id -g)``) with ``HOME`` and the
    working dir pointed at a per-job mount, into which ``logs/`` is written;
  * task/seed/tunables are passed through the job ``env`` (the image entrypoint
    reads ``TASK``/``SEED``/…), not a baked-in command.

Fractional ``gpus`` (a PCS scheduling notion, not a hard VRAM cap) is honoured as
a concurrency cap: ``floor(1/gpus)`` jobs share the single local GPU at once
(e.g. ``0.25`` -> 4 in parallel), each container getting ``--gpus all``.
"""

import os
import re
import shutil
import logging
import tarfile
import tempfile
import subprocess
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def _derive_concurrency(gpus: float) -> int:
    """How many jobs share one GPU, from a PCS-style fractional gpus value.

    Fractional gpus is not a hard VRAM cap (neither here nor in PCS) — it's a
    scheduling signal for how many of these jobs fit on one GPU. So ``0.40`` ->
    2, ``0.25`` -> 4, ``1.0`` -> 1. The ``+1e-6`` corrects float artefacts (e.g.
    ``1/0.2`` flooring to 4 instead of 5)."""
    if gpus and gpus > 0:
        return max(1, int(1.0 / gpus + 1e-6))
    return 1


class LocalRunnerError(RuntimeError):
    """Raised when docker is unavailable or a local job cannot be launched."""


class LocalRunner:
    """
    Build + run reward candidates on the local machine via docker.

    Args:
        image: Docker image *repository* name. Each job is built and run under its
            own ``<repo>:<job_id>`` tag (so concurrent builds never race on a
            shared tag), then that tag is removed once the job finishes.
        gpus: PCS-style fractional GPU request. Sets how many jobs share the local
            GPU at once — ``floor(1/gpus)`` (e.g. ``0.25`` -> 4 concurrent). Each
            container still gets ``--gpus all`` (cooperative sharing, as in PCS).
            0 runs CPU-only, one at a time.
        work_root: Where per-job working directories (the ``/work`` mount) are
            created. Defaults to a temp dir.
        run_as_user: Run the container as the current uid:gid (mirrors the PCS /
            CARES non-root contract). Set False to run as the image's default user.
        max_concurrent: Override the gpus-derived concurrency cap (jobs run at
            once). Defaults to ``floor(1/gpus)``.
    """

    # Job states mirror the coordinator's terminal vocabulary so the evaluator can
    # treat both clients identically (it only special-cases "succeeded").
    def __init__(
        self,
        image: str = "ard-local",
        gpus: float = 1.0,
        work_root: Optional[str] = None,
        run_as_user: bool = True,
        max_concurrent: Optional[int] = None,
    ):
        # Keep only the repository part; the per-job tag is appended at build time.
        self.image_repo = image.split(":", 1)[0]
        self.gpus = float(gpus)
        self.max_concurrent = int(max_concurrent) if max_concurrent \
            else _derive_concurrency(self.gpus)
        self.work_root = work_root or tempfile.mkdtemp(prefix="ard_local_work_")
        self.run_as_user = run_as_user
        os.makedirs(self.work_root, exist_ok=True)
        # job_id -> spec/result dict (see submit_job / _run_job).
        self._jobs: Dict[str, Dict] = {}
        self._docker = shutil.which("docker")
        if not self._docker:
            raise LocalRunnerError(
                "docker not found on PATH; single-machine mode needs docker"
            )
        logger.info(
            f"LocalRunner ready: image={self.image_repo} gpus={self.gpus} "
            f"-> up to {self.max_concurrent} job(s) concurrently on the local GPU"
        )

    # ------------------------------------------------------------------ utils
    def _run(self, args: List[str], **kwargs) -> subprocess.CompletedProcess:
        return subprocess.run([self._docker, *args], **kwargs)

    def _job_image(self, job_id: str) -> str:
        """Per-job image tag ``<repo>:<sanitised-job-id>`` (docker-tag safe)."""
        tag = re.sub(r"[^a-z0-9_.-]", "-", job_id.lower()).strip("-.") or "job"
        return f"{self.image_repo}:{tag}"

    def healthz(self) -> bool:
        """True if the docker daemon answers (parity with CoordinatorClient)."""
        try:
            proc = self._run(
                ["version", "--format", "{{.Server.Version}}"],
                capture_output=True, text=True, timeout=30,
            )
            return proc.returncode == 0
        except (OSError, subprocess.SubprocessError) as e:
            logger.error(f"Local docker health check failed: {e}")
            return False

    # ------------------------------------------------------------------- jobs
    def submit_job(
        self,
        tarball_path: str,
        output_paths: List[str],
        env: Optional[Dict[str, str]] = None,
        command: Optional[str] = None,
        build_args: Optional[Dict[str, str]] = None,
        gpus: float = 1,
        timeout_seconds: int = 3600,
    ) -> str:
        """
        Queue a candidate for local execution and return its job id.

        Cheap by design (mirrors the coordinator's upload-then-poll split): the
        actual ``docker build``/``docker run`` happens in :meth:`wait_for_all`,
        which keeps the evaluator's "Submitted N jobs; waiting…" log honest.
        """
        # Derive a stable, readable id from the codebase tarball name
        # (codebase_<tag>.tar.gz -> <tag>); fall back to a counter.
        base = os.path.basename(tarball_path)
        tag = base[len("codebase_"):-len(".tar.gz")] if base.startswith("codebase_") \
            and base.endswith(".tar.gz") else f"job_{len(self._jobs)}"
        job_id = tag
        self._jobs[job_id] = {
            "tarball_path": tarball_path,
            "output_paths": list(output_paths),
            "env": dict(env or {}),
            "command": command,
            "build_args": dict(build_args or {}),
            "gpus": float(gpus),
            "timeout_seconds": int(timeout_seconds),
            "status": "submitted",
            "exit_code": None,
            "error": None,
            "work_dir": os.path.join(self.work_root, tag),
        }
        logger.info(f"Queued local job {job_id}: env={env or {}} command={command!r}")
        return job_id

    def wait_for_all(
        self, job_ids: List[str], poll_interval: Optional[float] = None
    ) -> Dict[str, Dict]:
        """
        Build + run the queued jobs, up to ``max_concurrent`` sharing the GPU at
        once; return a ``{job_id: job}`` map.

        Each ``job`` carries ``status`` (succeeded / failed / timed_out) plus
        ``exit_code`` and ``error`` — the same shape the evaluator reads from a
        coordinator JobView. Each thread only mutates its own job dict, so the
        shared ``_jobs`` map needs no extra locking.
        """
        pending = [
            jid for jid in job_ids
            if self._jobs.get(jid, {}).get("status") == "submitted"
        ]
        for jid in job_ids:
            if jid not in self._jobs:
                logger.warning(f"Unknown local job id: {jid}")
        workers = max(1, min(self.max_concurrent, len(pending)))
        logger.info(f"Running {len(pending)} local job(s), {workers} at a time")
        with ThreadPoolExecutor(max_workers=workers) as executor:
            list(executor.map(lambda jid: self._run_job(jid, self._jobs[jid]), pending))
        return {jid: self._jobs[jid] for jid in job_ids if jid in self._jobs}

    def _run_job(self, job_id: str, job: Dict) -> None:
        """Execute one job: build its image, run the container, capture status.

        Uses a per-job ``<repo>:<job_id>`` tag and a per-job container name so
        several jobs can build and run at once without colliding; the tag is
        removed at the end so disk doesn't grow across iterations.
        """
        image = self._job_image(job_id)
        name = re.sub(r"[^a-zA-Z0-9_.-]", "-", f"ard_{job_id}")
        work_dir = job["work_dir"]
        if os.path.exists(work_dir):
            shutil.rmtree(work_dir, ignore_errors=True)
        os.makedirs(work_dir, exist_ok=True)

        try:
            # 1) Build the image from the candidate tarball (Dockerfile at its
            #    root). The build context is the gzipped tar piped on stdin; docker
            #    auto-detects the compression. This bakes the injected reward in.
            build_args = []
            for k, v in job["build_args"].items():
                build_args += ["--build-arg", f"{k}={v}"]
            logger.info(f"[{job_id}] docker build -> {image}")
            try:
                with open(job["tarball_path"], "rb") as ctx:
                    proc = self._run(
                        ["build", "-t", image, *build_args, "-"],
                        stdin=ctx, capture_output=True, text=True,
                    )
            except OSError as e:
                job.update(status="failed", error=f"docker build invocation failed: {e}")
                return
            if proc.returncode != 0:
                tail = (proc.stderr or proc.stdout or "")[-1500:]
                logger.error(f"[{job_id}] image build failed:\n{tail}")
                job.update(status="failed", exit_code=proc.returncode,
                           error=f"docker build failed: {tail[-500:]}")
                return

            # 2) Run the container the way PCS does: non-root, HOME + workdir on the
            #    per-job mount, env-driven entrypoint. logs/ lands in work_dir.
            run_args = ["run", "--rm", "--name", name]
            if self.gpus > 0:
                run_args += ["--gpus", "all"]
            if self.run_as_user:
                run_args += ["-u", f"{os.getuid()}:{os.getgid()}"]
            run_args += ["-e", "HOME=/work", "-w", "/work", "-v", f"{work_dir}:/work"]
            for k, v in job["env"].items():
                run_args += ["-e", f"{k}={v}"]
            run_args.append(image)
            if job["command"]:                     # optional CMD override
                run_args += ["sh", "-c", job["command"]]

            log_path = os.path.join(work_dir, "local_run.log")
            logger.info(f"[{job_id}] docker run (logs -> {log_path})")
            try:
                with open(log_path, "w") as log_fh:
                    proc = self._run(
                        run_args, stdout=log_fh, stderr=subprocess.STDOUT,
                        timeout=job["timeout_seconds"],
                    )
                job["exit_code"] = proc.returncode
                job["status"] = "succeeded" if proc.returncode == 0 else "failed"
                if proc.returncode != 0:
                    job["error"] = f"container exited {proc.returncode} (see {log_path})"
            except subprocess.TimeoutExpired:
                logger.warning(f"[{job_id}] timed out after {job['timeout_seconds']}s; killing")
                self._run(["kill", name], capture_output=True)
                job.update(status="timed_out",
                           error=f"exceeded timeout {job['timeout_seconds']}s")
            except OSError as e:
                job.update(status="failed", error=f"docker run invocation failed: {e}")
        finally:
            # Drop the per-job tag; shared base/cache layers are reference-counted
            # so this only frees this job's thin layers.
            self._run(["rmi", "-f", image], capture_output=True)

        if job["status"] != "succeeded":
            logger.warning(f"[{job_id}] {job['status']}: {job.get('error')}")

    def download_artifacts(self, job_id: str, dest_path: str) -> bool:
        """
        Pack the job's collected ``output_paths`` into ``dest_path`` (a .tar.gz).

        Returns False (without raising) when the job produced none of them — the
        same "no artifacts" signal the coordinator client gives on a 404.
        """
        job = self._jobs.get(job_id)
        if job is None:
            logger.warning(f"download_artifacts: unknown job {job_id}")
            return False
        work_dir = job["work_dir"]
        present = [p for p in job["output_paths"]
                   if os.path.exists(os.path.join(work_dir, p))]
        if not present:
            logger.warning(f"Job {job_id} produced no artifacts under {work_dir}")
            return False
        os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
        with tarfile.open(dest_path, "w:gz") as tar:
            for rel in present:
                tar.add(os.path.join(work_dir, rel), arcname=rel)
        logger.info(f"Packed artifacts for {job_id} -> {dest_path}")
        return True
