"""
HTTP client for the Parallel Coordination System (PCS).

ARD no longer runs training itself. It submits each candidate reward function as
a containerised job to a PCS coordinator, which owns scheduling, GPU-slot
allocation, SSH-to-worker dispatch, ``docker run``, log capture, cleanup and
artifact collection. This module is the thin client over that HTTP API.

Coordinator API (see parallel_coordination_system/README.md):
    POST   /jobs                       multipart: metadata(JSON) + codebase(.tar.gz) -> JobOut
    GET    /jobs/{id}                  -> JobOut (status, exit_code, has_artifacts, ...)
    GET    /jobs/{id}/logs             -> {"job_id", "logs"}
    POST   /jobs/{id}/cancel           -> JobOut
    GET    /jobs/{id}/artifacts        -> <id>.tar.gz (only when has_artifacts)
    GET    /healthz                    -> liveness

Auth is a bearer token (``Authorization: Bearer <token>``).
"""

import os
import time
import logging
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

# Terminal job states reported by the coordinator.
TERMINAL_STATES = frozenset({"succeeded", "failed", "cancelled", "timed_out"})


class CoordinatorError(RuntimeError):
    """Raised when the coordinator returns an error or is unreachable."""


class CoordinatorClient:
    """
    Minimal client for submitting and tracking PCS jobs.

    Args:
        base_url: Coordinator base URL, e.g. ``http://localhost:8000``.
        token: Bearer token. If None, read from ``token_env`` environment var.
        token_env: Name of the env var holding the token (default ``PCS_TOKEN``).
        poll_interval: Seconds between status polls in :meth:`wait_for_job`.
        request_timeout: Per-request HTTP timeout in seconds.
    """

    def __init__(
        self,
        base_url: str,
        token: Optional[str] = None,
        token_env: str = "PCS_TOKEN",
        poll_interval: float = 10.0,
        request_timeout: float = 60.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.token = token or os.getenv(token_env)
        if not self.token:
            raise CoordinatorError(
                f"No coordinator token provided (set ${token_env} or pass token=)"
            )
        self.poll_interval = poll_interval
        self.request_timeout = request_timeout
        self._session = requests.Session()
        self._session.headers.update({"Authorization": f"Bearer {self.token}"})

    # ------------------------------------------------------------------ utils
    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _raise_for_status(self, resp: requests.Response, context: str):
        if resp.status_code >= 400:
            body = resp.text[:500]
            raise CoordinatorError(
                f"{context} failed: HTTP {resp.status_code} {body}"
            )

    def healthz(self) -> bool:
        """Return True if the coordinator answers /healthz."""
        try:
            resp = self._session.get(self._url("/healthz"), timeout=self.request_timeout)
            return resp.status_code == 200
        except requests.RequestException as e:
            logger.error(f"Coordinator health check failed: {e}")
            return False

    # ------------------------------------------------------------------- jobs
    def submit_job(
        self,
        tarball_path: str,
        command: str,
        docker_image: str,
        output_paths: List[str],
        gpus: int = 1,
        timeout_seconds: int = 3600,
    ) -> str:
        """
        Submit a job and return its id.

        The codebase tarball expands to the project root on the worker; the
        worker runs ``command`` from there inside ``docker_image``. ``output_paths``
        are collected into the downloadable artifacts tarball.
        """
        import json

        metadata = {
            "command": command,
            "docker_image": docker_image,
            "output_paths": output_paths,
            "gpus": gpus,
            "timeout_seconds": timeout_seconds,
        }
        logger.info(f"Submitting job: {command!r} image={docker_image} gpus={gpus}")
        with open(tarball_path, "rb") as fh:
            files = {
                "metadata": (None, json.dumps(metadata), "application/json"),
                "codebase": (os.path.basename(tarball_path), fh, "application/gzip"),
            }
            try:
                resp = self._session.post(
                    self._url("/jobs"), files=files, timeout=self.request_timeout
                )
            except requests.RequestException as e:
                raise CoordinatorError(f"Job submission request failed: {e}") from e
        self._raise_for_status(resp, "Job submission")
        job = resp.json()
        logger.info(f"Job submitted: id={job['id']} status={job['status']}")
        return job["id"]

    def get_job(self, job_id: str) -> Dict:
        """Return the JobOut dict for a job."""
        try:
            resp = self._session.get(
                self._url(f"/jobs/{job_id}"), timeout=self.request_timeout
            )
        except requests.RequestException as e:
            raise CoordinatorError(f"get_job({job_id}) failed: {e}") from e
        self._raise_for_status(resp, f"get_job({job_id})")
        return resp.json()

    def wait_for_job(
        self, job_id: str, poll_interval: Optional[float] = None
    ) -> Dict:
        """
        Block until ``job_id`` reaches a terminal state and return its JobOut dict.

        Tolerates transient request failures (logs and retries on the next tick).
        """
        interval = poll_interval or self.poll_interval
        while True:
            try:
                job = self.get_job(job_id)
            except CoordinatorError as e:
                logger.warning(f"Polling {job_id} hit a transient error: {e}")
                time.sleep(interval)
                continue
            if job["status"] in TERMINAL_STATES:
                logger.info(
                    f"Job {job_id} terminal: {job['status']} "
                    f"(exit_code={job.get('exit_code')})"
                )
                return job
            time.sleep(interval)

    def wait_for_all(
        self, job_ids: List[str], poll_interval: Optional[float] = None
    ) -> Dict[str, Dict]:
        """
        Wait for every job in ``job_ids`` to finish.

        Jobs run concurrently on the coordinator (subject to its GPU slots); this
        polls them all and returns a ``{job_id: JobOut}`` map once all are terminal.
        """
        interval = poll_interval or self.poll_interval
        results: Dict[str, Dict] = {}
        pending = list(job_ids)
        while pending:
            still_pending = []
            for job_id in pending:
                try:
                    job = self.get_job(job_id)
                except CoordinatorError as e:
                    logger.warning(f"Polling {job_id} hit a transient error: {e}")
                    still_pending.append(job_id)
                    continue
                if job["status"] in TERMINAL_STATES:
                    results[job_id] = job
                else:
                    still_pending.append(job_id)
            pending = still_pending
            if pending:
                logger.info(
                    f"Waiting on {len(pending)}/{len(job_ids)} job(s): "
                    f"{', '.join(pending)}"
                )
                time.sleep(interval)
        return results

    def get_logs(self, job_id: str) -> str:
        """Return the full captured log text for a job."""
        try:
            resp = self._session.get(
                self._url(f"/jobs/{job_id}/logs"), timeout=self.request_timeout
            )
        except requests.RequestException as e:
            raise CoordinatorError(f"get_logs({job_id}) failed: {e}") from e
        self._raise_for_status(resp, f"get_logs({job_id})")
        return resp.json().get("logs", "")

    def cancel_job(self, job_id: str) -> Dict:
        """Cancel a job; returns the updated JobOut dict."""
        try:
            resp = self._session.post(
                self._url(f"/jobs/{job_id}/cancel"), timeout=self.request_timeout
            )
        except requests.RequestException as e:
            raise CoordinatorError(f"cancel_job({job_id}) failed: {e}") from e
        self._raise_for_status(resp, f"cancel_job({job_id})")
        return resp.json()

    def download_artifacts(self, job_id: str, dest_path: str) -> bool:
        """
        Download a job's artifacts tarball to ``dest_path``.

        Returns False (without raising) when the job produced no artifacts.
        """
        try:
            resp = self._session.get(
                self._url(f"/jobs/{job_id}/artifacts"),
                timeout=self.request_timeout,
                stream=True,
            )
        except requests.RequestException as e:
            raise CoordinatorError(f"download_artifacts({job_id}) failed: {e}") from e
        if resp.status_code == 404:
            logger.warning(f"Job {job_id} has no artifacts to download")
            return False
        self._raise_for_status(resp, f"download_artifacts({job_id})")
        os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
        with open(dest_path, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                if chunk:
                    fh.write(chunk)
        logger.info(f"Downloaded artifacts for {job_id} -> {dest_path}")
        return True
