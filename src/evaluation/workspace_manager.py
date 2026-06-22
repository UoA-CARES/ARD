"""
Workspace / codebase preparation for coordinator-dispatched evaluation.

ARD trains against the ``ard-isaaclab-tasks`` repo. For each candidate reward we
produce a self-contained ``.tar.gz`` of that repo with the proposed reward spliced
into the task env file, ready to upload as a PCS job's codebase. The pristine repo
is never mutated — every candidate gets a fresh copy in a temp directory.

(This replaces the old git-checkout + regex injection against an in-tree Isaac
project; injection is now AST-based — see :mod:`reward_injection`.)
"""

import os
import shutil
import tarfile
import logging
import tempfile
from typing import Optional

from .reward_injection import inject_reward, extract_method_source, RewardInjectionError

logger = logging.getLogger(__name__)

# Files/dirs never shipped in a job codebase (mirrors the PCS test harness).
_TAR_EXCLUDE = {".git", ".venv", "__pycache__", ".pytest_cache", ".mypy_cache"}


class WorkspaceManager:
    """
    Builds per-candidate job codebases from the ard-isaaclab-tasks repo.

    Args:
        tasks_repo: Path to the ard-isaaclab-tasks checkout (the pristine source).
        env_file_rel: Path of the task env file *relative to* ``tasks_repo`` whose
            ``_get_rewards`` is the injection target,
            e.g. ``source/ard_tasks/ard_tasks/tasks/direct/cartpole/cartpole_env.py``.
        build_root: Where to stage per-candidate copies (default: a temp dir).
    """

    def __init__(
        self,
        tasks_repo: str,
        env_file_rel: str,
        build_root: Optional[str] = None,
    ):
        self.tasks_repo = os.path.abspath(os.path.expanduser(tasks_repo))
        self.env_file_rel = env_file_rel
        # Resolve to an absolute path so staging/tarball locations are stable
        # regardless of CWD (the coordinator/local backends read the tarball by
        # path). ``mkdtemp`` already returns an absolute path.
        self.build_root = (
            os.path.abspath(os.path.expanduser(build_root))
            if build_root else tempfile.mkdtemp(prefix="ard_codebase_")
        )

        if not os.path.isdir(self.tasks_repo):
            raise ValueError(f"tasks_repo not found: {self.tasks_repo}")
        env_abs = os.path.join(self.tasks_repo, env_file_rel)
        if not os.path.isfile(env_abs):
            raise ValueError(f"Task env file not found: {env_abs}")
        os.makedirs(self.build_root, exist_ok=True)

    # ------------------------------------------------------------------ checks
    def validate(self) -> bool:
        """Confirm the pristine env file parses and exposes the reward template."""
        try:
            self.get_reward_template()
            return True
        except (OSError, RewardInjectionError) as e:
            logger.error(f"Workspace validation failed: {e}")
            return False

    def get_reward_template(self) -> str:
        """Return the pristine ``_get_rewards`` source (used to prompt the LLM)."""
        with open(os.path.join(self.tasks_repo, self.env_file_rel)) as fh:
            return extract_method_source(fh.read())

    def get_env_source(self) -> str:
        """Return the full pristine env-file source (LLM task context)."""
        with open(os.path.join(self.tasks_repo, self.env_file_rel)) as fh:
            return fh.read()

    # ------------------------------------------------------------------- build
    def build_codebase(self, reward_method_src: str, tag: str) -> str:
        """
        Stage a fresh repo copy with ``reward_method_src`` injected and pack it.

        Args:
            reward_method_src: LLM-proposed ``_get_rewards`` method source.
            tag: Unique label for this candidate (used in dir/tarball names).

        Returns:
            Absolute path to the produced ``.tar.gz`` codebase.
        """
        stage = os.path.join(self.build_root, f"stage_{tag}")
        if os.path.exists(stage):
            shutil.rmtree(stage)
        logger.info(f"Staging codebase for {tag} -> {stage}")
        shutil.copytree(
            self.tasks_repo,
            stage,
            ignore=shutil.ignore_patterns(*_TAR_EXCLUDE),
        )

        env_abs = os.path.join(stage, self.env_file_rel)
        with open(env_abs) as fh:
            original = fh.read()
        injected = inject_reward(original, reward_method_src)
        with open(env_abs, "w") as fh:
            fh.write(injected)
        logger.debug(f"Injected reward into {env_abs}")

        tarball = os.path.join(self.build_root, f"codebase_{tag}.tar.gz")
        with tarfile.open(tarball, "w:gz") as tar:
            # Pack the *contents* so the archive root is the project root.
            for entry in sorted(os.listdir(stage)):
                if entry in _TAR_EXCLUDE:
                    continue
                tar.add(os.path.join(stage, entry), arcname=entry)
        logger.info(f"Built codebase tarball: {tarball}")
        return tarball

    def cleanup(self):
        """Remove the staging directory tree."""
        if os.path.isdir(self.build_root):
            shutil.rmtree(self.build_root, ignore_errors=True)
