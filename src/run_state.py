"""
Run-level state for ``--resume``: what the refinement loop needs besides the
reward history (completed iteration count, warm-start checkpoint, LLM messages).

Stored as ``run_state.json`` in the run directory, written atomically.
"""

import json
import os
from typing import Any, Dict

STATE_FILENAME = "run_state.json"


def save_state(run_dir: str, state: Dict[str, Any]) -> str:
    """Atomically write ``state`` to ``<run_dir>/run_state.json``."""
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, STATE_FILENAME)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as fh:
        json.dump(state, fh, indent=2)
    os.replace(tmp_path, path)
    return path


def load_state(run_dir: str) -> Dict[str, Any]:
    """Read ``<run_dir>/run_state.json``. Raises FileNotFoundError if absent."""
    with open(os.path.join(run_dir, STATE_FILENAME)) as fh:
        return json.load(fh)
