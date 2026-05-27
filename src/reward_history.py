"""
Centralised, thread-safe record of the reward-refinement lifecycle.

One :class:`RewardRecord` captures the *entire* life of a single reward
candidate — from the LLM that proposed it, through the job that trained it, to
the fitness it scored and the feedback we sent back. :class:`RewardHistory` is
the one place that owns these records across every iteration and phase.

Why this exists
---------------
The loop used to scatter a candidate across positionally-aligned lists
(``reward_methods[i]`` / ``raw_responses[i]``), an idx-keyed ``logs`` list, and
a ``best_run`` dict. Correlating "which raw LLM response produced job X with
fitness Y, and what feedback did we send?" meant cross-referencing four
structures linked only by list position. A record collapses that into one
object that flows through generation -> evaluation -> judgement.

Concurrency
-----------
Reward generation is fanned out across a thread pool, so several threads
register records at once. Every mutation of the shared list goes through a
lock, so :meth:`RewardHistory.new_record` / :meth:`update` / reads are safe to
call concurrently. Per-record fields are otherwise written by the single-
threaded evaluate/score phases, which hold the record object directly.

Designed for a future where *all* candidates (not just the best) are fed back
to the LLM: every candidate is retained with its method, summary and fitness,
so :meth:`RewardHistory.for_iteration` already yields the full batch.
"""

import os
import json
import time
import logging
import threading
from typing import Dict, List, Optional
from dataclasses import dataclass, field, asdict
from math import isfinite

logger = logging.getLogger(__name__)

# Lifecycle status values, in rough order of progression.
STATUS_PENDING = "pending"          # record created, nothing attempted yet
STATUS_GENERATED = "generated"      # LLM produced a valid _get_rewards method
STATUS_GEN_FAILED = "gen_failed"    # LLM never produced a valid method
STATUS_BUILD_FAILED = "build_failed"   # reward injection / codebase build failed
STATUS_SUBMIT_FAILED = "submit_failed"  # coordinator rejected the job
STATUS_SUBMITTED = "submitted"      # job accepted, training in flight
STATUS_NO_ARTIFACTS = "no_artifacts"   # job finished but produced nothing
STATUS_NO_METRICS = "no_metrics"    # artifacts had no usable TensorBoard logs
# Terminal coordinator states (succeeded / failed / cancelled / timed_out) are
# stored verbatim.


@dataclass
class RewardRecord:
    """The full lifecycle of one reward candidate (one training job)."""

    # --- identity -----------------------------------------------------------
    iteration: int                       # 1-based refinement iteration
    index: int                           # position within this batch
    phase: str                           # "run" (exploration) | "eval" (scoring)
    tag: str                             # unique label, e.g. "iter1_run_0"
    seed: Optional[int] = None           # training seed, if pinned

    # --- generation (LLM) ---------------------------------------------------
    model: Optional[str] = None
    temperature: Optional[float] = None
    gen_seed: Optional[int] = None       # LLM sampler seed for this candidate
    raw_response: Optional[str] = None   # verbatim LLM completion
    reward_method: Optional[str] = None  # extracted _get_rewards source
    gen_error: Optional[str] = None      # why generation failed, if it did

    # --- dispatch / evaluation (coordinator) --------------------------------
    job_id: Optional[str] = None
    status: str = STATUS_PENDING
    eval_error: Optional[str] = None     # build/submit/run failure detail

    # --- captured artifacts -------------------------------------------------
    log_path: Optional[str] = None       # extracted run dir
    tb_path: Optional[str] = None        # TensorBoard event file
    summary_path: Optional[str] = None   # human-readable scalar summary

    # --- judgement (fitness scorer) -----------------------------------------
    fitness: float = field(default_factory=lambda: float("-inf"))
    selected_best: bool = False          # chosen as the batch winner
    feedback_text: Optional[str] = None  # exact feedback sent to the LLM

    # --- bookkeeping --------------------------------------------------------
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    @property
    def has_method(self) -> bool:
        """True if generation yielded a reward method to evaluate."""
        return bool(self.reward_method)

    @property
    def succeeded(self) -> bool:
        """True if the job trained and produced a fitness we can score."""
        return self.status == "succeeded" and self.tb_path is not None

    def to_dict(self) -> Dict:
        """JSON-safe dict (non-finite fitness becomes null)."""
        d = asdict(self)
        if not isfinite(self.fitness):
            d["fitness"] = None
        return d

    def __repr__(self):
        fit = f"{self.fitness:.4f}" if isfinite(self.fitness) else "n/a"
        return (f"RewardRecord(tag={self.tag}, status={self.status}, "
                f"fitness={fit}, best={self.selected_best})")


class RewardHistory:
    """
    Thread-safe owner of every :class:`RewardRecord` in a refinement run.

    Args:
        output_dir: Directory the history is persisted to (``reward_history.json``).
    """

    def __init__(self, output_dir: Optional[str] = None):
        self._lock = threading.Lock()
        self._records: List[RewardRecord] = []
        self.output_dir = output_dir

    # ------------------------------------------------------------------ writes
    def new_record(self, **fields) -> RewardRecord:
        """Create a record and append it atomically; returns the record."""
        record = RewardRecord(**fields)
        with self._lock:
            self._records.append(record)
        return record

    def update(self, record: RewardRecord, **fields) -> RewardRecord:
        """Atomically set fields on ``record`` and bump ``updated_at``."""
        with self._lock:
            for key, value in fields.items():
                setattr(record, key, value)
            record.updated_at = time.time()
        return record

    # ------------------------------------------------------------------- reads
    def all(self) -> List[RewardRecord]:
        """Snapshot of every record (safe to iterate without the lock)."""
        with self._lock:
            return list(self._records)

    def for_iteration(
        self, iteration: int, phase: Optional[str] = None
    ) -> List[RewardRecord]:
        """All records for an iteration (optionally one phase), ordered by index."""
        with self._lock:
            records = [
                r for r in self._records
                if r.iteration == iteration and (phase is None or r.phase == phase)
            ]
        return sorted(records, key=lambda r: r.index)

    def best_of(
        self, iteration: int, phase: Optional[str] = None
    ) -> Optional[RewardRecord]:
        """Highest-fitness record for an iteration/phase, or None if none scored."""
        scored = [r for r in self.for_iteration(iteration, phase) if isfinite(r.fitness)]
        return max(scored, key=lambda r: r.fitness) if scored else None

    # ------------------------------------------------------------- persistence
    def save_json(self, path: Optional[str] = None) -> Optional[str]:
        """Write the full history to JSON. Returns the path, or None if unset."""
        if path is None:
            if not self.output_dir:
                return None
            path = os.path.join(self.output_dir, "reward_history.json")
        with self._lock:
            payload = [r.to_dict() for r in self._records]
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(payload, fh, indent=2)
        logger.info(f"Reward history written to {path} ({len(payload)} records)")
        return path
