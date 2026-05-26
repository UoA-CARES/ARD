"""
Fitness judgement for evaluated reward candidates.

This is the "judge" half that was split out of the evaluator: given records the
evaluator has already trained and captured, it reads the fixed ``fitness_function``
metric from each one's TensorBoard log, writes it onto the record, and selects
the batch winner. The evaluator stays responsible only for dispatch + capture;
all metric reading and ranking lives here so the scoring policy can change
(e.g. mean-over-seeds, multi-objective) without touching dispatch.
"""

import os
import logging
from typing import List, Optional

from .result_processor import load_accumulator
from . import config

logger = logging.getLogger(__name__)


class FitnessScorer:
    """Reads the fitness metric from captured runs and ranks candidates."""

    def __init__(self, fitness_tag: str = config.FITNESS_METRIC):
        # Matched by suffix so whatever scope rl_games/IsaacAlgoObserver prefixes
        # it with (e.g. "Episode/fitness_function") still resolves.
        self.fitness_tag = fitness_tag

    # ---------------------------------------------------------------- scoring
    def score(self, record) -> float:
        """Read fitness from ``record.tb_path``, store it on the record, return it."""
        fitness = self.read_fitness(record.tb_path) if record.tb_path else float("-inf")
        record.fitness = fitness
        return fitness

    def score_all(self, records: List) -> List:
        """Score every captured record in ``records``; returns the same list."""
        for record in records:
            self.score(record)
        return records

    def read_fitness(self, tb_file: Optional[str]) -> float:
        """Return the max value of the fitness metric over training (-inf if absent)."""
        if not tb_file or not os.path.exists(tb_file):
            logger.error(f"TensorBoard file not found: {tb_file}")
            return float("-inf")
        try:
            ea = load_accumulator(tb_file)
            tag = self._resolve_tag(ea, self.fitness_tag)
            if tag is None:
                logger.warning(
                    f"Fitness tag {self.fitness_tag!r} not found. "
                    f"Available: {ea.scalars.Keys()}"
                )
                return float("-inf")
            events = ea.Scalars(tag)
            if not events:
                return float("-inf")
            return float(max(e.value for e in events))
        except Exception as e:  # noqa: BLE001 - TB parsing surfaces many error types
            logger.error(f"Error reading fitness from {tb_file}: {e}")
            return float("-inf")

    def _resolve_tag(self, ea, wanted: str) -> Optional[str]:
        """Resolve ``wanted`` to an actual scalar tag, matching by exact or suffix."""
        keys = ea.scalars.Keys()
        if wanted in keys:
            return wanted
        suffix = wanted.split("/")[-1]
        matches = [k for k in keys if k.split("/")[-1] == suffix or k.endswith(suffix)]
        if matches:
            if len(matches) > 1:
                logger.warning(f"Multiple tags match {wanted!r}: {matches}; using {matches[0]}")
            return matches[0]
        return None

    # ---------------------------------------------------------------- ranking
    @staticmethod
    def select_best(records: List):
        """
        Mark and return the highest-fitness record in ``records``.

        Considers only records with a finite fitness (i.e. successfully trained
        and scored). Returns None if none qualify. Sets ``selected_best`` on the
        winner and clears it on the rest of the batch.
        """
        from math import isfinite

        scored = [r for r in records if isfinite(r.fitness)]
        for r in records:
            r.selected_best = False
        if not scored:
            logger.warning("No scored candidates to select from")
            return None
        best = max(scored, key=lambda r: r.fitness)
        best.selected_best = True
        logger.info(f"Best candidate: {best}")
        return best
