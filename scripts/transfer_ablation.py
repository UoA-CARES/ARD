#!/usr/bin/env python3
"""
Two-iteration ablation of ARD's network transfer, across a range of seeds.

The question this answers is narrow: *what does carrying the previous
iteration's network into the next one actually buy?* A full refinement run
cannot answer it, because every iteration differs from the last in its reward
code as well as its initial weights. So this script runs exactly two iterations:

  iteration 1  a plain ARD start - base configs, cold, ``agent.sample``
               LLM-generated candidates, the winner picked by fitness.
  iteration 2  ONE batch of candidates, generated once from iteration 1's
               feedback, trained N times over - once per *arm* in
               configs/transfer_ablation_config.yaml. The arms differ only in
               their transfer settings (warm start on/off, plasticity
               injection - CBP neuron replacement - on/off),
               so two arms' results are a paired comparison of those settings
               over identical reward code.

The whole thing is repeated per seed (``42 44`` -> seeds 42, 43, 44), which is
where the noise estimate comes from. Unlike main.py's run phase, each job is
given ``--seed`` explicitly: the arms within a seed then share a training seed
(tightening the paired comparison) and different seeds are genuine independent
replicates rather than the same training run with different LLM sampling.

Scheduling
----------
``MAX_SENT_RUNS`` jobs may be with the HPC scheduler at any one time, counted
across every seed and arm. The scheduler state is tracked locally (this script
never asks the cluster what it is running) by a ProgressBoard, which exists to
stop a seed's stragglers from idling the cluster: iteration 2 of a seed cannot
start until every one of that seed's iteration-1 jobs has landed, so when seed
42 has 2 runs outstanding the remaining 8 slots are filled with seed 43's
iteration-1 work instead of waiting.

HPC backend only.

Usage:
    export OPENROUTER_API_KEY=...
    python scripts/transfer_ablation.py 42 44 --task Isaac-ARD-Repose-Cube-Shadow-Direct-v0
    python scripts/transfer_ablation.py 42 44 --task shadow_hand --dry-run

Results land in
<output_dir>/<task>/transfer_ablation_<timestamp>/{seed_<N>/, transfer_ablation_*.json}.
"""

import os
import sys
import json
import time
import copy
import logging
import argparse
import difflib
from dataclasses import dataclass, field
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from main import load_yaml_config, normalize_warm_start_cfg, resolve_task_config  # noqa: E402
from src.evaluation import RewardEvaluator, FitnessScorer  # noqa: E402
from src.refinement.llm_agent import EurekaAgent  # noqa: E402
from src.reward_history import (  # noqa: E402
    RewardHistory,
    RewardRecord,
    STATUS_GENERATED,
    STATUS_GEN_FAILED,
)

# How many jobs this script will have with the HPC scheduler at any one moment,
# summed over every seed and every arm. The cluster's own per-user cap
# (runner.hpc.max_active_jobs) is a separate, larger ceiling; this is the budget
# the ablation agrees to live within, and the number the ProgressBoard exists to
# keep full.
MAX_SENT_RUNS = 10

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,  # importing main.py already called basicConfig; ours must win
)
logger = logging.getLogger("transfer_ablation")

# Unit lifecycle. A unit is one HPC job: one candidate reward, in one arm, for
# one seed.
READY = "ready"          # has code, waiting for a slot in the window
IN_FLIGHT = "in_flight"  # submitted, not yet terminal
DONE = "done"            # terminal and captured
FAILED = "failed"        # generation, build, submit or training failed
SETTLED = (DONE, FAILED)
# An abandoned arm has no units at all - the reason is kept on ArmState.skip_reason
# - so there is deliberately no "skipped" unit state to leak into the counts.

ITER1 = 1
ITER2 = 2


class AblationConfigError(RuntimeError):
    """transfer_ablation_config.yaml asked for something that does not exist."""


# --------------------------------------------------------------------- config
# The keys an arm may carry, mapped to the base config each one overrides. Adding
# a third config here is the only change needed to let arms override it too.
_OVERRIDABLE = ("refineconfig", "settings")


def _merge_overrides(base: dict, overrides: dict, arm: str, target: str, path: str = ""):
    """Recursively merge `overrides` onto a copy of `base`, in place.

    Every leaf named in `overrides` must already exist in `base`. That is the
    point of this function rather than a plain dict update: an arm that sets
    `warm_start.enable` (or `plasticity` under the wrong parent) would otherwise
    be accepted, add a key no code reads, and quietly run as an exact duplicate
    of another arm - producing an ablation whose two arms are identical and whose
    difference is therefore reported as noise.
    """
    for key, value in overrides.items():
        here = f"{path}.{key}" if path else key
        if key not in base:
            hint = difflib.get_close_matches(key, list(base), n=1)
            suffix = f" (did you mean {hint[0]!r}?)" if hint else ""
            raise AblationConfigError(
                f"arm {arm!r}: {target}.yaml has no field {here!r}{suffix}; "
                f"available here: {sorted(base)}"
            )
        if isinstance(value, dict) != isinstance(base[key], dict):
            raise AblationConfigError(
                f"arm {arm!r}: {target}.yaml field {here!r} is "
                f"{type(base[key]).__name__}, but the override is "
                f"{type(value).__name__}"
            )
        if isinstance(value, dict):
            _merge_overrides(base[key], value, arm, target, here)
        else:
            base[key] = value


def _slug(overrides: dict, path: str = "") -> str:
    """A name for an unnamed arm, built from what it overrides."""
    parts = []
    for key, value in sorted(overrides.items()):
        if isinstance(value, dict):
            parts.append(_slug(value, key))
        else:
            parts.append(f"{key}_{value}")
    return "-".join(p for p in parts if p) or "default"


@dataclass
class ArmSpec:
    """One iteration-2 condition: the fully resolved configs it trains under."""

    name: str
    settings: dict
    refine_cfg: dict
    overrides: dict

    @property
    def warm(self) -> bool:
        return bool(normalize_warm_start_cfg(self.refine_cfg).get("enabled", False))

    @property
    def plasticity(self) -> bool:
        env = self.settings.get("runner", {}).get("env", {}) or {}
        return bool(env.get("plasticity", env.get("PLASTICITY", False)))

    @property
    def injection(self) -> Optional[str]:
        env = self.settings.get("runner", {}).get("env", {}) or {}
        value = env.get("plasticity_injection_strategy",
                        env.get("PLASTICITY_INJECTION_STRATEGY"))
        return str(value) if value else None


def load_arms(path: str, base_settings: dict, base_refine: dict) -> List[ArmSpec]:
    """Read and validate configs/transfer_ablation_config.yaml into ArmSpecs."""
    raw = load_yaml_config(path)
    if not isinstance(raw, list) or not raw:
        raise AblationConfigError(
            f"{path} must be a non-empty list of arms, got "
            f"{type(raw).__name__}"
        )

    arms: List[ArmSpec] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise AblationConfigError(
                f"{path} item {i} must be a mapping, got {type(item).__name__}"
            )
        overrides = {k: v for k, v in item.items() if k != "name"}
        name = str(item.get("name") or _slug(overrides))
        unknown = sorted(set(overrides) - set(_OVERRIDABLE))
        if unknown:
            raise AblationConfigError(
                f"arm {name!r} (item {i}): unknown key(s) {unknown}; "
                f"an arm may override {list(_OVERRIDABLE)} only"
            )
        if any(c in name for c in "/\\ \t"):
            raise AblationConfigError(
                f"arm name {name!r} must be a path- and tag-safe token "
                f"(no spaces or slashes): it names a directory and a job tag"
            )

        settings = copy.deepcopy(base_settings)
        refine_cfg = copy.deepcopy(base_refine)
        for target, dest in (("settings", settings), ("refineconfig", refine_cfg)):
            block = overrides.get(target)
            if block is None:
                continue
            if not isinstance(block, dict):
                raise AblationConfigError(
                    f"arm {name!r}: {target!r} must be a mapping, got "
                    f"{type(block).__name__}"
                )
            _merge_overrides(dest, block, name, target)
        arms.append(ArmSpec(name, settings, refine_cfg, overrides))

    names = [a.name for a in arms]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise AblationConfigError(f"duplicate arm name(s): {duplicates}")
    return arms


# ------------------------------------------------------------- progress board
@dataclass
class Unit:
    """One HPC job: one candidate reward, in one arm, for one seed."""

    seed: int
    arm: Optional[str]  # None => iteration 1
    iteration: int
    index: int
    tag: str
    record: RewardRecord
    state: str
    job: object = None  # HPCJob once submitted

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "arm": self.arm,
            "iteration": self.iteration,
            "index": self.index,
            "tag": self.tag,
            "state": self.state,
            "status": self.record.status,
            "job_id": self.record.job_id,
            "fitness": self.record.fitness if self.record.fitness > float("-inf") else None,
        }


@dataclass
class ArmState:
    """One arm's units for one seed."""

    spec: ArmSpec
    units: List[Unit] = field(default_factory=list)
    skip_reason: Optional[str] = None


@dataclass
class SeedState:
    """Everything one seed's replicate needs, and how far along it is."""

    seed: int
    agent: object                       # EurekaAgent - its own conversation
    history: RewardHistory              # -> seed_<N>/reward_history.json
    scorer: object                      # FitnessScorer (base refineconfig)
    out_dir: str
    iter1_units: Optional[List[Unit]] = None   # None until generated
    arms: Dict[str, ArmState] = field(default_factory=dict)
    best: Optional[RewardRecord] = None
    checkpoint_path: Optional[str] = None
    unlocked: bool = False

    def all_units(self) -> List[Unit]:
        units = list(self.iter1_units or [])
        for arm in self.arms.values():
            units.extend(arm.units)
        return units


class ProgressBoard:
    """Per-seed run progress, and the fill order that keeps the window busy.

    Holds no HPC or LLM state of its own: candidate generation and iteration-1
    judgement arrive through `hooks`, so the scheduling logic here can be driven
    with stubs and asserted on without a cluster.
    """

    def __init__(self, seeds: List[SeedState], arms: List[ArmSpec], hooks, state_path: str):
        self.seeds = seeds
        self.arms = arms
        self.hooks = hooks
        self.state_path = state_path

    # -------------------------------------------------------------- selection
    def next_ready(self) -> Optional[Unit]:
        """The next unit to submit, or None if every seed is blocked or finished.

        Seeds are walked in ascending order and each seed is taken as far as it
        can go before the next is considered, so the earliest seed finishes
        first. A seed that is *blocked* - iteration-1 jobs still outstanding, so
        iteration 2 has no winner to transfer from - simply falls through to the
        next seed, which is what stops stragglers from idling the window.
        """
        for seed_state in self.seeds:
            unit = self._next_for_seed(seed_state)
            if unit is not None:
                return unit
        return None

    def _next_for_seed(self, s: SeedState) -> Optional[Unit]:
        if s.iter1_units is None:
            self._materialise_iter1(s)
        for unit in s.iter1_units:
            if unit.state == READY:
                return unit
        if not all(u.state in SETTLED for u in s.iter1_units):
            return None  # blocked: iteration 2 has nothing to transfer from yet
        if not s.unlocked:
            self._unlock(s)
        for spec in self.arms:
            for unit in s.arms[spec.name].units:
                if unit.state == READY:
                    return unit
        return None

    # ---------------------------------------------------------- materialising
    def _materialise_iter1(self, s: SeedState) -> None:
        logger.info(f"[seed {s.seed}] generating iteration-1 candidates")
        specs = self.hooks.generate(s, ITER1)
        s.iter1_units = [
            self._unit(s, None, ITER1, k, f"iter1_run_{k}", spec)
            for k, spec in enumerate(specs)
        ]
        s.history.save_json()

    def _unlock(self, s: SeedState) -> None:
        """Iteration 1 has landed: judge it, then build every arm's units."""
        self.hooks.on_iter1_complete(s)
        specs = self.hooks.generate(s, ITER2) if s.best is not None else []

        for spec in self.arms:
            arm = ArmState(spec=spec)
            s.arms[spec.name] = arm
            if s.best is None:
                arm.skip_reason = "iteration 1 produced no scored candidate"
            elif spec.warm and not s.checkpoint_path:
                arm.skip_reason = (
                    f"iteration-1 winner {s.best.tag} has no checkpoint to transfer"
                )
            if arm.skip_reason:
                logger.error(f"[seed {s.seed}] arm {spec.name!r} skipped: {arm.skip_reason}")
                continue
            for k, cand in enumerate(specs):
                tag = f"iter2_{spec.name}_run_{k}"
                unit = self._unit(s, spec.name, ITER2, k, tag, cand)
                if spec.warm:
                    s.history.update(unit.record, warm_started_from=s.best.tag)
                arm.units.append(unit)

        s.unlocked = True
        s.history.save_json()
        self._write_warm_start_log(s)

    def _unit(self, s: SeedState, arm: Optional[str], iteration: int,
              index: int, tag: str, cand: dict) -> Unit:
        """Register one candidate on the seed's history and wrap it in a Unit."""
        record = s.history.new_record(
            iteration=iteration,
            index=index,
            phase="run",
            tag=tag,
            seed=s.seed,
            model=cand.get("model"),
            temperature=cand.get("temperature"),
            gen_seed=cand.get("gen_seed"),
            reward_method=cand.get("reward_method"),
            raw_response=cand.get("raw_response"),
            gen_error=cand.get("gen_error"),
            status=STATUS_GEN_FAILED if cand.get("gen_error") else STATUS_GENERATED,
        )
        return Unit(
            seed=s.seed, arm=arm, iteration=iteration, index=index, tag=tag,
            record=record, state=FAILED if cand.get("gen_error") else READY,
        )

    def _write_warm_start_log(self, s: SeedState) -> None:
        """The same audit trail main.py keeps: who was warm-started from whom."""
        entries = [
            {
                "iteration": ITER2,
                "arm": spec.name,
                "source_tag": s.best.tag,
                "source_iteration": s.best.iteration,
                "source_fitness": s.best.fitness,
                "checkpoint_path": s.checkpoint_path,
                "applied_to": [u.tag for u in s.arms[spec.name].units],
            }
            for spec in self.arms
            if spec.warm and s.arms[spec.name].units
        ]
        if not entries:
            return
        path = os.path.join(s.out_dir, "warm_start_history.json")
        with open(path, "w") as fh:
            json.dump(entries, fh, indent=2)

    # ------------------------------------------------------------- inspection
    def all_done(self) -> bool:
        for s in self.seeds:
            if s.iter1_units is None or not s.unlocked:
                return False
            if any(u.state not in SETTLED for u in s.all_units()):
                return False
        return True

    def snapshot(self) -> dict:
        return {
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "max_sent_runs": MAX_SENT_RUNS,
            "arms": [a.name for a in self.arms],
            "seeds": [
                {
                    "seed": s.seed,
                    "unlocked": s.unlocked,
                    "best_tag": s.best.tag if s.best else None,
                    "best_fitness": s.best.fitness if s.best else None,
                    "checkpoint_path": s.checkpoint_path,
                    "skipped_arms": {
                        name: arm.skip_reason
                        for name, arm in s.arms.items() if arm.skip_reason
                    },
                    "units": [u.to_dict() for u in s.all_units()],
                }
                for s in self.seeds
            ],
        }

    def save(self) -> None:
        tmp = self.state_path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(self.snapshot(), fh, indent=2)
        os.replace(tmp, self.state_path)  # never leave a half-written state file


# --------------------------------------------------------------------- hooks
class LLMHooks:
    """Candidate generation and iteration-1 judgement, via the Eureka agent."""

    def __init__(self, base_refine: dict):
        self.max_workers = int(base_refine.get("max_workers", 1))

    def generate(self, s: SeedState, iteration: int) -> List[dict]:
        """`agent.sample` candidates, fanned out across threads (network-bound).

        Mirrors main.py's generation phase, including its per-candidate
        `gen_seed`, so an iteration-1 batch here is the same draw main.py would
        have made for this seed.
        """
        agent = s.agent
        results: List[Optional[dict]] = [None] * agent.samples

        def _generate(k: int) -> None:
            gen_seed = s.seed + iteration * 1000 + k
            common = {
                "model": agent.model,
                "temperature": agent.temperature,
                "gen_seed": gen_seed,
            }
            try:
                method, raw = agent.func_gen(agent.messages, seed=gen_seed)
                results[k] = {**common, "reward_method": method, "raw_response": raw}
            except RuntimeError as e:
                logger.error(f"[seed {s.seed}] iter{iteration} gen {k} failed: {e}")
                results[k] = {**common, "gen_error": str(e)}

        with ThreadPoolExecutor(max_workers=min(agent.samples, self.max_workers)) as pool:
            list(pool.map(_generate, range(agent.samples)))
        return [r for r in results if r is not None]

    def on_iter1_complete(self, s: SeedState) -> None:
        """Score iteration 1, keep its winner's checkpoint, and feed it back.

        The feedback is what makes iteration 2 a real second iteration rather
        than a re-roll of iteration 1, and it happens exactly once per seed - so
        every arm generates from the identical conversation and, below, from the
        identical candidates.
        """
        records = [u.record for u in s.iter1_units]
        s.scorer.score_all(records)
        best = s.scorer.select_best(records)
        s.history.save_json()
        if best is None:
            logger.error(f"[seed {s.seed}] iteration 1 produced no scored candidate")
            return

        s.best = best
        s.checkpoint_path = best.checkpoint_path
        logger.info(
            f"[seed {s.seed}] iteration-1 winner {best.tag} fitness={best.fitness:.4f} "
            f"checkpoint={best.checkpoint_path}"
        )
        feedback = s.agent.receive_feedback(best.raw_response, summary_path=best.summary_path)
        s.history.update(best, feedback_text=feedback)
        s.history.save_json()


# ---------------------------------------------------------------- evaluators
class EvaluatorPool:
    """One RewardEvaluator per (seed, arm), built on first use.

    An evaluator bakes in its warm-start block and its `runner` config - which
    is where plasticity lives - so an arm cannot share one with another arm.
    Seeds are kept apart too, because each needs its own output tree, staging
    root and job-name prefix (the same namespacing scripts/run_seeds.py applies
    per seed).
    """

    def __init__(self, task_cfg: dict, base_settings: dict, base_refine: dict,
                 root: str, run_id: str):
        self.task_cfg = task_cfg
        self.base_settings = base_settings
        self.base_refine = base_refine
        self.root = root
        self.run_id = run_id
        self._cache: Dict[Tuple[int, Optional[str]], RewardEvaluator] = {}

    def get(self, seed: int, arm: Optional[ArmSpec]) -> RewardEvaluator:
        key = (seed, arm.name if arm else None)
        if key not in self._cache:
            settings = arm.settings if arm else self.base_settings
            refine_cfg = arm.refine_cfg if arm else self.base_refine
            runner = copy.deepcopy(settings["runner"])
            hpc = runner.setdefault("hpc", {})
            hpc["job_name_prefix"] = f"{hpc.get('job_name_prefix', 'ard')}_abl_s{seed}"
            build_root = settings.get("build_root")
            self._cache[key] = RewardEvaluator(
                tasks_repo=settings["tasks_repo"],
                env_file_rel=self.task_cfg["env_file"],
                task=self.task_cfg["task"],
                runner=runner,
                # Arms of one seed share an output tree; the arm name is in every
                # tag, so nothing collides and one seed_<N>/reward_history.json
                # covers the whole replicate.
                output_dir=os.path.join(self.root, f"seed_{seed}"),
                build_root=(
                    os.path.join(build_root, f"ablation_{self.run_id}", f"seed_{seed}")
                    if build_root else None
                ),
                warm_start=normalize_warm_start_cfg(refine_cfg),
                checkpoint_sel_mode=refine_cfg.get("candidate_checkpoint_sel_mode", "best"),
            )
        return self._cache[key]

    def terminate_all(self) -> None:
        for evaluator in self._cache.values():
            try:
                evaluator.runner.terminate()
            except Exception as e:  # noqa: BLE001 - best-effort teardown
                logger.warning(f"terminate failed: {e}")


# ----------------------------------------------------------------- scheduler
class AblationRunner:
    """Keeps MAX_SENT_RUNS jobs in flight, drawn from wherever work is unblocked."""

    def __init__(self, board: ProgressBoard, pool: EvaluatorPool, poll_seconds: float):
        self.board = board
        self.pool = pool
        self.poll_seconds = poll_seconds
        self.in_flight: List[Unit] = []

    def _arm_spec(self, unit: Unit) -> Optional[ArmSpec]:
        return next((a for a in self.board.arms if a.name == unit.arm), None)

    def _seed_state(self, unit: Unit) -> SeedState:
        return next(s for s in self.board.seeds if s.seed == unit.seed)

    def submit(self, unit: Unit) -> bool:
        """Build + push + submit one unit. False if it never reached the scheduler."""
        spec = self._arm_spec(unit)
        evaluator = self.pool.get(unit.seed, spec)
        # Iteration 1 is always cold (there is no previous winner); iteration 2
        # transfers only in the arms that asked to.
        checkpoint = self._seed_state(unit).checkpoint_path if spec and spec.warm else None
        logger.info(
            f"[{unit.tag}] submitting (seed {unit.seed}, "
            f"{'iter1' if unit.arm is None else unit.arm}, "
            f"{len(self.in_flight) + 1}/{MAX_SENT_RUNS} in flight)"
        )
        job = evaluator.submit_record(unit.record, checkpoint)
        if job is None:
            unit.state = FAILED  # submit_record already recorded why
            return False
        unit.job = job
        unit.state = IN_FLIGHT
        self.in_flight.append(unit)
        return True

    def harvest(self) -> int:
        """Poll every in-flight job; capture the terminal ones. Returns how many landed."""
        landed = 0
        for unit in list(self.in_flight):
            evaluator = self.pool.get(unit.seed, self._arm_spec(unit))
            status = evaluator.poll_record(unit.job)
            if not evaluator.runner.is_terminal(status):
                continue
            retry = evaluator.harvest_record(unit.record, unit.job, status)
            if retry is not None:
                unit.job = retry  # came back empty and was resubmitted
                continue
            self.in_flight.remove(unit)
            unit.state = DONE if unit.record.status == "succeeded" else FAILED
            landed += 1
            self._seed_state(unit).history.save_json()
        return landed

    def run(self) -> None:
        while not self.board.all_done():
            submitted = 0
            while len(self.in_flight) < MAX_SENT_RUNS:
                unit = self.board.next_ready()
                if unit is None:
                    break  # everything left is blocked on a job still running
                self.submit(unit)
                submitted += 1
                self.board.save()

            landed = self.harvest()
            self.board.save()

            if not self.in_flight and not submitted and not landed:
                # Nothing running, nothing startable, and not done: a gate that
                # can never open. Stop rather than spin.
                logger.error("no work in flight and nothing became ready; stopping")
                return
            if self.in_flight:
                time.sleep(self.poll_seconds)


# ------------------------------------------------------------------ reporting
def build_report(board: ProgressBoard, task: str, run_id: str) -> dict:
    """Per-seed, per-arm fitness, and each arm's delta against iteration 1."""
    report = {
        "task": task,
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "max_sent_runs": MAX_SENT_RUNS,
        "arms": [
            {"name": a.name, "warm_start": a.warm, "plasticity": a.plasticity,
             "plasticity_injection": a.injection, "overrides": a.overrides}
            for a in board.arms
        ],
        "seeds": {},
    }
    for s in board.seeds:
        iter1_records = [u.record for u in (s.iter1_units or [])]
        values, mean, std = s.scorer.summarise(iter1_records)
        entry = {
            "iter1": {
                "n": len(values), "mean": mean, "std": std,
                "best_tag": s.best.tag if s.best else None,
                "best_fitness": s.best.fitness if s.best else None,
            },
            "checkpoint_path": s.checkpoint_path,
            "arms": {},
        }
        for spec in board.arms:
            arm = s.arms.get(spec.name)
            if arm is None or arm.skip_reason:
                entry["arms"][spec.name] = {
                    "skipped": arm.skip_reason if arm else "never reached"
                }
                continue
            records = [u.record for u in arm.units]
            s.scorer.score_all(records)
            a_values, a_mean, a_std = s.scorer.summarise(records)
            best = max(a_values) if a_values else None
            entry["arms"][spec.name] = {
                "n": len(a_values),
                "mean": a_mean,
                "std": a_std,
                "best": best,
                "delta_best_vs_iter1": (
                    best - s.best.fitness if best is not None and s.best else None
                ),
                "values": a_values,
            }
        report["seeds"][str(s.seed)] = entry
    return report


def print_report(report: dict) -> None:
    seeds = sorted(report["seeds"], key=int)
    width = max([len(a["name"]) for a in report["arms"]] + [len("iter1 (best)")]) + 2
    header = "".ljust(width) + "".join(f"seed {sd:<18}" for sd in seeds)
    print("\n=== Transfer ablation: fitness by arm ===")
    print(header)
    row = "iter1 (best)".ljust(width)
    for sd in seeds:
        best = report["seeds"][sd]["iter1"]["best_fitness"]
        row += f"{'n/a' if best is None else f'{best:.4f}':<23}"
    print(row)
    for arm in report["arms"]:
        row = arm["name"].ljust(width)
        for sd in seeds:
            cell = report["seeds"][sd]["arms"].get(arm["name"], {})
            if "skipped" in cell:
                row += f"{'skipped':<23}"
            else:
                row += f"{cell['mean']:.4f}±{cell['std']:.4f} (n={cell['n']})".ljust(23)
        print(row)
    print("\nRows are the mean fitness over an arm's iteration-2 candidates; "
          "iter1 (best) is the winner those arms transferred from.\n")


# ------------------------------------------------------------------- dry run
def preview_command(task: str, runner_cfg: dict, warm_start_cfg: dict,
                    seed: int, warm: bool) -> str:
    """The command an arm's jobs will run, without standing up a backend.

    Borrows `_build_hpc_command` on a bare instance - the same trick
    scripts/cold_start_evaluate.py uses - so what --dry-run prints is assembled
    by the code that submits and cannot drift from it. This variant also carries
    the warm-start block, which is the dimension being ablated. The checkpoint is
    a placeholder with a pre-seeded epoch of 0 so nothing is read from disk; a
    real warm job's --max_iterations is higher by its checkpoint's own epoch.
    """
    stub = RewardEvaluator.__new__(RewardEvaluator)
    stub.task = task
    stub.env_extra = dict(runner_cfg.get("env", {}) or {})
    stub.hpc_extra_args = str((runner_cfg.get("hpc", {}) or {}).get("extra_args", "") or "")
    stub.warm_start_cfg = dict(warm_start_cfg or {})
    placeholder = "<iteration-1 winner checkpoint>"
    stub._epoch_cache = {placeholder: 0}
    return RewardEvaluator._build_hpc_command(stub, seed, placeholder if warm else None)


def dry_run(seeds: List[int], arms: List[ArmSpec], task_cfg: dict,
            base_settings: dict, base_refine: dict, samples: int) -> None:
    """Print the plan. Contacts nothing: no docker, no scheduler, no LLM."""
    total = len(seeds) * samples * (1 + len(arms))
    print(f"\ntask      : {task_cfg['task']}")
    print(f"seeds     : {seeds}")
    print(f"samples   : {samples} candidates per iteration")
    print(f"arms      : {[a.name for a in arms]}")
    print(f"jobs      : {len(seeds)} seeds x {samples} x (1 iter1 + {len(arms)} arms) = {total}")
    print(f"window    : {MAX_SENT_RUNS} in flight at once "
          f"(~{-(-total // MAX_SENT_RUNS)} full waves)")

    print("\nfirst wave (the order next_ready() hands out slots):")
    for i in range(min(MAX_SENT_RUNS, len(seeds) * samples)):
        seed = seeds[i // samples]
        print(f"  {i + 1:>2}. seed {seed} iter1_run_{i % samples}")
    print("  (iteration 2 of a seed unlocks only once all of its iteration-1 "
          "jobs have landed)")

    print(f"\niteration 1 command (seed {seeds[0]}, base configs):")
    print("  " + preview_command(
        task_cfg["task"], base_settings["runner"],
        normalize_warm_start_cfg(base_refine), seeds[0], warm=False))
    for arm in arms:
        print(f"\narm {arm.name!r} (warm_start={arm.warm}, plasticity={arm.plasticity}, "
              f"plasticity_injection={arm.injection}):")
        print("  " + preview_command(
            task_cfg["task"], arm.settings["runner"],
            normalize_warm_start_cfg(arm.refine_cfg), seeds[0], warm=arm.warm))
    print()


# ----------------------------------------------------------------------- main
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("seed_lo", type=int, help="First seed (inclusive)")
    parser.add_argument("seed_hi", type=int, help="Last seed (inclusive)")
    parser.add_argument("--task", required=True,
                        help="Registered task name or task dir; resolved against "
                             "settings.tasks_repo")
    parser.add_argument("--settings", default="configs/settings.yaml")
    parser.add_argument("--refineconfig", default="configs/refineconfig.yaml")
    parser.add_argument("--ablation-config", default="configs/transfer_ablation_config.yaml",
                        help="The iteration-2 arms")
    parser.add_argument("--output-dir", default=None,
                        help="Overrides settings.output_dir for this ablation only")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the plan and each arm's job command, then exit. "
                             "Contacts nothing.")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()

    if args.seed_lo > args.seed_hi:
        sys.exit(f"seed bounds are inclusive and ascending: got {args.seed_lo} {args.seed_hi}")
    seeds = list(range(args.seed_lo, args.seed_hi + 1))

    base_settings = load_yaml_config(args.settings)
    base_refine = load_yaml_config(args.refineconfig)
    task_cfg = load_yaml_config(resolve_task_config(args.task, base_settings["tasks_repo"]))
    arms = load_arms(args.ablation_config, base_settings, base_refine)
    samples = int(base_refine.get("agent", {}).get("sample", 4))

    if args.dry_run:
        dry_run(seeds, arms, task_cfg, base_settings, base_refine, samples)
        return 0

    backend = str(base_settings.get("runner", {}).get("backend", "local")).lower()
    if backend != "hpc":
        sys.exit(
            f"transfer_ablation drives the HPC scheduler directly and has no local "
            f"path; settings.runner.backend is {backend!r}"
        )
    hpc_cfg = base_settings["runner"].get("hpc", {}) or {}
    cap = int(hpc_cfg.get("max_active_jobs", 50))
    if MAX_SENT_RUNS > cap:
        sys.exit(
            f"MAX_SENT_RUNS={MAX_SENT_RUNS} exceeds the scheduler's "
            f"max_active_jobs={cap}; lower it at the top of this script"
        )

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    root = os.path.join(
        os.path.expanduser(args.output_dir or base_settings.get("output_dir", "./runs")),
        task_cfg["task"],
        f"transfer_ablation_{run_id}",
    )
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, "transfer_ablation_config.yaml"), "w") as fh:
        yaml.safe_dump(
            [{"name": a.name, "warm_start": a.warm, "plasticity": a.plasticity,
              "plasticity_injection": a.injection, "overrides": a.overrides}
             for a in arms],
            fh, sort_keys=False,
        )
    logger.info(
        f"Transfer ablation {run_id}: seeds {seeds}, arms {[a.name for a in arms]}, "
        f"{len(seeds) * samples * (1 + len(arms))} jobs, "
        f"{MAX_SENT_RUNS} in flight at once -> {root}"
    )

    # One evaluator is needed before anything else: the agent's prompt is built
    # from the pristine task sources it exposes, and those are the same for every
    # seed and arm.
    pool = EvaluatorPool(task_cfg, base_settings, base_refine, root, run_id)
    template_evaluator = pool.get(seeds[0], None)

    seed_states = []
    for seed in seeds:
        out_dir = os.path.join(root, f"seed_{seed}")
        os.makedirs(out_dir, exist_ok=True)
        seed_states.append(SeedState(
            seed=seed,
            # A conversation per seed: iteration 2's candidates are written in
            # reply to *this* seed's iteration-1 winner.
            agent=EurekaAgent(
                task_description=task_cfg["description"],
                reward_template=template_evaluator.get_reward_template(),
                env_source=template_evaluator.get_env_source(),
                agent_config=base_refine.get("agent", {}),
            ),
            history=RewardHistory(output_dir=out_dir),
            scorer=FitnessScorer(scoring_mode=base_refine.get("scoring_mode", "global_max")),
            out_dir=out_dir,
        ))

    board = ProgressBoard(
        seeds=seed_states, arms=arms, hooks=LLMHooks(base_refine),
        state_path=os.path.join(root, "transfer_ablation_state.json"),
    )
    runner = AblationRunner(board, pool, float(hpc_cfg.get("poll_seconds", 30)))

    try:
        runner.run()
    except KeyboardInterrupt:
        logger.warning("interrupted; cancelling in-flight jobs")
        pool.terminate_all()
        board.save()
        return 130
    finally:
        board.save()

    report = build_report(board, task_cfg["task"], run_id)
    with open(os.path.join(root, "transfer_ablation_report.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    print_report(report)
    logger.info(f"Transfer ablation complete: {root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
