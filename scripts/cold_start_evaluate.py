#!/usr/bin/env python3
"""
Re-train a warm-start experiment's per-iteration winning rewards from scratch.

A warm-start refinement run chains its iterations: iteration 1 cold-starts, and
every later iteration's candidates all resume from the previous iteration's
best-fitness checkpoint. Each winner's fitness therefore carries two confounded
effects -- a better reward function *and* the epochs of training it inherited.
This script builds the missing cold-start arm: for every iteration's winning
reward function it launches one fresh run (newly initialised networks, no
checkpoint), everything else held identical to the warm counterpart -- same task,
same network architecture and PPO hyperparameters (both come from the task's
rl_games_ppo_cfg.yaml, which is never touched), same num_envs / plasticity /
epoch budget, same fitness scoring.

Input is a finished experiment folder, e.g.
    runs/seed_42/Isaac-ARD-Repose-Cube-Shadow-Direct-v0/20260914-194139
which must hold `warm_start_history.json` (so the run was warm-started) and
`reward_history.json` (which carries every candidate's reward source verbatim --
no LLM is involved here).

What gets run, all derived from those two files:
  * Every `source_tag` in warm_start_history.json with `source_iteration >= 2`
    -- i.e. each iteration's winner, as the warm-start audit trail recorded it.
    Iteration 1's winner is skipped: iteration 1 already cold-started.
  * Plus the last iteration's winner, which no entry names as a source (there is
    no further iteration for it to warm-start). It is the highest-fitness
    `phase == "run"` record of that iteration. Note `selected_best` cannot be
    used for this: main.py re-runs `select_best` over *all* run records at the
    end, so only the single global winner keeps the flag.

Step offsets are *recorded, not applied*. Nothing in rl_games or train.py can
start a run's logging at a non-zero epoch -- a warm run's offset x-axis comes
from rl_games inheriting the epoch/frame counters out of the checkpoint it
loads. So each cold run logs epochs 1..MAX_ITERATIONS, and `step_offset` (the
epoch of the checkpoint its warm counterpart consumed) is written into
cold_start_history.json for the comparison to add at plot time. This is also why
every cold run gets exactly MAX_ITERATIONS epochs: the evaluator only extends
the budget when it is handed a checkpoint.

Everything about dispatch, collection and scoring is the pipeline's own:
RewardEvaluator submits/polls/collects (hpc or local backend, per settings.yaml),
ResultProcessor captures the artifacts, FitnessScorer reads the fitness. The only
new thing here is picking the rewards, and the bookkeeping that pairs each cold
run with its warm counterpart.

Output lands beside the originals, inside the source folder, so both arms sit
side by side and one `scripts/get_run_metrics.py <run_folder>` packages them:

    <run_folder>/iter3_run_1/            the original warm run (untouched)
    <run_folder>/iter3_run_1_cold/       its cold counterpart          (no --seeds)
    <run_folder>/iter3_run_1_cold/seed_42/   one subdir per seed       (--seeds 42 43)
    <run_folder>/cold_start_history.json the only file this script writes

Caveats worth knowing when reading the results:
  * `save_best_after` (100 epochs) bites at epoch 100 in a cold run, but was
    already satisfied from the first epoch in the warm counterpart, so the
    nn/<name>.pth "best" checkpoint means slightly different things in the two
    arms. Fitness is read from TensorBoard, not from checkpoints, so scoring is
    unaffected.
  * The job image fetches rl_games from GitHub at build time (RL_GAMES_REF
    defaults to master in the tasks repo Dockerfile) and HPC builds pass no
    --build-arg, so if the fork has moved since the original experiment the cold
    arm trains against different rl_games code. The SHA each original job used is
    printed from its container.log so a mismatch is visible before submitting.

Usage:
    python scripts/cold_start_evaluate.py runs/seed_42/<task>/<timestamp> --dry-run
    python scripts/cold_start_evaluate.py runs/seed_42/<task>/<timestamp>
    python scripts/cold_start_evaluate.py runs/seed_42/<task>/<timestamp> --seeds 42 43 44
"""

import os
import re
import sys
import json
import time
import shlex
import shutil
import logging
import argparse
from math import isfinite
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

try:
    from main import load_yaml_config, resolve_task_config
    from src.evaluation import RewardEvaluator, FitnessScorer, ResultProcessor, config
    from src.evaluation.result_processor import checkpoint_epoch
    from src.reward_history import RewardHistory, RewardRecord, STATUS_GENERATED
except ImportError as e:  # pragma: no cover - environment problem, not logic
    sys.exit(
        f"Cannot import the ARD pipeline from {REPO_ROOT}: {e}\n"
        "Run this from the ARD checkout with the pipeline's environment active."
    )

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,  # importing main.py already called basicConfig; ours must win
)
logger = logging.getLogger("cold_start_evaluate")

WARM_START_HISTORY = "warm_start_history.json"
REWARD_HISTORY = "reward_history.json"
COLD_HISTORY = "cold_start_history.json"

# Appended to the source tag to name the cold run. Load-bearing: the evaluator
# rmtree's <output_dir>/<tag> before collecting a job's results, and the output
# dir here IS the source experiment folder, so a cold tag that did not differ
# from its source tag would delete the original run.
COLD_SUFFIX = "_cold"

# Flags that legitimately differ between a warm run and its cold counterpart, so
# the preflight command comparison ignores them: the warm-start delivery itself,
# the epoch budget it extends, and the seed (which this script may pin).
_PREFLIGHT_IGNORED_FLAGS = {
    "--checkpoint",
    "--warm_start",
    "--warm_start_reset_optimizer",
    "--warm_start_reset_lr_schedule",
    "--warm_start_reset_obs_normalizer",
    "--warm_start_reset_value_normalizer",
    "--critic_warmup_epoch_count",
    "--max_iterations",
    "--seed",
}

_RL_GAMES_SHA_RE = re.compile(r"^\[hpc\] rl_games=(\S+)", re.MULTILINE)


@dataclass
class Target:
    """One winning reward function and the cold run(s) replacing its warm start."""

    source_tag: str                        # the warm run being re-done cold
    source_iteration: int                  # iteration that tag won
    source_fitness: Optional[float]        # its fitness, as the warm run scored it
    source_checkpoint: Optional[str]       # the checkpoint it warm-started FROM
    step_offset: Optional[int]             # that checkpoint's epoch = where the warm run's x-axis began
    reward_method: str
    raw_response: Optional[str] = None
    model: Optional[str] = None
    temperature: Optional[float] = None
    gen_seed: Optional[int] = None
    records: List[RewardRecord] = field(default_factory=list)

    @property
    def cold_tag(self) -> str:
        return f"{self.source_tag}{COLD_SUFFIX}"

    def to_json(self, scorer: Optional[FitnessScorer] = None) -> Dict:
        """The pairing, then this target's runs — see the module docstring's layout."""
        mean = std = None
        if scorer is not None:
            values, mean_, std_ = scorer.summarise(self.records)
            if values:
                mean, std = mean_, std_
        return {
            "source_tag": self.source_tag,
            "source_iteration": self.source_iteration,
            "source_fitness": self.source_fitness,
            "source_checkpoint": self.source_checkpoint,
            "step_offset": self.step_offset,
            "cold_tag": self.cold_tag,
            "cold_fitness_mean": mean,
            "cold_fitness_std": std,
            "runs": [r.to_dict() for r in self.records],
        }


# --------------------------------------------------------------------- inputs
def load_source_experiment(run_folder: str):
    """Read the source experiment's two history files, or explain what's missing."""
    for name in (WARM_START_HISTORY, REWARD_HISTORY):
        path = os.path.join(run_folder, name)
        if not os.path.isfile(path):
            missing = (
                f"{run_folder} has no {name}. "
                + (
                    "This script expects a warm-start experiment; a cold run writes "
                    "no warm-start audit trail, and there would be nothing to re-do "
                    "cold."
                    if name == WARM_START_HISTORY
                    else "Without it the winning reward functions cannot be recovered."
                )
            )
            raise SystemExit(missing)

    with open(os.path.join(run_folder, WARM_START_HISTORY)) as fh:
        warm_log = json.load(fh)
    with open(os.path.join(run_folder, REWARD_HISTORY)) as fh:
        reward_log = json.load(fh)
    if not warm_log:
        raise SystemExit(f"{os.path.join(run_folder, WARM_START_HISTORY)} is empty")
    return warm_log, reward_log


def resolve_step_offset(checkpoint_path: str) -> Optional[int]:
    """Epoch of a warm-start checkpoint: its file name first, its payload second."""
    epoch = checkpoint_epoch(checkpoint_path)
    if epoch is not None:
        return epoch
    # Renamed or hand-made checkpoint: fall back to the evaluator's own
    # stdlib-only .pth reader rather than re-implementing it (it never imports
    # torch, so this stays cheap and safe on a login node).
    try:
        return RewardEvaluator._checkpoint_epoch(checkpoint_path)
    except Exception as e:
        logger.warning(f"Cannot determine the epoch of {checkpoint_path}: {e}")
        return None


def build_targets(warm_log: List[Dict], reward_log: List[Dict]) -> List[Target]:
    """Pick each iteration's winning reward and the offset its warm run started at.

    Two lookups come out of warm_start_history.json: `source_tag` names the
    winner of `source_iteration`, and the entry whose `applied_to` contains a tag
    names the checkpoint that tag consumed (hence its logging offset).
    """
    by_tag = {r["tag"]: r for r in reward_log}
    run_records = [r for r in reward_log if r.get("phase") == "run"]
    if not run_records:
        raise SystemExit(f"No phase=run records in {REWARD_HISTORY}")

    offsets: Dict[str, Dict] = {}
    for entry in warm_log:
        offset = resolve_step_offset(entry["checkpoint_path"])
        for tag in entry.get("applied_to", []):
            offsets[tag] = {
                "checkpoint": entry["checkpoint_path"],
                "step_offset": offset,
            }

    # Winner per iteration, skipping iteration 1 (already cold-started).
    winners: Dict[int, str] = {
        int(e["source_iteration"]): e["source_tag"]
        for e in warm_log
        if int(e["source_iteration"]) >= 2
    }

    # The last iteration's winner warm-starts nothing, so no entry names it.
    last_iteration = max(int(r["iteration"]) for r in run_records)
    if last_iteration >= 2 and last_iteration not in winners:
        scored = [
            r for r in run_records
            if int(r["iteration"]) == last_iteration and r.get("fitness") is not None
        ]
        if scored:
            winners[last_iteration] = max(scored, key=lambda r: r["fitness"])["tag"]
        else:
            logger.warning(
                f"Iteration {last_iteration} has no scored candidate; its winner "
                "cannot be identified and is left out"
            )

    targets = []
    for iteration in sorted(winners):
        tag = winners[iteration]
        record = by_tag.get(tag)
        if record is None:
            logger.warning(f"{tag} won iteration {iteration} but is absent from "
                           f"{REWARD_HISTORY}; skipping")
            continue
        if not record.get("reward_method"):
            logger.warning(f"{tag} has no reward_method recorded; skipping")
            continue
        offset = offsets.get(tag)
        if offset is None:
            logger.warning(
                f"{tag} appears in no warm_start_history applied_to list, so the "
                "epoch its warm run started from is unknown; skipping"
            )
            continue
        targets.append(Target(
            source_tag=tag,
            source_iteration=iteration,
            source_fitness=record.get("fitness"),
            source_checkpoint=offset["checkpoint"],
            step_offset=offset["step_offset"],
            reward_method=record["reward_method"],
            raw_response=record.get("raw_response"),
            model=record.get("model"),
            temperature=record.get("temperature"),
            gen_seed=record.get("gen_seed"),
        ))

    if not targets:
        raise SystemExit("No iteration winner could be re-run cold; nothing to do")
    return targets


# -------------------------------------------------------------------- records
def cold_tag_for(target: Target, seed: Optional[int]) -> str:
    """Flat, filesystem- and registry-safe tag for one cold run."""
    return target.cold_tag if seed is None else f"{target.cold_tag}_s{seed}"


def final_dir_for(output_dir: str, target: Target, seed: Optional[int]) -> str:
    """Where a cold run's artifacts end up (per-seed runs are nested, see nest_seed_dirs)."""
    if seed is None:
        return os.path.join(output_dir, target.cold_tag)
    return os.path.join(output_dir, target.cold_tag, f"seed_{seed}")


def create_records(history: RewardHistory, targets: List[Target],
                   seeds: Optional[List[int]]) -> List[RewardRecord]:
    """One record per (winning reward, seed), carrying the source's provenance.

    `seed=None` (no --seeds) reproduces the warm counterparts exactly: their
    run-phase records also carried no seed, so no --seed flag is emitted and the
    task's own rl_games_ppo_cfg.yaml seed applies.
    """
    all_records = []
    for target in targets:
        for index, seed in enumerate(seeds or [None]):
            record = history.new_record(
                iteration=target.source_iteration,
                index=index,
                phase="run",
                tag=cold_tag_for(target, seed),
                seed=seed,
                model=target.model,
                temperature=target.temperature,
                gen_seed=target.gen_seed,
                reward_method=target.reward_method,
                raw_response=target.raw_response,
                status=STATUS_GENERATED,
            )
            target.records.append(record)
            all_records.append(record)
    return all_records


# ------------------------------------------------------------------ preflight
def preview_hpc_command(task: str, runner_cfg: Dict, seed: Optional[int]) -> str:
    """The command the cold jobs will run, without standing up a backend.

    `_build_hpc_command` needs only the task and `runner.env`/`runner.hpc.extra_args`
    to assemble its flags, so it is borrowed here on a bare instance. Borrowing it
    rather than re-deriving the flag list is the point: what the preflight compares
    and what `--dry-run` prints are then the same code that submits, and cannot
    drift from it. This also keeps planning usable with no docker and no HPC
    session -- the evaluator's own constructor demands both.
    """
    stub = RewardEvaluator.__new__(RewardEvaluator)
    stub.task = task
    stub.env_extra = dict(runner_cfg.get("env", {}) or {})
    stub.hpc_extra_args = str((runner_cfg.get("hpc", {}) or {}).get("extra_args", "") or "")
    return RewardEvaluator._build_hpc_command(stub, seed, None)


def parse_flags(command: str) -> Dict[str, object]:
    """Flag -> value (True when valueless) for the comparable part of a job command."""
    tokens = shlex.split(command)
    flags: Dict[str, object] = {}
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if not token.startswith("--"):
            i += 1
            continue
        if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
            flags[token] = tokens[i + 1]
            i += 2
        else:
            flags[token] = True
            i += 1
    return {k: v for k, v in flags.items() if k not in _PREFLIGHT_IGNORED_FLAGS}


def original_job_command(run_folder: str, source_tag: str) -> Optional[str]:
    """The verbatim command the original job ran, from the scheduler's status.json."""
    path = os.path.join(run_folder, source_tag, "status.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as fh:
            return (json.load(fh) or {}).get("command")
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Cannot read {path}: {e}")
        return None


def original_rl_games_sha(run_folder: str, source_tag: str) -> Optional[str]:
    """The rl_games commit the original job trained against, from its container.log."""
    path = os.path.join(run_folder, source_tag, "logs", "container.log")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, errors="replace") as fh:
            head = fh.read(4096)
    except OSError:
        return None
    match = _RL_GAMES_SHA_RE.search(head)
    return match.group(1) if match else None


def preflight(run_folder: str, targets: List[Target], cold_command: str) -> List[str]:
    """Compare the cold command against the originals' and report any drift.

    The run folder snapshots no config of its own, so settings.yaml may have been
    edited since the experiment ran. Everything that must match for the two arms
    to be comparable is on the recorded command line.
    """
    problems = []
    cold_flags = parse_flags(cold_command)
    seen_shas = {}
    for target in targets:
        command = original_job_command(run_folder, target.source_tag)
        if command is None:
            logger.warning(f"[{target.source_tag}] no status.json; cannot verify its "
                           "original command")
            continue
        warm_flags = parse_flags(command)
        for flag in sorted(set(cold_flags) | set(warm_flags)):
            if cold_flags.get(flag) != warm_flags.get(flag):
                problems.append(
                    f"{target.source_tag}: {flag} was {warm_flags.get(flag)!r} in the "
                    f"original run, would be {cold_flags.get(flag)!r} now"
                )
        sha = original_rl_games_sha(run_folder, target.source_tag)
        if sha:
            seen_shas.setdefault(sha, []).append(target.source_tag)

    for sha, tags in seen_shas.items():
        logger.info(f"Original rl_games commit {sha} ({len(tags)} run(s)); the cold "
                    "jobs build whatever the fork's RL_GAMES_REF resolves to now")
    if len(seen_shas) > 1:
        logger.warning("The original runs did not all use the same rl_games commit: "
                       + ", ".join(f"{s[:12]}={len(t)}" for s, t in seen_shas.items()))
    return problems


# --------------------------------------------------------------- reorganising
def _rewrite_paths(record: RewardRecord, old_dir: str, new_dir: str):
    """Repoint a moved run's captured paths, failing loudly if one doesn't land."""
    for attr in ("log_path", "tb_path", "summary_path", "checkpoint_path"):
        value = getattr(record, attr)
        if not value:
            continue
        if not value.startswith(old_dir):
            logger.warning(f"[{record.tag}] {attr}={value} is outside {old_dir}; "
                           "left as it is")
            continue
        moved = new_dir + value[len(old_dir):]
        if not os.path.exists(moved):
            raise RuntimeError(
                f"[{record.tag}] {attr} should now be {moved} but nothing is there"
            )
        setattr(record, attr, moved)


def nest_seed_dirs(output_dir: str, targets: List[Target]):
    """Move each seeded run into <cold_tag>/seed_<n>/ and fix up its captured paths.

    The evaluator derives a job's work dir from its tag (<output_dir>/<tag>) and a
    '/' in a tag would break the tarball name, the staging dir and the scheduler
    job name -- so the runs are submitted flat and nested afterwards.
    """
    for target in targets:
        for record in target.records:
            if record.seed is None:
                continue
            old_dir = os.path.join(output_dir, record.tag)
            new_dir = final_dir_for(output_dir, target, record.seed)
            if os.path.abspath(old_dir) == os.path.abspath(new_dir):
                continue
            if not os.path.isdir(old_dir):
                continue  # nothing collected (failed job), or already nested
            os.makedirs(os.path.dirname(new_dir), exist_ok=True)
            if os.path.isdir(new_dir):
                raise SystemExit(
                    f"Cannot nest {old_dir} into {new_dir}: it already exists. Move or "
                    "remove it, or re-run with --skip-existing."
                )
            shutil.move(old_dir, new_dir)
            logger.info(f"[{record.tag}] -> {new_dir}")
            _rewrite_paths(record, os.path.abspath(old_dir), os.path.abspath(new_dir))


def adopt_existing(records: List[RewardRecord], targets: List[Target],
                   output_dir: str, processor: ResultProcessor) -> List[RewardRecord]:
    """Split off records whose results are already on disk (--skip-existing).

    Jobs run for hours, so a re-run after an interruption must not resubmit what
    already finished. Adopted records are captured in place and scored with the
    rest.
    """
    by_tag = {r.tag: t for t in targets for r in t.records}
    to_run = []
    for record in records:
        target = by_tag[record.tag]
        final_dir = final_dir_for(output_dir, target, record.seed)
        captured = processor.capture(final_dir) if os.path.isdir(final_dir) else None
        if captured is None:
            to_run.append(record)
            continue
        record.status = "succeeded"
        record.log_path = captured.log_path
        record.tb_path = captured.tb_path
        record.summary_path = captured.summary_path
        record.checkpoint_path = captured.checkpoint_path
        logger.info(f"[{record.tag}] already has results in {final_dir}; skipping")
    return to_run


# -------------------------------------------------------------------- outputs
def write_history(path: str, meta: Dict, targets: List[Target],
                  scorer: Optional[FitnessScorer] = None) -> str:
    """Write cold_start_history.json — the pairing, the offsets and the records.

    Never RewardHistory.save_json(): that writes reward_history.json into the
    output dir, which here is the source experiment's own folder.
    """
    payload = dict(meta)
    payload["targets"] = [t.to_json(scorer) for t in targets]
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
    return path


def _fmt(value: Optional[float]) -> str:
    return "n/a" if value is None or not isfinite(value) else f"{value:.4f}"


def report(targets: List[Target], scorer: FitnessScorer):
    """warm vs cold, per iteration winner."""
    logger.info("=== Cold-start vs warm-start ===")
    logger.info(f"  {'source':<20} {'offset':>8} {'warm':>10} {'cold':>10} "
                f"{'std':>8} {'delta':>10}")
    for target in targets:
        values, mean, std = scorer.summarise(target.records)
        cold = mean if values else None
        warm = target.source_fitness
        delta = (cold - warm) if (cold is not None and warm is not None) else None
        logger.info(
            f"  {target.source_tag:<20} {str(target.step_offset):>8} {_fmt(warm):>10} "
            f"{_fmt(cold):>10} {_fmt(std if values else None):>8} {_fmt(delta):>10}"
        )


# ----------------------------------------------------------------------- main
def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Re-train a warm-start experiment's per-iteration winning "
                    "rewards from freshly initialised networks, and collect the "
                    "results beside the originals.",
    )
    parser.add_argument("run_folder", type=str,
                        help="finished warm-start experiment folder, e.g. "
                             "runs/seed_42/Isaac-ARD-Repose-Cube-Shadow-Direct-v0/20260914-194139")
    parser.add_argument("--settings", type=str, default="configs/settings.yaml",
                        help="backend / runner config to dispatch with (default: %(default)s)")
    parser.add_argument("--refineconfig", type=str, default="configs/refineconfig.yaml",
                        help="read for scoring_mode and candidate_checkpoint_sel_mode "
                             "(default: %(default)s)")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="RL training seed(s) for the cold runs: each becomes "
                             "train.py --seed, i.e. params.seed for PPO/env "
                             "initialisation. NOT run_seeds.py's --seeds, which sets "
                             "refineconfig.base_seed and drives the LLM sampler. "
                             "Omitted (default): one run per reward with no --seed, "
                             "exactly as the warm counterparts ran, so the task's "
                             "rl_games_ppo_cfg.yaml seed applies. Given: one run per "
                             "reward per seed, each under <tag>_cold/seed_<n>/")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="where cold runs are written (default: the run folder "
                             "itself, so they sit beside the originals)")
    parser.add_argument("--build-root", type=str, default=None,
                        help="codebase staging root (default: a cold_* subdir of "
                             "settings.build_root, so staging never collides)")
    parser.add_argument("--job-name-prefix", type=str, default=None,
                        help="scheduler job-name prefix (default: settings' prefix "
                             "plus _cold and the source seed)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="adopt, rather than resubmit, cold runs whose results "
                             "are already on disk")
    parser.add_argument("--strict", action="store_true",
                        help="abort if the cold command would differ from the "
                             "originals' (default: warn and continue)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan, the preflight and the exact commands; "
                             "submit nothing and write nothing")
    return parser


def derive_names(run_folder: str, settings: Dict, args) -> Dict[str, str]:
    """Output dir, build root and job-name prefix, namespaced to this experiment.

    Defaults keep the cold arm from colliding with anything: its own staging root
    (WorkspaceManager stages a full copy of the tasks repo per candidate) and its
    own scheduler prefix, both tagged with the source seed and timestamp the same
    way run_seeds.py tags a sweep.
    """
    stamp = os.path.basename(run_folder)
    seed_part = next(
        (p for p in Path(run_folder).parts if re.fullmatch(r"seed_\d+", p)), None
    )
    suffix = f"cold_{seed_part}_{stamp}" if seed_part else f"cold_{stamp}"

    build_root = args.build_root
    if build_root is None:
        base = settings.get("build_root") or os.path.join("runs", "_codebases")
        build_root = os.path.join(base, suffix)

    prefix = args.job_name_prefix
    if prefix is None:
        hpc = settings.get("runner", {}).get("hpc") or {}
        prefix = f"{hpc.get('job_name_prefix', 'ard')}_cold"
        if seed_part:
            prefix = f"{prefix}_s{seed_part.split('_')[1]}"

    return {
        "output_dir": args.output_dir or run_folder,
        "build_root": build_root,
        "job_name_prefix": prefix,
    }


def main():
    args = build_arg_parser().parse_args()

    run_folder = os.path.abspath(os.path.expanduser(args.run_folder))
    if not os.path.isdir(run_folder):
        raise SystemExit(f"Not a directory: {run_folder}")

    warm_log, reward_log = load_source_experiment(run_folder)
    targets = build_targets(warm_log, reward_log)

    settings = load_yaml_config(REPO_ROOT / args.settings)
    refine_cfg = load_yaml_config(REPO_ROOT / args.refineconfig)

    # The task is the directory the timestamped run folders live in; resolving it
    # against the tasks repo yields the env file whose compute_reward is replaced.
    task = os.path.basename(os.path.dirname(run_folder))
    task_cfg = load_yaml_config(resolve_task_config(task, settings["tasks_repo"]))
    if task_cfg["task"] != task:
        raise SystemExit(
            f"{run_folder} sits under '{task}' but that resolves to task "
            f"'{task_cfg['task']}'; pass a run folder whose parent names its task"
        )

    names = derive_names(run_folder, settings, args)
    runner_cfg = dict(settings["runner"])
    if isinstance(runner_cfg.get("hpc"), dict):
        runner_cfg["hpc"] = {**runner_cfg["hpc"],
                             "job_name_prefix": names["job_name_prefix"]}

    seeds = args.seeds
    planned = len(targets) * len(seeds or [None])
    logger.info(
        f"{len(targets)} iteration winner(s) x {len(seeds or [None])} seed(s) "
        f"= {planned} cold job(s) -> {names['output_dir']}"
    )
    for target in targets:
        logger.info(
            f"  iter {target.source_iteration:>2}: {target.source_tag} "
            f"(fitness {_fmt(target.source_fitness)}) warm-started at epoch "
            f"{target.step_offset} -> {target.cold_tag}"
            + ("" if seeds is None else f"/seed_{{{','.join(map(str, seeds))}}}")
        )

    # Checked, not trusted: the evaluator rmtree's each work dir before collecting
    # into it, and the output dir is the source experiment's own folder, so a cold
    # tag that collided with its source tag would delete the original run. Not an
    # assert -- this must hold under python -O too.
    for target in targets:
        for seed in seeds or [None]:
            tag = cold_tag_for(target, seed)
            if not tag.startswith(target.source_tag + COLD_SUFFIX) or tag == target.source_tag:
                raise SystemExit(
                    f"Refusing to run: cold tag {tag!r} does not distinguish itself "
                    f"from its source {target.source_tag!r}"
                )

    # Command and preflight before the backend: both are pure config, so a dry
    # run needs neither docker nor a live scheduler session.
    cold_command = None
    if settings["runner"].get("backend", "local").lower() == "hpc":
        cap = int((settings["runner"].get("hpc", {}) or {}).get(
            "max_active_jobs", config.DEFAULT_HPC_MAX_ACTIVE_JOBS))
        if planned > cap:
            raise SystemExit(
                f"{planned} jobs exceeds the scheduler's {cap}-job cap; submit fewer seeds"
            )
        cold_command = preview_hpc_command(task, runner_cfg, seeds[0] if seeds else None)
        logger.info(f"Cold job command: {cold_command}")
        problems = preflight(run_folder, targets, cold_command)
        if problems:
            for problem in problems:
                logger.error(f"preflight: {problem}")
            if args.strict:
                raise SystemExit("Aborting on the preflight mismatches above (--strict)")
            logger.warning("Continuing despite the mismatches above; the two arms may "
                           "not be comparable")
        else:
            logger.info("Preflight: the cold command matches the originals' apart from "
                        "the warm-start flags")

    history_path = os.path.join(names["output_dir"], COLD_HISTORY)
    meta = {
        "source_run": run_folder,
        "task": task,
        "env_file": task_cfg["env_file"],
        "created_at": time.time(),
        "settings": str(args.settings),
        "refineconfig": str(args.refineconfig),
        "backend": settings["runner"].get("backend", "local"),
        "max_iterations": settings["runner"].get("env", {}).get("MAX_ITERATIONS"),
        "num_envs": settings["runner"].get("env", {}).get("NUM_ENVS"),
        "plasticity": settings["runner"].get("env", {}).get("plasticity"),
        "scoring_mode": refine_cfg.get("scoring_mode", "global_max"),
        "checkpoint_sel_mode": refine_cfg.get("candidate_checkpoint_sel_mode", "best"),
        "seeds": seeds,
        "job_name_prefix": names["job_name_prefix"],
        "build_root": names["build_root"],
        "command": cold_command,
        "note": "Cold runs log epochs 1..max_iterations; add each target's "
                "step_offset to align them with their warm counterpart.",
    }

    if args.dry_run:
        logger.info(f"[dry-run] would write {history_path}")
        for target in targets:
            for seed in seeds or [None]:
                logger.info(f"[dry-run] {cold_tag_for(target, seed)} -> "
                            f"{final_dir_for(names['output_dir'], target, seed)}")
        return 0

    history = RewardHistory()
    records = create_records(history, targets, seeds)
    scorer = FitnessScorer(scoring_mode=refine_cfg.get("scoring_mode", "global_max"))
    processor = ResultProcessor(
        checkpoint_sel_mode=refine_cfg.get("candidate_checkpoint_sel_mode", "best")
    )

    # Settle what actually needs running before writing anything or touching a
    # backend, so a clash with an earlier attempt leaves no half-written history,
    # and a pure re-collection (--skip-existing over a finished set) needs no
    # docker or scheduler at all.
    to_run = records
    if args.skip_existing:
        to_run = adopt_existing(records, targets, names["output_dir"], processor)
    else:
        for target in targets:
            for seed in seeds or [None]:
                final_dir = final_dir_for(names["output_dir"], target, seed)
                if os.path.exists(final_dir):
                    raise SystemExit(
                        f"{final_dir} already exists. Re-run with --skip-existing to "
                        "keep it, or remove it first."
                    )

    # Written before anything is submitted: the pairing and the offsets are the
    # part that cannot be reconstructed if this process dies mid-batch.
    write_history(history_path, meta, targets, scorer)
    logger.info(f"Wrote {history_path}")

    if to_run:
        logger.info(f"Dispatching {len(to_run)} cold job(s) (no checkpoint, "
                    f"{meta['max_iterations']} epochs each)")
        evaluator = RewardEvaluator(
            tasks_repo=settings["tasks_repo"],
            env_file_rel=task_cfg["env_file"],
            task=task,
            runner=runner_cfg,
            output_dir=names["output_dir"],
            build_root=names["build_root"],
            warm_start=None,  # cold: no checkpoint is delivered, so no --warm_start flags
            checkpoint_sel_mode=refine_cfg.get("candidate_checkpoint_sel_mode", "best"),
        )
        try:
            evaluator.evaluate(to_run)  # no checkpoint_path -> cold, unextended budget
        finally:
            nest_seed_dirs(names["output_dir"], targets)
            write_history(history_path, meta, targets, scorer)
    else:
        logger.info("Every cold run already had results; nothing to dispatch")

    scorer.score_all(records)
    write_history(history_path, meta, targets, scorer)
    report(targets, scorer)
    logger.info(f"Results collected in {names['output_dir']}, paired in {history_path}")

    return 0 if all(r.succeeded for r in records) else 1


if __name__ == "__main__":
    sys.exit(main())
