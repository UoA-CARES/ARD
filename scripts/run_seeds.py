#!/usr/bin/env python3
"""
Run the refinement pipeline several times, once per base seed.

`main.py` takes no --seed flag: the seed comes from `base_seed` in
configs/refineconfig.yaml, and the output location comes from `output_dir` in
configs/settings.yaml. This driver therefore writes one throwaway copy of both
configs per seed (under runs/_seed_sweep/seed_<seed>/) and calls main.py with
them, so each run gets its own base_seed AND its own output tree. Without the
separate output tree the runs would overwrite each other: candidate tags
(`iter1_run_0`, ...) and `reward_history.json` are identical between runs.

Runs are sequential by default: on the hpc backend one run already fans
`agent.sample` jobs out onto the scheduler at once.

Usage:
    export OPENROUTER_API_KEY=...
    python scripts/run_seeds.py                       # seeds 42..46, shadow hand
    python scripts/run_seeds.py --seeds 44 45 46      # resume a partial sweep
    python scripts/run_seeds.py --task cartpole --dry-run

Results land in <output_dir>/seed_<seed>/<task>/, console log per seed in
runs/_seed_sweep/seed_<seed>/console.log.
"""

import os
import sys
import time
import copy
import logging
import argparse
import subprocess
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TASK = "Isaac-ARD-Repose-Cube-Shadow-Direct-v0"
DEFAULT_SEEDS = [50]

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_seeds")


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def write_seed_configs(settings, refine_cfg, seed, sweep_dir):
    """Write per-seed copies of settings.yaml / refineconfig.yaml.

    Everything that would otherwise collide between runs is suffixed with the
    seed: the output tree, the codebase staging root, and (on the hpc backend)
    the scheduler job-name prefix.
    """
    refine_cfg = copy.deepcopy(refine_cfg)
    refine_cfg["base_seed"] = seed

    settings = copy.deepcopy(settings)
    settings["output_dir"] = os.path.join(
        settings.get("output_dir", "./runs"), f"seed_{seed}"
    )
    if settings.get("build_root"):
        settings["build_root"] = os.path.join(settings["build_root"], f"seed_{seed}")
    hpc = settings.get("runner", {}).get("hpc")
    if isinstance(hpc, dict):
        hpc["job_name_prefix"] = f"{hpc.get('job_name_prefix', 'ard')}_s{seed}"

    sweep_dir.mkdir(parents=True, exist_ok=True)
    settings_path = sweep_dir / "settings.yaml"
    refine_path = sweep_dir / "refineconfig.yaml"
    with open(settings_path, "w") as f:
        yaml.safe_dump(settings, f, sort_keys=False)
    with open(refine_path, "w") as f:
        yaml.safe_dump(refine_cfg, f, sort_keys=False)
    return settings_path, refine_path


def run_one(task, settings_path, refine_path, log_path):
    """Run main.py --refine for one seed, teeing its output to log_path."""
    cmd = [
        sys.executable, "-u", "main.py", "--refine",
        "--task", task,
        "--settings", str(settings_path),
        "--refineconfig", str(refine_path),
    ]
    logger.info(f"$ {' '.join(cmd)}")
    with open(log_path, "w") as log:
        proc = subprocess.Popen(
            cmd, cwd=REPO_ROOT, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
            log.flush()
        return proc.wait()


def main():
    parser = argparse.ArgumentParser(
        description="Run the ARD refinement pipeline once per base seed"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
                        help=f"base seeds to run (default: {DEFAULT_SEEDS})")
    parser.add_argument("--task", type=str, default=DEFAULT_TASK,
                        help=f"task dir name or registered task ID (default: {DEFAULT_TASK})")
    parser.add_argument("--settings", type=str, default="configs/settings.yaml",
                        help="base settings YAML to copy per seed")
    parser.add_argument("--refineconfig", type=str, default="configs/refineconfig.yaml",
                        help="base refinement YAML to copy per seed")
    parser.add_argument("--sweep-dir", type=str, default="runs/_seed_sweep",
                        help="where per-seed configs and console logs are written")
    parser.add_argument("--dry-run", action="store_true",
                        help="write the per-seed configs and print the commands, run nothing")
    parser.add_argument("--stop-on-error", action="store_true",
                        help="stop the sweep if a seed fails (default: carry on)")
    args = parser.parse_args()

    if not args.dry_run and not os.environ.get("OPENROUTER_API_KEY"):
        logger.error("OPENROUTER_API_KEY is not set")
        return 1

    settings = load_yaml(REPO_ROOT / args.settings)
    refine_cfg = load_yaml(REPO_ROOT / args.refineconfig)
    sweep_root = REPO_ROOT / args.sweep_dir / args.task

    results = []
    for n, seed in enumerate(args.seeds, start=1):
        seed_dir = sweep_root / f"seed_{seed}"
        settings_path, refine_path = write_seed_configs(
            settings, refine_cfg, seed, seed_dir
        )
        logger.info(f"=== Seed {seed} ({n}/{len(args.seeds)}) -> {seed_dir} ===")

        if args.dry_run:
            logger.info(
                f"[dry-run] python -u main.py --refine --task {args.task} "
                f"--settings {settings_path} --refineconfig {refine_path}"
            )
            continue

        started = time.time()
        code = run_one(args.task, settings_path, refine_path, seed_dir / "console.log")
        elapsed = time.time() - started
        results.append((seed, code, elapsed))
        logger.info(
            f"=== Seed {seed} finished: exit={code} in {elapsed / 3600:.2f} h ==="
        )
        if code != 0 and args.stop_on_error:
            logger.error("Stopping the sweep (--stop-on-error)")
            break

    if args.dry_run:
        return 0

    logger.info("=== Sweep summary ===")
    for seed, code, elapsed in results:
        state = "ok" if code == 0 else f"FAILED (exit {code})"
        logger.info(f"  seed {seed}: {state}, {elapsed / 3600:.2f} h")
    return 0 if all(code == 0 for _, code, _ in results) else 1


if __name__ == "__main__":
    sys.exit(main())
