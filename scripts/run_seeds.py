#!/usr/bin/env python3
"""
Run the refinement pipeline several times, once per base seed.

`main.py` takes no --seed flag: the seed comes from `base_seed` in
configs/refineconfig.yaml, and the output location comes from `output_dir` in
configs/settings.yaml. This driver therefore writes one throwaway copy of both
configs per seed (under runs/_seed_sweep/<task>/seed_<seed>/) and calls main.py
with them, so each run gets its own base_seed AND its own output tree. Without
the separate output tree the runs would overwrite each other: candidate tags
(`iter1_run_0`, ...) and `reward_history.json` are identical between runs.

This script's flags ARE main.py's flags: it builds its parser from main.py's own
`build_parser()`, so every main.py flag is accepted here and forwarded verbatim
to every per-seed run, and a flag added to main.py later (--plasticity, ...)
needs no change here. The only extras are the sweep's own --seeds / --sweep-dir /
--dry-run / --stop-on-error. --settings and --refineconfig name the BASE configs
that are copied per seed; the copies are what main.py actually receives.

Nothing is defaulted: --task and --seeds must be passed, so what a sweep is
running is never a guess.

Runs are sequential: on the hpc backend one run already fans `agent.sample` jobs
out onto the scheduler at once.

Usage:
    export OPENROUTER_API_KEY=...
    python scripts/run_seeds.py --refine --task Isaac-ARD-Repose-Cube-Shadow-Direct-v0 \
        --seeds 42 43 44
    python scripts/run_seeds.py --refine --task cartpole --seeds 42 --warm-start --dry-run
    python scripts/run_seeds.py --refine --task shadow_hand --seeds 42 43 44 45 46 --warm-start

Results land in <output_dir>/seed_<seed>/<task>/, console log per seed in
runs/_seed_sweep/<task>/seed_<seed>/console.log.
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
sys.path.insert(0, str(REPO_ROOT))

try:
    from main import build_parser
except ImportError as e:  # pragma: no cover - environment problem, not logic
    sys.exit(
        f"Cannot import main.py from {REPO_ROOT}: {e}\n"
        "Run this from the ARD checkout with the pipeline's environment active."
    )

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,  # importing main.py already called basicConfig; ours must win
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


def _long_option(action):
    """The flag to emit for an action: its long form when it has one."""
    return next(
        (o for o in action.option_strings if o.startswith("--")),
        action.option_strings[0],
    )


def build_main_cmd(main_actions, args, overrides):
    """Rebuild main.py's argv from the flags we parsed with main.py's parser.

    Driven off the parser's own actions rather than a hand-written list, so
    whatever main.py accepts is what each seed gets. `overrides` replaces a
    flag's value by dest — used to point --settings/--refineconfig at this
    seed's config copies instead of the base ones the user named.

    An action shape this doesn't know how to render raises rather than being
    dropped: a silently missing flag would mean the sweep quietly ran a
    different experiment than the same command line on main.py.
    """
    argv = [sys.executable, "-u", "main.py"]
    for action in main_actions:
        value = overrides.get(action.dest, getattr(args, action.dest, None))
        flag = _long_option(action)

        if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction,
                               argparse._StoreConstAction)):
            # Flag-only: emit it exactly when it flips the default (a store_true
            # left False, or a store_false left True, must stay off the command).
            if value != action.default:
                argv.append(flag)
        elif isinstance(action, argparse._CountAction):
            argv.extend([flag] * int(value or 0))
        elif isinstance(action, argparse._AppendAction):
            for item in value or []:
                argv.extend([flag, str(item)])
        elif isinstance(action, argparse._StoreAction):
            if value is None:
                continue  # an unset optional: let main.py apply its own default
            if action.nargs in (None, "?"):
                argv.extend([flag, str(value)])
            else:  # "+", "*", or an int count
                if not value:
                    continue
                argv.extend([flag, *(str(v) for v in value)])
        else:
            raise ValueError(
                f"run_seeds cannot forward {flag} ({type(action).__name__}); "
                "teach build_main_cmd how to render it."
            )
    return argv


def run_one(cmd, log_path):
    """Run one main.py invocation, teeing its output to log_path."""
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


def build_arg_parser():
    """main.py's parser plus the sweep's own flags.

    Returns (parser, main_actions): `main_actions` is captured before the
    sweep-only flags are added, so the split between "forward to main.py" and
    "consumed here" keeps itself up to date as either side grows flags.
    """
    main_parser = build_parser(add_help=False)
    main_actions = [a for a in main_parser._actions if a.option_strings]

    # No default task: an experiment this long should never be a guess.
    for action in main_actions:
        if action.dest == "task":
            action.required = True

    parser = argparse.ArgumentParser(
        description="Run the ARD refinement pipeline once per base seed. Accepts "
                    "every main.py flag and forwards it to each per-seed run; "
                    "--settings/--refineconfig name the base configs copied per seed.",
        parents=[main_parser],
    )
    parser.add_argument("--seeds", type=int, nargs="+", required=True,
                        help="base seeds to run, e.g. --seeds 42 43 44")
    parser.add_argument("--sweep-dir", type=str, default="runs/_seed_sweep",
                        help="where per-seed configs and console logs are written")
    parser.add_argument("--dry-run", action="store_true",
                        help="write the per-seed configs and print the commands, run nothing")
    parser.add_argument("--stop-on-error", action="store_true",
                        help="stop the sweep if a seed fails (default: carry on)")
    return parser, main_actions


def main():
    parser, main_actions = build_arg_parser()
    args = parser.parse_args()

    if not args.refine:
        parser.error(
            "--refine is required: it is main.py's only mode, so without it every "
            "seed would just print main.py's help"
        )
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

        cmd = build_main_cmd(main_actions, args, {
            "settings": settings_path,
            "refineconfig": refine_path,
        })

        if args.dry_run:
            logger.info(f"[dry-run] $ {' '.join(cmd)}")
            continue

        started = time.time()
        code = run_one(cmd, seed_dir / "console.log")
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
