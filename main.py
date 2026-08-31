#!/usr/bin/env python3
"""
Entry point for the ARD (Autonomous RL Designer) reward-refinement pipeline.

Stage 2 — Automated reward refinement (Eureka-style):
  1. An LLM proposes complete `_get_rewards` methods for an ard-isaaclab-tasks env.
  2. Each candidate is spliced into a fresh copy of the task repo (AST injection)
     and built + run as a local docker job (PPO / rl_games), one at a time.
  3. Finished jobs are scored by the task's fixed `fitness_function` metric; the
     best candidate is re-trained `num_eval` times to de-noise its score, its
     checkpoint carried into the next iteration (warm-starting), and its
     training summary fed back to the LLM for the next round.

Usage:
    export OPENROUTER_API_KEY=...            # LLM key
    python main.py --refine                       # uses configs/taskconfig.yaml
    python main.py --refine --task cartpole        # by dir name; or --task Isaac-ARD-Humanoid-v0
    python main.py --refine --taskconfig configs/taskconfig.yaml \
                   --settings configs/settings.yaml --refineconfig configs/refineconfig.yaml
"""

import os
import json
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import yaml
from tqdm import tqdm

from src.refinement.llm_agent import EurekaAgent
from src.evaluation import RewardEvaluator, FitnessScorer
from src.reward_history import RewardHistory, STATUS_GENERATED, STATUS_GEN_FAILED

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_yaml_config(config_path):
    """Safely load a YAML configuration file."""
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration: {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"Config file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing {config_path}: {e}")
        raise


def resolve_task_config(task_name, tasks_repo):
    """Resolve a task selector to its ard_meta.yaml path inside tasks_repo.

    Each task directory under ``source/ard_tasks/ard_tasks/tasks/direct/<dir>/``
    carries an ``ard_meta.yaml`` with the same keys as configs/taskconfig.yaml.
    ``task_name`` may be either the directory name (e.g. ``locomotion``) or the
    registered task ID (e.g. ``Isaac-ARD-Humanoid-v0``).
    """
    tasks_repo = os.path.abspath(os.path.expanduser(tasks_repo))
    direct_root = os.path.join(tasks_repo, "source/ard_tasks/ard_tasks/tasks/direct")

    # Map both the directory name and the registered task ID to each meta file.
    by_dir, by_id = {}, {}
    for d in sorted(os.listdir(direct_root)):
        meta_path = os.path.join(direct_root, d, "ard_meta.yaml")
        if not os.path.isfile(meta_path):
            continue
        task_id = (yaml.safe_load(open(meta_path)) or {}).get("task")
        by_dir[d] = meta_path
        if task_id:
            by_id[task_id] = (d, meta_path)

    if task_name in by_dir:
        return by_dir[task_name]
    if task_name in by_id:
        return by_id[task_name][1]

    available = ", ".join(f"{d} ({tid})" for tid, (d, _) in sorted(by_id.items())) or "(none)"
    raise FileNotFoundError(
        f"No ard_meta.yaml for task '{task_name}' in {direct_root}. "
        f"Available: {available}"
    )


def run_refinement(settings, task_cfg, refine_cfg):
    """Run the Eureka refinement loop for one task."""
    tasks_repo = settings["tasks_repo"]
    # Timestamped per-execution directory: re-running a task must not delete the
    # previous execution's logs/checkpoints (each job rmtree's its own work_dir)
    # nor overwrite its reward_history.json.
    output_dir = os.path.join(
        os.path.expanduser(settings.get("output_dir", "./runs")),
        task_cfg["task"],
        datetime.now().strftime("%Y%m%d-%H%M%S"),
    )

    evaluator = RewardEvaluator(
        tasks_repo=tasks_repo,
        env_file_rel=task_cfg["env_file"],
        task=task_cfg["task"],
        runner=settings["runner"],
        output_dir=output_dir,
        build_root=settings.get("build_root"),
    )
    scorer = FitnessScorer()

    agent = EurekaAgent(
        task_description=task_cfg["description"],
        reward_template=evaluator.get_reward_template(),
        env_source=evaluator.get_env_source(),
        agent_config=refine_cfg.get("agent", {}),
    )

    iterations = int(refine_cfg.get("iteration", 1))
    num_eval = int(refine_cfg.get("num_eval", 1))
    base_seed = int(refine_cfg.get("base_seed", 0))
    max_workers = min(agent.samples, int(refine_cfg.get("max_workers", agent.samples)))
    warm_start = bool(refine_cfg.get("warm_start", False))

    # The previous iteration's winning candidate, whose checkpoint the next
    # iteration resumes from; None means cold start (always true for
    # iteration 1, since there is no previous best yet). Kept as the actual
    # record (not just its checkpoint path) so the log line and the
    # `warm_started_from` field persisted onto each new candidate both name
    # the exact source candidate — needed to verify, after the fact, that
    # warm-starting picked up the candidate and checkpoint it should have.
    warm_start_source = None

    # A dedicated, append-only audit trail of every warm-start decision, kept
    # separate from reward_history.json so "who was warm-started from whom"
    # can be checked at a glance without cross-referencing every candidate's
    # own record. Written to warm_start_history.json (see save_warm_start_log).
    warm_start_log = []

    def save_warm_start_log():
        path = os.path.join(output_dir, "warm_start_history.json")
        os.makedirs(output_dir, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(warm_start_log, fh, indent=2)
        return path

    # The single source of truth: every candidate's generation -> evaluation ->
    # judgement -> feedback lifecycle is recorded here, and it is thread-safe so
    # the generation fan-out below can register records concurrently.
    history = RewardHistory(output_dir=output_dir)

    for i in range(1, iterations + 1):
        logger.info(f"=== Refinement iteration {i}/{iterations} ===")

        # --- Generation phase: propose a batch of candidates -----------------
        # func_gen is a network-bound LLM call, so fan the samples out across
        # threads (the GIL is released during I/O). Threads only read
        # agent.messages (mutated later by receive_feedback) and each registers
        # its own record by index, so correctness no longer relies on ordering.
        def _generate(k):
            tag = f"iter{i}_run_{k}"
            # Distinct seed per candidate so the batch explores varied reward
            # designs instead of collapsing to one (identical prompts alone can
            # return identical completions under provider-side determinism).
            gen_seed = base_seed + i * 1000 + k
            try:
                method, raw = agent.func_gen(agent.messages, seed=gen_seed)
                history.new_record(
                    iteration=i, index=k, phase="run", tag=tag,
                    model=agent.model, temperature=agent.temperature,
                    gen_seed=gen_seed,
                    reward_method=method, raw_response=raw, status=STATUS_GENERATED,
                )
            except RuntimeError as e:
                logger.error(f"[{tag}] generation failed: {e}")
                history.new_record(
                    iteration=i, index=k, phase="run", tag=tag,
                    model=agent.model, temperature=agent.temperature,
                    gen_seed=gen_seed,
                    gen_error=str(e), status=STATUS_GEN_FAILED,
                )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            list(tqdm(
                executor.map(_generate, range(agent.samples)),
                total=agent.samples,
                desc=f"iter {i}: generating rewards",
            ))

        run_records = history.for_iteration(i, phase="run")

        # --- Run phase: dispatch + capture (evaluator), then judge (scorer) --
        logger.info(f"Evaluating {sum(r.has_method for r in run_records)} candidate(s)")
        if warm_start and warm_start_source:
            logger.info(
                f"Warm-starting from {warm_start_source.tag} "
                f"(fitness={warm_start_source.fitness:.4f}): "
                f"{warm_start_source.checkpoint_path}"
            )
            for r in run_records:
                history.update(r, warm_started_from=warm_start_source.tag)
            warm_start_log.append({
                "iteration": i,
                "source_tag": warm_start_source.tag,
                "source_iteration": warm_start_source.iteration,
                "source_fitness": warm_start_source.fitness,
                "checkpoint_path": warm_start_source.checkpoint_path,
                "applied_to": [r.tag for r in run_records],
            })
            save_warm_start_log()
        evaluator.evaluate(
            run_records,
            checkpoint_path=warm_start_source.checkpoint_path if warm_start and warm_start_source else None,
        )
        scorer.score_all(run_records)
        best = scorer.select_best(run_records)

        if best is None:
            logger.error("No candidate trained successfully; requesting a rewrite")
            seed_record = next((r for r in run_records if r.raw_response), None)
            feedback = agent.receive_feedback(
                seed_record.raw_response if seed_record else "", summary_path=None
            )
            if seed_record:
                history.update(seed_record, feedback_text=feedback)
            history.save_json()
            continue

        logger.info(f"Best candidate idx={best.index} fitness={best.fitness:.4f}")

        # --- Eval phase: re-train the best reward num_eval times to score it --
        # eval_records = [
        #     history.new_record(
        #         iteration=i, index=k, phase="eval", tag=f"iter{i}_eval_{k}",
        #         seed=base_seed + k + 1, model=agent.model, temperature=agent.temperature,
        #         reward_method=best.reward_method, raw_response=best.raw_response,
        #         status=STATUS_GENERATED,
        #     )
        #     for k in range(num_eval)
        # ]
        # evaluator.evaluate(
        #     eval_records,
        #     checkpoint_path=warm_start_source.checkpoint_path if warm_start else None,
        # )
        # scorer.score_all(eval_records)
        # best_eval = scorer.select_best(eval_records)
        # best_eval = None
        # winner = best_eval or best
        # summary_path = winner.summary_path
        # if best_eval:
        #     logger.info(f"Eval fitness (best of {num_eval}): {best_eval.fitness:.4f}")

        # Carry this iteration's winning checkpoint into the next iteration.
        if warm_start and best.checkpoint_path:
            warm_start_source = best

        # --- Feedback phase: fold the outcome back into the conversation -----
        # The summary must come from the run that was scored: the LLM is shown
        # its own code next to that same run's numbers. Pairing the code with a
        # different training run (a re-train on another seed) would describe two
        # different trajectories and make the reflection misleading.
        # Today only the winner is fed back; because the history retains every
        # candidate with its summary, feeding the whole batch back later is just
        # a different read of `run_records` — no structural change needed.
        feedback = agent.receive_feedback(
            best.raw_response, summary_path=best.summary_path
        )
        history.update(best, feedback_text=feedback)
        history.save_json()

    logger.info("Refinement loop complete")

    # --- Eval phase: score the run's best reward over num_eval seeds ---------
    # Once, after the search, on the best candidate of *all* iterations. Running
    # it per iteration would spend the multi-seed budget re-training local bests
    # that a later iteration discards, and still leave the final winner scored by
    # a single seed. `select_best` over every run record marks that winner (and
    # only it) as `selected_best` in the history.
    best = scorer.select_best([r for r in history.all() if r.phase == "run"])
    if best is None:
        logger.error("No candidate trained successfully in any iteration; skipping eval")
        return history

    logger.info(
        f"=== Eval phase: re-training {best.tag} (fitness {best.fitness:.4f}) "
        f"on {num_eval} seed(s) ==="
    )
    eval_records = [
        history.new_record(
            iteration=best.iteration, index=k, phase="eval",
            tag=f"{best.tag}_eval_{k}",
            seed=base_seed + k + 1, model=best.model, temperature=best.temperature,
            reward_method=best.reward_method, raw_response=best.raw_response,
            status=STATUS_GENERATED,
        )
        for k in range(num_eval)
    ]
    evaluator.evaluate(eval_records)
    scorer.score_all(eval_records)
    values, mean, std = scorer.summarise(eval_records)
    logger.info(
        f"Eval fitness over {len(values)}/{num_eval} seed(s): "
        f"mean={mean:.4f} std={std:.4f} "
        f"[{', '.join(f'{v:.4f}' for v in values)}]"
    )
    history.save_json()
    return history


def main():
    parser = argparse.ArgumentParser(description="ARD reward-refinement pipeline")
    parser.add_argument("--refine", action="store_true",
                        help="Run LLM-based reward-function refinement")
    parser.add_argument("--settings", type=str, default="configs/settings.yaml",
                        help="Path to settings YAML")
    parser.add_argument("--task", type=str, default=None,
                        help="Registered task name (resolves its ard_meta.yaml in "
                             "settings.tasks_repo). Takes precedence over --taskconfig.")
    parser.add_argument("--taskconfig", type=str, default="configs/taskconfig.yaml",
                        help="Path to task configuration YAML (used if --task is omitted)")
    parser.add_argument("--refineconfig", type=str, default="configs/refineconfig.yaml",
                        help="Path to refinement configuration YAML")
    parser.add_argument("--warm-start", action="store_true",
                        help="Resume each iteration from the previous iteration's "
                             "de-noised winner instead of random weights (default: "
                             "false, or refineconfig.yaml's warm_start).")
    args = parser.parse_args()

    settings = load_yaml_config(args.settings)
    if args.task:
        taskconfig_path = resolve_task_config(args.task, settings["tasks_repo"])
    else:
        taskconfig_path = args.taskconfig
    task_cfg = load_yaml_config(taskconfig_path)

    if args.refine:
        refine_cfg = load_yaml_config(args.refineconfig)
        if args.warm_start:
            refine_cfg["warm_start"] = True
        run_refinement(settings, task_cfg, refine_cfg)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
