#!/usr/bin/env python3
"""
Entry point for the ARD (Autonomous RL Designer) reward-refinement pipeline.

Stage 2 — Automated reward refinement (Eureka-style):
  1. An LLM proposes complete `_get_rewards` methods for an ard-isaaclab-tasks env.
  2. Each candidate is spliced into a fresh copy of the task repo (AST injection)
     and submitted as a job to the Parallel Coordination System (PCS) coordinator,
     which trains it (PPO / rl_games) on a GPU worker.
  3. Finished jobs are scored by the task's fixed `fitness_function` metric; the
     best candidate's training summary is fed back to the LLM for the next round.

Usage:
    export TOKEN=pcs_...                 # coordinator bearer token
    export OPENROUTER_API_KEY=...            # LLM key
    python main.py --refine                       # uses configs/taskconfig.yaml
    python main.py --refine --task cartpole        # by dir name; or --task Isaac-ARD-Humanoid-v0
    python main.py --refine --taskconfig configs/taskconfig.yaml \
                   --settings configs/settings.yaml --refineconfig configs/refineconfig.yaml
"""

import os
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor

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
    output_dir = os.path.join(
        os.path.expanduser(settings.get("output_dir", "./runs")), task_cfg["task"]
    )

    evaluator = RewardEvaluator(
        tasks_repo=tasks_repo,
        env_file_rel=task_cfg["env_file"],
        task=task_cfg["task"],
        coordinator=settings["coordinator"],
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
        evaluator.evaluate(run_records)
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
        eval_records = [
            history.new_record(
                iteration=i, index=k, phase="eval", tag=f"iter{i}_eval_{k}",
                seed=base_seed + k, model=agent.model, temperature=agent.temperature,
                reward_method=best.reward_method, raw_response=best.raw_response,
                status=STATUS_GENERATED,
            )
            for k in range(num_eval)
        ]
        evaluator.evaluate(eval_records)
        scorer.score_all(eval_records)
        best_eval = scorer.select_best(eval_records)

        summary_path = (best_eval or best).summary_path
        if best_eval:
            logger.info(f"Eval fitness (best of {num_eval}): {best_eval.fitness:.4f}")

        # --- Feedback phase: fold the outcome back into the conversation -----
        # Today only the winner is fed back; because the history retains every
        # candidate with its summary, feeding the whole batch back later is just
        # a different read of `run_records` — no structural change needed.
        feedback = agent.receive_feedback(best.raw_response, summary_path=summary_path)
        history.update(best, feedback_text=feedback)
        history.save_json()

    logger.info("Refinement loop complete")
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
    args = parser.parse_args()

    settings = load_yaml_config(args.settings)
    if args.task:
        taskconfig_path = resolve_task_config(args.task, settings["tasks_repo"])
    else:
        taskconfig_path = args.taskconfig
    task_cfg = load_yaml_config(taskconfig_path)

    if args.refine:
        refine_cfg = load_yaml_config(args.refineconfig)
        run_refinement(settings, task_cfg, refine_cfg)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
