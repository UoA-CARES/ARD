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
    export PCS_TOKEN=pcs_...                 # coordinator bearer token
    export OPENROUTER_API_KEY=...            # LLM key
    python main.py --refine
    python main.py --refine --taskconfig configs/taskconfig.yaml \
                   --settings configs/settings.yaml --refineconfig configs/refineconfig.yaml
"""

import os
import argparse
import logging

import yaml
from tqdm import tqdm

from src.refinement.llm_agent import EurekaAgent
from src.evaluation import RewardEvaluator

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
    )

    agent = EurekaAgent(
        task_description=task_cfg["description"],
        reward_template=evaluator.get_reward_template(),
        env_source=evaluator.get_env_source(),
        agent_config=refine_cfg.get("agent", {}),
    )

    iterations = int(refine_cfg.get("iteration", 1))
    num_eval = int(refine_cfg.get("num_eval", 1))
    base_seed = int(refine_cfg.get("base_seed", 0))
    max_iterations = int(task_cfg.get("max_iterations", 100))

    history = []
    for i in range(1, iterations + 1):
        logger.info(f"=== Refinement iteration {i}/{iterations} ===")

        # --- Run phase: generate and evaluate a batch of candidates ----------
        reward_methods, raw_responses = [], []
        for _ in tqdm(range(agent.samples), desc=f"iter {i}: generating rewards"):
            method, raw = agent.func_gen(agent.messages)
            reward_methods.append(method)
            raw_responses.append(raw)

        logger.info(f"Evaluating {len(reward_methods)} candidate(s)")
        best_run, run_logs = evaluator.evaluate(
            reward_methods, max_iterations=max_iterations, tag_prefix=f"iter{i}_run"
        )

        if best_run is None:
            logger.error("No candidate trained successfully; requesting a rewrite")
            agent.receive_feedback(raw_responses[0], summary_path=None)
            history.append({"iteration": i, "best_run": None})
            continue

        best_idx = best_run["idx"]
        best_method = reward_methods[best_idx]
        best_response = raw_responses[best_idx]
        logger.info(
            f"Best candidate idx={best_idx} fitness={best_run['fitness']:.4f}"
        )

        # --- Eval phase: re-train the best reward num_eval times to score it --
        seeds = [base_seed + k for k in range(num_eval)]
        best_eval, eval_logs = evaluator.evaluate(
            [best_method] * num_eval,
            max_iterations=max_iterations,
            tag_prefix=f"iter{i}_eval",
            seeds=seeds,
        )

        summary_path = (best_eval or best_run)["summary_path"]
        if best_eval:
            logger.info(f"Eval fitness (best of {num_eval}): {best_eval['fitness']:.4f}")
        agent.receive_feedback(best_response, summary_path=summary_path)

        history.append({
            "iteration": i,
            "best_run": best_run,
            "best_eval": best_eval,
            "run_logs": run_logs,
            "eval_logs": eval_logs,
        })

    logger.info("Refinement loop complete")
    return history


def main():
    parser = argparse.ArgumentParser(description="ARD reward-refinement pipeline")
    parser.add_argument("--refine", action="store_true",
                        help="Run LLM-based reward-function refinement")
    parser.add_argument("--settings", type=str, default="configs/settings.yaml",
                        help="Path to settings YAML")
    parser.add_argument("--taskconfig", type=str, default="configs/taskconfig.yaml",
                        help="Path to task configuration YAML")
    parser.add_argument("--refineconfig", type=str, default="configs/refineconfig.yaml",
                        help="Path to refinement configuration YAML")
    args = parser.parse_args()

    settings = load_yaml_config(args.settings)
    task_cfg = load_yaml_config(args.taskconfig)

    if args.refine:
        refine_cfg = load_yaml_config(args.refineconfig)
        run_refinement(settings, task_cfg, refine_cfg)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
