#!/usr/bin/env python3
"""
Main entry point for the Isaac Sim2Real RL Pipeline.

# Current only support Rl_Games

This script provides a unified interface to run the complete 3-step pipeline:
1. Simulation training with Isaac Lab
2. LLM-based performance refinement using Eureka framework
3. Sim-to-real transfer with domain randomization and DrEureka

Usage:
    
    # Config
    python main.py * --help  # Show all options
    python main.py * --version  # Show version info
    
    # Stage 1
    python main.py * --list-template-tasks  # List available official template RL tasks from IsaacLab
                   * --create   # Create a new custom RL task environment, your will access a interactive shell to define the task
                   --taskconfig TASKCONFIG  # /home/lee/code/Isaac-Sim2Real-Pipeline/configs/taskconfig.yaml.
                   --simtrain
                   --simtest  
                   --simplay

    # Stage 2
    python main.py --refine  # Run LLM-based reward function refinement using Eureka framework
                   --refineconfig REFINECONFIG  # Path to Eureka configuration file

    # Stage 3


    * not shown on the prototype
"""

import os
import yaml
import argparse
import subprocess
import logging
from src.refinement.llm_agent import EurekaAgent
from src.refinement.files_operation import load_env_cfg, load_prompts
from src.evaluation import RewardEvaluator
from src.utils.common import load_machine_pool
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_yaml_config(config_path):
    """Safely loads a YAML configuration file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Successfully loaded configuration from: {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"The file was not found at {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file {config_path}: {e}")
        raise




def main():
    """
    Main function to load configurations and run the main logic.
    """
    # Set up argument Parser to accept file paths from the command line
    parser = argparse.ArgumentParser(description="TODO:")
    parser.add_argument('--taskconfig', type=str, default="configs/taskconfig.yaml", required=False, help="Path to the task configuration YAML file.")
    parser.add_argument('--simtrain', action='store_true', help="Run simulation training using the provided task configuration")
    parser.add_argument('--simtest', action='store_true', help="Run simulation testing using the provided task configuration")
    parser.add_argument('--simplay', action='store_true', help="Run simulation play using the provided task configuration")
    parser.add_argument('--refine', action='store_true', help="Run LLM-based reward function refinement using Eureka framework")
    parser.add_argument('--refineconfig', type=str, default="configs/refineconfig.yaml", required=False, help="Path to the refine configuration YAML file.")
    
    args = parser.parse_args()

    # Setting Configurations: The localised workspace and python envionment
    settings_yaml = load_yaml_config("configs/settings.yaml")
    # Load task configuration
    task_yaml = load_yaml_config(args.taskconfig)
    
    # Build workspace path robustly using os.path.join, expanduser and abspath
    root = settings_yaml.get('workspace') if settings_yaml else None
    task_ws = task_yaml.get('workspace') if task_yaml else None
    if not root or not task_ws:
        logger.warning("'workspace' missing in settings.yaml or taskconfig.yaml; attempting to join available parts")
    workspace = os.path.abspath(os.path.expanduser(os.path.join(root or '', task_ws or '')))
    if not workspace:
        raise ValueError("Error: unable to determine workspace from settings.yaml and taskconfig.yaml")
    logger.info(f"Successfully determined local workspace: {workspace}")

    task_config = {
        "workspace": workspace,
        "env_cfg_path": os.path.join(workspace, task_yaml.get('env_cfg_path')),
        "logs_path": os.path.join(workspace, task_yaml.get('logs_path')),
        "task_description": task_yaml.get('description'),
    }

    # command parameters
    # task = task_yaml.get('task')
    # checkpoint = task_yaml.get('checkpoint') 
    # whichpython = settings_yaml.get('python_env')
    # num_envs = task_yaml.get('num_envs')
    # seed = task_yaml.get('seed')
    # max_iterations = task_yaml.get('max_iterations')

    if args.refine:
        logger.info(f"Loading refine configuration from {args.refineconfig}")
        refine_config = load_yaml_config(args.refineconfig)

        logger.info("Initializing environment and LLM agents")
        subprocess.run(["git", "checkout", "."], cwd=settings_yaml.get('workspace'))

        # Init LLM agents
        llm_agent = EurekaAgent(task_config=task_config, agent_config=refine_config.get('agent', {}))

        # Load machine pool
        machine_pool = load_machine_pool()

        # Initialize RewardEvaluator
        logger.info("Initializing RewardEvaluator")
        evaluator = RewardEvaluator(
            task_config=task_config,
            settings_config=settings_yaml,
            machine_pool=machine_pool
        )

        # Run refinement loop
        logger.info("Starting refinement loop")
        refine_logs = {}
        iteration = int(refine_config.get('iteration'))
        for i in range(1, iteration+1):
            logger.info(f"Refinement iteration {i}/{iteration}")
            logger.info("Generating reward functions with LLM")
            # Generate reward functions using LLM agent
            reward_func_list = []
            raw_response_list = []

            for _ in tqdm(range(llm_agent.samples), desc="Generating reward functions"):
                reward_func, raw_response = llm_agent.func_gen(llm_agent.messages)
                reward_func_list.append(reward_func)
                raw_response_list.append(raw_response)

            logger.info("Evaluating reward functions")

            # Create a copy of task_yaml for this iteration with modified logs_path
            iter_task_yaml = task_yaml.copy()
            iter_task_yaml['logs_path'] = os.path.join(task_yaml.get('logs_path'), f'iter_{i}')
            
            logger.info(f"Evaluating {len(reward_func_list)} reward functions for iteration {i}")
            best_run, logs = evaluator.evaluate(reward_func_list, task_yaml=iter_task_yaml, log_name_template="run_{idx}")
            logger.info(f"Best run index: {best_run['idx']}")
            refine_logs[f'iteration_{i}/run'] = [best_run, logs, reward_func_list, raw_response_list]
            
            logger.info(f"Running {refine_config.get('num_eval')} evaluation runs with best reward function")
            best_eval, logs = evaluator.evaluate([reward_func_list[best_run["idx"]]]*refine_config.get('num_eval'), task_yaml=iter_task_yaml, log_name_template="eval_{idx}")
            logger.info(f"Evaluation complete for iteration {i}")
            refine_logs[f'iteration_{i}/eval'] = [best_eval, logs, reward_func_list, raw_response_list]
            
            if best_eval:
                logger.info(f"Max consecutive successes: {best_eval['max_con_successes']}")
                llm_agent.receive_feedback(refine_logs, iteration=i)
            else:
                logger.error("Evaluation failed, stopping refinement loop")
                break

if __name__ == "__main__":
    main()