"""
Main evaluation orchestrator for reward function evaluation.

This module provides the high-level RewardEvaluator class that coordinates
the entire evaluation pipeline.
"""

import os
import logging
from typing import List, Dict, Optional

from .workspace_manager import WorkspaceManager
from .parallel_executor import ParallelExecutor
from .result_processor import ResultProcessor, EvaluationResult
from . import config
import time

logger = logging.getLogger(__name__)


class RewardEvaluator:
    """
    Orchestrates parallel evaluation of reward functions.

    This class coordinates the entire evaluation pipeline:
    1. Validates machine pool availability
    2. Prepares workspace with injected reward code for each evaluation
    3. Distributes training tasks across machine pool
    4. Executes training in parallel via remote pipeline
    5. Processes TensorBoard logs and collects results
    6. Selects best performing reward function

    Args:
        task_config: Dictionary containing task configuration
            Required keys:
            - workspace: Path to workspace directory
            - env_cfg_path: Path to environment configuration file
            - logs_path: Path to training logs directory
        settings_config: Dictionary containing global settings
            Required keys:
            - workspace: Path to global workspace
        machine_pool: List of available machines for training (format: "user@host")
        docker_dir: Path to docker directory (default: auto-detected from main.py location)
        max_retries: Maximum retry attempts for failed tasks (default: 0)
        timeout: Training timeout in seconds (default: 3600)

    Example:
        >>> evaluator = RewardEvaluator(
        ...     task_config={'workspace': '/path', 'env_cfg_path': '/path/env.py', 'logs_path': '/logs'},
        ...     settings_config={'workspace': '/path'},
        ...     machine_pool=['user@gpu1', 'user@gpu2']
        ... )
        >>> results = evaluator.evaluate(reward_function_list, logs=[])
        >>> print(f"Best result: {results['max_con_successes']} consecutive successes")
    """

    def __init__(
        self,
        task_config: Dict,
        settings_config: Dict,
        machine_pool: List[str],
        docker_dir: Optional[str] = None,
        max_retries: int = config.DEFAULT_MAX_RETRIES,
        timeout: int = config.DEFAULT_TRAINING_TIMEOUT
    ):
        # Validate inputs
        self._validate_config(task_config, settings_config)

        if not machine_pool:
            raise ValueError("Machine pool cannot be empty")

        self.task_config = task_config
        self.settings_config = settings_config
        self.machine_pool = machine_pool

        # Auto-detect docker_dir if not provided
        if docker_dir is None:
            # Assume docker/ is in the project root (same level as main.py)
            project_root = settings_config["workspace"]
            docker_dir = os.path.join(project_root, "docker")

        if not os.path.exists(docker_dir):
            raise ValueError(f"Docker directory not found: {docker_dir}")

        self.docker_dir = docker_dir

        # Initialize components
        self.workspace_manager = WorkspaceManager(
            workspace_path=settings_config.get('workspace'),
            env_cfg_path=task_config.get('env_cfg_path')
        )

        self.parallel_executor = ParallelExecutor(
            machine_pool=machine_pool,
            docker_dir=docker_dir,
            max_retries=max_retries,
            timeout=timeout
        )

        self.result_processor = ResultProcessor(task_config=task_config)

        logger.info(f"RewardEvaluator initialized with {len(machine_pool)} machines")

    def _validate_config(self, task_config: Dict, settings_config: Dict):
        """Validate that required configuration keys are present."""
        required_task_keys = ['workspace', 'env_cfg_path', 'logs_path']
        required_settings_keys = ['workspace']

        for key in required_task_keys:
            if key not in task_config:
                raise ValueError(f"task_config missing required key: '{key}'")

        for key in required_settings_keys:
            if key not in settings_config:
                raise ValueError(f"settings_config missing required key: '{key}'")

    def evaluate(
        self,
        reward_funcs: List[str],
        task_yaml: Optional[Dict] = None,
        log_name_template: str = "eval_{idx}"
    ) -> Optional[Dict]:
        """
        Evaluate multiple reward functions in parallel across machine pool.

        Args:
            reward_funcs: List of reward function code strings to evaluate
            logs: Optional list to append evaluation logs to
            task_yaml: Optional task configuration (if None, uses self.task_config)
            log_name_template: Template for naming log directories (default: "eval_{idx}")
        Returns:
            Dictionary with best evaluation result containing:
            - log_path: Path to the best training log
            - max_con_successes: Maximum consecutive successes achieved
            - tb_path: Path to TensorBoard logs
            - idx: Index of the best reward function
            Returns None if no successful evaluations
        """
        logs = []

        if not reward_funcs:
            logger.error("No reward functions provided for evaluation")
            return None, logs

        logger.info(f"Evaluating {len(reward_funcs)} reward functions across {len(self.machine_pool)} machines")

        # Validate workspace before starting
        if not self.workspace_manager.validate_workspace():
            logger.error("Workspace validation failed")
            return None, logs

        # Create training tasks
        tasks = self.parallel_executor.create_tasks(
            reward_funcs,
            log_name_template=log_name_template
        )

        # Prepare task parameters for remote execution
        # local_workspace
        task_params = self._build_task_params(task_yaml)

        # Execute all tasks in parallel (with workspace prep for each)
        process_results = []
        for task in tasks:
            # Prepare workspace with specific reward function
            logger.info(f"Preparing workspace for task {task.idx}")
            success = self.workspace_manager.prepare_for_evaluation(
                reward_func_code=task.reward_func,
                reset=True,
                pattern=config.REWARD_FUNCTION_PATTERN
            )

            if not success:
                logger.error(f"Failed to prepare workspace for task {task.idx}, skipping")
                continue

            # Spawn the training process
            proc = self.parallel_executor.spawn_process(task, task_params)
            process_results.append((proc, task))
            time.sleep(1)

        # Wait for all processes and collect results
        logger.info("Waiting for all training processes to complete...")
        evaluation_results = []

        for proc, task in process_results:
            process_result = self.parallel_executor.wait_for_process(proc, task)

            if process_result.success:
                # Process the training result
                eval_result = self.result_processor.process_training_result(
                    log_name=task.log_name,
                    idx=task.idx,
                    task_params=task_params,
                    machine=task.machine
                )

                if eval_result:
                    evaluation_results.append(eval_result)

                    # Append to logs if provided
                    logs.append({
                        'samples': task.idx,
                        'machine': task.machine,
                        'result': {
                            'log_path': eval_result.log_path,
                            'max_con_successes': eval_result.max_consecutive_successes,
                            'tb_path': eval_result.tb_path,
                            'idx': eval_result.idx
                        }
                    })
            else:
                logger.warning(f"Task {task.idx} did not complete successfully")

        # Select best result
        if not evaluation_results:
            logger.error("No successful evaluations")
            return None, logs

        best_result = ResultProcessor.select_best_result(
            evaluation_results,
            metric='max_consecutive_successes'
        )

        if best_result:
            result_dict = {
                'log_path': best_result.log_path,
                'max_con_successes': best_result.max_consecutive_successes,
                'tb_path': best_result.tb_path,
                'idx': best_result.idx
            }

            logger.info(
                f"Best evaluation: Task {best_result.idx} with "
                f"{best_result.max_consecutive_successes} consecutive successes"
            )

            return result_dict, logs

        return None, logs

    def _build_task_params(self, task_yaml: Optional[Dict] = None) -> Dict:
        """
        Build task parameters for remote execution.

        Args:
            task_yaml: Optional task configuration override

        Returns:
            Dictionary of task parameters
        """
        # Use provided task_yaml or fall back to self.task_config
        if task_yaml is None:
            # Try to reconstruct from task_config
            # This is a simplified version - may need adjustment based on actual usage
            task_yaml = {}

        params = {
            'task_name': task_yaml.get('task'),
            'task_folder': task_yaml.get('workspace'),
            'docker_name': task_yaml.get('docker_name', 'isaac'),
            'logs_folder': task_yaml.get('logs_path'),
            'training_config': task_yaml.get('config', ''),
            'local_workspace': self.settings_config.get('workspace'),
            'workspace_dir': os.path.expandvars(task_yaml.get('remote_workspace', '${HOME}/.temp_isaac')),
        }

        return params

    def get_machine_pool_status(self) -> Dict:
        """
        Get status information about the machine pool.

        Returns:
            Dictionary with machine pool statistics
        """
        return {
            'total_machines': len(self.machine_pool),
            'machines': self.machine_pool,
            'docker_dir': self.docker_dir,
            'timeout': self.parallel_executor.timeout,
            'max_retries': self.parallel_executor.max_retries
        }
