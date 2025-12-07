"""
Parallel execution module for distributed training.

This module handles:
- Distributing training tasks across a machine pool
- Managing parallel subprocess execution
- Process synchronization and result collection
- Error handling and retry logic
"""

import os
import subprocess
import logging
import threading
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Status of a training task."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class TrainingTask:
    """Represents a single training task."""
    idx: int
    reward_func: str
    machine: str
    machine_id: int
    log_name: str
    status: TaskStatus = TaskStatus.PENDING


@dataclass
class ProcessResult:
    """Result from a completed training process."""
    task: TrainingTask
    returncode: int
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    success: bool = False


class ParallelExecutor:
    """
    Manages parallel subprocess execution across a machine pool.

    Responsibilities:
    - Distributes tasks using round-robin scheduling
    - Spawns and monitors training subprocesses
    - Handles timeouts and retries
    - Collects results from completed processes

    Args:
        machine_pool: List of available machines (format: "user@host")
        docker_dir: Path to docker directory containing run scripts
        max_retries: Maximum number of retry attempts for failed tasks (default: 0)
        timeout: Timeout in seconds for each training process (default: 3600)

    Example:
        >>> executor = ParallelExecutor(["user@gpu1", "user@gpu2"], "/path/to/docker")
        >>> tasks = executor.create_tasks(reward_functions)
        >>> results = executor.execute_parallel(tasks, task_params)
    """

    def __init__(
        self,
        machine_pool: List[str],
        docker_dir: str,
        max_retries: int = 0,
        timeout: int = 3600
    ):
        if not machine_pool:
            raise ValueError("Machine pool cannot be empty")

        if not os.path.exists(docker_dir):
            raise ValueError(f"Docker directory does not exist: {docker_dir}")

        self.machine_pool = machine_pool
        self.docker_dir = docker_dir
        self.max_retries = max_retries
        self.timeout = timeout

        logger.info(f"ParallelExecutor initialized with {len(machine_pool)} machines")
        logger.debug(f"  Docker dir: {docker_dir}")
        logger.debug(f"  Timeout: {timeout}s")
        logger.debug(f"  Max retries: {max_retries}")

    def create_tasks(
        self,
        reward_funcs: List[str],
        log_name_template: str = "eval_{idx}"
    ) -> List[TrainingTask]:
        """
        Create training tasks with round-robin machine allocation.

        Args:
            reward_funcs: List of reward function code strings
            log_name_template: Template for log names (must contain {idx})

        Returns:
            List of TrainingTask objects
        """
        tasks = []

        for idx, reward_func in enumerate(reward_funcs):
            # Round-robin machine selection
            machine_idx = idx % len(self.machine_pool)
            machine = self.machine_pool[machine_idx]

            log_name = log_name_template.format(idx=idx)

            task = TrainingTask(
                idx=idx,
                reward_func=reward_func,
                machine=machine,
                machine_id=machine_idx,
                log_name=log_name
            )
            tasks.append(task)

            logger.debug(f"Task {idx}: machine[{machine_idx}] ({machine}) -> {log_name}")

        logger.info(f"Created {len(tasks)} tasks distributed across {len(self.machine_pool)} machines")
        return tasks

    def build_command(
        self,
        task: TrainingTask,
        task_params: Dict,
        log_name_template: str = "eval_{idx}"
    ) -> List[str]:
        """
        Build the command to execute for a training task.

        Args:
            task: TrainingTask object
            task_params: Dictionary containing task configuration parameters

        Returns:
            Command as list of strings
        """
        run_script = os.path.join(self.docker_dir, "run_remote_pipeline.sh")

        # Build command arguments matching run_remote_pipeline.sh parameters:
        # $1: ISAACLAB_TASK_NAME - The --task argument for your script
        # $2: LOCAL_WORKSPACE - Local workspace path
        # $3: TASK_FOLDER - The folder name of your task (the folder inside isaactasks/)
        # $4: LOGS_FOLDER_NAME - Path to logs folder on local machine
        # $5: TASK_TRAINING_CONFIG - Training config (can be empty string)
        # $6: REMOTE_TARGET - Remote target (user@host)
        cmd = [
            run_script,
            task_params.get('task_name'),
            task_params.get('local_workspace'),
            task_params.get('task_folder'),
            os.path.join(task_params.get('logs_folder'), log_name_template.format(idx=task.idx)),
            task_params.get('training_config', ''),
            task.machine
        ]

        return cmd

    def spawn_process(
        self,
        task: TrainingTask,
        task_params: Dict
    ) -> subprocess.Popen:
        """
        Spawn a subprocess for a training task.

        Args:
            task: TrainingTask object
            task_params: Dictionary containing task configuration

        Returns:
            subprocess.Popen object
        """
        cmd = self.build_command(task, task_params, log_name_template=task.log_name)

        logger.info(f"Starting training for task {task.idx} on machine[{task.machine_id}] ({task.machine})")
        logger.debug(f"  Command: {' '.join(cmd)}")

        # Create log file path for this task
        log_file_path = os.path.join(
            task_params.get('local_workspace'),
            task_params.get('logs_folder'),
            "command_outputs",
            f"{task.idx}_subprocess.log"
        )
        
        # Ensure log directory exists
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        
        # Open log file for writing
        log_file = open(log_file_path, 'w')
        
        logger.info(f"  Logging output to: {log_file_path}")
        
        proc = subprocess.Popen(
            cmd,
            cwd=self.docker_dir,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Store log file handle with process for cleanup
        proc._log_file = log_file
        
        # # Stream output in real-time
        # if proc.stdout:
        #     for line in proc.stdout:
        #         print(f"[Task {task.idx}] {line.rstrip()}")

        task.status = TaskStatus.RUNNING
        return proc

    def wait_for_process(
        self,
        proc: subprocess.Popen,
        task: TrainingTask
    ) -> ProcessResult:
        """
        Wait for a process to complete and collect results.

        Args:
            proc: subprocess.Popen object
            task: TrainingTask object

        Returns:
            ProcessResult object
        """
        try:
            stdout, stderr = proc.communicate(timeout=self.timeout)

            result = ProcessResult(
                task=task,
                returncode=proc.returncode,
                stdout=stdout,
                stderr=stderr,
                success=(proc.returncode == 0)
            )

            if result.success:
                task.status = TaskStatus.COMPLETED
                logger.info(f"Task {task.idx} completed successfully on machine[{task.machine_id}] ({task.machine})")
            else:
                task.status = TaskStatus.FAILED
                logger.error(f"Task {task.idx} failed on machine[{task.machine_id}] ({task.machine}) (exit code: {proc.returncode})")
                if stderr:
                    logger.error(f"  stderr: {stderr[:500]}")  # Log first 500 chars

            return result

        except subprocess.TimeoutExpired:
            logger.error(f"Task {task.idx} timed out after {self.timeout}s on machine[{task.machine_id}] ({task.machine})")
            proc.kill()
            proc.communicate()  # Clean up

            task.status = TaskStatus.TIMEOUT

            return ProcessResult(
                task=task,
                returncode=-1,
                stderr=f"Process timed out after {self.timeout} seconds",
                success=False
            )

        except Exception as e:
            logger.error(f"Error waiting for task {task.idx}: {e}")
            task.status = TaskStatus.FAILED

            return ProcessResult(
                task=task,
                returncode=-1,
                stderr=str(e),
                success=False
            )

    def execute_parallel(
        self,
        tasks: List[TrainingTask],
        task_params: Dict
    ) -> List[ProcessResult]:
        """
        Execute all tasks in parallel and collect results.

        Args:
            tasks: List of TrainingTask objects
            task_params: Dictionary containing task configuration

        Returns:
            List of ProcessResult objects
        """
        logger.info(f"Executing {len(tasks)} tasks in parallel")

        # Spawn all processes
        processes = []
        for task in tasks:
            proc = self.spawn_process(task, task_params)
            processes.append((proc, task))

        # Wait for all processes to complete
        logger.info("Waiting for all training processes to complete...")
        results = []

        for proc, task in processes:
            result = self.wait_for_process(proc, task)
            results.append(result)

        # Summary
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful

        logger.info(f"Parallel execution completed: {successful} successful, {failed} failed")

        return results

    def execute_with_retry(
        self,
        tasks: List[TrainingTask],
        task_params: Dict
    ) -> List[ProcessResult]:
        """
        Execute tasks with retry logic for failures.

        Args:
            tasks: List of TrainingTask objects
            task_params: Dictionary containing task configuration

        Returns:
            List of ProcessResult objects (final results after retries)
        """
        results = self.execute_parallel(tasks, task_params)

        if self.max_retries == 0:
            return results

        # Retry failed tasks
        for attempt in range(1, self.max_retries + 1):
            failed_tasks = [r.task for r in results if not r.success]

            if not failed_tasks:
                logger.info("All tasks successful, no retries needed")
                break

            logger.info(f"Retry attempt {attempt}/{self.max_retries} for {len(failed_tasks)} failed tasks")

            # Reset task status
            for task in failed_tasks:
                task.status = TaskStatus.PENDING

            # Re-execute failed tasks
            retry_results = self.execute_parallel(failed_tasks, task_params)

            # Update results with retry outcomes
            result_map = {r.task.idx: r for r in results}
            for retry_result in retry_results:
                result_map[retry_result.task.idx] = retry_result

            results = list(result_map.values())

            successful = sum(1 for r in results if r.success)
            logger.info(f"After retry {attempt}: {successful}/{len(results)} successful")

        return results

    def execute_sequential_per_machine(
        self,
        tasks: List[TrainingTask],
        task_params: Dict,
        workspace_prepare_func=None
    ) -> List[ProcessResult]:
        """
        Execute tasks sequentially per machine to prevent memory overflow.

        Tasks on the same machine instance run sequentially (one after another).
        Tasks on different machine instances run in parallel.

        Args:
            tasks: List of TrainingTask objects
            task_params: Dictionary containing task configuration
            workspace_prepare_func: Optional callback function to prepare workspace
                                   before each task. Should accept (task) and return bool.

        Returns:
            List of ProcessResult objects
        """
        # Group tasks by machine_id (not machine name, to handle duplicates)
        machine_tasks = defaultdict(list)
        for task in tasks:
            machine_tasks[task.machine_id].append(task)

        logger.info(f"Executing {len(tasks)} tasks sequentially per machine")
        for machine_id, machine_task_list in machine_tasks.items():
            machine_name = machine_task_list[0].machine
            logger.info(f"  Machine[{machine_id}] ({machine_name}): {len(machine_task_list)} tasks queued")

        # Thread-safe storage for results
        results_lock = threading.Lock()
        all_results = []

        def execute_machine_queue(machine_id: int, machine_task_list: List[TrainingTask]):
            """Execute all tasks for a specific machine instance sequentially."""
            machine_results = []
            machine_name = machine_task_list[0].machine

            logger.info(f"[Machine {machine_id} ({machine_name})] Starting sequential execution of {len(machine_task_list)} tasks")

            for task in machine_task_list:
                logger.info(f"[Machine {machine_id}] Processing task {task.idx} (queue position: {machine_task_list.index(task) + 1}/{len(machine_task_list)})")

                # Prepare workspace if callback provided
                if workspace_prepare_func:
                    logger.info(f"[Machine {machine_id}] Preparing workspace for task {task.idx}")
                    success = workspace_prepare_func(task)
                    if not success:
                        logger.error(f"[Machine {machine_id}] Failed to prepare workspace for task {task.idx}, skipping")
                        # Create a failed result
                        result = ProcessResult(
                            task=task,
                            returncode=-1,
                            stderr="Workspace preparation failed",
                            success=False
                        )
                        task.status = TaskStatus.FAILED
                        machine_results.append(result)
                        continue

                    # Sleep after workspace preparation to ensure files are ready before upload
                    logger.debug(f"[Machine {machine_id}] Waiting for workspace to stabilize before remote execution")
                    time.sleep(2.0)

                # Spawn and immediately wait for this task to complete
                proc = self.spawn_process(task, task_params)
                result = self.wait_for_process(proc, task)
                machine_results.append(result)

                logger.info(f"[Machine {machine_id}] Task {task.idx} {'completed successfully' if result.success else 'failed'}")

            logger.info(f"[Machine {machine_id}] Completed all {len(machine_task_list)} tasks")

            # Store results in thread-safe manner
            with results_lock:
                all_results.extend(machine_results)

        # Create and start a thread for each machine instance
        threads = []
        for machine_id, machine_task_list in machine_tasks.items():
            thread = threading.Thread(
                target=execute_machine_queue,
                args=(machine_id, machine_task_list),
                name=f"Machine-{machine_id}"
            )
            thread.start()
            threads.append(thread)
            time.sleep(30)  # Stagger thread starts slightly

        # Wait for all machine threads to complete
        logger.info("Waiting for all machines to complete their task queues...")
        for thread in threads:
            thread.join()

        # Summary
        successful = sum(1 for r in all_results if r.success)
        failed = len(all_results) - successful

        logger.info(f"Sequential execution completed: {successful} successful, {failed} failed")

        # Sort results by task index for consistent ordering
        all_results.sort(key=lambda r: r.task.idx)

        return all_results
