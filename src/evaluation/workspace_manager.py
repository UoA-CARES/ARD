"""
Workspace management module for evaluation.

This module handles workspace preparation including:
- Git checkout operations to reset workspace state
- Reward function code injection into environment files
- Workspace cleanup and validation
"""

import os
import re
import subprocess
import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)


class WorkspaceManager:
    """
    Manages workspace preparation for evaluation runs.

    Responsibilities:
    - Reset workspace to clean state via git checkout
    - Inject reward function code into environment configuration files
    - Validate workspace state before training

    Args:
        workspace_path: Path to the workspace directory
        env_cfg_path: Path to the environment configuration file

    Example:
        >>> manager = WorkspaceManager("/path/to/workspace", "/path/to/env_cfg.py")
        >>> manager.prepare_for_evaluation(reward_function_code)
    """

    # Default regex pattern for matching reward functions
    DEFAULT_REWARD_PATTERN = r'@torch\.jit\.script\s*\n*def\s+compute_rewards\s*\([^)]*\).*?return\s+total_reward, reward_components'

    def __init__(self, workspace_path: str, env_cfg_path: str):
        self.workspace_path = workspace_path
        self.env_cfg_path = env_cfg_path

        if not os.path.exists(workspace_path):
            raise ValueError(f"Workspace path does not exist: {workspace_path}")

        if not os.path.exists(env_cfg_path):
            raise ValueError(f"Environment config file does not exist: {env_cfg_path}")

    def reset_workspace(self) -> bool:
        """
        Reset workspace to clean state using git checkout.

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Resetting workspace: {self.workspace_path}")
            result = subprocess.run(
                ["git", "checkout", "."],
                cwd=self.workspace_path,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                logger.debug("Workspace reset successful")
                # Sleep to ensure file system settles after git checkout
                time.sleep(0.2)
                return True
            else:
                logger.error(f"Git checkout failed: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            logger.error("Git checkout timed out after 30 seconds")
            return False
        except Exception as e:
            logger.error(f"Error resetting workspace: {e}")
            return False

    def inject_reward_function(
        self,
        reward_func_code: str,
        pattern: Optional[str] = None
    ) -> bool:
        """
        Inject reward function code into the environment configuration file.

        Args:
            reward_func_code: The reward function code to inject
            pattern: Optional regex pattern to match the function to replace.
                    If None, uses DEFAULT_REWARD_PATTERN

        Returns:
            True if successful, False otherwise
        """
        if pattern is None:
            pattern = self.DEFAULT_REWARD_PATTERN

        try:
            logger.info(f"Injecting reward function into: {self.env_cfg_path}")

            # Read the target script file
            with open(self.env_cfg_path, 'r') as f:
                script_content = f.read()

            # Compile the regex pattern
            regex = re.compile(pattern, re.DOTALL)

            # Check if pattern matches
            if not regex.search(script_content):
                logger.error(
                    f"Could not find target function in {self.env_cfg_path} "
                    f"using pattern: {pattern[:50]}..."
                )
                return False

            # Replace the matched function with new code
            new_script_content = regex.sub(reward_func_code, script_content, count=1)

            # Verify that replacement actually changed something
            if new_script_content == script_content:
                logger.warning("Reward function injection resulted in no changes")
                return False

            # Write the modified content back
            with open(self.env_cfg_path, 'w') as f:
                f.write(new_script_content)

            logger.debug(f"Successfully injected reward function ({len(reward_func_code)} chars)")
            return True

        except FileNotFoundError:
            logger.error(f"Environment config file not found: {self.env_cfg_path}")
            return False
        except Exception as e:
            logger.error(f"Error injecting reward function: {e}")
            return False

    def prepare_for_evaluation(
        self,
        reward_func_code: str,
        reset: bool = True,
        pattern: Optional[str] = None
    ) -> bool:
        """
        Prepare workspace for evaluation run.

        This is a convenience method that combines reset and injection.

        Args:
            reward_func_code: The reward function code to inject
            reset: Whether to reset workspace before injection (default: True)
            pattern: Optional regex pattern for function matching

        Returns:
            True if preparation successful, False otherwise
        """
        if reset:
            if not self.reset_workspace():
                logger.error("Failed to reset workspace")
                return False

        if not self.inject_reward_function(reward_func_code, pattern):
            logger.error("Failed to inject reward function")
            return False

        logger.info("Workspace prepared successfully")
        return True

    def validate_workspace(self) -> bool:
        """
        Validate that workspace is in a good state for evaluation.

        Checks:
        - Workspace directory exists
        - Environment config file exists
        - Workspace is a git repository

        Returns:
            True if valid, False otherwise
        """
        checks = []

        # Check workspace exists
        if os.path.exists(self.workspace_path):
            checks.append(("Workspace exists", True))
        else:
            checks.append(("Workspace exists", False))

        # Check env config exists
        if os.path.exists(self.env_cfg_path):
            checks.append(("Env config exists", True))
        else:
            checks.append(("Env config exists", False))

        # Check if git repo
        git_dir = os.path.join(self.workspace_path, ".git")
        if os.path.exists(git_dir):
            checks.append(("Is git repository", True))
        else:
            checks.append(("Is git repository", False))

        # Log results
        all_passed = all(result for _, result in checks)
        for check_name, result in checks:
            status = "✓" if result else "✗"
            logger.debug(f"  {status} {check_name}")

        if all_passed:
            logger.info("Workspace validation passed")
        else:
            logger.error("Workspace validation failed")

        return all_passed


# Maintain backward compatibility - standalone function
def write_code_to_file(func_strings: str, target_script: str, rules: str):
    """
    Legacy function for writing code to file.

    DEPRECATED: Use WorkspaceManager.inject_reward_function() instead.
    """
    # Extract workspace from target_script path
    workspace = os.path.dirname(os.path.dirname(target_script))
    manager = WorkspaceManager(workspace, target_script)
    return manager.inject_reward_function(func_strings, pattern=rules)
