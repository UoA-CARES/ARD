import os
import re
# import glob
# from pathlib import Path
# import numpy as np
# from tensorboard.backend.event_processing import event_accumulator

# from src.evaluation.result_processor import (
#     read_tb,
#     get_latest_checkpoint_dir,
#     summarize_tensorboard
# )
# from src.evaluation.workspace_manager import write_code_to_file


def load_prompts():
    # Reading configuration files from agent_config directory
    config_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "refinement", "agent_config")
    code_output_tip_path = os.path.join(config_dir, "code_output_tip.txt")
    initial_system_path = os.path.join(config_dir, "initial_system.txt")
    initial_user_path = os.path.join(config_dir, "initial_user.txt")
    # feedback
    code_feedback_path = os.path.join(config_dir, "code_feedback.txt")
    execution_error_feedback_path = os.path.join(config_dir, "execution_error_feedback.txt")
    policy_feedback_path = os.path.join(config_dir, "policy_feedback.txt")

    # Read the configuration files
    try:
        with open(code_output_tip_path, 'r') as f:
            code_output_tip = f.read()
        with open(initial_system_path, 'r') as f:
            initial_system = f.read()
        with open(initial_user_path, 'r') as f:
            initial_user = f.read()
        with open(code_feedback_path, 'r') as f:
            code_feedback = f.read()
        with open(execution_error_feedback_path, 'r') as f:
            execution_error_feedback = f.read()
        with open(policy_feedback_path, 'r') as f:
            policy_feedback = f.read()
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Configuration file not found: {e}")
    return {
        "code_output_tip": code_output_tip,
        "initial_system": initial_system,
        "initial_user": initial_user,
        "code_feedback": code_feedback,
        "execution_error_feedback": execution_error_feedback,
        "policy_feedback": policy_feedback,
    }


def load_env_cfg(env_cfg_path):
    # read the env config from env_cfg_path which is a python scripts containing useful infos for the isaaclab simultions, 
    # and copy observation (_get_observations) and related codes as a strings
    
    # Read the environment config file
    try:
        with open(env_cfg_path, 'r') as f:
            env_config_content = f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Environment config file not found: {env_cfg_path}")

    # Extract reward calculation code
    reward_code = ""
    compute_rewards_pattern = re.compile(
        r'@torch\.jit\.script\s*\n*def\s+compute_rewards\s*\([^)]*\).*?return\s+total_reward, reward_components', re.DOTALL
    )
    match = compute_rewards_pattern.search(env_config_content)
    if match:
        reward_code = match.group(0)
    else:
        raise ValueError("Could not find the compute_rewards function in the environment config file.")

    # Combine all extracted information
    env_cfg_dict = {
        "env_code": env_config_content,
        "reward_code": reward_code,
    }
    return env_cfg_dict