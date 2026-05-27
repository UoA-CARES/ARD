"""
File helpers for the refinement layer.

Reward extraction now lives in :mod:`src.evaluation.reward_injection`
(``extract_method_source``), which pulls the task's ``_get_rewards`` method via
the AST. This module just loads the prompt templates.
"""

import os


def load_prompts():
    """Load the LLM prompt templates from agent_config/."""
    config_dir = os.path.join(os.path.dirname(__file__), "agent_config")
    names = {
        "code_output_tip": "code_output_tip.txt",
        "initial_system": "initial_system.txt",
        "initial_user": "initial_user.txt",
        "code_feedback": "code_feedback.txt",
        "execution_error_feedback": "execution_error_feedback.txt",
        "policy_feedback": "policy_feedback.txt",
    }
    prompts = {}
    for key, fname in names.items():
        path = os.path.join(config_dir, fname)
        try:
            with open(path, "r") as f:
                prompts[key] = f.read()
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Prompt template not found: {path}") from e
    return prompts
