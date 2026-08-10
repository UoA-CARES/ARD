"""
VLM Agent for autonomous feedback of Eureka reward design.

The agent generates natural language feedback on task performance based on video input,
which is then used as reward reflection for the next iteration of the Eureka pipeline.
"""


import re
import os
import logging

from openai import OpenAI

class VLMFeedbackAgent:
    """
    VLM feedback generator.

    Args:
        task_description: Natural-language description of the task goal.
        env_source: Full task env-class source (VLM task context).
        agent_config: {model, base_url, sample, temperature?}.
        # TODO: Add other vars
    """

    def __init__(
        self,
        task_description: str,
        env_source: str,
        agent_config: dict,
        # TODO: Add other vars
    ):
        self.task_description = task_description
        self.env_source = env_source

        self.model = agent_config.get("model")
        self.base_url = agent_config.get("base_url")
        self.samples = int(agent_config.get("sample", 4))
        self.temperature = float(agent_config.get("temperature", 0.8))
        self.top_p = float(agent_config.get("top_p", 1.0))

        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OPENROUTER_API_KEY environment variable is not set."
            )