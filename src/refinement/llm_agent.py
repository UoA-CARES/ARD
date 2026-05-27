"""
LLM Agent for autonomous reward design (Eureka-style).

The agent proposes complete ``_get_rewards(self)`` methods for an ard-isaaclab-tasks
environment, then iterates on them using training feedback (per-component scalar
trends + the fixed ``fitness_function`` evaluation metric). The proposed method is
spliced into the task env via AST and dispatched for training by the evaluator.
"""

import re
import os
import logging

from openai import OpenAI

from src.refinement.files_operation import load_prompts

logger = logging.getLogger(__name__)

# A valid proposal must define a `_get_rewards(self ...)` method.
GET_REWARDS_RE = re.compile(r"def\s+_get_rewards\s*\(\s*self\b", re.DOTALL)

# Code-block extraction patterns, most-specific first.
_CODE_PATTERNS = [
    r"```python(.*?)```",
    r"```(.*?)```",
]


class EurekaAgent:
    """
    LLM reward designer.

    Args:
        task_description: Natural-language description of the task goal.
        reward_template: Pristine ``_get_rewards`` source (shown as inspiration).
        env_source: Full task env-class source (LLM task context).
        agent_config: {model, base_url, sample, temperature?}.
    """

    def __init__(
        self,
        task_description: str,
        reward_template: str,
        env_source: str,
        agent_config: dict,
    ):
        self.prompts = load_prompts()
        self.task_description = task_description
        self.reward_template = reward_template
        self.env_source = env_source

        self.model = agent_config.get("model")
        self.base_url = agent_config.get("base_url")
        self.samples = int(agent_config.get("sample", 4))
        self.temperature = float(agent_config.get("temperature", 0.8))
        self.top_p = float(agent_config.get("top_p", 1.0))

        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            logger.warning("OPENROUTER_API_KEY is not set; LLM calls will fail.")
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

        self.code_output_tip = self.prompts["code_output_tip"]

        self.messages = self._init_messages()

    def _init_messages(self):
        system_content = (
            self.prompts["initial_system"].format(
                task_reward_template=self.reward_template
            )
            + self.code_output_tip
        )
        user_content = self.prompts["initial_user"].format(
            task_obs_code_string=self.env_source,
            task_description=self.task_description,
        )
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

    def receive_feedback(self, best_response_text: str, summary_path: str = None) -> str:
        """
        Fold the previous iteration's outcome into the conversation.

        Args:
            best_response_text: Raw LLM response that produced the best run.
            summary_path: Path to that run's training_summary.txt, or None if the
                iteration failed entirely (signals a hard reset).

        Returns:
            The exact feedback message text appended to the conversation (so the
            caller can record what was sent back to the LLM).
        """
        if summary_path and os.path.exists(summary_path):
            with open(summary_path, "r") as f:
                summary = f.read()
            feedback_content = (
                self.prompts["policy_feedback"]
                + "\n"
                + summary
                + "\n"
                + self.prompts["code_feedback"]
            )
        else:
            feedback_content = self.prompts["execution_error_feedback"].format(
                traceback_msg="No reward function trained successfully this "
                "iteration. Rewrite an entirely new reward function."
            )
        feedback_content += self.code_output_tip

        assistant_msg = {"role": "assistant", "content": best_response_text}
        user_msg = {"role": "user", "content": feedback_content}
        if len(self.messages) == 2:
            self.messages += [assistant_msg, user_msg]
        else:
            # Keep the window to system + initial-user + last assistant/user pair.
            self.messages[-2] = assistant_msg
            self.messages[-1] = user_msg
        return feedback_content

    def func_gen(self, messages, seed=None):
        """
        Query the LLM and return (method_source, raw_response).

        ``seed`` varies the sampler per candidate so identical prompts no longer
        collapse to identical completions (and stays reproducible). Retries until a
        response contains a valid ``_get_rewards`` method.
        """
        max_retries = 10
        for attempt in range(1, max_retries + 1):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    seed=seed,
                    n=1,
                    messages=messages,
                )
                response = completion.choices[0].message.content
                code = self._extract_method(response)
                if code is not None:
                    return code, response
                logger.warning(f"Attempt {attempt}: no valid _get_rewards in response")
            except Exception as e:  # noqa: BLE001 - surface API/transport errors and retry
                logger.warning(f"Attempt {attempt} failed: {e}")
        raise RuntimeError(
            "Failed to generate a valid _get_rewards method after 10 attempts."
        )

    @staticmethod
    def _extract_method(response: str):
        """Pull the first fenced code block that defines a _get_rewards method."""
        for pattern in _CODE_PATTERNS:
            for match in re.findall(pattern, response, re.DOTALL):
                block = match.strip()
                if GET_REWARDS_RE.search(block):
                    return block
        # Fall back to the raw response if it itself is a bare method.
        if GET_REWARDS_RE.search(response):
            return response.strip()
        return None
