"""
VLM Agent for autonomous feedback of Eureka reward design.

The agent generates natural language feedback on task performance based on video input,
which is then used as reward reflection for the next iteration of the Eureka pipeline.
"""


import os
import logging
import base64

from openai import OpenAI

logger = logging.getLogger(__name__)

class VLMFeedbackAgent:
    """
    VLM feedback generator.

    Args:
        task_description: Natural-language description of the task goal.
        agent_config: {model, base_url, sample, temperature...}.
    """

    def __init__(
        self,
        task_description: str,
        agent_config: dict,
    ):
        self.model = agent_config.get("model")
        self.base_url = agent_config.get("base_url")
        self.temperature = float(agent_config.get("temperature", 0.8))
        self.max_output_tokens = int(agent_config.get("max_output_tokens", 600))
        self.timeout_seconds = int(agent_config.get("timeout_seconds", 90))
        self.task_description = task_description

        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            logger.warning("OPENROUTER_API_KEY is not set; LLM calls will fail.")
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

        self.raw_response = None  # Store the raw response for debugging

        self.sys_message = self._init_sys_message()

    def _init_sys_message(self):
        """
        Initialise system message for the VLM model based on the task description.
        """
        system_prompt_path = os.path.join(os.path.dirname(__file__), "vlm_critic.txt")
        with open(system_prompt_path, "r") as f:
            system_prompt = f.read()

        return [{"role": "system", "content": system_prompt}]

    def critique_video(self, video_path: str, seed: int = None) -> str:
        """
        Critiques a video and returns a natural language feedback.
        """
        video_content = self._build_video_content(video_path)
        messages = self._build_messages([video_content])
        logging.info("Video Critique requested. ")
        return self._call_vlm(messages, seed=seed)

    def critique_images(self, frame_paths: list[str], seed: int = None) -> str:
        """
        Critiques a sequence of images and returns a natural language feedback.
        """
        sequence_note = {"type": "text", "text": "The following frames are sequential frames from a video clip, in chronological order."}
        image_content = self._build_image_content(frame_paths)
        messages = self._build_messages([sequence_note] + image_content)
        logging.info("Image Critique requested. ")
        return self._call_vlm(messages, seed=seed)

    def _build_video_content(self, video_path: str) -> dict:
        """
        Read, encode, and prepare the video content for the VLM model.
        """
        # Assumes video path is valid
        with open(video_path, "rb") as f:
            encoded_video = base64.b64encode(f.read()).decode('utf-8')
        return {
            "type": "video_url",
            "video_url": {"url": f"data:video/mp4;base64,{encoded_video}"},
        }
        
    def _build_image_content(self, frame_paths: list[str]) -> list[dict]:
        """
        Read, encode, and prepare the image content for the VLM model.
        """

        content_list = []

        for frame_path in frame_paths:
            with open(frame_path, "rb") as f:
                encoded_image = base64.b64encode(f.read()).decode('utf-8')

            content_list.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{encoded_image}"},
            })
        return content_list

    def _build_messages(self, content_items: list[dict]) -> list[dict]:
        """
        Builds a list of messages for the VLM model based on the provided content items.
        """
        content = [{"type": "text", 
                    "text": f"Critique this task: {self.task_description}"}] + content_items

        return self.sys_message + [{"role": "user", "content": content}]


    def _call_vlm(self, messages: list[dict], seed: int = None) -> str:
        """
        VLM API call
        """
        max_retries = 10

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_output_tokens,
                    timeout=self.timeout_seconds,
                    seed=seed
                )
                feedback = response.choices[0].message.content
                if feedback is not None:
                    logger.info(f"VLM feedback received")
                    logger.info(f"Storing raw response to attribute 'raw_response' for debugging.")
                        
                    return feedback
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
        raise RuntimeError(
            f"Failed to get a valid response from the VLM model after {max_retries} attempts."
        )



    
    

    



        