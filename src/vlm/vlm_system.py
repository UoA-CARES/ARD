"""
Top level VLM class that handles all preprocessing, paths and arguments for the VLM module.
Gathers all necessary information and passes it to the VLMFeedbackAgent for processing.
Stores images (if any) and gets feedback from the VLMFeedbackAgent.

"""

import os
import logging
import shutil
import cv2

from ..evaluation.config import VLM_FEEDBACK_FILE
from .vlm_agent import VLMFeedbackAgent

logger = logging.getLogger(__name__)


class VLM:
    """ 
    Top level VLM class that handles all preprocessing, paths and arguments for the VLM module.
    Must be called with task_description and agent_config already loaded and processed.

    Args:
        video_path: Path to the input video file.
        task_description: The main task description.
        task_specific_information: Any task-specific information.
        agent_config: {model, base_url, sample, temperature...}.
        seed: Optional seed for reproducibility.
    """

    def __init__(
        self,
        video_path: str,
        task_description: str,
        task_specific_information: str,
        agent_config: dict,
        seed: int = None,
    ):
        self.agent_config = agent_config

        # Validate video path
        if not os.path.isfile(video_path) or not video_path.endswith(".mp4"):
            raise FileNotFoundError(f"Video file not found or invalid format: {video_path}")
        self.video_path = video_path
        self.as_images = agent_config.get("as_images", False)
        self.seed = seed
        self.sample_rate = agent_config.get("sample_rate", 4)  # Default frame rate for slicing is 4 fps
        self.FRAME_CAP = agent_config.get("frame_cap", 1000)  # Get frame cap from config, default to 1000 if not provided

        # Build the task description with any task-specific information
        self.task_description = f"{task_description}\n{task_specific_information}"

    def send_input_to_vlm(self):
        """
        Send the video or sliced frames to the VLMFeedbackAgent for critique and get feedback.
        """
        vlm_agent = VLMFeedbackAgent(self.task_description, self.agent_config)
        if not self.as_images:
            # Critique the video directly
            feedback = vlm_agent.critique_video(self.video_path, seed=self.seed)
        else:
            # Slice the video into frames and critique the frames
            frame_list = self._slice_video_into_frames(self.video_path, self.sample_rate)
            feedback = vlm_agent.critique_images(frame_list, seed=self.seed)

        return feedback

    def save_vlm_feedback(self, feedback: str, output_dir: str):
        """
        Saves the VLM feedback to a text file in "training_record" subdir where "training_summary.txt" is located.

        Args:
            feedback: The feedback string to save.
            output_dir: Path to "training_record" subdir. Will be called via `winner.summary_path` from `main.py`.
        """
        with open(os.path.join(output_dir, VLM_FEEDBACK_FILE), "w") as f:
            f.write(feedback)

    def _slice_video_into_frames(self, video_path: str, fps: int) -> list[str]:
        """
        Slice a video into frames and save them as images.

        Args:
            video_path: Path to the input video file. Sliced frames will be saved in a subdirectory named 'frames' in the same directory as the video.
            fps: Frames per second to extract (optional)."""

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        parent_dir = os.path.dirname(video_path)
        output_dir = os.path.join(parent_dir, "frames")

        # wipe the dir and make a new one
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        # Get the original FPS of the video
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = total_frames / original_fps

        # Calc expected frames
        expected_frames = video_duration * fps

        if expected_frames > self.FRAME_CAP:
            sample_fps = fps * self.FRAME_CAP / expected_frames
        else:
            sample_fps = fps

        # Calc frame skip interval
        frame_skip_interval = int(original_fps / sample_fps)
        frame_skip_interval = max(1, frame_skip_interval)  # Ensure at least 1

        frame_counter = 0
        saved_frames = 0
        image_list = []

        # Slice video into frames and save them
        while cap.isOpened() and saved_frames < self.FRAME_CAP:
            timestamp = frame_counter / original_fps
            ret, frame = cap.read()
            if not ret:                 # End of video
                break

            # Match frame skip interval to sample fps
            if frame_counter % frame_skip_interval == 0:
                framepath = os.path.join(output_dir, f"frame_{saved_frames:04d}.png")

                cv2.imwrite(framepath, frame)
                image_list.append({"frame_path": framepath, "timestamp": timestamp})
                saved_frames += 1

            frame_counter += 1

        # Close video
        cap.release()

        # Return the list of saved frame paths
        return image_list