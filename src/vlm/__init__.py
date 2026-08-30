"""
VLM Evaluation Module for Critiquing Videos and Images
This module provides functionality to evaluate videos and images using a Visual Language Model (VLM).
Returns natural language feedback based on the the input video or images, which is then appended onto Reward Reflection
for the next iteration of the Eureka pipeline.

The VLM currently only supports 'hpc' execution backend. In which it submits a single `play.py` job to the CARES HPC Scheduler, which generates
videos of the robot performing the task. The VLM then critiques the video and returns natural language feedback.
This job is entirely separate from the training jobs that are run in the main Eureka pipeline, and is only used for evaluation purposes.

Main classes:
- VLM: Top level. Processes video/images and settings.
- VLMFeedbackAgent: Generates natural language feedback based on the video/images and task description.
"""

from .vlm_system import VLM
from .vlm_agent import VLMFeedbackAgent

__all__ = [
    "VLM",
    "VLMFeedbackAgent",
]

__version__ = "0.1.0"
