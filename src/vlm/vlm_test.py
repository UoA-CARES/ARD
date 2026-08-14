"""
Standalone test script for the VLM module.

This takes in an individual video file from the ard-isaaclab-tasks repo and runs
the VLM module on it. It is not intended to be run as part of the main Eureka
pipeline.

The VLM acts as a critic to the performance of the robot in the video and outputs
a structured JSONL file containing natural language feedback.

This feedback is meant to be used as reward reflection for the next iteration of
the Eureka pipeline, but in this script we are just analyzing what it outputs.
"""

import os
import shutil
import sys
import argparse
import logging
import yaml
import cv2

from pathlib import Path
import json

# Allow running this file directly via:
#   python src/vlm/vlm_test.py --video_path ...
# without setting PYTHONPATH.
SRC_ROOT = str(Path(__file__).resolve().parents[1])
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from vlm.vlm_agent import VLMFeedbackAgent

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# For initial call 
# 1. Parse cli arguments (--as_images, --video_path)
# 2. Load config from refineconfig.yamlm pass "vlm" section to VLMFeedbackAgent
# 3. Retrieve task description from hard coded txt file


# For video processing
# 1. Retrieve video path from cli args
# if --as_images is raised
# 1a. Slice video into frames
# else
# 2. Construct VLMFeedbackAgent with task description and config
# 3. Pass video/frames to VLMFeedbackAgent

# For slicing video into frames
# 1. If there are already images from a previous run, clear them out
# 2. Use OpenCV to read video and slice into frames
# 3. Save frames to dedicated folder (need to save so we can look at them later)

# After getting feedback (hardcoded path for now)
# 1. define jsonl file structure
# 1a. if save_raw_response is true, save raw response to jsonl file
# 2. write feedback to jsonl file

def slice_video_into_frames(video_path: str, output_dir: str, fps: int)-> list[str]:
    """
    Slice a video into frames and save them as images.

    Args:
        video_path: Path to the input video file.
        output_dir: Directory where the frames will be saved.
        fps: Frames per second to extract (optional)."""

    FRAME_CAP = 30 # Maximum number of frames to extract

    # wipe the dir and make a new one
    path_to_output_dir = os.path.join(SRC_ROOT, "..", "runs", "vlm_test_outputs", output_dir)
    if os.path.exists(path_to_output_dir):
        shutil.rmtree(path_to_output_dir)
    os.makedirs(path_to_output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    # Get the original FPS of the video
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / original_fps

    # Calc expected frames
    expected_frames = video_duration * fps

    if expected_frames > FRAME_CAP:
        sample_fps = fps * FRAME_CAP / expected_frames
    else:
        sample_fps = fps

    # Calc frame skip interval
    frame_skip_interval = int(original_fps / sample_fps)
    frame_skip_interval = max(1, frame_skip_interval)  # Ensure at least 1

    frame_counter = 0
    saved_frames = 0

    # Slice video into frames and save them
    while cap.isOpened() and saved_frames < FRAME_CAP:
        ret, frame = cap.read()
        if not ret:                 # End of video
            break

        # Match frame skip interval to sample fps
        if frame_counter % frame_skip_interval == 0:
            frame_filename = os.path.join(path_to_output_dir, f"frame_{saved_frames:04d}.png")
            cv2.imwrite(frame_filename, frame)
            saved_frames += 1

        frame_counter += 1

    # Close video
    cap.release()

    # Return the list of saved frame paths
    image_list = []
    for filename in sorted(os.listdir(path_to_output_dir)):
        if filename.endswith(".png"):
            image_list.append(os.path.join(path_to_output_dir, filename))
    return image_list    

def main():

    # For easy copy paste
    # /home/andrew/ard-isaaclab-tasks/logs/rl_games/shadow_hand/2026-07-28_13-56-38/videos/play/rl-video-step-0-camera_0.mp4

    parser = argparse.ArgumentParser(description="VLM Test Script")
    parser.add_argument(
        "--video_path",
        type=str,
        required=True,
        help="Path to the video file to be critiqued.",
    )
    parser.add_argument(
        "--as_images",
        action="store_true",
        help="If set, the video will be sliced into frames and critiqued as images.",
    )
    args = parser.parse_args()

    # Load configuration
    sys_cfg_path = os.path.join(SRC_ROOT, "..", "configs", "refineconfig.yaml")
    try:
        with open(sys_cfg_path, "r") as f:
            sys_cfg = (yaml.safe_load(f))["vlm"]
            logger.info(f"Loaded VLM configuration from {sys_cfg_path}")
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {sys_cfg_path}")
        raise

    # Hardcoded task descriptions for testing
    task_path = os.path.join(os.path.dirname(__file__), "task_description.txt")
    try: 
        with open(task_path, "r") as f:
            task_description = f.read()
    except FileNotFoundError:
        logger.error(f"Task description file not found: {task_path}")
        raise

    # Get the date-time of the video
    grandparent_path = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(args.video_path)))
    )
    grandparent_name = os.path.basename(grandparent_path)

    # Get feedback based on input video or sliced frames
    if not args.as_images:
        # Critique the video directly
        vlm_agent = VLMFeedbackAgent(task_description, sys_cfg)
        feedback = vlm_agent.critique_video(args.video_path)
    else:
        # slice the video
        SAMPLE_FPS = 4                      # fps of Nvidia Cosmos Reason-1
        image_output_dir = f"{grandparent_name}__{os.path.basename(args.video_path)}_output_frames"
        image_list = slice_video_into_frames(args.video_path, image_output_dir, SAMPLE_FPS)
        # Critique the sliced frames
        vlm_agent = VLMFeedbackAgent(task_description, sys_cfg)
        feedback = vlm_agent.critique_images(image_list)

    # Save feedback to JSONL file
    jsonl_record = {
        "video_name": os.path.basename(args.video_path),
        "model:": sys_cfg.get("model"),
        "task_description": task_description,
        "date_time": grandparent_name,  
        "vlm_feedback": feedback,
        "human_feedback": None,  # Placeholder for human feedback
    }
    if sys_cfg.get("save_raw_response", False) and hasattr(vlm_agent, "raw_response"):          # TO FIX
        jsonl_record["raw_response"] = vlm_agent.raw_response
    if sys_cfg.get("jsonl_enabled", True):
        jsonl_path = os.path.join(SRC_ROOT, "..", "runs", "vlm_test_outputs", f"{grandparent_name}_{os.path.basename(args.video_path)}.jsonl")
        os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
        with open(jsonl_path, "a", encoding="utf-8") as f:
            json.dump(jsonl_record, f, indent=1)
            # f.write("\n")
    else:
        logger.info("JSONL logging is disabled; feedback not saved to file.")

if __name__ == "__main__":
    main()
