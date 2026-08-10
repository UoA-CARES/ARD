"""Standalone test script for the VLM module.

This takes in an invidual video file from the ard-isaaclab-tasks repo and runs the VLM module on it. It is not intended to be run as part of the main eureka pipeline.
The VLM acts as a critic to the performance of the robot in the video and outputs a structure JSONL file containing natural language feedback.
This feedback is meant to be used as reward reflection for the next iteration of the Eureka pipeline, but in this script we are just analysing 
what it outputs. 
"""

import os
import argparse
import logging
import yaml


logging.basicConfig(                                                        # -- from main.py
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

def load_yaml_config(config_path):                                          # -- from main.py
    """Safely load a YAML configuration file."""
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration: {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"Config file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing {config_path}: {e}")
        raise





# Get default prompt path
def get_default_prompt_path() -> str:
    """Get the default prompt path for the VLM module."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(
        os.path.join(script_dir, "..", "vlm", "vlm_critic.txt")
    )

# check file path
def check_file_path(file_path: str) -> bool:
    """Check if the given file path exists."""
    if not os.path.isfile(file_path):
        logger.error(f"File not found: {file_path}")
        return False
    logger.info(f"File exists: {file_path}")
    return True

# Input video to VLM module


# Get VLM output


# Save VLM output to JSONL file

def main():
    # video path for copy paste
    # /home/andrew/ard-isaaclab-tasks/logs/rl_games/shadow_hand/2026-07-28_13-56-38/videos/play/rl-video-step-0-camera_0.mp4


    parser = argparse.ArgumentParser(prog="vlm_test", description="Standalone test script for the VLM module.")
    parser.add_argument("--video_path", type=str, required=True, default="path/to/default/video.mp4",
        help="Path to the input video file.")
    parser.add_argument("--prompt_path", type=str, default=get_default_prompt_path(),
        help="Path to the prompt file. Defaults to ")
    
    args = parser.parse_args()

    if check_file_path(args.video_path):
        logger.info(f"Video file found")
    if not args.prompt_path or not check_file_path(args.prompt_path):
        args.prompt_path = get_default_prompt_path()  # get default
        logger.info(f"Using prompt file: {args.prompt_path}")
    with open(args.prompt_path, "r") as f:
        prompt_content = f.read()
        # print(f"Prompt content:\n{prompt_content}")
        


    refine_cfg = load_yaml_config("configs/refineconfig.yaml")
    print(f"refine_cfg: {refine_cfg['vlm']['model']}")
    


if __name__ == "__main__":
    main()