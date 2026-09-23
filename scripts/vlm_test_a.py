'''
Mass testing script.
Calls all 600 VLM calls, gives output and graphs shit
'''

import os
import shutil
import sys
import logging
import cv2
import glob
import re
import json

from pydantic import BaseModel, Field, ConfigDict
from typing import List, Literal, Optional, Dict
from pathlib import Path

import yaml

##### Image Parameters #####
FRAME_CAP = 100 # Maximum number of frames to extract
SAMPLE_FPS = 8 # fps of Nvidia Cosmos Reason-1
############################

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Allow running this file directly via:
#   python src/vlm/vlm_test.py --video_path ...
# without setting PYTHONPATH.
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.vlm.vlm_agent import VLMFeedbackAgent

###############################################################
###############################################################
# Detected behaviour tags for VLM output
BehaviorTag = Literal[
    "contact loss",
    "jitter",
    "cube instability",
    "tossing",
    "static hold",
    "smooth rotation",
]

class Metadata(BaseModel):
    video_file_name: str = Field(description="Name of the video file.")
    camera_angle: Optional[str] = Field(default=None, description="Camera angle identifier.")
    fps: Optional[int] = Field(default=None, description="Frames per second used for slicing.")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility.")
    model_name: Optional[str] = Field(default=None, description="Name of the VLM model used for evaluation.")
    raw_feedback: Optional[str] = Field(default=None, description="Raw feedback from the VLM agent before extraction.")

class FeedbackSchema(BaseModel):
    model_config = ConfigDict(extra="forbid") 
    detected_behaviours: List[BehaviorTag] = Field(description="Observed behaviors.") 

class VLMRecord(BaseModel):
    metadata: Metadata = Field(description="Metadata about the test.")
    feedback: FeedbackSchema = Field(description="Feedback from the VLM agent.")
    visual_score: float = Field(ge=1.0, le=3.0, description="Visual score from 1 to 3")

RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "feedback_schema",
        "strict": True,
        "schema": FeedbackSchema.model_json_schema(),
    },
}

###############################################################
###############################################################
# Load CFG
def load_cfgs(is_vlm: bool = True):
    """
    Load configuration files for VLM testing.
    Returns:
        sys_cfg: System configuration dictionary.
        task_description: Task description string.
    """
    if is_vlm:
        sys_cfg_path = os.path.join(PROJECT_ROOT, "configs", "refineconfig.yaml")
        system_prompt_path = os.path.join(os.path.dirname(__file__), "test_a", "vlm_critic_test.txt")
    else:
        sys_cfg_path = os.path.join(os.path.dirname(__file__), "test_a", "llm_config.yaml")
        system_prompt_path = os.path.join(os.path.dirname(__file__), "test_a", "vlm_extractor.txt")

    # task desc always the same
    task_description_path = os.path.join(os.path.dirname(__file__), "test_a", "task_description_test.txt")

    try:
        with open(sys_cfg_path, "r") as f:
            sys_cfg = yaml.safe_load(f).get("vlm", {})
            logger.info(f"Loaded VLM configuration from {sys_cfg_path}")
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {sys_cfg_path}")
        raise
    try:
        with open(task_description_path, "r") as f:
            task_description = f.read()
            logger.info(f"Loaded task description from {task_description_path}")
    except FileNotFoundError:
        logger.error(f"Task description file not found: {task_description_path}")
        raise


    return sys_cfg, system_prompt_path, task_description
# Slicing
def slice_video_into_frames(video_path: str, output_dir: str, fps: int) -> list[dict]:
    """
    Slice a video into frames and save them as images.
    """
    # Make sure output dir exists
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Could not open video file: {video_path}")
        return []

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if original_fps == 0 or total_frames == 0:
        logger.warning(f"Video file appears empty or unreadable: {video_path}")
        cap.release()
        return []

    video_duration = total_frames / original_fps
    expected_frames = video_duration * fps

    if expected_frames > FRAME_CAP:
        sample_fps = fps * FRAME_CAP / expected_frames
    else:
        sample_fps = fps

    frame_skip_interval = max(1, int(original_fps / sample_fps))

    frame_counter = 0
    saved_frames = 0
    image_list = []

    while cap.isOpened() and saved_frames < FRAME_CAP:
        timestamp = frame_counter / original_fps
        ret, frame = cap.read()
        if not ret:
            break

        if frame_counter % frame_skip_interval == 0:
            framepath = os.path.join(output_dir, f"frame_{saved_frames:04d}.png")
            
            success = cv2.imwrite(framepath, frame)
            if success:
                image_list.append({"frame_path": framepath, "timestamp": timestamp})
                # FIX 2: Move log statement inside loop to log every saved frame
                logger.info(f"Saved frame {saved_frames} -> {framepath}")
            else:
                logger.error(f"Failed to write image frame to: {framepath}")
                
            saved_frames += 1

        frame_counter += 1

    cap.release()
    return image_list
def slice(video_dir: str):
    fps = [8, 15, 30]
    
    # Target folders explicitly matching 'checkpoint_' pattern
    checkpoint_folders = [
        f for f in os.listdir(video_dir) 
        if f.startswith("checkpoint_") and f.endswith("_videos")
    ]
    
    logger.info(f"Found {len(checkpoint_folders)} target checkpoint folders: {checkpoint_folders}")

    for video_folder in checkpoint_folders:
        folder_path = os.path.join(video_dir, video_folder)
        
        # Check if 'play' subfolder exists, fallback to main checkpoint directory if not
        play_dir = os.path.join(folder_path, "play")
        target_search_dir = play_dir if os.path.isdir(play_dir) else folder_path
        
        logger.info(f"Scanning for videos in: {target_search_dir}")

        for video_file in os.listdir(target_search_dir):
            video_file_path = os.path.join(target_search_dir, video_file)
            
            if not os.path.isfile(video_file_path) or not video_file.lower().endswith(('.mp4', '.avi', '.mov')):
                continue

            for fps_value in fps:
                output_file_dir = f"frames/fps_{fps_value}/frames_{video_file}"
                output_dir = os.path.join(folder_path, output_file_dir)
                
                logger.info(f"Processing '{video_file}' @ {fps_value} FPS -> {output_dir}")
                slice_video_into_frames(video_file_path, output_dir, fps_value)
# Recording
def save_record_to_json(record: VLMRecord, output_file: Path | str):
    """Appends a VLMRecord to a JSON array file safely."""
    data = []

    # Read existing records if file exists and is non-empty
    if output_file.exists() and output_file.stat().st_size > 0:
        try:
            with open(output_file, "r") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            logger.warning(
                f"Could not parse {output_file}. Starting fresh list."
            )

    # Convert Pydantic record to dict (Pydantic V2 model_dump)
    data.append(record.model_dump(mode="json"))

    # Write updated list back to file
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)
def discover_eval_targets(video_dir: str, fps: int = 8):
    """Discovers checkpoint and camera directories matching the requested FPS."""
    base_path = Path(video_dir)

    checkpoint_folders = sorted(
        base_path.glob("checkpoint_*_videos"),
        key=lambda p: int(m.group(1))
        if (m := re.search(r"checkpoint_(\d+)", p.name))
        else 999999,
    )

    for checkpoint_dir in checkpoint_folders:
        # Match any folder inside frames/fps_{fps}/ matching frames_rl-video-step-0-camera_*
        fps_dir = checkpoint_dir / "frames" / f"fps_{fps}"
        if not fps_dir.exists():
            continue

        # Look for matching directories (including those ending in .mp4)
        camera_dirs = sorted([d for d in fps_dir.glob("frames_*") if d.is_dir()])

        for angle_dir in camera_dirs:
            frame_files = sorted([str(p) for p in angle_dir.glob("*.png")])
            if frame_files:
                yield checkpoint_dir, angle_dir, frame_files
def run_eval(
    vlm_agent: VLMFeedbackAgent,
    llm_agent: VLMFeedbackAgent,
    frame_files: list[str],
    checkpoint_dir: Path,
    angle_dir: Path,
    output_file: Path,
    fps: int,
    num_seeds: int = 10,
    model_name: Optional[str] = None
):
    """Runs seed evaluations for a single frame sequence and appends each to disk."""
    cam_match = re.search(r"(camera_\d+)", angle_dir.name)
    camera_angle = cam_match.group(1) if cam_match else angle_dir.name
    
    # Strip 'frames_' prefix and ensure clean .mp4 filename for metadata
    raw_video_name = angle_dir.name.replace("frames_", "")
    if not raw_video_name.endswith(".mp4"):
        raw_video_name += ".mp4"
        
    video_file_name = f"{checkpoint_dir.name}/{raw_video_name}"

    # Format frame strings into dicts with frame_path AND calculated timestamp
    formatted_frames = [
        {
            "frame_path": path,
            "timestamp": round(idx / fps, 4)
        }
        for idx, path in enumerate(frame_files)
    ]

    for seed in range(num_seeds):
        # 1. Get feedback from VLM agent
        raw_feedback = vlm_agent.critique_images(formatted_frames, seed=seed)

        # 2. Extract the feedback from the VLM agent into structured JSON using the LLM agent
        messages = [
            llm_agent.sys_message[0],
            {"role": "user", "content": llm_agent.task_description},
            {"role": "user", "content": raw_feedback["reasoning"]}
        ]

        logger.info("Calling LLM")
        json_response_str: str = llm_agent._call_vlm(messages, seed=seed, response_format=RESPONSE_FORMAT)

        try:
            # 2. Parse & validate JSON output
            feedback: FeedbackSchema = FeedbackSchema.model_validate_json(
                json_response_str
            )

            # 3. Construct Metadata
            meta = Metadata(
                video_file_name=video_file_name,
                camera_angle=camera_angle,
                fps=fps,
                seed=seed,
                model_name=model_name,
                raw_feedback=raw_feedback["reasoning"]
            )

            # 4. Save Record
            record = VLMRecord(metadata=meta, feedback=feedback, visual_score=raw_feedback["score"])
            save_record_to_json(record, output_file)
            logger.info(
                f"Saved record for {checkpoint_dir.name} | {camera_angle} | FPS={fps} | Seed={seed}"
            )

        except Exception as e:
            logger.error(
                f"Failed parsing/validating for {checkpoint_dir.name}/{camera_angle} (Seed {seed}): {e}"
            )
            logger.debug(f"Raw Output: {json_response_str}")
# Testing
def run_angle_experiment(
    video_dir: str,
    vlm_agent: VLMFeedbackAgent,
    llm_agent: VLMFeedbackAgent,
    fixed_fps: int = 8,
    num_seeds: int = 10,
    output_json_path: Path | str = "angle_experiment_results.json",
):
    output_file = Path(output_json_path)

    for chk_dir, angle_dir, frame_files in discover_eval_targets(
        video_dir, fps=fixed_fps
    ):
        logger.info(
            f"[ANGLE EXP] Processing {chk_dir.name} | {angle_dir.name} across {num_seeds} seeds..."
        )
        run_eval(
            vlm_agent=vlm_agent,
            llm_agent=llm_agent,
            frame_files=frame_files,
            checkpoint_dir=chk_dir,
            angle_dir=angle_dir,
            output_file=output_file,
            fps=fixed_fps,
            num_seeds=num_seeds,
            model_name=vlm_agent.model
        )
def run_fps_experiment(
    video_dir: str,
    vlm_agent: VLMFeedbackAgent,
    llm_agent: VLMFeedbackAgent,
    fps_list: list[int] = [8, 15, 30],
    fixed_camera_angle: str = "camera_0",
    num_seeds: int = 10,
    output_json_path: Path | str = "fps_experiment_results.json",
):
    output_file = Path(output_json_path)

    for fps in fps_list:
        for chk_dir, angle_dir, frame_files in discover_eval_targets(
            video_dir, fps=fps
        ):
            # Lock to single fixed angle for fair FPS comparison
            if fixed_camera_angle not in angle_dir.name:
                continue

            logger.info(
                f"[FPS EXP] Processing FPS={fps} | {chk_dir.name} across {num_seeds} seeds..."
            )
            run_eval(
                vlm_agent=vlm_agent,
                llm_agent=llm_agent,
                frame_files=frame_files,
                checkpoint_dir=chk_dir,
                angle_dir=angle_dir,
                output_file=output_file,
                fps=fps,
                num_seeds=num_seeds,
                model_name=vlm_agent.model
            )
def run_model_experiment(
    video_dir: str,
    vlm_agents: Dict[str, VLMFeedbackAgent],  # {"gemini-3.6": agent1, "qwen3-30b": agent2}
    llm_agent: VLMFeedbackAgent,
    fixed_fps: int = 8,
    fixed_camera_angle: str = "camera_0",
    num_seeds: int = 10,
    output_json_path: Path | str = "angle_experiment_results.json",
):
    output_file = Path(output_json_path)

    # Pre-discover frame directories matching the fixed conditions
    targets = [
        (chk, angle, frames)
        for chk, angle, frames in discover_eval_targets(
            video_dir, fps=fixed_fps
        )
        if fixed_camera_angle in angle.name
    ]

    for model_name, vlm_agent in vlm_agents.items():
        for chk_dir, angle_dir, frame_files in targets:
            logger.info(
                f"[MODEL EXP] Model={model_name} | {chk_dir.name} across {num_seeds} seeds..."
            )
            run_eval(
                vlm_agent=vlm_agent,
                llm_agent=llm_agent,
                frame_files=frame_files,
                checkpoint_dir=chk_dir,
                angle_dir=angle_dir,
                output_file=output_file,
                fps=fixed_fps,
                num_seeds=num_seeds,
                model_name=model_name,
            )
# Calculating results and graphing


def main():
    #### SLICING PHASE
    # slice(base_video_path) DONE

    # Paths and settings
    # Base path where videos live
    base_video_path = Path("/home/andrew/Desktop/results/test-A/videos")

    # Dynamic output directory: one level up from base_video_path -> /home/andrew/Desktop/results/test-A/results
    output_json_dir = base_video_path.parent / "results"
    output_json_dir.mkdir(parents=True, exist_ok=True)

    # Instantiate usual VLM feedback agent
    sys_cfg, system_prompt_path, task_description = load_cfgs(is_vlm=True)
    vlm = VLMFeedbackAgent(task_description, sys_cfg, system_prompt_path=system_prompt_path)

    # Intantiate LLM for extracting structured feedback from VLM output
    sys_cfg_llm, system_prompt_path_llm, task_description = load_cfgs(is_vlm=False)
    llm = VLMFeedbackAgent(task_description, sys_cfg_llm, system_prompt_path=system_prompt_path_llm)

    # Angle test
    run_angle_experiment(
        video_dir=base_video_path,
        vlm_agent=vlm,
        llm_agent=llm,
        fixed_fps=8,
        num_seeds=10,
        output_json_path=output_json_dir / "angle_test_results.json",
    )

    # FPS test
    run_fps_experiment(
        video_dir=base_video_path,
        vlm_agent=vlm,
        llm_agent=llm,
        fps_list=[8, 15, 30],
        fixed_camera_angle="camera_0",
        num_seeds=10,
        output_json_path=output_json_dir / "fps_test_results.json",
    )

    # Model test

    sys_cfg_qwen = sys_cfg.copy()
    sys_cfg_qwen["model"] = "qwen/qwen3-vl-30b-a3b-instruct"
    sys_cfg_gemini_36 = sys_cfg.copy()  
    sys_cfg_gemini_36["model"] = "google/gemini-3.6-flash"
    sys_cfg_gemini_31 = sys_cfg.copy()
    sys_cfg_gemini_31["model"] = "google/gemini-3.1-pro-preview"

    vlm_agents = {
        "qwen/qwen3-vl-30b-a3b-instruct": VLMFeedbackAgent(task_description, sys_cfg_qwen, system_prompt_path=system_prompt_path),
        "google/gemini-3.6-flash": VLMFeedbackAgent(task_description, sys_cfg_gemini_36, system_prompt_path=system_prompt_path),
        "google/gemini-3.1-pro-preview": VLMFeedbackAgent(task_description, sys_cfg_gemini_31, system_prompt_path=system_prompt_path),
    }

    run_model_experiment(
        video_dir=base_video_path,
        vlm_agents=vlm_agents,
        llm_agent=llm,
        fixed_fps=8,
        fixed_camera_angle="camera_0",
        num_seeds=10,
        output_json_path=output_json_dir / "model_test_results.json",
    )


if __name__ == "__main__":
    main()

