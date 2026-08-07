"""Standalone test script for the VLM module.

This takes in an invidual video file from the ard-isaaclab-tasks repo and runs the VLM module on it. It is not intended to be run as part of the main eureka pipeline.
The VLM acts as a critic to the performance of the robot in the video and outputs a structure JSONL file containing natural language feedback.
This feedback is meant to be used as reward reflection for the next iteration of the Eureka pipeline, but in this script we are just analysing 
what it outputs. 
"""

import os
import argparse

# Argparse


# Get path to video 


# Input video to VLM module


# Get VLM output


# Save VLM output to JSONL file


