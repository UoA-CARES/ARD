# VLM Test Script
To run a one-shot VLM api call on a singular video. By default, the VLM will take in an mp4 file.

Response is generated in ~/runs/vlm_test_outputs/{date_time}_{video_name}.jsonl.
If `--as_images` is raised. The images will be stored in a folder under the same name

`--video_path`: Specifies the path to a chosen video {REQUIRED}
`--as_images`: Slice the video into images. This is because some VLM models do not take video (e.g mp4) input naitively. 
`--seed`: Specify a random seed to vary the response. Defaults to none.
`--frame_rate`: Specify a frame rate for image slicing. Capped to 15fps. Does nothing if specified without `as_images`. Defaults to 4 if not specified. 

```bash
# Example call for a sliced video with seed 2
python src/vlm/vlm_test.py --video_path {path/to/video_file} --as_images --seed 2
```