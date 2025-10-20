#!/bin/bash
# Inference runner script for pixi
# Usage: pixi run infer <video_path> [additional_args]

export KMP_DUPLICATE_LIB_OK=TRUE

if [ $# -eq 0 ]; then
    echo "Usage: pixi run infer <video_path> [additional_args]"
    echo "Example: pixi run infer latest.mp4 --show --show-chart"
    exit 1
fi

VIDEO_PATH="$1"
shift  # Remove first argument (video path)

# Run with all arguments
python main.py --source "$VIDEO_PATH" --model halfyield-model.pt "$@"