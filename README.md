# HalfYield Runner

YOLOv12n-seg inference pipeline for pecan half/piece classification.  
Includes:

- Fixed ROI crop (500×600 at (x=400, y=300)) → resized to 640×640 for model input  
- Segmentation contours mapped back to full-frame  
- Per-class confidence thresholds (`half`, `obscured`, `piece`, `shell`)  
- Rolling, time-integrated half/piece ratio over a sliding window  
- Interactive video controls (seek, pause, slow mode)  
- Optional integrated ratio chart overlay  

---

## Requirements

- Python 3.11–3.13  
- GPU with CUDA 12.4 support (for acceleration), or CPU fallback  

All dependencies are declared in `pyproject.toml` and pinned in `uv.lock`.

---

## Installation

[uv](https://github.com/astral-sh/uv) is the recommended package manager.  
Install uv with one line:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then, from this project directory:

```bash
uv sync
```

This will create and populate a virtual environment using `pyproject.toml` + `uv.lock`.

---

## Running Inference

Basic usage:

```bash
uv run main.py --source video.mp4 --model halfyield-model.pt --show --show-chart
```

### Key Options

- `--source PATH` (required): input video file  
- `--model PATH`: YOLOv12n-seg model checkpoint (default: `halfyield-model.pt`)  
- `--device`: device spec, e.g. `cuda:0` or `cpu`  
- `--save PATH`: optional output video path to write results  
- `--show`: display a live OpenCV window  
- `--show-chart`: overlay rolling integrated ratio chart (requires `--show`)  

### Interactive Controls (when `--show` is enabled)

- **SPACE** — pause/play  
- **← / →** — skip seconds while playing; prev/next frame while paused  
- **, / .** — prev/next frame (forces paused)  
- **S** — toggle slow mode (200 ms delay)  
- **Q / ESC** — quit  

---

## Example

Run on sample video, saving annotated output:

```bash
uv run main.py \\
  --source video.mp4 \\
  --model halfyield-model.pt \\
  --save annotated.mp4 \\
  --show \\
  --show-chart \\
  --half-conf 0.35 --piece-conf 0.30
```

---

## Project Layout

```
halfyield-runner/
├── main.py             # Inference + visualization script
├── halfyield-model.pt  # YOLOv12n-seg model weights
├── video.mp4           # Example video input
├── pyproject.toml      # Project metadata + dependencies
├── uv.lock             # Frozen dependency lockfile
└── README.md           # This file
```
