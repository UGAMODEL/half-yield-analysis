# HalfYield Runner

## Operator workflow

Start the file-picker launcher:

```bash
python desktop.py
```

1. Select a recorded processing video.
2. Choose a CSV report location.
3. Select a rolling measurement window.
4. Select **Start review**.
5. Use **Pause**, **Back**, **Ahead**, or the frame controls to inspect the video.
6. Select **Finish / save CSV** to close the review and save the report.

The launcher requires Python with Tk support. For Homebrew Python 3.11, install `python-tk@3.11`.
If Tk is unavailable, start the same dashboard from the command line:

```bash
python main.py --source latest.mp4 --show
```

The dashboard shows half share, piece share, a trend, and video position.
The video remains visible at the end until you finish or seek to another position.
The dashboard always shows the trend. The legacy chart overlay flags apply to neither its size nor its position.
Saved annotated videos do not include dashboard controls.

**Measurement:** Half share equals detected half area divided by detected half and piece area.
Shell and obscured material do not contribute to this measurement. This measurement is not weight yield.
An empty reading means that the application has insufficient detected area or time samples.
Check camera alignment before each review. The measurement region is fixed at the coordinates described below.

**Report scope:** Each seek or frame step clears the measurement history and starts a new report segment.
Resume playback after inspection to collect samples. The report requires at least two playback samples.
The CSV includes only the segment after the most recent seek or step.
The launcher suggests a timestamped report name and asks before it replaces an existing report.
Command-line report naming retains its existing behavior.


### Visual style

The interface follows the supplied UGA Visual Style guide's palette and typography.
Bulldog Red (`#BA0C2F`), Arch Black, and Chapel Bell White establish the brand.
Stegeman neutral tints support control surfaces. Olympic (`#004E60`) identifies measurements and active controls.
White text appears on red backgrounds. Red text appears on white backgrounds.
The interface does not pair red text with black backgrounds or black text with red backgrounds.

The dashboard bundles Oswald at Medium weight for headings and readings, and Merriweather Sans for control labels and supporting text.
Font licenses are in `assets/fonts`. Text masks are cached to limit rendering work during playback.
The optional Tk launcher uses installed Merriweather Sans, with the approved Georgia font as its fallback.
The launcher uses Georgia for its title.


### Calibration

Select **Calibration** in the review window to show six sliders.
Drag a slider and release it to apply the value. Each change starts a new report segment.
Pause playback to compare detections on the same frame. Resume playback to collect new report samples.

- **Half, piece, obscured, and shell confidence:** Minimum accepted detection confidence for each class, from 0% to 100%.
- **Half mass / area:** Approximate grams per model-space square pixel for halves.
- **Piece mass / area:** Approximate grams per model-space square pixel for pieces.

The mass sliders use a logarithmic scale from `0.000001` to `1` gram per model-space square pixel.
Both default to `0.001`. These values are placeholders, not measured pecan coefficients.
Estimate each coefficient separately: divide a weighed sample's mass by its detected area under the same camera setup.
Use model-space mask area at the 640-pixel model input size, not original video pixels.
Use distinct material samples. Do not count the same material across repeated video frames.

Estimated half mass share equals `half_area * half_coefficient / (half_area * half_coefficient + piece_area * piece_coefficient)`.
The dashboard retains the area share and shows the estimated half mass share separately.
This estimate is not a direct weight measurement or a throughput measurement.
The CSV records all confidence thresholds, both coefficients, and estimated mass shares for the segment and each time bin.
Calibration values apply to the current review. Use command-line options to reuse precise values in another review:

```bash
python main.py --source latest.mp4 --show --half-conf 0.35 --piece-conf 0.30 \
  --half-mass-per-area 0.0012 --piece-mass-per-area 0.0008
```


YOLOv12n-seg inference pipeline for pecan half/piece classification.  
Includes:

- Fixed ROI crop (500×600 at (x=400, y=300)) → resized to 640×640 for model input  
- Segmentation contours mapped back to full-frame  
- Per-class confidence thresholds (`half`, `obscured`, `piece`, `shell`)  
- Rolling, time-integrated half/piece ratio over a sliding window  
- Interactive video controls (seek, pause, slow mode)  
- Optional integrated ratio chart overlay  
- **Automatic hardware acceleration detection (CUDA/MPS/CPU)**

---

## Hardware Acceleration Support

The project automatically detects and uses the best available hardware acceleration:

- **NVIDIA GPUs**: CUDA acceleration (Linux/Windows)
- **Apple Silicon**: Metal Performance Shaders (MPS) acceleration (macOS)
- **CPU fallback**: Works on all systems without dedicated GPU

No manual configuration required - the optimal device is selected automatically!

---

## Requirements

- Python 3.11–3.13  
- For GPU acceleration:
  - **Linux/Windows**: NVIDIA GPU with CUDA 12.4+ support
  - **macOS**: Apple Silicon (M1/M2/M3) for Metal acceleration

---

## Installation

### Option 1: Using Pixi (Recommended)

[Pixi](https://pixi.sh) provides cross-platform package management with automatic hardware-specific dependencies:

```bash
# Install pixi
curl -fsSL https://pixi.sh/install.sh | bash

# Install platform-appropriate dependencies automatically
pixi install

# Run the application with flexible arguments
pixi run infer video.mp4 --show --show-chart

# Or use predefined tasks
pixi run run-latest   # For latest.mp4
pixi run demo         # For demo with video.mp4
```

Pixi automatically installs:
- **Linux/Windows**: PyTorch with CUDA support + nvidia-ml-py
- **macOS**: PyTorch with Metal Performance Shaders support

**Note for macOS users**: If you encounter OpenMP warnings, pixi tasks automatically set the `KMP_DUPLICATE_LIB_OK=TRUE` environment variable to resolve this common PyTorch issue.

### Option 2: Using uv

[uv](https://github.com/astral-sh/uv) is an alternative package manager:

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Run the application (set OpenMP variable on macOS if needed)
export KMP_DUPLICATE_LIB_OK=TRUE  # macOS only
uv run main.py --source video.mp4 --model halfyield-model.pt --show --show-chart
```

---

## Device Testing

Test your hardware acceleration setup:

```bash
# With pixi
pixi run test-device

# With uv (macOS: set OpenMP variable if needed)
export KMP_DUPLICATE_LIB_OK=TRUE  # macOS only
uv run test_device.py
```

This will show:
- Available acceleration options (CUDA/MPS/CPU)
- Recommended device for your system
- Performance verification

---

## Running Inference

Basic usage:

```bash
# With pixi - flexible inference with custom arguments
pixi run infer latest.mp4 --show --show-chart

# Or enter pixi shell for full control
pixi shell
python main.py --source video.mp4 --model halfyield-model.pt --show --show-chart
exit

# Or use predefined tasks
pixi run run-latest  # Runs with latest.mp4
pixi run demo        # Runs with video.mp4

# With uv  
uv run main.py --source video.mp4 --model halfyield-model.pt --show --show-chart
```

### Key Options

- `--source PATH` (required): input video file  
- `--model PATH`: YOLOv12n-seg model checkpoint (default: `halfyield-model.pt`)  
- `--device`: device spec (e.g., `cuda:0`, `mps`, `cpu`). **Auto-detected if not specified**
- `--save PATH`: optional output video path to write results  
- `--show`: display a live OpenCV window  
- `--show-chart`: overlay rolling integrated ratio chart (requires `--show`)

### Yield Analysis Options

- `--yield-report`: generate yield analysis report on exit (default: enabled)
- `--no-yield-report`: disable yield analysis report
- `--yield-output PATH`: output path for yield analysis CSV (default: auto-generated)
- `--yield-bin-seconds N`: time bin duration in seconds for yield analysis (default: 10.0)

### Device Selection

The application automatically selects the optimal device:
1. **CUDA GPU** (if available on Linux/Windows)
2. **Apple Metal (MPS)** (if available on macOS with Apple Silicon)  
3. **CPU** (fallback)

To manually override device selection:
```bash
# Force CPU usage
pixi run infer video.mp4 --device cpu --show

# Force specific CUDA device  
pixi run infer video.mp4 --device cuda:1 --show

# Force MPS on macOS
pixi run infer video.mp4 --device mps --show
```  

### Interactive Controls (when `--show` is enabled)

- **SPACE** — pause/play  
- **← / →** — skip seconds while playing; prev/next frame while paused  
- **, / .** — prev/next frame (forces paused)  
- **S** — toggle slow mode (200 ms delay)  
- **Q / ESC** — quit  

---

## Example

Run on sample video, saving annotated output with yield analysis:

```bash
# Using pixi
pixi run infer video.mp4 \
  --save annotated.mp4 \
  --show \
  --show-chart \
  --half-conf 0.35 --piece-conf 0.30 \
  --yield-bin-seconds 5.0

# Using uv
uv run main.py \
  --source video.mp4 \
  --model halfyield-model.pt \
  --save annotated.mp4 \
  --show \
  --show-chart \
  --half-conf 0.35 --piece-conf 0.30 \
  --yield-output custom_yield_report.csv
```

### Yield Analysis Output

When processing completes, the program automatically generates:

1. **Console Output**: Summary statistics and time-series data
2. **CSV Report**: Detailed analysis with binned yield data over time

Example output:
```
============================================================
HALF YIELD ANALYSIS REPORT  
============================================================
Video Duration: 120.50 seconds
Total Half Area (integrated): 15420.30
Total Piece Area (integrated): 8967.45  
Total Area (half + piece): 24387.75
NET AVERAGE YIELD: 0.6323 (63.23%)
============================================================

Yield over time (10-second bins):
Time (s)    Yield    Half Area   Piece Area
--------------------------------------------
     0.0   0.6850      1285.40       589.20
    10.0   0.6205      1456.80       890.15
    20.0   0.6441      1388.92       773.28
    ...

Detailed analysis saved to: video_yield_analysis.csv
============================================================
```

---

## Performance Tips

### Hardware Acceleration Benefits
- **CUDA (NVIDIA GPU)**: 5-10x faster inference compared to CPU
- **MPS (Apple Silicon)**: 2-4x faster inference on M1/M2/M3 Macs
- **CPU**: Reliable fallback, recommended for development/testing

### Optimal Settings by Hardware
```bash
# High-end NVIDIA GPU (RTX 3080+)
--device cuda:0 --conf 0.25 --half-conf 0.30

# Apple Silicon Mac (M1/M2/M3)  
--device mps --conf 0.30 --half-conf 0.35

# CPU (any system)
--device cpu --conf 0.35 --half-conf 0.40
```

### Memory Considerations
- **GPU**: Ensure sufficient VRAM (2GB+ recommended)
- **System RAM**: 8GB+ recommended for video processing
- **Storage**: SSD recommended for large video files

---

## Troubleshooting

### Device Issues
1. **CUDA not detected on Linux/Windows**:
   ```bash
   # Check NVIDIA driver installation
   nvidia-smi
   
   # Verify PyTorch CUDA installation
   pixi run python -c "import torch; print(torch.cuda.is_available())"
   ```

2. **MPS not available on macOS**:
   - Ensure you have Apple Silicon (M1/M2/M3) Mac
   - Update to macOS 12.3+ and latest PyTorch
   ```bash
   pixi run python -c "import torch; print(torch.backends.mps.is_available())"
   ```

3. **OpenMP warnings on macOS**:
   - Common with PyTorch installations
   - Pixi tasks automatically set `KMP_DUPLICATE_LIB_OK=TRUE`
   - For manual runs: `export KMP_DUPLICATE_LIB_OK=TRUE`

4. **Slow performance**:
   - Run `pixi run test-device` to verify acceleration
   - Reduce video resolution or confidence thresholds
   - Use `--device cpu` if GPU memory is insufficient

### Installation Issues
- **Pixi not found**: Ensure pixi is installed and in PATH
- **Platform detection**: Pixi should automatically detect your platform
- **Missing dependencies**: Try `pixi install --locked` to use exact versions

---

## Project Layout

```
halfyield-runner/
├── main.py             # Inference + visualization script  
├── halfyield-model.pt  # YOLOv12n-seg model weights
├── test_device.py      # Hardware acceleration testing
├── test_yield.py       # Yield analysis testing
├── run_inference.sh    # Pixi inference runner script
├── pyproject.toml      # Python project configuration
├── pixi.toml           # Pixi project configuration  
├── video.mp4           # Example video input
└── README.md           # This file
```

