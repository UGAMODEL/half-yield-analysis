# HalfYield Runner

Analyze recorded pecan processing videos with a YOLO segmentation model.
Review detections in an operator dashboard and export time-binned CSV measurements.
The repository includes the model checkpoint, dashboard fonts, and automated tests.
Supply your own input videos.

## Setup

Use Python 3.11 or 3.12 with [uv](https://github.com/astral-sh/uv).
The dependency files currently specify Python `>=3.11, <=3.13`.
Run all commands from the repository root.

```bash
uv sync
uv run python desktop.py
```

The file-picker launcher requires Tk support in your Python installation.
If Tk is unavailable, use the command-line dashboard:

```bash
uv run python main.py --source /path/to/video.mp4 --show
```

The application selects CUDA, MPS, or CPU based on PyTorch availability.
GPU support depends on your installed PyTorch build and system drivers.
Use `--device cpu`, `--device cuda:0`, or `--device mps` to choose a device explicitly.

## Review a video

1. Start the launcher.
2. Select a video and a CSV report location.
3. Choose a rolling measurement window.
4. Select **Start review**.
5. Inspect playback with the dashboard controls.
6. Select **Finish / save CSV**.

See the [operator guide](docs/operator-guide.md) for report scope, calibration, and interface details.
The review window remains open at the end of the video until you finish or seek.

### Measurement

Half share equals detected half area divided by detected half and piece area.
Shell and obscured material do not contribute to this measurement.
The rolling measurement integrates areas over video time.
An empty reading means the application has insufficient detected area or time samples.

The fixed measurement region is 500 × 600 pixels, with its top-left corner at `(400, 300)`.
The model receives this crop resized to 640 × 640 pixels.
Check camera alignment before each review.
The region constants are in `main.py`.

Area share is not weight yield.
The separate mass estimate requires measured half and piece coefficients for your camera setup.
The default coefficients are placeholders.
See [calibration instructions](docs/operator-guide.md#calibration) before you use mass estimates.

### Report scope

Each seek, frame step, or calibration change clears the measurement history and starts a new report segment.
Resume playback to collect samples after inspection.
Reports require at least two playback samples and contain only the current segment.
The launcher suggests timestamped names and asks before replacing an existing report.

## Command-line use

Process a video without the dashboard:

```bash
uv run python main.py --source /path/to/video.mp4 --yield-output report.csv
```

Review a video and save annotated output:

```bash
uv run python main.py --source /path/to/video.mp4 \
  --show --save annotated.mp4 --yield-output report.csv \
  --half-conf 0.35 --piece-conf 0.30 --yield-bin-seconds 5
```

| Option | Purpose | Default |
| --- | --- | --- |
| `--source PATH` | Input video | Required |
| `--model PATH` | Model checkpoint | `halfyield-model.pt` |
| `--device DEVICE` | Inference device | Automatic |
| `--show` | Open the operator dashboard | Disabled |
| `--save PATH` | Save annotated video without dashboard controls | Disabled |
| `--conf N` | Base detection confidence | `0.30` |
| `--half-conf N`, `--piece-conf N` | Class confidence thresholds | Base confidence |
| `--obscured-conf N`, `--shell-conf N` | Other class confidence thresholds | Base confidence |
| `--ratio-window-sec N` | Rolling measurement window | `60` seconds |
| `--yield-output PATH` | CSV report path | `<video_stem>_yield_analysis.csv` in the working directory |
| `--yield-bin-seconds N` | Report bin duration | `10` seconds |
| `--no-yield-report` | Disable report generation | Reports enabled |

The dashboard always shows its trend.
Legacy chart options, including `--show-chart`, do not change dashboard layout.
Use the full help for parallel processing, GPU, and encoding options:

```bash
uv run python main.py --help
```

### Keyboard controls

| Key | Action |
| --- | --- |
| Space | Pause or resume |
| Left / Right | Seek during playback, or step frames while paused |
| `,` / `.` | Step backward or forward and pause |
| S | Toggle slow mode |
| Q / Escape | Finish review |

## Development

Run the automated tests:

```bash
uv run python -m unittest discover -s tests -v
```

The tests cover calibration, dashboard controls, playback transitions, CSV metadata, and measurement history.
Playback tests mock video capture and model inference and do not require a display or GPU.

`pyproject.toml` defines the uv dependencies.
Local environments, lockfiles, input videos, and generated CSV files are currently ignored by Git.

### Repository layout

| Path | Purpose |
| --- | --- |
| `main.py` | Inference, playback, video output, and CSV reports |
| `desktop.py` | File-picker launcher |
| `operator_dashboard.py` | Review dashboard and controls |
| `calibration.py` | Confidence and mass coefficient controls |
| `yield_history.py` | Rolling measurement history |
| `hmi_style.py` | Dashboard typography and drawing helpers |
| `assets/fonts/` | Bundled fonts and their licenses |
| `halfyield-model.pt` | Model checkpoint |
| `run_inference.sh` | Positional video argument wrapper for uv |
| `tune_multi_gpu.py` | Experimental tuner with a machine-specific input path |
| `tests/` | Automated tests |
| `docs/operator-guide.md` | Detailed operator instructions |

## Troubleshooting

- **Tk is unavailable:** Use `main.py --source /path/to/video.mp4 --show` from your environment.
- **CUDA is unavailable:** Check `nvidia-smi` and your PyTorch installation. Use `--device cpu` to run without a GPU.
- **MPS is unavailable:** Check your PyTorch build and macOS hardware support. Use `--device cpu` as a fallback.
- **OpenMP errors on macOS:** Check for conflicting OpenMP libraries in your Python environment.
- **No report after inspection:** Resume playback after seeking, stepping, or changing calibration to collect at least two samples.

## License

See [LICENSE](LICENSE) for the GNU Affero General Public License, version 3.
Bundled fonts have separate [SIL Open Font Licenses](assets/fonts/README.md).
