# HalfYield Runner

Analyze recorded pecan processing videos with a YOLO segmentation model.
Review detections in an operator dashboard and export time-binned CSV measurements.
The repository includes the model checkpoint, dashboard fonts, and automated tests.
Supply your own input videos.

## Install and start

Use Python 3.11 or 3.12 with [uv](https://github.com/astral-sh/uv).
Install the app directly from this Git repository:

```bash
uv tool install --python 3.12 git+https://github.com/UGAMODEL/half-yield-analysis.git
halfyield
```

The command opens a local browser interface. It prints the URL if the browser does not open automatically.
The installed package includes the model, fonts, and HTML/CSS/JS assets.
You can run it from any working directory. Tk is not required.

For local development:

```bash
uv sync
uv run halfyield
```

Start a review directly:

```bash
halfyield --source /path/to/video.mp4 --show
```

The application selects CUDA, MPS, or CPU based on PyTorch availability.
GPU support depends on your installed PyTorch build and system drivers.
Use `--device cpu`, `--device cuda:0`, or `--device mps` to choose a device explicitly.

## Review a video

1. Run `halfyield`.
2. Choose a video file or enter its local path. Enter an optional CSV report path.
3. Choose a rolling measurement window.
4. Select **Start review**.
5. Inspect playback with the dashboard controls.
6. Select **Finish / save CSV**.

See the [operator guide](docs/operator-guide.md) for report scope, calibration, and interface details.
The review remains open at the end of the video until you finish or seek.
Close the browser tab or window to finish the review and save its current report segment.
A browser disconnect also stops the runner after 20 seconds without a heartbeat.
Reloading the page ends the current session. Start `halfyield` again for another review.

### Measurement

Half share equals detected half area divided by detected half and piece area.
Shell and obscured material do not contribute to this measurement.
The rolling measurement integrates areas over video time.
An empty reading means the application has insufficient detected area or time samples.

The fixed measurement region is 500 × 600 pixels, with its top-left corner at `(400, 300)`.
The model receives this crop resized to 640 × 640 pixels.
Check camera alignment before each review.
The region constants are in `halfyield/main.py`.

Area share is not weight yield.
The separate mass estimate requires measured half and piece coefficients for your camera setup.
The default coefficients are placeholders.
See [calibration instructions](docs/operator-guide.md#calibration) before you use mass estimates.

### Report scope

Each seek, frame step, or calibration change clears the measurement history and starts a new report segment.
Resume playback to collect samples after inspection.
Reports require at least two playback samples and contain only the current segment.
The browser uses a timestamped report name by default and rejects existing report paths.
Reports default to the working directory. Choosing a file uploads a temporary copy to the local runner.
Enter a local video path to avoid copying a large file.

## Command-line use

Process a video without the dashboard:

```bash
halfyield --source /path/to/video.mp4 --yield-output report.csv
```

Review a video and save annotated output:

```bash
halfyield --source /path/to/video.mp4 \
  --show --save annotated.mp4 --yield-output report.csv \
  --half-conf 0.35 --piece-conf 0.30 --yield-bin-seconds 5
```

| Option | Purpose | Default |
| --- | --- | --- |
| `--source PATH` | Input video | Required |
| `--model PATH` | Model checkpoint | Bundled checkpoint |
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

The dashboard shows its trend by default. Use **View settings** to hide it or change control spacing.
Legacy chart options, including `--show-chart`, do not change dashboard layout.
Use the full help for parallel processing, GPU, and encoding options:

```bash
halfyield --help
```

### Keyboard controls

| Key | Action |
| --- | --- |
| Space | Pause or resume |
| Left / Right | Seek during playback, or step frames while paused |
| `,` / `.` | Step backward or forward and pause |
| S | Toggle slow mode |
| Q / Escape | Finish review |

## Browser and runner

The Python runner serves the interface on a random localhost port.
HTML, CSS, JavaScript, and fonts load from the installed package.
The browser renders controls, measurement values, calibration inputs, and an SVG trend chart.
The runner decodes video, runs inference, and sends annotated JPEG frames and JSON measurements.
The interface does not require an external web service or a JavaScript build tool.

Closing the tab sends a finish request. A heartbeat timeout handles missing close events.
The runner saves the current report segment before it shuts down.

## Development

Run the automated tests:

```bash
uv run python -m unittest discover -s tests -v
```

The tests cover calibration, dashboard controls, playback transitions, CSV metadata, and measurement history.
Playback tests mock video capture and model inference and do not require a display or GPU.

`pyproject.toml` defines the uv dependencies.
Local environments, the uv lockfile, input videos, and generated CSV files are ignored by Git.

### Repository layout

| Path | Purpose |
| --- | --- |
| `halfyield/main.py` | Inference, playback, video output, and CSV reports |
| `halfyield/cli.py` | Installed command entry point |
| `halfyield/web_review.py`, `halfyield/web_server.py` | Local HTTP server and runner controls |
| `halfyield/web/` | Browser interface and client-side chart |
| `halfyield/operator_dashboard.py` | Review dashboard and controls |
| `halfyield/calibration.py` | Confidence and mass coefficient controls |
| `halfyield/yield_history.py` | Rolling measurement history |
| `halfyield/hmi_style.py` | Dashboard typography and drawing helpers |
| `halfyield/assets/fonts/` | Bundled fonts and their licenses |
| `halfyield/halfyield-model.pt` | Model checkpoint |
| `run_inference.sh` | Positional video argument wrapper for uv |
| `tune_multi_gpu.py` | Experimental tuner with a machine-specific input path |
| `tests/` | Automated tests |
| `docs/operator-guide.md` | Detailed operator instructions |

## Troubleshooting

- **Browser does not open:** Open the localhost URL printed by the runner.
- **CUDA is unavailable:** Check `nvidia-smi` and your PyTorch installation. Use `--device cpu` to run without a GPU.
- **MPS is unavailable:** Check your PyTorch build and macOS hardware support. Use `--device cpu` as a fallback.
- **OpenMP errors on macOS:** Check for conflicting OpenMP libraries in your Python environment.
- **No report after inspection:** Resume playback after seeking, stepping, or changing calibration to collect at least two samples.

## License

See [LICENSE](LICENSE) for the GNU Affero General Public License, version 3.
Bundled fonts have separate [SIL Open Font Licenses](halfyield/assets/fonts/README.md).
