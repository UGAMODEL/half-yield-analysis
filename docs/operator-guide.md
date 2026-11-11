# Operator guide

Run commands from the repository root after you install the dependencies.

## Operator workflow

Start the file-picker launcher:

```bash
uv run python desktop.py
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
uv run python main.py --source latest.mp4 --show
```

The dashboard shows half share, piece share, a trend, and video position.
The video remains visible at the end until you finish or seek to another position.
The dashboard always shows the trend. The legacy chart overlay flags apply to neither its size nor its position.
Saved annotated videos do not include dashboard controls.

**Measurement:** Half share equals detected half area divided by detected half and piece area.
Shell and obscured material do not contribute to this measurement. This measurement is not weight yield.
An empty reading means that the application has insufficient detected area or time samples.
Check camera alignment before each review. The measurement region is fixed at the coordinates in the [README](../README.md#measurement).

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
uv run python main.py --source latest.mp4 --show --half-conf 0.35 --piece-conf 0.30 \
  --half-mass-per-area 0.0012 --piece-mass-per-area 0.0008
```

