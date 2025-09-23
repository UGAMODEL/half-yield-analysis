#!/usr/bin/env python3
"""
YOLOv12n-seg inference with:
 - Fixed ROI crop (500x600 at (x=400, y=300)) -> resize to 640x640 for model
 - Segmentation contours mapped back to full-frame + class label above each contour
 - Per-class confidence thresholds for: half, obscured, piece, shell (fallback to --conf)
 - Rolling, time-integrated ratio over window ΔT:
       ratio(t) = ∫_{t-ΔT}^{t} half(u) du / ∫_{t-ΔT}^{t} [half(u)+piece(u)] du
   (integration via piecewise-constant left-Riemann over sample intervals)
 - Interactive controls:
      SPACE = pause/play
      ← / →  = -/+ skip seconds while playing; prev/next frame while paused
      , / .  = prev/next frame (forces paused)
      S      = toggle slow mode (200ms waitKey)
      Q/ESC  = quit
"""

import argparse
import sys
from pathlib import Path
from collections import deque

import cv2
import numpy as np

# Ultralytics YOLO
try:
    from ultralytics import YOLO
except Exception:
    print("Error: Ultralytics not installed. Try: pip install ultralytics")
    raise

# ---- ROI & model input ----
# Crop a 500 (width) x 600 (height) region whose top-left is (x=400, y=300).
ROI_X, ROI_Y = 400, 300
ROI_W, ROI_H = 500, 600
MODEL_SIZE = 640  # 640x640 input for the model

SKIP_DEFAULT = 5  # seconds to skip on arrow keys when playing


# ======================= Utilities =======================

def apply_seek(cap, delta_seconds):
    """Seek by +/- seconds, using timestamps when available, else frames."""
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cur_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        if np.isfinite(cur_ms) and cur_ms >= 0:
            dur_ms = (cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps) * 1000.0
            new_ms = max(0.0, min(dur_ms, cur_ms + delta_seconds * 1000.0))
            cap.set(cv2.CAP_PROP_POS_MSEC, new_ms)
        else:
            cur_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            new_frame = max(0, cur_frame + int(delta_seconds * fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
    except Exception:
        pass


def apply_step_frame(cap, delta_frames):
    """Move exactly +/- N frames (e.g., -1 or +1), clamped to start."""
    try:
        cur_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        new_frame = max(0, cur_frame + int(delta_frames))
        cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
    except Exception:
        pass


def parse_key(k):
    """Normalize cv2.waitKey result across platforms."""
    if k in (81, 2424832):      # left arrow
        return 'left'
    if k in (83, 2555904):      # right arrow
        return 'right'
    if k in (ord(','),):        # prev frame
        return 'prev_frame'
    if k in (ord('.'),):        # next frame
        return 'next_frame'
    if k in (ord('a'), ord('A'), ord('j'), ord('J')):
        return 'left'
    if k in (ord('d'), ord('D'), ord('l'), ord('L')):
        return 'right'
    if k in (ord(' '),):        # spacebar
        return 'toggle_pause'
    if k in (ord('s'), ord('S')):
        return 'toggle_slow'
    if k in (ord('q'), ord('Q'), 27):  # q or ESC
        return 'quit'
    return None


def names_maps(names):
    """Ultralytics model.names could be list or dict {id: name}."""
    if isinstance(names, dict):
        id_to_name = names
    else:
        id_to_name = {i: n for i, n in enumerate(names)}
    name_to_id = {v: k for k, v in id_to_name.items()}
    return id_to_name, name_to_id


def class_ids_for(substrings, id_to_name):
    """Return set of IDs whose name contains any substring (case-insensitive)."""
    subs = [s.lower() for s in substrings]
    out = set()
    for cid, nm in id_to_name.items():
        nm_l = str(nm).lower()
        if any(s in nm_l for s in subs):
            out.add(int(cid))
    return out


# --------- Time-integrated ratio helpers ---------

def integrated_ratio_last(samples, window_ms):
    """
    Compute time-integrated ratio over the last `window_ms` ending at the most recent sample.
    samples: deque/list of (t_ms, half_area, piece_area), t_ms strictly increasing.
    Uses left-Riemann (area at segment's left endpoint) per interval.
    """
    if len(samples) < 2:
        return None

    t_end = samples[-1][0]
    cutoff = t_end - window_ms
    N = 0.0
    D = 0.0

    # Iterate segments [i] -> [i+1], include overlap with [cutoff, t_end]
    for i in range(len(samples) - 1):
        t0, half0, piece0 = samples[i]
        t1, _, _ = samples[i + 1]

        seg_start = max(t0, cutoff)
        seg_end = t1
        if seg_end <= seg_start:
            continue

        dt = (seg_end - seg_start) / 1000.0  # seconds (unit cancels; consistent)
        N += half0 * dt
        D += (half0 + piece0) * dt

    if D <= 0.0:
        return None
    return N / D


def integrated_ratio_series(samples, window_ms):
    """
    Build a time series of integrated ratio values for plotting.
    Returns list[(t_ms, ratio)] aligned to the sample times.
    Efficient O(n) two-pointer with prefix integrals over segments.
    """
    n = len(samples)
    if n < 2:
        return []

    t = np.array([s[0] for s in samples], dtype=np.float64)
    Ah = np.array([s[1] for s in samples], dtype=np.float64)  # half
    Ap = np.array([s[2] for s in samples], dtype=np.float64)  # piece

    # Segment durations (ms) between samples; last point has no outgoing segment
    dt = np.diff(t)  # length n-1, in ms

    # Prefix integrals over segments using left value:
    # cum[k] = sum_{i<k} f[i] * dt[i], so cum[0]=0, cum[1]=A[0]*dt[0], ...
    cumN = np.zeros(n, dtype=np.float64)
    cumD = np.zeros(n, dtype=np.float64)
    # Fill from i=1..n-1 using segment i-1
    for k in range(1, n):
        cumN[k] = cumN[k - 1] + Ah[k - 1] * dt[k - 1]
        cumD[k] = cumD[k - 1] + (Ah[k - 1] + Ap[k - 1]) * dt[k - 1]

    out = []
    s = 0  # left window pointer (index into samples)
    ms_window = float(window_ms)

    for k in range(n):
        t_end = t[k]
        cutoff = t_end - ms_window

        # Advance s while the next sample is still <= cutoff
        while s + 1 < n and t[s + 1] <= cutoff:
            s += 1

        if cutoff <= t[0]:
            # Entire window lies within [t[0], t[k]]
            N = cumN[k]
            D = cumD[k]
        else:
            # Window crosses segment [s, s+1]; add partial from cutoff->t[s+1]
            # and full segments (s+1..k-1) via prefix differences
            # Guard k==0 handled above, here k>=1 always has cum arrays valid
            # Ensure s < k for valid window
            # Partial duration within segment s:
            partial_dt = max(0.0, t[s + 1] - cutoff)  # ms
            N = (cumN[k] - cumN[s + 1]) + Ah[s] * partial_dt
            D = (cumD[k] - cumD[s + 1]) + (Ah[s] + Ap[s]) * partial_dt

        if D > 0:
            out.append((t_end, float(N / D)))
        else:
            out.append((t_end, None))

    return out


def draw_ratio_chart_time(canvas, samples_ratio, w, h, corner="bl", label_prefix="60s ratio"):
    """
    Time-based ratio chart.
    `samples_ratio` is a list of (t_ms, ratio_or_None), typically from integrated_ratio_series().
    X-axis spans [t_now - window, t_now], Y-axis is 0..1.
    """
    if not samples_ratio:
        return

    H, W = canvas.shape[:2]
    if corner == "tl":
        x0, y0 = 12, 12
    elif corner == "tr":
        x0, y0 = W - w - 12, 12
    elif corner == "br":
        x0, y0 = W - w - 12, H - h - 12
    else:  # "bl"
        x0, y0 = 12, H - h - 12

    panel = canvas[y0:y0+h, x0:x0+w]
    overlay = panel.copy()
    cv2.rectangle(overlay, (0, 0), (w-1, h-1), (50, 50, 50), thickness=-1)
    cv2.addWeighted(overlay, 0.4, panel, 0.6, 0, panel)
    cv2.rectangle(panel, (0, 0), (w-1, h-1), (200, 200, 200), 1)

    # grid/ticks
    for frac in (0.0, 0.5, 1.0):
        y = int((1.0 - frac) * (h-1))
        cv2.line(panel, (0, y), (w-1, y), (160, 160, 160), 1)
        cv2.putText(panel, f"{frac:.1f}", (4, max(12, y-4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (230, 230, 230), 1, cv2.LINE_AA)

    # time range
    t_valid = [tr[0] for tr in samples_ratio if tr[1] is not None]
    if not t_valid:
        canvas[y0:y0+h, x0:x0+w] = panel
        return
    t_start = min(t_valid)
    t_end   = samples_ratio[-1][0]
    window = max(1.0, t_end - t_start)  # ms

    # polyline points spaced by timestamp (not index)
    pts = []
    last_val = None
    for (t_ms, r) in samples_ratio:
        if r is None:
            continue
        x = int(((t_ms - t_start) / window) * (w-1))
        y = int((1.0 - max(0.0, min(1.0, r))) * (h-1))
        pts.append([x, y])
        last_val = r

    if len(pts) >= 2:
        pts = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(panel, [pts], isClosed=False, color=(0, 255, 255), thickness=2)

    # latest value label
    if last_val is not None:
        cv2.putText(panel, f"{label_prefix}: {last_val:.3f}",
                    (8, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)

    canvas[y0:y0+h, x0:x0+w] = panel


# ======================= Core processing =======================

def process_frame(
    frame,
    model,
    HALF_IDS, OBSCURED_IDS, PIECE_IDS, SHELL_IDS,
    base_conf, class_thresh, min_area_px
):
    """
    - Crop ROI -> resize -> model.predict
    - Per-class confidence filtering
    - Accumulate areas (per-frame, in model space) for half & piece
    - Map contours back to full frame, draw + label
    - Return (annotated_frame, half_area_sum, piece_area_sum)
    """
    H, W = frame.shape[:2]

    # Crop ROI bounds and check
    x1, y1 = ROI_X, ROI_Y
    x2, y2 = min(ROI_X + ROI_W, W), min(ROI_Y + ROI_H, H)
    if x1 >= x2 or y1 >= y2:
        return frame, 0.0, 0.0

    crop = frame[y1:y2, x1:x2]
    resized = cv2.resize(crop, (MODEL_SIZE, MODEL_SIZE), interpolation=cv2.INTER_LINEAR)

    # Inference with base conf (stage-1 filter)
    results = model.predict(resized, imgsz=MODEL_SIZE, verbose=False, conf=base_conf)

    polygons = []  # (poly_xy_full, class_id)
    half_area_sum = 0.0
    piece_area_sum = 0.0

    if len(results) > 0:
        r = results[0]
        has_masks = getattr(r, "masks", None) is not None and getattr(r.masks, "xy", None) is not None
        has_boxes = getattr(r, "boxes", None) is not None

        if has_masks and has_boxes and len(r.masks.xy) and len(r.boxes.cls):
            classes = r.boxes.cls.detach().cpu().numpy().astype(int)
            confs   = r.boxes.conf.detach().cpu().numpy().astype(float)

            # Per-detection thresholds start from base_conf; override per class if provided
            thresh = np.full_like(confs, float(base_conf), dtype=float)

            if HALF_IDS:
                thr = class_thresh.get("half")
                if thr is not None:
                    thresh[np.isin(classes, list(HALF_IDS))] = float(thr)

            if OBSCURED_IDS:
                thr = class_thresh.get("obscured")
                if thr is not None:
                    thresh[np.isin(classes, list(OBSCURED_IDS))] = float(thr)

            if PIECE_IDS:
                thr = class_thresh.get("piece")
                if thr is not None:
                    thresh[np.isin(classes, list(PIECE_IDS))] = float(thr)

            if SHELL_IDS:
                thr = class_thresh.get("shell")
                if thr is not None:
                    thresh[np.isin(classes, list(SHELL_IDS))] = float(thr)

            keep = np.where(confs >= thresh)[0]

            # scale back to full-frame
            sx, sy = ROI_W / MODEL_SIZE, ROI_H / MODEL_SIZE

            for i in keep:
                seg_xy = r.masks.xy[i]
                if seg_xy is None or len(seg_xy) == 0:
                    continue

                # Robust model-space area (binarize float mask if needed)
                try:
                    m = r.masks.data[i]                     # torch HxW float
                    area_i = float((m > 0.5).sum().item())  # pixels in MODEL space
                except Exception:
                    cnt_model = np.asarray(seg_xy, np.float32).reshape(-1, 1, 2)
                    area_i = float(abs(cv2.contourArea(cnt_model)))

                if area_i >= float(min_area_px):
                    cid = int(classes[i])
                    # Areas for integration (only half & piece contribute to ratio)
                    if cid in HALF_IDS:
                        half_area_sum += area_i
                    if cid in PIECE_IDS:
                        piece_area_sum += area_i

                # Map polygon to full frame for drawing
                seg_xy = np.asarray(seg_xy, dtype=np.float32)
                seg_xy[:, 0] = seg_xy[:, 0] * sx + ROI_X
                seg_xy[:, 1] = seg_xy[:, 1] * sy + ROI_Y
                polygons.append((seg_xy, int(classes[i])))

    # Compose output frame
    out = frame.copy()
    cv2.rectangle(out, (ROI_X, ROI_Y), (ROI_X + ROI_W, ROI_Y + ROI_H), (255, 255, 255), 1)

    id_to_name, _ = names_maps(model.names)

    for poly, cid in polygons:
        pts = np.round(poly).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

        # class label above centroid
        M = cv2.moments(pts)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            label = str(id_to_name.get(int(cid), int(cid)))
            cv2.putText(out, label, (cx, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

    return out, half_area_sum, piece_area_sum


# ======================= Main =======================

def main():
    parser = argparse.ArgumentParser(description="YOLOv12n-seg contour tracking with integrated ratio chart and per-class thresholds.")
    parser.add_argument("--source", required=True, help="Path to input video file.")
    parser.add_argument("--model", default="halfyield-model.pt", help="Path to YOLOv12n-seg model (.pt).")
    parser.add_argument("--device", default=None, help="Device string (e.g., '0', 'cuda:0', or 'cpu').")
    parser.add_argument("--save", default=None, help="Optional output video path to write.")
    parser.add_argument("--show", action="store_true", help="Show a live window.")

    # Confidence thresholds
    parser.add_argument("--conf", type=float, default=0.30,
                        help="Base confidence threshold for all classes")
    parser.add_argument("--half-conf", type=float, default=None,
                        help="Confidence required for class 'half' (overrides --conf)")
    parser.add_argument("--obscured-conf", type=float, default=None,
                        help="Confidence required for class 'obscured' (overrides --conf)")
    parser.add_argument("--piece-conf", type=float, default=None,
                        help="Confidence required for class 'piece/pieces' (overrides --conf)")
    parser.add_argument("--shell-conf", type=float, default=None,
                        help="Confidence required for class 'shell' (overrides --conf)")
    parser.add_argument("--min-area", type=float, default=30.0,
                        help="Ignore masks smaller than this area in model-space pixels")

    # Interaction / chart
    parser.add_argument("--skip-seconds", type=float, default=SKIP_DEFAULT,
                        help="Seconds to skip with left/right keys (default: 5)")
    parser.add_argument("--ratio-window-sec", type=float, default=60.0,
                        help="Seconds of history for the integrated ratio window (default: 60s)")
    parser.add_argument("--chart-width", type=int, default=320, help="Ratio chart width (px)")
    parser.add_argument("--chart-height", type=int, default=120, help="Ratio chart height (px)")
    parser.add_argument("--chart-pos", type=str, default="bl",
                        help="Chart corner: tl, tr, bl, br (default: bl)")
    parser.add_argument("--show-chart", action="store_true",
                        help="Render rolling integrated ratio chart overlay (requires --show)")

    args = parser.parse_args()

    src = Path(args.source)
    if not src.exists():
        print(f"Error: source video not found: {src}")
        sys.exit(1)

    # Load model
    model = YOLO(args.model)
    if args.device is not None:
        model.to(args.device)

    # Build class-id sets once from model names
    id_to_name, _ = names_maps(model.names)
    HALF_IDS     = class_ids_for(["half"], id_to_name)
    OBSCURED_IDS = class_ids_for(["obscur"], id_to_name)  # matches 'obscured'
    PIECE_IDS    = class_ids_for(["piece", "pieces"], id_to_name)
    SHELL_IDS    = class_ids_for(["shell"], id_to_name)

    class_thresh = {
        "half":     args.half_conf,
        "obscured": args.obscured_conf,
        "piece":    args.piece_conf,
        "shell":    args.shell_conf,
    }

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        print(f"Error: failed to open video: {src}")
        sys.exit(1)

    # Prepare writer if needed
    writer = None
    if args.save is not None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(str(args.save), fourcc, fps, (width, height), True)

    window_name = "YOLOv12n-seg Contour Tracking"
    if args.show:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    paused = False
    slow = False
    # Store raw areas per timestamp (t_ms, half_area, piece_area)
    area_hist = deque()
    last_frame = None  # reuse current frame while paused

    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
                last_frame = frame
            else:
                # When paused, use the last captured frame
                if last_frame is None:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    last_frame = frame
                frame = last_frame

            pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)

            out, half_area, piece_area = process_frame(
                frame, model,
                HALF_IDS, OBSCURED_IDS, PIECE_IDS, SHELL_IDS,
                base_conf=args.conf,
                class_thresh=class_thresh,
                min_area_px=args.min_area,
            )

            # Only append when we’re not paused and time advanced
            if not paused and np.isfinite(pos_ms):
                area_hist.append((float(pos_ms), float(half_area), float(piece_area)))

            # Trim by time window (keep only segments overlapping last ratio-window-sec)
            window_ms = float(args.ratio_window_sec) * 1000.0
            if area_hist:
                t_now = area_hist[-1][0]
                cutoff = t_now - window_ms
                while len(area_hist) > 1 and area_hist[0][0] < cutoff:
                    # Keep at least one sample before cutoff to define the leading segment
                    area_hist.popleft()

            # Save only while playing to keep linear output
            if writer is not None and not paused:
                writer.write(out)

            if args.show:
                # Overlay UI hints
                overlay = out.copy()
                hint = ("SPACE: pause/play | "
                        f"←/→: -{float(args.skip_seconds):.1f}s/+{float(args.skip_seconds):.1f}s (play) "
                        "| prev/next frame (pause) | ,/.: prev/next | S: slow | Q: quit")
                cv2.putText(overlay, hint, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (255, 255, 255), 2, cv2.LINE_AA)
                state = "PAUSED" if paused else "PLAYING"
                cv2.putText(overlay, state, (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 255, 255), 2, cv2.LINE_AA)

                # Compute current integrated ratio and (optionally) series for chart
                ratio_now = integrated_ratio_last(list(area_hist), window_ms)
                if ratio_now is not None:
                    cv2.putText(overlay, f"Integrated {int(args.ratio_window_sec)}s ratio: {ratio_now:.3f}",
                                (12, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (255, 255, 255), 2, cv2.LINE_AA)

                cv2.addWeighted(overlay, 0.35, out, 0.65, 0, out)

                if args.show_chart:
                    series = integrated_ratio_series(list(area_hist), window_ms)
                    draw_ratio_chart_time(out, series, args.chart_width, args.chart_height,
                                          corner=args.chart_pos, label_prefix=f"{int(args.ratio_window_sec)}s ratio")

                cv2.imshow(window_name, out)
                k = cv2.waitKey(200 if slow else (150 if paused else 1)) & 0xFFFF
                action = parse_key(k)

                if action == 'quit':
                    break

                elif action == 'toggle_pause':
                    paused = not paused
                    continue

                elif action == 'toggle_slow':
                    slow = not slow
                    continue

                elif action == 'left':
                    if paused:
                        # prev frame
                        apply_step_frame(cap, -1)
                        ret, frame = cap.read()
                        if ret:
                            last_frame = frame
                    else:
                        apply_seek(cap, -float(args.skip_seconds))
                    continue

                elif action == 'right':
                    if paused:
                        # next frame
                        apply_step_frame(cap, +1)
                        ret, frame = cap.read()
                        if ret:
                            last_frame = frame
                    else:
                        apply_seek(cap, +float(args.skip_seconds))
                    continue

                elif action == 'prev_frame':
                    apply_step_frame(cap, -1)
                    ret, frame = cap.read()
                    if ret:
                        last_frame = frame
                    paused = True
                    continue

                elif action == 'next_frame':
                    apply_step_frame(cap, +1)
                    ret, frame = cap.read()
                    if ret:
                        last_frame = frame
                    paused = True
                    continue

            else:
                # Headless: keep writing if requested (already handled above)
                pass

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
