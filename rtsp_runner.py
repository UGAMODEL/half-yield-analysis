#!/usr/bin/env python3
"""Real-time RTSP runner optimized for a single NVIDIA GPU.

This script reuses the core ROI + post-processing logic from ``main.py`` but is tuned
for low-latency streaming use-cases:

* Single-GPU CUDA execution (defaults to ``cuda:0`` when available)
* RTSP input with automatic reconnection and buffer flushing
* Optional NVDEC hardware acceleration for decode
* Drop-frame strategy to stay real-time when inference lags the stream
"""
from __future__ import annotations

import argparse
import signal
import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO  # type: ignore[attr-defined]

from main import (
    class_ids_for,
    get_gpu_memory_info,
    get_optimal_device,
    names_maps,
    process_frame,
)


def _prefer_single_cuda_device(explicit: Optional[str]) -> str:
    """Return the best single-GPU device string."""
    if explicit:
        return explicit

    if torch.cuda.is_available():
        try:
            torch.cuda.set_device(0)
        except Exception:
            pass
        return "cuda:0"

    detected = get_optimal_device()
    return detected


def _install_sigint_handler(stop_flag: dict) -> None:
    def handler(signum, frame):  # noqa: ARG001
        stop_flag["stop"] = True
        print("\nStopping… (Ctrl+C)")

    signal.signal(signal.SIGINT, handler)


def _open_rtsp_capture(url: str, use_nvdec: bool, hw_device: int, buffer_size: int) -> cv2.VideoCapture:
    backends = [cv2.CAP_FFMPEG, cv2.CAP_GSTREAMER]
    last_error = None

    for backend in backends:
        try:
            cap = cv2.VideoCapture(url, backend)
            if not cap.isOpened():
                last_error = f"Backend {backend} could not open stream"
                continue

            if use_nvdec and torch.cuda.is_available():
                try:
                    cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
                    cap.set(cv2.CAP_PROP_HW_DEVICE, hw_device)
                except Exception:
                    pass

            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)
            except Exception:
                pass

            return cap
        except Exception as exc:  # pragma: no cover - OpenCV backend errors vary
            last_error = str(exc)
            continue

    raise RuntimeError(f"Unable to open RTSP stream: {last_error or 'unknown error'}")


def _flush_buffer(cap: cv2.VideoCapture, extra_grabs: int) -> None:
    for _ in range(extra_grabs):
        cap.grab()


def _ensure_writer(path: Optional[Path], fps: float, width: int, height: int, use_nvenc: bool) -> Optional[cv2.VideoWriter]:
    if path is None:
        return None

    path.parent.mkdir(parents=True, exist_ok=True)
    from main import create_video_writer  # Lazy import to avoid circular dependencies

    return create_video_writer(str(path), fps, width, height, use_nvenc=use_nvenc)


def _overlay_stats(frame: np.ndarray, fps: float, latency_ms: float, drops: int) -> None:
    text = f"FPS {fps:4.1f} | latency {latency_ms:5.1f} ms | dropped {drops}"
    cv2.putText(
        frame,
        text,
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


def _run_stream(args: argparse.Namespace) -> None:
    device = _prefer_single_cuda_device(args.device)
    print(f"Using device: {device}")

    model = YOLO(args.model)
    model.to(device)
    if "cuda" in device and torch.cuda.is_available():
        try:
            idx = int(device.split(":", 1)[1]) if ":" in device else int(device)
            torch.cuda.set_device(idx)
        except Exception:
            pass
        torch.backends.cudnn.benchmark = True
        model_core = getattr(model, "model", None)
        if model_core is not None and hasattr(model_core, "half"):
            try:
                model_core.half()
            except Exception:
                pass

    id_to_name, _ = names_maps(model.names)
    HALF_IDS = class_ids_for(["half"], id_to_name)
    OBSCURED_IDS = class_ids_for(["obscur"], id_to_name)
    PIECE_IDS = class_ids_for(["piece", "pieces"], id_to_name)
    SHELL_IDS = class_ids_for(["shell"], id_to_name)
    class_thresh = {
        "half": args.half_conf,
        "obscured": args.obscured_conf,
        "piece": args.piece_conf,
        "shell": args.shell_conf,
    }

    stop_flag = {"stop": False}
    _install_sigint_handler(stop_flag)

    cap = _open_rtsp_capture(args.rtsp_url, args.nvdec, args.hw_device, args.buffer_size)

    ok, test_frame = cap.read()
    if not ok:
        raise RuntimeError("Unable to read initial frame from RTSP stream")

    height, width = test_frame.shape[:2]
    print(f"RTSP stream ready at {width}x{height} pixels")

    writer = _ensure_writer(args.record, args.stream_fps, width, height, args.nvenc)

    fps_window = deque(maxlen=max(5, int(args.stats_interval * 30)))
    drops = 0
    last_stats_at = time.time()
    frame_id = 0

    if torch.cuda.is_available() and "cuda" in device:
        info = get_gpu_memory_info()
        if info:
            print(f"GPU memory: {info['free_gb']:.1f}GB free / {info['total_gb']:.1f}GB total")

    # Warmup optional flush
    if args.flush_frames:
        _flush_buffer(cap, args.flush_frames)

    try:
        with torch.inference_mode():
            last_batch_start = time.time()
            batch_frames: list[np.ndarray] = []
            batch_metadata: list[Tuple[int, float]] = []

            while not stop_flag["stop"]:
                grabbed, frame = cap.read()
                if not grabbed:
                    drops += 1
                    time.sleep(args.reconnect_delay)
                    cap.release()
                    cap = _open_rtsp_capture(args.rtsp_url, args.nvdec, args.hw_device, args.buffer_size)
                    continue

                frame_id += 1
                batch_frames.append(frame)
                batch_metadata.append((frame_id, time.time()))

                if args.flush_frames:
                    _flush_buffer(cap, args.flush_frames)

                now = time.time()
                should_process = (
                    len(batch_frames) >= args.batch_size
                    or (now - last_batch_start) >= args.batch_timeout
                )

                if not should_process:
                    continue

                start = time.time()
                results = []
                for frm in batch_frames:
                    annotated, half_area, piece_area = process_frame(
                        frm,
                        model,
                        HALF_IDS,
                        OBSCURED_IDS,
                        PIECE_IDS,
                        SHELL_IDS,
                        args.conf,
                        class_thresh,
                        args.min_area,
                    )
                    results.append((annotated, half_area, piece_area))
                latency_ms = (time.time() - start) * 1000.0

                for (annotated, _half, _piece), (fid, ts) in zip(results, batch_metadata, strict=False):
                    fps_window.append(ts)
                    if args.overlay_stats:
                        fps = 0.0
                        if len(fps_window) >= 2:
                            duration = fps_window[-1] - fps_window[0]
                            if duration > 0:
                                fps = len(fps_window) / duration
                        _overlay_stats(annotated, fps, latency_ms / max(1, len(batch_frames)), drops)

                    if args.show:
                        cv2.imshow("RTSP Runner", annotated)
                        key = cv2.waitKey(1) & 0xFF
                        if key in (ord("q"), 27):
                            stop_flag["stop"] = True
                            break
                    if writer is not None:
                        writer.write(annotated)

                batch_frames.clear()
                batch_metadata.clear()
                last_batch_start = time.time()

                if (time.time() - last_stats_at) >= args.stats_interval:
                    fps = 0.0
                    if len(fps_window) >= 2:
                        duration = fps_window[-1] - fps_window[0]
                        if duration > 0:
                            fps = len(fps_window) / duration
                    print(
                        f"Stats — FPS: {fps:4.1f}, latency: {latency_ms/ max(1,len(results)):5.1f} ms, drops: {drops}, processed frames: {frame_id}"
                    )
                    last_stats_at = time.time()

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Real-time RTSP runner optimized for single NVIDIA GPU (CUDA)."
    )
    parser.add_argument("--rtsp-url", required=True, help="RTSP URL to subscribe to (rtsp://…)")
    parser.add_argument("--model", default="halfyield-model.pt", help="Path to YOLOv12n-seg model")
    parser.add_argument("--device", default=None, help="Torch device override (default: prefer cuda:0)")
    parser.add_argument("--conf", type=float, default=0.30, help="Base confidence threshold")
    parser.add_argument("--half-conf", type=float, default=None, help="Class-specific threshold for half")
    parser.add_argument("--obscured-conf", type=float, default=None, help="Class-specific threshold for obscured")
    parser.add_argument("--piece-conf", type=float, default=None, help="Class-specific threshold for piece")
    parser.add_argument("--shell-conf", type=float, default=None, help="Class-specific threshold for shell")
    parser.add_argument("--min-area", type=float, default=30.0, help="Minimum mask area to keep (model space pixels)")

    parser.add_argument("--batch-size", type=int, default=8, help="Micro-batch size for inference")
    parser.add_argument("--batch-timeout", type=float, default=0.08, help="Max seconds to wait before forcing inference")
    parser.add_argument("--nvdec", action="store_true", default=True, help="Enable NVDEC if available")
    parser.add_argument("--nvenc", action="store_true", default=True, help="Enable NVENC for optional recording")
    parser.add_argument("--hw-device", type=int, default=0, help="CUDA device index for NVDEC/NVENC")
    parser.add_argument("--buffer-size", type=int, default=2, help="Decoder buffer size (frames)")
    parser.add_argument("--flush-frames", type=int, default=0, help="Extra frames to grab+drop each iteration to reduce latency")
    parser.add_argument("--reconnect-delay", type=float, default=2.0, help="Seconds to wait before reconnect attempts")

    parser.add_argument("--show", action="store_true", help="Display annotated stream window")
    parser.add_argument("--overlay-stats", action="store_true", help="Overlay FPS/latency metrics on frames")
    parser.add_argument("--record", type=Path, default=None, help="Optional path to save annotated stream")
    parser.add_argument("--stream-fps", type=float, default=30.0, help="Recording FPS if --record is enabled")
    parser.add_argument("--stats-interval", type=float, default=5.0, help="Seconds between console stats updates")

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        _run_stream(args)
    except Exception as exc:  # pragma: no cover - runtime errors are informative
        print(f"Error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
