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
import threading
import queue
import time

import cv2
import numpy as np

# Ultralytics YOLO
try:
    from ultralytics import YOLO
except Exception:
    print("Error: Ultralytics not installed. Try: pip install ultralytics")
    raise

# PyTorch for device detection
try:
    import torch
except ImportError:
    torch = None
    print("Warning: PyTorch not available. Falling back to CPU inference.")

# ---- ROI & model input ----
# Crop a 500 (width) x 600 (height) region whose top-left is (x=400, y=300).
ROI_X, ROI_Y = 400, 300
ROI_W, ROI_H = 500, 600
MODEL_SIZE = 640  # 640x640 input for the model

SKIP_DEFAULT = 5  # seconds to skip on arrow keys when playing


# ======================= Device Detection =======================

def get_optimal_device():
    """
    Automatically detect the best available device for inference.
    Returns the optimal device string for the current platform.
    """
    if torch is None:
        print("PyTorch not available. Using CPU.")
        return "cpu"
    
    # Check for CUDA (NVIDIA GPU)
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        if device_count > 1:
            # Multi-GPU setup
            print(f"CUDA available: {device_count} device(s). Multi-GPU setup detected.")
            for i in range(device_count):
                device_name = torch.cuda.get_device_name(i)
                memory_gb = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                print(f"  GPU {i}: {device_name} ({memory_gb:.1f}GB)")
            return "cuda"  # Return generic cuda for multi-GPU handling
        else:
            device_name = torch.cuda.get_device_name(0)
            memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"CUDA available: {device_count} device(s). Using GPU: {device_name} ({memory_gb:.1f}GB)")
            return "cuda:0"
    
    # Check for MPS (Apple Silicon Metal Performance Shaders)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("Apple Metal Performance Shaders (MPS) available. Using MPS acceleration.")
        return "mps"
    
    # Fallback to CPU
    print("No GPU acceleration available. Using CPU.")
    return "cpu"


def get_gpu_devices():
    """Get list of available CUDA devices."""
    if not torch or not torch.cuda.is_available():
        return []
    
    devices = []
    for i in range(torch.cuda.device_count()):
        devices.append({
            'id': i,
            'name': torch.cuda.get_device_name(i),
            'memory_gb': torch.cuda.get_device_properties(i).total_memory / (1024**3),
            'device_str': f"cuda:{i}"
        })
    return devices


def get_optimal_batch_size(device, model_size=640, max_batch_size=256, memory_fraction=0.8):
    """
    Determine optimal batch size based on available GPU memory.
    """
    if not torch or not torch.cuda.is_available() or "cuda" not in device:
        return min(8, max_batch_size)  # Conservative default for non-CUDA
    
    try:
        # Get GPU memory info
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        available_memory = gpu_memory * memory_fraction
        
        # Estimate memory per frame (rough calculation)
        # Model input: 3 channels * model_size^2 * 4 bytes (float32) * 2 (input + gradients)
        bytes_per_frame = 3 * model_size * model_size * 4 * 2
        
        # Add overhead for model weights and intermediate activations (rough estimate)
        model_overhead = 2 * 1024**3  # ~2GB for model + activations
        
        # Calculate max batch size
        usable_memory = available_memory - model_overhead
        theoretical_max = int(usable_memory / bytes_per_frame)
        
        # Apply safety factors and limits
        safe_batch_size = min(theoretical_max // 2, max_batch_size)  # 50% safety margin
        safe_batch_size = max(safe_batch_size, 1)  # At least 1
        
        # Round to power of 2 for better GPU utilization
        power_of_2 = 1
        while power_of_2 * 2 <= safe_batch_size:
            power_of_2 *= 2
        
        optimal_batch_size = power_of_2
        
        print(f"GPU Memory: {gpu_memory / (1024**3):.1f}GB, "
              f"Calculated optimal batch size: {optimal_batch_size}")
        
        return optimal_batch_size
        
    except Exception as e:
        print(f"Could not determine optimal batch size: {e}. Using default.")
        return min(16, max_batch_size)


def create_video_writer(output_path, fps, width, height, use_nvenc=True):
    """
    Create VideoWriter with NVENC hardware encoding if available.
    Falls back to CPU encoding if NVENC is not available.
    """
    if use_nvenc and torch and torch.cuda.is_available():
        try:
            # Try NVENC backends
            nvenc_configs = [
                (cv2.CAP_FFMPEG, 'H264'),  # H.264 NVENC
                (cv2.CAP_FFMPEG, 'HEVC'),  # H.265 NVENC
            ]
            
            for backend, codec in nvenc_configs:
                try:
                    fourcc = cv2.VideoWriter_fourcc(*codec)
                    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height), True)
                    
                    # Test if it works
                    if writer.isOpened():
                        print(f"NVENC hardware encoding enabled ({codec})")
                        return writer
                    writer.release()
                except Exception as e:
                    print(f"NVENC codec {codec} failed: {e}")
                    continue
                    
            print("NVENC not available, falling back to CPU encoding")
        except Exception as e:
            print(f"NVENC initialization failed: {e}")
    
    # Fallback to standard CPU encoding
    try:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height), True)
        if writer.isOpened():
            print("Using CPU video encoding")
            return writer
        writer.release()
    except Exception as e:
        print(f"MP4V encoding failed: {e}")
    
    # Final fallback
    try:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height), True)
        if writer.isOpened():
            print("Using XVID CPU encoding")
            return writer
    except Exception as e:
        print(f"XVID encoding failed: {e}")
    
    raise RuntimeError(f"Could not create video writer for: {output_path}")


def create_video_capture(video_path, use_nvdec=True):
    """
    Create VideoCapture with NVDEC hardware decoding if available.
    Falls back to CPU decoding if NVDEC is not available.
    """
    # Try NVDEC first if requested and CUDA is available
    if use_nvdec and torch and torch.cuda.is_available():
        try:
            # Try different NVDEC backends
            nvdec_backends = [
                cv2.CAP_FFMPEG,  # FFmpeg with NVDEC
                cv2.CAP_GSTREAMER,  # GStreamer with NVDEC
            ]
            
            for backend in nvdec_backends:
                try:
                    cap = cv2.VideoCapture(str(video_path), backend)
                    
                    # Configure NVDEC properties
                    cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
                    cap.set(cv2.CAP_PROP_HW_DEVICE, 0)  # Use GPU 0
                    
                    # Test if it works
                    if cap.isOpened():
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
                            print(f"NVDEC hardware decoding enabled (backend: {backend})")
                            return cap
                        cap.release()
                except Exception as e:
                    print(f"NVDEC backend {backend} failed: {e}")
                    continue
                    
            print("NVDEC not available, falling back to CPU decoding")
        except Exception as e:
            print(f"NVDEC initialization failed: {e}")
    
    # Fallback to standard CPU decoding
    cap = cv2.VideoCapture(str(video_path))
    if cap.isOpened():
        print("Using CPU video decoding")
        return cap
    else:
        raise RuntimeError(f"Could not open video file: {video_path}")


def get_gpu_memory_info():
    """Get GPU memory information for optimization."""
    if not torch or not torch.cuda.is_available():
        return None
    
    try:
        device = torch.cuda.current_device()
        total_memory = torch.cuda.get_device_properties(device).total_memory
        allocated_memory = torch.cuda.memory_allocated(device)
        cached_memory = torch.cuda.memory_reserved(device)
        free_memory = total_memory - allocated_memory
        
        return {
            'total_gb': total_memory / (1024**3),
            'allocated_gb': allocated_memory / (1024**3),
            'cached_gb': cached_memory / (1024**3),
            'free_gb': free_memory / (1024**3),
        }
    except Exception:
        return None

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


# ======================= Yield Analysis =======================

def calculate_yield_statistics(area_hist, bin_duration_sec=10.0):
    """
    Calculate yield statistics from area history.
    
    Args:
        area_hist: deque of (t_ms, half_area, piece_area) tuples
        bin_duration_sec: Duration of time bins in seconds for binned analysis
    
    Returns:
        dict containing:
        - net_average: Overall yield ratio
        - total_half_area: Sum of all half areas
        - total_piece_area: Sum of all piece areas
        - total_area: Sum of half + piece areas
        - duration_sec: Total video duration processed
        - binned_data: List of (bin_start_sec, bin_yield, bin_half, bin_piece) tuples
    """
    if len(area_hist) < 2:
        return {
            'net_average': 0.0,
            'total_half_area': 0.0,
            'total_piece_area': 0.0,
            'total_area': 0.0,
            'duration_sec': 0.0,
            'binned_data': []
        }
    
    # Convert to lists for easier processing
    samples = list(area_hist)
    
    # Calculate overall statistics using integration over time
    total_half_integrated = 0.0
    total_piece_integrated = 0.0
    
    # Integrate using left-Riemann sum (same as the ratio calculation)
    for i in range(len(samples) - 1):
        t0, half0, piece0 = samples[i]
        t1, _, _ = samples[i + 1]
        dt_sec = (t1 - t0) / 1000.0  # Convert ms to seconds
        
        total_half_integrated += half0 * dt_sec
        total_piece_integrated += piece0 * dt_sec
    
    # Calculate net average yield
    total_integrated = total_half_integrated + total_piece_integrated
    net_average = total_half_integrated / total_integrated if total_integrated > 0 else 0.0
    
    # Calculate duration
    duration_sec = (samples[-1][0] - samples[0][0]) / 1000.0 if len(samples) > 1 else 0.0
    
    # Calculate binned statistics
    binned_data = []
    if duration_sec > 0:
        bin_duration_ms = bin_duration_sec * 1000.0
        start_time = samples[0][0]
        end_time = samples[-1][0]
        
        current_bin_start = start_time
        while current_bin_start < end_time:
            current_bin_end = min(current_bin_start + bin_duration_ms, end_time)
            
            # Find samples in this bin
            bin_half = 0.0
            bin_piece = 0.0
            
            for i in range(len(samples) - 1):
                t0, half0, piece0 = samples[i]
                t1, _, _ = samples[i + 1]
                
                # Check if segment overlaps with bin
                seg_start = max(t0, current_bin_start)
                seg_end = min(t1, current_bin_end)
                
                if seg_end > seg_start:
                    dt_sec = (seg_end - seg_start) / 1000.0
                    bin_half += half0 * dt_sec
                    bin_piece += piece0 * dt_sec
            
            # Calculate yield for this bin
            bin_total = bin_half + bin_piece
            bin_yield = bin_half / bin_total if bin_total > 0 else 0.0
            
            bin_start_sec = (current_bin_start - start_time) / 1000.0
            binned_data.append((bin_start_sec, bin_yield, bin_half, bin_piece))
            
            current_bin_start = current_bin_end
    
    return {
        'net_average': net_average,
        'total_half_area': total_half_integrated,
        'total_piece_area': total_piece_integrated,
        'total_area': total_integrated,
        'duration_sec': duration_sec,
        'binned_data': binned_data
    }


def save_yield_report(stats, output_path=None, video_path=None):
    """
    Save yield analysis to CSV file and print to stdout.
    
    Args:
        stats: Dictionary from calculate_yield_statistics()
        output_path: Optional path for CSV output (defaults to video_name + '_yield_analysis.csv')
        video_path: Original video path for naming
    """
    import csv
    from datetime import datetime
    
    # Generate default output path if not provided
    if output_path is None:
        if video_path:
            video_stem = Path(video_path).stem
            output_path = f"{video_stem}_yield_analysis.csv"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"yield_analysis_{timestamp}.csv"
    
    # Print summary to stdout
    print("\n" + "="*60)
    print("HALF YIELD ANALYSIS REPORT")
    print("="*60)
    print(f"Video Duration: {stats['duration_sec']:.2f} seconds")
    print(f"Total Half Area (integrated): {stats['total_half_area']:.2f}")
    print(f"Total Piece Area (integrated): {stats['total_piece_area']:.2f}")
    print(f"Total Area (half + piece): {stats['total_area']:.2f}")
    print(f"NET AVERAGE YIELD: {stats['net_average']:.4f} ({stats['net_average']*100:.2f}%)")
    print("="*60)
    
    if stats['binned_data']:
        print(f"\nYield over time (10-second bins):")
        print(f"{'Time (s)':>8} {'Yield':>8} {'Half Area':>12} {'Piece Area':>12}")
        print("-" * 44)
        for bin_start, bin_yield, bin_half, bin_piece in stats['binned_data']:
            print(f"{bin_start:8.1f} {bin_yield:8.4f} {bin_half:12.2f} {bin_piece:12.2f}")
    
    # Save to CSV
    try:
        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header information
            writer.writerow(['# Half Yield Analysis Report'])
            writer.writerow(['# Generated:', datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
            if video_path:
                writer.writerow(['# Video:', str(video_path)])
            writer.writerow(['# Duration (seconds):', f"{stats['duration_sec']:.2f}"])
            writer.writerow(['# Net Average Yield:', f"{stats['net_average']:.6f}"])
            writer.writerow(['# Total Half Area:', f"{stats['total_half_area']:.2f}"])
            writer.writerow(['# Total Piece Area:', f"{stats['total_piece_area']:.2f}"])
            writer.writerow([''])
            
            # Write binned data
            writer.writerow(['Time_Start_Seconds', 'Yield_Ratio', 'Half_Area_Integrated', 'Piece_Area_Integrated', 'Total_Area'])
            for bin_start, bin_yield, bin_half, bin_piece in stats['binned_data']:
                writer.writerow([f"{bin_start:.1f}", f"{bin_yield:.6f}", f"{bin_half:.2f}", f"{bin_piece:.2f}", f"{bin_half + bin_piece:.2f}"])
        
        print(f"\nDetailed analysis saved to: {output_path}")
        
    except Exception as e:
        print(f"\nWarning: Could not save CSV file '{output_path}': {e}")
    
    print("="*60 + "\n")


# ======================= Parallel Processing =======================

class MultiGPUFrameProcessor:
    """Multi-GPU frame processor for maximum throughput on dual GPU systems."""
    
    def __init__(self, model_path, class_config, args, max_queue_size=50, batch_size=32):
        self.model_path = model_path
        self.class_config = class_config
        self.args = args
        # Create separate queues for each GPU to ensure load balancing
        self.gpu_devices = get_gpu_devices()
        self.frame_queues = [queue.Queue(maxsize=max_queue_size//len(self.gpu_devices)) for _ in self.gpu_devices]
        self.result_queue = queue.Queue(maxsize=max_queue_size)
        self.stop_event = threading.Event()
        self.batch_size = batch_size
        self.workers = []
        self.next_gpu = 0  # Round-robin assignment
        
        if not self.gpu_devices:
            raise RuntimeError("No CUDA devices available for multi-GPU processing")
        
        print(f"MultiGPU: Created {len(self.frame_queues)} separate queues for load balancing")
    
    def start_workers(self):
        """Start GPU workers, one per available GPU."""
        for i, gpu_info in enumerate(self.gpu_devices):
            worker = threading.Thread(
                target=self._gpu_worker_loop, 
                args=(gpu_info, i),  # Pass queue index
                name=f"GPU-{gpu_info['id']}-Worker"
            )
            worker.daemon = True
            worker.start()
            self.workers.append(worker)
            print(f"Started worker for GPU {gpu_info['id']}: {gpu_info['name']}")
    
    def stop_workers(self):
        """Stop all GPU workers."""
        self.stop_event.set()
        # Add sentinel values to unblock workers
        for frame_queue in self.frame_queues:
            try:
                frame_queue.put(None, timeout=1.0)
            except queue.Full:
                pass
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=3.0)
    
    def _gpu_worker_loop(self, gpu_info, queue_index):
        """Worker loop for a specific GPU."""
        gpu_id = gpu_info['id']
        device_str = gpu_info['device_str']
        frame_queue = self.frame_queues[queue_index]  # Use dedicated queue
        
        # Load model on this specific GPU
        try:
            from ultralytics import YOLO
            model = YOLO(self.model_path)
            model.to(device_str)
            
            # GPU-specific optimizations
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.set_device(gpu_id)
                    torch.cuda.empty_cache()
            except Exception:
                pass  # Fall back gracefully if torch not available
            
            print(f"GPU {gpu_id} model loaded and ready")
            
        except Exception as e:
            print(f"Failed to initialize model on GPU {gpu_id}: {e}")
            return
        
        HALF_IDS, OBSCURED_IDS, PIECE_IDS, SHELL_IDS = self.class_config
        print(f"GPU {gpu_id} worker started, waiting for frames...")
        
        while not self.stop_event.is_set():
            try:
                # Collect a batch of frames for this GPU
                batch_items = []
                
                # Collect frames with timeout to avoid hanging
                for i in range(self.batch_size):
                    try:
                        # Use shorter timeout for first frame, longer for subsequent
                        timeout = 1.0 if i == 0 else 0.1
                        item = frame_queue.get(timeout=timeout)
                        if item is None:  # Sentinel value
                            if batch_items:
                                break  # Process what we have
                            else:
                                print(f"GPU {gpu_id} worker exiting due to sentinel")
                                return  # Exit worker
                        batch_items.append(item)
                        
                        # Debug output for first batch
                        if len(batch_items) == 1 and batch_items[0][0] < 5:
                            print(f"GPU {gpu_id} got first frame {batch_items[0][0]}")
                            
                    except queue.Empty:
                        if batch_items:
                            # Process partial batch if we have some frames
                            print(f"GPU {gpu_id} processing partial batch of {len(batch_items)} frames")
                            break
                        elif i == 0:
                            # No frames available, continue waiting
                            continue
                        else:
                            # Got some frames but timed out getting more
                            break
                
                if not batch_items:
                    continue
                
                # Process batch on this GPU
                try:
                    import torch
                    print(f"GPU {gpu_id} starting batch processing of {len(batch_items)} frames")
                    if torch.cuda.is_available():
                        with torch.cuda.device(gpu_id):
                            batch_results = self._process_frame_batch_gpu(
                                batch_items, model, gpu_id, HALF_IDS, OBSCURED_IDS, PIECE_IDS, SHELL_IDS
                            )
                    else:
                        batch_results = self._process_frame_batch_gpu(
                            batch_items, model, gpu_id, HALF_IDS, OBSCURED_IDS, PIECE_IDS, SHELL_IDS
                        )
                        
                    print(f"GPU {gpu_id} completed batch processing, got {len(batch_results)} results")
                    
                    # Put results back
                    for result in batch_results:
                        try:
                            self.result_queue.put(result, timeout=1.0)
                            if result[0] < 5:  # Debug first few results
                                print(f"GPU {gpu_id} added result for frame {result[0]} to result queue")
                        except queue.Full:
                            print(f"GPU {gpu_id} result queue full, skipping remaining results")
                            break  # Skip remaining if queue is full
                                
                except Exception as e:
                    print(f"GPU {gpu_id} batch processing error: {e}")
                    # Fall back to individual processing
                    for item in batch_items:
                        try:
                            frame_id, frame, pos_ms = item
                            out, half_area, piece_area = process_frame(
                                frame, model,
                                HALF_IDS, OBSCURED_IDS, PIECE_IDS, SHELL_IDS,
                                base_conf=self.args.conf,
                                class_thresh={
                                    "half": self.args.half_conf,
                                    "obscured": self.args.obscured_conf, 
                                    "piece": self.args.piece_conf,
                                    "shell": self.args.shell_conf,
                                },
                                min_area_px=self.args.min_area,
                            )
                            self.result_queue.put((frame_id, out, half_area, piece_area, pos_ms), timeout=1.0)
                        except Exception as inner_e:
                            print(f"GPU {gpu_id} individual frame processing error: {inner_e}")
                            continue
                    
            except Exception as e:
                print(f"GPU {gpu_id} worker error: {e}")
                continue
    
    def _process_frame_batch_gpu(self, batch_items, model, gpu_id, HALF_IDS, OBSCURED_IDS, PIECE_IDS, SHELL_IDS):
        """Process a batch of frames on a specific GPU."""
        if not batch_items:
            return []
        
        try:
            # Prepare batch data
            batch_frames = []
            batch_metadata = []
            valid_items = []
            
            for frame_id, frame, pos_ms in batch_items:
                H, W = frame.shape[:2]
                x1, y1 = ROI_X, ROI_Y
                x2, y2 = min(ROI_X + ROI_W, W), min(ROI_Y + ROI_H, H)
                
                if x1 >= x2 or y1 >= y2:
                    batch_metadata.append((frame_id, frame, pos_ms, None))
                    continue
                
                valid_items.append((frame_id, frame, pos_ms, x1, y1, x2, y2))
            
            if not valid_items:
                return [(item[0], item[1], 0.0, 0.0, item[2]) for item in batch_items]
            
            # Batch preprocessing
            batch_list = []
            for i, (frame_id, frame, pos_ms, x1, y1, x2, y2) in enumerate(valid_items):
                crop = frame[y1:y2, x1:x2]
                resized = cv2.resize(crop, (MODEL_SIZE, MODEL_SIZE), interpolation=cv2.INTER_LINEAR)
                batch_list.append(resized)
                batch_frames.append(frame)
                batch_metadata.append((frame_id, frame, pos_ms, resized))
            
            # GPU batch inference
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
                
            results = model.predict(
                batch_list, 
                imgsz=MODEL_SIZE, 
                verbose=False, 
                conf=self.args.conf,
                device=f"cuda:{gpu_id}",
                half=True,  # Use FP16 for speed
                augment=False,
            )
            
            # Process results
            batch_results = []
            result_idx = 0
            
            for frame_id, frame, pos_ms, resized_crop in batch_metadata:
                if resized_crop is None:
                    batch_results.append((frame_id, frame, 0.0, 0.0, pos_ms))
                    continue
                
                if result_idx < len(results):
                    r = results[result_idx]
                    result_idx += 1
                    
                    out, half_area, piece_area = self._process_single_result_gpu(
                        frame, r, HALF_IDS, OBSCURED_IDS, PIECE_IDS, SHELL_IDS, model
                    )
                    batch_results.append((frame_id, out, half_area, piece_area, pos_ms))
                else:
                    batch_results.append((frame_id, frame, 0.0, 0.0, pos_ms))
            
            return batch_results
            
        except Exception as e:
            if "out of memory" in str(e).lower():
                print(f"GPU {gpu_id} out of memory with batch size {len(batch_items)}. Consider reducing batch size.")
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
            else:
                print(f"GPU {gpu_id} batch processing failed: {e}")
            return []
    
    def _process_single_result_gpu(self, frame, result, HALF_IDS, OBSCURED_IDS, PIECE_IDS, SHELL_IDS, model):
        """Process a single inference result on GPU."""
        # Reuse the existing _process_single_result logic
        processor = FrameProcessor(model, self.class_config, self.args, 1, 1, 1)
        return processor._process_single_result(frame, result, HALF_IDS, OBSCURED_IDS, PIECE_IDS, SHELL_IDS)
    
    def add_frame(self, frame_id, frame, pos_ms):
        """Add frame for processing (round-robin across GPUs)."""
        try:
            # Round-robin assignment to GPU queues
            gpu_queue = self.frame_queues[self.next_gpu]
            gpu_queue.put((frame_id, frame, pos_ms), block=False)
            
            # Debug output for first few frames
            if frame_id < 6:
                print(f"DEBUG: MultiGPU added frame {frame_id} to GPU {self.next_gpu} queue (queue size: {gpu_queue.qsize()})")
            
            # Move to next GPU (round-robin)
            self.next_gpu = (self.next_gpu + 1) % len(self.gpu_devices)
            return True
        except queue.Full:
            if frame_id < 6:
                print(f"DEBUG: MultiGPU queue {self.next_gpu} full when adding frame {frame_id}")
            return False
    
    def get_result(self, timeout=0.1):
        """Get processed result (non-blocking)."""
        try:
            result = self.result_queue.get(timeout=timeout)
            if result and result[0] < 3:  # Debug first few results
                print(f"DEBUG: MultiGPU got result for frame {result[0]} (result queue size: {self.result_queue.qsize()})")
            return result
        except queue.Empty:
            return None


class FrameProcessor:
    """Multi-threaded frame processor with batch processing for improved GPU utilization."""
    
    def __init__(self, model, class_config, args, max_queue_size=20, num_workers=2, batch_size=4):
        self.model = model
        self.class_config = class_config
        self.args = args
        self.frame_queue = queue.Queue(maxsize=max_queue_size)
        self.result_queue = queue.Queue(maxsize=max_queue_size)
        self.stop_event = threading.Event()
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.workers = []
        
    def start_workers(self):
        """Start worker threads for parallel processing."""
        for i in range(self.num_workers):
            worker = threading.Thread(target=self._worker_loop, name=f"Worker-{i}")
            worker.daemon = True
            worker.start()
            self.workers.append(worker)
    
    def stop_workers(self):
        """Stop all worker threads."""
        self.stop_event.set()
        # Add sentinel values to unblock workers
        for _ in range(self.num_workers):
            try:
                self.frame_queue.put(None, timeout=1.0)
            except queue.Full:
                pass
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=2.0)
    
    def _worker_loop(self):
        """Worker thread main loop with batch processing."""
        HALF_IDS, OBSCURED_IDS, PIECE_IDS, SHELL_IDS = self.class_config
        
        while not self.stop_event.is_set():
            try:
                # Collect a batch of frames
                batch_items = []
                
                # Try to get batch_size items, but don't wait too long for a full batch
                for _ in range(self.batch_size):
                    try:
                        item = self.frame_queue.get(timeout=0.1)
                        if item is None:  # Sentinel value
                            if batch_items:
                                break  # Process what we have
                            else:
                                return  # Exit worker
                        batch_items.append(item)
                    except queue.Empty:
                        if batch_items:
                            break  # Process partial batch
                        else:
                            continue  # Keep waiting
                
                if not batch_items:
                    continue
                
                # Process batch
                try:
                    batch_results = self._process_frame_batch(batch_items, HALF_IDS, OBSCURED_IDS, PIECE_IDS, SHELL_IDS)
                    
                    # Put results back
                    for result in batch_results:
                        try:
                            self.result_queue.put(result, timeout=1.0)
                        except queue.Full:
                            break  # Skip remaining if queue is full
                            
                except Exception as e:
                    print(f"Batch processing error: {e}")
                    # Fall back to individual processing
                    for item in batch_items:
                        try:
                            frame_id, frame, pos_ms = item
                            out, half_area, piece_area = process_frame(
                                frame, self.model,
                                HALF_IDS, OBSCURED_IDS, PIECE_IDS, SHELL_IDS,
                                base_conf=self.args.conf,
                                class_thresh={
                                    "half": self.args.half_conf,
                                    "obscured": self.args.obscured_conf, 
                                    "piece": self.args.piece_conf,
                                    "shell": self.args.shell_conf,
                                },
                                min_area_px=self.args.min_area,
                            )
                            self.result_queue.put((frame_id, out, half_area, piece_area, pos_ms), timeout=1.0)
                        except Exception as inner_e:
                            print(f"Individual frame processing error: {inner_e}")
                            continue
                    
            except Exception as e:
                print(f"Worker error: {e}")
                continue
    
    def _process_frame_batch(self, batch_items, HALF_IDS, OBSCURED_IDS, PIECE_IDS, SHELL_IDS):
        """Process a batch of frames efficiently using optimized GPU batch inference."""
        if not batch_items:
            return []
        
        try:
            # Prepare batch data with GPU-optimized preprocessing
            batch_frames = []
            batch_metadata = []
            
            # Pre-allocate numpy array for better memory efficiency
            valid_items = []
            for frame_id, frame, pos_ms in batch_items:
                H, W = frame.shape[:2]
                
                # Crop ROI bounds and check
                x1, y1 = ROI_X, ROI_Y
                x2, y2 = min(ROI_X + ROI_W, W), min(ROI_Y + ROI_H, H)
                if x1 >= x2 or y1 >= y2:
                    batch_metadata.append((frame_id, frame, pos_ms, None))
                    continue
                
                valid_items.append((frame_id, frame, pos_ms, x1, y1, x2, y2))
            
            if not valid_items:
                return [(item[0], item[1], 0.0, 0.0, item[2]) for item in batch_items]
            
            # Batch preprocessing - vectorized operations
            batch_tensor = np.zeros((len(valid_items), MODEL_SIZE, MODEL_SIZE, 3), dtype=np.uint8)
            
            for i, (frame_id, frame, pos_ms, x1, y1, x2, y2) in enumerate(valid_items):
                crop = frame[y1:y2, x1:x2]
                resized = cv2.resize(crop, (MODEL_SIZE, MODEL_SIZE), interpolation=cv2.INTER_LINEAR)
                batch_tensor[i] = resized
                
                batch_frames.append(frame)
                batch_metadata.append((frame_id, frame, pos_ms, resized))
            
            # Convert to list for YOLO (YOLO expects list of images)
            batch_list = [batch_tensor[i] for i in range(len(valid_items))]
            
            # Batch inference with optimized settings for high-throughput
            if torch and torch.cuda.is_available():
                torch.cuda.empty_cache()  # Clear cache before batch
                
            results = self.model.predict(
                batch_list, 
                imgsz=MODEL_SIZE, 
                verbose=False, 
                conf=self.args.conf,
                device=self.model.device,
                # Optimize for throughput
                half=(torch and torch.cuda.is_available()),  # Use FP16 on CUDA for 2x speed
                augment=False,  # Disable augmentation for speed
            )
            
            # Process results
            batch_results = []
            result_idx = 0
            
            for frame_id, frame, pos_ms, resized_crop in batch_metadata:
                if resized_crop is None:
                    # Invalid crop
                    batch_results.append((frame_id, frame, 0.0, 0.0, pos_ms))
                    continue
                
                # Get corresponding result
                if result_idx < len(results):
                    r = results[result_idx]
                    result_idx += 1
                    
                    # Process this frame's result
                    out, half_area, piece_area = self._process_single_result(
                        frame, r, HALF_IDS, OBSCURED_IDS, PIECE_IDS, SHELL_IDS
                    )
                    batch_results.append((frame_id, out, half_area, piece_area, pos_ms))
                else:
                    # No result available
                    batch_results.append((frame_id, frame, 0.0, 0.0, pos_ms))
            
            return batch_results
            
        except Exception as e:
            if torch and "out of memory" in str(e).lower():
                print(f"GPU out of memory with batch size {len(batch_items)}. Consider reducing batch size.")
                # Clear GPU cache and retry with smaller batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                print(f"Batch processing failed: {e}")
            return []
    
    def _process_single_result(self, frame, result, HALF_IDS, OBSCURED_IDS, PIECE_IDS, SHELL_IDS):
        """Process a single inference result."""
        polygons = []  # (poly_xy_full, class_id)
        half_area_sum = 0.0
        piece_area_sum = 0.0

        has_masks = getattr(result, "masks", None) is not None and getattr(result.masks, "xy", None) is not None
        has_boxes = getattr(result, "boxes", None) is not None

        if has_masks and has_boxes and len(result.masks.xy) and len(result.boxes.cls):
            classes = result.boxes.cls.detach().cpu().numpy().astype(int)
            confs   = result.boxes.conf.detach().cpu().numpy().astype(float)

            # Per-detection thresholds
            thresh = np.full_like(confs, float(self.args.conf), dtype=float)

            class_thresh = {
                "half": self.args.half_conf,
                "obscured": self.args.obscured_conf, 
                "piece": self.args.piece_conf,
                "shell": self.args.shell_conf,
            }

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
                seg_xy = result.masks.xy[i]
                if seg_xy is None or len(seg_xy) == 0:
                    continue

                # Robust model-space area
                try:
                    m = result.masks.data[i]
                    area_i = float((m > 0.5).sum().item())
                except Exception:
                    cnt_model = np.asarray(seg_xy, np.float32).reshape(-1, 1, 2)
                    area_i = float(abs(cv2.contourArea(cnt_model)))

                if area_i >= float(self.args.min_area):
                    cid = int(classes[i])
                    # Areas for integration
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

        id_to_name, _ = names_maps(self.model.names)

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

    def add_frame(self, frame_id, frame, pos_ms):
        """Add frame for processing (non-blocking)."""
        try:
            self.frame_queue.put((frame_id, frame, pos_ms), block=False)
            return True
        except queue.Full:
            return False
    
    def get_result(self, timeout=0.1):
        """Get processed result (non-blocking)."""
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None


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
    parser.add_argument("--device", default=None, 
                        help="Device string (e.g., '0', 'cuda:0', 'mps', or 'cpu'). "
                             "If not specified, automatically detects the best available device.")
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
    
    # Hardware acceleration options
    parser.add_argument("--nvdec", action="store_true", default=True,
                        help="Enable NVDEC hardware video decoding on CUDA devices (default: True)")
    parser.add_argument("--no-nvdec", action="store_false", dest="nvdec",
                        help="Disable NVDEC and use CPU video decoding")
    
    # Performance options
    parser.add_argument("--parallel", action="store_true", default=False,
                        help="Enable parallel frame processing for improved performance (headless mode only)")
    parser.add_argument("--num-workers", type=int, default=2,
                        help="Number of worker threads for parallel processing (default: 2)")
    parser.add_argument("--queue-size", type=int, default=20,
                        help="Maximum queue size for parallel processing (default: 20)")
    parser.add_argument("--multi-gpu", action="store_true", default=False,
                        help="Use multi-GPU processing if multiple CUDA devices are available")
    parser.add_argument("--batch-size", type=int, default=0,
                        help="Batch size for GPU inference (default: 0 = auto-detect based on GPU)")
    parser.add_argument("--max-batch-size", type=int, default=256,
                        help="Maximum batch size to attempt (default: 256)")
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.8,
                        help="Fraction of GPU memory to use for batching (default: 0.8)")
    parser.add_argument("--nvenc", action="store_true", default=True,
                        help="Enable NVENC hardware video encoding for output (default: True)")
    
    # Yield analysis options
    parser.add_argument("--yield-report", action="store_true", default=True,
                        help="Generate yield analysis report on exit (default: True)")
    parser.add_argument("--no-yield-report", action="store_false", dest="yield_report",
                        help="Disable yield analysis report")
    parser.add_argument("--yield-output", type=str, default=None,
                        help="Output path for yield analysis CSV (default: auto-generated)")
    parser.add_argument("--yield-bin-seconds", type=float, default=10.0,
                        help="Time bin duration in seconds for yield analysis (default: 10.0)")

    args = parser.parse_args()

    src = Path(args.source)
    if not src.exists():
        print(f"Error: source video not found: {src}")
        sys.exit(1)

    # Load model
    model = YOLO(args.model)
    
    # Set device - use automatic detection if not specified
    if args.device is not None:
        device = args.device
        print(f"Using specified device: {device}")
    else:
        device = get_optimal_device()
    
    model.to(device)

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

    cap = create_video_capture(src, use_nvdec=args.nvdec)
    
    # Get video properties for debug information
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    duration_sec = total_frames / fps if fps > 0 else 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video loaded: {src}")
    print(f"  Resolution: {width}x{height}")
    print(f"  Total frames: {total_frames}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Duration: {duration_sec:.2f} seconds ({duration_sec/60:.1f} minutes)")
    print(f"  ROI: {ROI_W}x{ROI_H} at ({ROI_X},{ROI_Y})")
    
    # Show GPU memory info if using CUDA
    if torch and torch.cuda.is_available():
        gpu_info = get_gpu_memory_info()
        if gpu_info:
            print(f"  GPU Memory: {gpu_info['free_gb']:.1f}GB free / {gpu_info['total_gb']:.1f}GB total")
    
    print("="*50)

    # Prepare writer if needed
    writer = None
    if args.save is not None:
        writer = create_video_writer(args.save, fps, width, height, use_nvenc=args.nvenc)

    window_name = "YOLOv12n-seg Contour Tracking"
    if args.show:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    paused = False
    slow = False
    # Store raw areas per timestamp (t_ms, half_area, piece_area)
    area_hist = deque()  # For real-time ratio calculation (trimmed to window)
    complete_area_hist = deque()  # For final yield analysis (never trimmed)
    last_frame = None  # reuse current frame while paused
    
    # Progress tracking
    frame_count = 0
    last_progress_update = 0
    progress_update_interval = max(1, total_frames // 100)  # Update every 1% or at least every frame

    # Initialize parallel processing if requested and not in GUI mode
    processor = None
    if args.parallel and not args.show:
        # Auto-detect optimal batch size if not specified
        if args.batch_size == 0:
            optimal_batch_size = get_optimal_batch_size(device, MODEL_SIZE, args.max_batch_size, args.gpu_memory_fraction)
        else:
            optimal_batch_size = args.batch_size
            
        class_config = (HALF_IDS, OBSCURED_IDS, PIECE_IDS, SHELL_IDS)
        
        # Check for multi-GPU support
        gpu_devices = get_gpu_devices()
        if args.multi_gpu and len(gpu_devices) > 1:
            print(f"Multi-GPU mode enabled: {len(gpu_devices)} GPUs detected")
            processor = MultiGPUFrameProcessor(args.model, class_config, args, args.queue_size, optimal_batch_size)
            processor.start_workers()
            print(f"Multi-GPU processing started: {len(gpu_devices)} GPU workers, batch size: {optimal_batch_size}")
        else:
            if args.multi_gpu and len(gpu_devices) <= 1:
                print("Multi-GPU requested but only 1 GPU available, falling back to single-GPU mode")
            processor = FrameProcessor(model, class_config, args, args.queue_size, args.num_workers, optimal_batch_size)
            processor.start_workers()
            print(f"Parallel processing enabled: {args.num_workers} workers, batch size: {optimal_batch_size}, queue size: {args.queue_size}")
        
        # GPU-specific optimizations
        if torch and torch.cuda.is_available():
            print("GPU optimizations enabled: FP16 precision, memory management")
            # Warm up the GPU - skip for multi-GPU as each worker handles its own warmup
            if not (args.multi_gpu and len(gpu_devices) > 1):
                try:
                    dummy_input = np.zeros((1, MODEL_SIZE, MODEL_SIZE, 3), dtype=np.uint8)
                    _ = model.predict([dummy_input], verbose=False, imgsz=MODEL_SIZE)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    print("GPU warmup complete")
                except Exception as e:
                    print(f"GPU warmup failed: {e}")
            else:
                print("Multi-GPU mode: individual GPU warmup handled by workers")

    try:
        if processor is not None:
            # Parallel processing mode (headless only)
            pending_results = {}  # frame_id -> (pos_ms, timestamp)
            next_frame_id = 0
            ret = True  # Initialize ret
            
            while True:
                # Read and queue frames
                while len(pending_results) < args.queue_size:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                    if processor.add_frame(next_frame_id, frame, pos_ms):
                        pending_results[next_frame_id] = (pos_ms, time.time())
                        next_frame_id += 1
                    else:
                        # Queue full, process results
                        break
                
                # Process results in order
                result = processor.get_result(timeout=0.1)
                if result is not None:
                    result_frame_id, out, half_area, piece_area, result_pos_ms = result
                    
                    # Process this result
                    frame_count += 1
                    
                    # Update progress
                    if (frame_count - last_progress_update) >= progress_update_interval:
                        progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                        
                        bar_width = 40
                        filled = int(bar_width * progress / 100)
                        bar = '█' * filled + '░' * (bar_width - filled)
                        
                        print(f"\rProcessing: [{bar}] {progress:.1f}% "
                              f"(Frame {frame_count}/{total_frames})", end='', flush=True)
                        last_progress_update = frame_count
                    
                    # Store data for analysis
                    if np.isfinite(result_pos_ms):
                        data_point = (float(result_pos_ms), float(half_area), float(piece_area))
                        complete_area_hist.append(data_point)
                    
                    # Save video frame
                    if writer is not None:
                        writer.write(out)
                    
                    # Clean up
                    if result_frame_id in pending_results:
                        del pending_results[result_frame_id]
                
                # Check if we're done
                if not ret and len(pending_results) == 0:
                    break
                
                # Small delay to prevent busy waiting
                if not pending_results:
                    time.sleep(0.001)
        else:
            # Original single-threaded processing
            while True:
                if not paused:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    last_frame = frame
                    frame_count += 1
                    
                    # Update progress bar (only when not showing GUI to avoid spam)
                    if not args.show and (frame_count - last_progress_update) >= progress_update_interval:
                        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                        progress = (current_frame / total_frames) * 100 if total_frames > 0 else 0
                        elapsed_sec = (cap.get(cv2.CAP_PROP_POS_MSEC) or 0) / 1000.0
                        
                        # Simple progress bar
                        bar_width = 40
                        filled = int(bar_width * progress / 100)
                        bar = '█' * filled + '░' * (bar_width - filled)
                        
                        print(f"\rProcessing: [{bar}] {progress:.1f}% "
                              f"(Frame {current_frame}/{total_frames}, "
                              f"Time: {elapsed_sec:.1f}s/{duration_sec:.1f}s)", end='', flush=True)
                        last_progress_update = frame_count
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

                # Only append when we're not paused and time advanced
                if not paused and np.isfinite(pos_ms):
                    data_point = (float(pos_ms), float(half_area), float(piece_area))
                    area_hist.append(data_point)
                    complete_area_hist.append(data_point)  # Keep complete history for final analysis

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
        # Stop parallel processor if running
        if processor is not None:
            processor.stop_workers()
            
        cap.release()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()
        
        # Final progress completion (only if we were showing progress)
        if not args.show and frame_count > 0:
            print("\n✓ Processing complete!")
        
        # Generate yield analysis report
        if args.yield_report and len(complete_area_hist) >= 2:
            try:
                stats = calculate_yield_statistics(complete_area_hist, args.yield_bin_seconds)
                save_yield_report(stats, args.yield_output, args.source)
            except Exception as e:
                print(f"Warning: Could not generate yield report: {e}")
        elif args.yield_report and len(complete_area_hist) < 2:
            print("Warning: Insufficient data for yield analysis (need at least 2 samples)")


if __name__ == "__main__":
    main()
