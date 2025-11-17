#!/usr/bin/env python3
"""
Dedicated GPU Pipeline Architecture:
- GPU 0: Video decode + preprocessing + batching
- GPU 1: Pure inference processing
"""

import queue
import threading
import time
import torch
import cv2
import numpy as np

class DedicatedGPUPipeline:
    """
    Two-GPU pipeline with dedicated roles:
    - GPU 0: NVDEC decode + preprocessing 
    - GPU 1: YOLO inference
    """
    
    def __init__(self, model_path, class_config, args, batch_size=512):
        self.model_path = model_path
        self.class_config = class_config
        self.args = args
        self.batch_size = batch_size
        
        # Queues for pipeline communication
        self.preprocessed_queue = queue.Queue(maxsize=10)  # Batches ready for inference
        self.result_queue = queue.Queue(maxsize=50)        # Inference results
        self.stop_event = threading.Event()
        
        # Workers
        self.decode_worker = None
        self.inference_worker = None
        
        # GPU assignments
        self.decode_gpu = 0      # GPU 0: Decode + preprocess
        self.inference_gpu = 1   # GPU 1: Pure inference
        
        print(f"Dedicated GPU Pipeline:")
        print(f"  GPU {self.decode_gpu}: NVDEC decode + preprocessing")
        print(f"  GPU {self.inference_gpu}: YOLO inference")
        
    def start_workers(self, video_capture):
        """Start the dedicated GPU workers."""
        # Start decode/preprocess worker on GPU 0
        self.decode_worker = threading.Thread(
            target=self._decode_worker_loop,
            args=(video_capture,),
            name="GPU0-Decode-Worker"
        )
        self.decode_worker.daemon = True
        self.decode_worker.start()
        
        # Start inference worker on GPU 1
        self.inference_worker = threading.Thread(
            target=self._inference_worker_loop,
            name="GPU1-Inference-Worker"
        )
        self.inference_worker.daemon = True
        self.inference_worker.start()
        
        print("Dedicated GPU workers started")
    
    def stop_workers(self):
        """Stop all workers."""
        self.stop_event.set()
        
        # Send sentinel to unblock inference worker
        try:
            self.preprocessed_queue.put(None, timeout=1.0)
        except queue.Full:
            pass
            
        # Wait for workers
        if self.decode_worker:
            self.decode_worker.join(timeout=5.0)
        if self.inference_worker:
            self.inference_worker.join(timeout=5.0)
    
    def _decode_worker_loop(self, cap):
        """GPU 0: Dedicated decode and preprocessing worker."""
        try:
            import torch
            torch.cuda.set_device(self.decode_gpu)
            torch.cuda.empty_cache()
            print(f"GPU {self.decode_gpu} decode worker started")
            
            frame_id = 0
            batch_frames = []
            batch_metadata = []
            
            while not self.stop_event.is_set():
                try:
                    # Read frame with NVDEC (already on GPU 0)
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                    
                    # Preprocess on GPU 0
                    processed_frame = self._preprocess_frame_gpu0(frame, frame_id)
                    if processed_frame is not None:
                        batch_frames.append(processed_frame)
                        batch_metadata.append((frame_id, frame, pos_ms))
                    
                    frame_id += 1
                    
                    # When batch is ready, send to inference GPU
                    if len(batch_frames) >= self.batch_size:
                        batch_data = {
                            'frames': batch_frames,
                            'metadata': batch_metadata,
                            'batch_id': frame_id // self.batch_size
                        }
                        
                        try:
                            self.preprocessed_queue.put(batch_data, timeout=1.0)
                            if frame_id < 1000:  # Debug early batches
                                print(f"GPU {self.decode_gpu} sent batch {batch_data['batch_id']} ({len(batch_frames)} frames) to inference")
                            batch_frames = []
                            batch_metadata = []
                        except queue.Full:
                            print(f"GPU {self.decode_gpu} preprocessed queue full, dropping batch")
                            batch_frames = []
                            batch_metadata = []
                
                except Exception as e:
                    print(f"GPU {self.decode_gpu} decode error: {e}")
                    continue
            
            # Send final partial batch
            if batch_frames:
                batch_data = {
                    'frames': batch_frames,
                    'metadata': batch_metadata,
                    'batch_id': -1  # Final batch
                }
                try:
                    self.preprocessed_queue.put(batch_data, timeout=1.0)
                except queue.Full:
                    pass
                    
        except Exception as e:
            print(f"GPU {self.decode_gpu} decode worker failed: {e}")
    
    def _inference_worker_loop(self):
        """GPU 1: Dedicated inference worker."""
        try:
            import torch
            from ultralytics import YOLO
            
            # Load model on inference GPU
            torch.cuda.set_device(self.inference_gpu)
            model = YOLO(self.model_path)
            model.to(f"cuda:{self.inference_gpu}")
            torch.cuda.empty_cache()
            
            print(f"GPU {self.inference_gpu} inference worker started, model loaded")
            
            HALF_IDS, OBSCURED_IDS, PIECE_IDS, SHELL_IDS = self.class_config
            
            while not self.stop_event.is_set():
                try:
                    # Get preprocessed batch from GPU 0
                    batch_data = self.preprocessed_queue.get(timeout=1.0)
                    if batch_data is None:  # Sentinel
                        break
                    
                    batch_frames = batch_data['frames']
                    batch_metadata = batch_data['metadata']
                    batch_id = batch_data['batch_id']
                    
                    if batch_id >= 0 and batch_id < 10:  # Debug early batches
                        print(f"GPU {self.inference_gpu} processing batch {batch_id} ({len(batch_frames)} frames)")
                    
                    # Pure inference on GPU 1
                    with torch.cuda.device(self.inference_gpu):
                        results = model.predict(
                            batch_frames,
                            imgsz=640,  # MODEL_SIZE
                            verbose=False,
                            conf=self.args.conf,
                            device=f"cuda:{self.inference_gpu}",
                            half=True,
                            augment=False,
                        )
                    
                    # Process results
                    for i, (frame_id, original_frame, pos_ms) in enumerate(batch_metadata):
                        if i < len(results):
                            result = results[i]
                            # Process single result (simplified)
                            processed_frame, half_area, piece_area = self._process_inference_result(
                                original_frame, result, HALF_IDS, OBSCURED_IDS, PIECE_IDS, SHELL_IDS
                            )
                            
                            # Send result back
                            result_data = (frame_id, processed_frame, half_area, piece_area, pos_ms)
                            try:
                                self.result_queue.put(result_data, timeout=0.1)
                            except queue.Full:
                                break  # Skip if result queue full
                        
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"GPU {self.inference_gpu} inference error: {e}")
                    continue
                    
        except Exception as e:
            print(f"GPU {self.inference_gpu} inference worker failed: {e}")
    
    def _preprocess_frame_gpu0(self, frame, frame_id):
        """Preprocess frame on GPU 0 (decode GPU)."""
        try:
            # ROI extraction
            from main import ROI_X, ROI_Y, ROI_W, ROI_H, MODEL_SIZE
            
            H, W = frame.shape[:2]
            x1, y1 = ROI_X, ROI_Y
            x2, y2 = min(ROI_X + ROI_W, W), min(ROI_Y + ROI_H, H)
            
            if x1 >= x2 or y1 >= y2:
                return None
            
            # Extract ROI and resize
            crop = frame[y1:y2, x1:x2]
            resized = cv2.resize(crop, (MODEL_SIZE, MODEL_SIZE), interpolation=cv2.INTER_LINEAR)
            
            return resized
            
        except Exception as e:
            print(f"Preprocessing error for frame {frame_id}: {e}")
            return None
    
    def _process_inference_result(self, frame, result, HALF_IDS, OBSCURED_IDS, PIECE_IDS, SHELL_IDS):
        """Process inference result (simplified version)."""
        # This would use the existing process_frame logic
        # For now, return dummy values
        return frame, 0.0, 0.0
    
    def get_result(self, timeout=0.1):
        """Get processed result."""
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None