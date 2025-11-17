#!/usr/bin/env python3
"""
Test script to verify device detection and acceleration support.
"""

import sys
import platform

def test_device_detection():
    """Test device detection functionality."""
    print(f"System: {platform.system()} {platform.machine()}")
    print(f"Python: {sys.version}")
    
    # Test PyTorch availability
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        
        # Test CUDA
        if torch.cuda.is_available():
            print(f"CUDA available: YES")
            print(f"CUDA devices: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("CUDA available: NO")
        
        # Test MPS (Apple Silicon)
        if hasattr(torch.backends, 'mps'):
            if torch.backends.mps.is_available():
                print("MPS (Metal Performance Shaders) available: YES")
            else:
                print("MPS (Metal Performance Shaders) available: NO")
        else:
            print("MPS (Metal Performance Shaders): Not supported in this PyTorch version")
        
        # Test CPU
        print(f"CPU threads: {torch.get_num_threads()}")
        
        # Show recommended device
        from main import get_optimal_device
        optimal_device = get_optimal_device()
        print(f"Recommended device: {optimal_device}")
        
        # Test tensor creation on each available device
        devices_to_test = ["cpu"]
        if torch.cuda.is_available():
            devices_to_test.append("cuda:0")
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            devices_to_test.append("mps")
        
        print("\nTesting tensor operations:")
        for device in devices_to_test:
            try:
                x = torch.randn(100, 100, device=device)
                y = torch.randn(100, 100, device=device)
                z = torch.mm(x, y)
                print(f"  {device}: OK (tensor shape: {z.shape})")
            except Exception as e:
                print(f"  {device}: FAILED ({e})")
        
    except ImportError:
        print("PyTorch not available")
    
    # Test Ultralytics
    try:
        from ultralytics import YOLO
        print("Ultralytics YOLO: Available")
    except ImportError:
        print("Ultralytics YOLO: Not available")

if __name__ == "__main__":
    test_device_detection()