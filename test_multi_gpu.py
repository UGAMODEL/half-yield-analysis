#!/usr/bin/env python3
"""
Test script for multi-GPU support detection.
Run this to verify your dual Tesla P40 setup is detected properly.
"""

import sys
import os

# Add current directory to path to import main.py functions
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from main import get_gpu_devices, get_optimal_device
    
    print("=== Multi-GPU Detection Test ===")
    
    # Test device detection
    device = get_optimal_device()
    print(f"Optimal device: {device}")
    
    # Test GPU enumeration
    gpu_devices = get_gpu_devices()
    print(f"\nGPU devices found: {len(gpu_devices)}")
    
    if gpu_devices:
        for i, gpu_info in enumerate(gpu_devices):
            print(f"  GPU {i}: {gpu_info}")
        
        if len(gpu_devices) > 1:
            print(f"\n✅ Multi-GPU setup detected! {len(gpu_devices)} GPUs available")
            print("   You can use --multi-gpu flag to enable distributed processing")
        else:
            print(f"\n⚠️  Only {len(gpu_devices)} GPU detected")
    else:
        print("\n❌ No CUDA GPUs detected")
    
    print("\n=== Test Complete ===")
    
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're in the correct directory with main.py")
except Exception as e:
    print(f"Error during test: {e}")