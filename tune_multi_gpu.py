#!/usr/bin/env python3
"""
Multi-GPU Performance Tuner
Automatically finds optimal settings for your dual Tesla P40 setup.
"""

import subprocess
import sys
import time
import re
import os

def run_test(batch_size, queue_size, duration=30):
    """Run a short test and measure performance."""
    cmd = [
        sys.executable, "main.py",
        "--source", "/mnt/nfs_share/camera1/camera1_auto_blower_separator_20251020_102330.mp4",
        "--parallel",
        "--multi-gpu", 
        "--nvdec",
        "--nvenc",
        "--batch-size", str(batch_size),
        "--queue-size", str(queue_size),
        "--max-batch-size", str(batch_size)
    ]
    
    print(f"Testing: batch_size={batch_size}, queue_size={queue_size}")
    
    try:
        start_time = time.time()
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        frames_processed = 0
        last_frame = 0
        
        # Monitor for specified duration
        while time.time() - start_time < duration:
            line = process.stdout.readline()
            if not line:
                break
                
            # Look for progress updates
            match = re.search(r'Frame (\d+)/\d+', line)
            if match:
                current_frame = int(match.group(1))
                if current_frame > last_frame:
                    frames_processed = current_frame
                    last_frame = current_frame
            
            # Print interesting lines
            if any(keyword in line for keyword in ['GPU', 'Processing:', 'ERROR', 'memory']):
                print(f"  {line.strip()}")
        
        # Kill the process
        process.terminate()
        process.wait(timeout=5)
        
        elapsed = time.time() - start_time
        fps = frames_processed / elapsed if elapsed > 0 else 0
        
        print(f"  Result: {frames_processed} frames in {elapsed:.1f}s = {fps:.1f} FPS\n")
        return fps
        
    except Exception as e:
        print(f"  Error: {e}\n")
        return 0

def main():
    print("=== Multi-GPU Performance Tuner ===")
    print("Testing different batch sizes and queue sizes...")
    print()
    
    # Test different configurations
    configs = [
        (64, 100),   # Small batches
        (128, 200),  # Medium batches  
        (192, 300),  # Large batches
        (256, 400),  # Max batches
    ]
    
    results = []
    
    for batch_size, queue_size in configs:
        fps = run_test(batch_size, queue_size, duration=45)
        results.append((batch_size, queue_size, fps))
        
        if fps == 0:
            print(f"Skipping further tests due to error with batch_size={batch_size}")
            break
        
        # Give GPU some time to cool down
        time.sleep(5)
    
    # Find best configuration
    if results:
        best = max(results, key=lambda x: x[2])
        batch_size, queue_size, fps = best
        
        print("=== RESULTS ===")
        for bs, qs, f in results:
            marker = " ← BEST" if (bs, qs) == (batch_size, queue_size) else ""
            print(f"batch_size={bs:3d}, queue_size={qs:3d}: {f:5.1f} FPS{marker}")
        
        print(f"\nOptimal settings for your dual Tesla P40 setup:")
        print(f"--batch-size {batch_size} --queue-size {queue_size}")
        print(f"Expected performance: {fps:.1f} FPS")
        
        # Generate optimized command
        print(f"\nOptimized command:")
        print(f"python main.py --source VIDEO.mp4 \\")
        print(f"    --parallel --multi-gpu --nvdec --nvenc \\")
        print(f"    --batch-size {batch_size} --queue-size {queue_size}")
    else:
        print("No successful tests completed!")

if __name__ == "__main__":
    main()