#!/usr/bin/env python3
"""
Test script for yield analysis functionality
"""

import sys
from collections import deque
from main import calculate_yield_statistics, save_yield_report

def test_yield_analysis():
    """Test the yield analysis functions with sample data"""
    
    # Create sample area history data (t_ms, half_area, piece_area)
    # Simulating 30 seconds of data with varying yields
    area_hist = deque()
    
    # First 10 seconds: 70% yield (high half content)
    for i in range(10):
        t_ms = i * 1000.0  # 1 second intervals
        half_area = 700.0  # high half
        piece_area = 300.0  # low piece
        area_hist.append((t_ms, half_area, piece_area))
    
    # Next 10 seconds: 50% yield (equal half/piece)  
    for i in range(10, 20):
        t_ms = i * 1000.0
        half_area = 500.0
        piece_area = 500.0
        area_hist.append((t_ms, half_area, piece_area))
    
    # Last 10 seconds: 30% yield (low half content)
    for i in range(20, 30):
        t_ms = i * 1000.0
        half_area = 300.0
        piece_area = 700.0
        area_hist.append((t_ms, half_area, piece_area))
    
    print("Testing yield analysis with sample data...")
    print("Sample data: 30 seconds, varying from 70% -> 50% -> 30% yield")
    
    # Calculate statistics
    stats = calculate_yield_statistics(area_hist, bin_duration_sec=10.0)
    
    print(f"Expected overall yield: ~50% (average of 70%, 50%, 30%)")
    print(f"Calculated net average: {stats['net_average']:.4f} ({stats['net_average']*100:.2f}%)")
    
    # Generate report
    save_yield_report(stats, "test_yield_analysis.csv", "test_video.mp4")
    
    # Verify binned data
    print(f"\nBinned data verification:")
    expected_yields = [0.70, 0.50, 0.30]
    for i, (bin_start, bin_yield, bin_half, bin_piece) in enumerate(stats['binned_data']):
        expected = expected_yields[i] if i < len(expected_yields) else 0.0
        print(f"Bin {i}: Expected {expected:.2f}, Got {bin_yield:.4f} - {'✓' if abs(bin_yield - expected) < 0.01 else '✗'}")

if __name__ == "__main__":
    test_yield_analysis()