# Multi-GPU Support Guide

## Overview
The video inference pipeline now supports multi-GPU processing to maximize throughput on systems with multiple CUDA devices.

## Features
- **Automatic GPU Detection**: Detects all available CUDA GPUs
- **Load Balancing**: Distributes batch processing across all GPUs
- **Per-GPU Workers**: Each GPU gets its own worker thread with dedicated model
- **Memory Optimization**: Each GPU manages its own memory and cache
- **Batch Processing**: Large batches are distributed across GPUs for maximum efficiency

## Usage

### Single GPU (Default)
```bash
python main.py --source video.mp4 --parallel --nvdec
```

### Multi-GPU (Dual Tesla P40)
```bash
python main.py --source video.mp4 --parallel --multi-gpu --nvdec
```

### Optimized for Dual Tesla P40 Setup
```bash
python main.py --source video.mp4 \
    --parallel \
    --multi-gpu \
    --nvdec \
    --nvenc \
    --batch-size 0 \
    --max-batch-size 256 \
    --queue-size 50
```

## Command Line Options

- `--multi-gpu`: Enable multi-GPU processing (requires --parallel)
- `--batch-size 0`: Auto-detect optimal batch size per GPU (recommended)
- `--max-batch-size 256`: Maximum batch size to try (good for high-memory GPUs)
- `--queue-size 50`: Frame queue size (larger for multi-GPU setups)

## Performance Benefits

### Expected Performance on Dual Tesla P40:
- **Single GPU**: ~250 batch size, 1 GPU worker
- **Multi-GPU**: ~250 batch size per GPU, 2 GPU workers
- **Theoretical Speedup**: Up to 2x throughput with proper load balancing

### Hardware Acceleration Stack:
1. **NVDEC**: Hardware video decoding (CPU offload)
2. **Multi-GPU**: Parallel inference processing
3. **NVENC**: Hardware video encoding (CPU offload)
4. **FP16**: Half precision for 2x GPU speed

## Testing

Run the test script to verify your setup:
```bash
python test_multi_gpu.py
```

Expected output for dual Tesla P40:
```
=== Multi-GPU Detection Test ===
Optimal device: cuda:0
CUDA available: 2 device(s). Multi-GPU setup detected.
  GPU 0: Tesla P40 (23.9 GB)
  GPU 1: Tesla P40 (23.9 GB)

GPU devices found: 2
  GPU 0: {'id': 0, 'name': 'Tesla P40', 'memory_gb': 23.9, 'device_str': 'cuda:0'}
  GPU 1: {'id': 1, 'name': 'Tesla P40', 'memory_gb': 23.9, 'device_str': 'cuda:1'}

✅ Multi-GPU setup detected! 2 GPUs available
   You can use --multi-gpu flag to enable distributed processing
```

## Architecture

### Multi-GPU Processing Flow:
1. **Frame Producer**: Single thread reads video frames with NVDEC
2. **GPU Workers**: One worker per GPU, each with dedicated model
3. **Batch Distribution**: Frames distributed across GPU queues
4. **Parallel Inference**: Each GPU processes its batch independently
5. **Result Collector**: Results merged in original frame order
6. **Video Writer**: Single thread writes output with NVENC

### Memory Management:
- Each GPU loads its own model copy
- Independent memory management per GPU
- Automatic cache clearing on OOM
- FP16 precision for memory efficiency

## Troubleshooting

### If multi-GPU isn't working:
1. Check GPU detection: `python test_multi_gpu.py`
2. Verify CUDA installation: `nvidia-smi`
3. Check PyTorch CUDA: `python -c "import torch; print(torch.cuda.device_count())"`

### Performance optimization:
- Use `--batch-size 0` for auto-detection
- Increase `--queue-size` for multi-GPU setups
- Monitor GPU utilization with `nvidia-smi`
- Enable all hardware acceleration: `--nvdec --nvenc`

### Memory issues:
- Reduce `--max-batch-size` if OOM occurs
- Check GPU memory with `nvidia-smi`
- Each GPU needs ~8GB for large batches