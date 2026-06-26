# Holoscan Headless Simulation Validation

## Status

The deterministic synthetic graph completed successfully on 2026-06-27. This
is runtime evidence for Holoscan graph composition and message flow, not
production-model or target-device performance evidence.

## Environment

- Docker Desktop engine 29.5.3 with WSL2
- NVIDIA GeForce RTX 3070 Laptop GPU, 8,192 MiB
- Windows driver 566.36, CUDA capability reported as 12.7
- Holoscan SDK 3.11.0 CUDA 12 dGPU container
- image digest:
  `sha256:bbb62e52a661693ffc3d7804657e001c86e30f2b25fb46b092b546982e62c232`

The container reported CUDA minor-version compatibility mode because it was
built with CUDA 12.8 while the host driver reported CUDA 12.7 support.

## Command

```bash
python3 deployment/holoscan/app.py \
  --model synthetic \
  --camera synthetic \
  --pose synthetic \
  --headless \
  --duration 15 \
  --output-jsonl release/holoscan-simulation.jsonl
```

## Verified results

- process exit code: `0`
- structured JSONL records: `450`
- frame index range: `0..449`, continuous
- source-time span: `14.967 s`
- pose status: all records reported `pose_ok=true`
- predictor states observed: `warming_up`, `straight`, `yield`, `overtake`
- final progress: `30/30`
- scheduler reported graph execution finished and destroyed its context

## Boundary

This run validates the synthetic source, synthetic pose adapter, synthetic
predictor, operator connections, fixed-count scheduling, headless sink, and
JSONL serialization. It does not validate camera acquisition, MediaPipe,
trt_pose, packaged Torch model inference, prediction accuracy, real-time
latency, power, thermal behavior, or Jetson deployment.

An attempted packaged bus-model run in this specific Holoscan 3.11 container
encountered a native PyTorch/Python crash when Torch APIs were called from the
GXF worker thread. The standalone TorchScript checkpoints loaded and executed
outside that worker. Production integration therefore still requires a
target-supported Torch/LibTorch configuration or isolation of model inference
from the GXF Python worker.
