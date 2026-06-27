# Holoscan Headless Simulation Validation

## Status

The deterministic synthetic graph and the packaged bus ONNX predictor both
completed successfully on 2026-06-27. This is runtime evidence for graph
composition, message flow, and real model execution. It is not target-device
performance or prediction-accuracy evidence.

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

## Real bus-model validation

The source and pose were kept deterministic so this run isolates the packaged
model and graph integration:

```bash
python3 deployment/holoscan/app.py \
  --model bus \
  --runtime onnx \
  --onnx-model-dir model_package/onnx \
  --camera synthetic \
  --pose synthetic \
  --headless \
  --duration 15 \
  --fps 30 \
  --output-jsonl release/holoscan-onnx-real-model.jsonl
```

Verified results:

- process exit code: `0`
- structured JSONL records: `450`, continuous frame range `0..449`
- pose status: `450/450` records reported `pose_ok=true`
- first Stage1 prediction: frame `39`
- Stage2 feature buffer reached `120/120`
- first Stage2 prediction: frame `396`, `overtake (0.81)`
- final Stage2 prediction: `overtake (0.83)`
- scheduler reported graph execution finished and destroyed its context
- no traceback, segmentation fault, timeout, or worker-process failure

These labels and confidence values come from deterministic synthetic poses and
are integration fixtures, not accuracy measurements.

## Live camera and MediaPipe validation

Docker Desktop did not expose the Windows camera as `/dev/video0`. A local
host-only MJPEG bridge supplied frames through
`http://host.docker.internal:8890/stream.mjpg`. No video or image frames were
recorded by the validation.

```bash
python3 deployment/holoscan/app.py \
  --model bus \
  --runtime onnx \
  --camera stream \
  --camera-url http://host.docker.internal:8890/stream.mjpg \
  --pose mediapipe \
  --duration 15 \
  --fps 30 \
  --headless \
  --output-jsonl release/holoscan-live-camera.jsonl
```

The configured duration corresponds to 450 requested input frames. It is not a
wall-clock performance measurement.

- process exit code: `0`
- structured JSONL records: `450`, continuous frame range `0..449`
- pose status: `450/450` records reported `pose_ok=true`
- first Stage1 prediction: frame `20`
- Stage2 feature buffer reached `120/120`
- first Stage2 prediction: frame `184`
- observed timestamp span: `39.677 s` (`11.32` processed frames/s)
- scheduler and isolated ONNX worker exited normally
- no traceback, segmentation fault, timeout, or camera read failure

The observed prediction was an uncontrolled webcam integration output. It is
not a ground-truth label or an accuracy result. The processed frame rate is
reported for transparency and is not presented as a latency or throughput
benchmark.

## Boundary

The runs validate the synthetic source and pose adapter, a Windows webcam
transported through the local bridge, MediaPipe pose extraction, packaged bus
ONNX models, process-isolated inference bridge, operator connections,
fixed-count scheduling, headless sink, and JSONL serialization. They do not
validate direct V4L2 camera acquisition, trt_pose, direct PyTorch inference
inside a GXF worker, prediction accuracy, real-time latency, power, thermal
behavior, or Jetson deployment.

Direct PyTorch and ONNX Runtime calls from the GXF Python worker both produced
a native Python thread-state crash at the first Stage1 inference. The validated
ONNX path fixes that integration issue by running ONNX Runtime in a spawned
process and exchanging ordered requests through bounded synchronous queues.
