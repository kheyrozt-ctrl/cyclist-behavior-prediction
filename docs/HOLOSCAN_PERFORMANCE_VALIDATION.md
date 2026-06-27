# Holoscan Desktop Performance Validation

## Status

This report records integration throughput and per-stage timing on the tested
Docker Desktop system. It is not Jetson evidence, an accuracy comparison, or a
sensor-to-action latency measurement.

Environment:

- NVIDIA Holoscan SDK 3.11.0 CUDA 12 dGPU container
- NVIDIA GeForce RTX 3070 Laptop GPU
- Windows camera transported through the local MJPEG bridge
- MediaPipe 0.10.21 CPU pose inference
- ONNX Runtime 1.27 CPU model inference in the isolated worker
- 640 × 480 camera stream

## MediaPipe complexity comparison

Each short run requested 120 frames. The person and camera scene were not a
controlled accuracy dataset, so pose coverage is a run-validity signal rather
than a quality score.

| Complexity | Pose coverage | Processed fps | Pose p50 / p95 / p99 (ms) | Operator p50 / p95 / p99 (ms) | Decision |
|---:|---:|---:|---:|---:|---|
| 0 | 21/120 (17.5%) | 37.64 | 17.54 / 21.47 / 25.49 | 22.13 / 27.35 / 30.60 | rejected: insufficient pose coverage |
| 1 | 120/120 (100%) | 27.18 | 23.56 / 35.11 / 45.46 | 30.17 / 44.36 / 53.25 | selected |
| 2 | 119/120 (99.17%) | 11.41 | 75.23 / 88.82 / 91.73 | 82.84 / 94.21 / 100.74 | rejected: slow |

## Full selected-configuration run

The 450-frame complexity-1 run completed with:

- process exit code `0`
- pose coverage `450/450`
- processed rate `27.99 fps`
- first Stage1 output at frame `37`
- first Stage2 output at frame `380`
- final Stage2 buffer `120/120`
- pose timing: p50 `23.59 ms`, p95 `30.92 ms`, p99 `34.81 ms`
- predictor timing: p50 `0.58 ms`, p95 `6.38 ms`, p99 `7.27 ms`
- operator timing: p50 `30.48 ms`, p95 `40.19 ms`, p99 `44.35 ms`
- source-to-graph timing: p50 `30.94 ms`, p95 `40.81 ms`,
  p99 `45.36 ms`
- no graph, camera, MediaPipe, or inference-worker failure

The valid-pose run did not sustain 30 fps and does not meet a strict 30 fps
acceptance gate.

Headless rendering was subsequently disabled because the sink discards the
annotated pixels. A no-pose run then processed 35.86 fps with operator p95
`20.99 ms`; because pose coverage was `0/450`, that result is only an
unloaded-path ceiling and is not a replacement for the valid-pose result.

## Reproduction

On Docker Desktop:

```powershell
deployment\holoscan\validate_live_camera.ps1
```

On Linux/Jetson with V4L2:

```bash
bash deployment/holoscan/validate_v4l2.sh 0 450 30
```

Summarize any JSONL run with:

```bash
python tools/summarize_holoscan_run.py release/holoscan-live-validation.jsonl
```

## Remaining acceptance work

- repeat the optimized headless run with valid pose coverage;
- run direct V4L2 acquisition on the target Linux/Jetson device;
- measure power mode, clocks, temperature, memory, drops, and a sustained soak;
- measure capture timestamps at the sensor boundary before making
  sensor-to-action latency claims;
- compare the selected pose configuration against a labeled replay before
  treating its speed/coverage tradeoff as a quality-preserving release.
