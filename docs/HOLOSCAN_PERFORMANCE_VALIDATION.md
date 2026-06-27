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

That run did not sustain 30 fps. Headless rendering was subsequently disabled
because the sink discards the annotated pixels.

## Optimized headless run

The one-click validation script then completed a second 450-frame,
complexity-1 run:

- process exit code `0`
- pose coverage `449/450` (`99.78%`)
- processed rate `33.93 fps`
- first Stage1 output at frame `43`
- first Stage2 output at frame `412`
- final Stage2 buffer `120/120`
- pose timing: p50 `23.62 ms`, p95 `28.32 ms`, p99 `32.16 ms`
- predictor timing: p50 `0.54 ms`, p95 `3.47 ms`, p99 `7.16 ms`
- operator timing: p50 `25.11 ms`, p95 `31.03 ms`, p99 `33.91 ms`
- source-to-graph timing: p50 `25.53 ms`, p95 `31.68 ms`,
  p99 `34.57 ms`
- no graph, camera, MediaPipe, or inference-worker failure

The short-run average throughput exceeded 30 fps and the operator p95 stayed
within the 33.33 ms frame period. Operator p99 was `33.91 ms`, so this does not
establish that every frame meets a strict 30 fps deadline. It is also not a
sustained soak or target-device result.

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

- run direct V4L2 acquisition on the target Linux/Jetson device;
- measure power mode, clocks, temperature, memory, drops, and a sustained soak;
- measure capture timestamps at the sensor boundary before making
  sensor-to-action latency claims;
- compare the selected pose configuration against a labeled replay before
  treating its speed/coverage tradeoff as a quality-preserving release.
