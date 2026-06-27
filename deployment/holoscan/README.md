# Holoscan Cyclist Prediction Graph

This graph implements `camera -> pose/model inference -> display + structured
prediction publication` using the existing unified predictors.

## Platform requirement

The NVIDIA Holoscan runtime is not available as a native Windows Python wheel.
Run this graph on a supported Linux x86_64/CUDA system or Jetson. Windows users
should run the standard unified web launcher instead:

```powershell
.\.venv\Scripts\python.exe unified_prediction\run_web.py
```

Install the graph on the supported Linux/Jetson environment:

```bash
python3 -m pip install -r deployment/holoscan/requirements.txt
python3 deployment/holoscan/app.py \
  --model bus \
  --runtime onnx \
  --onnx-model-dir model_package/onnx \
  --pose mediapipe
```

The ONNX bus-model path runs inference in a spawned Python process. This is
intentional: calling PyTorch or ONNX Runtime directly from the GXF Python
operator worker caused a native Python thread-state crash in Holoscan 3.11.
The graph uses synchronous, bounded IPC so frame and prediction ordering remain
deterministic. `--onnx-worker-timeout` controls the startup and inference
timeout (default: 120 seconds). ONNX is the default bus runtime and
`model_package/onnx` is the default model directory; the flags are shown
explicitly above for reproducibility.

## Headless simulation

Use the official CUDA 12 dGPU container on an Ubuntu workstation or cloud GPU:

```bash
docker pull nvcr.io/nvidia/clara-holoscan/holoscan:v3.11.0-cuda12-dgpu
docker run --rm --gpus all -it \
  -v "$PWD:/workspace" \
  -w /workspace \
  nvcr.io/nvidia/clara-holoscan/holoscan:v3.11.0-cuda12-dgpu
```

Inside the container, install the application dependencies and run the graph
without a camera or display:

```bash
python3 -m pip install -r requirements-unified.txt
python3 deployment/holoscan/app.py \
  --model synthetic \
  --camera synthetic \
  --pose synthetic \
  --headless \
  --duration 15 \
  --torch-threads 1 \
  --output-jsonl holoscan_predictions.jsonl
```

The synthetic source runs at the requested `--fps`, supplies deterministic
normalized 33-point poses, stops automatically, and writes one structured JSON
record per frame. It validates graph composition, message flow, synthetic
predictor state progression, and output serialization. It does not
execute the packaged Torch models and is not camera, pose-estimation, accuracy,
latency, power, thermal, or production-hardware evidence.

`--torch-threads` applies only to the direct PyTorch runtime. The validated bus
path is `--runtime onnx`.

See `docs/HOLOSCAN_SIMULATION_VALIDATION.md` for the recorded container,
hardware, commands, verified synthetic and real-model outputs, and explicit
validation boundaries. The shipped ONNX files were checked against the source
TorchScript files; see `docs/MODEL_EXPORT_EQUIVALENCE.md`.

On Windows, an activated virtual environment provides `python.exe`, not
necessarily `python3.exe`. Therefore `python3` can resolve to the Microsoft
Store interpreter outside the virtual environment. Use
`.\.venv\Scripts\python.exe` explicitly for Windows commands.

Press `q` in the display window to stop. Select `--pose trt` only when the
Jetson trt_pose dependencies and model are installed.

The graph publishes an annotated frame and a structured prediction dictionary
on separate output ports. It intentionally keeps acquisition, inference, and
display as separate operators so bounded queues, recorders, network transmitters,
or Holoviz can replace individual stages without changing model semantics.

The bus ONNX predictor and deterministic simulation inputs have been executed
in the official Holoscan 3.11 dGPU container. Camera acquisition, MediaPipe,
Jetson deployment, latency, power, thermal, and throughput still require
target-device validation and are not asserted by this repository.
