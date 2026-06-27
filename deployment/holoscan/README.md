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

## Docker Desktop webcam bridge

Docker Desktop on Windows does not expose a Windows webcam as `/dev/video0`
inside its Linux VM. Run the local bridge from PowerShell:

```powershell
.\.venv\Scripts\python.exe deployment\holoscan\camera_bridge.py `
  --webcam-index 0 --width 640 --height 480 --fps 30
```

The bridge binds to `127.0.0.1`, serves only an in-memory MJPEG stream, and does
not record frames. Docker Desktop makes it available to a container at
`host.docker.internal`.

For a headless official Holoscan container, MediaPipe installs a GUI OpenCV
wheel that needs `libGL.so.1`. Replace it with the headless wheel before
starting the graph:

```bash
python3 -m pip install "numpy<2" "mediapipe==0.10.21" \
  "onnxruntime>=1.27,<2"
python3 -m pip uninstall -y opencv-python opencv-contrib-python
python3 -m pip install "opencv-contrib-python-headless<4.12"

python3 deployment/holoscan/app.py \
  --model bus \
  --runtime onnx \
  --camera stream \
  --camera-url http://host.docker.internal:8890/stream.mjpg \
  --pose mediapipe \
  --headless
```

For a non-headless Linux installation, use
`deployment/holoscan/requirements.txt` and provide the normal OpenCV GUI system
libraries.

The reproducible container packages those headless dependencies:

```powershell
docker build -t cyclist-holoscan:3.11 `
  -f deployment/holoscan/Dockerfile deployment/holoscan
```

For a complete 450-frame Windows camera validation, double-click
`validate_live_camera.bat` or run:

```powershell
deployment\holoscan\validate_live_camera.ps1
```

The script builds the image if needed, starts and stops the camera bridge,
runs the real MediaPipe/ONNX graph, and writes a JSON timing summary under
`release/`. It does not save camera images or video.

On a supported Linux/Jetson system with `/dev/video0`, run:

```bash
bash deployment/holoscan/validate_v4l2.sh 0 450 30
```

That command is an acceptance-test entry point, not evidence that this
repository has executed on Jetson hardware.

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

MediaPipe complexity defaults to `1`. Use `--mediapipe-complexity 0|1|2` to
change it. Complexity 0 was faster but failed the pose-coverage gate in the
recorded webcam test; complexity 2 was substantially slower. Headless mode
skips skeleton and text rendering because the sink discards annotated pixels.
Every JSONL record includes separate pose, predictor, operator, and
source-to-graph timing fields.

See `docs/HOLOSCAN_SIMULATION_VALIDATION.md` for the recorded container,
hardware, commands, verified synthetic and real-model outputs, and explicit
validation boundaries. The shipped ONNX files were checked against the source
TorchScript files; see `docs/MODEL_EXPORT_EQUIVALENCE.md`.
Measured desktop timing and its limitations are documented in
`docs/HOLOSCAN_PERFORMANCE_VALIDATION.md`.

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

The bus ONNX predictor has been executed with both deterministic poses and a
Windows webcam/MediaPipe stream in the official Holoscan 3.11 dGPU container.
Direct V4L2 camera acquisition, Jetson deployment, latency, power, thermal, and
throughput still require target-device validation and are not asserted by this
repository.
