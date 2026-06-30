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
sh edge_deployment/holoscan/start_holoscan.sh \
  --model bus \
  --camera realsense \
  --pose mediapipe
```

On JetPack 5/Python 3.8 the launcher creates an isolated environment with the
last compatible official SDK wheel, Holoscan 0.6.0. It keeps JetPack-provided
PyTorch, MediaPipe, CUDA, and camera packages available through
`--system-site-packages`. Newer CUDA 12 platforms install the current
`holoscan-cu12` SDK line from the same requirements file.

The default camera is an Intel RealSense color stream, opened through
`pyrealsense2`. For a regular UVC webcam, use
`--camera webcam --webcam-index 0`. RealSense exposes several `/dev/video*`
subdevices, so opening `/dev/video0` directly with OpenCV is not reliable.

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
python3 edge_deployment/holoscan/app.py \
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

`--torch-threads` applies only to the real `bus` and `intersection` model modes.

See `docs/HOLOSCAN_SIMULATION_VALIDATION.md` for the recorded container,
hardware, command, verified outputs, and explicit validation boundary.

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

This is a reference graph whose Python source is syntax-checked in CI. Successful
execution and latency, power, thermal, and throughput results require validation
on the target Linux/Jetson device and are not asserted by this repository.
