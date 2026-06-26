# Unified Web Launcher

## Purpose

The Unified Web Launcher runs the bus-stop and intersection cyclist-prediction
pipelines from a local browser. It uses a fixed, single-screen English interface
with bicycle-only iconography and no page or sidebar scrolling.

The launcher does not upload video. Camera frames, pose estimation, model
inference, JPEG encoding, and browser delivery remain on the host machine.

## Start the application

### Windows

Double-click `start_unified_web.bat` in the repository root.

The first run creates `.venv` with Python 3.11 when available and installs
`requirements-unified.txt`. Later runs reuse the environment. Keep the terminal
window open while the console is running.

### Linux

Mark the launcher executable once:

```bash
chmod +x start_unified_web.sh
```

Then double-click `start_unified_web.sh` and select **Run as a program**, or run:

```bash
./start_unified_web.sh
```

The Linux environment uses `--system-site-packages` so Jetson installations can
reuse NVIDIA PyTorch, CUDA, RealSense, and related platform packages.

### Manual start

```bash
python unified_prediction/run_web.py
```

Options:

```text
--host HOST       Bind address; default 127.0.0.1
--port PORT       HTTP port; default 8765
--no-browser      Do not open the system browser automatically
```

Use `--host 0.0.0.0` only on a trusted network. The local server has no user
authentication or TLS termination.

## Interface

The left panel configures:

- prediction scenario: bus stop or intersection;
- pose engine: MediaPipe or trt_pose;
- camera: standard OpenCV camera or Intel RealSense;
- camera index, requested width, height, and frame rate;
- intersection checkpoint fold and compute device;
- optional bus-stop upper-body direction-label swap.

The right panel contains:

- the latest annotated camera frame;
- pipeline state and measured processing FPS;
- cyclist/pose detection state;
- model output lines;
- temporal sequence-buffer progress.

The browser requests the latest available JPEG frame from `/api/frame`. This is
deliberately independent of inference cadence: slow model inference lowers the
display update rate instead of accumulating an unbounded browser-frame queue.

## Runtime states

| State | Meaning |
|---|---|
| `idle` | No camera or model pipeline is active |
| `starting` | Dependencies, model checkpoints, pose engine, or camera are loading |
| `running` | Frames are being captured and processed |
| `stopping` | The stop event has been sent and resources are closing |
| `error` | Startup or runtime failed; the UI displays the exception message |

## Dependency compatibility

The MediaPipe adapter uses the BlazePose Solutions API. The supported x86/Windows
package is `mediapipe==0.10.21`; newer Tasks-only packages such as 0.10.35 do not
provide `mediapipe.solutions`. The unified requirements also keep NumPy below 2
and use the matching OpenCV contrib package.

Jetson/aarch64 environments use the platform-specific MediaPipe path declared in
`requirements-unified.txt` and may still require the setup procedure in
`cyclist_inference/setup.sh` or `pose_detection/setup.sh`.

## Troubleshooting

### `No module named torch`

Close a manually started server and use the platform launcher. It creates and
populates `.venv`. On Windows, a manual repair is:

```powershell
.venv\Scripts\python -m pip install -r requirements-unified.txt
```

### `mediapipe has no attribute solutions`

The wrong MediaPipe generation is installed. Run the platform launcher again,
or repair Windows manually:

```powershell
.venv\Scripts\python -m pip install --force-reinstall "mediapipe==0.10.21" "numpy<2" "opencv-contrib-python<4.12"
```

### GBK codec decode error

The model configuration is UTF-8 and current loaders explicitly request that
encoding. Restart the launcher to use the updated code.

### No video after Start

1. Read the red error text under the Start/Stop controls.
2. Verify the selected camera ID is not in use by another program.
3. Select **Standard camera** for a normal USB/integrated webcam.
4. Select **Intel RealSense** only when `pyrealsense2` and the device are present.
5. Use `Ctrl+F5` once if an older browser asset remains cached.

The UI shows the first frame only after the model, pose engine, and camera have
started successfully. Model loading can take noticeably longer than page load.

### Interface appears clipped

The console targets desktop displays at least 900 pixels wide and uses the
available viewport without document scrolling. Use the browser at 100% zoom on
small displays.

## Stop and shutdown

Use **Stop** to release the camera and pose engine while keeping the web console
open. Close the terminal or press `Ctrl+C` to stop the local HTTP server.
