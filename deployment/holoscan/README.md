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
python3 deployment/holoscan/app.py --model bus --pose mediapipe
```

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
