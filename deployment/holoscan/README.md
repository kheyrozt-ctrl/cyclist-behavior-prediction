# Holoscan Cyclist Prediction Graph

This graph implements `camera -> pose/model inference -> display + structured
prediction publication` using the existing unified predictors.

Install on a supported Linux/Jetson Holoscan environment:

```bash
python3 -m pip install -r deployment/holoscan/requirements.txt
python3 deployment/holoscan/app.py --model bus --pose mediapipe
```

Press `q` in the display window to stop. Select `--pose trt` only when the
Jetson trt_pose dependencies and model are installed.

The graph publishes an annotated frame and a structured prediction dictionary
on separate output ports. It intentionally keeps acquisition, inference, and
display as separate operators so bounded queues, recorders, network transmitters,
or Holoviz can replace individual stages without changing model semantics.

This implementation is a functional graph. Jetson latency, power, thermal, and
throughput claims require measurement on the target device and are not asserted
by this repository.
