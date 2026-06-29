# Open-Source Training And Inference Code

This directory contains the Apache-2.0 code release.

- `training/intersection_intention_legacy/`: original intersection gesture-to-maneuver training lineage.
- `training/bus_stop_v5_public/`: V5 public-training and combined bus-stop model lineage.
- `inference/intersection_runtime/`: original real-time intersection inference support.
- `inference/unified_prediction/`: unified prediction runtime and web launcher.
- `model_artifacts/bus_stop_torchscript/`: packaged bus-stop TorchScript model files.
- `pose_backends/trt_pose/`: pose-estimation backend code and metadata.
- `launchers/`: repository-level launcher scripts and requirements.

For the current unified demo, use `inference/unified_prediction/start_web.bat`
on Windows or `inference/unified_prediction/start_web.sh` on Linux/macOS.
