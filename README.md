# Cyclist Behavior Prediction Workspace

[![License](https://img.shields.io/badge/code-Apache--2.0-blue.svg)](LICENSE)

This repository contains real-time cyclist behavior prediction pipelines for:

- `unified_prediction/`: shared real-time app with one camera/pose/display loop and selectable predictors.
- `cyclist_inference/`: intersection cyclist intention prediction.
- `model_package/`: bus-stop cyclist behavior prediction using TorchScript models.
- `pose_detection/`: RealSense/trt_pose pose detection utilities and model files.
- `cyclistprediction/`: source training/inference code and lightweight trained weights needed by `cyclist_inference`.

Large raw/training datasets may be present in a local workspace, but remain
excluded from Git. Virtual environments, generated videos, and old nested Git
histories are also intentionally excluded from this flattened repository.

## Unified Launcher

### Web launcher (recommended)

Run the local web console with a parameter panel and live annotated preview:

```bash
python3 unified_prediction/run_web.py
```

Or launch it by double-clicking the platform script in the repository root:

- Windows: `start_unified_web.bat`
- Linux: `start_unified_web.sh`

On first launch, the script creates `.venv` and installs the packages from
`requirements-unified.txt`. Later launches reuse that environment. Internet
access is required only for the first setup. On Jetson, the Linux environment
inherits system packages so it can reuse NVIDIA's PyTorch/CUDA installation.

On Linux, mark the script as executable once after downloading:

```bash
chmod +x start_unified_web.sh
```

Some Linux file managers default to opening shell scripts as text. Select
"Run as a program" when prompted, or enable executable text files in the file
manager preferences.

The browser opens at `http://127.0.0.1:8765`. To access it from another device
on the same trusted network, use `--host 0.0.0.0` and open the machine's LAN IP.

The web console provides an English, single-screen traffic-prediction interface
with model, pose, camera, resolution, frame-rate, fold, and compute-device
controls. The right panel shows annotated camera frames, pose status, current
prediction output, and sequence-buffer progress. See
[`docs/UNIFIED_WEB_LAUNCHER.md`](docs/UNIFIED_WEB_LAUNCHER.md) for operation and
troubleshooting details.

### Desktop launcher

Run this from the repository root to choose a model in a GUI window:

```bash
python3 unified_prediction/run_prediction.py
```

The launcher lets you choose:

- bus-stop or intersection model
- MediaPipe or trt_pose backend
- RealSense or webcam source
- live display or headless video recording
- optional test duration and output file

You can also select a model directly:

```bash
# Bus-stop model: straight / yield / overtake
python3 unified_prediction/run_prediction.py --model bus --pose mediapipe
python3 unified_prediction/run_prediction.py --model bus --pose trt --bus-swap-upper-labels

# Intersection model: Crossing / LeftTurn / RightTurn
python3 unified_prediction/run_prediction.py --model intersection --pose trt

# Use a normal OpenCV webcam instead of RealSense
python3 unified_prediction/run_prediction.py --model bus --camera webcam --webcam-index 0

# SSH/headless test
python3 unified_prediction/run_prediction.py --model bus --headless --duration 30 -o bus_test.avi
python3 unified_prediction/run_prediction.py --model intersection --headless --duration 30 -o intersection_test.avi
```

## Bus-stop Real-time Prediction

```bash
cd model_package
python3 bus_stop_predict.py --pose mediapipe
python3 bus_stop_predict.py --pose trt
python3 bus_stop_predict.py --pose mediapipe --headless --duration 30 -o bus_stop_test.avi
```

If the upper-limb left/right labels appear reversed while head direction is correct:

```bash
python3 bus_stop_predict.py --swap-upper-labels
```

## Intersection Real-time Prediction

```bash
cd cyclist_inference
python3 cyclist_predict.py --pose trt
python3 cyclist_predict.py --pose mediapipe
python3 cyclist_predict.py --pose trt --headless --duration 30 -o intersection_test.avi
```

## Local datasets

The current workspace includes ignored cyclist skeleton, video, parsed trial,
and result data under `cyclistprediction/`. These files are intentionally not
tracked because of their size. See
[`docs/DATASET_INVENTORY.md`](docs/DATASET_INVENTORY.md) for the observed local
inventory, schemas, intended roles, and handling constraints.

## Public dataset release

The sanitized 2D skeleton subset is published on Hugging Face:

- [Kheyro/cyclist-intention-2d-keypoints](https://huggingface.co/datasets/Kheyro/cyclist-intention-2d-keypoints)
- CC BY 4.0
- V5 release: 32 participants, 559 scene clips, and 99,734 frames
- participant-disjoint train, validation, and test splits
- Stage1 gesture intervals plus Stage2 `straight`, `yield`, and `overtake` labels
- no raw video, original filenames, calendar dates, session IDs, source paths,
  or direct participant IDs

The reproducible V5 builder is `tools/build_hf_dataset_v5.py`. Release construction,
quality gates, privacy controls, and publication details are documented in
[`docs/PUBLIC_DATASET_RELEASE.md`](docs/PUBLIC_DATASET_RELEASE.md).

## Holoscan graph

A functional Holoscan application is provided under `deployment/holoscan/`. It
separates camera acquisition, pose/model inference, and display/structured
prediction publication into operators:

```bash
python3 -m pip install -r deployment/holoscan/requirements.txt
python3 deployment/holoscan/app.py --model bus --pose mediapipe
```

See [`deployment/holoscan/README.md`](deployment/holoscan/README.md). Hardware
performance claims require measurement on the target Jetson and are not inferred
from desktop execution.

## License

Source code is licensed under the Apache License 2.0. Dataset releases are
licensed separately under CC BY 4.0. See `LICENSE` and `NOTICE`.
