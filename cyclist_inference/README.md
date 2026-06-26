# Cyclist Intention Prediction - Real-Time Inference

Real-time cyclist maneuver prediction on Jetson AGX Orin using an Intel RealSense D435 camera. A two-stage Bi-LSTM pipeline classifies cyclist gestures frame-by-frame, then predicts the intended maneuver (left turn, right turn, or crossing).

## Architecture

```
RealSense D435 (640x480 @ 30fps)
        |
  create_pose_adapter("mediapipe" | "trt")
        |
  PoseAdapter (abstract)
   /          \
MediaPipeAdapter   TrtPoseAdapter
(33 keypoints)     (18 keypoints -> mapped to 33)
        \         /
   33 keypoints -> 66 features
        |
Sliding window buffer (30 frames)
        |
   +----+----+
   |         |
Explicit   Implicit
Gesture    Gesture
Classifier Classifier
(Bi-LSTM)  (Bi-LSTM)
   |         |
   +----+----+
        |
Gesture weight encoding -> gesture buffer (719 entries)
        |
Maneuver Predictor (Bi-LSTM -> LeftTurn / RightTurn / Crossing)
        |
Display overlay with predictions
```

## Pose Estimation Backends

Two backends are available via the `--pose` flag:

### MediaPipe Pose

MediaPipe BlazePose provides 33 keypoints covering the full body including face, hands, and feet. Uses `model_complexity=2` (most accurate) with detection confidence 0.8 and tracking confidence 0.5. Input is the full camera resolution (640x480); coordinates are returned normalized to [0, 1].

### NVIDIA trt_pose

ResNet18 with attention layers, producing 18 COCO keypoints + neck. Optimized with TensorRT FP16 for the Jetson platform. Input is resized to 224x224 with ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]). Post-processing uses ParseObjects on confidence maps and part affinity fields. Coordinates are normalized to [0, 1].

The 18 trt_pose keypoints are mapped to MediaPipe's 33 using direct mapping, duplication, and interpolation (see Keypoint Mapping Methodology below).

### Backend comparison

| | MediaPipe | trt_pose |
|---|---|---|
| Keypoints | 33 (BlazePose) | 18 (COCO + neck) |
| Architecture | BlazePose (MediaPipe) | ResNet18 + attention |
| Inference backend | CPU (MediaPipe) | TensorRT FP16 (GPU) |
| Input resolution | 640x480 (native) | 224x224 |
| Measured FPS | Not benchmarked here | Not benchmarked here |
| Coordinate format | Normalized [0,1] | Normalized [0,1] |

## Keypoint Mapping Methodology

When using `--pose trt`, the 18 trt_pose keypoints are mapped to MediaPipe's 33-keypoint format so the LSTM gesture classifiers receive the same 66-feature input they were trained on.

### Mapping table

| MP Idx | MediaPipe Name | trt_pose Idx | Strategy |
|--------|---------------|-------------|----------|
| 0 | nose | 0 | direct |
| 1 | left_eye_inner | 1 | duplicate (left_eye) |
| 2 | left_eye | 1 | direct |
| 3 | left_eye_outer | 1 | duplicate (left_eye) |
| 4 | right_eye_inner | 2 | duplicate (right_eye) |
| 5 | right_eye | 2 | direct |
| 6 | right_eye_outer | 2 | duplicate (right_eye) |
| 7 | left_ear | 3 | direct |
| 8 | right_ear | 4 | direct |
| 9 | mouth_left | 0, 17, 1 | interpolate: 70% nose + 30% neck, shift 15% toward left_eye |
| 10 | mouth_right | 0, 17, 2 | interpolate: 70% nose + 30% neck, shift 15% toward right_eye |
| 11 | left_shoulder | 5 | direct |
| 12 | right_shoulder | 6 | direct |
| 13 | left_elbow | 7 | direct |
| 14 | right_elbow | 8 | direct |
| 15 | left_wrist | 9 | direct |
| 16 | right_wrist | 10 | direct |
| 17 | left_pinky | 9 | duplicate (left_wrist) |
| 18 | right_pinky | 10 | duplicate (right_wrist) |
| 19 | left_index | 9 | duplicate (left_wrist) |
| 20 | right_index | 10 | duplicate (right_wrist) |
| 21 | left_thumb | 9 | duplicate (left_wrist) |
| 22 | right_thumb | 10 | duplicate (right_wrist) |
| 23 | left_hip | 11 | direct |
| 24 | right_hip | 12 | direct |
| 25 | left_knee | 13 | direct |
| 26 | right_knee | 14 | direct |
| 27 | left_ankle | 15 | direct |
| 28 | right_ankle | 16 | direct |
| 29 | left_heel | 15 | duplicate (left_ankle) |
| 30 | right_heel | 16 | duplicate (right_ankle) |
| 31 | left_foot_index | 15 | duplicate (left_ankle) |
| 32 | right_foot_index | 16 | duplicate (right_ankle) |

### Mapping strategies

- **Direct mapping** (13 keypoints): 1:1 correspondence between trt_pose and MediaPipe keypoints (nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles).
- **Duplication** (18 keypoints): MediaPipe keypoint copies the nearest available trt_pose keypoint. Eye inner/outer variants copy the eye center; hand keypoints (pinky, index, thumb) copy the wrist; foot keypoints (heel, foot index) copy the ankle.
- **Weighted interpolation** (2 keypoints): Mouth corners are synthesized as 70% nose + 30% neck position, then shifted 15% laterally toward the corresponding eye.

### Minimum detection threshold

Returns `None` (no pose) if fewer than 5 trt_pose keypoints are detected. This prevents feeding near-empty feature vectors to the LSTM classifiers.

### Limitations

- **Hand keypoints** (MP 17-22): Collapse to the wrist position. Fine-grained finger detail is unavailable from trt_pose.
- **Foot keypoints** (MP 29-32): Collapse to the ankle position. Heel/toe distinction is unavailable.
- **Mouth points** (MP 9-10): Synthetic approximations from head geometry, not detected from the image.

These limitations have low impact on cyclist gesture classification, which primarily relies on shoulder/elbow/wrist positions and head orientation.

### Pipeline timing

| Stage | Trigger | Latency from start |
|-------|---------|-------------------|
| Pose detection | Every frame | Immediate |
| Gesture classification | After 30 pose frames collected (~1s) | ~1 second |
| Maneuver prediction | After 719 gesture entries accumulated (~24s) | ~24 seconds |

After the initial fill, maneuver predictions update every frame.

## Hardware

- NVIDIA Jetson AGX Orin
- Intel RealSense D435 (USB 3.0 port)

## Prerequisites

The following must be installed system-wide (they are inherited via `--system-site-packages`):

- Python 3.8
- PyTorch with CUDA (e.g. `torch 2.0.0+nv23.05`)
- OpenCV (`cv2`)
- pyrealsense2

## Setup

```bash
bash setup.sh
cd ~/mathias_ws/cyclist_inference
python3 -m venv venv --system-site-packages --without-pip
source venv/bin/activate
curl -sS https://bootstrap.pypa.io/pip/3.8/get-pip.py | python3
pip install -r requirements.txt
```

This creates a virtual environment at `./venv` with system-site-packages and installs MediaPipe, scikit-learn, pandas, and numpy. A smoke test at the end verifies all required modules are importable.

To activate the environment manually:

```bash
source venv/bin/activate
```

## Usage

### With display (default, MediaPipe backend)

```bash
python3 cyclist_predict.py
```

Press `q` to quit.

### Use trt_pose backend

```bash
python3 cyclist_predict.py --pose trt
```

### Headless mode (save to video)

```bash
python3 cyclist_predict.py --headless -o recording.avi
python3 cyclist_predict.py --pose trt --headless -o recording.avi
```

Press `Ctrl+C` to stop.

### Select model fold

Models were trained with 5-fold cross-validation. Default is fold 5.

```bash
python3 cyclist_predict.py --fold 3
```

### Custom model paths

```bash
python3 cyclist_predict.py \
    --explicit-model /path/to/explicit.pkl \
    --implicit-model /path/to/implicit.pkl \
    --maneuver-model /path/to/maneuver.pkl
```

### Force CPU inference

```bash
python3 cyclist_predict.py --device cpu
```

### All options

| Flag | Default | Description |
|------|---------|-------------|
| `--explicit-model` | `../cyclistprediction/.../ExplicitModels/best_model_fold_5.pkl` | Explicit gesture model weights |
| `--implicit-model` | `../cyclistprediction/.../ImplicitModels/best_model_fold_5.pkl` | Implicit gesture model weights |
| `--maneuver-model` | `../cyclistprediction/.../ManueverPrediction/Models/best_model_fold_5.pkl` | Maneuver prediction model weights |
| `--fold` | `5` | Shorthand to select fold 1-5 |
| `--pose` | `mediapipe` | Pose estimation backend (`mediapipe` or `trt`) |
| `--headless` | off | Disable display window, save to file |
| `-o`, `--output` | `output.avi` | Output video path (headless mode) |
| `--device` | auto (CUDA if available) | Torch device (`cuda:0`, `cpu`) |

## Models

All three models are bidirectional LSTMs with dropout and a fully-connected output layer.

| Model | Input | Sequence length | LSTM layers | Hidden size | Output classes | Parameters |
|-------|-------|----------------|-------------|-------------|----------------|------------|
| Explicit gesture | 66 (33 keypoints x 2) | 30 frames | 4 | 128 | 3 | 1,387,267 |
| Implicit gesture | 66 (33 keypoints x 2) | 30 frames | 4 | 128 | 4 | 1,387,524 |
| Maneuver | 2 (gesture weights) | 719 frames | 2 | 128 | 3 | 531,203 |

### Gesture classes and weight encoding

**Explicit gestures** (hand signals):

| Index | Label | Weight |
|-------|-------|--------|
| 0 | Right_Hand_Explicit | +1 |
| 1 | neutral_explicit | 0 |
| 2 | Left_Hand_Explicit | -1 |

**Implicit gestures** (head/body cues):

| Index | Label | Weight |
|-------|-------|--------|
| 0 | Right_Look | +1 |
| 1 | neutral_implicit | 0 |
| 2 | Left_Overshoulder | -2 |
| 3 | Left_Look | -1 |

**Maneuver classes** (alphabetical via LabelEncoder):

| Index | Label |
|-------|-------|
| 0 | Crossing |
| 1 | LeftTurn |
| 2 | RightTurn |

## Display overlay

The video output shows:

- **Skeleton**: Pose landmarks drawn on the cyclist (from whichever backend is selected)
- **FPS**: Current processing frame rate
- **Explicit**: Current explicit gesture classification
- **Implicit**: Current implicit gesture classification
- **Maneuver**: Predicted maneuver (color-coded: green=Crossing, blue=LeftTurn, red=RightTurn)
- **Progress bar**: Gesture buffer fill towards the 719 entries needed for maneuver prediction

## File structure

```
cyclist_inference/
    README.md              # This file
    setup.sh               # Environment setup script
    requirements.txt       # pip dependencies
    models.py              # Bi-LSTM model class definitions
    pose_adapter.py        # Pose backend abstraction (MediaPipe / trt_pose)
    cyclist_predict.py     # Main real-time inference script
```

## Differences from training code

- **Device handling**: The original model definitions use a global `device` variable for `h0`/`c0` tensor initialization. The inference models use `x.device` instead, making them portable across devices without side effects.
- **No `return_hidden`**: The gesture model `forward()` methods drop the `return_hidden` parameter since inference only needs final class predictions.
- **Unified `ManeuverLSTMClassifier`**: Renamed from `LSTMClassifier` to avoid ambiguity with the gesture classifiers.

## Troubleshooting

**No RealSense camera detected**: Check USB 3.0 connection. Verify with `lsusb | grep 8086` (expect `8086:0b5c` for D435).

**MediaPipe install fails**: On aarch64/Python 3.8, try `pip install mediapipe==0.10.9` explicitly. If that fails, check for community aarch64 wheels.

**Low FPS**: MediaPipe `model_complexity=2` is the most accurate but slowest. Try `--pose trt` for significantly faster inference. Alternatively, edit `pose_adapter.py` and set `model_complexity=1` in `MediaPipeAdapter.__init__` for faster MediaPipe inference at the cost of some pose accuracy.

**Model not found**: Ensure the `cyclistprediction/` directory is at `../cyclistprediction/` relative to this directory, or provide explicit paths with `--explicit-model`, `--implicit-model`, and `--maneuver-model`.

**trt_pose model not found**: Ensure `pose_detection/models/resnet18_baseline_att_224x224_A_epoch_249.pth` and `pose_detection/models/human_pose.json` exist. Run `bash ../pose_detection/download_models.sh` if needed.

**trt_pose import error with `--pose mediapipe`**: The backends use lazy imports — trt_pose is only imported when `--pose trt` is selected, so MediaPipe mode works without trt_pose installed (and vice versa).

**TensorRT optimization fails (DECONVOLUTION weight count errors)**: This is a known issue with `fp16_mode=True` and deconvolution layers — FP16 conversion halves the weight count, causing a mismatch. The system automatically falls back to PyTorch CUDA inference, which is still fast on the Orin. To force FP32 TensorRT instead, change `fp16_mode=True` to `fp16_mode=False` in `pose_detection/pose_detect.py`. If a stale TRT cache exists from a previous attempt, delete `pose_detection/models/resnet18_baseline_att_224x224_A_trt.pth` before retrying.
