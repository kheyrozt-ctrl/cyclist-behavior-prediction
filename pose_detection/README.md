# Skeleton Pose Detection on Jetson AGX Orin

Real-time human skeleton pose detection using an Intel RealSense D435 depth camera and NVIDIA's [trt_pose](https://github.com/NVIDIA-AI-IOT/trt_pose) library, running on a Jetson AGX Orin.

## Hardware

- **NVIDIA Jetson AGX Orin** — L4T R35.4.1 (JetPack 5.1.2), Ubuntu 20.04, Python 3.8
- **Intel RealSense D435** — USB 3.0 depth camera (RGB stream used for pose detection)

## Software Stack

| Component | Version | Purpose |
|-----------|---------|---------|
| CUDA | 11.4 | GPU compute (via `nvidia-jetpack`) |
| TensorRT | 8.5 | Optional inference optimization |
| PyTorch | 2.0.0+nv23.05 | Neural network framework (NVIDIA Jetson wheel) |
| torchvision | 0.15.1 | Image transforms, ResNet18 backbone |
| trt_pose | 0.0.1 | Pose estimation model + keypoint parsing |
| torch2trt | 0.5.0 | PyTorch-to-TensorRT converter (optional) |
| pyrealsense2 | 2.54.2 | RealSense camera SDK (built from source with `FORCE_RSUSB_BACKEND`) |
| OpenCV | 4.13.0 | Image processing and display |

## Model

**ResNet18 + Attention** (`resnet18_baseline_att_224x224_A`) from the trt_pose project:
- Input: 224x224 RGB image, ImageNet-normalized
- Output: 18 COCO keypoints (nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles) + part affinity fields
- Inference backend: PyTorch CUDA (with optional TensorRT acceleration)
- Performance: not benchmarked in this repository; measure throughput and
  latency on the target Jetson, software stack, power mode, and camera profile

## Project Structure

```
pose_detection/
├── setup.sh              # Installs all system + Python dependencies from scratch
├── download_models.sh    # Downloads pre-trained model weights and topology
├── pose_detect.py        # Main application: RealSense -> trt_pose -> skeleton overlay
├── fallback_dnn.py       # Lightweight OpenCV DNN fallback (no PyTorch needed)
├── requirements.txt      # Python package list
├── .gitignore
├── README.md
├── venv/                 # Python venv (--system-site-packages)
└── models/               # Downloaded model files
    ├── resnet18_baseline_att_224x224_A_epoch_249.pth  (82 MB)
    ├── human_pose.json                                 (COCO topology)
    ├── pose_deploy_linevec.prototxt                    (DNN fallback proto)
    └── pose_iter_440000.caffemodel                     (DNN fallback weights)
```

## Setup

### 1. Install dependencies

```bash
bash setup.sh
```

This installs everything from scratch in the correct order: pip, nvidia-jetpack (CUDA/TensorRT/cuDNN), numpy, OpenCV, librealsense (built from source), PyTorch, torchvision (built from source), torch2trt, and trt_pose. Takes ~30-45 minutes. The script is idempotent and skips already-installed components.

### 2. Set up the virtual environment

```bash
cd /home/agx-orin/mathias_ws/pose_detection
python3 -m venv venv --system-site-packages --without-pip
source venv/bin/activate
curl -sS https://bootstrap.pypa.io/pip/3.8/get-pip.py | python3
pip install -r requirements.txt
```

> **Note:** The venv uses `--system-site-packages` to inherit PyTorch, trt_pose, torch2trt, and pyrealsense2 which are built from source with CUDA/TensorRT support. Do not reinstall these inside the venv.

### 3. Download models

```bash
bash download_models.sh
```

### 4. Run

```bash
source venv/bin/activate

# Live display (requires monitor or X forwarding)
python3 pose_detect.py

# Over SSH — saves output to video file
python3 pose_detect.py --headless
python3 pose_detect.py --headless -o output.avi
```

Press `q` to quit (live mode) or `Ctrl+C` (headless mode).

## Fallback: OpenCV DNN

A lightweight fallback script that uses OpenCV's DNN module with an OpenPose
Caffe model. No PyTorch or TensorRT is required. CPU throughput is
hardware-dependent and is not claimed here.

```bash
python3 fallback_dnn.py
python3 fallback_dnn.py --webcam       # Use webcam instead of RealSense
python3 fallback_dnn.py --headless     # Save to file
```

## How It Works

1. **Camera capture** — The RealSense D435 streams 640x480 BGR frames at 30 FPS via `pyrealsense2`
2. **Preprocessing** — Each frame is resized to 224x224, converted to RGB, and ImageNet-normalized
3. **Inference** — The ResNet18 attention model produces a confidence map (keypoint heatmaps) and part affinity fields (connection strengths)
4. **Parsing** — `trt_pose.plugins` extracts peaks from heatmaps, scores connections via PAFs, and assembles them into full skeletons
5. **Visualization** — Keypoints are drawn as colored circles and connected joints as lines (left side = blue, right side = red, center = green)

## Skeleton Color Scheme

| Body Side | Color | Keypoints |
|-----------|-------|-----------|
| Left | Blue (BGR: 255,0,0) | Left ear, shoulder, elbow, wrist, hip, knee, ankle |
| Right | Red (BGR: 0,0,255) | Right ear, shoulder, elbow, wrist, hip, knee, ankle |
| Center | Green (BGR: 0,255,0) | Nose, eyes |

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `RuntimeError: failed to set power state` | Install udev rules: `sudo cp /tmp/pose_detection_build/librealsense/config/99-realsense-libusb.rules /etc/udev/rules.d/ && sudo udevadm control --reload-rules && sudo udevadm trigger`, then replug the camera |
| No RealSense camera detected | Check USB 3.0 connection: `lsusb \| grep 8086` (expect `8086:0b5c` for D435) |
| TensorRT optimization fails | The script automatically falls back to PyTorch CUDA inference |
| `libopenblas.so.0 not found` | `sudo apt install libopenblas-base` |
| Display issues over SSH | Use `--headless` flag to save to video file |
| PyTorch wheel URL broken | Check [NVIDIA forums](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048) for updated Jetson wheels |

## Key Design Decisions

- **librealsense built with `FORCE_RSUSB_BACKEND=ON`** — avoids kernel patching which is unreliable on Jetson L4T kernels
- **TensorRT is optional** — the app gracefully falls back to PyTorch CUDA if TRT conversion fails (version compatibility issues between torch2trt and TensorRT 8.5)
- **TRT model cached to disk** — first-run optimization takes ~30s, subsequent runs load instantly
- **Topology function inlined** — `coco_category_to_topology()` is copied from `trt_pose.coco` to avoid pulling in `pycocotools` as a dependency
