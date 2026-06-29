#!/usr/bin/env bash
# setup.sh — Install all dependencies for trt_pose skeleton detection on Jetson AGX Orin
# Run: bash setup.sh
# Expected time: ~30-45 minutes (mostly librealsense build + torchvision build)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="${SCRIPT_DIR}/models"
BUILD_DIR="/tmp/pose_detection_build"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log()  { echo -e "${GREEN}[SETUP]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
fail() { echo -e "${RED}[FAIL]${NC} $*"; exit 1; }

mkdir -p "$BUILD_DIR" "$MODELS_DIR"

# ─── Step 1: pip3 + basic system packages ───────────────────────────────────
log "Installing pip3 and system packages..."
sudo apt-get update
sudo apt-get install -y python3-pip python3-dev python3-setuptools \
    build-essential cmake git pkg-config libssl-dev libusb-1.0-0-dev \
    libgtk-3-dev libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev

# ─── Step 2: JetPack / CUDA / TensorRT ─────────────────────────────────────
if ! command -v nvcc &>/dev/null; then
    log "Installing nvidia-jetpack (CUDA, TensorRT, cuDNN)... This may take a while (~5-7 GB)."
    sudo apt-get install -y nvidia-jetpack
else
    log "nvcc already found, skipping nvidia-jetpack install."
fi

# ─── Step 3: CUDA environment variables ─────────────────────────────────────
CUDA_ENV_BLOCK='
# CUDA environment (added by pose_detection setup.sh)
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
'

if ! grep -q "CUDA_HOME=/usr/local/cuda" ~/.bashrc 2>/dev/null; then
    log "Adding CUDA environment variables to ~/.bashrc"
    echo "$CUDA_ENV_BLOCK" >> ~/.bashrc
fi

# Source for current session
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}

log "CUDA version: $(nvcc --version 2>/dev/null | grep 'release' || echo 'not yet available')"

# ─── Step 4: Python packages ───────────────────────────────────────────────
log "Installing numpy and opencv-python..."
pip3 install numpy==1.24.4
pip3 install opencv-python Pillow gdown

# ─── Step 5: librealsense SDK (build from source) ──────────────────────────
if python3 -c "import pyrealsense2" 2>/dev/null; then
    log "pyrealsense2 already installed, skipping librealsense build."
else
    log "Building librealsense v2.54.2 from source (~15-20 min)..."
    REALSENSE_DIR="${BUILD_DIR}/librealsense"

    if [ ! -d "$REALSENSE_DIR" ]; then
        git clone --depth 1 --branch v2.54.2 https://github.com/IntelRealSense/librealsense.git "$REALSENSE_DIR"
    fi

    mkdir -p "${REALSENSE_DIR}/build"
    cd "${REALSENSE_DIR}/build"

    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_PYTHON_BINDINGS=true \
        -DPYTHON_EXECUTABLE=$(which python3) \
        -DBUILD_EXAMPLES=false \
        -DBUILD_GRAPHICAL_EXAMPLES=false \
        -DFORCE_RSUSB_BACKEND=ON \
        -DBUILD_WITH_CUDA=OFF

    make -j$(nproc)
    sudo make install

    # Install Python bindings
    PYREALSENSE_SO=$(find "${REALSENSE_DIR}/build" -name "pyrealsense2*.so" | head -1)
    if [ -n "$PYREALSENSE_SO" ]; then
        SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")
        sudo cp "$PYREALSENSE_SO" "$SITE_PACKAGES/"
        # Also copy the pyrealsense2 directory if it exists
        PYRS_DIR=$(dirname "$PYREALSENSE_SO")
        if [ -d "${PYRS_DIR}/pyrealsense2" ]; then
            sudo cp -r "${PYRS_DIR}/pyrealsense2" "$SITE_PACKAGES/"
        fi
    fi

    cd "$SCRIPT_DIR"

    python3 -c "import pyrealsense2; print('pyrealsense2 version:', pyrealsense2.__version__)" \
        || warn "pyrealsense2 import failed — you may need to set PYTHONPATH manually."
fi

# ─── Step 6: PyTorch 2.0 (NVIDIA Jetson wheel) ─────────────────────────────
if python3 -c "import torch; print('PyTorch', torch.__version__)" 2>/dev/null; then
    log "PyTorch already installed."
else
    log "Installing PyTorch 2.0 for Jetson (JP5.1.2, aarch64)..."
    TORCH_WHEEL="torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl"
    TORCH_URL="https://developer.download.nvidia.com/compute/redist/jp/v512/${TORCH_WHEEL}"

    cd "$BUILD_DIR"
    if [ ! -f "$TORCH_WHEEL" ]; then
        wget -q --show-progress "$TORCH_URL" -O "$TORCH_WHEEL" || {
            warn "PyTorch wheel download failed. The URL may have changed."
            warn "Check: https://forums.developer.nvidia.com/t/pytorch-for-jetson/"
            warn "Look for: PyTorch v2.0 — JetPack 5.1.2 wheel"
            fail "Cannot continue without PyTorch."
        }
    fi
    pip3 install "$TORCH_WHEEL"
    cd "$SCRIPT_DIR"

    python3 -c "import torch; print('PyTorch', torch.__version__, '| CUDA:', torch.cuda.is_available())"
fi

# ─── Step 7: torchvision 0.15.1 (build from source) ────────────────────────
if python3 -c "import torchvision; print('torchvision', torchvision.__version__)" 2>/dev/null; then
    log "torchvision already installed."
else
    log "Building torchvision 0.15.1 from source..."
    VISION_DIR="${BUILD_DIR}/torchvision"

    if [ ! -d "$VISION_DIR" ]; then
        git clone --depth 1 --branch v0.15.1 https://github.com/pytorch/vision.git "$VISION_DIR"
    fi

    cd "$VISION_DIR"
    python3 setup.py install --user
    cd "$SCRIPT_DIR"

    python3 -c "import torchvision; print('torchvision', torchvision.__version__)"
fi

# ─── Step 8: torch2trt ─────────────────────────────────────────────────────
if python3 -c "import torch2trt" 2>/dev/null; then
    log "torch2trt already installed."
else
    log "Installing torch2trt..."
    T2T_DIR="${BUILD_DIR}/torch2trt"

    if [ ! -d "$T2T_DIR" ]; then
        git clone https://github.com/NVIDIA-AI-IOT/torch2trt.git "$T2T_DIR"
    fi

    cd "$T2T_DIR"
    python3 setup.py install --user
    cd "$SCRIPT_DIR"
fi

# ─── Step 9: trt_pose ──────────────────────────────────────────────────────
if python3 -c "import trt_pose" 2>/dev/null; then
    log "trt_pose already installed."
else
    log "Installing trt_pose..."
    TRT_POSE_DIR="${BUILD_DIR}/trt_pose"

    if [ ! -d "$TRT_POSE_DIR" ]; then
        git clone https://github.com/NVIDIA-AI-IOT/trt_pose.git "$TRT_POSE_DIR"
    fi

    cd "$TRT_POSE_DIR"
    python3 setup.py install --user
    cd "$SCRIPT_DIR"
fi

# ─── Done ───────────────────────────────────────────────────────────────────
log "========================================="
log "Setup complete!"
log "========================================="
log "Next steps:"
log "  1. bash download_models.sh"
log "  2. python3 pose_detect.py"
log ""
log "If this is a fresh shell, run: source ~/.bashrc"
