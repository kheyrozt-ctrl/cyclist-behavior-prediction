#!/usr/bin/env bash
# download_models.sh — Download trt_pose pre-trained model and COCO topology
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="${SCRIPT_DIR}/models"
mkdir -p "$MODELS_DIR"

GREEN='\033[0;32m'
NC='\033[0m'
log() { echo -e "${GREEN}[DOWNLOAD]${NC} $*"; }

# ─── ResNet18 trt_pose model ───────────────────────────────────────────────
MODEL_FILE="${MODELS_DIR}/resnet18_baseline_att_224x224_A_epoch_249.pth"
if [ -f "$MODEL_FILE" ]; then
    log "Model already downloaded: $(basename "$MODEL_FILE")"
else
    log "Downloading resnet18_baseline_att_224x224_A model..."
    # Google Drive file ID for the trt_pose resnet18 model
    GDRIVE_ID="1XYDdCUdiF2xxx4rznmLb62SdOUZuoNbd"
    pip3 install gdown -q 2>/dev/null || true
    gdown "https://drive.google.com/uc?id=${GDRIVE_ID}" -O "$MODEL_FILE" || {
        echo "gdown failed. Trying direct URL..."
        wget -q --show-progress \
            "https://github.com/NVIDIA-AI-IOT/trt_pose/releases/download/v0.0.1/resnet18_baseline_att_224x224_A_epoch_249.pth" \
            -O "$MODEL_FILE" || {
            echo ""
            echo "Automatic download failed. Please download manually:"
            echo "  Model: resnet18_baseline_att_224x224_A_epoch_249.pth"
            echo "  From:  https://github.com/NVIDIA-AI-IOT/trt_pose"
            echo "  Place in: $MODELS_DIR/"
            exit 1
        }
    }
fi
log "Model size: $(du -h "$MODEL_FILE" | cut -f1)"

# ─── COCO topology (human_pose.json) ───────────────────────────────────────
TOPOLOGY_FILE="${MODELS_DIR}/human_pose.json"
if [ -f "$TOPOLOGY_FILE" ]; then
    log "Topology already downloaded: $(basename "$TOPOLOGY_FILE")"
else
    log "Downloading human_pose.json topology..."
    wget -q --show-progress \
        "https://raw.githubusercontent.com/NVIDIA-AI-IOT/trt_pose/master/tasks/human_pose/human_pose.json" \
        -O "$TOPOLOGY_FILE"
fi

# ─── OpenCV DNN fallback model (optional) ──────────────────────────────────
DNN_PROTO="${MODELS_DIR}/pose_deploy_linevec.prototxt"
DNN_MODEL="${MODELS_DIR}/pose_iter_440000.caffemodel"

if [ -f "$DNN_MODEL" ]; then
    log "OpenCV DNN model already downloaded."
else
    log "Downloading OpenCV DNN fallback model (MobileNet OpenPose, ~8MB)..."
    wget -q --show-progress \
        "https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/models/pose/body_25/pose_deploy.prototxt" \
        -O "${MODELS_DIR}/pose_deploy_body25.prototxt" 2>/dev/null || true

    # Use the lighter COCO model for the DNN fallback
    wget -q --show-progress \
        "https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/models/pose/coco/pose_deploy_linevec.prototxt" \
        -O "$DNN_PROTO" 2>/dev/null || true

    wget -q --show-progress \
        "http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/coco/pose_iter_440000.caffemodel" \
        -O "$DNN_MODEL" 2>/dev/null || {
        warn "DNN fallback model download failed (optional, only needed for fallback_dnn.py)."
    }
fi

log "========================================="
log "All models downloaded to: $MODELS_DIR/"
ls -lh "$MODELS_DIR/"
log "========================================="
