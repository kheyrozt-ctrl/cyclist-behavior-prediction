#!/usr/bin/env bash
set -e

echo "===== OS ====="
cat /etc/os-release || true
echo

echo "===== Jetson L4T / JetPack ====="
cat /etc/nv_tegra_release || true
apt-cache show nvidia-jetpack 2>/dev/null | grep Version || true
echo

echo "===== Architecture ====="
uname -a
dpkg --print-architecture || true
echo

echo "===== Python ====="
which python3 || true
python3 --version || true
python3 -m pip --version || true
echo

echo "===== glibc ====="
ldd --version | head -n 1 || true
echo

echo "===== CUDA ====="
which nvcc || true
nvcc --version || true
echo

echo "===== NVIDIA / Jetson GPU ====="
which tegrastats || true
nvidia-smi || true
echo

echo "===== TensorRT ====="
dpkg -l | grep -i tensorrt || true
python3 - <<'PY' || true
try:
    import tensorrt as trt
    print("TensorRT Python:", trt.__version__)
except Exception as e:
    print("TensorRT Python import failed:", e)
PY
echo

echo "===== PyTorch ====="
python3 - <<'PY' || true
try:
    import torch
    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA version:", torch.version.cuda)
except Exception as e:
    print("PyTorch import failed:", e)
PY
echo

echo "===== Holoscan ====="
python3 - <<'PY' || true
try:
    import holoscan
    print("Holoscan installed:", holoscan.__version__)
except Exception as e:
    print("Holoscan import failed:", e)
PY
echo

echo "===== Docker ====="
docker --version || true
docker info 2>/dev/null | grep -i runtime || true
echo

echo "===== RealSense ====="
which realsense-viewer || true
rs-enumerate-devices || true
echo

echo "===== Project files ====="
pwd
ls -la
echo
find . -maxdepth 3 -type f | grep -E "holoscan|app.py|requirements.txt|camera.py|predict" || true


