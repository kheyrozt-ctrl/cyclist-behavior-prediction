#!/bin/bash
# Setup script for cyclist intention prediction inference on Jetson AGX Orin
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"

echo "=== Cyclist Inference Environment Setup ==="

# Create venv with system-site-packages (for PyTorch, CUDA, pyrealsense2)
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment with system-site-packages..."
    python3 -m venv --system-site-packages "$VENV_DIR"
else
    echo "Virtual environment already exists at $VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

echo "Python: $(python3 --version)"
echo "Pip: $(pip --version)"

# Upgrade pip
pip install --upgrade pip

# Install MediaPipe (may need fallback on aarch64)
echo "Installing MediaPipe..."
if ! pip install mediapipe 2>/dev/null; then
    echo "Direct install failed, trying mediapipe==0.10.9..."
    if ! pip install mediapipe==0.10.9 2>/dev/null; then
        echo "WARNING: MediaPipe installation failed."
        echo "You may need to build from source or find an aarch64 wheel."
        echo "Try: pip install mediapipe-lite"
        pip install mediapipe-lite 2>/dev/null || true
    fi
fi

# Install remaining dependencies
echo "Installing remaining dependencies..."
pip install -r "$SCRIPT_DIR/requirements.txt"

# Smoke test
echo ""
echo "=== Smoke Test ==="
python3 -c "
import sys
modules = {}
for name in ['torch', 'numpy', 'pandas', 'sklearn', 'cv2']:
    try:
        __import__(name)
        modules[name] = 'OK'
    except ImportError:
        modules[name] = 'MISSING'

# MediaPipe check
try:
    import mediapipe
    modules['mediapipe'] = 'OK'
except ImportError:
    modules['mediapipe'] = 'MISSING'

# pyrealsense2 check
try:
    import pyrealsense2
    modules['pyrealsense2'] = 'OK'
except ImportError:
    modules['pyrealsense2'] = 'MISSING'

print('Module status:')
all_ok = True
for name, status in modules.items():
    print(f'  {name}: {status}')
    if status == 'MISSING':
        all_ok = False

if not all_ok:
    print('\nWARNING: Some modules are missing. The pipeline may not work correctly.')
    sys.exit(1)
else:
    print('\nAll modules OK!')
"

echo ""
echo "=== Setup Complete ==="
echo "Activate with: source $VENV_DIR/bin/activate"
echo "Run with: python3 cyclist_predict.py"
