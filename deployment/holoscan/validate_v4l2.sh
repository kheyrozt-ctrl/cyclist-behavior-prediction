#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

camera_index="${1:-0}"
frames="${2:-450}"
fps="${3:-30}"
if [[ ! "$camera_index" =~ ^[0-9]+$ || ! "$frames" =~ ^[1-9][0-9]*$ || ! "$fps" =~ ^[1-9][0-9]*$ ]]; then
  echo "Usage: $0 [non-negative-camera-index] [positive-frame-count] [positive-fps]" >&2
  exit 2
fi
duration="$(python3 - "$frames" "$fps" <<'PY'
import sys
print(int(sys.argv[1]) / int(sys.argv[2]))
PY
)"
output="release/holoscan-v4l2-validation.jsonl"

python3 deployment/holoscan/app.py \
  --model bus \
  --runtime onnx \
  --camera webcam \
  --webcam-index "$camera_index" \
  --pose mediapipe \
  --mediapipe-complexity 1 \
  --duration "$duration" \
  --fps "$fps" \
  --headless \
  --output-jsonl "$output"

python3 tools/summarize_holoscan_run.py \
  "$output" \
  --output release/holoscan-v4l2-validation-summary.json
