#!/usr/bin/env sh

set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)
VENV="$PROJECT_ROOT/.venv-holoscan-jp5"
PYTHON="$VENV/bin/python"

if [ ! -x "$PYTHON" ]; then
    python3 -m venv --system-site-packages "$VENV"
fi

if ! "$PYTHON" -c 'import filelock, os, sys; raise SystemExit(0 if os.path.realpath(filelock.__file__).startswith(os.path.realpath(sys.prefix) + os.sep) else 1)' >/dev/null 2>&1; then
    "$PYTHON" -m pip install --ignore-installed "filelock<4"
fi

if ! "$PYTHON" -c 'import holoscan, cv2, numpy' >/dev/null 2>&1; then
    "$PYTHON" -m pip install --upgrade "pip<25"
    "$PYTHON" -m pip install -r "$SCRIPT_DIR/requirements.txt"
fi

cd "$PROJECT_ROOT"
exec "$PYTHON" "$SCRIPT_DIR/app.py" "$@"
