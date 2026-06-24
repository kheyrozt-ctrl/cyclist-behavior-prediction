#!/usr/bin/env sh

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$SCRIPT_DIR" || exit 1
VENV_PYTHON="$SCRIPT_DIR/.venv/bin/python"

if [ ! -x "$VENV_PYTHON" ]; then
    if command -v python3 >/dev/null 2>&1; then
        BASE_PYTHON=python3
    elif command -v python >/dev/null 2>&1; then
        BASE_PYTHON=python
    else
        printf '\nPython 3 was not found. Install Python 3 and try again.\n'
        printf 'Press Enter to close...'
        read -r _answer
        exit 1
    fi

    printf '\n[SETUP] Creating the Unified Python environment...\n'
    "$BASE_PYTHON" -m venv --system-site-packages "$SCRIPT_DIR/.venv" || exit 1
fi

if ! "$VENV_PYTHON" -c 'import torch, cv2, numpy, sklearn, pandas; from mediapipe import solutions' >/dev/null 2>&1; then
    printf '\n[SETUP] Installing prediction dependencies. The first run may take several minutes...\n'
    "$VENV_PYTHON" -m pip install --upgrade pip || exit 1
    "$VENV_PYTHON" -m pip install -r requirements-unified.txt || exit 1
fi

"$VENV_PYTHON" unified_prediction/run_web.py
EXIT_CODE=$?

if [ "$EXIT_CODE" -ne 0 ]; then
    printf '\nUnified Vision Console stopped with exit code %s.\n' "$EXIT_CODE"
    printf 'Review the error above.\nPress Enter to close...'
    read -r _answer
fi

exit "$EXIT_CODE"
