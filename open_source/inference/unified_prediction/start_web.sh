#!/usr/bin/env sh

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$SCRIPT_DIR" || exit 1
VENV_PYTHON="$SCRIPT_DIR/.venv/bin/python"

find_python() {
    for candidate in python3.11 python3.12 python3 python; do
        if command -v "$candidate" >/dev/null 2>&1 &&
            "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] in ((3, 11), (3, 12)) else 1)' >/dev/null 2>&1; then
            printf '%s' "$candidate"
            return 0
        fi
    done
    return 1
}

if [ ! -x "$VENV_PYTHON" ]; then
    BASE_PYTHON=$(find_python) || {
        printf '\nPython 3.11 or 3.12 is required for this runtime.\n'
        printf 'Install Python 3.11/3.12, delete .venv if it exists, then try again.\n'
        printf 'Press Enter to close...'
        read -r _answer
        exit 1
    }
    printf '\n[SETUP] Creating the Unified Python environment...\n'
    "$BASE_PYTHON" -m venv --system-site-packages "$SCRIPT_DIR/.venv" || exit 1
fi

if ! "$VENV_PYTHON" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] in ((3, 11), (3, 12)) else 1)' >/dev/null 2>&1; then
    "$VENV_PYTHON" -c 'import sys; print("Current venv Python:", sys.version)'
    printf '\nThis venv uses an incompatible Python version.\n'
    printf 'Delete %s and rerun after installing Python 3.11 or 3.12.\n' "$SCRIPT_DIR/.venv"
    exit 1
fi

if ! "$VENV_PYTHON" -c 'import torch, cv2, numpy, sklearn, pandas; from mediapipe import solutions' >/dev/null 2>&1; then
    printf '\n[SETUP] Installing prediction dependencies. The first run may take several minutes...\n'
    "$VENV_PYTHON" -m pip install --upgrade pip || exit 1
    "$VENV_PYTHON" -m pip install -r requirements.txt || exit 1
fi

"$VENV_PYTHON" run_web.py
EXIT_CODE=$?

if [ "$EXIT_CODE" -ne 0 ]; then
    printf '\nUnified Vision Console stopped with exit code %s.\n' "$EXIT_CODE"
    printf 'Review the error above.\nPress Enter to close...'
    read -r _answer
fi

exit "$EXIT_CODE"
