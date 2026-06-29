#!/usr/bin/env python3
"""Executable entrypoint for the unified cyclist prediction app."""

import os
import sys


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from unified_prediction.unified_app import main


if __name__ == "__main__":
    raise SystemExit(main())
