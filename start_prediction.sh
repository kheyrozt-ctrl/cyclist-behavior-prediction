#!/usr/bin/env sh
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$SCRIPT_DIR/open_source/inference/unified_prediction" || exit 1
exec sh ./start_web.sh
