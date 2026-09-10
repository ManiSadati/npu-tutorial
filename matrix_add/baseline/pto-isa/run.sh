#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" -c 'import numpy' || { echo "Install NumPy in PYTHON_BIN's environment." >&2; exit 1; }
if [[ $# == 0 ]]; then
  bash "$LAB_ROOT/build.sh"
elif [[ $# != 1 || $1 != --no-build ]]; then
  echo "Usage: bash run.sh [--no-build]" >&2; exit 2
fi
ulimit -S -n "$(ulimit -H -n)"
[[ -x "$BUILD_DIR/verify" ]] || { echo "Run bash build.sh first." >&2; exit 1; }
ARTIFACT_DIR="$(mktemp -d "$BUILD_DIR/sim-XXXXXXXX")"
cd "$ARTIFACT_DIR"
exec > >(tee run.log) 2>&1
printf 'A5 simulator artifacts: %s\nOpen-file limit: %s\n' "$ARTIFACT_DIR" "$(ulimit -S -n)"
"$PYTHON_BIN" "$LAB_ROOT/generate.py" --a a.bin --b b.bin
# Timeout covers simulator startup, kernel execution, copies, and shutdown.
timeout --kill-after=10s "${SIM_TIMEOUT_SEC:-1800}s" "$BUILD_DIR/verify" a.bin b.bin output.bin
"$PYTHON_BIN" "$LAB_ROOT/compare.py" --a a.bin --b b.bin --output output.bin
