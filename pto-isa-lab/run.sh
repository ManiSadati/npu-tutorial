#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
if [[ $# == 0 ]]; then
  "$LAB_ROOT/build.sh"
elif [[ $# != 1 || $1 != --no-build ]]; then
  echo "Usage: $0 [--no-build]" >&2; exit 2
fi
[[ -x "$BUILD_DIR/verify" ]] || { echo "Run ./build.sh first." >&2; exit 1; }
ARTIFACT_DIR="$(mktemp -d "$BUILD_DIR/sim-XXXXXXXX")"
cd "$ARTIFACT_DIR"
printf 'A5 simulator artifacts: %s\n' "$ARTIFACT_DIR"
timeout --kill-after=10s "${SIM_TIMEOUT_SEC:-600}s" "$BUILD_DIR/verify" 2>&1 | tee run.log
