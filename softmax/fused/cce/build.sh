#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
[[ $# == 0 ]] || { echo "Usage: $0" >&2; exit 2; }
[[ -x "$CCEC" ]] || { echo "Missing ccec: $CCEC; set CANN_ROOT." >&2; exit 1; }
[[ -f "$SIM_LIB_DIR/libruntime_camodel.so" ]] || { echo "Missing simulator: $SIM_LIB_DIR" >&2; exit 1; }
mkdir -p "$BUILD_DIR"
exec > >(tee "$BUILD_DIR/build.log") 2>&1
"$CCEC" -fPIC -shared --cce-fatobj-link --cce-aicore-arch="$DEVICE_ARCH" \
  -DREGISTER_BASE -O2 -std=c++17 \
  -I "$CANN_ROOT/pkg_inc" -I "$CANN_ROOT/pkg_inc/profiling" \
  -I "$CANN_ROOT/pkg_inc/runtime/runtime" \
  -x cce "$LAB_ROOT/src/kernel.cpp" -x cce "$LAB_ROOT/src/launch.cpp" \
  -L "$SIM_LIB_DIR" -L "$CANN_ROOT/lib64" -lruntime_camodel \
  -o "$BUILD_DIR/libkernel.so"
"${CXX:-g++}" -O2 -std=c++17 "$LAB_ROOT/../../baseline/common/host.cpp" \
  -I "$CANN_ROOT/include" -L "$BUILD_DIR" -L "$SIM_LIB_DIR" -L "$CANN_ROOT/lib64" \
  -Wl,-rpath,'$ORIGIN' -lkernel -Wl,--no-as-needed -lruntime_camodel -Wl,--as-needed -lascendcl \
  -ltiling_api -lplatform -lc_sec -ldl -lnnopbase -lpthread -o "$BUILD_DIR/verify"
printf 'Build passed: %s\n' "$BUILD_DIR/verify"
{
  date -u '+Built: %Y-%m-%dT%H:%M:%SZ'
  printf 'CANN_ROOT=%s\nSOC_VERSION=%s\nDEVICE_ARCH=%s\nCCEC=%s\n' "$CANN_ROOT" "$SOC_VERSION" "$DEVICE_ARCH" "$CCEC"
  sha256sum "$LAB_ROOT"/src/*.cpp "$LAB_ROOT/../../baseline/common/host.cpp" "$LAB_ROOT"/{build,env,run}.sh
} > "$BUILD_DIR/build-info.txt"
