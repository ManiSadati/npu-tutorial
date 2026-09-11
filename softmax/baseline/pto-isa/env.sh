#!/usr/bin/env bash
# Source from build.sh/run.sh or from your shell.
LAB_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export CANN_ROOT="${CANN_ROOT:-${LAB_ROOT}/../../../pto-isa-lab/toolchain}"
export PTO_ISA_ROOT="${PTO_ISA_ROOT:-${HOME}/pto-isa}"
export SOC_VERSION="${SOC_VERSION:-Ascend950PR_9599}"
export DEVICE_ARCH="${DEVICE_ARCH:-dav-c310-vec}"
export BUILD_DIR="${BUILD_DIR:-${LAB_ROOT}/build}"
BUILD_DIR="$(realpath -m -- "$BUILD_DIR")"
export CCEC="${CCEC:-${CANN_ROOT}/bin/ccec}"
export SIM_LIB_DIR="${CANN_ROOT}/tools/simulator/${SOC_VERSION}/lib"
export LD_LIBRARY_PATH="${BUILD_DIR}:${SIM_LIB_DIR}:${CANN_ROOT}/lib64:${CANN_ROOT}/x86_64-linux/lib64:${CANN_ROOT}/x86_64-linux/devlib:${CANN_ROOT}/x86_64-linux/devlib/linux/x86_64:${LD_LIBRARY_PATH:-}"
export ASCEND_HOME_PATH="$CANN_ROOT"
export ASCEND_OPP_PATH="$CANN_ROOT/opp"
