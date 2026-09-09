# PTO-ISA A5 lab

Standalone kernel workspace for this Bluezone server. Builds real A5 device code
and executes it through the installed CANN operator simulator (`runtime_camodel`).
This is simulator validation, not a physical NPU run or hardware timing result.

## Run

```bash
cd ~/npu-tutorial/pto-isa-lab
./run.sh                # compile kernel + host, run and verify
./build.sh              # compile only
./run.sh --no-build     # rerun the existing binary
```

No conda activation or Planner scripts are required. Scripts also work when
invoked by absolute path from another directory.

## Write kernels

`src/kernel.cpp` uses PTO-ISA tiles: `TASSIGN`, `TLOAD`, `TADD`, and `TSTORE`.

- `src/kernel.cpp`: edit the device kernel here.
- `src/launch.cpp`: kernel declaration and `<<<1, nullptr, stream>>>` launch.
- `src/host.cpp`: ACL allocation, input generation, launch, output comparison, cleanup.
- `env.sh`: toolkit, target and library paths.
- `build.sh`: compiles `.cpp` device sources using `ccec -x cce`, links a shared
  kernel library, then builds the C++ host with `g++`.
- `run.sh`: builds by default, runs with a timeout and retains simulator traces.

The example computes `c = a + b` for **256 FP32 elements in one vector block**.
Three trials cover mixed positive/negative values, cancellation to zero, and
addition of zero. Every result must match exactly; the host initializes output
to NaN to detect missing stores. Exit status is nonzero on build, runtime,
comparison, cleanup, or timeout failure.

When changing shapes or arguments, update all three C++ files together.
The example has no dynamic length or tail handling. UB offsets are bytes;
each of its three buffers occupies 1024 bytes. Preserve the MTE2 → vector →
MTE3 events and the final barrier when experimenting with memory movement.

## Configuration

`toolchain` is a local symlink to the existing CANN 9.1.0 beta.3 installation.
The toolkit remains external to this workspace. Override it when moving folders:

```bash
CANN_ROOT=/path/to/cann ./run.sh
BUILD_DIR="$PWD/build-experiment" ./run.sh
SIM_TIMEOUT_SEC=900 ./run.sh --no-build
```

Defaults: `SOC_VERSION=Ascend950PR_9599`, `DEVICE_ARCH=dav-c310-vec`,
`SIM_TIMEOUT_SEC=600`, `CXX=g++`. `CCEC` can override the compiler executable.
Changing target variables requires a compatible toolkit and kernel; this
example is tested only with the defaults. Do not treat `--no-build` as rebuilding
for a changed target or changed sources.

The host link deliberately retains `libruntime_camodel` with `--no-as-needed`
before `libascendcl`; dropping it can select the hardware runtime and cause
`aclInit` to fail. `env.sh` puts simulator libraries first and supplies CANN's
HAL stub library path needed on this server.

## Output

- `build/libkernel.so`: device kernel plus host launch wrapper.
- `build/verify`: numerical verifier.
- `build/build.log`: latest compiler output.
- `build/build-info.txt`: build configuration and source hashes.
- `build/sim-*/run.log`: one log per run; look for three `Trial ... PASS` lines.
- `build/sim-*/*.dump`: simulator instruction/memory traces.

Simulator startup and tracing can take several minutes. CANN may also write
its normal diagnostic logs under `~/ascend/log`. Build and trace files are
regenerable; each simulator run gets its own directory.

## Local references

The setup was derived from the working CANN launch pattern in
`~/mani-PTO/softmax/run_sim.sh` and the PTO-ISA A5 TADD example at
`~/pto-isa/tests/npu/a5/src/st/testcase/tadd/tadd_kernel.cpp`.
Planner context: `~/Planner/AGENT.md` and
`~/Planner/PTO-ISA/coding-guide/repo-and-validation.md`.

PTO headers default to `~/pto-isa/include`; override with
`PTO_ISA_ROOT=/path/to/pto-isa ./run.sh`. The tested checkout is
`896d8ec69aaf5b623fead5afcae7a657fa784a2b`; its existing local change to
`TBinOp.hpp` is whitespace only and was left untouched.

## Simulator error: Too many open files

`run.sh` raises its soft open-file limit to the account's hard limit before
launching the simulator. This affects only the script and its child processes.
If the simulator still reports `Too many open files`, inspect `ulimit -Sn` and
`ulimit -Hn`. An administrator must increase the session's hard limit if it is
too low; the script cannot exceed it. Start a new login session after an account
limit change. No kernel rebuild is needed: use `./run.sh --no-build`.
