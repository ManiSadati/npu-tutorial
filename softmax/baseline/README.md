# Row-softmax baseline: 10 × 1024

Two editable A5 kernels compute softmax independently along each 1024-element
row, using FP32 input, FP32 arithmetic and FP32 output. Both use the stable
formula `exp(x - max(x)) / sum(exp(x - max(x)))` and support finite inputs.
The fixed launch is four vector blocks, processing 3, 3, 2 and 2 rows respectively.

## Run on Bluezone

```bash
cd ~/npu-tutorial/softmax/baseline/cce
bash run.sh

cd ~/npu-tutorial/softmax/baseline/pto-isa
bash run.sh
```

Each run builds the kernel and host, generates the same seeded 10×1024 input,
executes the A5 operator simulator, writes the output tensor, and runs the Python
comparison. `bash build.sh` builds only; `bash run.sh --no-build` reruns without
compiling. Scripts work from any current directory. Failures return nonzero.

Prerequisites are the existing CANN 9.1.0 beta.3 setup (ccec, A5 simulator), g++,
and Python 3 with NumPy. PTO-ISA additionally needs its header checkout.

```bash
export CANN_ROOT=/path/to/cann
export PTO_ISA_ROOT=/path/to/pto-isa   # PTO-ISA only; default ~/pto-isa
export PYTHON_BIN=python3           # any Python with NumPy
```

If CANN_ROOT is unset, each workspace uses the original lab's toolchain symlink
in this repository. Set CANN_ROOT explicitly when sharing with other accounts.
Neither Planner nor conda activation is required. Existing CCEC overrides still
apply; unset CCEC if it points to a different installation.

The scripts raise the soft open-file limit to the account's hard limit for the
simulator's trace files. The default timeout is **1800 seconds (30 minutes)**,
plus 10 seconds for forced shutdown. Override with `SIM_TIMEOUT_SEC=3600`.
Simulator timing is not hardware performance; these are functional baselines.

## Layout and rewrite points

- `cce/src/kernel.cpp`: explicit DMA, SIMD reductions, exp/div and pipeline events.
- `pto-isa/src/kernel.cpp`: TLOAD, TROWMAX, TROWEXPANDSUB, TEXP, TROWSUM,
  TROWEXPANDDIV, TSTORE. Separate UB tiles and full barriers between stages make
  dependencies explicit for initial rewrites.
- Each `src/launch.cpp`: shared ABI and four-block launch geometry.
- `common/host.cpp`: shared ACL launcher and binary tensor I/O.
- Each `generate.py` and `compare.py`: runnable wrappers around the common Python
  implementations, keeping inputs and tolerances identical between backends.

Kernel dimensions, launch geometry and host/comparison sizes are fixed. Change
these together when adding other shapes. The baseline does not handle arbitrary
strides, partial rows, NaN or infinity inputs. Each block waits for its output
store before reusing UB for its next row; removing synchronization requires
preserving those dependencies.

## Tensor files and comparisons

Each run prints its unique `build/sim-XXXXXXXX` directory. It contains:

- `input.bin`, `output.bin`, `reference.bin`: contiguous row-major FP32 tensors,
  10×1024, each exactly 40960 bytes; reference.bin is the float64 NumPy
  reference rounded to FP32.
- `run.log`: simulator output and comparison summary.
- Simulator `.dump` traces.

The checker computes the reference in float64, compares every element with
`rtol=2e-5, atol=2e-7`, requires finite nonnegative output, and checks every row
sum against one with tolerance `2e-5`. Inputs include seeded random rows,
constant rows, large positive/negative offsets, a wide ramp, and peaked rows.
Missing stores are detected by initializing device output to NaN.

Run comparisons manually from either backend folder:

```bash
python3 compare.py --input build/sim-XXXXXXXX/input.bin \
  --output build/sim-XXXXXXXX/output.bin
```

Compare both implementations using paths from their successful runs:

```bash
python3 compare.py --input /path/to/cce-run/input.bin \
  --output /path/to/cce-run/output.bin --other /path/to/pto-run/output.bin
```

Both outputs must use the same input; default generation is identical. Load a
tensor for inspection with `np.fromfile(path, dtype=np.float32).reshape(10, 1024)`.
Keep the entire baseline directory when copying it: both backends use `common/`.

## Validation status

Both implementations passed all 10240 output comparisons in the A5 simulator.
See [VALIDATION.md](VALIDATION.md) for errors, artifact paths and the direct
backend comparison. PTO's TROWMAX uses a negative-infinity initializer, which
triggers simulator `vec_err_idata_inf_nan_t0` diagnostics on VMAX in this passing
configuration; consult the validation notes rather than treating these specific
messages alone as a numerical failure.

## Operator profiling

`bash run.sh` now invokes `$CANN_ROOT/tools/msopprof/bin/msopprof simulator`
around the host executable and then runs the same numerical checker. It uses
the existing env.sh paths; `activate_ptoas` and .bashrc sourcing are unnecessary.
A bare `msopprof` selects device profiling, which is not the mode used here.

Reports are under each new `build/sim-*/profile/OPPROF_*/simulator/` directory:
`trace.json` plus per-core `*_instr_exe.csv` files. `PROFILE_CORE_ID` defaults
to 0 (the profiler's core group; on this simulator its report includes vector
subcores core0.veccore0 and core0.veccore1). All kernel blocks still execute and
the complete output is checked. `MSOPPROF_BIN` can override the executable.
`PROFILE_TIMEOUT_MIN` defaults to 30 minutes; the outer SIM_TIMEOUT_SEC also
remains in effect.

The installed profiler may warn about PC-start lookup and absent debug_line
information. Instruction traces are still generated, but these builds lack
source-line/call-stack attribution. Adding compiler `-g` is a separate optional
build change. Profiling can perturb simulation ticks; compare runs made with
identical profiling settings.
