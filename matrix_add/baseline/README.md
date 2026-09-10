# Single-core matrix addition: 100 × 512

Compute `C[row, col] = A[row, col] + B[row, col]` for two contiguous FP32
100×512 matrices. This is elementwise addition processed row by row, with no
reduction along the row.

## Run

```bash
cd ~/npu-tutorial/matrix_add/baseline/cce
bash run.sh

cd ~/npu-tutorial/matrix_add/baseline/pto-isa
bash run.sh
```

Each run builds, generates identical seeded A/B matrices, runs the A5 simulator,
saves C and compares all 51200 elements with NumPy FP32 addition. Exact equality
is required for this finite, normal-valued test data. Cases include cancellation,
zero inputs and distinct row/column values to expose indexing errors. Output is
initialized to NaN to catch missing stores, including the final row.

## One core

Both launchers use `matrix_add<<<1, nullptr, stream>>>`: exactly one vector block.
That block loops from row 0 through row 99, reusing three 2048-byte UB buffers.
No rows are distributed across cores. Each row completes its GM store before
reusing the buffers. The target is A5 `dav-c310-vec`.

The simulator initializes many model cores; the `AicWrapper attach AIC ...`
messages describe initialization, not work distribution. Inspect `block_start`
and `block_end` lines for the actual executing AIV block. A one-block launch
does not pin a particular physical core number across separate hardware runs.

## Setup and files

Setup matches the existing labs: CANN with ccec and the Ascend950PR_9599
simulator, g++, Python with NumPy, and PTO-ISA headers for that backend.

```bash
export CANN_ROOT=/path/to/cann
export PTO_ISA_ROOT=/path/to/pto-isa  # only PTO-ISA; defaults to ~/pto-isa
```

Default CANN paths use the original labs' toolchain symlinks. Use explicit
CANN_ROOT when sharing. No Planner scripts or conda activation are required.
`PYTHON_BIN` selects a Python with NumPy. The scripts raise the soft open-file
limit to the account's hard limit and use a 1800-second timeout, configurable
with SIM_TIMEOUT_SEC. `bash run.sh --no-build` skips compilation.

- `cce/src/kernel.cpp`: raw A5 DMA, eight 64-lane vector additions per row, sync.
- `pto-isa/src/kernel.cpp`: row tiles using TLOAD/TADD/TSTORE with the same sync.
- Each `src/launch.cpp`: one-block launch; changing this changes core allocation.
- `common/host.cpp`: shared ACL host and binary I/O for two inputs and one output.
- Each `generate.py`/`compare.py`: runnable wrappers sharing `common/` Python code.
- Each `build/sim-*`: a.bin, b.bin, output.bin, reference.bin, run.log and traces.

Each tensor is 204800 bytes, row-major FP32, shape (100,512). Load with
`np.fromfile(path, dtype=np.float32).reshape(100, 512)`.

To compare an existing output, from either backend folder:

```bash
python3 compare.py --a /path/to/a.bin --b /path/to/b.bin \
  --output /path/to/output.bin
```

Append `--other /path/to/other-output.bin` to compare both implementations;
both must have used identical A and B. Keep the whole baseline directory when
copying it, since the backends use the shared common folder.

Both implementations passed all output checks on one simulated AIV core. See
[VALIDATION.md](VALIDATION.md) for logs, ticks and wall times.
