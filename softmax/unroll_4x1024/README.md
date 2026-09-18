# Row softmax: loops versus explicit register reuse

Both CCE kernels compute full stable softmax for **4x1024 FP32** input and
return all 4096 normalized values. They target A5 `dav-c310-vec`, run one
vector block, and use the same single 16 KiB input DMA and single 16 KiB
output DMA. This is ordinary max-then-exp/sum softmax, without online repair.

`baseline` has an outer four-row loop and three inner sixteen-vector loops:
find max, calculate/save exponentials and sum, then reload/divide/store.
Reductions use `vcmax`/`vcadd` followed by register `vdup` broadcasts.

`unrolled` explicitly expands every row and vector operation. Each row keeps
sixteen input vectors live through the maximum calculation, overwrites those
registers with exponentials, sums them, and finally divides/stores. There are
no source loops or intermediate exponential UB stores. Separate lexical
scopes let registers be reused between rows; the kernel does not need to
retain all 64 input vectors simultaneously. Loads/stores use constant offsets.
The arithmetic reduction order and `vdiv` are retained from baseline.

Expected data-vector instruction counts, before compiler effects:

| Operation | Baseline | Unrolled |
| --- | ---: | ---: |
| UB loads | 192 | 64 |
| UB stores | 128 | 64 |
| Exponentials | 64 | 64 |
| Divisions | 64 | 64 |
| Horizontal max / sum | 4 / 4 | 4 / 4 |

This tests explicit register reuse together with unrolling and constant
offsets; it does not isolate each change's individual contribution. Compiler
scheduling or spills can affect the actual instructions; see validation.
Both retain the same UB output offset 32768 and end address 49152 for matching
layout. The unrolled kernel leaves the middle 16 KiB unused; compacting that
gap is not part of this test. No artificial 20.5 KiB limit applies here.

## Run

```bash
cd ~/npu-tutorial/softmax/unroll_4x1024/baseline
bash run.sh
cd ../unrolled
bash run.sh
```

Use `--no-build` to rerun an existing matching binary. Both use `common/` host,
generator and checker files. Inputs include seeded random values, random rows
offset by +1000 and -1000, and a ramp from -80 to 80. The checker compares all
outputs against stable NumPy float64 softmax with `rtol=2e-5`, `atol=2e-7`,
checks finiteness/nonnegativity, and checks each row sum within `2e-5` of one.

The runners use the existing CANN toolchain setup and `msopprof simulator`;
set `CANN_ROOT` if needed. Profiles and tensors appear in `build/sim-*/`.
Simulator ticks and wall time are not physical hardware latency measurements.
See [VALIDATION.md](VALIDATION.md) for measured instruction counts, timings,
and numerical comparisons.
