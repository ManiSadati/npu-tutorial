# Simulator validation — 2026-09-09

Branch: `mani/softmax-rewrite`. CANN 9.1.0 beta.3, `dav-c310-vec`,
`Ascend950PR_9599`, CCE optimization `-O2`.
PTO-ISA checkout: `896d8ec69aaf5b623fead5afcae7a657fa784a2b` (existing whitespace
change in TBinOp.hpp retained; no PTO-ISA source edits).

Both builds passed and both complete simulator runs exited 0. Each compared
all 102400 FP32 values against stable float64 NumPy softmax and checked all
100 row sums. The input files are byte-identical.

| Backend | Maximum absolute error vs NumPy | Maximum row-sum error | Result |
| --- | --- | --- | --- |
| CCE | 8.93486203e-08 | 1.88037390e-07 | PASS, 102400/102400 |
| PTO-ISA | 1.33515099e-07 | 1.90507188e-07 | PASS, 102400/102400 |

Direct comparison:

```text
NumPy float64 reference: PASS; correct=102400/102400, max_abs_error=8.93486203e-08, max_row_sum_error=1.8803739e-07
Other backend vs reference: PASS; correct=102400/102400, max_abs_error=1.33515099e-07, max_row_sum_error=1.90507188e-07
Backend comparison: PASS; correct=102400/102400, max_abs_error=1.78813934e-07, max_row_sum_error=1.8803739e-07
```

Saved run artifacts (relative to this directory):

- CCE: `cce/build/sim-v4lL1ICe/run.log`
- PTO-ISA: `pto-isa/build/sim-8NDc0KgA/run.log`

Each run directory includes input.bin, output.bin, reference.bin and simulator
traces. Build directories are ignored by git. Local simulator wall times were
about 295 seconds for CCE and 415 seconds for PTO-ISA, with concurrent runs;
these are not hardware kernel times or a controlled performance comparison.

The Python checker also accepted a reference tensor and rejected a deliberately
corrupted last-row element. All shell scripts passed `bash -n`.

## PTO simulator diagnostics

The simulator prints `vec_err_idata_inf_nan_t0` on VMAX while executing TROWMAX.
The checked-out PTO header `include/pto/npu/a5/TRowReduce.hpp` initializes the
maximum with `Padding<float>::Min`; `common.hpp` defines it as the -infinity
bit pattern 0xff800000. These diagnostics occurred in the passing run above;
outputs remained finite and matched the reference. No simulator errors are
filtered or hidden. Investigate any different diagnostics or failed comparisons.

This validates the fixed finite-input FP32 baseline in the operator simulator,
not physical A5 hardware or arbitrary shapes/dtypes.
