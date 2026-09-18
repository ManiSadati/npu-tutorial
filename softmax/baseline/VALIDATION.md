# Softmax validation: 10 × 1024 — 2026-09-14

All three rebuilt implementations exited 0 and passed all 10240 output
comparisons against stable float64 NumPy softmax, including all 10 row sums.
Input files match byte-for-byte. Each run launches four AIV blocks; row counts
are 3, 3, 2, 2. The FP32 shape, host I/O and shared Python checks are consistent.

| Backend | Simulator wall time | Simulator ticks | Max absolute error |
| --- | --- | --- | --- |
| Baseline CCE | 42.8595 seconds | 5601 | 1.76337554e-08 |
| Baseline PTO-ISA | 66.3285 seconds | 8302 | 1.95239250e-08 |
| Fused CCE | 46.3338 seconds | 5748 | 1.95239250e-08 |

These are simulator measurements, not physical A5 kernel latency. Runs were
concurrent. CANN 9.1.0 beta.3, dav-c310-vec, Ascend950PR_9599, -O2.
The known PTO TROWMAX negative-infinity-initializer diagnostics also appeared
in this passing run. The checker and tolerances are unchanged.

- cce log (relative to softmax/): `baseline/cce/build/rows10-validation/sim-pDgt1ZT6/run.log`
- pto log (relative to softmax/): `baseline/pto-isa/build/rows10-validation/sim-FvGnnpcr/run.log`
- fused log (relative to softmax/): `fused/cce/build/rows10-validation/sim-J55vjtfW/run.log`

Every tensor is 40960 bytes, shape 10×1024, FP32. Rebuild with `bash run.sh`
after updating; old 100-row binaries/artifacts cannot be reused with the new
checker. Validation used separate `build/rows10-validation` directories.

Direct comparisons against baseline CCE:

```text
NumPy float64 reference: PASS; correct=10240/10240, max_abs_error=1.76337554e-08, max_row_sum_error=6.99883502e-08
Other backend vs reference: PASS; correct=10240/10240, max_abs_error=1.9523925e-08, max_row_sum_error=9.69705892e-08
Backend comparison: PASS; correct=10240/10240, max_abs_error=2.98023224e-08, max_row_sum_error=6.99883502e-08
NumPy float64 reference: PASS; correct=10240/10240, max_abs_error=1.76337554e-08, max_row_sum_error=6.99883502e-08
Other backend vs reference: PASS; correct=10240/10240, max_abs_error=1.9523925e-08, max_row_sum_error=4.82841815e-08
Backend comparison: PASS; correct=10240/10240, max_abs_error=2.98023224e-08, max_row_sum_error=6.99883502e-08
```

## Historical 100-row validation

The following results describe the previous shape.

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

## msopprof runner validation — 2026-09-14

All three minimal runner commands exited 0, produced parsed profiler reports,
and passed the unchanged full-tensor comparison. Commands use the existing
env.sh and the absolute CANN msopprof path; activate_ptoas is not sourced.
Validation used host execution because msopprof local control sockets are
blocked in the Codex sandbox.

- baseline CCE: `baseline/cce/build/rows10-validation/sim-QI0e5ryR/profile/OPPROF_20260914184302_HRQBYOBHCSHQHXSG/simulator/` — parsed timeline and 2 nonempty instruction CSV files; all 10240 outputs passed.
- baseline PTO-ISA: `baseline/pto-isa/build/rows10-validation/sim-1mFtZ8h4/profile/OPPROF_20260914184205_HJQDKVTRCTWKPFFE/simulator/` — parsed timeline and 2 nonempty instruction CSV files; all 10240 outputs passed.
- fused CCE: `fused/cce/build/rows10-validation/sim-W7iSjkJw/profile/OPPROF_20260914184209_MSYPAGHMDTVABAYL/simulator/` — parsed timeline and 2 nonempty instruction CSV files; all 10240 outputs passed.

These runs retain the installed profiler warnings about PC-start fallback and
missing debug_line information. They provide instruction/timeline reports,
without source-line call stacks.
