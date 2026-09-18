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

# Fused CCE validation — 2026-09-10

- Built and ran with the baseline A5 target (`dav-c310-vec`, CANN 9.1.0 beta.3,
  simulator Ascend950PR_9599, `-O2`), four blocks, 100×1024 FP32.
- Complete simulator run exited 0; all 102400 outputs passed the unchanged
  NumPy reference checker, including the 100 row-sum checks.
- Input bytes match the saved baseline run exactly.
- `python3 verify_repair.py` passed every prefix invariant and the final
  numerical check; deliberately omitting either repair fails.
- All new shell scripts passed `bash -n`.

```text
NumPy float64 reference: PASS; correct=102400/102400, max_abs_error=8.93486203e-08, max_row_sum_error=1.8803739e-07
Other backend vs reference: PASS; correct=102400/102400, max_abs_error=8.93486203e-08, max_row_sum_error=1.8803739e-07
Backend comparison: PASS; correct=102400/102400, max_abs_error=2.98023224e-08, max_row_sum_error=1.8803739e-07
```

Artifacts relative to `softmax/`:

- Fused: `fused/cce/build/sim-0SZF7oaI/run.log`
- Direct comparison: `fused/cce/build/sim-0SZF7oaI/baseline-comparison.log`
- Baseline output: `baseline/cce/build/sim-v4lL1ICe/output.bin`

The new run reports 38028 simulator ticks; the saved baseline run reports
36407 ticks. These runs are not a controlled performance study, and the fused
variant has not demonstrated a speedup. Removing the intermediate buffer adds
repair exponentials and a recurrence dependency. No physical A5 performance
claim is made.

The kernel has five localized edits: 14 inserted and 9 deleted source lines
relative to baseline, with no baseline file changes or hidden helper bodies.

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
