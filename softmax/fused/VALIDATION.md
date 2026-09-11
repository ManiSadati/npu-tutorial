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
