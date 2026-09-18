# Validation: 32 rows, usual compiler settings without an unroll pragma

Validated 2026-09-17 with CANN 9.1.0-beta.3, CCEC `-O2`, `dav-c310-vec`,
`Ascend950PR_9599`, and one vector block. All six new variants ran concurrently.
The baseline and pragma variants below are saved earlier runs, not rerun here.

The new build script is byte-identical to `../row_loop/build.sh`. The only
kernel-source change is deletion of `#pragma clang loop unroll(disable)`
(one instance per batch). No other compiler options were added or removed.
Builds and shell syntax checks passed.

| Metric | Previous row-loop pragma | New, no pragma |
| --- | ---: | ---: |
| Total ticks | 15750 | 15751 |
| Summed VF duration (us) | 2.254 | 2.254 |
| Vector loads | 512 | 512 |
| Vector stores | 512 | 512 |
| Vector-loop setup instructions | 2 | 2 |

Original baseline: 16291 saved ticks. All instruction-name/count
aggregates match the pragma run, and VF/input/output DMA durations match.
The compiler chose to retain the row loop without the directive. Total-tick
variation is not evidence that removing the pragma improves the compiled
vector computation. No extra UB spill traffic or demand instruction-cache
lookup-MISS entries appeared.

Both input and output tensors are byte-identical to the saved pragma run.
All 32768 normalized outputs and row sums passed the same NumPy checker:

```text
NumPy float64 reference: PASS; correct=32768/32768, max_abs_error=8.93486203e-08, max_row_sum_error=1.79018744e-07
```

VF batch durations (us): 1.127, 1.127.
Summed input/output DMA: 2.940/2.738 us.
Model wall time: 107.4870 s, including contention from parallel runs.
No UB overlap or vector arithmetic diagnostics appeared. The usual profiler
PC/source-information messages remained, but usable reports were generated.

Artifact directory: `build/sim-7or5bfg0/`, containing tensors,
`run.log`, and `profile/OPPROF_*/simulator/trace.json` plus instruction CSVs.

This validates the normal `-O2` case locally, not physical A5 performance.
The six shapes are single new runs compared with saved earlier runs; small
tick differences should not be treated as robust speedups. See the
[combined comparison](../../ROW_LOOP_RESULTS.md).
