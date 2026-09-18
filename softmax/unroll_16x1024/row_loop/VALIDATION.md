# Validation: retained row loop, 16x1024 softmax

Validated 2026-09-17 with CANN 9.1.0-beta.3, `dav-c310-vec`,
`Ascend950PR_9599`, CCEC `-O2`, one vector block. The six new row-loop
variants ran concurrently. Existing baseline/fully-unrolled versions were
not rebuilt or rerun for this comparison. Builds and shell syntax checks passed.

| Metric | Historical baseline | Historical fully unrolled | New row loop |
| --- | ---: | ---: | ---: |
| Total ticks | 8935 | 8588 | 8545 |
| Summed VF duration (us) | 1.339 | 1.129 | 1.127 |

The new run passed all 16384 outputs against stable NumPy float64 softmax.
Input and output files were byte-identical to the earlier fully unrolled run.

```text
NumPy float64 reference: PASS; correct=16384/16384, max_abs_error=8.93486203e-08, max_row_sum_error=1.79018744e-07
```

Measured data vector loads/stores: **256/256**, matching the
fully unrolled access counts without extra UB spill traffic. The profiler
reports **1 vector-loop setup instruction(s)**: one per batch,
confirming the row loop was retained. There are 256 vector EXP,
256 vector DIV, and 16 horizontal max/sum instructions each.
The instruction-cache log has **zero** `instr req lookup tag MISS` entries.
This is a log-event observation, not a guarantee about all cache behavior.

VF batch durations (us): 1.127.
Summed input/output DMA durations: 1.460/1.358 us.
Model wall time: 57.7959 seconds; concurrent runs include host contention.
No UB overlap or vector arithmetic diagnostics were reported. Usual profiler
PC/source-information messages remained, but usable reports were produced.

Artifact directory relative to this variant: `build/sim-4r20jonN/`.
It contains tensors, `run.log`, and profiler traces/instruction CSVs.

These are single new simulator runs compared with saved earlier measurements,
not a simultaneous rerun of all three implementations or physical hardware
measurements. Small tick differences should not be overinterpreted. See
[the combined comparison](../../ROW_LOOP_RESULTS.md).
