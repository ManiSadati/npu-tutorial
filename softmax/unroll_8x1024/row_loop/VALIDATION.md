# Validation: retained row loop, 8x1024 softmax

Validated 2026-09-17 with CANN 9.1.0-beta.3, `dav-c310-vec`,
`Ascend950PR_9599`, CCEC `-O2`, one vector block. The six new row-loop
variants ran concurrently. Existing baseline/fully-unrolled versions were
not rebuilt or rerun for this comparison. Builds and shell syntax checks passed.

| Metric | Historical baseline | Historical fully unrolled | New row loop |
| --- | ---: | ---: | ---: |
| Total ticks | 5531 | 5429 | 5456 |
| Summed VF duration (us) | 0.694 | 0.601 | 0.598 |

The new run passed all 8192 outputs against stable NumPy float64 softmax.
Input and output files were byte-identical to the earlier fully unrolled run.

```text
NumPy float64 reference: PASS; correct=8192/8192, max_abs_error=2.40262591e-08, max_row_sum_error=8.67686105e-08
```

Measured data vector loads/stores: **128/128**, matching the
fully unrolled access counts without extra UB spill traffic. The profiler
reports **1 vector-loop setup instruction(s)**: one per batch,
confirming the row loop was retained. There are 128 vector EXP,
128 vector DIV, and 8 horizontal max/sum instructions each.
The instruction-cache log has **zero** `instr req lookup tag MISS` entries.
This is a log-event observation, not a guarantee about all cache behavior.

VF batch durations (us): 0.598.
Summed input/output DMA durations: 0.857/0.768 us.
Model wall time: 34.3028 seconds; concurrent runs include host contention.
No UB overlap or vector arithmetic diagnostics were reported. Usual profiler
PC/source-information messages remained, but usable reports were produced.

Artifact directory relative to this variant: `build/sim-kgVkPJfq/`.
It contains tensors, `run.log`, and profiler traces/instruction CSVs.

These are single new simulator runs compared with saved earlier measurements,
not a simultaneous rerun of all three implementations or physical hardware
measurements. Small tick differences should not be overinterpreted. See
[the combined comparison](../../ROW_LOOP_RESULTS.md).
