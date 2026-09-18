# A5 validation: 32x1024 full row softmax

Validated 2026-09-17: CANN 9.1.0-beta.3, `dav-c310-vec`,
`Ascend950PR_9599`, CCEC `-O2`, one vector block. All six 16/20/32-row
baseline/unrolled simulations ran concurrently as requested. Builds and shell
syntax checks passed. Pairs have matching DMA geometry and UB addresses.

| Metric | Baseline | Unrolled/register reuse |
| --- | ---: | ---: |
| Total ticks | 16291 | 18587 |
| Summed VF duration (us) | 2.595 | 3.821 |
| Summed input DMA duration (us) | 2.901 | 2.957 |
| Summed output DMA duration (us) | 2.741 | 2.735 |
| Model wall time (s) | 123.755 | 121.664 |
| Vector loads | 1536 | 512 |
| Vector stores | 1024 | 512 |
| Vector-loop setup instructions | 98 | 0 |
| Vector exponentials | 512 | 512 |
| Vector divisions | 512 | 512 |
| Horizontal maxima | 32 | 32 |
| Horizontal sums | 32 | 32 |

Both passed all 32768 outputs against stable NumPy float64 softmax.
Inputs and outputs were byte-identical within the pair. Maximum absolute error
was 8.93486203e-8; maximum row-sum error was 1.79018744e-7 for both.
No UB overlap or vector arithmetic diagnostics were reported. The usual missing
PC/source-information profiler messages remained; reports were generated.

Unrolled used 14.09% more ticks and 47.24% more VF time.
Counts include all `RV_VLD*`/`RV_VST*` addressing forms and match the intended
accesses, with no additional UB spill traffic. Unrolled has no vector-loop
setup instructions. Both compute the same number of exponentials, divisions,
and horizontal reductions. This tests unrolling and register reuse together.

These are simulator measurements, not physical A5 performance. Concurrent
wall times include host contention. Separate shapes and batching policies
must not be treated as a controlled linear scaling experiment.

Artifacts relative to this folder:

- `baseline/build/sim-q8EK6pCA/`
- `unrolled/build/sim-QEgd47q6/`

Each includes tensors, `run.log`, and `profile/OPPROF_*/simulator/trace.json`
plus instruction CSVs.

## Second-batch slowdown

Both versions process two 16-row batches, reusing 192 KiB UB with full output
completion before reuse. VF durations for batches 1 and 2 were:

| Batch | Baseline | Unrolled |
| --- | ---: | ---: |
| First | 1.339 us | 1.129 us |
| Second | 1.256 us | 2.692 us |

The unrolled first batch retains the 16-row benefit. Its second batch has a
long delay before the first RV instruction and several large issue gaps.
The `dump/core0.veccore0.rvec.icache1.dump` log contains seven
`instr req lookup tag MISS` entries for unrolled, versus zero for baseline,
and 144 prefetch-MISS entries versus one. Entries are log events, not unique
misses or a calibrated stall-cycle count. The instruction-cache evidence is
consistent with a code-fetch penalty in the large unrolled program. It is
not evidence of data-register spills, and this experiment does not isolate
instruction-cache behavior as the sole cause. A parallel repeat checks whether
the observed timing pattern persists without modifying either binary.

## Parallel repeat of the 32-row pair

Both kernels reran concurrently without rebuilding and passed all 32768
outputs. Inputs and outputs were byte-identical to their first runs.

| Variant | Repeat ticks | Batch VF durations (us) |
| --- | ---: | --- |
| baseline | 16242 | 1.339, 1.256 |
| unrolled | 18518 | 1.129, 2.692 |

Unrolled again took about 14% more ticks. The second-batch vector slowdown
repeated; the larger row count did not produce a monotonically growing benefit.

Repeat artifacts:

- `baseline/build/sim-sSJbKjdZ/`
- `unrolled/build/sim-2lTa5efB/`
