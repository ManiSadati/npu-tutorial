# A5 validation: 20x1024 full row softmax

Validated 2026-09-17: CANN 9.1.0-beta.3, `dav-c310-vec`,
`Ascend950PR_9599`, CCEC `-O2`, one vector block. All six 16/20/32-row
baseline/unrolled simulations ran concurrently as requested. Builds and shell
syntax checks passed. Pairs have matching DMA geometry and UB addresses.

| Metric | Baseline | Unrolled/register reuse |
| --- | ---: | ---: |
| Total ticks | 10584 | 10227 |
| Summed VF duration (us) | 1.661 | 1.427 |
| Summed input DMA duration (us) | 1.747 | 1.789 |
| Summed output DMA duration (us) | 1.669 | 1.662 |
| Model wall time (s) | 81.6246 | 71.1798 |
| Vector loads | 960 | 320 |
| Vector stores | 640 | 320 |
| Vector-loop setup instructions | 61 | 0 |
| Vector exponentials | 320 | 320 |
| Vector divisions | 320 | 320 |
| Horizontal maxima | 20 | 20 |
| Horizontal sums | 20 | 20 |

Both passed all 20480 outputs against stable NumPy float64 softmax.
Inputs and outputs were byte-identical within the pair. Maximum absolute error
was 8.93486203e-8; maximum row-sum error was 1.79018744e-7 for both.
No UB overlap or vector arithmetic diagnostics were reported. The usual missing
PC/source-information profiler messages remained; reports were generated.

Unrolled used 3.37% fewer ticks and 14.09% less VF time.
Counts include all `RV_VLD*`/`RV_VST*` addressing forms and match the intended
accesses, with no additional UB spill traffic. Unrolled has no vector-loop
setup instructions. Both compute the same number of exponentials, divisions,
and horizontal reductions. This tests unrolling and register reuse together.

These are simulator measurements, not physical A5 performance. Concurrent
wall times include host contention. Separate shapes and batching policies
must not be treated as a controlled linear scaling experiment.

Artifacts relative to this folder:

- `baseline/build/sim-lFYkNKll/`
- `unrolled/build/sim-zwFtRH48/`

Each includes tensors, `run.log`, and `profile/OPPROF_*/simulator/trace.json`
plus instruction CSVs.

## Maximum resident batch

Twenty rows use 245760 bytes (240 KiB) with three 80 KiB buffers. This is the
maximum for the unchanged baseline layout in 248 KiB UB. It is not the maximum
possible after buffer aliasing/compaction or streaming. Unrolled leaves the
exponential-buffer address region unused to keep the comparison matched.
