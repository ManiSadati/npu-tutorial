# A5 validation: 16x1024 full row softmax

Validated 2026-09-17: CANN 9.1.0-beta.3, `dav-c310-vec`,
`Ascend950PR_9599`, CCEC `-O2`, one vector block. All six 16/20/32-row
baseline/unrolled simulations ran concurrently as requested. Builds and shell
syntax checks passed. Pairs have matching DMA geometry and UB addresses.

| Metric | Baseline | Unrolled/register reuse |
| --- | ---: | ---: |
| Total ticks | 8935 | 8588 |
| Summed VF duration (us) | 1.339 | 1.129 |
| Summed input DMA duration (us) | 1.438 | 1.474 |
| Summed output DMA duration (us) | 1.376 | 1.363 |
| Model wall time (s) | 62.3128 | 56.648 |
| Vector loads | 768 | 256 |
| Vector stores | 512 | 256 |
| Vector-loop setup instructions | 49 | 0 |
| Vector exponentials | 256 | 256 |
| Vector divisions | 256 | 256 |
| Horizontal maxima | 16 | 16 |
| Horizontal sums | 16 | 16 |

Both passed all 16384 outputs against stable NumPy float64 softmax.
Inputs and outputs were byte-identical within the pair. Maximum absolute error
was 8.93486203e-8; maximum row-sum error was 1.79018744e-7 for both.
No UB overlap or vector arithmetic diagnostics were reported. The usual missing
PC/source-information profiler messages remained; reports were generated.

Unrolled used 3.88% fewer ticks and 15.68% less VF time.
Counts include all `RV_VLD*`/`RV_VST*` addressing forms and match the intended
accesses, with no additional UB spill traffic. Unrolled has no vector-loop
setup instructions. Both compute the same number of exponentials, divisions,
and horizontal reductions. This tests unrolling and register reuse together.

These are simulator measurements, not physical A5 performance. Concurrent
wall times include host contention. Separate shapes and batching policies
must not be treated as a controlled linear scaling experiment.

Artifacts relative to this folder:

- `baseline/build/sim-C1LoKEbd/`
- `unrolled/build/sim-vZqIwPO8/`

Each includes tensors, `run.log`, and `profile/OPPROF_*/simulator/trace.json`
plus instruction CSVs.
