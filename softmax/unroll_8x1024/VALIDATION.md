# Validation: 8x1024 row softmax

Validated 2026-09-17 with CANN 9.1.0-beta.3, `dav-c310-vec`,
`Ascend950PR_9599`, CCEC `-O2`, one vector block per kernel. Both builds and
shell syntax checks passed. The two 2-row and two 8-row simulations were
launched concurrently, as requested. Each has separate binaries and artifacts.
No older 4-row simulation was rerun in this batch.

| Metric | Looped baseline | Unrolled/register reuse |
| --- | ---: | ---: |
| Total simulator ticks | 5531 | 5429 |
| VF duration (us) | 0.694 | 0.601 |
| Input DMA duration (us) | 0.848 | 0.873 |
| Output DMA duration (us) | 0.758 | 0.763 |
| Model wall time (s) | 46.5712 | 44.0804 |
| Vector loads | 384 | 128 |
| Vector stores | 256 | 128 |
| Vector-loop setup instructions | 25 | 0 |
| Vector exponentials | 128 | 128 |
| Vector divisions | 128 | 128 |
| Horizontal maxima | 8 | 8 |
| Horizontal sums | 8 | 8 |

Both compared all 8192 normalized outputs against NumPy float64 and
passed. Inputs were byte-identical within the pair; outputs were also
byte-identical. Maximum absolute error was 2.40262591e-8 and maximum row-sum
error was 8.67686105e-8 for both.

Unrolled used 1.84% fewer total ticks and 13.40% less VF time.
The total-tick difference for two rows is effectively a tie, not evidence of
a robust end-to-end speedup. These are single parallel-run measurements;
wall times include host contention and should not be compared directly with
the earlier sequential four-row runs. Simulator ticks are not hardware latency.

Instruction counts include all `RV_VLD*` and `RV_VST*` addressing forms.
They match the intended input/output accesses with no extra UB spill traffic.
Unrolled has no executed vector-loop setup instructions. EXP/DIV and horizontal
reduction counts match baseline; unrolling plus register reuse saves memory
operations and intermediate-buffer barriers without changing the softmax
formula. Both share input/output DMA sizes and UB addresses within their pair.
No UB overlap or vector arithmetic diagnostics appeared. The profiler emitted
its usual missing PC/source information messages but generated usable traces.

Artifacts relative to this folder:

- `baseline/build/sim-dGmlY8qA/`
- `unrolled/build/sim-jetusqMl/`

Each contains input/output/reference tensors, `run.log`, and
`profile/OPPROF_*/simulator/trace.json` plus instruction CSVs.
