# A5 simulator validation: 4x1024 looped versus unrolled softmax

Validated 2026-09-17, CANN 9.1.0-beta.3, `dav-c310-vec`,
`Ascend950PR_9599`, CCEC `-O2`, one vector block. Both builds and shell syntax
checks passed. Simulations ran sequentially with identical profiling settings.
Each kernel loads the complete 16 KiB input once and writes 16 KiB output once.

## First comparison

| Metric | Looped baseline | Fully unrolled/register reuse |
| --- | ---: | ---: |
| NumPy float64 comparison | 4096/4096 PASS | 4096/4096 PASS |
| Maximum absolute error | 2.40262591e-8 | 2.40262591e-8 |
| Maximum row-sum error | 8.67686105e-8 | 8.67686105e-8 |
| Total simulator ticks | 3866 | 3840 |
| Vector-function duration | 0.372 us | 0.335 us |
| Input DMA duration | 0.509 us | 0.541 us |
| Output DMA duration | 0.461 us | 0.458 us |
| Profile core duration | 1.58 us | 1.57 us |
| Model wall time | 23.0256 s | 23.7686 s |
| Data vector loads | 192 | 64 |
| Data vector stores | 128 | 64 |
| Vector EXP / DIV | 64 / 64 | 64 / 64 |
| Horizontal MAX / ADD | 4 / 4 | 4 / 4 |
| Vector store/load memory barriers | 4 | 0 |
| Executed vector-loop setup instructions | 13 | 0 |

Inputs and outputs were byte-identical between variants. Both passed all
four row normalization checks, and their direct comparison had zero error.
The unrolled profile counts 16 `RV_VLDI` plus 48 `RV_VLDS` loads, and 16
`RV_VSTI` plus 48 `RV_VSTS` stores. Baseline uses `RV_VLD`/`RV_VST`.
These counts match exactly the requested data movement without additional
UB spill/reload traffic. The emitted unrolled trace has no vector loop setup.

Register reuse reduces UB load instructions by 66.7% and stores by 50%.
It leaves the arithmetic count unchanged and removes intermediate-buffer
barriers. Vector-function duration fell by **9.95%**, while total ticks fell
by only **0.67%**. Input/output DMA and other overhead remain substantial;
input DMA timing also differed slightly despite the same logical transfer.
This comparison changes unrolling, register reuse, and address calculation
together, so it does not isolate each contribution.

Artifacts relative to this directory:

- `baseline/build/sim-sZP8LgVq/`
- `unrolled/build/sim-CofOrW93/`

Each contains tensors, `run.log`, and the parsed profiler timeline/instruction
CSVs. The profiler emitted its usual missing PC/source information messages,
but produced usable traces and completed successfully. Simulator ticks and
wall time do not establish hardware latency or a general speedup.

## Sequential repeat

Both kernels were rerun without rebuilding to check the small total-tick gap.

| Metric | Looped baseline | Unrolled/register reuse |
| --- | ---: | ---: |
| Total ticks | 3868 | 3854 |
| VF duration | 0.372 us | 0.335 us |
| Model wall time | 22.9076 s | 22.9857 s |

Both passed 4096/4096 outputs again. Inputs and outputs were byte-identical to
their first runs. Input/output DMA durations were unchanged from the first
pair. The vector improvement repeated exactly, while total-tick reduction
was only 0.36% (0.67% initially). These runs support a roughly 10% vector-phase
improvement but only a small observed end-to-end simulator improvement.

Repeat artifacts:

- `baseline/build/sim-80TUXcjT/`
- `unrolled/build/sim-j15lUxvs/`
