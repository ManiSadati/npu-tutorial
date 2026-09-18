# Row softmax: 20x1024 loops versus register reuse

Both CCE kernels return full normalized 20x1024 FP32 softmax on one A5
vector block. This extends the [4-row experiment](../unroll_4x1024/README.md).

- `baseline`: row loop plus max, exp/sum, and division loops; saves
  exponentials to UB and reloads them.
- `unrolled`: explicit operations for all 20 rows, with no source loops;
  each row's sixteen vectors remain in registers through max, exp/sum and
  division. Registers can be reused between rows. No online repair is used.

## UB and batching

Both process 1 batch(es) of 20 rows. Each batch uses one
81920-byte GM input DMA and one equally sized output DMA, with matching
barriers and offsets in both versions. Total logical input and output traffic
is 81920 bytes each. The 20-row output is never transposed.

The shared UB layout has input at 0, baseline exponentials at 81920,
and output at 163840, ending at 245760 bytes. The unrolled kernel
leaves the middle buffer region unused to retain matching addresses.

The configured UB is 253952 bytes (248 KiB). With this three-buffer layout,
`floor(253952 / (3 * 1024 * 4)) = 20` rows fit in one batch: 245760 bytes
(240 KiB). Twenty-one rows need 258048 bytes and exceed the limit. This is
not a universal softmax row limit: compacting or overwriting buffers changes
capacity. For example, two separate input/output buffers alone could hold
31 rows. Here both variants keep the original matched three-buffer layout.

For 32 rows both kernels explicitly process two batches of 16 rows and wait
for output completion before reusing UB. Thus unrolled still has no source
loops, but the 32-row workload has two input/output DMA pairs and two vector
scopes. It is not a single-resident-matrix test like the smaller shapes.

Expected loads/stores: baseline 960/640, unrolled 320/320; both
perform 320 vector exponentials/divisions and 20 horizontal maxima/sums.
This tests unrolling and explicit register reuse together.

## Run and check

```bash
cd ~/npu-tutorial/softmax/unroll_20x1024/baseline
bash run.sh
cd ../unrolled
bash run.sh
```

Use `--no-build` only with a matching existing binary. Both variants share
`common/` host, input generator and NumPy float64 checker. Tests include random
rows, +/-1000 offsets, a wide ramp, a constant row, a peak and alternating
values. Every output is checked with `rtol=2e-5`, `atol=2e-7`, finiteness and
nonnegativity; each row sum must be within `2e-5` of one.

The runners use the existing CANN configuration and `msopprof simulator`.
See [VALIDATION.md](VALIDATION.md) for the six concurrent 16/20/32-row runs.
Concurrent simulator wall time includes host contention. Simulator ticks
and profile timings are not physical A5 measurements.
