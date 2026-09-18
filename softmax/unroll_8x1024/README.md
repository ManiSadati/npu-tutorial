# Row softmax: 8x1024 loops versus register reuse

Both CCE kernels return full normalized 8x1024 FP32 softmax, on one A5
vector block. They use one 32768-byte input DMA and one equally sized
output DMA. This extends the [4-row experiment](../unroll_4x1024/README.md).

- `baseline`: outer row loop, inner loops for max, exp/sum, and division;
  exponentials are saved to UB and reloaded.
- `unrolled`: explicit operations for all 8 rows, no source loops; each row's
  sixteen vectors remain in registers through max, exp/sum, and division.
  Rows have separate register lifetimes. No online repair is used.

Both retain the same UB output offset 65536 and address span 98304 bytes.
The unrolled version leaves the intermediate buffer region unused. There is
no artificial 20.5 KiB UB restriction in these experiments.

Expected vector loads/stores: baseline 384/256, unrolled 128/128.
Both perform 128 exponentials, 128 divisions, and 8 horizontal maxima
and sums. The comparison changes register reuse and unrolling together.

```bash
cd ~/npu-tutorial/softmax/unroll_8x1024/baseline
bash run.sh
cd ../unrolled
bash run.sh
```

Both use the same local `common/` host and Python generation/checking files.
`bash run.sh --no-build` reuses a matching binary. Setup and profiling match
the 4-row experiment. Output is checked against stable NumPy float64 softmax
with `rtol=2e-5`, `atol=2e-7`, finiteness/nonnegativity, and row-sum tolerance
`2e-5`. Inputs contain seeded random values and a large positive row offset;
when present, later rows add a negative offset, a wide ramp, a constant row,
a peak, and alternating values.

See [VALIDATION.md](VALIDATION.md) for the requested parallel-run results.
Simulator ticks are not physical hardware latency; concurrent simulator wall
clock times also reflect host contention.
