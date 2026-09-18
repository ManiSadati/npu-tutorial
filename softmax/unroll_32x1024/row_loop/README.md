# 32x1024 softmax: row loop with unrolled vector operations

This new variant retains the row loop and explicitly expands the sixteen
64-lane vectors within each row. The `baseline` and `unrolled` siblings are
unchanged. `#pragma clang loop unroll(disable)` requests that the compiler
retain the row loop; validation inspects executed loop instructions.

Each row loads its input vectors once, retains them through the maximum
reduction, then overwrites those registers with exponentials. After the sum
reduction, it divides and stores without saving/reloading exponentials in UB.
Arithmetic and reduction order match the fully unrolled variant. Only one row
must be live at a time. There is no online max/sum repair.

The kernel runs one vector block, processes 2 batch(es) of 16
rows, and returns all 32768 normalized FP32 values. UB input/output offsets,
DMA sizes and barriers match the sibling variants: input at 0, output at
131072, highest used address ending at 196608 bytes. The old
exponential-buffer address range remains unused. Each batch has one input DMA,
one vector scope and one output DMA; 32 rows use two explicit batches.

Expected data loads/stores are 512/512, matching fully unrolled and
avoiding baseline's 1536/1024. Row offsets are calculated at runtime;
vector offsets within each row are fixed. This trades repeated row-loop
control/address work for a smaller instruction body.

```bash
cd ~/npu-tutorial/softmax/unroll_32x1024/row_loop
bash run.sh
# Reuse this variant's build:
bash run.sh --no-build
```

The host, generator and checker are shared with the siblings in `../common`.
The checker validates all values against NumPy float64 softmax, finiteness,
nonnegativity and row normalization. Profiling uses the same CANN/A5 setup.
See [VALIDATION.md](VALIDATION.md) for this run and
[the combined comparison](../../ROW_LOOP_RESULTS.md) for all six shapes.
