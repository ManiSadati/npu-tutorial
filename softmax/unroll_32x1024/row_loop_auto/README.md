# 32x1024 softmax: compiler-selected row-loop optimization

A separate variant of `../row_loop` with only the
`#pragma clang loop unroll(disable)` directive removed. All earlier variants
are preserved. The build script is byte-identical to the row-loop variant:
standard CCEC `-O2`, `-std=c++17`, A5 `dav-c310-vec` and the existing required
runtime/link options. No additional unroll flags or pragmas are used.

The source keeps a row loop and explicitly expands sixteen vector operations
per row. Inputs remain in registers through max; those registers retain the
exponentials through sum and division. The compiler may optimize/unroll the
row loop according to its usual heuristics. Validation inspects actual executed
loop instructions and data accesses rather than assuming the source loop stays.

One vector block computes the full normalized 32x1024 output. DMA geometry,
UB addresses, inputs, tolerances, arithmetic, and batching match sibling
variants. Thirty-two rows use two explicit sixteen-row batches. This experiment
changes no other aspect of the kernel or its host.

```bash
cd ~/npu-tutorial/softmax/unroll_32x1024/row_loop_auto
bash run.sh
# Reuse a matching build:
bash run.sh --no-build
```

Requires the same CANN toolchain, host compiler and NumPy as the siblings.
The runner profiles with `msopprof simulator`, then checks every output and
row normalization against NumPy float64. Host/generator/checker remain shared
in `../common`; keep that folder when copying the example.

See [VALIDATION.md](VALIDATION.md) and the
[combined comparison](../../ROW_LOOP_RESULTS.md). All six new shapes ran
concurrently; simulator wall time includes contention, and simulator ticks
are not physical hardware latency measurements.
