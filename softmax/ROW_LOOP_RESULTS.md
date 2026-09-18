# Softmax: retain the row loop, unroll only vectors within each row

Validated 2026-09-17. Six new `row_loop` variants were added without changing
the existing baseline or fully unrolled kernels. All six new variants ran
concurrently. The baseline and fully unrolled numbers below are saved earlier
runs, not reruns in this batch. All use one vector block and matching shape,
GM layout, UB addresses, DMA geometry and numerical output contract per shape.

## Results

| Rows × 1024 | Baseline ticks | Fully unrolled ticks | Row-loop ticks | Baseline VF (us) | Fully unrolled VF (us) | Row-loop VF (us) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 3012 | 3010 | 3054 | 0.201 | 0.196 | 0.198 |
| 4 | 3866 | 3840 | 3845 | 0.372 | 0.335 | 0.331 |
| 8 | 5531 | 5429 | 5456 | 0.694 | 0.601 | 0.598 |
| 16 | 8935 | 8588 | 8545 | 1.339 | 1.129 | 1.127 |
| 20 | 10584 | 10227 | 10100 | 1.661 | 1.427 | 1.392 |
| 32 | 16291 | 18587 | 15750 | 2.595 | 3.821 | 2.254 |

All values passed NumPy float64 validation and were bit-identical to the
saved fully unrolled outputs for their respective shapes. Every new trace
retains one row-loop setup per batch, and preserves the expected one input
vector load and one output vector store per 64 values, without extra spills.

For 2–20 rows the vector times are close to fully unrolled. At 32 rows,
retaining the row loop avoids the earlier second-batch slowdown: both batches
take **1.127 us**, compared with fully unrolled **1.129 + 2.692 us**. Total
ticks are **15.3% fewer than fully unrolled** and **3.3% fewer than baseline**
using the first historical runs. The earlier 32-row repeat gave 16242/18518
baseline/unrolled ticks and the same VF pattern, so the comparison is not
dependent on selecting the first historical pair.

The new 32-row instruction-cache log has zero demand lookup-MISS entries;
the earlier fully unrolled log had seven. This supports the instruction-fetch
explanation for that slowdown. It does not prove every saved cycle is due to
the cache: retaining a loop also changes scheduling and address calculation.
Fewer data accesses alone do not explain the change because the two versions
have the same load/store counts.

## Method and limits

Keep `for (row ...)` with `#pragma clang loop unroll(disable)`. Inside it,
explicitly name and operate on sixteen vector registers. Input stays live
through max; those same registers hold exponentials through sum and division.
There is no intermediate exponential UB buffer traffic and no online repair.

Brief pseudocode below starts after GM input is loaded into UB. Each `x[row,i]`
is one 64-element vector; `horizontal_*` reduces its lanes to a scalar.

```text
Baseline:
  for row:
    m = vector(-infinity)
    for i = 0..15: m = max(m, load_UB(x[row,i]))
    M = horizontal_max(m); s = vector(0)
    for i = 0..15:
      e = exp(load_UB(x[row,i]) - M)
      store_UB(saved_exp[row,i], e); s += e
    S = horizontal_sum(s); wait_for_exp_stores()
    for i = 0..15: store_UB(y[row,i], load_UB(saved_exp[row,i]) / S)

Inner-unrolled, row loop:
  for row:
    v0, ..., v15 = load_UB(x[row,0]), ..., load_UB(x[row,15])
    M = horizontal_max(elementwise_max(v0, ..., v15))
    v0, ..., v15 = exp(v0 - M), ..., exp(v15 - M)
    S = horizontal_sum(v0 + ... + v15)
    store_UB(y[row,0], v0 / S); ...; store_UB(y[row,15], v15 / S)
```

The `...` operations are explicit source statements, not inner loops; the
fully unrolled version also expands the outer row loop. Scalar M/S are
broadcast for vector arithmetic. The normal-settings variant below needs no pragma.

Why it helps: the 16-row traces show UB loads **768 → 256**, stores
**512 → 256**, and intermediate-buffer barriers **16 → 0**, with unchanged
math counts and no extra spills. VF time falls **1.339 → 1.127 us** (15.8%);
largely unchanged GM transfers limit the overall tick improvement to about 4–5%.

Unrolling is not mathematically necessary: a compiler could retain values in
registers from loop-based source too. But a runtime-indexed temporary array
does not guarantee register storage; if placed in UB, reloads remain. Explicit
vector names make reuse straightforward. Keeping the outer row loop also
avoids the large instruction stream and observed 32-row cache slowdown.

The 32-row case still uses two explicit 16-row batches. Twenty rows remains
the largest resident batch for the matched three-buffer address layout.
No input transposition or inter-row grouping was introduced. All rows are
processed individually, reusing registers.

Single new runs and historical comparisons do not establish hardware timings
or a universal speed ranking. In particular, tiny differences for small shapes
are insufficient to claim an overall speedup. Concurrent simulator wall times
are not physical kernel latency.

## New folders

- [2 rows](unroll_2x1024/row_loop/README.md): `softmax/unroll_2x1024/row_loop/`
- [4 rows](unroll_4x1024/row_loop/README.md): `softmax/unroll_4x1024/row_loop/`
- [8 rows](unroll_8x1024/row_loop/README.md): `softmax/unroll_8x1024/row_loop/`
- [16 rows](unroll_16x1024/row_loop/README.md): `softmax/unroll_16x1024/row_loop/`
- [20 rows](unroll_20x1024/row_loop/README.md): `softmax/unroll_20x1024/row_loop/`
- [32 rows](unroll_32x1024/row_loop/README.md): `softmax/unroll_32x1024/row_loop/`

Run `bash run.sh` inside any new folder. Each README links to its validation
record and artifact directory. The variants share their shape’s existing
`common/` host/generator/checker; retain that directory when copying them.

## Normal compiler settings: no row-loop pragma

Added separate `row_loop_auto/` variants for every shape. Only the pragma
preventing row-loop unrolling was removed; build scripts and all other
kernel logic remain identical to `row_loop`. All six new simulations ran
concurrently and passed, with outputs bit-identical to the saved pragma runs.

| Rows × 1024 | Baseline ticks | Fully unrolled ticks (improvement) | Inner unrolled, row loop ticks (improvement) |
| --- | ---: | ---: | ---: |
| 2 | 3012 | 3010 (+0.07%) | 3059 (-1.56%) |
| 4 | 3866 | 3840 (+0.67%) | 3844 (+0.57%) |
| 8 | 5531 | 5429 (+1.84%) | 5471 (+1.08%) |
| 16 | 8935 | 8588 (+3.88%) | 8501 (+4.86%) |
| 20 | 10584 | 10227 (+3.37%) | 10090 (+4.67%) |
| 32 | 16291 | 18587 (-14.09%) | 15751 (+3.31%) |

All three columns use the usual `-O2` settings without unroll pragmas.
Improvement is `(baseline_ticks - variant_ticks) / baseline_ticks × 100`;
positive means fewer ticks, negative means more ticks. These are existing
measurements from different batches; no simulations were rerun for this table.
The row-loop column uses `row_loop_auto`, not the pragma variant.

The compiler retained the row loop without being instructed to do so. All
instruction-name/count aggregates and VF/DMA phase durations match the pragma
runs across all six shapes. The small total-tick differences therefore do not
show a vector-performance benefit from removing the pragma. No extra spills
or demand instruction-cache lookup-MISS entries were observed.

The normal-settings 16-row run has 4.9% fewer total ticks than the saved
original baseline (8501 vs 8935). This supports the same register-reuse
improvement without a special unroll directive; it does not establish a
precise repeatable 4.9% hardware speedup.

New runnable folders and per-shape validation links:

- [2 rows, no pragma](unroll_2x1024/row_loop_auto/README.md)
- [4 rows, no pragma](unroll_4x1024/row_loop_auto/README.md)
- [8 rows, no pragma](unroll_8x1024/row_loop_auto/README.md)
- [16 rows, no pragma](unroll_16x1024/row_loop_auto/README.md)
- [20 rows, no pragma](unroll_20x1024/row_loop_auto/README.md)
- [32 rows, no pragma](unroll_32x1024/row_loop_auto/README.md)
