# Fused CCE row softmax with reduction repair

Inspired by `~/math-rewrite.md`, sections 3.1 and Benchmark C: introduce running
max/normalizer state, repair the old normalizer when the maximum changes, and
fuse the formerly dependent maximum/exponential/sum stages.

```bash
cd ~/npu-tutorial/softmax/fused/cce
bash run.sh
```

Uses the baseline's 100×1024 FP32 contract, four vector blocks, DMA/sync sequence,
A5 `dav-c310-vec` target, compiler options, host, input generator and Python
checker. `bash run.sh --no-build` skips rebuilding. Setup and timeout behavior
are the same as [baseline](../baseline/README.md). The common host and Python
code are read directly from `../baseline/common`, so retain the baseline folder.

## Five localized mathematical edits

1. Initialize the running normalizer before the first loop.
2. Save the old per-lane maximum before updating it.
3. Repair the running normalizer and accumulate the new exponential inside that loop.
4. Replace the second scan with a repair from per-lane maxima to the row maximum.
5. Recompute final exponentials from input in the output loop.

The literal kernel diff is **14 inserted lines and 9 deleted lines**, a net
increase of five lines; it is not a five-line patch. No helper or macro hides
additional kernel logic. Inspect it with:

```bash
diff -u ../baseline/cce/src/kernel.cpp cce/src/kernel.cpp
```

## Repair derivation

For lane j after consuming a prefix of 64-element chunks, retain:

```text
m[j] = max_k x[k,j]
s[j] = sum_k exp(x[k,j] - m[j])
```

For the next value v[j]:

```text
new_m[j] = max(m[j], v[j])
alpha[j] = exp(m[j] - new_m[j])
new_s[j] = alpha[j] * s[j] + exp(v[j] - new_m[j])
```

In real arithmetic, each old summand becomes
`exp(x - m) * exp(m - new_m) = exp(x - new_m)`, proving that the prefix invariant
is preserved. Initialize m to the most negative finite float and s to zero.

The lanes may have different maxima. Once all sixteen chunks have been read:

```text
M = max_j m[j]
S = sum_j s[j] * exp(m[j] - M)
y[k,j] = exp(x[k,j] - M) / S
```

The second repair is essential: summing the uncorrected per-lane normalizers
would normalize each lane against a different scale. FP32 reassociation and
exponential rounding change numerical results; correctness uses the baseline's
existing tolerances, not bitwise equality or a formal IEEE equivalence claim.

`python3 verify_repair.py` checks the invariant at every prefix in FP32 against
a float64 reference, with growing/falling maxima, lane offsets, extreme constant
rows and a late peak. It also verifies that dropping either repair fails.

## Fusion and cost

The maximum/exp/sum statistics now share one input scan. Final output still
needs a second scan after the denominator is known. The baseline had three
vector scans (two input scans and one exponential-buffer scan); this version
has two input scans and removes the 4096-byte-per-row exponential store/load.
The now-unused `ex` pointer declaration is retained to minimize source changes.
GM traffic, kernel launch count and final output size are unchanged.

Repair and output recomputation add exponentials and a loop-carried dependency.
This is a fusion experiment, not a demonstrated performance improvement.
Use simulator output for functional checks; measure hardware before claiming a
speedup. Input domain, shape and tolerances remain those of the baseline.

See [VALIDATION.md](VALIDATION.md) for the passing simulator/reference/baseline comparisons and performance limitations.
