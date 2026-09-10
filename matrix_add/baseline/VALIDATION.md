# Single-core addition validation: 10 × 512 — 2026-09-10

Both rebuilt implementations exited 0 and passed exact comparisons for all
5120 outputs. Each run has exactly one block_start and block_end event,
AIV core_id=0, block_id=0. A/B inputs match byte-for-byte across the backends.

| Backend | Correct outputs | Maximum error | Simulator ticks | Simulator wall time |
| --- | --- | --- | --- | --- |
| CCE | 5120/5120 | 0 | 12890 | 126.803 seconds |
| PTO-ISA | 5120/5120 | 0 | 12972 | 124.524 seconds |

These are simulator measurements, not physical A5 kernel times. Runs were
concurrent. CANN 9.1.0 beta.3, dav-c310-vec, Ascend950PR_9599, -O2.

```text
NumPy FP32 addition: PASS; correct=5120/5120, failed_rows=[], max_abs_error=0
Other backend vs NumPy: PASS; correct=5120/5120, failed_rows=[], max_abs_error=0
Backend comparison: PASS; correct=5120/5120, failed_rows=[], max_abs_error=0
```

Current artifacts relative to this directory:

- CCE: `cce/build/sim-wAFqf9QS/run.log`
- PTO-ISA: `pto-isa/build/sim-EwsAcVFn/run.log`

Each run contains 20480-byte a.bin, b.bin, output.bin and reference.bin files.
Old 100-row artifacts are retained in separate run directories; their outputs
are incompatible with the new checker. Rebuild after updating to the new shape
using `bash run.sh`, rather than reusing an old binary with `--no-build`.

## Historical 100-row run

The following results describe the previous shape, not the current testcase.

# Single-core addition validation — 2026-09-10

Both A5 simulator runs exited 0 and passed exact NumPy FP32 addition checks
for all 51200 elements (100 rows). Both have exactly one block_start and one
block_end event: AIV core_id=0, block_id=0. All rows execute in that one block.
Input A and B files match byte-for-byte across the backends.

| Backend | Correct outputs | Max absolute error | Simulator ticks | Simulator wall time |
| --- | --- | --- | --- | --- |
| CCE | 51200/51200 | 0 | 118953 | 711.879 seconds |
| PTO-ISA | 51200/51200 | 0 | 119753 | 747.458 seconds |

These are operator simulator measurements, not physical A5 kernel times.
Runs were concurrent; wall times are not a controlled performance comparison.
CANN 9.1.0 beta.3, dav-c310-vec, Ascend950PR_9599 simulator, -O2.

```text
NumPy FP32 addition: PASS; correct=51200/51200, failed_rows=[], max_abs_error=0
Other backend vs NumPy: PASS; correct=51200/51200, failed_rows=[], max_abs_error=0
Backend comparison: PASS; correct=51200/51200, failed_rows=[], max_abs_error=0
```

Saved artifacts relative to this folder:

- CCE: `cce/build/sim-BoECwNpD/run.log`
- PTO-ISA: `pto-isa/build/sim-uZ0ioitt/run.log`

Each run directory also contains a.bin, b.bin, output.bin, reference.bin and
simulator traces. Build artifacts are ignored by git. The Python checker
accepted the reference output and rejected a corrupted final element (NaN,
representing a missing store). Both sets of shell scripts passed bash -n.
