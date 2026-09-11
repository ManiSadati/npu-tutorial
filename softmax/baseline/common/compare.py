"""Validate all 100x1024 results, optionally comparing the other backend too."""
import argparse
from pathlib import Path
import numpy as np

def load(path):
    if Path(path).stat().st_size != 100 * 1024 * 4:
        raise ValueError(f"{path}: expected exactly 409600 bytes (100x1024 FP32)")
    return np.fromfile(path, dtype=np.float32).reshape(100, 1024)

def check(label, got, expected):
    close = np.isclose(got, expected, rtol=2e-5, atol=2e-7)
    good = np.isfinite(got) & (got >= 0) & close
    row_error = np.abs(got.astype(np.float64).sum(axis=1) - 1)
    passed = bool(good.all() and (row_error <= 2e-5).all())
    print(f"{label}: {'PASS' if passed else 'FAIL'}; "
          f"correct={good.sum()}/{got.size}, max_abs_error={np.max(np.abs(got-expected)):.9g}, "
          f"max_row_sum_error={row_error.max():.9g}")
    for row, col in np.argwhere(~good)[:8]:
        print(f"  [{row}, {col}] got={got[row,col]:.9g} expected={expected[row,col]:.9g}")
    return passed

p = argparse.ArgumentParser()
p.add_argument("--input", required=True)
p.add_argument("--output", required=True)
p.add_argument("--other", help="Optional output from the other backend; must use the same input")
a = p.parse_args()
x = load(a.input).astype(np.float64)
if not np.isfinite(x).all():
    raise ValueError("This baseline supports finite input only")
e = np.exp(x - x.max(axis=1, keepdims=True))
reference = e / e.sum(axis=1, keepdims=True)
reference.astype(np.float32).tofile(Path(a.output).with_name("reference.bin"))
y = load(a.output)
ok = check("NumPy float64 reference", y, reference)
if a.other:
    other = load(a.other)
    ok = check("Other backend vs reference", other, reference) and ok
    ok = check("Backend comparison", y, other) and ok
raise SystemExit(0 if ok else 1)
