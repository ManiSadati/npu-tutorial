"""Check every element of C = A + B, including the final row."""
import argparse
from pathlib import Path
import numpy as np

def load(path):
    if Path(path).stat().st_size != 10 * 512 * 4:
        raise ValueError(f"{path}: expected exactly 20480 bytes (10x512 FP32)")
    return np.fromfile(path, dtype=np.float32).reshape(10, 512)

def check(label, got, expected):
    correct = np.isfinite(got) & (got == expected)
    failed_rows = np.flatnonzero(~correct.all(axis=1))
    print(f"{label}: {'PASS' if correct.all() else 'FAIL'}; "
          f"correct={correct.sum()}/{got.size}, failed_rows={failed_rows.tolist()}, "
          f"max_abs_error={np.max(np.abs(got.astype(np.float64)-expected)):.9g}")
    for row, col in np.argwhere(~correct)[:8]:
        print(f"  [{row},{col}] got={got[row,col]} expected={expected[row,col]}")
    return bool(correct.all())

p = argparse.ArgumentParser()
p.add_argument("--a", required=True)
p.add_argument("--b", required=True)
p.add_argument("--output", required=True)
p.add_argument("--other", help="Other backend output for identical A and B")
args = p.parse_args()
a, b = load(args.a), load(args.b)
if not (np.isfinite(a).all() and np.isfinite(b).all()):
    raise ValueError("Test inputs must be finite")
reference = np.add(a, b, dtype=np.float32)
reference.tofile(Path(args.output).with_name("reference.bin"))
y = load(args.output)
ok = check("NumPy FP32 addition", y, reference)
if args.other:
    other = load(args.other)
    ok = check("Other backend vs NumPy", other, reference) and ok
    ok = check("Backend comparison", y, other) and ok
raise SystemExit(0 if ok else 1)
