"""Identical seeded FP32 matrix inputs for both single-core backends."""
import argparse
import numpy as np
p = argparse.ArgumentParser()
p.add_argument("--a", required=True)
p.add_argument("--b", required=True)
p.add_argument("--seed", type=int, default=2026)
args = p.parse_args()
rng = np.random.default_rng(args.seed)
a = rng.normal(0, 10, (100, 512)).astype(np.float32)
b = rng.normal(0, 10, (100, 512)).astype(np.float32)
a[0] = 0; b[0] = 0
b[1] = -a[1]
b[2] = 0
# Distinct row/column coordinates expose row indexing and missing tail rows.
a[3] = np.arange(512, dtype=np.float32); b[3] = -256
a[4::3] = np.arange(4, 100, 3, dtype=np.float32)[:, None]
a.tofile(args.a); b.tofile(args.b)
print(f"Generated A and B: shape=(100,512), FP32, seed={args.seed}")
