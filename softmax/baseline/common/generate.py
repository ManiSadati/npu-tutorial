"""Identical finite FP32 inputs for both softmax backends."""
import argparse
import numpy as np
p = argparse.ArgumentParser()
p.add_argument("--output", required=True)
p.add_argument("--seed", type=int, default=2026)
a = p.parse_args()
rng = np.random.default_rng(a.seed)
x = rng.normal(0, 3, (100, 1024)).astype(np.float32)
x[0] = 0
x[1] = 1000
x[2] = -1000
x[3] = np.linspace(-80, 80, 1024, dtype=np.float32)
x[4] = -40; x[4, 511] = 40
x[5] = np.where(np.arange(1024) % 2, -20, 20)
x[6] += 1000
x[7] -= 1000
x.tofile(a.output)
print(f"Generated {x.shape} FP32 input, seed={a.seed}: {a.output}")
