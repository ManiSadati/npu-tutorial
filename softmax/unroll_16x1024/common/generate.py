import argparse
import numpy as np
p = argparse.ArgumentParser()
p.add_argument('--output', required=True)
p.add_argument('--seed', type=int, default=2026)
a = p.parse_args()
rng = np.random.default_rng(a.seed)
x = rng.normal(0, 3, (16, 1024)).astype(np.float32)
x[1] += 1000
if len(x) > 2: x[2] -= 1000
if len(x) > 3: x[3] = np.linspace(-80, 80, 1024, dtype=np.float32)
x[4] = 0
x[5] = -40; x[5, 511] = 40
x[6] = np.where(np.arange(1024) % 2, -20, 20)
x.tofile(a.output)
print(f'Generated {x.shape} FP32 input, seed={a.seed}: {a.output}')
