"""Check lane-wise repair invariants and exercise both repair terms in FP32."""
import numpy as np


def online(x, local_repair=True, lane_repair=True, check_prefix=False):
    m = np.full((x.shape[0], 64), -np.finfo(np.float32).max, dtype=np.float32)
    s = np.zeros_like(m)
    for k in range(16):
        v = x[:, k * 64:(k + 1) * 64]
        new = np.maximum(m, v)
        if local_repair:
            s *= np.exp(m - new)
        s += np.exp(v - new)
        m = new
        if check_prefix:
            prefix = x[:, :(k + 1) * 64].reshape(x.shape[0], k + 1, 64).astype(np.float64)
            expected = np.exp(prefix - m[:, None, :]).sum(axis=1)
            np.testing.assert_allclose(s, expected, rtol=2e-6, atol=2e-6)
    maximum = m.max(axis=1, keepdims=True)
    if lane_repair:
        s *= np.exp(m - maximum)
    return np.exp(x - maximum) / s.sum(axis=1, keepdims=True)


if __name__ == '__main__':
    rng = np.random.default_rng(2026)
    x = rng.normal(0, 3, (100, 1024)).astype(np.float32)
    # Increasing and decreasing chunk/lane maxima force nontrivial repairs.
    ramp = np.arange(16, dtype=np.float32).repeat(64) * 4
    x[:25] += ramp
    x[25:50] -= ramp
    x[50:75] += np.tile(np.arange(64, dtype=np.float32), 16)
    x[75] = 1000; x[76] = -1000; x[77] = 0
    x[78] = -40; x[78, -1] = 40
    x[79:] += 1000
    z = x.astype(np.float64)
    e = np.exp(z - z.max(axis=1, keepdims=True))
    reference = e / e.sum(axis=1, keepdims=True)
    result = online(x, check_prefix=True)
    np.testing.assert_allclose(result, reference, rtol=2e-5, atol=2e-7)
    np.testing.assert_allclose(result.sum(axis=1), 1, rtol=0, atol=2e-5)
    assert not np.allclose(online(x, local_repair=False), reference, rtol=2e-5, atol=2e-7)
    assert not np.allclose(online(x, lane_repair=False), reference, rtol=2e-5, atol=2e-7)
    print('PASS: every prefix invariant and final FP32 output; omitting either repair fails.')
