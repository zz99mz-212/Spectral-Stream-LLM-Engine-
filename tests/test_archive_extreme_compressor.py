"""
ExtremeCompressor validation test.
Tests hierarchical compression at 100:1 to 5000:1 on realistic synthetic FP32 weights.
"""
import sys
import numpy as np

sys.path.insert(0, '/home/mike/Documents/Github/SpectralStream')
from spectralstream.unified_intelligence_engine import ExtremeCompressor, QualityValidator


def create_realistic_weight(shape, effective_rank=None, noise_level=0.001):
    """Create a weight matrix with realistic LLM properties:
    - Low-rank structure (power law singular value decay)
    - Heavy-tailed singular values (decay ~ 1/i^1.5)
    - Small amount of Gaussian noise
    """
    m, n = shape
    if effective_rank is None:
        effective_rank = min(m, n) // 4
    p = 1.5
    k = min(m, n)
    s = np.arange(1, k + 1, dtype=np.float64) ** (-p)
    s = s / np.linalg.norm(s) * np.sqrt(m * n) * 0.1
    U, _ = np.linalg.qr(np.random.randn(m, k))
    V, _ = np.linalg.qr(np.random.randn(n, k))
    W = U[:, :len(s)] @ np.diag(s) @ V.T[:len(s), :]
    W += np.random.randn(m, n).astype(np.float64) * noise_level
    return W.astype(np.float32)


if __name__ == '__main__':
    np.random.seed(42)
    v = QualityValidator()
    ec = ExtremeCompressor()

    test_configs = [
        ('small_dense', (256, 512), 64, 0.001),
        ('medium_attn', (1536, 2048), 128, 0.001),
        ('medium_ffn', (1536, 6144), 128, 0.001),
        ('large_embed', (25600, 512), 512, 0.01),
    ]

    results = []
    for name, shape, rank, noise in test_configs:
        t = create_realistic_weight(shape, rank, noise)
        fp32_size = t.nbytes
        print(f'\n{name:15s} shape={str(shape):20s} fp32={fp32_size/1024**2:.1f}MB')
        for target in [100, 500, 2000, 5000]:
            try:
                c = ec.compress(t, target_ratio=float(target))
                d = ec.decompress(c)
                m = v.evaluate(t, d)
                ratio = fp32_size / max(c['compressed_bytes'], 1)
                print(f'  Target {target:>5d}:1 -> {ratio:>8.1f}:1  '
                      f'SNR={m["snr_db"]:6.1f}dB  '
                      f'rel_err={m["relative_error"]:.6f}  '
                      f'{"OK" if m["relative_error"] < 0.01 else "LO"}')
                results.append({
                    'name': name, 'target': target, 'ratio': ratio,
                    'snr_db': m['snr_db'], 'rel_err': m['relative_error'],
                    'passed': m['relative_error'] < 0.01
                })
            except Exception as ex:
                print(f'  Target {target:>5d}:1 -> ERROR: {ex}')

    print(f'\n{"=" * 60}')
    print('Summary:')
    passes = sum(1 for r in results if r['passed'])
    print(f'  Tests passed: {passes}/{len(results)}')
    for r in results:
        status = 'OK' if r['passed'] else 'LO'
        print(f'  {status} {r["name"]:15s} target={r["target"]:5d}:1 '
              f'actual={r["ratio"]:7.1f}:1 '
              f'SNR={r["snr_db"]:5.1f}dB err={r["rel_err"]:.6f}')
