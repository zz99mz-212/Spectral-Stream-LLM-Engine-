"""
Advanced quantization methods beyond simple block quant.
"""

from typing import Dict, Tuple
import numpy as np
from spectralstream.core.math_primitives import LloydMaxQuantizer


def _snr(orig, recon):
    o = orig.ravel().astype(np.float64)
    r = recon.ravel().astype(np.float64)
    mse = np.mean((o - r) ** 2)
    return float(10.0 * np.log10(np.var(o) / (mse + 1e-30)))


def _norm_quantiles(n_levels: int = 16) -> np.ndarray:
    """Quantiles of N(0,1), symmetric, scaled to [-1, 1]."""
    return np.array([
        -1.0, -0.70756877, -0.54220910, -0.41681885,
        -0.31090474, -0.21594631, -0.12734098, -0.04209538,
        0.04209538, 0.12734098, 0.21594631, 0.31090474,
        0.41681885, 0.54220910, 0.70756877, 1.0,
    ], dtype=np.float32)


def nf4_quantize(tensor: np.ndarray, block_size: int = 64) -> Tuple[Dict, float, float]:
    """NormalFloat4 quantization from QLoRA paper.
    Uses quantiles of N(0,1) as levels.
    """
    t = np.asarray(tensor, dtype=np.float32)
    s = t.shape
    f = t.reshape(-1, s[-1])
    nr, nc = f.shape
    p = (block_size - nc % block_size) % block_size
    if p:
        f = np.pad(f, ((0, 0), (0, p)))
    nb = f.shape[1] // block_size
    b = f.reshape(nr, nb, block_size)
    sc = np.maximum(np.max(np.abs(b), axis=2, keepdims=True), 1e-10)
    norm = np.clip(b / sc, -1.0, 1.0)
    levels = _norm_quantiles(16)
    idx = np.argmin(np.abs(norm[:, :, :, None] - levels[None, None, None, :]), axis=3).astype(np.uint8)
    pairs = idx.reshape(nr, nb, block_size // 2, 2)
    packed = (pairs[..., 0].astype(np.uint8) << 4) | pairs[..., 1].astype(np.uint8)
    comp = {'q': packed, 'sc': sc.astype(np.float32), 'shape': s, 'bs': block_size, 'p': p, 'method': 'nf4'}
    ratio = float(t.nbytes / (packed.nbytes + sc.nbytes))
    uv_e = (packed.astype(np.uint16) >> 4).astype(np.uint8)
    uv_o = (packed & 0x0F).astype(np.uint8)
    uv_d = np.empty((nr, nb, block_size), dtype=np.uint8)
    uv_d[..., 0::2] = uv_e
    uv_d[..., 1::2] = uv_o
    d = sc * levels[uv_d]
    d = d.reshape(f.shape)
    if p:
        d = d[:, :-p]
    rv = d.reshape(s)
    return comp, ratio, _snr(t, rv)


def kmeans_quantize(tensor: np.ndarray, n_clusters: int = 16) -> Tuple[Dict, float, float]:
    """K-means based quantization. Cluster weights into k centroids."""
    t = np.asarray(tensor, dtype=np.float32)
    flat = t.ravel()
    rng = np.random.RandomState(42)
    centroids = np.zeros(n_clusters, dtype=np.float32)
    centroids[0] = flat[rng.randint(len(flat))]
    for i in range(1, n_clusters):
        dists = np.min((flat[:, None] - centroids[:i][None, :]) ** 2, axis=1)
        probs = dists / (dists.sum() + 1e-30)
        centroids[i] = flat[rng.choice(len(flat), p=probs)]
    for _ in range(20):
        labels = np.argmin(np.abs(flat[:, None] - centroids[None, :]), axis=1).astype(np.int32)
        new_c = np.array([flat[labels == i].mean() if np.any(labels == i) else centroids[i] for i in range(n_clusters)])
        if np.allclose(centroids, new_c, atol=1e-6):
            break
        centroids = new_c
    labels = np.argmin(np.abs(flat[:, None] - centroids[None, :]), axis=1).astype(np.uint8)
    comp = {'labels': labels, 'centroids': centroids.astype(np.float32), 'shape': t.shape, 'n_clusters': n_clusters, 'method': 'kmeans_quant'}
    ratio = float(t.nbytes / (labels.nbytes + centroids.nbytes))
    rv = centroids[labels].reshape(t.shape).astype(np.float32)
    return comp, ratio, _snr(t, rv)


def _e8_nearest(x):
    """Find nearest E8 lattice point to an 8D vector."""
    v1 = np.floor(x + 0.5)
    d1 = v1 - x
    if int(np.sum(v1)) % 2 != 0:
        i = np.argmax(np.abs(d1))
        v1[i] -= np.sign(d1[i]) if d1[i] != 0 else 1.0
    v2 = np.floor(2.0 * x + 0.5) / 2.0
    d2 = v2 - x
    if int(np.sum(2.0 * v2)) % 2 != 0:
        i = np.argmax(np.abs(d2))
        v2[i] -= np.sign(d2[i]) * 0.5 if d2[i] != 0 else 0.5
    return v1 if np.sum((v1 - x) ** 2) < np.sum((v2 - x) ** 2) else v2


def e8_lattice_quantize(tensor: np.ndarray, block_size: int = 8) -> Tuple[Dict, float, float]:
    """E8 lattice quantization. Uses Gosset lattice for optimal sphere packing."""
    t = np.asarray(tensor, dtype=np.float32).ravel()
    n_orig = len(t)
    p = (8 - n_orig % 8) % 8
    if p:
        t = np.pad(t, (0, p))
    n_blocks_8d = len(t) // 8
    blocks = t.reshape(-1, 8)
    sc = np.maximum(np.max(np.abs(blocks), axis=1, keepdims=True), 1e-10)
    norm = blocks / sc
    lattice_pts = np.array([_e8_nearest(norm[i]) for i in range(n_blocks_8d)])
    stored = np.round(lattice_pts * 2.0).astype(np.int8)
    comp = {'pts': stored, 'sc': sc.astype(np.float32).ravel(), 'shape': tensor.shape, 'p': p, 'method': 'e8_lattice'}
    ratio = float(tensor.nbytes / (stored.nbytes + sc.nbytes))
    lp = stored.astype(np.float32) / 2.0
    d = (lp * sc).ravel()
    if p:
        d = d[:-p]
    rv = d.reshape(tensor.shape).astype(np.float32)
    return comp, ratio, _snr(tensor, rv)


def lloyd_max_quantize(tensor: np.ndarray, n_levels: int = 16) -> Tuple[Dict, float, float]:
    """Lloyd-Max optimal quantizer. Iteratively optimizes decision boundaries."""
    t = np.asarray(tensor, dtype=np.float32)
    q = LloydMaxQuantizer(n_bits=int(np.log2(n_levels)))
    q.train(t)
    indices, centroids = q.compress(t)
    comp = {'indices': indices, 'centroids': centroids, 'shape': t.shape, 'n_bits': q.n_bits, 'method': 'lloyd_max'}
    ratio = float(t.nbytes / (indices.nbytes + centroids.nbytes))
    rv = q.decompress(indices, t.shape).astype(np.float32)
    return comp, ratio, _snr(t, rv)


def stochastic_round_quantize(tensor: np.ndarray, bits: int = 8) -> Tuple[Dict, float, float]:
    """Stochastic rounding: round up/down with probability proportional to
    distance from nearest quantization levels. Unbiased rounding."""
    t = np.asarray(tensor, dtype=np.float32)
    s = t.shape
    f = t.reshape(-1, s[-1])
    nr, nc = f.shape
    bs = 256
    p = (bs - nc % bs) % bs
    if p:
        f = np.pad(f, ((0, 0), (0, p)))
    nb = f.shape[1] // bs
    b = f.reshape(nr, nb, bs)
    sc = np.maximum(np.max(np.abs(b), axis=2, keepdims=True), 1e-10)
    rng = np.random.RandomState(42)
    max_q = 2 ** (bits - 1) - 1
    norm = b / sc
    floor = np.floor(norm * max_q)
    frac = norm * max_q - floor
    rnd = (rng.uniform(size=frac.shape) < np.abs(frac)).astype(np.float32) * np.sign(frac + 1e-30)
    q = (floor + rnd).clip(-max_q, max_q).astype(np.int16)
    comp = {'q': q, 'sc': sc.astype(np.float32), 'shape': s, 'bits': bits, 'p': p, 'bs': bs, 'method': 'stochastic_round'}
    ratio = float(t.nbytes / (q.nbytes + sc.nbytes))
    d = sc * (q.astype(np.float32) / max_q)
    d = d.reshape(f.shape)
    if p:
        d = d[:, :-p]
    rv = d.reshape(s)
    return comp, ratio, _snr(t, rv)


def outlier_aware_quantize(tensor: np.ndarray, bits: int = 4, outlier_threshold: float = 3.0) -> Tuple[Dict, float, float]:
    """Detect and preserve outlier values separately.
    Keeps outlier in FP16, quantizes the rest with more precision.
    """
    t = np.asarray(tensor, dtype=np.float32)
    flat = t.ravel()
    mu, std = float(np.mean(flat)), float(np.std(flat))
    mask = np.abs(flat - mu) > outlier_threshold * std
    mask_packed = np.packbits(mask)
    outliers = flat[mask].astype(np.float16)
    inliers = flat[~mask]
    bs = 128
    n_in = len(inliers)
    p = (bs - n_in % bs) % bs
    if p:
        inliers = np.pad(inliers, (0, p))
    nb = len(inliers) // bs
    bi = inliers.reshape(-1, bs)
    sc_in = np.maximum(np.max(np.abs(bi), axis=1, keepdims=True), 1e-10)
    max_q = 2 ** (bits - 1) - 1
    q = np.round(bi / sc_in * max_q).clip(-max_q, max_q).astype(np.int8)
    comp = {'q': q, 'sc_in': sc_in.astype(np.float32), 'outliers': outliers, 'mask': mask_packed,
            'shape': t.shape, 'bits': bits, 'p': p, 'n_in': n_in, 'method': 'outlier_aware'}
    comp_bytes = q.nbytes + sc_in.nbytes + outliers.nbytes + mask_packed.nbytes
    ratio = float(t.nbytes / max(comp_bytes, 1))
    d_in = sc_in * (q.astype(np.float32) / max_q)
    full_mask = np.unpackbits(mask_packed)[:len(flat)].astype(bool)
    d_flat = np.zeros(len(flat), dtype=np.float32)
    d_flat[~full_mask] = d_in.ravel()[:n_in]
    d_flat[full_mask] = np.asarray(outliers, dtype=np.float32)
    rv = d_flat.reshape(t.shape)
    return comp, ratio, _snr(t, rv)


def adaptive_group_quantize(tensor: np.ndarray, bits: int = 4, n_groups: int = 8) -> Tuple[Dict, float, float]:
    """Adaptive grouping: split channels into groups based on magnitude,
    quantize each group with its own scale.
    """
    t = np.asarray(tensor, dtype=np.float32)
    s = t.shape
    f = t.reshape(-1, s[-1])
    nr, nc = f.shape
    norms = np.linalg.norm(f, axis=0)
    order = np.argsort(norms)
    group_size = nc // n_groups
    groups = []
    for g in range(n_groups):
        start = g * group_size
        end = start + group_size if g < n_groups - 1 else nc
        col_idx = order[start:end]
        groups.append(col_idx)
    max_q = 2 ** (bits - 1) - 1
    q_data = np.empty_like(f, dtype=np.int8)
    sc_data = np.zeros((nr, n_groups), dtype=np.float32)
    for g, cols in enumerate(groups):
        block = f[:, cols]
        sc = np.maximum(np.max(np.abs(block), axis=1, keepdims=True), 1e-10)
        q_data[:, cols] = np.round(block / sc * max_q).clip(-max_q, max_q).astype(np.int8)
        sc_data[:, g] = sc.ravel()
    comp = {'q': q_data, 'sc': sc_data, 'groups': np.concatenate(groups).astype(np.int32), 'group_sizes': np.array([len(g) for g in groups], dtype=np.int32),
            'shape': s, 'n_groups': n_groups, 'method': 'adaptive_group'}
    ratio = float(t.nbytes / (q_data.nbytes + sc_data.nbytes + comp['groups'].nbytes + comp['group_sizes'].nbytes))
    d = np.zeros_like(f, dtype=np.float32)
    for g, cols in enumerate(groups):
        d[:, cols] = sc_data[:, g:g+1] * (q_data[:, cols].astype(np.float32) / max_q)
    rv = d.reshape(s)
    return comp, ratio, _snr(t, rv)


if __name__ == '__main__':
    t = np.random.randn(256, 256).astype(np.float32)
    for fn, nm in [(nf4_quantize, 'NF4'), (kmeans_quantize, 'KMeans'),
                   (e8_lattice_quantize, 'E8Lattice'), (lloyd_max_quantize, 'LloydMax'),
                   (stochastic_round_quantize, 'StochasticRound'), (outlier_aware_quantize, 'OutlierAware'),
                   (adaptive_group_quantize, 'AdaptiveGroup')]:
        data, ratio, snr = fn(t)
        print(f"{nm}: {ratio:.2f}x, SNR={snr:.1f}dB")
