# --- packf32.py ---
"""Module extracted from cascades.py — packf32."""

from __future__ import annotations


def _pack_f32(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()
def _int4_quant(t: np.ndarray) -> Tuple[np.ndarray, float, float]:
    t_min, t_max = t.min(), t.max()
    scale = (t_max - t_min) / 15.0 if t_max != t_min else 1.0
    q = np.round((t - t_min) / scale).clip(0, 15).astype(np.uint8)
    return q, scale, t_min
def _int8_quant(t: np.ndarray) -> Tuple[np.ndarray, float, float]:
    t_min, t_max = t.min(), t.max()
    scale = (t_max - t_min) / 255.0 if t_max != t_min else 1.0
    q = np.round((t - t_min) / scale).clip(0, 255).astype(np.uint8)
    return q, scale, t_min
def _dct_2d_block(t: np.ndarray, block_size: int = 8) -> np.ndarray:
    h, w = t.shape
    out = np.zeros_like(t, dtype=np.float32)
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            blk = t[i : i + block_size, j : j + block_size]
            if blk.size == 0:
                continue
            dct_rows = np.apply_along_axis(_dct_1d, 1, blk)
            out[i : i + block_size, j : j + block_size] = np.apply_along_axis(
                _dct_1d, 0, dct_rows
            )
    return out
def _wavelet_haar_1d(x: np.ndarray) -> np.ndarray:
    n = len(x)
    if n < 2 or (n & (n - 1)) != 0:
        return x.copy()
    out = x.copy().astype(np.float64)
    h = n
    while h > 1:
        half = h // 2
        for i in range(half):
            a = out[i]
            b = out[i + half]
            out[i] = (a + b) / np.sqrt(2)
            out[i + half] = (a - b) / np.sqrt(2)
        h = half
    return out.astype(np.float32)
def _svd_truncate(
    t: np.ndarray, rank: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    m = t.reshape(t.shape[0], -1)
    U, S, Vt = np.linalg.svd(m, full_matrices=False)
    k = min(rank, len(S))
    return (
        U[:, :k].astype(np.float32),
        S[:k].astype(np.float32),
        Vt[:k, :].astype(np.float32),
        k,
    )
def _tt_decompose(t: np.ndarray, rank: int) -> List[np.ndarray]:
    """Simple Tensor Train decomposition via sequential SVD."""
    shape = t.shape
    d = len(shape)
    cores = []
    current = t.copy().astype(np.float64)
    r_prev = 1
    for k in range(d - 1):
        nk = shape[k]
        current = current.reshape(r_prev * nk, -1)
        U, S, Vt = np.linalg.svd(current, full_matrices=False)
        rk = min(rank, len(S))
        core = U[:, :rk].reshape(r_prev, nk, rk).astype(np.float32)
        cores.append(core)
        current = (S[:rk, None] * Vt[:rk, :]).astype(np.float64)
        r_prev = rk
    core_last = current.reshape(r_prev, shape[-1], 1).astype(np.float32)
    cores.append(core_last)
    return cores
def _rans_encode(symbols: np.ndarray, freqs: np.ndarray) -> bytes:
    """Simplified rANS encode."""
    state = 1
    out = []
    cumsum = np.zeros(len(freqs) + 1, dtype=np.int64)
    cumsum[1:] = np.cumsum(freqs)
    total = int(cumsum[-1])
    for s in symbols:
        s_i = int(s) % len(freqs)
        c = int(cumsum[s_i])
        f = int(freqs[s_i])
        while state >= (total << 16):
            out.append(state & 0xFF)
            state >>= 8
        state = (state // f) * total + c + (state % f)
    while state > 0:
        out.append(state & 0xFF)
        state >>= 8
    return bytes(out)
def _lz77_compress(
    data: np.ndarray, window: int = 32
) -> Tuple[List[Tuple[int, int, float]], np.ndarray]:
    flat = data.ravel()
    matches = []
    residuals = []
    i = 0
    while i < len(flat):
        best_len, best_dist = 0, 0
        for d in range(1, min(window, i) + 1):
            ml = 0
            while i + ml < len(flat) and abs(flat[i + ml] - flat[i - d + ml]) < 0.01:
                ml += 1
            if ml > best_len:
                best_len = ml
                best_dist = d
        if best_len >= 2:
            matches.append((best_dist, best_len))
            i += best_len
        else:
            matches.append((-1, 0))
            residuals.append(flat[i])
            i += 1
    return matches, np.array(residuals, dtype=np.float32)
def _cascade_compress(
    tensor: np.ndarray, stages: List[Dict], meta: Dict
) -> Tuple[bytes, dict]:
    """Generic cascade: apply each stage in sequence, unify metadata."""
    flat = tensor.ravel().astype(np.float32)
    residual = flat.copy()
    stage_data = []
    for stage in stages:
        name = stage["name"]
        params = stage.get("params", {})
        stage_data.append((name, params))
    n_stages = len(stage_data)
    level = params.get("level", n_stages) if "level" in params else n_stages
    stages_used = stage_data[:level]
    t = flat.copy()
    for sname, sparams in stages_used:
        if sname == "dct":
            sz = int(np.sqrt(len(t)))
            mat = t[: sz * sz].reshape(sz, sz)
            dct = _dct_2d_block(mat)
            keep = sparams.get("keep", 0.3)
            flat_dct = dct.ravel()
            nk = max(int(len(flat_dct) * keep), 1)
            idx = np.argsort(-np.abs(flat_dct))[:nk]
            vals = flat_dct[idx]
            t = np.zeros(len(flat_dct), dtype=np.float32)
            t[idx] = vals
        elif sname == "wavelet":
            sz = int(np.sqrt(len(t)))
            t = _wavelet_haar_1d(t[: sz * sz])
        elif sname == "tt":
            rank = sparams.get("rank", 4)
            shape_2d = (int(t.shape[0] ** 0.5), int(t.shape[0] ** 0.5))
            mat = t[: shape_2d[0] * shape_2d[1]].reshape(shape_2d[0], shape_2d[1])
            cores = _tt_decompose(mat, rank)
            t = np.concatenate([c.ravel() for c in cores])
        elif sname == "svd":
            rank = sparams.get("rank", 8)
            m = int(t.shape[0] ** 0.5)
            mat = t[: m * m].reshape(m, m)
            U, S, Vt, k = _svd_truncate(mat, rank)
            t = np.concatenate([U.ravel(), S, Vt.ravel()])
        elif sname == "quant":
            bits = sparams.get("bits", 4)
            if bits <= 4:
                q, scale, zero = _int4_quant(t)
                t = q.astype(np.float32)
            else:
                q, scale, zero = _int8_quant(t)
                t = q.astype(np.float32)
        elif sname == "entropy":
            n_bins = sparams.get("n_bins", 64)
            t_norm = (
                ((t - t.min()) / (t.max() - t.min() + 1e-30) * (n_bins - 1))
                .astype(np.int32)
                .clip(0, n_bins - 1)
            )
            freqs = np.bincount(t_norm, minlength=n_bins).astype(np.int64)
            freqs = np.maximum(freqs, 1)
            encoded = _rans_encode(t_norm, freqs)
            t = np.frombuffer(encoded, dtype=np.uint8).astype(np.float32)
        elif sname == "lz77":
            matches, residuals = _lz77_compress(t)
            t = np.concatenate([residuals, np.array([len(matches)], dtype=np.float32)])
        elif sname == "spike":
            bits = sparams.get("bits", 4)
            q, scale, zero = _int4_quant(t)
            mask = (np.abs(t) > np.percentile(np.abs(t), 70)).astype(np.float32)
            t = q.astype(np.float32) * mask
    out_data = _pack_f32(t)
    meta_out = dict(
        meta,
        shape=tensor.shape,
        n_stages=level,
        stage_names=[s[0] for s in stages_used],
    )
    return out_data, meta_out