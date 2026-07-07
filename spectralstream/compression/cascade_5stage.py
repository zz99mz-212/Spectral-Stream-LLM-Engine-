from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spectralstream.compression.honest_metrics import serialized_nbytes, ErrorMetrics
from spectralstream.compression._dtype_utils import (
    detect_storage_dtype,
    convert_to_storage,
    convert_from_storage,
    encode_dtype_code,
    decode_dtype_code,
)
from spectralstream.compression.methods.functional.ergodic_hyperfunction import (
    ErgodicHyperfunction,
)


_IRRATIONAL_PRIMES: np.ndarray = np.array(
    [
        2,
        3,
        5,
        7,
        11,
        13,
        17,
        19,
        23,
        29,
        31,
        37,
        41,
        43,
        47,
        53,
        59,
        61,
        67,
        71,
        73,
        79,
        83,
        89,
        97,
        101,
        103,
        107,
        109,
        113,
        127,
        131,
        137,
        139,
        149,
        151,
        157,
        163,
        167,
        173,
        179,
        181,
        191,
        193,
        197,
        199,
        211,
        223,
        227,
        229,
        233,
        239,
        241,
        251,
        257,
        263,
        269,
        271,
        277,
        281,
        283,
        293,
    ],
    dtype=np.float64,
)


def _closest_divisor(n: int, target: int) -> int:
    target = max(1, min(target, n))
    for step in range(n):
        for d in (target - step, target + step):
            if 1 <= d <= n and n % d == 0:
                return d
    return 1


def _factor_d(n: int, d: int) -> List[int]:
    if d <= 1:
        return [n]
    s = max(1, int(round(n ** (1.0 / d))))
    for step in range(n):
        for candidate in (s - step, s + step):
            if 1 <= candidate <= n and n % candidate == 0:
                return [candidate] + _factor_d(n // candidate, d - 1)
    return [1] + _factor_d(n, d - 1)


def _matrix_fold_dims(m: int, n: int, d: int) -> Tuple[List[int], List[int]]:
    if d == 2:
        return [m], [n]
    if d == 3:
        s = max(1, int(round(math.sqrt(m))))
        m1 = _closest_divisor(m, s)
        m2 = m // m1
        return [m1, m2], [n]
    if d >= 4:
        s = max(1, int(round(math.sqrt(m))))
        m1 = _closest_divisor(m, s)
        m2 = m // m1 if m1 < m else 1
        t = max(1, int(round(math.sqrt(n))))
        n1 = _closest_divisor(n, t)
        n2 = n // n1 if n1 < n else 1
        return [m1, m2], [n1, n2]
    return [m], [n]


def _auto_fold_dims(m: int, n: int, max_dims: int = 5) -> Tuple[List[int], List[int]]:
    """Auto-compute fold dimensions that maximize the first TT mode size.

    Unlike _matrix_fold_dims which always splits rows first, this function
    chooses whether to split rows or columns based on aspect ratio, ensuring
    the first dimension is as large as possible for better SVD resolution.
    """
    total = m * n
    if max_dims <= 2:
        return [m], [n]

    if m >= n:
        big, small = m, n
        split_big = True
    else:
        big, small = n, m
        split_big = False

    if max_dims >= 4 and small >= 16:
        b1 = _closest_divisor(big, max(1, int(round(math.sqrt(big)))))
        b2 = big // b1 if b1 < big else 1
        s1 = _closest_divisor(small, max(1, int(round(math.sqrt(small)))))
        s2 = small // s1 if s1 < small else 1
        row_dims, col_dims = ([b1, b2], [s1, s2]) if split_big else ([s1, s2], [b1, b2])
    elif max_dims == 3:
        b1 = _closest_divisor(big, max(1, int(round(math.sqrt(big)))))
        b2 = big // b1 if b1 < big else 1
        row_dims, col_dims = ([b1, b2], [small]) if split_big else ([small], [b1, b2])
    else:
        row_dims, col_dims = [m], [n]

    return row_dims, col_dims


def _einsort_stage1(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mtx = np.asarray(matrix, dtype=np.float64)
    col_norms = np.linalg.norm(mtx, axis=0)
    col_order = np.argsort(-col_norms)
    mtx = mtx[:, col_order]
    row_norms = np.linalg.norm(mtx, axis=1)
    row_order = np.argsort(-row_norms)
    mtx = mtx[row_order, :]
    return mtx, row_order.astype(np.int32), col_order.astype(np.int32)


def _inverse_permute(
    matrix: np.ndarray, row_perm: np.ndarray, col_perm: np.ndarray
) -> np.ndarray:
    inv_row = np.argsort(row_perm)
    inv_col = np.argsort(col_perm)
    return matrix[inv_row, :][:, inv_col]


def _tt_svd_decompose(
    matrix: np.ndarray,
    target_ratio: float,
    tt_rank: Optional[int] = None,
    d: int = 4,
    storage_dtype: np.dtype = np.dtype("float16"),
    energy_threshold: float = 0.95,
) -> Tuple[List[np.ndarray], np.ndarray]:
    m, n = matrix.shape
    # Use auto fold that maximizes first dimension
    row_dims, col_dims = _auto_fold_dims(m, n, max_dims=d)
    all_dims = row_dims + col_dims
    effective_d = len(all_dims)
    reshaped = matrix.reshape(*row_dims, *col_dims)
    total_el = m * n

    r = tt_rank
    if r is None:
        target_storage = total_el * 4 / max(target_ratio, 1.0)
        avg_side = max(1, int(round(math.sqrt(sum(all_dims) / len(all_dims)))))
        approx_rank = max(
            4, min(128, int(math.sqrt(target_storage / (effective_d * avg_side))))
        )
        r = max(4, min(approx_rank, sum(all_dims) // len(all_dims)))
    r = max(4, r)

    cores = []
    current = np.asarray(reshaped, dtype=np.float64)
    prev_r = 1
    for k in range(effective_d - 1):
        ik = all_dims[k]
        unfolded = current.reshape(prev_r * ik, -1)
        target_rk = min(r, *unfolded.shape)
        if target_rk < 1:
            target_rk = 1

        U, S, Vt = np.linalg.svd(unfolded, full_matrices=False)

        # Energy-based rank selection: keep enough SVs to capture energy_threshold
        if len(S) > 1:
            s_cumsum = np.cumsum(S**2)
            s_total = float(s_cumsum[-1])
            if s_total > 1e-30:
                n_energy = int(
                    np.searchsorted(s_cumsum / s_total, energy_threshold) + 1
                )
            else:
                n_energy = 1
            rk = min(target_rk, max(n_energy, 1), len(S) - 1)
            rk = max(2, rk)
        else:
            rk = 1

        core = U[:, :rk].reshape(prev_r, ik, rk)
        cores.append(convert_to_storage(core, storage_dtype))
        current = (S[:rk, None] * Vt[:rk, :]).reshape(rk, -1)
        prev_r = rk

    last_core = current.reshape(prev_r, all_dims[-1], 1)
    cores.append(convert_to_storage(last_core, storage_dtype))
    tt_recon = _tt_reconstruct(cores, all_dims, (m, n))
    residual = matrix - tt_recon
    return cores, residual


def _svd_truncated(
    matrix: np.ndarray,
    target_ratio: float,
    storage_dtype: np.dtype = np.dtype("float16"),
    energy_threshold: float = 0.99,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Truncated SVD as an alternative to TT-SVD for smaller matrices.

    Uses energy-based rank selection then stores U[:,:r] @ diag(S[:r]) format
    to avoid storing Vt separately (Vt = SVD output, reconstruction uses U*diag(S)*Vt).
    Returns (U, S, Vt, reconstruction, residual).
    """
    m, n = matrix.shape
    total_el = m * n

    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)

    # Energy-based rank
    s_cumsum = np.cumsum(S**2)
    s_total = float(s_cumsum[-1])
    if s_total > 1e-30:
        min_rank = int(np.searchsorted(s_cumsum / s_total, energy_threshold) + 1)
    else:
        min_rank = 1

    # Also consider target_ratio
    ratio_driven_rank = max(2, int(total_el * 4 / (m + n + 1) / max(target_ratio, 1)))

    r = min(max(min_rank, ratio_driven_rank), min(m, n))
    r = min(r, len(S) - 1) if len(S) > 1 else 1
    r = max(2, r)

    U_r = U[:, :r]
    S_r = S[:r]
    Vt_r = Vt[:r, :]

    recon = (U_r * S_r) @ Vt_r
    residual = matrix.astype(np.float64) - recon.astype(np.float64)

    return (
        convert_to_storage(U_r, storage_dtype),
        convert_to_storage(S_r, storage_dtype),
        convert_to_storage(Vt_r, storage_dtype),
        recon.astype(np.float64),
        residual,
    )


def _tt_reconstruct(
    cores: List[np.ndarray],
    all_dims: List[int],
    matrix_shape: Tuple[int, int],
    storage_dtype: np.dtype = np.float16,
) -> np.ndarray:
    def _to_f64(x):
        return convert_from_storage(x, storage_dtype).astype(np.float64)

    result = _to_f64(cores[0])
    for core in cores[1:]:
        result = np.tensordot(result, _to_f64(core), axes=([-1], [0]))
    m, n = matrix_shape
    row_dims = all_dims[:1]
    col_dims = all_dims[1:]
    return result.reshape(m, n)


def _sparse_residual_stage3(
    residual: np.ndarray,
    topk_ratio: float = 0.01,
    use_2_4: bool = False,
    storage_dtype: np.dtype = np.float16,
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    flat = residual.ravel().astype(np.float64)
    n = len(flat)
    if use_2_4:
        padded = np.zeros(int(math.ceil(n / 4) * 4), dtype=np.float64)
        padded[:n] = flat
        blocks = padded.reshape(-1, 4)
        abs_vals = np.abs(blocks)
        top2_idx = np.argpartition(abs_vals, 2, axis=1)[:, :2]
        mask = np.zeros_like(blocks, dtype=bool)
        rows_2 = np.repeat(np.arange(blocks.shape[0]), 2)
        cols_2 = top2_idx.ravel()
        mask[rows_2, cols_2] = True
        sparse_vals = blocks[mask]
        indices = np.where(mask.ravel())[0].astype(np.int16)
    else:
        n_keep = max(1, int(n * topk_ratio))
        abs_flat = np.abs(flat)
        top_idx = np.argpartition(-abs_flat, n_keep - 1)[:n_keep]
        indices = np.sort(top_idx).astype(np.int16)
        sparse_vals = flat[indices]
    scale = float(np.max(np.abs(sparse_vals))) if len(sparse_vals) > 0 else 1.0
    if scale > 1e-10:
        sparse_vals = sparse_vals / scale
    sparse_recon = np.zeros(n, dtype=np.float64)
    sparse_recon[indices.astype(np.int64)] = sparse_vals * scale
    residual2 = flat - sparse_recon
    return (
        indices.astype(np.int16),
        convert_to_storage(sparse_vals, storage_dtype),
        np.float32(scale),
        residual2.reshape(residual.shape),
    )


def _sparse_reconstruct(
    indices: np.ndarray,
    values: np.ndarray,
    scale: float,
    n: int,
    storage_dtype: np.dtype = np.float16,
) -> np.ndarray:
    recon = np.zeros(n, dtype=np.float64)
    idx = indices.astype(np.int64)
    valid = idx < n
    recon[idx[valid]] = convert_from_storage(values, storage_dtype)[valid].astype(
        np.float64
    ) * float(scale)
    return recon


def _ergodic_trajectory_stage4(
    values: np.ndarray,
    n_channels: int = 16,
    storage_dtype: np.dtype = np.float16,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    flat = values.ravel().astype(np.float64)
    n = len(flat)
    n_chan = min(n_channels, max(1, n // 4))
    if n_chan < 1:
        n_chan = 1
    block_size = int(math.ceil(n / n_chan))
    padded = np.zeros(n_chan * block_size, dtype=np.float64)
    padded[:n] = flat
    blocks = padded.reshape(n_chan, block_size)
    n_avail = min(n_chan, len(_IRRATIONAL_PRIMES))
    alphas = np.sqrt(_IRRATIONAL_PRIMES[:n_avail])
    t = np.arange(block_size, dtype=np.float64)
    A_out = np.zeros(n_avail, dtype=np.float64)
    phi_out = np.zeros(n_avail, dtype=np.float64)
    bias_out = np.zeros(n_avail, dtype=np.float64)
    for c in range(n_avail):
        y = blocks[c]
        sin_at = np.sin(alphas[c] * t)
        cos_at = np.cos(alphas[c] * t)
        X = np.column_stack([sin_at, cos_at, np.ones(block_size, dtype=np.float64)])
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        b1, b2, bc = float(beta[0]), float(beta[1]), float(beta[2])
        A_out[c] = math.sqrt(b1 * b1 + b2 * b2)
        phi_out[c] = math.atan2(b2, b1)
        bias_out[c] = bc
    ergodic_recon = np.zeros(n, dtype=np.float64)
    for c in range(n_avail):
        start = c * block_size
        end = min(start + block_size, n)
        seg_len = end - start
        if seg_len <= 0:
            continue
        t_seg = np.arange(seg_len, dtype=np.float64)
        seg_recon = A_out[c] * np.sin(alphas[c] * t_seg + phi_out[c]) + bias_out[c]
        ergodic_recon[start:end] = seg_recon
    residual2 = flat - ergodic_recon
    return (
        convert_to_storage(alphas, storage_dtype),
        convert_to_storage(A_out, storage_dtype),
        convert_to_storage(phi_out, storage_dtype),
        convert_to_storage(bias_out, storage_dtype),
        residual2.reshape(values.shape),
    )


def _ergodic_reconstruct(
    alphas: np.ndarray,
    A: np.ndarray,
    phi: np.ndarray,
    bias: np.ndarray,
    n: int,
    storage_dtype: np.dtype = np.float16,
) -> np.ndarray:
    alphas_f64 = convert_from_storage(alphas, storage_dtype).astype(np.float64)
    A_f64 = convert_from_storage(A, storage_dtype).astype(np.float64)
    phi_f64 = convert_from_storage(phi, storage_dtype).astype(np.float64)
    bias_f64 = convert_from_storage(bias, storage_dtype).astype(np.float64)
    n_chan = min(len(alphas_f64), len(A_f64), len(phi_f64), len(bias_f64))
    if n_chan == 0:
        return np.zeros(n, dtype=np.float64)
    block_size = int(math.ceil(n / n_chan))
    recon = np.zeros(n, dtype=np.float64)
    for c in range(n_chan):
        start = c * block_size
        end = min(start + block_size, n)
        seg_len = end - start
        if seg_len <= 0:
            continue
        t_seg = np.arange(seg_len, dtype=np.float64)
        recon[start:end] = float(A_f64[c]) * np.sin(
            float(alphas_f64[c]) * t_seg + float(phi_f64[c])
        ) + float(bias_f64[c])
    return recon


def _siren_fit_2d(
    residual: np.ndarray,
    original_shape: Tuple[int, int],
    hidden_dim: int = 32,
    n_epochs: int = 200,
    storage_dtype: np.dtype = np.float16,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    rows, cols = original_shape
    n = rows * cols
    row_coords = np.linspace(-1.0, 1.0, rows)
    col_coords = np.linspace(-1.0, 1.0, cols)
    r_grid, c_grid = np.meshgrid(row_coords, col_coords, indexing="ij")
    coords = np.column_stack([r_grid.ravel(), c_grid.ravel()]).astype(np.float64)
    target = residual.ravel().astype(np.float64)
    t_mean = float(np.mean(target))
    t_std = float(np.std(target)) + 1e-10
    target_norm = (target - t_mean) / t_std
    rng = np.random.RandomState(42)
    w1 = (rng.randn(2, hidden_dim) * (6.0 / 2.0**0.5)).astype(np.float64)
    b1 = (rng.uniform(-np.pi, np.pi, hidden_dim)).astype(np.float64)
    wo = np.zeros(hidden_dim, dtype=np.float64)
    bo = 0.0
    lr = 0.005
    for epoch in range(n_epochs):
        h = np.sin(coords @ w1 + b1)
        pred = h @ wo + bo
        loss = pred - target_norm
        grad_wo = (h.T @ loss) / n
        grad_bo = float(np.mean(loss))
        wo -= lr * grad_wo
        bo -= lr * grad_bo
        if epoch % 50 == 49:
            lr *= 0.5
    bo = bo * t_std + t_mean
    wo = wo * t_std
    return (
        convert_to_storage(w1, storage_dtype),
        convert_to_storage(b1, storage_dtype),
        convert_to_storage(wo, storage_dtype),
        np.float32(bo),
    )


def _siren_reconstruct(
    w1: np.ndarray,
    b1: np.ndarray,
    wo: np.ndarray,
    bo: float,
    shape: Tuple[int, int],
    storage_dtype: np.dtype = np.float16,
) -> np.ndarray:
    rows, cols = shape
    row_coords = np.linspace(-1.0, 1.0, rows)
    col_coords = np.linspace(-1.0, 1.0, cols)
    r_grid, c_grid = np.meshgrid(row_coords, col_coords, indexing="ij")
    coords = np.column_stack([r_grid.ravel(), c_grid.ravel()]).astype(np.float64)
    h = np.sin(
        coords @ convert_from_storage(w1, storage_dtype).astype(np.float64)
        + convert_from_storage(b1, storage_dtype).astype(np.float64)
    )
    pred = h @ convert_from_storage(wo, storage_dtype).astype(np.float64) + float(bo)
    return pred.reshape((rows, cols)).astype(np.float64)


def _detect_flat_spectrum(matrix: np.ndarray, sample_size: int = 256) -> bool:
    """Detect if a matrix has a flat singular spectrum (LLM weights).

    LLM weights have near-Gaussian distributions with flat SVD spectra,
    meaning SVD/TT-SVD cannot compress them effectively. Detection uses
    a fast randomized SVD on a subsample.

    Returns True if >30% of rank is needed for 90% energy (flat spectrum).
    """
    m, n = matrix.shape
    k = min(m, n, sample_size)
    if k < 16:
        return False
    try:
        rng = np.random.RandomState(42)
        if m <= k or n <= k:
            U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
            S = S[: len(S) // 2]
        else:
            probe = rng.randn(n, k)
            Y = matrix @ probe
            Q, _ = np.linalg.qr(Y)
            B = Q.T @ matrix
            _, S, _ = np.linalg.svd(B, full_matrices=False)
        if len(S) < 4:
            return False
        cum_energy = np.cumsum(S**2)
        total = float(cum_energy[-1])
        if total < 1e-30:
            return False
        r_90 = int(np.searchsorted(cum_energy / total, 0.90) + 1)
        rank_frac = r_90 / max(len(S), 1)
        return rank_frac > 0.3
    except Exception:
        return False


def _block_quant(
    arr: np.ndarray, n_bits: int = 4, block_size: int = 64
) -> Tuple[np.ndarray, np.ndarray]:
    """Block-wise uniform quantization with bit packing.

    INT4: range [-7, 7], packed 2 per byte.
    INT2: range [-1, 1], packed 4 per byte.
    INT8: range [-127, 127], 1 per byte.

    Returns (codes_packed, scales) where codes_packed is a compact uint8
    array with bit-packed values, and scales (float16) has one per block.
    """
    flat = arr.ravel().astype(np.float32)
    n = len(flat)
    n_blocks = max(1, (n + block_size - 1) // block_size)
    scales = np.zeros(n_blocks, dtype=np.float16)

    if n_bits == 8:
        packed_len = n
    elif n_bits == 4:
        packed_len = (n + 1) // 2
    elif n_bits == 2:
        packed_len = (n + 3) // 4
    else:
        packed_len = n

    packed = np.zeros(packed_len, dtype=np.uint8)
    max_q = float((1 << (n_bits - 1)) - 1)
    offset = max_q + 1

    for i in range(n_blocks):
        start = i * block_size
        end = min(start + block_size, n)
        block = flat[start:end]
        scale = float(np.max(np.abs(block)))
        if scale < 1e-10:
            scale = 1.0
        scales[i] = np.float16(scale)
        q = np.clip(np.round(block / scale * max_q), -max_q, max_q)
        q_off = (q.astype(np.int8) + offset).astype(np.uint8)

        if n_bits == 8:
            packed[start:end] = q_off
        elif n_bits == 4:
            for j, v in enumerate(q_off):
                src_idx = start + j
                if src_idx >= n:
                    break
                pack_idx = src_idx // 2
                shift = (src_idx % 2) * 4
                packed[pack_idx] = packed[pack_idx] | np.uint8(v << shift)
        elif n_bits == 2:
            for j, v in enumerate(q_off):
                src_idx = start + j
                if src_idx >= n:
                    break
                pack_idx = src_idx // 4
                shift = (src_idx % 4) * 2
                packed[pack_idx] = packed[pack_idx] | np.uint8(v << shift)

    return packed, scales


def _block_dequant(
    codes: np.ndarray,
    scales: np.ndarray,
    shape: tuple,
    n_bits: int = 4,
    block_size: int = 64,
) -> np.ndarray:
    """Dequantize bit-packed block-quantized data."""
    n = int(np.prod(shape))
    flat = np.zeros(n, dtype=np.float32)
    n_blocks = max(1, (n + block_size - 1) // block_size)
    max_q = float((1 << (n_bits - 1)) - 1)
    offset = max_q + 1

    for i in range(n_blocks):
        start = i * block_size
        end = min(start + block_size, n)
        s = float(scales[i])

        for j in range(start, end):
            if n_bits == 8:
                raw_val = int(codes[j])
            elif n_bits == 4:
                idx = j // 2
                shift = 4 * (j % 2)
                raw_val = (int(codes[idx]) >> shift) & 0xF
            elif n_bits == 2:
                idx = j // 4
                shift = 2 * (j % 4)
                raw_val = (int(codes[idx]) >> shift) & 0x3
            else:
                raw_val = 0
            raw_signed = raw_val - offset
            flat[j] = float(raw_signed) * s / max_q

    return flat.reshape(shape)


class FiveStageCascade:
    def __init__(
        self,
        tt_rank: Optional[int] = None,
        sparse_topk_ratio: float = 0.01,
        ergodic_n_channels: int = 16,
        siren_hidden_dim: int = 32,
        siren_n_epochs: int = 200,
        d: int = 4,
        use_2_4_sparsity: bool = False,
        energy_threshold: float = 0.99,
    ):
        self.tt_rank = tt_rank
        self.sparse_topk_ratio = sparse_topk_ratio
        self.ergodic_n_channels = ergodic_n_channels
        self.siren_hidden_dim = siren_hidden_dim
        self.siren_n_epochs = siren_n_epochs
        self.d = d
        self.use_2_4_sparsity = use_2_4_sparsity
        self.energy_threshold = energy_threshold
        self._cascade_storage_dtype: np.dtype = np.float16

    def compress(
        self, tensor: np.ndarray, target_ratio: float = 200.0
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        self._cascade_storage_dtype = detect_storage_dtype(tensor)
        t = np.asarray(tensor, dtype=np.float64)
        if t.ndim == 1:
            return self._compress_1d(t, target_ratio)
        if t.ndim != 2:
            orig_shape = t.shape
            t = t.reshape(t.shape[0], -1)
            payload, meta = self._compress_2d(t, target_ratio)
            meta["original_shape"] = orig_shape
            return payload, meta
        return self._compress_2d(t, target_ratio)

    def decompress(
        self, payload: Dict[str, Any], metadata: Dict[str, Any]
    ) -> np.ndarray:
        self._cascade_storage_dtype = decode_dtype_code(
            metadata.get("_storage_dtype", 0)
        )
        shape = metadata.get("original_shape")
        if shape is None:
            raise ValueError("metadata must contain original_shape")
        if len(shape) == 1:
            return self._decompress_1d(payload, metadata)
        return self._decompress_2d(payload, metadata)

    def _compress_1d(
        self, tensor: np.ndarray, target_ratio: float
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        n = len(tensor)
        side = max(1, int(math.isqrt(n)))
        while n % side != 0 and side > 1:
            side -= 1
        if side < 2:
            side = int(math.ceil(math.sqrt(n)))
            padded = np.zeros(side * side, dtype=tensor.dtype)
            padded[:n] = tensor
            matrix = padded.reshape(side, side)
        else:
            matrix = tensor.copy().reshape(side, n // side)
        payload, metadata = self._compress_2d(matrix, target_ratio)
        metadata["original_shape_1d"] = tensor.shape
        metadata["_1d_n"] = n
        metadata["_1d_side"] = side
        return payload, metadata

    def _decompress_1d(
        self, payload: Dict[str, Any], metadata: Dict[str, Any]
    ) -> np.ndarray:
        n = metadata.get("_1d_n", metadata.get("original_shape_1d", (1,))[0])
        side = metadata.get("_1d_side", 1)
        meta_2d = dict(metadata)
        meta_2d["original_shape"] = (side, max(1, int(math.ceil(n / side))))
        matrix_recon = self._decompress_2d(payload, meta_2d)
        flat = matrix_recon.ravel()[:n]
        return flat.reshape(metadata["original_shape_1d"]).astype(np.float32)

    def _compress_2d(
        self, matrix: np.ndarray, target_ratio: float
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        sd = self._cascade_storage_dtype
        m, n = matrix.shape
        t0_total = time.time()

        # ── Path Selection ──────────────────────────────────────────────
        # LLM weights have flat SVD spectra — SVD/TT-SVD cannot compress
        # them effectively. The pipeline auto-detects this and falls back
        # to block quantization on the permuted matrix.
        #
        # Detection: if SVD energy decays slowly (need >30% rank for 90% energy),
        # the tensor is "flat-spectrum" and SVD is pointless.
        flat_spectrum = _detect_flat_spectrum(matrix)

        if flat_spectrum or target_ratio > 100:
            # ── Quantization Path (for LLM weights) ──────────────────
            # Stage 1: EinSort permutation (reduces quantization entropy)
            permuted, row_perm, col_perm = _einsort_stage1(matrix)

            # Stage 2: Aggressive block quantization
            # Target: 200:1 means ~0.16 bits/weight vs FP32, ~0.08 bits/weight vs BF16
            # Block INT2 gives 16:1 per stage. Need multiple residual stages.
            n_stages = max(1, int(np.ceil(np.log(max(target_ratio, 2)) / np.log(16))))
            n_stages = min(n_stages, 6)  # cap at 6 stages (diminishing returns)

            residual = permuted.astype(np.float64)
            recon_sum = np.zeros_like(residual)
            quant_stages = []

            for si in range(n_stages):
                source = np.ascontiguousarray(residual, dtype=np.float32)
                if si == 0:
                    codes, scales = _block_quant(source, 2, block_size=64)
                    recon_block = _block_dequant(codes, scales, source.shape, 2, 64)
                elif si == n_stages - 1 and n_stages > 2:
                    codes, scales = _block_quant(source, 4, block_size=64)
                    recon_block = _block_dequant(codes, scales, source.shape, 4, 64)
                else:
                    codes, scales = _block_quant(source, 2, block_size=64)
                    recon_block = _block_dequant(codes, scales, source.shape, 2, 64)
                recon_sum += recon_block.astype(np.float64)
                residual = permuted.astype(np.float64) - recon_sum
                quant_stages.append((codes, scales))

            payload: Dict[str, Any] = {
                "s1_row_perm": row_perm,
                "s1_col_perm": col_perm,
                "s2_type": "quant",
                "s2_n_stages": n_stages,
                "s2_quant_stages": quant_stages,
            }
            used_stages = [1, 2]
            primary_rel_error = float(np.var(residual)) / max(
                float(np.var(permuted)), 1e-30
            )

        else:
            # ── SVD/TT Path (for structured tensors) ──────────────────
            permuted, row_perm, col_perm = _einsort_stage1(matrix)
            use_full_svd = (self.d <= 2) or (max(m, n) <= 4096 and m * n <= 16_000_000)

            if use_full_svd:
                U_r, S_r, Vt_r, svd_recon, primary_residual = _svd_truncated(
                    permuted,
                    target_ratio,
                    storage_dtype=sd,
                    energy_threshold=self.energy_threshold,
                )
                cores_or_svd = [U_r, S_r, Vt_r]
                decomp_key = "svd"
            else:
                cores, primary_residual = _tt_svd_decompose(
                    permuted,
                    target_ratio,
                    tt_rank=self.tt_rank,
                    d=self.d,
                    storage_dtype=sd,
                )
                cores_or_svd = cores
                decomp_key = "tt"

            residual_var = float(np.var(primary_residual))
            orig_var = float(np.var(permuted))
            primary_rel_error = residual_var / max(orig_var, 1e-30)

            payload: Dict[str, Any] = {
                "s1_row_perm": row_perm,
                "s1_col_perm": col_perm,
                "s2_type": decomp_key,
                "s2_cores": cores_or_svd,
            }
            used_stages = [1, 2]

        metadata: Dict[str, Any] = {
            "original_shape": matrix.shape,
            "dims": list(matrix.shape),
            "decomposition": payload["s2_type"],
            "used_stages": used_stages,
            "primary_rel_error": float(primary_rel_error),
            "_storage_dtype": int(encode_dtype_code(sd)),
        }
        return payload, metadata

    def _decompress_2d(
        self, payload: Dict[str, Any], metadata: Dict[str, Any]
    ) -> np.ndarray:
        sd = self._cascade_storage_dtype
        shape = metadata["original_shape"]
        m, n = shape
        decomp_type = payload.get("s2_type", "tt")

        if decomp_type == "quant":
            # Quantization path
            n_stages = payload.get("s2_n_stages", 1)
            quant_stages = payload.get("s2_quant_stages", [])
            recon_sum = np.zeros((m, n), dtype=np.float64)
            for si, (codes, scales) in enumerate(quant_stages):
                n_bits = 4 if (si == n_stages - 1 and n_stages > 2) else 2
                recon_block = _block_dequant(
                    codes, scales, (m, n), n_bits, block_size=64
                )
                recon_sum += recon_block.astype(np.float64)
            permuted_recon = recon_sum
        elif decomp_type == "svd":
            U_r = convert_from_storage(payload["s2_cores"][0], sd).astype(np.float64)
            S_r = convert_from_storage(payload["s2_cores"][1], sd).astype(np.float64)
            Vt_r = convert_from_storage(payload["s2_cores"][2], sd).astype(np.float64)
            permuted_recon = (U_r * S_r) @ Vt_r
        else:
            row_dims, col_dims = _auto_fold_dims(m, n, max_dims=self.d)
            all_dims = row_dims + col_dims
            permuted_recon = _tt_reconstruct(
                payload["s2_cores"],
                all_dims,
                (m, n),
                storage_dtype=sd,
            )

        result = _inverse_permute(
            permuted_recon,
            payload["s1_row_perm"],
            payload["s1_col_perm"],
        ).reshape(shape)
        return result.astype(np.float32)


def compress_cascade(
    tensor: np.ndarray,
    target_ratio: float = 200.0,
    **kwargs,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    cascade = FiveStageCascade(**kwargs)
    return cascade.compress(tensor, target_ratio)


def decompress_cascade(
    payload: Dict[str, Any],
    metadata: Dict[str, Any],
) -> np.ndarray:
    cascade = FiveStageCascade()
    return cascade.decompress(payload, metadata)


class Cascade5StageMethod:
    """Standard compression-method interface for engine compatibility.

    Wraps FiveStageCascade so the engine's method discovery and registry
    can treat it like any other built-in method.
    """

    name = "cascade_5stage"
    category = "unified"

    def __init__(self, **kwargs):
        self._cascade = FiveStageCascade(**kwargs)
        self._kwargs = kwargs

    def compress(
        self,
        tensor: np.ndarray,
        target_ratio: float = 200.0,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        payload, metadata = self._cascade.compress(tensor, target_ratio)
        return payload, metadata

    def decompress(
        self, payload: Dict[str, Any], metadata: Dict[str, Any]
    ) -> np.ndarray:
        return self._cascade.decompress(payload, metadata)

    def __repr__(self) -> str:
        return f"Cascade5StageMethod({self._kwargs})"
