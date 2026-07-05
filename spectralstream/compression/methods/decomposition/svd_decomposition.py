"""
SVD and related matrix decomposition methods.
All methods return: (compressed: dict, ratio: float, snr_db: float)
Uses randomized SVD for matrices > 1024×1024 to prevent OOM.
"""

from __future__ import annotations

import gc
from typing import Dict, Tuple

import numpy as np
from scipy.linalg import svd as _scipy_svd

_SVD_SIZE_LIMIT = 1024 * 1024  # use randomized SVD above this element count


def _bytes(obj) -> int:
    if isinstance(obj, np.ndarray):
        return obj.nbytes
    if isinstance(obj, dict):
        return sum(_bytes(v) for _, v in obj.items())
    if isinstance(obj, (list, tuple)):
        return sum(_bytes(x) for x in obj)
    return 8


def _snr(orig: np.ndarray, recon: np.ndarray) -> float:
    o = orig.astype(np.float64).ravel()
    r = recon.astype(np.float64).ravel()
    mse = np.mean((o - r) ** 2)
    return float(10.0 * np.log10(np.mean(o**2) / (mse + 1e-30)))


def _to_2d(tensor: np.ndarray) -> np.ndarray:
    t = np.asarray(tensor, dtype=np.float64)
    if t.ndim > 2:
        t = t.reshape(t.shape[0], -1)
    return t


def _safe_svd(t: np.ndarray, full_matrices: bool = False) -> Tuple:
    """Dispatch to full SVD or randomized SVD based on matrix size."""
    m, n = t.shape
    if t.size > _SVD_SIZE_LIMIT and min(m, n) > 512:
        return _randomized_svd_internal(t, min(m, n) // 4)
    return _scipy_svd(t, full_matrices=full_matrices)


def _randomized_svd_internal(
    X: np.ndarray, n_components: int, n_oversamples: int = 10, n_iter: int = 3
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Memory-efficient randomized SVD — avoids forming large SVD."""
    m, n = X.shape
    k = min(n_components + n_oversamples, min(m, n))
    rng = np.random.RandomState(42)
    O = rng.randn(n, k).astype(X.dtype)
    Y = X @ O
    for _ in range(n_iter):
        Y = X @ (X.T @ Y)
    Q, _ = np.linalg.qr(Y)
    B = Q.T @ X
    Ub, s, Vt = _scipy_svd(B, full_matrices=False)
    U = Q @ Ub[:, :n_components]
    S = s[:n_components]
    Vh = Vt[:n_components, :]
    return U, S, Vh


def svd_truncated(
    tensor: np.ndarray, rank: int = None, energy_threshold: float = 0.99
) -> Tuple[Dict, float, float]:
    """Truncated SVD: W ~ U @ diag(s) @ Vh.
    Auto-rank selection based on energy threshold.
    Uses randomized SVD for matrices > 1024×1024.
    """
    t = _to_2d(tensor)
    m, n = t.shape
    k_target = rank or min(m, n)

    if t.size > _SVD_SIZE_LIMIT and min(m, n) > 512:
        max_rank = min(k_target, 256)
        U, s, Vt = _randomized_svd_internal(t, max_rank)
        if rank is None:
            cum = np.cumsum(s**2) / (np.sum(s**2) + 1e-30)
            k_target = int(np.searchsorted(cum, energy_threshold)) + 1
        rank = min(k_target, len(s))
        U, s, Vt = U[:, :rank], s[:rank], Vt[:rank, :]
    else:
        U, s, Vt = _scipy_svd(t, full_matrices=False)
        if rank is None:
            cum = np.cumsum(s**2) / (np.sum(s**2) + 1e-30)
            rank = int(np.searchsorted(cum, energy_threshold)) + 1
        rank = min(rank, len(s))
        U, s, Vt = U[:, :rank], s[:rank], Vt[:rank, :]

    c = {
        "U": U.astype(np.float32),
        "s": s.astype(np.float32),
        "Vt": Vt.astype(np.float32),
        "shape": tensor.shape,
    }
    recon = (U * s) @ Vt
    recon = recon.reshape(tensor.shape)

    del U, s, Vt
    gc.collect()

    return c, tensor.nbytes / max(_bytes(c), 1), _snr(tensor, recon)


def randomized_svd(
    tensor: np.ndarray, rank: int = None, n_oversamples: int = 5
) -> Tuple[Dict, float, float]:
    """Randomized SVD for faster decomposition of large matrices.
    Uses randomized range finder to approximate SVD.
    """
    t = _to_2d(tensor)
    m, n = t.shape
    k = min(rank or int(0.9 * min(m, n)), min(m, n))
    k = max(1, k)
    r = min(k + n_oversamples, min(m, n))

    rng = np.random.RandomState(42)
    Q, _ = np.linalg.qr(t @ rng.randn(n, r))
    B = Q.T @ t
    Ub, s, Vt = _scipy_svd(B, full_matrices=False)
    k = min(k, len(s))
    U = Q @ Ub[:, :k]
    s, Vt = s[:k], Vt[:k, :]

    c = {
        "U": U[:, :k].astype(np.float32),
        "s": s.astype(np.float32),
        "Vt": Vt.astype(np.float32),
        "shape": tensor.shape,
    }
    recon = (U[:, :k] * s) @ Vt
    recon = recon.reshape(tensor.shape)

    del Q, B, Ub, U, s, Vt
    gc.collect()

    return c, tensor.nbytes / max(_bytes(c), 1), _snr(tensor, recon)


def cur_decomposition(
    tensor: np.ndarray, rank: int = None
) -> Tuple[Dict, float, float]:
    """CUR matrix decomposition: W ~ C @ U @ R.
    Selects actual columns (C) and rows (R) of the original matrix.
    Uses randomized SVD for leverage score computation on large matrices.
    """
    t = _to_2d(tensor)
    m, n = t.shape
    k = min(rank or max(1, int(0.1 * min(m, n))), min(m, n))
    k = max(1, k)

    if t.size > _SVD_SIZE_LIMIT and min(m, n) > 512:
        U_full, _, Vt_full = _randomized_svd_internal(t, k)
    else:
        U_full, _, Vt_full = _scipy_svd(t, full_matrices=False)

    pr = np.sum(U_full[:, :k] ** 2, axis=1)
    pr = np.maximum(pr, 0.0)
    pr /= np.sum(pr)
    pc = np.sum(Vt_full[:k, :].T ** 2, axis=1)
    pc = np.maximum(pc, 0.0)
    pc /= np.sum(pc)

    rng = np.random.RandomState(42)
    ri = np.sort(rng.choice(m, size=k, p=pr, replace=False))
    ci = np.sort(rng.choice(n, size=k, p=pc, replace=False))

    C_mat = t[:, ci]
    R_mat = t[ri, :]
    U_cur = np.linalg.pinv(C_mat) @ t @ np.linalg.pinv(R_mat)
    recon = C_mat @ U_cur @ R_mat

    c = {
        "C": C_mat.astype(np.float32),
        "U": U_cur.astype(np.float32),
        "R": R_mat.astype(np.float32),
        "row_idx": ri.astype(np.int32),
        "col_idx": ci.astype(np.int32),
        "shape": tensor.shape,
    }

    del U_full, Vt_full, C_mat, R_mat, U_cur
    gc.collect()

    return c, tensor.nbytes / max(_bytes(c), 1), _snr(tensor, recon)


def nystrom_approximation(
    tensor: np.ndarray, rank: int = 32
) -> Tuple[Dict, float, float]:
    """Nystrom approximation for symmetric positive semi-definite matrices.
    Uses column sampling to approximate the full matrix.
    """
    t = _to_2d(tensor)
    n = t.shape[0]
    t_sym = (t + t.T) * 0.5
    k = min(rank, n)
    k = max(1, k)

    rng = np.random.RandomState(42)
    idx = np.sort(rng.choice(n, size=k, replace=False))

    C = t_sym[:, idx]
    W11 = C[idx, :]
    ev, evec = np.linalg.eigh(W11)
    ev = np.maximum(ev, 0.0)
    pos = ev > 1e-12 * ev.max() if ev.max() > 0 else ev > 1e-30

    if np.any(pos):
        W11_pinv = (evec[:, pos] / ev[pos]) @ evec[:, pos].T
    else:
        W11_pinv = np.zeros((k, k), dtype=np.float64)

    recon = C @ W11_pinv @ C.T

    c = {
        "C": C.astype(np.float32),
        "W11_pinv": W11_pinv.astype(np.float32),
        "col_idx": idx.astype(np.int32),
        "shape": tensor.shape,
    }

    del C, W11, ev, evec, W11_pinv
    gc.collect()

    return c, tensor.nbytes / max(_bytes(c), 1), _snr(tensor, recon)


if __name__ == "__main__":
    t = np.random.randn(128, 128).astype(np.float32)
    data, ratio, snr = svd_truncated(t)
    print(f"SVD: {ratio:.2f}x, SNR={snr:.1f}dB")
    data, ratio, snr = randomized_svd(t, rank=32)
    print(f"RSVD: {ratio:.2f}x, SNR={snr:.1f}dB")
    data, ratio, snr = cur_decomposition(t, rank=32)
    print(f"CUR: {ratio:.2f}x, SNR={snr:.1f}dB")
    data, ratio, snr = nystrom_approximation(t @ t.T, rank=32)
    print(f"Nystrom: {ratio:.2f}x, SNR={snr:.1f}dB")
