"""
Structured matrix decompositions for specialized matrix types.
All methods return: (compressed: dict, ratio: float, snr_db: float)
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np
from scipy.linalg import svd


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
    if t.ndim == 0:
        t = t.reshape(1, 1)
    elif t.ndim == 1:
        t = t.reshape(1, -1)
    elif t.ndim > 2:
        t = t.reshape(t.shape[0], -1)
    return t


def kronecker_decompose(
    tensor: np.ndarray, shape_a: Tuple[int, int] = None
) -> Tuple[Dict, float, float]:
    """Kronecker product decomposition: W ~ A (X) B.
    Finds optimal A, B minimizing ||W - A(X)B||_F.
    Uses Schmidt decomposition via SVD of the permuted matrix.
    """
    t = _to_2d(tensor)
    m, n = t.shape

    if shape_a is None:
        a = int(math.isqrt(m))
        while m % a != 0:
            a -= 1
        shape_a = (a, m // a)

    a, b = shape_a
    if m % a != 0 or n % b != 0:
        raise ValueError(f"shape_a {shape_a} does not divide tensor shape {(m, n)}")
    c, d = m // a, n // b

    W_r = t.reshape(a, c, b, d).transpose(0, 2, 1, 3).reshape(a * b, c * d)
    U, s, Vt = svd(W_r, full_matrices=False)
    A = math.sqrt(max(s[0], 0.0)) * U[:, 0].reshape(a, b)
    B = math.sqrt(max(s[0], 0.0)) * Vt[0, :].reshape(c, d)
    recon = np.kron(A, B)

    c_dict: Dict = {
        "A": A.astype(np.float32),
        "B": B.astype(np.float32),
        "shape": tensor.shape,
    }
    return c_dict, tensor.nbytes / max(_bytes(c_dict), 1), _snr(tensor, recon)


def block_diagonal_decompose(
    tensor: np.ndarray, n_blocks: int = 4
) -> Tuple[Dict, float, float]:
    """Block diagonal approximation with block-wise SVD.
    Splits matrix into diagonal blocks and SVD-truncates each.
    """
    t = _to_2d(tensor)
    m, n = t.shape
    bm, bn = m // n_blocks, n // n_blocks

    Us, Ss, Vts = [], [], []
    for k in range(n_blocks):
        i0, j0 = k * bm, k * bn
        block = t[i0 : i0 + bm, j0 : j0 + bn]
        U, s, Vt = svd(block, full_matrices=False)
        r = max(1, min(len(s), min(bm, bn) // 2))
        Us.append(U[:, :r].astype(np.float32))
        Ss.append(s[:r].astype(np.float32))
        Vts.append(Vt[:r, :].astype(np.float32))

    recon = np.zeros_like(t)
    for k in range(n_blocks):
        i0, j0 = k * bm, k * bn
        U, s, Vt = (
            Us[k].astype(np.float64),
            Ss[k].astype(np.float64),
            Vts[k].astype(np.float64),
        )
        recon[i0 : i0 + bm, j0 : j0 + bn] = (U * s) @ Vt

    c_dict: Dict = {
        "U_blocks": Us,
        "s_blocks": Ss,
        "Vt_blocks": Vts,
        "n_blocks": n_blocks,
        "block_shape": (bm, bn),
        "shape": tensor.shape,
    }
    return c_dict, tensor.nbytes / max(_bytes(c_dict), 1), _snr(tensor, recon)


def toeplitz_decompose(tensor: np.ndarray) -> Tuple[Dict, float, float]:
    """Toeplitz matrix approximation.
    W[i,j] ~ w[i-j] (constant along diagonals).
    Optimal w(k) = mean of entries on k-th diagonal.
    Uses m-based shift to handle rectangular matrices (m < n).
    """
    t = _to_2d(tensor)
    m, n = t.shape
    w = np.zeros(m + n - 1)
    for k in range(-(m - 1), n):
        diag = np.diag(t, k=k)
        w[k + m - 1] = np.mean(diag) if len(diag) > 0 else 0.0

    i = np.arange(m)[:, None]
    j = np.arange(n)[None, :]
    recon = w[j - i + m - 1]

    c_dict: Dict = {
        "w": w.astype(np.float32),
        "shape": tensor.shape,
    }
    return c_dict, tensor.nbytes / max(_bytes(c_dict), 1), _snr(tensor, recon)


def hankel_decompose(tensor: np.ndarray) -> Tuple[Dict, float, float]:
    """Hankel matrix approximation.
    W[i,j] ~ w[i+j] (constant along anti-diagonals).
    """
    t = _to_2d(tensor)
    m, n = t.shape
    w = np.zeros(m + n - 1)
    for k in range(m + n - 1):
        i0 = max(0, k - n + 1)
        i1 = min(m, k + 1)
        if i0 < i1:
            w[k] = np.mean(t[np.arange(i0, i1), k - np.arange(i0, i1)])
        else:
            w[k] = 0.0

    i = np.arange(m)[:, None]
    j = np.arange(n)[None, :]
    recon = w[i + j]

    c_dict: Dict = {
        "w": w.astype(np.float32),
        "shape": tensor.shape,
    }
    return c_dict, tensor.nbytes / max(_bytes(c_dict), 1), _snr(tensor, recon)


def butterfly_factorize(
    tensor: np.ndarray, n_levels: int = None
) -> Tuple[Dict, float, float]:
    """Butterfly factorization: W = B_0 @ B_1 @ ... @ B_{L-1}
    Each B_i is block-diagonal of 2x2 matrices. O(N log N) params.
    Uses greedy level-by-level SVD-based fitting.
    """
    t = _to_2d(tensor)
    m, n = t.shape
    N = 1 << (max(m, n) - 1).bit_length()
    L = n_levels or int(math.log2(N))

    W = np.zeros((N, N), dtype=np.float64)
    W[:m, :n] = t
    cur = W.copy()
    blocks = []

    for level in range(L):
        s = N // (1 << (level + 1))
        nb = N // (2 * s)
        i_base = np.arange(nb) * 2 * s
        k = np.arange(s)
        is0 = (i_base[:, None] + k[None, :]).ravel()
        is1 = is0 + s
        level_blocks = np.zeros((nb * s, 2, 2), dtype=np.float64)
        level_blocks[:, 0, 0] = cur[is0, is0]
        level_blocks[:, 0, 1] = cur[is0, is1]
        level_blocks[:, 1, 0] = cur[is1, is0]
        level_blocks[:, 1, 1] = cur[is1, is1]
        blocks.append(level_blocks.astype(np.float32))

        if level < L - 1:
            a = level_blocks[:, 0, 0]
            b = level_blocks[:, 0, 1]
            c = level_blocks[:, 1, 0]
            d = level_blocks[:, 1, 1]
            det = a * d - b * c
            det = np.where(np.abs(det) < 1e-30, 1e-30, det)
            inv_det = 1.0 / det
            B_inv_00 = d * inv_det
            B_inv_01 = -b * inv_det
            B_inv_10 = -c * inv_det
            B_inv_11 = a * inv_det
            cur_next = cur.copy()
            rows_i = cur[is0]
            rows_ip1 = cur[is1]
            cur_next[is0] = B_inv_00[:, None] * rows_i + B_inv_01[:, None] * rows_ip1
            cur_next[is1] = B_inv_10[:, None] * rows_i + B_inv_11[:, None] * rows_ip1
            cur = cur_next

    recon = np.eye(N, dtype=np.float64)
    for level in reversed(range(L)):
        s = N // (1 << (level + 1))
        nb = N // (2 * s)
        i_base = np.arange(nb) * 2 * s
        kk = np.arange(s)
        is0 = (i_base[:, None] + kk[None, :]).ravel()
        is1 = is0 + s
        B = np.zeros((N, N), dtype=np.float64)
        blk = blocks[level]
        B[is0, is0] = blk[:, 0, 0]
        B[is0, is1] = blk[:, 0, 1]
        B[is1, is0] = blk[:, 1, 0]
        B[is1, is1] = blk[:, 1, 1]
        recon = B @ recon
    recon = recon[:m, :n]

    c_dict: Dict = {
        "blocks": blocks,
        "N": N,
        "L": L,
        "shape": tensor.shape,
    }
    return c_dict, tensor.nbytes / max(_bytes(c_dict), 1), _snr(tensor, recon)


def circulant_decompose(tensor: np.ndarray) -> Tuple[Dict, float, float]:
    """Circulant matrix approximation. Diagonalized by FFT.
    W = F^-1 @ diag(F @ w) @ F where w is first column.
    Optimal w(k) = mean of entries on k-th cyclic diagonal.
    """
    t = _to_2d(tensor)
    n = t.shape[0]
    if t.shape[0] != t.shape[1]:
        n = max(t.shape)
        t_pad = np.zeros((n, n), dtype=np.float64)
        t_pad[: t.shape[0], : t.shape[1]] = t
        t = t_pad

    i = np.arange(n)[:, None]
    k = np.arange(n)[None, :]
    c = np.mean(t[i, (i - k) % n], axis=0)

    row = np.arange(n)[:, None]
    col = np.arange(n)[None, :]
    recon = c[(row - col) % n]
    recon = recon[: tensor.shape[0], : tensor.shape[1]]

    c_dict: Dict = {
        "c": c.astype(np.float32),
        "shape": tensor.shape,
    }
    return c_dict, tensor.nbytes / max(_bytes(c_dict), 1), _snr(tensor, recon)


if __name__ == "__main__":
    t = np.random.randn(128, 128).astype(np.float32)
    data, ratio, snr = kronecker_decompose(t, shape_a=(16, 8))
    print(f"Kronecker: {ratio:.2f}x, SNR={snr:.1f}dB")
    data, ratio, snr = block_diagonal_decompose(t, n_blocks=4)
    print(f"BlockDiag: {ratio:.2f}x, SNR={snr:.1f}dB")
    data, ratio, snr = toeplitz_decompose(t)
    print(f"Toeplitz: {ratio:.2f}x, SNR={snr:.1f}dB")
    data, ratio, snr = hankel_decompose(t)
    print(f"Hankel: {ratio:.2f}x, SNR={snr:.1f}dB")
    data, ratio, snr = butterfly_factorize(t)
    print(f"Butterfly: {ratio:.2f}x, SNR={snr:.1f}dB")
    data, ratio, snr = circulant_decompose(t)
    print(f"Circulant: {ratio:.2f}x, SNR={snr:.1f}dB")
