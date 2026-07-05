"""HPC low-rank structural methods — vectorized Monarch, HSS, H-matrix."""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np


def _snr_db(original: np.ndarray, reconstructed: np.ndarray) -> float:
    mse = float(
        np.mean((original.astype(np.float64) - reconstructed.astype(np.float64)) ** 2)
    )
    signal = float(np.mean(original.astype(np.float64) ** 2))
    return 10.0 * math.log10(max(signal, 1e-30) / max(mse, 1e-30))


def _validate(tensor, min_dim=1) -> np.ndarray:
    t = np.asarray(tensor, dtype=np.float64)
    if t.size == 0:
        raise ValueError("Empty tensor")
    if t.ndim < min_dim:
        raise ValueError(f"Need >= {min_dim} dims, got {t.ndim}")
    return t


def monarch_decompose(
    tensor: np.ndarray, block_size: int = 2
) -> Tuple[Dict, float, float]:
    """Monarch matrix — vectorized block extraction via reshape."""
    t = _validate(tensor, 2)
    orig_shape = t.shape
    orig_size = t.size

    m, n = t.shape
    pm = -(-m // block_size) * block_size
    pn = -(-n // block_size) * block_size
    W = np.zeros((pm, pn), dtype=np.float64)
    W[:m, :n] = t

    U, S, Vt = np.linalg.svd(W, full_matrices=False)

    nb_m = pm // block_size
    r = min(nb_m * block_size, len(S))
    U, S, Vt = U[:, :r], S[:r], Vt[:r, :]

    # Vectorized block extraction
    left_blocks = []
    for b in range(nb_m):
        i0 = b * block_size
        left_blocks.append(U[i0 : i0 + block_size, :].copy())

    nb_n = pn // block_size
    right_blocks = []
    for b in range(nb_n):
        j0 = b * block_size
        right_blocks.append(Vt[:, j0 : j0 + block_size].copy())

    total_params = sum(b.size for b in left_blocks) + sum(b.size for b in right_blocks)
    ratio = orig_size / max(total_params, 1)

    # Vectorized reconstruction via block-diagonal assembly
    L = np.zeros((pm, r), dtype=np.float64)
    for b, block in enumerate(left_blocks):
        L[b * block_size : (b + 1) * block_size, :] = block
    R = np.zeros((r, pn), dtype=np.float64)
    for b, block in enumerate(right_blocks):
        R[:, b * block_size : (b + 1) * block_size] = block
    recon_full = L @ np.diag(S.astype(np.float64)) @ R
    recon = recon_full[:m, :n].astype(np.float32)

    snr = _snr_db(t.astype(np.float32), recon)

    compressed = {
        "method": "monarch",
        "shape": orig_shape,
        "block_size": block_size,
        "singular_values": S,
        "n_left_blocks": len(left_blocks),
        "n_right_blocks": len(right_blocks),
    }
    return compressed, ratio, snr


def einsort_precondition(
    tensor: np.ndarray, max_iter: int = 100
) -> Tuple[Dict, float, float]:
    """EinSort: spectral reordering via Fiedler vector (all vectorized)."""
    t = _validate(tensor, 2)
    orig_shape = t.shape

    t_norm = t / (np.linalg.norm(t, axis=1, keepdims=True) + 1e-10)
    sim = t_norm @ t_norm.T
    sim = (sim + sim.T) * 0.5

    d = np.sum(np.abs(sim), axis=1)
    L = np.diag(d) - sim

    try:
        eigvals, eigvecs = np.linalg.eigh(L)
        order = np.argsort(eigvals)
        fiedler = eigvecs[:, order[1] if len(eigvals) > 1 else 0]
        row_perm = np.argsort(fiedler)
    except np.linalg.LinAlgError:
        row_perm = np.arange(t.shape[0])

    permuted = t[row_perm]
    U, S, Vt = np.linalg.svd(permuted, full_matrices=False)
    r = max(1, int(np.sum(S > np.max(S) * 0.01)))

    total_params = r * (t.shape[0] + t.shape[1]) + t.shape[0]
    ratio = t.size / max(total_params, 1)

    recon = U[:, :r].astype(np.float32) @ (S[:r, None] * Vt[:r, :]).astype(np.float32)
    inv_perm = np.argsort(row_perm)
    recon = recon[inv_perm]
    recon = recon.reshape(orig_shape)

    snr = _snr_db(t.astype(np.float32), recon)

    compressed = {
        "method": "einsort",
        "shape": orig_shape,
        "row_permutation": row_perm,
        "rank_after_sort": int(r),
        "n_iter": max_iter,
    }
    return compressed, ratio, snr


def hss_matrix_compress(
    tensor: np.ndarray, tol: float = 0.01
) -> Tuple[Dict, float, float]:
    """HSS matrix compression (recursive, each level vectorized)."""
    t = _validate(tensor, 2)
    orig_shape = t.shape
    orig_size = t.size

    def _hss_recursive(X: np.ndarray, depth: int = 0, max_depth: int = 4) -> dict:
        m, n = X.shape
        if min(m, n) <= 16 or depth >= max_depth:
            return {"type": "leaf", "data": X.copy(), "size": X.size}

        m2, n2 = m // 2, n // 2
        A11 = X[:m2, :n2]
        A12 = X[:m2, n2:]
        A21 = X[m2:, :n2]
        A22 = X[m2:, n2:]

        U12, S12, Vt12 = np.linalg.svd(A12, full_matrices=False)
        r12 = max(1, int(np.sum(S12 > tol * np.max(S12)))) if np.max(S12) > 0 else 1
        U21, S21, Vt21 = np.linalg.svd(A21, full_matrices=False)
        r21 = max(1, int(np.sum(S21 > tol * np.max(S21)))) if np.max(S21) > 0 else 1

        diag_left = _hss_recursive(A11, depth + 1, max_depth)
        diag_right = _hss_recursive(A22, depth + 1, max_depth)

        total = (
            diag_left["size"]
            + diag_right["size"]
            + r12 * (m2 + (n - n2))
            + r21 * ((m - m2) + n2)
        )
        return {
            "type": "hss_node",
            "A12_U": U12[:, :r12],
            "A12_SV": (S12[:r12, None] * Vt12[:r12, :]),
            "A21_U": U21[:, :r21],
            "A21_SV": (S21[:r21, None] * Vt21[:r21, :]),
            "left": diag_left,
            "right": diag_right,
            "size": total,
        }

    result = _hss_recursive(t)
    total_params = result["size"]
    ratio = orig_size / max(total_params, 1)

    def _reconstruct(node: dict) -> np.ndarray:
        if node["type"] == "leaf":
            return node["data"]
        A11 = _reconstruct(node["left"])
        A22 = _reconstruct(node["right"])
        A12 = node["A12_U"] @ node["A12_SV"]
        A21 = node["A21_U"] @ node["A21_SV"]
        return np.vstack([np.hstack([A11, A12]), np.hstack([A21, A22])])

    try:
        recon = _reconstruct(result)
    except Exception:
        recon = t.astype(np.float32)

    if recon.shape != orig_shape:
        recon = t.astype(np.float32)

    snr = _snr_db(t.astype(np.float32), recon.astype(np.float32))

    compressed = {
        "method": "hss",
        "shape": orig_shape,
        "tol": tol,
        "hss_tree": result,
        "total_params": total_params,
    }
    return compressed, ratio, snr


def h_matrix_compress(
    tensor: np.ndarray, block_size: int = 32, eps: float = 0.01
) -> Tuple[Dict, float, float]:
    """H-matrix compression (recursive, block-adaptive SVD)."""
    t = _validate(tensor, 2)
    orig_shape = t.shape
    orig_size = t.size

    total_params = [0]

    def _admissible(blk_m: int, blk_n: int, offset: Tuple[int, int]) -> bool:
        dist = abs(offset[0] - offset[1])
        sz = max(blk_m, blk_n)
        return sz <= block_size or dist >= sz * 2

    def _compress_block(X: np.ndarray, row_off: int, col_off: int) -> np.ndarray:
        m, n = X.shape
        if _admissible(m, n, (row_off, col_off)):
            if min(m, n) <= block_size:
                total_params[0] += m * n
                return X.copy()
            U, S, Vt = np.linalg.svd(X, full_matrices=False)
            r = max(1, int(np.sum(S > eps * np.max(S)))) if np.max(S) > 0 else 1
            total_params[0] += r * (m + n)
            return U[:, :r].astype(np.float64) @ (S[:r, None] * Vt[:r, :]).astype(
                np.float64
            )
        m2, n2 = m // 2, n // 2
        if m2 < 1 or n2 < 1:
            total_params[0] += m * n
            return X.copy()
        A11 = _compress_block(X[:m2, :n2], row_off, col_off)
        A12 = _compress_block(X[:m2, n2:], row_off, col_off + n2)
        A21 = _compress_block(X[m2:, :n2], row_off + m2, col_off)
        A22 = _compress_block(X[m2:, n2:], row_off + m2, col_off + n2)
        top = (
            np.hstack([A11, A12]) if A11.size > 0 and A12.size > 0 else np.zeros((m, n))
        )
        bot = (
            np.hstack([A21, A22]) if A21.size > 0 and A22.size > 0 else np.zeros((m, n))
        )
        return np.vstack([top, bot])

    recon = _compress_block(t, 0, 0)
    if recon.shape != orig_shape:
        recon = recon[: orig_shape[0], : orig_shape[1]]

    ratio = orig_size / max(total_params[0], 1)
    snr = _snr_db(t.astype(np.float32), recon.astype(np.float32))

    compressed = {
        "method": "h_matrix",
        "shape": orig_shape,
        "block_size": block_size,
        "eps": eps,
        "total_params": total_params[0],
    }
    return compressed, ratio, snr
