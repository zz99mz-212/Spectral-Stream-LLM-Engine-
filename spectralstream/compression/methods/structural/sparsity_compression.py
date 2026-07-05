"""HPC sparsity — vectorized Wanda, Hessian-free SparseGPT, batched pruning."""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np


def _snr_db(original: np.ndarray, reconstructed: np.ndarray) -> float:
    mse = float(
        np.mean((original.astype(np.float64) - reconstructed.astype(np.float64)) ** 2)
    )
    signal = float(np.mean(original.astype(np.float64) ** 2))
    return 10.0 * math.log10(max(signal, 1e-30) / max(mse, 1e-30))


def _validate(tensor) -> np.ndarray:
    t = np.asarray(tensor, dtype=np.float32)
    if t.size == 0:
        raise ValueError("Empty tensor")
    return t


def magnitude_prune(
    tensor: np.ndarray, sparsity: float = 0.5
) -> Tuple[Dict, float, float]:
    """Magnitude pruning: zero out smallest absolute weights. O(n) via argpartition."""
    t = _validate(tensor)
    orig_shape = t.shape
    orig_size = t.size

    n_keep = max(1, int(orig_size * (1.0 - sparsity)))
    abs_flat = np.abs(t).ravel()
    if n_keep < orig_size:
        kth = orig_size - n_keep
        threshold = -np.partition(-abs_flat, kth - 1)[kth - 1]
        mask = abs_flat >= threshold
        if np.sum(mask) != n_keep:
            order = np.argpartition(-abs_flat, n_keep - 1)[:n_keep]
            mask = np.zeros(orig_size, dtype=bool)
            mask[order] = True
        mask = mask.reshape(orig_shape)
    else:
        mask = np.ones(orig_shape, dtype=bool)

    kept_values = t[mask]
    recon = np.zeros(orig_size, dtype=np.float32)
    recon[mask.ravel()] = kept_values
    recon = recon.reshape(orig_shape)

    mask_bytes = int(math.ceil(orig_size / 8))
    compressed_bytes = kept_values.size * 4 + mask_bytes
    ratio = (orig_size * 4) / max(compressed_bytes, 1)
    snr = _snr_db(t, recon)

    compressed = {
        "method": "magnitude_prune",
        "shape": orig_shape,
        "sparsity": sparsity,
        "n_kept": int(n_keep),
        "n_total": orig_size,
        "mask": mask,
        "values": kept_values,
    }
    return compressed, ratio, snr


def gradient_prune(
    tensor: np.ndarray, importance_scores: np.ndarray = None, sparsity: float = 0.5
) -> Tuple[Dict, float, float]:
    """Gradient-based pruning."""
    t = _validate(tensor)
    orig_shape = t.shape
    orig_size = t.size

    if importance_scores is not None:
        scores = np.asarray(importance_scores, dtype=np.float32).ravel()
        if scores.size != orig_size:
            scores = np.abs(t).ravel()
    else:
        rng = np.random.RandomState(42)
        grad = rng.randn(orig_size).astype(np.float32)
        scores = np.abs(t.ravel()) * np.abs(grad)

    n_keep = max(1, int(orig_size * (1.0 - sparsity)))
    if n_keep < orig_size:
        threshold = -np.partition(-scores, n_keep - 1)[n_keep - 1]
        mask_flat = scores >= threshold
        if np.sum(mask_flat) != n_keep:
            order = np.argpartition(-scores, n_keep - 1)[:n_keep]
            mask_flat = np.zeros(orig_size, dtype=bool)
            mask_flat[order] = True
    else:
        mask_flat = np.ones(orig_size, dtype=bool)

    mask = mask_flat.reshape(orig_shape)
    kept_values = t[mask]
    recon = np.zeros(orig_size, dtype=np.float32)
    recon[mask_flat] = kept_values
    recon = recon.reshape(orig_shape)

    mask_bytes = int(math.ceil(orig_size / 8))
    compressed_bytes = kept_values.size * 4 + mask_bytes
    ratio = (orig_size * 4) / max(compressed_bytes, 1)
    snr = _snr_db(t, recon)

    compressed = {
        "method": "gradient_prune",
        "shape": orig_shape,
        "sparsity": sparsity,
        "n_kept": int(n_keep),
        "mask": mask,
        "values": kept_values,
        "scores": scores[mask_flat],
    }
    return compressed, ratio, snr


def movement_prune(
    tensor: np.ndarray, prev_weights: np.ndarray = None, sparsity: float = 0.5
) -> Tuple[Dict, float, float]:
    """Movement pruning: prune weights moving toward zero."""
    t = _validate(tensor)
    orig_shape = t.shape
    orig_size = t.size

    if prev_weights is not None:
        prev = np.asarray(prev_weights, dtype=np.float32)
        scores = np.abs(t).ravel() - np.abs(prev).ravel()
    else:
        scores = np.abs(t).ravel()

    n_keep = max(1, int(orig_size * (1.0 - sparsity)))
    if n_keep < orig_size:
        threshold = -np.partition(-scores, n_keep - 1)[n_keep - 1]
        mask_flat = scores >= threshold
        if np.sum(mask_flat) != n_keep:
            order = np.argpartition(-scores, n_keep - 1)[:n_keep]
            mask_flat = np.zeros(orig_size, dtype=bool)
            mask_flat[order] = True
    else:
        mask_flat = np.ones(orig_size, dtype=bool)

    mask = mask_flat.reshape(orig_shape)
    kept_values = t[mask]
    recon = np.zeros(orig_size, dtype=np.float32)
    recon[mask_flat] = kept_values
    recon = recon.reshape(orig_shape)

    mask_bytes = int(math.ceil(orig_size / 8))
    compressed_bytes = kept_values.size * 4 + mask_bytes
    ratio = (orig_size * 4) / max(compressed_bytes, 1)
    snr = _snr_db(t, recon)

    compressed = {
        "method": "movement_prune",
        "shape": orig_shape,
        "sparsity": sparsity,
        "n_kept": int(n_keep),
        "mask": mask,
        "values": kept_values,
    }
    return compressed, ratio, snr


def wanda_prune(
    tensor: np.ndarray, input_features: np.ndarray, sparsity: float = 0.5
) -> Tuple[Dict, float, float]:
    """Wanda pruning — vectorized row-wise top-k selection."""
    t = _validate(tensor)
    orig_shape = t.shape
    orig_size = t.size

    input_f = np.asarray(input_features, dtype=np.float32)
    if t.ndim == 1:
        t = t.reshape(1, -1)
    if input_f.ndim == 1:
        input_f = input_f.reshape(-1, 1)

    col_norms = np.linalg.norm(input_f, axis=0)
    abs_w = np.abs(t)
    scores = abs_w * col_norms.reshape(1, -1) if len(col_norms) == t.shape[1] else abs_w

    n_per_row = max(1, int(t.shape[1] * (1.0 - sparsity)))

    # Vectorized per-row top-k: partition across all rows simultaneously
    sort_idx = np.argpartition(-scores, n_per_row - 1, axis=1)
    keep_idx = sort_idx[:, :n_per_row]
    mask = np.zeros_like(t, dtype=bool)
    row_arange = np.arange(t.shape[0])[:, None]
    mask[row_arange, keep_idx] = True

    kept_values = t[mask]
    recon = np.zeros_like(t)
    recon[mask] = kept_values
    if len(orig_shape) == 1:
        recon = recon.ravel()[: orig_shape[0]]

    n_kept = int(np.sum(mask))
    mask_bytes = int(math.ceil(orig_size / 8))
    compressed_bytes = n_kept * 4 + mask_bytes
    ratio = (orig_size * 4) / max(compressed_bytes, 1)
    snr = _snr_db(
        np.asarray(tensor, dtype=np.float32).reshape(orig_shape),
        recon.reshape(orig_shape),
    )

    compressed = {
        "method": "wanda_prune",
        "shape": orig_shape,
        "sparsity": sparsity,
        "n_kept": n_kept,
        "n_per_row": n_per_row,
        "mask": mask,
        "values": kept_values,
    }
    return compressed, ratio, snr


def sparsegpt_prune(
    tensor: np.ndarray, hessian: np.ndarray = None, sparsity: float = 0.5
) -> Tuple[Dict, float, float]:
    """SparseGPT — optimized column processing with precomputed H_inv and slice updates."""
    t = _validate(tensor)
    orig_shape = t.shape
    orig_size = t.size

    if t.ndim == 1:
        t = t.reshape(-1, 1)

    m, n = t.shape

    if hessian is not None:
        H = np.asarray(hessian, dtype=np.float64)
        if H.shape != (n, n):
            H = np.eye(n, dtype=np.float64)
    else:
        rng = np.random.RandomState(42)
        X = rng.randn(n, n * 2).astype(np.float64)
        H = X @ X.T + 1e-4 * np.eye(n)

    trace_H = np.trace(H)
    H_reg = H + (1e-6 * trace_H / n) * np.eye(n)
    try:
        H_inv = np.linalg.inv(H_reg)
    except np.linalg.LinAlgError:
        H_inv = np.eye(n, dtype=np.float64)

    h_inv_diag = np.diag(H_inv)

    W = t.astype(np.float64).copy()
    mask = np.ones((m, n), dtype=bool)
    n_prune_total = int(n * sparsity)
    cols_pruned = 0

    # Precompute adaptive pruning per column
    for col in range(n):
        w_col = W[:, col]
        h_inv_cc = H_inv[col, col]
        importance = (w_col * w_col) / (2.0 * h_inv_cc + 1e-30)

        n_prune_now = max(0, n_prune_total - (n - col - 1))
        if n_prune_now <= 0 or col >= n:
            continue

        # O(n) selection via argpartition
        prune_idx = np.argpartition(-importance, n_prune_now - 1)[:n_prune_now]

        # Error compensation: vectorized outer product update on suffix
        deltas = w_col[prune_idx] / (h_inv_cc + 1e-30)
        # Use slice on suffix for cache efficiency
        W[np.ix_(prune_idx, np.arange(col, n))] += deltas[:, None] * H_inv[col, col:]
        w_col[prune_idx] = 0.0
        mask[prune_idx, col] = False
        cols_pruned += n_prune_now

    kept_values = t[mask.reshape(orig_shape)]
    recon = np.zeros_like(W, dtype=np.float32)
    recon[mask] = kept_values.astype(np.float32)
    if len(orig_shape) == 1:
        recon = recon.ravel()[: orig_shape[0]]

    n_kept = int(np.sum(mask))
    mask_bytes = int(math.ceil(orig_size / 8))
    compressed_bytes = n_kept * 4 + mask_bytes
    ratio = (orig_size * 4) / max(compressed_bytes, 1)
    snr = _snr_db(np.asarray(tensor, dtype=np.float32), recon.reshape(orig_shape))

    compressed = {
        "method": "sparsegpt",
        "shape": orig_shape,
        "sparsity": sparsity,
        "n_kept": n_kept,
        "mask": mask.reshape(orig_shape),
        "values": kept_values,
    }
    return compressed, ratio, snr
