"""
Block-structured compression methods.
All methods return: (compressed: dict, ratio: float, snr_db: float)
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

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


def block_sparse_compress(
    tensor: np.ndarray, block_size: int = 16, density: float = 0.5
) -> Tuple[Dict, float, float]:
    """Block-sparse compression. Keep only the most important blocks.
    Each block is either fully kept or fully zeroed.
    """
    t = _validate(tensor)
    orig_shape = t.shape
    orig_size = t.size
    block_size = max(1, int(block_size))
    density = float(np.clip(density, 0.0, 1.0))
    flat = t.ravel()

    # Pad to block boundary
    n = flat.size
    pad = -n % block_size
    if pad:
        flat = np.pad(flat, (0, pad), mode="constant")
    n_padded = len(flat)

    # Reshape into blocks
    n_blocks = n_padded // block_size
    blocks = flat.reshape(n_blocks, block_size)
    norms = np.linalg.norm(blocks, axis=1)

    # Keep top density blocks
    n_keep = max(1, min(int(density * n_blocks), n_blocks))
    threshold = np.sort(norms)[-n_keep] if n_keep < n_blocks else -1.0
    mask = norms >= threshold
    # Ensure exactly n_keep blocks kept (break ties)
    if np.sum(mask) != n_keep:
        order = np.argsort(-norms)
        mask[:] = False
        mask[order[:n_keep]] = True

    kept_values = blocks[mask].ravel()

    # Reconstruct
    recon_flat = np.zeros(n, dtype=np.float32)
    elem_mask = np.repeat(mask, block_size)[:n]
    recon_flat[elem_mask] = kept_values
    recon = recon_flat.reshape(orig_shape)

    # Compressed bytes: kept float32 values + bitmask
    mask_bytes = int(math.ceil(n_blocks / 8))
    compressed_bytes = kept_values.size * 4 + mask_bytes
    ratio = (orig_size * 4) / max(compressed_bytes, 1)

    snr = _snr_db(t, recon)

    compressed = {
        "method": "block_sparse",
        "shape": orig_shape,
        "block_size": block_size,
        "density": density,
        "n_blocks_kept": int(n_keep),
        "n_blocks_total": n_blocks,
        "mask": mask,
    }

    return compressed, ratio, snr


def structured_nm_sparsity(
    tensor: np.ndarray, n: int = 2, m: int = 4
) -> Tuple[Dict, float, float]:
    """N:M structured sparsity. Fully vectorized per-group top-N selection."""
    t = _validate(tensor)
    orig_shape = t.shape
    orig_size = t.size
    flat = t.ravel()

    pad = -orig_size % m
    if pad:
        flat = np.pad(flat, (0, pad), mode="constant")
    n_total = len(flat)
    n_groups = n_total // m

    groups = flat.reshape(n_groups, m)
    sort_idx = np.argpartition(-np.abs(groups), n - 1, axis=1)
    mask_flat = np.zeros(n_total, dtype=bool)
    row_offsets = np.repeat(np.arange(n_groups), n)
    col_pos = sort_idx[:, :n].ravel()
    mask_flat[row_offsets * m + col_pos] = True

    mask = mask_flat[:orig_size].reshape(orig_shape)
    kept_values = t[mask]

    recon = np.zeros_like(flat[:orig_size], dtype=np.float32)
    recon[mask.ravel()] = kept_values
    recon = recon.reshape(orig_shape)

    compressed_bytes = kept_values.size * 4 + kept_values.size * 2
    ratio = (orig_size * 4) / max(compressed_bytes, 1)
    snr = _snr_db(t, recon)

    compressed = {
        "method": "structured_nm",
        "shape": orig_shape,
        "n": n,
        "m": m,
        "mask": mask,
        "values": kept_values,
    }
    return compressed, ratio, snr


def group_lasso(
    tensor: np.ndarray, lambda_reg: float = 0.01
) -> Tuple[Dict, float, float]:
    """Group Lasso compression. Encourage group-wise sparsity (rows or columns).
    Entire rows/columns of weights are zeroed out if unimportant.
    """
    t = _validate(tensor)
    orig_shape = t.shape
    orig_size = t.size

    if t.ndim == 1:
        t = t.reshape(-1, 1)

    m, n = t.shape
    t_abs = np.abs(t)

    # Row norms
    row_norms = np.linalg.norm(t, axis=1)
    row_thresh = lambda_reg * np.max(row_norms)
    row_mask = row_norms >= row_thresh

    # Column norms (on row-pruned version)
    t_pruned = t[row_mask] if np.any(row_mask) else t
    col_norms = (
        np.linalg.norm(t_pruned, axis=0)
        if t_pruned.ndim > 1 and t_pruned.size > 0
        else np.ones(n)
    )
    col_thresh = lambda_reg * np.max(col_norms) if np.max(col_norms) > 0 else 0.0
    col_mask = col_norms >= col_thresh

    # Apply both masks
    recon = np.zeros((m, n), dtype=np.float32)
    if np.any(row_mask) and np.any(col_mask):
        recon[np.ix_(row_mask, col_mask)] = t[np.ix_(row_mask, col_mask)]

    # Restore original shape
    if len(orig_shape) == 1:
        recon = recon.ravel()[: orig_shape[0]]

    # Count stored elements
    n_kept = int(np.sum(row_mask) * np.sum(col_mask))
    n_zeros_removed = m * n - n_kept

    # Storage: kept float32 values + row_mask bits + col_mask bits
    compressed_bytes = n_kept * 4 + int(math.ceil(m / 8)) + int(math.ceil(n / 8))
    ratio = (orig_size * 4) / max(compressed_bytes, 1)
    snr = _snr_db(t.reshape(m, n) if t.ndim > 1 else t, recon)

    compressed = {
        "method": "group_lasso",
        "shape": orig_shape,
        "lambda": lambda_reg,
        "row_mask": row_mask,
        "col_mask": col_mask,
        "n_rows_kept": int(np.sum(row_mask)),
        "n_cols_kept": int(np.sum(col_mask)),
    }

    return compressed, ratio, snr
