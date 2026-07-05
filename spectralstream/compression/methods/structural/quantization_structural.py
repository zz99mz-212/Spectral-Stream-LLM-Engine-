"""
Structural methods that combine quantization with structural properties.
"""

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
    t = np.asarray(tensor, dtype=np.float64)
    if t.size == 0:
        raise ValueError("Empty tensor")
    return t


def _lloyd_max_quantize(
    data: np.ndarray, n_bits: int
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Lloyd-Max quantization — vectorized centroid update via np.bincount."""
    flat = data.ravel().astype(np.float64)
    n_levels = 1 << n_bits
    mu, sigma = np.mean(flat), np.std(flat)
    scale = max(abs(mu - 4.0 * sigma), abs(mu + 4.0 * sigma), 1e-8)
    normalized = np.clip(flat / scale, -1.0, 1.0)

    centroids = np.linspace(-1.0, 1.0, n_levels)
    for _ in range(50):
        boundaries = (centroids[1:] + centroids[:-1]) * 0.5
        indices = np.searchsorted(boundaries, normalized, side="right")
        indices = np.clip(indices, 0, n_levels - 1)
        sums = np.bincount(indices, weights=normalized, minlength=n_levels)
        counts = np.bincount(indices, minlength=n_levels)
        mask = counts > 0
        new_c = centroids.copy()
        new_c[mask] = sums[mask] / counts[mask]
        if np.allclose(centroids, new_c, atol=1e-6):
            centroids = new_c
            break
        centroids = new_c

    boundaries = (centroids[1:] + centroids[:-1]) * 0.5
    indices = np.searchsorted(boundaries, normalized, side="right")
    indices = np.clip(indices, 0, n_levels - 1).astype(np.uint8)
    return indices, centroids * scale, scale


def delta_quantize(
    tensor: np.ndarray, reference_tensor: np.ndarray = None, bits: int = 4
) -> Tuple[Dict, float, float]:
    """Delta quantization: compress difference from reference tensor.
    If reference is similar (e.g., adjacent layer), residuals are small.
    """
    t = _validate(tensor)
    orig_shape = t.shape
    orig_size = t.size

    if reference_tensor is not None:
        ref = np.asarray(reference_tensor, dtype=np.float64)
        if ref.shape != orig_shape:
            ref = np.zeros_like(t)
    else:
        # Use mean as reference if none provided
        ref = np.full_like(t, np.mean(t))

    residual = t - ref

    # Quantize the residual (small range → fewer bits)
    indices, centroids, scale = _lloyd_max_quantize(residual, bits)

    # Reconstruct
    quantized_residual = centroids[indices.reshape(orig_shape)]
    recon = ref.astype(np.float64) + quantized_residual

    # Storage: indices packed (bits per element), centroids (float64 * n_levels), ref (float64 * n)
    n_levels = 1 << bits
    compressed_bytes = (
        int(math.ceil(orig_size * bits / 8)) + n_levels * 8 + ref.size * 8
    )
    ratio = (orig_size * 4) / max(compressed_bytes, 1)
    snr = _snr_db(t.astype(np.float32), recon.astype(np.float32))

    compressed = {
        "method": "delta_quantize",
        "shape": orig_shape,
        "bits": bits,
        "indices": indices,
        "centroids": centroids,
        "scale": scale,
        "reference_mean": float(np.mean(ref)),
    }

    return compressed, ratio, snr


def hierarchical_delta(
    tensor: np.ndarray, levels: int = 3
) -> Tuple[Dict, float, float]:
    """Hierarchical delta encoding: pyramid of differences.
    Store coarse version, then deltas at finer levels.
    """
    t = _validate(tensor)
    orig_shape = t.shape
    orig_size = t.size

    flat = t.ravel()
    pyramid = []
    current = flat.copy()

    for level in range(levels):
        coarse_len = max(1, int(math.ceil(len(current) / 2)))
        # Coarse: downsample (average pairs)
        if len(current) % 2 == 1:
            coarse = np.zeros(coarse_len, dtype=np.float64)
            coarse[:-1] = (current[::2] + current[1::2]) * 0.5
            coarse[-1] = current[-1]
        else:
            coarse = (current[::2] + current[1::2]) * 0.5

        # Detail: difference from coarse
        detail = np.zeros(len(current), dtype=np.float64)
        detail[::2] = (
            current[::2] - np.repeat(coarse, 2)[::2][: len(current[::2])]
            if coarse_len > 1
            else current[::2] - coarse[0]
        )
        if len(current) > 1:
            detail[1::2] = current[1::2] - np.repeat(coarse, 2)[: len(current)][1::2]

        # Quantize detail with Lloyd-Max
        indices, centroids, scale = _lloyd_max_quantize(detail, max(2, 8 - level))

        pyramid.append(
            {
                "level": level,
                "detail_indices": indices,
                "detail_centroids": centroids,
                "detail_scale": scale,
                "detail_size": len(detail),
                "bits": max(2, 8 - level),
            }
        )
        current = coarse

    # Store coarsest level as float64
    coarsest = current

    # Reconstruct
    recon = coarsest.copy()
    for level_data in reversed(pyramid):
        detail_indices = level_data["detail_indices"]
        centroids = level_data["detail_centroids"]
        detail = centroids[detail_indices.astype(np.int32)]
        recon_coarse = recon.copy()
        recon = np.zeros(level_data["detail_size"], dtype=np.float64)
        n_c = len(recon_coarse)
        recon[: min(n_c * 2, len(recon)) : 2] = (
            recon_coarse[: min(n_c, len(recon) // 2)]
            + detail[: min(n_c * 2, len(recon)) : 2]
        )
        if len(recon) > 1:
            recon[1::2] = recon_coarse[: min(n_c, len(recon) // 2)] + detail[1::2]
    recon = recon[:orig_size].reshape(orig_shape)

    # Storage
    compressed_bytes = coarsest.size * 8  # coarsest as float64
    for ld in pyramid:
        n_levels = 1 << ld["bits"]
        compressed_bytes += (
            int(math.ceil(ld["detail_size"] * ld["bits"] / 8)) + n_levels * 8
        )

    ratio = (orig_size * 4) / max(compressed_bytes, 1)
    snr = _snr_db(t.astype(np.float32), recon.astype(np.float32))

    compressed = {
        "method": "hierarchical_delta",
        "shape": orig_shape,
        "levels": levels,
        "pyramid": pyramid,
        "coarsest": coarsest,
    }

    return compressed, ratio, snr


def basis_share(
    tensor: np.ndarray, shared_basis: np.ndarray = None, n_basis: int = 8
) -> Tuple[Dict, float, float]:
    """Basis sharing: express multiple tensors in shared basis.
    Only store coefficients, not the basis vectors.
    """
    t = _validate(tensor)
    orig_shape = t.shape
    orig_size = t.size

    if t.ndim == 1:
        t = t.reshape(1, -1)

    m, n = t.shape

    if shared_basis is not None:
        basis = np.asarray(shared_basis, dtype=np.float64)
        if basis.shape[0] != n:
            # Basis dimension mismatch, learn from tensor
            U, S, Vt = np.linalg.svd(t, full_matrices=False)
            basis = Vt[:n_basis, :].T if Vt.shape[0] >= n_basis else Vt.T
    else:
        # Learn basis from tensor via SVD
        U, S, Vt = np.linalg.svd(t, full_matrices=False)
        basis = (
            Vt[:n_basis, :].T
            if Vt.shape[0] >= n_basis
            else np.eye(n, dtype=np.float64)[:, :n_basis]
        )

    k = basis.shape[1]

    # Project tensor onto basis: coefficients = t @ basis  (t: m×n, basis: n×k)
    coefficients = t.astype(np.float64) @ basis

    # Reconstruct: t ≈ coefficients @ basis.T
    recon = (coefficients @ basis.T).astype(np.float32)
    if len(orig_shape) == 1:
        recon = recon.ravel()[: orig_shape[0]]

    # Storage: coefficients (float64) + basis vectors (float64, but marked as "shared")
    # If basis is truly shared externally, we only count coefficients
    compressed_bytes = coefficients.size * 8
    if shared_basis is None:
        # Must store basis too
        compressed_bytes += basis.size * 8

    ratio = (orig_size * 4) / max(compressed_bytes, 1)
    snr = _snr_db(np.asarray(tensor, dtype=np.float32), recon)

    compressed = {
        "method": "basis_share",
        "shape": orig_shape,
        "n_basis": k,
        "coefficients": coefficients,
        "basis": basis if shared_basis is None else None,
        "basis_is_shared": shared_basis is not None,
    }

    return compressed, ratio, snr
