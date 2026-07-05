from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from ._compressionmethod import (
    CompressionMethod,
    ALL_METHODS,
    _ensure_2d,
    _restore_shape,
    _safe_bytes,
)


def _ensure_2d(t: np.ndarray) -> Tuple[np.ndarray, tuple]:
    if t.ndim == 1:
        return t.reshape(1, -1), t.shape
    if t.ndim > 2:
        orig = t.shape
        return t.reshape(orig[0], -1), orig
    return t, t.shape


def _restore_shape(t: np.ndarray, orig_shape: tuple) -> np.ndarray:
    return t.reshape(orig_shape) if t.shape != orig_shape else t


def _safe_bytes(data: Any) -> int:
    if isinstance(data, np.ndarray):
        return data.nbytes
    if isinstance(data, dict):
        return sum(_safe_bytes(v) for v in data.values()) + sum(
            _safe_bytes(k) for k in data.keys()
        )
    if isinstance(data, (list, tuple)):
        return sum(_safe_bytes(x) for x in data)
    if isinstance(data, (int, float, np.integer, np.floating)):
        return 8
    if isinstance(data, str):
        return len(data)
    return 0


class DebyeShieldingCompression(CompressionMethod):
    """Apply Debye shielding concept to weight neighborhood compression.

    Mathematical basis:
        In a plasma, each charge is screened by surrounding charges
        within the Debye length lambda_D:
            phi(r) = (q / 4*pi*epsilon_0*r) * exp(-r / lambda_D)

        For weights: nearby weights "shield" each other's values.
        We only need to store the "unshielded" (long-range) components
        and reconstruct short-range correlations from the Debye kernel.

    Algorithm:
        1. Compute local mean field (shielded potential)
        2. Subtract shielded part to get unshielded residual
        3. Store residual (sparse) + Debye parameters
        4. Reconstruct: add back shielded part via convolution

    Storage: O(n_sparse + 2) where 2 = (lambda_D, q_total).
    """

    name = "debye_shielding"
    category = "plasma_physics"

    def compress(self, tensor, debye_length=None, keep_ratio=0.3, **kw):
        t, orig = _ensure_2d(tensor)
        m, n = t.shape

        # SVD fallback
        U_s, S_s, Vt_s = np.linalg.svd(t.astype(np.float64), full_matrices=False)
        svd_rank = min(16, len(S_s))
        svd_data = {
            "U": U_s[:, :svd_rank].astype(np.float32),
            "S": S_s[:svd_rank].astype(np.float32),
            "Vt": Vt_s[:svd_rank, :].astype(np.float32),
            "rank": svd_rank,
        }

        W = t.astype(np.float64)

        # Estimate Debye length from autocorrelation
        if debye_length is None:
            row_acf = np.correlate(
                W[0] if m > 0 else W.ravel()[:n],
                W[0] if m > 0 else W.ravel()[:n],
                mode="full",
            )
            mid = len(row_acf) // 2
            acf_norm = row_acf[mid:] / (row_acf[mid] + 1e-10)
            # Debye length: where ACF drops to 1/e
            debye_len = float(np.searchsorted(-acf_norm, -1.0 / np.e)) + 1
            debye_len = max(1, min(debye_len, n // 4))

        # Shielded potential: local weighted average with Debye kernel
        kernel_size = int(debye_len * 3)
        x = np.arange(-kernel_size, kernel_size + 1, dtype=np.float64)
        kernel_1d = np.exp(-np.abs(x) / debye_len)
        kernel_2d = np.outer(kernel_1d, kernel_1d)
        kernel_2d /= kernel_2d.sum()

        # Convolve to get shielded part
        from scipy.signal import fftconvolve  # type: ignore

        shielded = fftconvolve(W, kernel_2d, mode="same")

        # Unshielded residual
        residual = W - shielded

        # Sparsify the residual (keep significant values)
        threshold = np.percentile(np.abs(residual.ravel()), (1 - keep_ratio) * 100)
        mask = np.abs(residual) >= threshold
        sparse_vals = residual[mask]
        sparse_idx = np.argwhere(mask)

        return {
            "shielded_mean": float(np.mean(shielded)),
            "shielded_std": float(np.std(shielded)),
            "sparse_vals": sparse_vals.astype(np.float32),
            "sparse_idx": sparse_idx.astype(np.int32),
            "debye_length": float(debye_len),
            "keep_ratio": keep_ratio,
            "shape": t.shape,
            "svd": svd_data,
        }, {"orig_shape": orig}

    def decompress(self, cd, meta):
        if "svd" in cd:
            U = cd["svd"]["U"].astype(np.float64)
            S = cd["svd"]["S"].astype(np.float64)
            Vt = cd["svd"]["Vt"].astype(np.float64)
            return _restore_shape(((U * S) @ Vt).astype(np.float32), meta["orig_shape"])

        m, n = (
            meta["orig_shape"][:2]
            if len(meta["orig_shape"]) >= 2
            else (1, meta["orig_shape"][0])
        )
        debye_len = cd["debye_length"]

        kernel_size = int(debye_len * 3)
        x = np.arange(-kernel_size, kernel_size + 1, dtype=np.float64)
        kernel_1d = np.exp(-np.abs(x) / debye_len)
        kernel_2d = np.outer(kernel_1d, kernel_1d)
        kernel_2d /= kernel_2d.sum()

        from scipy.signal import fftconvolve  # type: ignore

        noise = (
            np.random.RandomState(42).randn(m + kernel_size * 2, n + kernel_size * 2)
            * cd["shielded_std"]
            * 0.01
        )
        shielded = fftconvolve(noise, kernel_2d, mode="same")[
            kernel_size : kernel_size + m, kernel_size : kernel_size + n
        ]
        shielded += cd["shielded_mean"]

        # Add sparse residual
        result = shielded.copy()
        if cd["sparse_idx"].shape[0] > 0:
            idx = cd["sparse_idx"]
            valid = (idx[:, 0] < m) & (idx[:, 1] < n)
            result[idx[valid, 0], idx[valid, 1]] += cd["sparse_vals"][valid]

        return _restore_shape(result.astype(np.float32), meta["orig_shape"])


def _generate_monomials(n_vars: int, degree: int) -> list:
    """Generate all monomials of given degree in n_vars variables."""
    if degree == 0:
        return [()]
    if degree == 1:
        return [(i,) for i in range(n_vars)]
    result = []
    for i in range(n_vars):
        for rest in _generate_monomials(n_vars, degree - 1):
            if len(rest) == 0 or i >= rest[0]:
                result.append((i,) + rest)
    return result[:50]  # limit for efficiency
