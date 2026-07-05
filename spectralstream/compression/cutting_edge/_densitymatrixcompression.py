from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from ._compressionmethod import CompressionMethod, ALL_METHODS, _ensure_2d, _restore_shape, _safe_bytes


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
        return sum(_safe_bytes(v) for v in data.values()) + sum(_safe_bytes(k) for k in data.keys())
    if isinstance(data, (list, tuple)):
        return sum(_safe_bytes(x) for x in data)
    if isinstance(data, (int, float, np.integer, np.floating)):
        return 8
    if isinstance(data, str):
        return len(data)
    return 0

class DensityMatrixCompression(CompressionMethod):
    """Represent weight distribution as a quantum density matrix.

    Mathematical basis:
        The density matrix rho = sum_i p_i |psi_i><psi_i| encodes
        the full statistical structure of the weight distribution.
        Eigenvalue decomposition: rho = sum_k lambda_k |e_k><e_k|

        By keeping only the top-K eigenvalues, we compress the
        distribution while preserving its most important features.

    Algorithm:
        1. Compute empirical covariance: Sigma = (1/n) W^T W
        2. Eigendecomposition: Sigma = Q Lambda Q^T
        3. Keep top-K eigenvalues (thermal state approximation)
        4. Store: eigenvectors Q_K, eigenvalues Lambda_K, mean mu

    Storage: O(K*n + K) instead of O(m*n).
    """
    name = "density_matrix"
    category = "quantum_mechanics"

    def compress(self, tensor, n_components=16, **kw):
        t, orig = _ensure_2d(tensor)
        m, n = t.shape
        k = min(n_components, min(m, n))

        mu = t.mean(axis=0).astype(np.float64)
        centered = t.astype(np.float64) - mu[None, :]

        # Empirical covariance (density matrix analog)
        cov = (centered.T @ centered) / max(m - 1, 1)

        # Eigendecomposition
        eigvals, eigvecs = np.linalg.eigh(cov)

        # Sort by eigenvalue magnitude (descending)
        order = np.argsort(np.abs(eigvals))[::-1]
        eigvals = eigvals[order[:k]]
        eigvecs = eigvecs[:, order[:k]]

        # Project data onto eigenvectors
        coefficients = centered @ eigvecs  # (m, k)

        return {
            "eigvals": eigvals.astype(np.float64),
            "eigvecs": eigvecs.astype(np.float32),
            "coeffs": coefficients.astype(np.float32),
            "mu": mu.astype(np.float32),
            "k": k,
            "shape": t.shape,
        }, {"orig_shape": orig}

    def decompress(self, cd, meta):
        mu = cd["mu"].astype(np.float64)
        eigvecs = cd["eigvecs"].astype(np.float64)
        coeffs = cd["coeffs"].astype(np.float64)

        # Reconstruction: W = mu + coeffs @ eigvecs^T
        result = mu[None, :] + coeffs @ eigvecs.T
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

