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

class ResonanceCompression(CompressionMethod):
    """Find resonant frequencies (eigenvalues) of W^T W for compression.

    Mathematical basis:
        The resonant frequencies of a matrix are the eigenvalues of W^T W.
        These are the squared singular values: omega_k = sigma_k^2.

        We decompose the weight matrix into resonant modes:
            W = sum_k sigma_k * u_k * v_k^T

        and store only the significant resonant modes.  The key insight is
        that many real-world weight matrices have a few dominant resonances.

    Algorithm:
        1. Compute eigenvalues of W^T W (or equivalently, singular values)
        2. Identify dominant resonant modes
        3. Store: mode indices, amplitudes, eigenvectors
        4. Reconstruct by superposition of resonant modes

    Storage: O(K * (m + n)) where K = number of significant resonances.
    """
    name = "resonance"
    category = "hybrid"

    def compress(self, tensor, energy_threshold=0.99, **kw):
        t, orig = _ensure_2d(tensor)
        m, n = t.shape
        W = t.astype(np.float64)

        # Compute resonant frequencies (singular values)
        U, S, Vt = np.linalg.svd(W, full_matrices=False)

        # Find number of modes needed for energy threshold
        total_energy = np.sum(S ** 2)
        cumulative = np.cumsum(S ** 2) / (total_energy + 1e-30)
        n_modes = int(np.searchsorted(cumulative, energy_threshold)) + 1
        n_modes = max(1, min(n_modes, len(S)))

        # Store resonant modes
        return {
            "U": U[:, :n_modes].astype(np.float32),
            "S": S[:n_modes].astype(np.float32),
            "Vt": Vt[:n_modes, :].astype(np.float32),
            "n_modes": n_modes,
            "total_energy": float(total_energy),
            "shape": t.shape,
        }, {"orig_shape": orig}

    def decompress(self, cd, meta):
        result = cd["U"].astype(np.float64) @ np.diag(cd["S"].astype(np.float64)) @ cd["Vt"].astype(np.float64)
        return _restore_shape(result.astype(np.float32), meta["orig_shape"])

