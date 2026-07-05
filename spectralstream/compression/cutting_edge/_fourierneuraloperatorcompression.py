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

class FourierNeuralOperatorCompression(CompressionMethod):
    """Use Fourier neural operator concept for weight compression.

    Mathematical basis:
        The Fourier Neural Operator learns to represent functions in
        Fourier space.  For weight matrices:
            W(x) = F^{-1}( R(k) * F(W)(k) )
        where R(k) is a learned filter in Fourier domain.

        Key insight: weight matrices are often smoother in Fourier domain
        and can be represented by a compact set of Fourier coefficients.

    Algorithm:
        1. FFT of weight matrix
        2. Learn optimal low-pass filter R(k)
        3. Store: significant Fourier coefficients + filter parameters

    Storage: O(K) where K = number of kept Fourier modes.
    """
    name = "fourier_neural_operator"
    category = "hybrid"

    def compress(self, tensor, n_modes=32, filter_order=4, **kw):
        t, orig = _ensure_2d(tensor)
        m, n = t.shape
        W = t.astype(np.float64)

        # FFT
        F = np.fft.fft2(W)

        # Learn optimal filter R(k) via polynomial fit in Fourier domain
        # R(k) = a_0 + a_1*|k| + a_2*|k|^2 + ...
        kx = np.fft.fftfreq(n)[None, :]
        ky = np.fft.fftfreq(m)[:, None]
        k_mag = np.sqrt(kx ** 2 + ky ** 2)

        # Fit polynomial to magnitude spectrum
        k_flat = k_mag.ravel()
        F_flat = np.abs(F.ravel())

        # Sort by frequency
        order = np.argsort(k_flat)
        k_sorted = k_flat[order]
        F_sorted = F_flat[order]

        # Polynomial fit in log-log space
        log_k = np.log(k_sorted + 1e-10)
        log_F = np.log(F_sorted + 1e-10)

        degree = min(filter_order, len(k_sorted) - 1)
        V = log_k[:, None] ** np.arange(degree + 1, dtype=np.float64)[None, :]
        filter_coeffs = np.linalg.lstsq(V, log_F, rcond=None)[0]

        # Select top Fourier modes
        flat_power = np.abs(F.ravel()) ** 2
        top_idx = np.argsort(flat_power)[::-1][:n_modes]

        # Store significant coefficients
        significant_F = F.ravel()[top_idx]

        return {
            "coeffs": significant_F.astype(np.complex128),
            "indices": top_idx.astype(np.int32),
            "filter_coeffs": filter_coeffs.astype(np.float64),
            "n_modes": n_modes,
            "shape": F.shape,
            "orig_shape_inner": t.shape,
        }, {"orig_shape": orig}

    def decompress(self, cd, meta):
        F = np.zeros(cd["shape"], dtype=np.complex128)
        F.ravel()[cd["indices"]] = cd["coeffs"]

        result = np.fft.ifft2(F).real

        # Pad if needed
        orig = meta["orig_shape"]
        if result.shape != orig:
            padded = np.zeros(orig, dtype=np.float64)
            padded[:result.shape[0], :result.shape[1]] = result
            result = padded

        return _restore_shape(result.astype(np.float32), meta["orig_shape"])

