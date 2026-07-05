from __future__ import annotations

import cmath
import math
import struct
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _ser(arr: np.ndarray) -> bytes:
    return np.ascontiguousarray(arr).tobytes()

def _deser(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

def _quantize(t: np.ndarray, bits: int = 8) -> Tuple[np.ndarray, float, float]:
    lo, hi = t.min(), t.max()
    if hi - lo < 1e-30:
        return np.zeros_like(t, dtype=np.uint8), lo, hi
    scale = (2**bits - 1) / (hi - lo)
    q = np.round((t - lo) * scale).astype(np.uint8)
    return q, float(scale), float(lo)

def _dequantize(q: np.ndarray, scale: float, lo: float, dtype=np.float32) -> np.ndarray:
    return (q.astype(dtype) / scale + lo).astype(dtype)

class FractionalFourier:
    """G17: F^α[w](u) = ∫K_α(u,t)w(t)dt, optimal T-F rotation."""

    name = "fractional_fourier"
    category = "novel_signal"

    def _frft(self, x: np.ndarray, alpha: float) -> np.ndarray:
        n = len(x)
        phi = alpha * math.pi / 2
        A = cmath.exp(-1j * (math.pi / 4 - phi / 2)) / math.sqrt(
            abs(math.sin(phi)) + 1e-30
        )
        t = np.linspace(-1, 1, n)
        chirp1 = np.exp(1j * math.pi * t**2 * (math.tan(phi / 2) + 1e-30))
        chirp2 = np.exp(-1j * math.pi * t**2 / (math.sin(phi) + 1e-30))
        y = chirp1 * x * chirp2
        return A * chirp1 * np.fft.fft(y) / math.sqrt(n)

    def compress(
        self, tensor: np.ndarray, n_alpha_candidates: int = 8
    ) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        flat = t.ravel()
        n = len(flat)
        best_alpha = 0.0
        best_sparsity = float("inf")
        best_coeffs = None
        for a in np.linspace(0.01, 1.99, n_alpha_candidates):
            coeffs = self._frft(flat, a)
            sparsity = float(np.sum(np.abs(coeffs) < np.percentile(np.abs(coeffs), 50)))
            if sparsity < best_sparsity:
                best_sparsity = sparsity
                best_alpha = a
                best_coeffs = coeffs
        thr = np.percentile(np.abs(best_coeffs), 75)
        keep = np.abs(best_coeffs) > thr
        idx = np.argwhere(keep).ravel().astype(np.int32)
        vals = best_coeffs[keep]
        meta = dict(shape=tensor.shape, n=n, alpha=best_alpha)
        data = struct.pack("!d", best_alpha) + _ser(idx) + vals.tobytes()
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        n = metadata["n"]
        alpha = struct.unpack("!d", data[:8])[0]
        # idx: int32 (4 bytes), val: complex128 (16 bytes)
        n_idx = (len(data) - 8) // 20
        idx = _deser(data[8 : 8 + n_idx * 4], np.int32).ravel()
        vals = np.frombuffer(data[8 + n_idx * 4 :], dtype=np.complex128, count=n_idx)
        coeffs = np.zeros(n, dtype=np.complex128)
        for i, v in zip(idx, vals):
            if i < n:
                coeffs[i] = v
        recon = self._frft(coeffs, -alpha).real
        return recon.reshape(shape).astype(np.float32)
