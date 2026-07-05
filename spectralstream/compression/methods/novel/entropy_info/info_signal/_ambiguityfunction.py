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

class AmbiguityFunction:
    """G18: A(τ,f_d) = ∫w(t)w*(t-τ)exp(-i2πf_d t)dt, 2D ambiguity surface."""

    name = "ambiguity_function"
    category = "novel_signal"

    def compress(
        self, tensor: np.ndarray, n_lags: int = 16, n_dopplers: int = 16
    ) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        flat = t.ravel()
        n = len(flat)
        lags = np.linspace(0, n - 1, n_lags).astype(int)
        dopplers = np.linspace(-0.5, 0.5, n_dopplers)
        surface = np.zeros((n_lags, n_dopplers), dtype=np.complex128)
        for i, tau in enumerate(lags):
            for j, fd in enumerate(dopplers):
                idx = np.arange(max(0, tau), min(n, n))
                w1 = flat[idx]
                w2 = flat[idx - tau] if tau > 0 else flat[idx]
                min_len = min(len(w1), len(w2))
                if min_len > 0:
                    surf = np.sum(
                        w1[:min_len]
                        * np.conj(w2[:min_len])
                        * np.exp(-2j * math.pi * fd * np.arange(min_len))
                    )
                    surface[i, j] = surf
        peak_mask = np.abs(surface) > np.percentile(np.abs(surface), 80)
        idx_keep = np.argwhere(peak_mask)
        vals_keep = surface[peak_mask]
        meta = dict(shape=tensor.shape, n=n, n_lags=n_lags, n_dopplers=n_dopplers)
        # store real/imag as float32 pairs
        real_imag = np.stack(
            [vals_keep.real.astype(np.float32), vals_keep.imag.astype(np.float32)],
            axis=-1,
        )
        data = _ser(idx_keep.astype(np.int16)) + real_imag.tobytes()
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        n = metadata["n"]
        n_lags = metadata["n_lags"]
        n_dopplers = metadata["n_dopplers"]
        n_pts = (len(data) - 4) // 8
        idx = _deser(data[: n_pts * 4], np.int16).reshape(-1, 2)
        ri = np.frombuffer(data[n_pts * 4 :], dtype=np.float32).reshape(-1, 2)
        vals = ri[:, 0] + 1j * ri[:, 1]
        surface = np.zeros((n_lags, n_dopplers), dtype=np.complex64)
        for (i, j), v in zip(idx, vals):
            if i < n_lags and j < n_dopplers:
                surface[i, j] = v
        recon = np.zeros(n, dtype=np.float32)
        for i in range(min(n_lags, n)):
            row_energy = np.sum(np.abs(surface[i]))
            recon[i] = row_energy
        return recon.reshape(shape).astype(np.float32)
