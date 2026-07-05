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

class Beamforming:
    """G7: w(θ) = Σ a_n(θ)s_n, delay-and-sum steering vectors."""

    name = "beamforming"
    category = "novel_signal"

    def compress(self, tensor: np.ndarray, n_angles: int = 16) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        if t.ndim < 2:
            t = t.reshape(1, -1)
        m, n = t.shape
        angles = np.linspace(0, math.pi, n_angles)
        steering = np.exp(1j * np.arange(n)[None, :] * angles[:, None])
        weights = steering @ t.T.astype(np.complex128)
        w_real = weights.real.astype(np.float16)
        w_imag = weights.imag.astype(np.float16)
        meta = dict(shape=tensor.shape, m=m, n=n, n_angles=n_angles)
        data = _ser(w_real) + _ser(w_imag)
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        m, n = metadata["m"], metadata["n"]
        na = metadata["n_angles"]
        ba = na * m * 2
        w_real = _deser(data[:ba], np.float16).reshape(na, m)
        w_imag = _deser(data[ba:], np.float16).reshape(na, m)
        weights = w_real + 1j * w_imag
        angles = np.linspace(0, math.pi, na)
        steering = np.exp(1j * np.arange(n)[None, :] * angles[:, None])
        t_hat = np.linalg.pinv(steering) @ weights
        return t_hat.T.real.reshape(shape).astype(np.float32)
