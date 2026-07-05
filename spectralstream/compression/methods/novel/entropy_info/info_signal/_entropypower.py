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

class EntropyPower:
    """F11: d/dt h(X+√t Z) = ½J(X+√t Z), Fisher info rate estimation."""

    name = "entropy_power"
    category = "novel_info"

    def compress(self, tensor: np.ndarray, noise_levels: int = 8) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        flat = t.ravel()
        h_est = 0.0
        for sigma in np.logspace(-3, 0, noise_levels):
            noisy = flat + np.random.RandomState(42).randn(len(flat)) * sigma
            std = float(noisy.std())
            h_est += 0.5 * math.log(2 * math.pi * math.e * std**2 + 1e-30)
        h_est /= noise_levels
        bits = max(2, int(np.ceil(h_est / math.log(2))))
        bits = min(bits, 16)
        q, scale, lo = _quantize(t, bits)
        meta = dict(shape=tensor.shape, scale=scale, lo=lo, bits=bits, h_est=h_est)
        data = _ser(q)
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        scale = metadata["scale"]
        lo = metadata["lo"]
        n = int(np.prod(shape))
        q = _deser(data[:n], np.uint8)
        return _dequantize(q, scale, lo).reshape(shape)
