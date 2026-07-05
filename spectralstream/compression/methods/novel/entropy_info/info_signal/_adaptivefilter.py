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

class AdaptiveFilter:
    """G8: LMS adaptive filter coefficients — w_{n+1} = w_n + μ e_n x_n."""

    name = "adaptive_filter"
    category = "novel_signal"

    def compress(
        self, tensor: np.ndarray, mu: float = 0.01, n_iter: int = 100
    ) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        flat = t.ravel()
        n = len(flat)
        w = np.zeros(n)
        x = np.random.RandomState(42).randn(n)
        errors: List[float] = []
        for _ in range(n_iter):
            e = flat - w * x
            w = w + mu * e * x
            errors.append(float(np.mean(e**2)))
        q, scale, lo = _quantize(w.astype(np.float64), bits=8)
        meta = dict(shape=tensor.shape, n=n, mu=mu, n_iter=n_iter, scale=scale, lo=lo)
        data = _ser(q) + np.array(errors, dtype=np.float16).tobytes()
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        n = metadata["n"]
        scale = metadata["scale"]
        lo = metadata["lo"]
        q = _deser(data[:n], np.uint8)
        w = _dequantize(q, scale, lo)
        x = np.random.RandomState(42).randn(n)
        return (w * x).reshape(shape).astype(np.float32)
