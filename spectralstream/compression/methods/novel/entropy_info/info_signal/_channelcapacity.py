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

class ChannelCapacity:
    """F14: C = max I(X;Y), Blahut-Arimoto algorithm for rate allocation."""

    name = "channel_capacity"
    category = "novel_info"

    def compress(
        self, tensor: np.ndarray, n_symbols: int = 16, n_iter: int = 100
    ) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        flat = t.ravel()
        hist, _ = np.histogram(flat, bins=n_symbols, density=True)
        p_x = hist / (hist.sum() + 1e-30)
        p_y_given_x = np.random.uniform(0.5, 1.5, (n_symbols, n_symbols))
        p_y_given_x /= p_y_given_x.sum(axis=1, keepdims=True)
        p_y = np.ones(n_symbols) / n_symbols
        for _ in range(n_iter):
            p_xy = p_x[:, None] * p_y_given_x
            p_y = p_xy.sum(axis=0)
            p_y /= p_y.sum() + 1e-30
            p_x_given_y = p_xy / (p_y[None, :] + 1e-30)
            mi = np.sum(p_xy * np.log(p_y_given_x / (p_y[None, :] + 1e-30) + 1e-30))
        capacity = max(0.0, float(mi))
        bits = max(1, min(16, int(np.ceil(capacity / math.log(2)))))
        q, scale, lo = _quantize(t, bits)
        meta = dict(
            shape=tensor.shape, scale=scale, lo=lo, bits=bits, capacity=capacity
        )
        data = _ser(q)
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        scale = metadata["scale"]
        lo = metadata["lo"]
        n = int(np.prod(shape))
        q = _deser(data[:n], np.uint8)
        return _dequantize(q, scale, lo).reshape(shape)
