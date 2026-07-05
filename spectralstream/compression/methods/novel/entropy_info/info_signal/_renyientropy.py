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

class RenyiEntropy:
    """F12: H_α(p) = 1/(1-α)log Σ p_i^α, α-tuned per tensor."""

    name = "renyi_entropy"
    category = "novel_info"

    def compress(self, tensor: np.ndarray, alpha: float = 0.5) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        flat = t.ravel()
        p = np.abs(flat) / (np.sum(np.abs(flat)) + 1e-30)
        alpha = max(0.01, min(10.0, alpha))
        if abs(alpha - 1.0) < 1e-6:
            h = -np.sum(p * np.log(p + 1e-30))
        else:
            h = math.log(np.sum(p**alpha) + 1e-30) / (1 - alpha)
        bits = max(1, min(16, int(np.ceil(h / math.log(2)))))
        q, scale, lo = _quantize(t, bits)
        meta = dict(shape=tensor.shape, scale=scale, lo=lo, bits=bits, alpha=alpha)
        data = _ser(q)
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        scale = metadata["scale"]
        lo = metadata["lo"]
        n = int(np.prod(shape))
        q = _deser(data[:n], np.uint8)
        return _dequantize(q, scale, lo).reshape(shape)
