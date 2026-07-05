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

class ShrinkageEstimation:
    """F5: James-Stein ŵ_JS = (1 - (p-2)σ²/||w||²)w."""

    name = "shrinkage_estimation"
    category = "novel_info"

    def compress(
        self, tensor: np.ndarray, sigma2_est: Optional[float] = None
    ) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        p = t.size
        w_norm2 = float(np.sum(t**2))
        if sigma2_est is None:
            sigma2_est = float(np.var(t))
        alpha = max(0.0, 1.0 - (p - 2) * sigma2_est / (w_norm2 + 1e-30))
        shrunk = alpha * t
        residual = t - shrunk
        q, scale, lo = _quantize(residual, bits=6)
        meta = dict(shape=tensor.shape, alpha=alpha, scale=scale, lo=lo)
        data = _ser(q)
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        alpha = metadata["alpha"]
        scale = metadata["scale"]
        lo = metadata["lo"]
        n = int(np.prod(shape))
        q = _deser(data[:n], np.uint8)
        residual = _dequantize(q, scale, lo)
        return residual.reshape(shape).astype(np.float32)
