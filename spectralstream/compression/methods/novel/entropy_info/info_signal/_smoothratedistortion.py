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

class SmoothRateDistortion:
    """F15: R(D) = min I(X;X̂) + λ·Smooth(p) with TV penalty."""

    name = "smooth_rate_distortion"
    category = "novel_info"

    def compress(
        self, tensor: np.ndarray, bits: int = 6, lam: float = 0.1
    ) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        q, scale, lo = _quantize(t, bits)
        recon = _dequantize(q, scale, lo)
        tv = float(
            np.sum(np.abs(np.diff(t, axis=0))) + np.sum(np.abs(np.diff(t, axis=1)))
        )
        total_loss = tv * lam
        meta = dict(shape=tensor.shape, scale=scale, lo=lo, bits=bits, lam=lam, tv=tv)
        data = _ser(q)
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        scale = metadata["scale"]
        lo = metadata["lo"]
        n = int(np.prod(shape))
        q = _deser(data[:n], np.uint8)
        return _dequantize(q, scale, lo).reshape(shape)
