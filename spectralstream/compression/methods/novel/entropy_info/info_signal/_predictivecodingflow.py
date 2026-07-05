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

class PredictiveCodingFlow:
    """F3: H(W) = Σ H(w_i | w_{<i}), autoregressive delta encoding."""

    name = "predictive_coding_flow"
    category = "novel_info"

    def compress(self, tensor: np.ndarray, axis: int = -1) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        if axis != -1:
            t = np.moveaxis(t, axis, -1)
        pred = np.zeros_like(t)
        pred[..., 1:] = t[..., :-1]
        delta = t - pred
        q, scale, lo = _quantize(delta, bits=8)
        meta = dict(
            shape=tensor.shape,
            scale=scale,
            lo=lo,
            first=t[..., 0].astype(np.float16).tobytes(),
        )
        data = _ser(q)
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        scale = metadata["scale"]
        lo = metadata["lo"]
        n = int(np.prod(shape))
        q = _deser(data[:n], np.uint8).reshape(shape)
        delta = _dequantize(q, scale, lo)
        first = _deser(metadata["first"], np.float16)
        delta[..., 0] = first
        recon = np.cumsum(delta, axis=-1)
        return recon.astype(np.float32)
