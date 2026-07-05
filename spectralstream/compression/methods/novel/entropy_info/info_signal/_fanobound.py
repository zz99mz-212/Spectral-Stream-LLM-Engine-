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

class FanoBound:
    """F8: H(e) + P(e)log(|W|-1) >= H(W|Ŵ), error-exponent guided quantization."""

    name = "fano_bound"
    category = "novel_info"

    def compress(
        self, tensor: np.ndarray, target_pe: float = 0.01
    ) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        n_classes = int(np.ceil(np.sqrt(t.size)))
        n_classes = max(2, min(n_classes, 256))
        bits_per = int(np.ceil(np.log2(n_classes)))
        q, scale, lo = _quantize(t, bits=bits_per)
        he = -target_pe * math.log2(target_pe + 1e-30) - (1 - target_pe) * math.log2(
            1 - target_pe + 1e-30
        )
        meta = dict(
            shape=tensor.shape,
            scale=scale,
            lo=lo,
            bits=bits_per,
            he=he,
            target_pe=target_pe,
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
