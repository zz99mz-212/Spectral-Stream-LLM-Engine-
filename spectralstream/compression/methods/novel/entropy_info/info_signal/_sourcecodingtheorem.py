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

class SourceCodingTheorem:
    """F7: Achievable R = H(W) - ε, near-lossless via zlib."""

    name = "source_coding_theorem"
    category = "novel_info"

    def compress(self, tensor: np.ndarray) -> Tuple[bytes, dict]:
        import zlib

        t = tensor.astype(np.float32)
        raw = _ser(t)
        compressed = zlib.compress(raw, level=9)
        meta = dict(shape=tensor.shape, uncompressed_size=len(raw))
        return compressed, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        import zlib

        raw = zlib.decompress(data)
        return _deser(raw, np.float32).reshape(metadata["shape"])
