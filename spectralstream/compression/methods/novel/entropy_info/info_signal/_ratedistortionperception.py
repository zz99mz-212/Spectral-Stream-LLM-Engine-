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

class RateDistortionPerception:
    """F1: R(D,P) = I(X;X̂) with perceptual constraint via histogram matching."""

    name = "rate_distortion_perception"
    category = "novel_info"

    def compress(
        self, tensor: np.ndarray, bits: int = 6, bins: int = 64
    ) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        q, scale, lo = _quantize(t, bits)
        target_hist, _ = np.histogram(t.ravel(), bins=bins, density=True)
        meta = dict(shape=tensor.shape, scale=scale, lo=lo, bits=bits, bins=bins)
        data = _ser(q)
        data += _ser(target_hist.astype(np.float32))
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        scale = metadata["scale"]
        lo = metadata["lo"]
        bits = metadata["bits"]
        bins = metadata["bins"]
        n = int(np.prod(shape))
        nq = n
        q = _deser(data[:nq], np.uint8).reshape(shape)
        th = _deser(data[nq:], np.float32)
        recon = _dequantize(q, scale, lo)
        recon_flat = recon.ravel()
        r_hist, r_edges = np.histogram(recon_flat, bins=bins, density=True)
        r_cdf = np.cumsum(r_hist)
        r_cdf /= r_cdf[-1] + 1e-30
        t_cdf = np.cumsum(th)
        t_cdf /= t_cdf[-1] + 1e-30
        mapped = np.interp(recon_flat, r_edges[:-1] + np.diff(r_edges) / 2, t_cdf)
        return mapped.reshape(shape).astype(np.float32)
