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

class FisherRaoMetric:
    """F13: g_ij = E[∂log p/∂θ_i · ∂log p/∂θ_j], Fréchet mean on manifold."""

    name = "fisher_rao_metric"
    category = "novel_info"

    def compress(self, tensor: np.ndarray, n_bins: int = 32) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        flat = t.ravel()
        hist, edges = np.histogram(flat, bins=n_bins, density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        eps = 1e-30
        grad = np.gradient(np.log(hist + eps))
        fisher = np.mean(grad**2)
        # Fréchet mean: use histogram centroids weighted by density
        frechet = float(np.sum(centers * hist) / np.sum(hist))
        resid = t - frechet
        bits = max(2, int(np.clip(8.0 / (fisher + eps), 1, 16)))
        q, scale, lo = _quantize(resid, bits)
        meta = dict(
            shape=tensor.shape,
            scale=scale,
            lo=lo,
            bits=bits,
            frechet=frechet,
            fisher=fisher,
        )
        data = _ser(q)
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        scale = metadata["scale"]
        lo = metadata["lo"]
        frechet = metadata["frechet"]
        n = int(np.prod(shape))
        q = _deser(data[:n], np.uint8)
        return (_dequantize(q, scale, lo) + frechet).reshape(shape)
