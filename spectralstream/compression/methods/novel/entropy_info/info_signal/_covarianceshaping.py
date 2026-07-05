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

class CovarianceShaping:
    """G11: Ŵ = C^{1/2}Z where Z is white, Cholesky factor storage."""

    name = "covariance_shaping"
    category = "novel_signal"

    def compress(self, tensor: np.ndarray, block_size: int = 32) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        if t.ndim < 2:
            t = t.reshape(-1, 1)
        m, n = t.shape
        bs = min(block_size, n)
        # estimate covariance from a block
        C = np.cov(t[:, :bs].T) + np.eye(bs) * 1e-6
        L = np.linalg.cholesky(C)
        Z = np.random.RandomState(42).randn(m, bs)
        recon_block = Z @ L.T
        residual = t[:, :bs] - recon_block
        q, scale, lo = _quantize(residual, bits=6)
        meta = dict(shape=tensor.shape, m=m, n=n, bs=bs, scale=scale, lo=lo)
        data = _ser(L.astype(np.float16)) + _ser(q)
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        m, n = metadata["m"], metadata["n"]
        bs = metadata["bs"]
        scale = metadata["scale"]
        lo = metadata["lo"]
        L_bytes = bs * bs * 2
        L = _deser(data[:L_bytes], np.float16).reshape(bs, bs)
        q = _deser(data[L_bytes:], np.uint8)
        residual = _dequantize(q, scale, lo).reshape(m, bs)
        Z = np.random.RandomState(42).randn(m, bs)
        recon = Z @ L.T + residual
        # extend to full size
        full = np.zeros((m, n), dtype=np.float32)
        full[:, :bs] = recon
        for j in range(bs, n):
            full[:, j] = recon[:, j % bs]
        return full.reshape(shape).astype(np.float32)
