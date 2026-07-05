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

class MUSICAlgorithm:
    """G13: MUSIC pseudospectrum — subspace decomposition."""

    name = "music_algorithm"
    category = "novel_signal"

    def compress(self, tensor: np.ndarray, n_sources: int = 4) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        if t.ndim < 2:
            t = t.reshape(-1, 1)
        m, n = t.shape
        k = min(n_sources, min(m, n) - 1)
        C = t.T @ t / m
        eigvals, eigvecs = np.linalg.eigh(C)
        idx = np.argsort(-eigvals)
        Us = eigvecs[:, idx[:k]]
        Un = eigvecs[:, idx[k:]]
        meta = dict(shape=tensor.shape, m=m, n=n, k=k)
        data = _ser(Us.astype(np.float16)) + _ser(Un.astype(np.float16))
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        m, n = metadata["m"], metadata["n"]
        k = metadata["k"]
        Us_bytes = n * k * 2
        Us = _deser(data[:Us_bytes], np.float16).reshape(n, k)
        n_un = n - k
        Un = _deser(data[Us_bytes:], np.float16).reshape(n, n_un) if n_un > 0 else None
        Pm = Us @ Us.T
        noise = np.random.RandomState(42).randn(m, n)
        t_hat = noise @ Pm.T
        return t_hat.reshape(shape).astype(np.float32)
