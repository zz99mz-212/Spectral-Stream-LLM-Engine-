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

class MatrixPencil:
    """G12: Hankel SVD decomposition — H = UΣV^T, system order."""

    name = "matrix_pencil"
    category = "novel_signal"

    def compress(self, tensor: np.ndarray) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        flat = t.ravel()
        n = len(flat)
        # Build Hankel matrix
        L = n // 2
        H = np.zeros((L, n - L + 1))
        for i in range(L):
            H[i] = flat[i : i + n - L + 1]
        U, S, Vt = np.linalg.svd(H, full_matrices=False)
        cum = np.cumsum(S) / np.sum(S)
        r = int(np.searchsorted(cum, 0.95)) + 1
        S_r = S[:r].astype(np.float16)
        Vt_r = Vt[:r, :].astype(np.float16)
        meta = dict(shape=tensor.shape, n=n, L=L, r=r)
        data = _ser(S_r) + _ser(Vt_r)
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        n = metadata["n"]
        L = metadata["L"]
        r = metadata["r"]
        S_r = _deser(data[: r * 2], np.float16)
        Vt_r = _deser(data[r * 2 :], np.float16).reshape(r, n - L + 1)
        U = np.random.RandomState(42).randn(L, r).astype(np.float64)
        U /= np.linalg.norm(U, axis=0, keepdims=True) + 1e-30
        Hr = (U * S_r) @ Vt_r
        # average anti-diagonals
        recon = np.zeros(n, dtype=np.float32)
        counts = np.zeros(n, dtype=np.int32)
        for i in range(L):
            for j in range(n - L + 1):
                idx = i + j
                recon[idx] += float(Hr[i, j])
                counts[idx] += 1
        recon = recon / np.maximum(counts, 1)
        return recon.reshape(shape)
