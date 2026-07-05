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

class EmpiricalModeDecomp:
    """G16: Huang-Hilbert sifting — w(t) = Σ IMF_k(t) + r(t)."""

    name = "empirical_mode_decomp"
    category = "novel_signal"

    def _sift(self, sig: np.ndarray, max_imfs: int = 5) -> List[np.ndarray]:
        imfs: List[np.ndarray] = []
        r = sig.copy()
        for _ in range(max_imfs):
            h = r.copy()
            for _ in range(10):
                env_upper = np.maximum.accumulate(h)
                env_lower = -np.maximum.accumulate(-h)
                mean = (env_upper + env_lower) / 2
                h = h - mean
            imfs.append(h.copy())
            r = r - h
        return imfs

    def compress(self, tensor: np.ndarray, n_imfs: int = 3) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        flat = t.ravel()
        imfs = self._sift(flat, n_imfs)
        residual = flat - sum(imfs)
        imf_data = np.concatenate(imfs).astype(np.float16)
        q, scale, lo = _quantize(residual, bits=6)
        meta = dict(shape=tensor.shape, n=len(flat), n_imfs=n_imfs, scale=scale, lo=lo)
        data = _ser(imf_data) + _ser(q)
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        n = metadata["n"]
        ni = metadata["n_imfs"]
        scale = metadata["scale"]
        lo = metadata["lo"]
        imf_bytes = n * ni * 2
        imf_data = _deser(data[:imf_bytes], np.float16).reshape(ni, n)
        q = _deser(data[imf_bytes:], np.uint8)
        residual = _dequantize(q, scale, lo)
        recon = imf_data.sum(axis=0) + residual[:n]
        return recon.reshape(shape).astype(np.float32)
