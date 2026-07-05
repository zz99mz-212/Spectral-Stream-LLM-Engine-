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

class ESPRITAlgorithm:
    """G14: TLS ESPRIT frequency estimation."""

    name = "esprit_algorithm"
    category = "novel_signal"

    def compress(self, tensor: np.ndarray, n_sources: int = 4) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        flat = t.ravel()
        n = len(flat)
        k = min(n_sources, n // 2)
        L = n // 2
        H = np.zeros((L, n - L + 1))
        for i in range(L):
            H[i] = flat[i : i + n - L + 1]
        U, S, _ = np.linalg.svd(H, full_matrices=False)
        Us = U[:, :k]
        Us1 = Us[:-1, :]
        Us2 = Us[1:, :]
        Psi, _, _, _ = np.linalg.lstsq(Us1, Us2, rcond=None)
        freqs = np.angle(np.linalg.eigvals(Psi))
        amplitudes = S[:k].astype(np.float16)
        meta = dict(shape=tensor.shape, n=n, k=k)
        data = _ser(freqs.astype(np.float16)) + _ser(amplitudes)
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        n = metadata["n"]
        k = metadata["k"]
        freqs = _deser(data[: k * 2], np.float16)
        amps = _deser(data[k * 2 :], np.float16)
        t = np.arange(n)
        recon = np.zeros(n, dtype=np.float64)
        for i in range(k):
            recon += float(amps[i]) * np.cos(float(freqs[i]) * t)
        return recon.reshape(shape).astype(np.float32)
