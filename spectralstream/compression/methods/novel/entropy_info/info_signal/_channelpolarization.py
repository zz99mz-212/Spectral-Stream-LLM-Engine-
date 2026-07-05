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

class ChannelPolarization:
    """F4: Polar transform G^⊗n, information concentrates in polarized channels."""

    name = "channel_polarization"
    category = "novel_info"

    def compress(
        self, tensor: np.ndarray, threshold: float = 0.1
    ) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        flat = t.ravel()
        n = len(flat)
        m = 1
        while m < n:
            m *= 2
        padded = np.pad(flat, (0, m - n))
        # Kronecker power polar transform
        x = padded.copy()
        d = 0
        while (1 << d) < m:
            d += 1
        for i in range(d):
            step = 1 << i
            for j in range(0, m, 2 * step):
                a = x[j : j + step].copy()
                b = x[j + step : j + 2 * step].copy()
                x[j : j + step] = a + b
                x[j + step : j + 2 * step] = a - b
        energy = np.abs(x)
        thr = np.percentile(energy, (1 - threshold) * 100)
        keep = energy >= thr
        idx = np.argwhere(keep).ravel().astype(np.int32)
        vals = x[keep].astype(np.float16)
        meta = dict(shape=tensor.shape, orig_n=n, m=m)
        data = _ser(idx) + vals.tobytes()
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        orig_n = metadata["orig_n"]
        m = metadata["m"]
        n_idx = (len(data) - 4) // 6
        idx = _deser(data[: n_idx * 4], np.int32).ravel()
        vals = np.frombuffer(data[n_idx * 4 :], dtype=np.float16)
        x = np.zeros(m, dtype=np.float64)
        for i, v in zip(idx, vals):
            if i < m:
                x[i] = float(v)
        d = 0
        while (1 << d) < m:
            d += 1
        for i in range(d - 1, -1, -1):
            step = 1 << i
            for j in range(0, m, 2 * step):
                a = x[j : j + step].copy()
                b = x[j + step : j + 2 * step].copy()
                x[j : j + step] = (a + b) / 2
                x[j + step : j + 2 * step] = (a - b) / 2
        return x[:orig_n].reshape(shape).astype(np.float32)
