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

class HarmonicInversion:
    """G15: Hilbert transform — instantaneous amplitude + phase."""

    name = "harmonic_inversion"
    category = "novel_signal"

    def compress(self, tensor: np.ndarray) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        flat = t.ravel()
        analytic = np.fft.fft(flat)
        n = len(flat)
        # Hilbert via FFT
        h = np.zeros(n, dtype=np.complex128)
        h[0] = analytic[0]
        h[1 : (n + 1) // 2] = 2 * analytic[1 : (n + 1) // 2]
        if n % 2 == 0:
            h[n // 2] = analytic[n // 2]
        hilbert = np.fft.ifft(h)
        amp = np.abs(hilbert).astype(np.float16)
        phase = np.angle(hilbert).astype(np.float16)
        meta = dict(shape=tensor.shape, n=n)
        data = _ser(amp) + _ser(phase)
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        n = metadata["n"]
        amp = _deser(data[: n * 2], np.float16)
        phase = _deser(data[n * 2 :], np.float16)
        return (amp * np.cos(phase)).reshape(shape).astype(np.float32)
