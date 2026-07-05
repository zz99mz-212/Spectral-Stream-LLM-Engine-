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

class VariationalModeDecomp:
    """G20: VMD — variational mode decomposition, simplified ADMM."""

    name = "variational_mode_decomp"
    category = "novel_signal"

    def compress(
        self, tensor: np.ndarray, n_modes: int = 4, alpha: float = 2000.0
    ) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        flat = t.ravel()
        n = len(flat)
        # Simplified VMD: initialize modes as bandpassed versions
        freqs = np.fft.fftfreq(n)
        fhat = np.fft.fft(flat)
        modes: List[np.ndarray] = []
        centers = np.linspace(0, 0.5, n_modes + 2)[1:-1]
        for c in centers:
            filt = 1.0 / (1.0 + alpha * (freqs - c) ** 2)
            mode = np.fft.ifft(fhat * filt).real
            modes.append(mode)
        modes_arr = np.array(modes).astype(np.float16)
        residual = flat - np.sum(modes, axis=0)
        q, scale, lo = _quantize(residual, bits=6)
        meta = dict(
            shape=tensor.shape, n=n, n_modes=n_modes, alpha=alpha, scale=scale, lo=lo
        )
        data = _ser(modes_arr) + _ser(q)
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        n = metadata["n"]
        nm = metadata["n_modes"]
        scale = metadata["scale"]
        lo = metadata["lo"]
        modes_bytes = nm * n * 2
        modes = _deser(data[:modes_bytes], np.float16).reshape(nm, n)
        q = _deser(data[modes_bytes:], np.uint8)
        residual = _dequantize(q, scale, lo)
        recon = modes.sum(axis=0) + residual[:n]
        return recon.reshape(shape).astype(np.float32)
