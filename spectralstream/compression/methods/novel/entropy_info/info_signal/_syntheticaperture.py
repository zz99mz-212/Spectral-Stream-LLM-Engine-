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

class SyntheticAperture:
    """G6: Range-Doppler — w(x) = ∫A(ξ)exp(-i4πR(ξ)/λ)dξ."""

    name = "synthetic_aperture"
    category = "novel_signal"

    def compress(
        self, tensor: np.ndarray, n_range_bins: int = 16
    ) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        flat = t.ravel()
        n = len(flat)
        # Range-Doppler: DFT across segments
        segs = np.array_split(flat, n_range_bins)
        doppler = np.zeros(n_range_bins, dtype=np.complex128)
        for i, seg in enumerate(segs):
            doppler[i] = np.fft.fft(seg)[0]
        A = np.abs(doppler).astype(np.float16)
        phi = np.angle(doppler).astype(np.float16)
        meta = dict(shape=tensor.shape, n=n, n_range_bins=n_range_bins)
        data = _ser(A) + _ser(phi)
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        n = metadata["n"]
        nrb = metadata["n_range_bins"]
        A = _deser(data[: nrb * 2], np.float16)
        phi = _deser(data[nrb * 2 :], np.float16)
        doppler = A * np.exp(1j * phi)
        seg_len = int(np.ceil(n / nrb))
        recon = np.zeros(n, dtype=np.float32)
        for i in range(nrb):
            start = i * seg_len
            end = min(start + seg_len, n)
            recon[start:end] = float(doppler[i].real)
        return recon.reshape(shape)
