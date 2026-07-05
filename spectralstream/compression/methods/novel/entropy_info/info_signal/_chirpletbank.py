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

class ChirpletBank:
    """G3: W(t) = Σ A_k exp(i(α_k t² + β_k t + γ_k)), adaptive chirplet bank."""

    name = "chirplet_bank"
    category = "novel_signal"

    def compress(self, tensor: np.ndarray, n_chirplets: int = 8) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        flat = t.ravel()
        n = len(flat)
        x = np.linspace(-1, 1, n)
        coeffs: List[float] = []
        residual = flat.copy()
        rng = np.random.RandomState(42)
        for _ in range(n_chirplets):
            alpha = rng.uniform(-10, 10)
            beta = rng.uniform(-10, 10)
            gamma = rng.uniform(0, 2 * math.pi)
            chirp = np.exp(1j * (alpha * x**2 + beta * x + gamma))
            A = np.vdot(residual, chirp) / n
            coeffs.extend([A.real, A.imag, alpha, beta, gamma])
            residual -= (A * chirp).real
        ca = np.array(coeffs, dtype=np.float16)
        meta = dict(shape=tensor.shape, n_chirplets=n_chirplets, n=n)
        data = ca.tobytes()
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        nc = metadata["n_chirplets"]
        n = metadata["n"]
        ca = _deser(data, np.float16)
        x = np.linspace(-1, 1, n)
        recon = np.zeros(n, dtype=np.float64)
        for k in range(nc):
            bs = k * 5
            Ar, Ai = float(ca[bs]), float(ca[bs + 1])
            alpha, beta, gamma = float(ca[bs + 2]), float(ca[bs + 3]), float(ca[bs + 4])
            A = Ar + 1j * Ai
            chirp = np.exp(1j * (alpha * x**2 + beta * x + gamma))
            recon += (A * chirp).real
        return recon.reshape(shape).astype(np.float32)
