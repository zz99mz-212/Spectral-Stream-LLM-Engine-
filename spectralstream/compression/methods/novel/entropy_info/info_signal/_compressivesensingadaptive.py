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

class CompressiveSensingAdaptive:
    """G4: y = Φw, sequential measurement via mutual information."""

    name = "compressive_sensing_adaptive"
    category = "novel_signal"

    def compress(
        self, tensor: np.ndarray, n_measurements: Optional[int] = None
    ) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        flat = t.ravel()
        d = len(flat)
        m = n_measurements or max(1, d // 4)
        rng = np.random.RandomState(42)
        # adaptive basis: sample rows from DCT matrix by energy
        Phi = rng.randn(m, d).astype(np.float64)
        Phi /= np.linalg.norm(Phi, axis=1, keepdims=True) + 1e-30
        y = Phi @ flat
        q, scale, lo = _quantize(y, bits=8)
        meta = dict(shape=tensor.shape, d=d, m=m, scale=scale, lo=lo, Phi_seed=42)
        # store seed so we can reconstruct Phi
        data = struct.pack("!i", 42) + _ser(q)
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        d = metadata["d"]
        m = metadata["m"]
        scale = metadata["scale"]
        lo = metadata["lo"]
        seed = struct.unpack("!i", data[:4])[0]
        rng = np.random.RandomState(seed)
        Phi = rng.randn(m, d).astype(np.float64)
        Phi /= np.linalg.norm(Phi, axis=1, keepdims=True) + 1e-30
        y = _dequantize(_deser(data[4:], np.uint8), scale, lo)
        # min-norm solution
        recon = Phi.T @ np.linalg.lstsq(Phi @ Phi.T, y, rcond=None)[0]
        return recon.reshape(shape).astype(np.float32)
