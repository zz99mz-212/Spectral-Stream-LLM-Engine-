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

class MatchingPursuitMulti:
    """G19: W = D·A, simultaneous OMP across rows."""

    name = "matching_pursuit_multi"
    category = "novel_signal"

    def compress(
        self, tensor: np.ndarray, n_atoms: int = 8, n_nonzero: int = 4
    ) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        if t.ndim < 2:
            t = t.reshape(-1, 1)
        m, n = t.shape
        rng = np.random.RandomState(42)
        D = rng.randn(n_atoms, n).astype(np.float64)
        D /= np.linalg.norm(D, axis=1, keepdims=True) + 1e-30
        # Simultaneous OMP
        residual = t.copy()
        active_set: List[int] = []
        A = np.zeros((m, n_atoms), dtype=np.float64)
        for _ in range(n_nonzero):
            proj = residual @ D.T  # (m, n_atoms)
            energies = np.sum(proj**2, axis=0)
            best = np.argmax(energies)
            if best in active_set:
                break
            active_set.append(best)
            A[:, best] = np.mean(proj, axis=1) if m > 1 else proj[:, best]
            residual = t - A[:, active_set] @ D[active_set]
        meta = dict(
            shape=tensor.shape,
            m=m,
            n=n,
            D=_ser(D.astype(np.float16)),
            active=active_set,
        )
        data = _ser(A[:, active_set].astype(np.float16))
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        m, n = metadata["m"], metadata["n"]
        active = metadata["active"]
        na = len(active)
        D = _deser(metadata["D"], np.float16).reshape(-1, n)
        A_sub = _deser(data, np.float16).reshape(m, na)
        recon = A_sub @ D[active]
        return recon.reshape(shape).astype(np.float32)
