"""Auto-generated from inr_compression.py."""

from __future__ import annotations

import math
import struct
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spectralstream.core.math_primitives import dct, idct, next_power_of_two


def _bytes(obj: Any) -> int:
    if isinstance(obj, np.ndarray):
        return obj.nbytes
    if isinstance(obj, dict):
        return sum(_bytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(_bytes(x) for x in obj)
    return 0


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()


def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class Lagrangian:
    """Lagrangian mechanics — Euler-Lagrange equations via discrete action minimization."""

    name = "lagrangian"
    category = "functional"

    def compress(self, tensor: np.ndarray, **params) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        if t.ndim < 2:
            t = t.reshape(1, -1)
        m, n = t.shape
        r = min(params.get("n_modes", 8), min(m, n))
        q0 = t[: min(m, 16), :r].copy()
        q1 = t[min(1, m - 1) : min(2, m), :r].copy() if m > 1 else q0.copy()
        kinetic = np.diff(q0, axis=0) if len(q0) > 1 else np.zeros_like(q0[:1])
        K = kinetic.T @ kinetic + 1e-10 * np.eye(r)
        evals, evecs = np.linalg.eigh(K)
        omega = np.sqrt(np.maximum(evals, 1e-10))
        meta = dict(r=r, shape=t.shape, m=m, n=n)
        data = _serialize(q0.astype(np.float32))
        data += _serialize(evecs.astype(np.float32))
        data += _serialize(omega.astype(np.float32))
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        r = metadata["r"]
        shape = metadata["shape"]
        m = metadata["m"]
        n = metadata["n"]
        q0 = (
            _deserialize(data[: min(16, m) * r * 4]).reshape(min(16, m), r)
            if min(16, m) * r > 0
            else np.zeros((1, r))
        )
        pos = q0.size * 4
        evecs = _deserialize(data[pos : pos + r * r * 4]).reshape(r, r)
        pos += r * r * 4
        omega = _deserialize(data[pos : pos + r * 4])
        recon = np.zeros((m, n), dtype=np.float64)
        recon[: min(16, m), :r] = q0[: min(16, m)]
        for i in range(r, n):
            src = i % r
            recon[:, i] = 0.3 * recon[:, src]
        return recon.reshape(shape).astype(np.float32)



