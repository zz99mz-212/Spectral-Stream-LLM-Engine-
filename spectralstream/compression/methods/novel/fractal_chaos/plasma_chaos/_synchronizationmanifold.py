from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class SynchronizationManifold:
    """Kuramoto model: dθ_i/dt = ω_i + K/N Σ sin(θ_j - θ_i)."""

    name = "synchronization_manifold"
    category = "novel_chaos"

    def compress(
        self, tensor: np.ndarray, K_coupling: float = 1.0
    ) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        if t.ndim < 2:
            t = t.reshape(1, -1)
        m, n = t.shape

        U, S, Vt = np.linalg.svd(t, full_matrices=False)
        k = min(8, len(S))

        omega = S[:k]
        theta0 = np.arctan2(U[:, 0], U[:, -1])[:k] if k > 0 else np.zeros(k)

        phase_diff = omega[:, None] - omega[None, :]
        sin_terms = K_coupling * np.sin(theta0[:, None] - theta0[None, :])
        dtheta = phase_diff + sin_terms
        sync_order = float(np.mean(np.abs(np.mean(np.exp(1j * dtheta), axis=0))))

        meta = dict(shape=tensor.shape, k=k, K=K_coupling, sync_order=sync_order)
        data = _serialize(omega.astype(np.float32))
        data += _serialize(theta0.astype(np.float32))
        data += _serialize(U[:, :k].astype(np.float32))
        data += _serialize(S[:k].astype(np.float32))
        data += _serialize(Vt[:k, :].astype(np.float32))
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        k = metadata["k"]

        pos = 0
        omega = _deserialize(data[: k * 4])
        pos += k * 4
        theta0 = _deserialize(data[pos : pos + k * 4])
        pos += k * 4
        m = shape[0]
        U = _deserialize(data[pos : pos + m * k * 4]).reshape(m, k)
        pos += m * k * 4
        S = _deserialize(data[pos : pos + k * 4])
        pos += k * 4
        Vt = _deserialize(data[pos : pos + k * shape[-1] * 4]).reshape(k, shape[-1])

        return ((U * S) @ Vt).reshape(shape).astype(np.float32)
