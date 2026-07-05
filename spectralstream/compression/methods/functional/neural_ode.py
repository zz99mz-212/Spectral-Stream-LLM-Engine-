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

class NeuralODE:
    """Neural ODE — treat rows as ODE trajectory, fit initial condition + dynamics."""

    name = "neural_ode"
    category = "functional"

    def compress(self, tensor: np.ndarray, **params) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        if t.ndim < 2:
            t = t.reshape(1, -1)
        m, n = t.shape
        dt = 1.0 / max(m, 1)
        n_modes = min(params.get("n_modes", 16), n)
        u0 = t[0, :n_modes].copy()
        K = np.zeros((n_modes, n_modes), dtype=np.float64)
        for i in range(1, min(m, 64)):
            if i < len(t) and i - 1 < m:
                grad = (t[i, :n_modes] - t[i - 1, :n_modes]) / dt
                K += np.outer(grad, t[i - 1, :n_modes])
        U_k, S_k, Vt_k = np.linalg.svd(K, full_matrices=False)
        r = min(8, len(S_k))
        meta = dict(n_modes=n_modes, r=r, shape=t.shape, m=m, n=n, dt=dt)
        data = _serialize(u0.astype(np.float32))
        data += _serialize(U_k[:, :r].astype(np.float32))
        data += _serialize(S_k[:r].astype(np.float32))
        data += _serialize(Vt_k[:r, :].astype(np.float32))
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        n_modes = metadata["n_modes"]
        r = metadata["r"]
        shape = metadata["shape"]
        m = metadata["m"]
        n = metadata["n"]
        dt = metadata["dt"]
        pos = 0
        u0 = _deserialize(data[: n_modes * 4])
        pos += n_modes * 4
        Uk = _deserialize(data[pos : pos + n_modes * r * 4]).reshape(n_modes, r)
        pos += n_modes * r * 4
        Sk = _deserialize(data[pos : pos + r * 4])
        pos += r * 4
        Vtk = _deserialize(data[pos : pos + r * n_modes * 4]).reshape(r, n_modes)
        K = (Uk * Sk) @ Vtk
        recon = np.zeros((m, n), dtype=np.float64)
        recon[0, :n_modes] = u0
        for i in range(1, m):
            recon[i, :n_modes] = recon[i - 1, :n_modes] + dt * (
                K @ recon[i - 1, :n_modes]
            )
        return recon.reshape(shape).astype(np.float32)



