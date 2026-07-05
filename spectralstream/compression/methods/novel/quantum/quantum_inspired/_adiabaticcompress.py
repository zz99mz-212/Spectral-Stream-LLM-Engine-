from __future__ import annotations

import math
import struct
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class AdiabaticCompress:
    """Adiabatic theorem: H(t) = (1-t/T)H_0 + (t/T)H_1.
    Store ground state evolution via slow parameter variation.
    """

    name = "adiabatic_compress"
    category = "quantum_compression"

    def compress(
        self,
        tensor: np.ndarray,
        n_time_steps: int = 20,
        subspace_dim: int = 8,
    ) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        orig_shape = t.shape
        if t.ndim > 2:
            t = t.reshape(t.shape[0], -1)
        m, n = t.shape
        d = min(subspace_dim, m, n)
        U0, S0_full, Vt0 = np.linalg.svd(t, full_matrices=False)
        ground = U0[:, :d] @ np.diag(S0_full[:d]) @ Vt0[:d, :]
        trajectory = []
        for step in range(n_time_steps):
            s = step / max(n_time_steps - 1, 1)
            interpolated = (1 - s) * t + s * ground
            projection = U0[:, :d].T @ interpolated @ Vt0[:d, :].T
            trajectory.append(projection.astype(np.float32))
        U0_f32 = U0[:, :d].astype(np.float32)
        Vt0_f32 = Vt0[:d, :].astype(np.float32)
        meta = dict(
            shape=orig_shape,
            m=m,
            n=n,
            d=d,
            n_time_steps=n_time_steps,
        )
        data = struct.pack("<III", m, n, d)
        data += struct.pack("<I", n_time_steps)
        data += _serialize(U0_f32)
        data += _serialize(Vt0_f32)
        for proj in trajectory:
            data += _serialize(proj)
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        m, n, d = struct.unpack_from("<III", data, 0)
        pos = 12
        n_steps = struct.unpack_from("<I", data, pos)[0]
        pos += 4
        U0 = _deserialize(data[pos : pos + m * d * 4]).reshape(m, d)
        pos += m * d * 4
        Vt0 = _deserialize(data[pos : pos + d * n * 4]).reshape(d, n)
        pos += d * n * 4
        traj = []
        for _ in range(n_steps):
            traj.append(_deserialize(data[pos : pos + d * d * 4]).reshape(d, d))
            pos += d * d * 4
        recon_compressed = U0 @ traj[-1].astype(np.float64) @ Vt0
        return (
            recon_compressed[: shape[0], : shape[1]].reshape(shape).astype(np.float32)
        )
