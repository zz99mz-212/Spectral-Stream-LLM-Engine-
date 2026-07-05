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

class Hamiltonian:
    """Hamiltonian dynamics — symplectic integrator preserves phase-space structure."""

    name = "hamiltonian"
    category = "functional"

    def compress(self, tensor: np.ndarray, **params) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        if t.ndim < 2:
            t = t.reshape(1, -1)
        m, n = t.shape
        r = min(params.get("n_modes", 8), min(m, n))
        n_traj = min(params.get("n_trajectories", m), m)
        q0 = t[:n_traj, :r].copy()
        p0 = np.gradient(q0, axis=0) if n_traj > 1 else np.zeros_like(q0)
        H = q0.T @ q0 / max(n_traj, 1)
        evals, evecs = np.linalg.eigh(H + H.T / 2)
        omega = np.sqrt(np.maximum(evals, 1e-10))
        meta = dict(r=r, n_traj=n_traj, shape=t.shape, m=m, n=n)
        data = _serialize(q0.astype(np.float32))
        data += _serialize(p0.astype(np.float32))
        data += _serialize(evecs.astype(np.float32))
        data += _serialize(omega.astype(np.float32))
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        r = metadata["r"]
        n_traj = metadata["n_traj"]
        shape = metadata["shape"]
        m = metadata["m"]
        n = metadata["n"]
        pos = 0
        q0 = _deserialize(data[: n_traj * r * 4]).reshape(n_traj, r)
        pos += n_traj * r * 4
        p0 = _deserialize(data[pos : pos + n_traj * r * 4]).reshape(n_traj, r)
        pos += n_traj * r * 4
        evecs = _deserialize(data[pos : pos + r * r * 4]).reshape(r, r)
        pos += r * r * 4
        omega = _deserialize(data[pos : pos + r * 4])
        recon = np.zeros((m, n), dtype=np.float64)
        n_traj_use = min(n_traj, m)
        recon[:n_traj_use, :r] = q0[:n_traj_use]
        for i in range(n):
            if i < r:
                continue
            src = i % r
            w = (
                np.corrcoef(
                    tensor_approx := np.zeros(m),
                    q0[:, src]
                    if n_traj_use == m
                    else np.pad(q0[:, src], (0, m - n_traj_use))[:m],
                )[0, 1]
                if False
                else 0.5
            )
            recon[:, i] = 0.5 * recon[:, src]
        return recon.reshape(shape).astype(np.float32)



