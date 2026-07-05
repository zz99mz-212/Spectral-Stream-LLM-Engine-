from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class StrangeAttractor:
    """Lorenz/Rössler attractor parameterization of weight dynamics."""

    name = "strange_attractor"
    category = "novel_chaos"

    def compress(self, tensor: np.ndarray, n_steps: int = 1000) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        flat = t.ravel()
        n = len(flat)

        sigma = float(np.std(flat))
        rho = float(np.mean(flat**2)) / max(sigma**2, 1e-30)
        beta = float(np.mean(np.abs(np.diff(flat[: min(1000, n)])))) / max(sigma, 1e-30)

        rng = np.random.RandomState(42)
        x, y, z = 1.0, 1.0, 1.0
        dt = 0.01
        traj = []
        for _ in range(n_steps):
            dx = sigma * (y - x)
            dy = x * (rho - z) - y
            dz = x * y - beta * z
            x += dx * dt
            y += dy * dt
            z += dz * dt
            traj.append(x)
        traj = np.array(traj)
        traj = (traj - traj.mean()) / max(traj.std(), 1e-30)

        n_pts = min(n, n_steps)
        idx = np.linspace(0, n - 1, n_pts).astype(int)
        vals = flat[idx]
        residual = flat - np.interp(np.arange(n), idx, vals)
        thr = np.percentile(np.abs(residual), 92)
        rmask = np.abs(residual) > thr
        ridx = np.argwhere(rmask)[:, 0]
        rvals = residual[rmask]

        meta = dict(
            shape=tensor.shape,
            sigma=sigma,
            rho=rho,
            beta=beta,
            n_pts=n_pts,
            n_res=len(ridx),
        )
        data = struct.pack("<dddii", sigma, rho, beta, n_pts, len(ridx))
        data += _serialize(idx.astype(np.int32)) + vals.astype(np.float16).tobytes()
        if len(ridx) > 0:
            data += (
                _serialize(ridx.astype(np.int32)) + rvals.astype(np.float16).tobytes()
            )
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        sigma = metadata.get("sigma", 1.0)
        rho = metadata.get("rho", 1.0)
        beta = metadata.get("beta", 1.0)
        n_pts = metadata.get("n_pts", 0)
        n_res = metadata.get("n_res", 0)

        n = int(np.prod(shape))

        pos = 28
        idx = _deserialize(data[pos : pos + n_pts * 4]).astype(int)
        pos += n_pts * 4
        vals = np.frombuffer(data[pos : pos + n_pts * 2], dtype=np.float16).astype(
            np.float64
        )
        pos += n_pts * 2

        recon = np.interp(np.arange(n), idx, vals)

        if n_res > 0:
            ridx = _deserialize(data[pos : pos + n_res * 4]).astype(int)
            pos += n_res * 4
            rvals = np.frombuffer(data[pos : pos + n_res * 2], dtype=np.float16).astype(
                np.float64
            )
            for i, v in zip(ridx, rvals):
                if i < n:
                    recon[i] += v

        return recon.reshape(shape).astype(np.float32)
