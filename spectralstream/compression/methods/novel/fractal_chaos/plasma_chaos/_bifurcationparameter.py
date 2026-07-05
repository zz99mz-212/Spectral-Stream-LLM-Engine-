from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class BifurcationParameter:
    """Bifurcation diagram: w_{n+1} = f(w_n, μ), fixed points as f(μ)."""

    name = "bifurcation_parameter"
    category = "novel_chaos"

    def compress(self, tensor: np.ndarray, n_mu: int = 16) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        flat = t.ravel()
        n = len(flat)

        mu_vals = np.linspace(0.5, 4.0, n_mu)
        fixed_pts = []
        for mu in mu_vals:
            x = 0.5
            for _ in range(500):
                x = mu * x * (1 - x)
            pts = []
            for _ in range(100):
                x = mu * x * (1 - x)
                pts.append(x)
            fixed_pts.append(np.mean(pts[-20:]))
        fixed_pts = np.array(fixed_pts)

        n_pts = min(n, n_mu)
        idx = np.linspace(0, n - 1, n_pts).astype(int)
        vals = flat[idx]
        residual = flat - np.interp(np.arange(n), idx, vals)
        thr = np.percentile(np.abs(residual), 92)
        rmask = np.abs(residual) > thr
        ridx = np.argwhere(rmask)[:, 0]
        rvals = residual[rmask]

        meta = dict(shape=tensor.shape, n_mu=n_mu, n_res=len(ridx))
        data = _serialize(mu_vals.astype(np.float32))
        data += _serialize(fixed_pts.astype(np.float32))
        data += _serialize(idx.astype(np.int32)) + vals.astype(np.float16).tobytes()
        if len(ridx) > 0:
            data += (
                _serialize(ridx.astype(np.int32)) + rvals.astype(np.float16).tobytes()
            )
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        n_mu = metadata["n_mu"]
        n_res = metadata.get("n_res", 0)
        n = int(np.prod(shape))

        pos = 0
        mu_vals = _deserialize(data[: n_mu * 4])
        pos += n_mu * 4
        fixed_pts = _deserialize(data[pos : pos + n_mu * 4])
        pos += n_mu * 4

        idx = _deserialize(data[pos : pos + n_mu * 4]).astype(int)
        pos += n_mu * 4
        vals = np.frombuffer(data[pos : pos + n_mu * 2], dtype=np.float16).astype(
            np.float64
        )
        pos += n_mu * 2

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
