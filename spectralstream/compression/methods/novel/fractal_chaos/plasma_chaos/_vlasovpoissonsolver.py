from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class VlasovPoissonSolver:
    """Vlasov-Poisson solver: ∂f/∂t + v·∂f/∂x + E·∂f/∂v = 0, coarse grid distribution."""

    name = "vlasov_poisson_solver"
    category = "novel_physics"

    def compress(self, tensor: np.ndarray, grid_size: int = 16) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        orig = t.copy()
        flat = orig.ravel()

        hist, edges = np.histogram(flat, bins=grid_size, density=True)
        centers = (edges[:-1] + edges[1:]) * 0.5

        phi = np.cumsum(hist) / hist.sum()
        E_field = -np.gradient(phi) * grid_size
        f_vel = np.gradient(hist) * grid_size
        vlasov_flux = E_field * f_vel
        vn = (vlasov_flux - vlasov_flux.min()) / (
            vlasov_flux.max() - vlasov_flux.min() + 1e-30
        )

        P = np.zeros((grid_size, grid_size), dtype=np.float64)
        for i in range(grid_size):
            vi = int(np.clip(vn[i] * (grid_size - 1), 0, grid_size - 1))
            P[i, vi] = hist[i] + 1e-10

        thr = np.percentile(P[P > 0], 35)
        mask = P > thr
        idx = np.argwhere(mask)
        vals = P[mask]

        n_pts = len(vals)
        meta = dict(shape=orig.shape, grid_size=grid_size, n_pts=n_pts)
        data = _serialize(centers.astype(np.float32))
        data += struct.pack("<i", n_pts)
        data += _serialize(idx.astype(np.int16)) + vals.astype(np.float16).tobytes()
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        grid_size = metadata["grid_size"]
        n_pts = metadata.get("n_pts", 0)
        centers = _deserialize(data[: grid_size * 4])
        pos = grid_size * 4
        if n_pts == 0:
            n_pts = int(np.frombuffer(data[pos : pos + 4], dtype=np.int32)[0])
        else:
            n_pts = int(np.frombuffer(data[pos : pos + 4], dtype=np.int32)[0])
        pos += 4
        idx = np.frombuffer(data[pos : pos + n_pts * 4], dtype=np.int16).reshape(-1, 2)
        pos += n_pts * 4
        vals = np.frombuffer(data[pos : pos + n_pts * 2], dtype=np.float16).astype(
            np.float64
        )

        Pr = np.zeros((grid_size, grid_size), dtype=np.float64)
        if len(idx) > 0:
            valid = (idx[:, 0] < grid_size) & (idx[:, 1] < grid_size)
            Pr[idx[valid, 0], idx[valid, 1]] = vals[: np.sum(valid)]

        marginal = Pr.sum(axis=1) + 1e-30
        marginal /= marginal.sum()
        cum = np.cumsum(marginal)
        n = int(np.prod(shape))
        q = np.linspace(0, 1, n)
        gi = np.clip(np.searchsorted(cum, q), 0, grid_size - 1)
        return centers[gi].reshape(shape).astype(np.float32)
