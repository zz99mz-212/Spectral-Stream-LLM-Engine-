from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class SymplecticReduction:
    """C12. SYMPLECTIC-REDUCTION: Marsden-Weinstein μ⁻¹(0)/G."""

    name = "symplectic_reduction"
    category = "novel_topological"

    def compress(self, tensor: np.ndarray, dim_g: int = 2) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        m, n = t.shape
        dg = min(dim_g, m, n)

        U, S, Vt = np.linalg.svd(t, full_matrices=False)
        r_target = max(1, len(S) - 2 * dg)
        r = min(max(1, r_target), len(S))

        U_r = U[:, :r]
        S_r = S[:r]
        Vt_r = Vt[:r, :]

        theta = np.arctan2(U_r[1:, :], U_r[:-1, :] + 1e-30)
        moment_map = np.mean(theta, axis=0)

        meta = dict(shape=t.shape, r=r, dim_g=dg)
        data = (
            _serialize(U_r.astype(np.float32))
            + _serialize(S_r.astype(np.float32))
            + _serialize(Vt_r.astype(np.float32))
            + _serialize(moment_map.astype(np.float32))
        )
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        r = metadata["r"]
        m, n = shape

        U_r = _deserialize(data[: m * r * 4]).reshape(m, r)
        pos = m * r * 4
        S_r = _deserialize(data[pos : pos + r * 4])
        pos += r * 4
        Vt_r = _deserialize(data[pos : pos + r * n * 4]).reshape(r, n)
        pos += r * n * 4
        moment_map = _deserialize(data[pos : pos + r * 4])

        recon = (U_r * S_r) @ Vt_r

        g_rot = np.eye(r, dtype=np.float64)
        for i in range(min(r, len(moment_map))):
            angle = float(moment_map[i]) * 0.01
            c, s = math.cos(angle), math.sin(angle)
            g_rot[i, i] = c
            if i + 1 < r:
                g_rot[i, i + 1] = -s
                g_rot[i + 1, i] = s

        recon = (U_r @ g_rot * S_r) @ Vt_r
        return recon.astype(np.float32)
