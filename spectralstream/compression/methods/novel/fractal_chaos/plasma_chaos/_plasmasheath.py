from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class PlasmaSheath:
    """Exponential boundary: W(x) = W_bulk + (W_wall - W_bulk)exp(-x/λ_D)."""

    name = "plasma_sheath"
    category = "novel_physics"

    def compress(
        self, tensor: np.ndarray, debye_length: float = 0.05
    ) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        if t.ndim < 2:
            t = t.reshape(1, -1)
        m, n = t.shape

        bulk = (
            float(np.median(t[m // 4 : 3 * m // 4, n // 4 : 3 * n // 4]))
            if m > 2 and n > 2
            else float(np.mean(t))
        )
        wall_l = float(np.mean(t[:, 0]))
        wall_r = float(np.mean(t[:, -1]))
        wall_t = float(np.mean(t[0, :]))
        wall_b = float(np.mean(t[-1, :]))
        wall = float(np.mean([wall_l, wall_r, wall_t, wall_b]))

        x = np.linspace(0, 1, n)[None, :]
        y = np.linspace(0, 1, m)[:, None]
        d_left = x
        d_right = 1 - x
        d_top = y
        d_bottom = 1 - y
        d = np.minimum(np.minimum(d_left, d_right), np.minimum(d_top, d_bottom))

        sheath = bulk + (wall - bulk) * np.exp(-d / max(debye_length, 1e-10))
        residual = t - sheath

        thr = np.percentile(np.abs(residual), 92)
        mask = np.abs(residual) > thr
        ridx = np.argwhere(mask)
        rvals = residual[mask]

        meta = dict(
            shape=tensor.shape,
            debye_length=debye_length,
            bulk=bulk,
            wall=wall,
            n_res=len(rvals),
        )
        data = struct.pack("<dff", debye_length, bulk, wall)
        data += _serialize(ridx.astype(np.int16)) + rvals.astype(np.float16).tobytes()
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        debye_length = metadata.get("debye_length", 0.05)
        bulk = metadata.get("bulk", 0.0)
        wall = metadata.get("wall", 0.0)
        n_res = metadata.get("n_res", 0)

        m, n = shape
        x = np.linspace(0, 1, n)[None, :]
        y = np.linspace(0, 1, m)[:, None]
        d_left = x
        d_right = 1 - x
        d_top = y
        d_bottom = 1 - y
        d = np.minimum(np.minimum(d_left, d_right), np.minimum(d_top, d_bottom))

        recon = bulk + (wall - bulk) * np.exp(-d / max(debye_length, 1e-10))

        pos = 16
        if n_res > 0:
            ridx = np.frombuffer(data[pos : pos + n_res * 4], dtype=np.int16).reshape(
                -1, 2
            )
            pos += n_res * 4
            rvals = np.frombuffer(data[pos : pos + n_res * 2], dtype=np.float16).astype(
                np.float64
            )
            for (ii, jj), vv in zip(ridx, rvals):
                if ii < m and jj < n:
                    recon[ii, jj] += vv

        return recon.astype(np.float32)
