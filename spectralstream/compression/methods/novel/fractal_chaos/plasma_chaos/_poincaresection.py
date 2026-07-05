from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class PoincareSection:
    """Return map P: Σ → Σ on Poincaré section."""

    name = "poincare_section"
    category = "novel_chaos"

    def compress(self, tensor: np.ndarray, section_dim: int = 8) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        if t.ndim < 2:
            t = t.reshape(1, -1)
        m, n = t.shape

        U, S, Vt = np.linalg.svd(t, full_matrices=False)
        k = min(section_dim, len(S))
        cum = np.cumsum(S) / np.sum(S)
        r = int(np.searchsorted(cum, 0.90)) + 1
        r = min(r, k)

        section_indices = np.linspace(0, n - 1, min(section_dim, n)).astype(int)
        P_map = Vt[:r, :r] if r > 1 else np.array([[Vt[0, 0]]])

        recon_part = (U[:, :r] * S[:r]) @ Vt[:r, :]
        residual = t - recon_part
        thr = np.percentile(np.abs(residual), 90)
        rmask = np.abs(residual) > thr
        ridx = np.argwhere(rmask)
        rvals = residual[rmask]

        meta = dict(shape=tensor.shape, r=r, section_dim=section_dim, n_res=len(rvals))
        data = _serialize(U[:, :r].astype(np.float32))
        data += _serialize(S[:r].astype(np.float32))
        data += _serialize(Vt[:r, :].astype(np.float32))
        data += _serialize(P_map.astype(np.float32).ravel())
        data += _serialize(section_indices.astype(np.int32))
        data += struct.pack("<i", len(ridx))
        data += _serialize(ridx.astype(np.int16)) + rvals.astype(np.float16).tobytes()
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        r = metadata["r"]
        n_res = metadata.get("n_res", 0)
        m, n = shape

        pos = 0
        U = _deserialize(data[: m * r * 4]).reshape(m, r)
        pos += m * r * 4
        S = _deserialize(data[pos : pos + r * 4])
        pos += r * 4
        Vt = _deserialize(data[pos : pos + r * n * 4]).reshape(r, n)
        pos += r * n * 4

        recon = (U * S) @ Vt

        p_map_size = r * r
        pos += p_map_size * 4
        pos += min(metadata["section_dim"], n) * 4

        if n_res > 0:
            n_r = int(np.frombuffer(data[pos : pos + 4], dtype=np.int32)[0])
            pos += 4
            if n_r > 0:
                ridx = np.frombuffer(data[pos : pos + n_r * 4], dtype=np.int16).reshape(
                    -1, 2
                )
                pos += n_r * 4
                rvals = np.frombuffer(
                    data[pos : pos + n_r * 2], dtype=np.float16
                ).astype(np.float64)
                for (ii, jj), vv in zip(ridx, rvals):
                    if ii < m and jj < n:
                        recon[ii, jj] += vv

        return recon.reshape(shape).astype(np.float32)
