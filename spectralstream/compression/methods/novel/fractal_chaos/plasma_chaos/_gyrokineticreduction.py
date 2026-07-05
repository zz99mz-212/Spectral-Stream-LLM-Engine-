from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class GyrokineticReduction:
    """Gyrokinetic reduction: 6D→5D via gyroaveraging, remove gyrophase dimension."""

    name = "gyrokinetic_reduction"
    category = "novel_physics"

    def compress(
        self, tensor: np.ndarray, n_gyro_angles: int = 8
    ) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        orig = t.copy()
        m, n = orig.shape
        step = max(1, min(m, n) // 16)
        rng = np.random.RandomState(42 + hash(tensor.shape) % 1000)

        gc = np.zeros_like(orig)
        for _ in range(n_gyro_angles):
            th = rng.uniform(0, 2 * np.pi)
            dx = int(step * np.cos(th))
            dy = int(step * np.sin(th))
            gc += np.roll(np.roll(orig, dx, axis=0), dy, axis=1)
        gc /= n_gyro_angles
        gp = orig - gc

        U, S, Vt = np.linalg.svd(gc, full_matrices=False)
        cum = np.cumsum(S) / np.sum(S)
        r = int(np.searchsorted(cum, 0.90)) + 1

        thr = np.percentile(np.abs(gp), 92)
        mask = np.abs(gp) > thr
        gidx = np.argwhere(mask)
        gvals = gp[mask]

        meta = dict(shape=tensor.shape, rank=r, n_gp=len(gvals))
        data = _serialize(U[:, :r].astype(np.float32))
        data += _serialize(S[:r].astype(np.float32))
        data += _serialize(Vt[:r, :].astype(np.float32))
        data += struct.pack("<i", len(gidx))
        data += _serialize(gidx.astype(np.int16)) + gvals.astype(np.float16).tobytes()
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        rank = metadata["rank"]
        m, n = shape

        pos = 0
        U = _deserialize(data[: m * rank * 4]).reshape(m, rank)
        pos += m * rank * 4
        S = _deserialize(data[pos : pos + rank * 4])
        pos += rank * 4
        Vt = _deserialize(data[pos : pos + rank * n * 4]).reshape(rank, n)
        pos += rank * n * 4
        recon = (U * S) @ Vt

        n_gp_data = data[pos : pos + 4]
        pos += 4
        if len(n_gp_data) >= 4:
            n_pts = int(np.frombuffer(n_gp_data, dtype=np.int32)[0])
            if n_pts > 0:
                gidx = np.frombuffer(
                    data[pos : pos + n_pts * 4], dtype=np.int16
                ).reshape(-1, 2)
                pos += n_pts * 4
                gvals = np.frombuffer(
                    data[pos : pos + n_pts * 2], dtype=np.float16
                ).astype(np.float64)
                for (ii, jj), vv in zip(gidx, gvals):
                    if ii < m and jj < n:
                        recon[ii, jj] += vv

        return recon.reshape(shape).astype(np.float32)
