from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class RecurrencePlot:
    """Recurrence quantification: R_ij = Θ(ε - ||x_i - x_j||), RQA measures."""

    name = "recurrence_plot"
    category = "novel_chaos"

    def compress(self, tensor: np.ndarray, epsilon: float = 0.1) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        if t.ndim < 2:
            t = t.reshape(1, -1)
        m, n = t.shape

        data_dim = min(m, 64)
        data_pts = t[:data_dim, :].reshape(data_dim, -1)
        n_pts = data_pts.shape[0]

        dists = np.zeros((n_pts, n_pts))
        for i in range(n_pts):
            diff = data_pts - data_pts[i : i + 1]
            dists[i] = np.sqrt(np.sum(diff**2, axis=1))

        R = (dists < epsilon).astype(np.float64)
        RR = float(np.mean(R))
        DET = float(np.sum(R * np.roll(R, 1, axis=0)) / max(np.sum(R), 1))
        LAM = float(np.sum(R * np.roll(R, 1, axis=1)) / max(np.sum(R), 1))
        p = np.sum(R, axis=0) / max(np.sum(R), 1)
        p = p[p > 0]
        ENTR = float(-np.sum(p * np.log(p + 1e-30))) if len(p) > 0 else 0.0

        U, S, Vt = np.linalg.svd(t, full_matrices=False)
        cum = np.cumsum(S) / np.sum(S)
        r = int(np.searchsorted(cum, 0.85)) + 1

        meta = dict(
            shape=tensor.shape,
            RR=RR,
            DET=DET,
            LAM=LAM,
            ENTR=ENTR,
            epsilon=epsilon,
            rank=r,
        )
        data = struct.pack("<ddddd", RR, DET, LAM, ENTR, epsilon)
        data += _serialize(U[:, :r].astype(np.float32))
        data += _serialize(S[:r].astype(np.float32))
        data += _serialize(Vt[:r, :].astype(np.float32))
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        rank = metadata["rank"]

        pos = 40
        m = shape[0]
        n = shape[-1]

        U = _deserialize(data[pos : pos + m * rank * 4]).reshape(m, rank)
        pos += m * rank * 4
        S = _deserialize(data[pos : pos + rank * 4])
        pos += rank * 4
        Vt = _deserialize(data[pos : pos + rank * n * 4]).reshape(rank, n)

        return ((U * S) @ Vt).reshape(shape).astype(np.float32)
