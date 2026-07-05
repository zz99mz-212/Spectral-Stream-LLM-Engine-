from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class RicciFlow:
    """C11. RICCI-FLOW: ∂g/∂t = -2 Ric(g) geometric flow smoothing."""

    name = "ricci_flow"
    category = "novel_topological"

    def compress(
        self, tensor: np.ndarray, n_steps: int = 5, dt: float = 0.1
    ) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        m, n = t.shape

        flow = t.copy()
        for _ in range(n_steps):
            grad_xx = np.gradient(np.gradient(flow, axis=0), axis=0)
            grad_yy = np.gradient(np.gradient(flow, axis=1), axis=1)
            ricci = -(grad_xx + grad_yy) * 0.5
            flow = flow + dt * 2 * ricci

        residual = t - flow
        thr = np.percentile(np.abs(residual), 90)
        mask = np.abs(residual) > thr
        ridx = np.argwhere(mask)
        rvals = residual[mask]

        U, S, Vt = np.linalg.svd(flow, full_matrices=False)
        r = max(1, np.sum(S > np.max(S) * 0.02))

        meta = dict(shape=t.shape, r=int(r), n_steps=n_steps)
        data = (
            _serialize(U[:, :r].astype(np.float32))
            + _serialize(S[:r].astype(np.float32))
            + _serialize(Vt[:r, :].astype(np.float32))
            + _serialize(ridx.astype(np.int16))
            + rvals.astype(np.float16).tobytes()
        )
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        r = metadata["r"]
        m, n = shape

        pos = 0
        U = _deserialize(data[: m * r * 4]).reshape(m, r)
        pos += m * r * 4
        S = _deserialize(data[pos : pos + r * 4])
        pos += r * 4
        Vt = _deserialize(data[pos : pos + r * n * 4]).reshape(r, n)
        pos += r * n * 4

        flow = (U * S) @ Vt

        remaining = data[pos:]
        n_pts = len(remaining) // (4 + 2) if len(remaining) >= 6 else 0
        if n_pts > 0:
            ridx = _deserialize(remaining[: n_pts * 4]).astype(int).reshape(-1, 2)
            rvals = np.frombuffer(remaining[n_pts * 4 :], dtype=np.float16).astype(
                np.float64
            )
            for i in range(min(n_pts, len(ridx), len(rvals))):
                ri, rj = ridx[i]
                if ri < m and rj < n:
                    flow[ri, rj] += rvals[i]

        return flow.astype(np.float32)
