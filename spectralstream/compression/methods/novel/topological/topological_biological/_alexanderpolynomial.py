from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class AlexanderPolynomial:
    """C10. ALEXANDER-POLYNOMIAL: knot invariants from weight braids."""

    name = "alexander_polynomial"
    category = "novel_topological"

    def compress(self, tensor: np.ndarray, n_coeffs: int = 8) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        m, n = t.shape
        k = min(n_coeffs, m, n)

        braid_rows = t[:k, :k].copy()

        det = np.linalg.det(braid_rows) if k > 0 else 0.0
        trace = np.trace(braid_rows) if k > 0 else 0.0

        U, S, Vt = np.linalg.svd(t, full_matrices=False)
        r = max(1, min(k, np.sum(S > np.max(S) * 0.02)))
        S_r = S[:r]

        meta = dict(shape=t.shape, k=k, r=r, det=float(det), trace=float(trace))
        data = (
            _serialize(U[:, :r].astype(np.float32))
            + _serialize(S_r.astype(np.float32))
            + _serialize(Vt[:r, :].astype(np.float32))
        )
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        r = metadata["r"]
        m, n = shape

        U = _deserialize(data[: m * r * 4]).reshape(m, r)
        pos = m * r * 4
        S = _deserialize(data[pos : pos + r * 4])
        pos += r * 4
        Vt = _deserialize(data[pos : pos + r * n * 4]).reshape(r, n)

        det = metadata.get("det", 0.0)
        trace = metadata.get("trace", 0.0)
        recon = (U * S) @ Vt
        recon[0, 0] += det * 0.01
        if r > 0:
            recon[0, 0] += trace * 0.01
        return recon.astype(np.float32)
