from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class WeilPetersson:
    """C13. WEIL-PETERSSON: Fenchel-Nielsen (ℓ_i, τ_i) coordinates."""

    name = "weil_petersson"
    category = "novel_topological"

    def compress(self, tensor: np.ndarray, n_pairs: int = 8) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        m, n = t.shape
        k = min(n_pairs, m, n)

        U, S, Vt = np.linalg.svd(t, full_matrices=False)
        r = max(1, min(k, len(S)))
        S_r = S[:r]
        S_log = np.log(np.maximum(S_r, 1e-30))
        length_coords = S_log[:k]
        twist_coords = np.arctan2(
            U[:k, :k].real if hasattr(U, "real") else U[:k, :k],
            Vt[:k, :k].real if hasattr(Vt, "real") else Vt[:k, :k],
        )

        meta = dict(shape=t.shape, r=r, k=k)
        data = (
            _serialize(U[:, :r].astype(np.float32))
            + _serialize(S_r.astype(np.float32))
            + _serialize(Vt[:r, :].astype(np.float32))
            + _serialize(length_coords[:k].astype(np.float32))
            + _serialize(twist_coords[:k, :k].astype(np.float32))
        )
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        r = metadata["r"]
        k = metadata["k"]
        m, n = shape

        pos = 0
        U_r = _deserialize(data[: m * r * 4]).reshape(m, r)
        pos += m * r * 4
        S_r = _deserialize(data[pos : pos + r * 4])
        pos += r * 4
        Vt_r = _deserialize(data[pos : pos + r * n * 4]).reshape(r, n)
        pos += r * n * 4
        lengths = _deserialize(data[pos : pos + k * 4])
        pos += k * 4
        twists = _deserialize(data[pos : pos + k * k * 4]).reshape(k, k)

        recon = (U_r * S_r) @ Vt_r

        for i in range(min(k, r)):
            angle = float(twists[i % k, i % k]) if k > 0 else 0.0
            recon += (
                np.outer(U_r[:, i], Vt_r[i, :])
                * (math.exp(lengths[i]) if i < len(lengths) else 1.0)
                * angle
                * 0.01
            )

        return recon.astype(np.float32)
