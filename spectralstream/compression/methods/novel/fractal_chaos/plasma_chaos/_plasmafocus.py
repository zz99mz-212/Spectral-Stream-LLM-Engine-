from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class PlasmaFocus:
    """Pinch force balance: J×B = ∇p, physics-constrained row/column structure."""

    name = "plasma_focus"
    category = "novel_physics"

    def compress(self, tensor: np.ndarray, rank: int = 8) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        if t.ndim < 2:
            t = t.reshape(1, -1)
        m, n = t.shape

        U, S, Vt = np.linalg.svd(t, full_matrices=False)
        k = min(rank, len(S))

        pressure = np.gradient(np.log(S[:k] + 1e-30))
        keep_mask = (
            np.abs(pressure) > np.percentile(np.abs(pressure), 40)
            if k > 1
            else np.ones(k, dtype=bool)
        )
        k2 = max(1, int(np.sum(keep_mask)))

        meta = dict(shape=tensor.shape, rank=k2)
        data = _serialize(U[:, :k2].astype(np.float32))
        data += _serialize(S[:k2].astype(np.float32))
        data += _serialize(Vt[:k2, :].astype(np.float32))
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        rank = metadata["rank"]

        U = _deserialize(data[: shape[0] * rank * 4]).reshape(shape[0], rank)
        pos = shape[0] * rank * 4
        S = _deserialize(data[pos : pos + rank * 4])
        pos += rank * 4
        Vt = _deserialize(data[pos : pos + rank * shape[-1] * 4]).reshape(
            rank, shape[-1]
        )

        return ((U * S) @ Vt).reshape(shape).astype(np.float32)
