from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class HallMHD:
    """Hall effect MHD: whistler physics, ε_H = d_i/L determines scale threshold."""

    name = "hall_mhd"
    category = "novel_physics"

    def compress(
        self, tensor: np.ndarray, hall_parameter: float = 0.2
    ) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        if t.ndim < 2:
            t = t.reshape(1, -1)
        m, n = t.shape

        U, S, Vt = np.linalg.svd(t, full_matrices=False)
        k = len(S)

        d_i = hall_parameter
        scale = np.arange(1, k + 1)
        ion_scales = scale * d_i / max(m, n)
        hall_weight = 1.0 / (1.0 + ion_scales**2)
        retained = hall_weight > np.percentile(hall_weight, 60)
        k2 = max(1, int(np.sum(retained)))
        k2 = min(k2, k)

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
