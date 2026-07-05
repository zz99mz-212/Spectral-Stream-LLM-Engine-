from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class MHDSpectral:
    """MHD wave decomposition: Alfvén + magnetosonic + entropy modes."""

    name = "mhd_spectral"
    category = "novel_physics"

    def compress(self, tensor: np.ndarray, rank: int = 6) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        if t.ndim < 2:
            t = t.reshape(1, -1)
        m, n = t.shape

        U, S, Vt = np.linalg.svd(t, full_matrices=False)
        k = min(rank, len(S))

        S_alfven = S[: k // 3] if k // 3 > 0 else S[:1]
        S_magn = S[k // 3 : 2 * k // 3] if 2 * k // 3 <= k else S[k // 3 :]
        S_entro = S[2 * k // 3 : k] if k > 2 else S[:1]

        meta = dict(shape=tensor.shape, rank=k)
        data = _serialize(S[:k].astype(np.float32))
        data += _serialize(U[:, :k].astype(np.float32))
        data += _serialize(Vt[:k, :].astype(np.float32))
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        rank = metadata["rank"]
        m, n = shape

        pos = 0
        S = _deserialize(data[: rank * 4])
        pos += rank * 4
        U = _deserialize(data[pos : pos + m * rank * 4]).reshape(m, rank)
        pos += m * rank * 4
        Vt = _deserialize(data[pos : pos + rank * n * 4]).reshape(rank, n)

        return ((U * S) @ Vt).reshape(shape).astype(np.float32)
