from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np

from ..._common import _block_int8_fallback, _block_int8_decompress


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class CategoryEndofunctor:
    """Endofunctor F:C→C: convolutional kernel SVD acting on object/arrow pairs."""

    name = "categoryendofunctor"
    category = "unified_physics_quantum2"

    def compress(self, tensor: np.ndarray, rank: int = 8) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64).reshape(tensor.shape[0], -1)
        m, n = t.shape
        k = min(rank, m, n)
        U, S, Vt = np.linalg.svd(t, full_matrices=False)
        k = min(k, len(S))
        buf = struct.pack("<III", m, n, k)
        buf += _serialize(U[:, :k]) + _serialize(S[:k]) + _serialize(Vt[:k, :])
        return bytes(buf), {"shape": tensor.shape}

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        m, n, k = struct.unpack_from("<III", data, 0)
        pos = 12
        U_k = np.frombuffer(data[pos : pos + m * k * 4], dtype=np.float32).reshape(m, k)
        pos += m * k * 4
        S_k = np.frombuffer(data[pos : pos + k * 4], dtype=np.float32)
        pos += k * 4
        Vt_k = np.frombuffer(data[pos : pos + k * n * 4], dtype=np.float32).reshape(
            k, n
        )
        return ((U_k * S_k) @ Vt_k).astype(np.float32).reshape(metadata["shape"])
