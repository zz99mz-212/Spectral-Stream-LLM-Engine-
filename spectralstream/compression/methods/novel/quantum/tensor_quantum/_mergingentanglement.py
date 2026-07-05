from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()


def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()


class MergingEntanglement:
    """Merging entanglement via 2D SVD with left/right block reconstruction."""

    name = "merging_entanglement"
    category = "tensor_quantum"

    def compress(self, tensor: np.ndarray, rank: int = 8) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        orig_shape = t.shape
        if t.ndim == 2:
            m, n = t.shape
            rk = min(rank, min(m, n))
            u, s, vt = np.linalg.svd(t, full_matrices=False)
            rk = min(rk, len(s))
            meta = dict(shape=orig_shape, rank=rk)
            data = struct.pack("<i", rk)
            data += _serialize(u[:, :rk].astype(np.float32))
            data += _serialize(s[:rk].astype(np.float32))
            data += _serialize(vt[:rk, :].astype(np.float32))
            return data, meta
        meta = dict(shape=orig_shape, rank=0)
        data = struct.pack("<i", 0) + _serialize(t.ravel().astype(np.float32))
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        rk = struct.unpack_from("<i", data, 0)[0]
        if rk == 0:
            flat = _deserialize(data[4:])
            return flat[: int(np.prod(shape))].reshape(shape).astype(np.float32)
        pos = 4
        m, n = shape
        u = _deserialize(data[pos : pos + m * rk * 4]).reshape(m, rk)
        pos += m * rk * 4
        s = _deserialize(data[pos : pos + rk * 4])
        pos += rk * 4
        vt = _deserialize(data[pos : pos + rk * n * 4]).reshape(rk, n)
        return ((u * s) @ vt).astype(np.float32)
