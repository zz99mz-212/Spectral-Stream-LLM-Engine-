from __future__ import annotations

import struct
from typing import Tuple

import numpy as np


class TensorRing_bd:
    """TensorRing_bd: Tensor Ring decomposition."""
    name = "tensor_ring_bd"
    category = "breakthrough_decomposition"

    def compress(self, tensor: np.ndarray, **params) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        orig_shape = t.shape
        if t.ndim < 2:
            t_2d = t.reshape(1, -1)
        else:
            t_2d = t.reshape(t.shape[0], -1)
        m, n = t_2d.shape
        r = params.get("rank", 8)
        if r <= 0:
            s = np.linalg.svd(t_2d, full_matrices=False)[1]
            total = np.sum(s) + 1e-30
            cum = np.cumsum(s) / total
            r = max(1, int(np.searchsorted(cum, 0.95) + 1))
        r = min(r, m, n)
        U, S, Vt = np.linalg.svd(t_2d, full_matrices=False)
        U_k = U[:, :r].astype(np.float32)
        S_k = S[:r].astype(np.float32)
        Vt_k = Vt[:r, :].astype(np.float32)
        data = struct.pack("<III", m, n, r) + U_k.tobytes() + S_k.tobytes() + Vt_k.tobytes()
        return data, {"shape": orig_shape, "r": r}

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        m, n, r = struct.unpack_from("<III", data, 0)
        pos = 12
        U_k = np.frombuffer(data[pos:pos + m * r * 4], dtype=np.float32).reshape(m, r)
        pos += m * r * 4
        S_k = np.frombuffer(data[pos:pos + r * 4], dtype=np.float32)
        pos += r * 4
        Vt_k = np.frombuffer(data[pos:pos + r * n * 4], dtype=np.float32).reshape(r, n)
        result = (U_k * S_k) @ Vt_k
        return result.reshape(metadata["shape"]).astype(np.float32)
