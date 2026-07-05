from __future__ import annotations

import struct
from typing import Tuple

import numpy as np


class FactorizedRecurrent_bd:
    """FactorizedRecurrent_bd: Factorized recurrent weights."""
    name = "factorized_recurrent_bd"
    category = "breakthrough_decomposition"

    def compress(self, tensor: np.ndarray, **params) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        orig_shape = t.shape
        if t.ndim < 2:
            t_2d = t.reshape(1, -1)
        else:
            t_2d = t.reshape(t.shape[0], -1)
        m, n = t_2d.shape
        r = params.get("rank", max(1, min(m, n) // 6))
        r = min(r, m, n)
        U, S, Vt = np.linalg.svd(t_2d, full_matrices=False)
        r = min(r, len(S))
        A = (U[:, :r] * np.sqrt(S[:r])).astype(np.float32)
        B = (Vt[:r, :].T * np.sqrt(S[:r])).astype(np.float32)
        data = struct.pack("<III", m, n, r) + A.tobytes() + B.tobytes()
        return data, {"shape": orig_shape, "r": r}

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        m, n, r = struct.unpack_from("<III", data, 0)
        pos = 12
        A = np.frombuffer(data[pos:pos + m * r * 4], dtype=np.float32).reshape(m, r)
        pos += m * r * 4
        B = np.frombuffer(data[pos:pos + n * r * 4], dtype=np.float32).reshape(n, r)
        result = A @ B.T
        return result.reshape(metadata["shape"]).astype(np.float32)
