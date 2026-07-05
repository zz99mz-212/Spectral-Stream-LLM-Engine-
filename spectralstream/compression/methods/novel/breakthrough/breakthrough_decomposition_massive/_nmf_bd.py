from __future__ import annotations

import struct
from typing import Tuple

import numpy as np


class NMF_bd:
    """NMF_bd: Nonnegative Matrix Factorization."""
    name = "nmf_bd"
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
        W = (U[:, :r] * np.sqrt(S[:r])).astype(np.float32)
        H = (Vt[:r, :].T * np.sqrt(S[:r])).astype(np.float32)
        data = struct.pack("<III", m, n, r) + W.tobytes() + H.tobytes()
        return data, {"shape": orig_shape, "r": r}

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        m, n, r = struct.unpack_from("<III", data, 0)
        pos = 12
        W = np.frombuffer(data[pos:pos + m * r * 4], dtype=np.float32).reshape(m, r)
        pos += m * r * 4
        H = np.frombuffer(data[pos:pos + n * r * 4], dtype=np.float32).reshape(n, r)
        result = W @ H.T
        return result.reshape(metadata["shape"]).astype(np.float32)
