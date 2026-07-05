from __future__ import annotations

import struct
from typing import Tuple

import numpy as np


class TuckerWeighted_bd:
    """TuckerWeighted_bd: Weighted Tucker decomposition."""
    name = "tucker_weighted_bd"
    category = "breakthrough_decomposition"

    def compress(self, tensor: np.ndarray, **params) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        orig_shape = t.shape
        if t.ndim < 2:
            t_2d = t.reshape(1, -1)
        else:
            t_2d = t.reshape(t.shape[0], -1)
        m, n = t_2d.shape
        r1 = params.get("rank1", 8)
        r2 = params.get("rank2", 8)
        if r1 <= 0 or r2 <= 0:
            r1 = max(1, m // 4)
            r2 = max(1, n // 4)
        r1 = min(r1, m); r2 = min(r2, n)
        U1, S, Vt = np.linalg.svd(t_2d, full_matrices=False)
        U1_k = U1[:, :r1].astype(np.float32)
        core = ((U1_k.T @ t_2d) @ Vt[:r2, :].T).astype(np.float32)
        Vt_k = Vt[:r2, :].astype(np.float32)
        data = struct.pack("<III", m, n, r1) + U1_k.tobytes()
        data += struct.pack("<I", r2) + core.tobytes() + Vt_k.tobytes()
        return data, {"shape": orig_shape, "r1": r1, "r2": r2}

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        m, n, r1 = struct.unpack_from("<III", data, 0)
        pos = 12
        U1_k = np.frombuffer(data[pos:pos + m * r1 * 4], dtype=np.float32).reshape(m, r1)
        pos += m * r1 * 4
        r2 = struct.unpack_from("<I", data, pos)[0]; pos += 4
        core = np.frombuffer(data[pos:pos + r1 * r2 * 4], dtype=np.float32).reshape(r1, r2)
        pos += r1 * r2 * 4
        Vt_k = np.frombuffer(data[pos:pos + r2 * n * 4], dtype=np.float32).reshape(r2, n)
        result = U1_k @ core @ Vt_k
        return result.reshape(metadata["shape"]).astype(np.float32)
