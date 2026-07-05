from __future__ import annotations

import struct
from typing import Tuple

import numpy as np


class RobustSVD:
    """RobustSVD:         k = params.get('rank', max(1, min(m, n) // 4))."""
    name = "robust_svd"
    category = "breakthrough_decomposition"

    def compress(self, tensor: np.ndarray, **params) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        orig_shape = t.shape
        if t.ndim < 2:
            t_2d = t.reshape(1, -1)
        else:
            t_2d = t.reshape(t.shape[0], -1)
        mat = t_2d.copy()
        mean = np.mean(mat)
        std = max(np.std(mat), 1e-10)
        mat = (mat - mean) / std

        m, n = mat.shape
        k = params.get('rank', max(1, min(m, n) // 4))
        k = min(k, m, n)
        U, S, Vt = np.linalg.svd(mat, full_matrices=False)
        k = min(k, len(S))
        U_k = U[:, :k].astype(np.float32)
        S_k = S[:k].astype(np.float32)
        Vt_k = Vt[:k, :].astype(np.float32)

        extra_meta = {}
        extra_meta.update(dict(mean=float(mean), std=float(std)))

        data = struct.pack("<III", m, n, k)
        data += U_k.tobytes() + S_k.tobytes() + Vt_k.tobytes()
        return data, {"shape": orig_shape, "k": k, **extra_meta}

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        m, n, k = struct.unpack_from("<III", data, 0)
        pos = 12
        U_k = np.frombuffer(data[pos:pos + m * k * 4], dtype=np.float32).reshape(m, k)
        pos += m * k * 4
        S_k = np.frombuffer(data[pos:pos + k * 4], dtype=np.float32)
        pos += k * 4
        Vt_k = np.frombuffer(data[pos:pos + k * n * 4], dtype=np.float32).reshape(k, n)
        mean, std = metadata.get('mean', 0.0), metadata.get('std', 1.0)

        result = (U_k * S_k) @ Vt_k

        result = result * std + mean
        return result.reshape(metadata["shape"]).astype(np.float32)
