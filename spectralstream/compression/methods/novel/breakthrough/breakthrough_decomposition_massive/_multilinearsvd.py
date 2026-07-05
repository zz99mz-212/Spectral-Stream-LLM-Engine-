from __future__ import annotations

import struct
from typing import Tuple

import numpy as np


class MultilinearSVD:
    """MultilinearSVD:         k = params.get('rank', max(1, min(m, n) // 5))."""
    name = "multilinear_svd"
    category = "breakthrough_decomposition"

    def compress(self, tensor: np.ndarray, **params) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        orig_shape = t.shape
        if t.ndim < 2:
            t_2d = t.reshape(1, -1)
        else:
            t_2d = t.reshape(t.shape[0], -1)
        mat = t_2d.copy()
        mat = np.ascontiguousarray(mat)
        if dct is not None:
            mat = dct(mat, norm='ortho', axis=0)

        m, n = mat.shape
        k = params.get('rank', max(1, min(m, n) // 5))
        k = min(k, m, n)
        U, S, Vt = np.linalg.svd(mat, full_matrices=False)
        k = min(k, len(S))
        U_k = U[:, :k].astype(np.float32)
        S_k = S[:k].astype(np.float32)
        Vt_k = Vt[:k, :].astype(np.float32)

        extra_meta = {}


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

        result = (U_k * S_k) @ Vt_k

        if idct is not None:
            result = idct(result, norm='ortho', axis=0)
        return result.reshape(metadata["shape"]).astype(np.float32)
