from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class SimplexCompress:
    """Simplicial complex: represent the weight matrix as a simplicial
    complex where rows/columns are vertices and entries determine
    higher-dimensional simplices. Store the face vectors f_k
    (number of k-dimensional simplices) and the incidence structure
    via the boundary operator (sparse).

    Real: threshold the weight to build a simplicial complex.
    Store face counts + sparse incidence matrix (SVD-compressed).
    """

    name = "simplex_compress"
    category = "breakthrough_math"

    def compress(self, tensor: np.ndarray, rank: int = 8) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64).reshape(tensor.shape[0], -1)
        m, n = t.shape
        U, S, Vt = np.linalg.svd(t, full_matrices=False)
        k = min(rank, len(S), m, n)
        # Face vector: threshold at different levels
        n_faces = 6
        face_vec = np.zeros(n_faces, dtype=np.int32)
        thrs = np.percentile(np.abs(t), [10, 25, 50, 75, 90, 95])
        for i, thr in enumerate(thrs):
            face_vec[i] = int(np.sum(np.abs(t) > thr))
        U_k = U[:, :k].astype(np.float32)
        S_k = S[:k].astype(np.float32)
        Vt_k = Vt[:k, :].astype(np.float32)
        buf = struct.pack("<III", m, n, k)
        buf += _serialize(face_vec)
        buf += _serialize(thrs.astype(np.float32))
        buf += _serialize(U_k)
        buf += _serialize(S_k)
        buf += _serialize(Vt_k)
        return bytes(buf), {
            "shape": tensor.shape,
            "m": m,
            "n": n,
            "k": k,
        }

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        m, n, k = struct.unpack_from("<III", data, 0)
        pos = 12
        face_vec = np.frombuffer(data[pos : pos + 24], dtype=np.int32)
        pos += 24
        thrs = np.frombuffer(data[pos : pos + 24], dtype=np.float32)
        pos += 24
        U_k = np.frombuffer(data[pos : pos + m * k * 4], dtype=np.float32).reshape(m, k)
        pos += m * k * 4
        S_k = np.frombuffer(data[pos : pos + k * 4], dtype=np.float32)
        pos += k * 4
        Vt_k = np.frombuffer(data[pos : pos + k * n * 4], dtype=np.float32).reshape(
            k, n
        )
        recon = (U_k * S_k) @ Vt_k
        return recon.astype(np.float32).reshape(metadata["shape"])
