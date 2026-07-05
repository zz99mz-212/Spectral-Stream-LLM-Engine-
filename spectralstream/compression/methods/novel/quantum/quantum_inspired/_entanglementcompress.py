from __future__ import annotations

import math
import struct
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class EntanglementCompress:
    """Use Schmidt decomposition across tensor pairs to capture entanglement.
    W_ij = Σ_k λ_k u_i^A v_j^B. Store Schmidt coefficients λ_k and vectors.
    """

    name = "entanglement_compress"
    category = "quantum_compression"

    def compress(self, tensor: np.ndarray, schmidt_rank: int = 8) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        orig_shape = t.shape
        if t.ndim < 2:
            t = t.reshape(-1, 1)
        m, n = t.shape
        r = min(schmidt_rank, m, n)
        U, S, Vt = np.linalg.svd(t, full_matrices=False)
        k = min(r, len(S))
        lam = S[:k].astype(np.float32)
        u = U[:, :k].astype(np.float32)
        v = Vt[:k, :].astype(np.float32)
        meta = dict(
            shape=orig_shape,
            m=m,
            n=n,
            k=k,
        )
        data = struct.pack("<III", m, n, k)
        data += _serialize(lam)
        data += _serialize(u)
        data += _serialize(v)
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        m, n, k = struct.unpack_from("<III", data, 0)
        pos = 12
        lam = _deserialize(data[pos : pos + k * 4])
        pos += k * 4
        u = _deserialize(data[pos : pos + m * k * 4]).reshape(m, k)
        pos += m * k * 4
        v = _deserialize(data[pos : pos + k * n * 4]).reshape(k, n)
        recon = (u * lam) @ v
        return recon.reshape(shape).astype(np.float32)
