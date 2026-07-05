from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class LyapunovSpectrum:
    """Lyapunov spectrum: top-k exponents + tangent basis for attractor dimension."""

    name = "lyapunov_spectrum"
    category = "novel_chaos"

    def compress(self, tensor: np.ndarray, k_exponents: int = 4) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        if t.ndim < 2:
            t = t.reshape(1, -1)
        m, n = t.shape

        U, S, Vt = np.linalg.svd(t, full_matrices=False)
        k = min(k_exponents, len(S))

        lyap_exponents = np.log(S[:k] + 1e-30)
        tangent_basis = U[:, :k]

        meta = dict(shape=tensor.shape, k=k)
        data = _serialize(lyap_exponents.astype(np.float32))
        data += _serialize(tangent_basis.astype(np.float32))
        data += _serialize(S[:k].astype(np.float32))
        data += _serialize(Vt[:k, :].astype(np.float32))
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        k = metadata["k"]
        m, n = shape

        pos = 0
        lyap = _deserialize(data[: k * 4])
        pos += k * 4
        U = _deserialize(data[pos : pos + m * k * 4]).reshape(m, k)
        pos += m * k * 4
        S = np.exp(_deserialize(data[pos : pos + k * 4]))
        pos += k * 4
        Vt = _deserialize(data[pos : pos + k * n * 4]).reshape(k, n)

        return ((U * S) @ Vt).reshape(shape).astype(np.float32)
