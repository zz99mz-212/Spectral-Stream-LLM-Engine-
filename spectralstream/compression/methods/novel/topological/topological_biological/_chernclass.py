from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class ChernClass:
    """C17. CHERN-CLASS: c_k(E) = [P_k(F_∇)] characteristic classes."""

    name = "chern_class"
    category = "novel_topological"

    def compress(self, tensor: np.ndarray, n_chern: int = 4) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        m, n = t.shape
        k = min(n_chern, m, n)

        try:
            U, S, Vt = np.linalg.svd(t, full_matrices=False)
        except np.linalg.LinAlgError:
            U = np.eye(m, k)
            S = np.ones(k)
            Vt = np.eye(k, n)

        r = min(k, len(S))
        curvature = S[:r] ** 2
        chern_numbers = np.zeros(r, dtype=np.float64)
        for j in range(r):
            chern_numbers[j] = np.sum(curvature[: j + 1]) / (2 * np.pi)

        meta = dict(shape=t.shape, r=r)
        data = (
            _serialize(U[:, :r].astype(np.float32))
            + _serialize(S[:r].astype(np.float32))
            + _serialize(Vt[:r, :].astype(np.float32))
            + _serialize(chern_numbers.astype(np.float32))
        )
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        r = metadata["r"]
        m, n = shape

        U_r = _deserialize(data[: m * r * 4]).reshape(m, r)
        pos = m * r * 4
        S_r = _deserialize(data[pos : pos + r * 4])
        pos += r * 4
        Vt_r = _deserialize(data[pos : pos + r * n * 4]).reshape(r, n)
        pos += r * n * 4
        chern = _deserialize(data[pos : pos + r * 4])

        S_adjusted = S_r * (1.0 + 0.05 * chern)
        return ((U_r * S_adjusted) @ Vt_r).astype(np.float32)
