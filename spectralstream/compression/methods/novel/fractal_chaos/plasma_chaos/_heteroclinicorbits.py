from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class HeteroclinicOrbits:
    """Saddle connections: store saddles + connecting paths."""

    name = "heteroclinic_orbits"
    category = "novel_chaos"

    def compress(self, tensor: np.ndarray, n_saddles: int = 4) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        if t.ndim < 2:
            t = t.reshape(1, -1)
        m, n = t.shape

        U, S, Vt = np.linalg.svd(t, full_matrices=False)

        saddle_idx = np.linspace(0, len(S) - 1, min(n_saddles, len(S))).astype(int)
        saddles = []
        connections = []
        for i in range(len(saddle_idx) - 1):
            s1 = saddle_idx[i]
            s2 = saddle_idx[i + 1]
            saddles.append(U[:, s1] * S[s1])
            saddles.append(U[:, s2] * S[s2])
            path = Vt[s1 : s1 + 1, :] + Vt[s2 : s2 + 1, :]
            connections.append(path)

        data = b""
        for s in saddles:
            data += _serialize(s.astype(np.float32))
        for c in connections:
            data += _serialize(c.astype(np.float32))
        data += _serialize(np.array([len(saddles), len(connections)], dtype=np.int32))

        k = max(1, min(8, len(S)))
        data = _serialize(U[:, :k].astype(np.float32))
        data += _serialize(S[:k].astype(np.float32))
        data += _serialize(Vt[:k, :].astype(np.float32))

        meta = dict(shape=tensor.shape, k=k, n_saddles=n_saddles)
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        k = metadata["k"]

        U = _deserialize(data[: shape[0] * k * 4]).reshape(shape[0], k)
        pos = shape[0] * k * 4
        S = _deserialize(data[pos : pos + k * 4])
        pos += k * 4
        Vt = _deserialize(data[pos : pos + k * shape[-1] * 4]).reshape(k, shape[-1])

        return ((U * S) @ Vt).reshape(shape).astype(np.float32)
