from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class CrisisInduced:
    """Boundary crisis: γ_c determines attractor size, adaptive compression."""

    name = "crisis_induced"
    category = "novel_chaos"

    def compress(
        self, tensor: np.ndarray, crisis_threshold: float = 0.5
    ) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        flat = t.ravel()

        gamma = float(np.std(flat))
        gamma_c = crisis_threshold * gamma
        pre_crisis = gamma < gamma_c

        damping = 1.0 if pre_crisis else max(0.1, gamma_c / max(gamma, 1e-30))

        if t.ndim < 2:
            t = t.reshape(1, -1)
        m, n = t.shape

        U, S, Vt = np.linalg.svd(t, full_matrices=False)
        S_damped = S * damping
        cum = np.cumsum(S_damped) / np.sum(S_damped)
        k = int(np.searchsorted(cum, 0.85)) + 1
        k = min(k, len(S))

        meta = dict(
            shape=tensor.shape,
            rank=k,
            gamma=gamma,
            gamma_c=gamma_c,
            pre_crisis=bool(pre_crisis),
            damping=float(damping),
        )
        data = _serialize(U[:, :k].astype(np.float32))
        data += _serialize(S_damped[:k].astype(np.float32))
        data += _serialize(Vt[:k, :].astype(np.float32))
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        rank = metadata["rank"]

        U = _deserialize(data[: shape[0] * rank * 4]).reshape(shape[0], rank)
        pos = shape[0] * rank * 4
        S = _deserialize(data[pos : pos + rank * 4])
        pos += rank * 4
        damping = metadata.get("damping", 1.0)
        S = S / max(damping, 1e-30)
        Vt = _deserialize(data[pos : pos + rank * shape[-1] * 4]).reshape(
            rank, shape[-1]
        )

        return ((U * S) @ Vt).reshape(shape).astype(np.float32)
