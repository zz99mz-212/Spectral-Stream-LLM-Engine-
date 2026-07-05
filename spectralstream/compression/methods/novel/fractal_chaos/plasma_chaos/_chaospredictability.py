from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class ChaosPredictability:
    """Predictability horizon: T_λ = 1/λ_max, principled truncation threshold."""

    name = "chaos_predictability"
    category = "novel_chaos"

    def compress(
        self, tensor: np.ndarray, horizon_mult: float = 1.0
    ) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        if t.ndim < 2:
            t = t.reshape(1, -1)
        m, n = t.shape

        U, S, Vt = np.linalg.svd(t, full_matrices=False)
        flat = t.ravel()

        dS = np.diff(np.log(S + 1e-30))
        lyap_max = float(np.max(np.abs(dS[: min(10, len(dS))]))) if len(dS) > 0 else 0.1
        T_lambda = 1.0 / max(lyap_max, 1e-30)
        T_pred = T_lambda * horizon_mult

        keep_fraction = min(1.0, T_pred / max(m, n))
        k = max(1, int(keep_fraction * len(S)))
        k = min(k, len(S))

        meta = dict(shape=tensor.shape, rank=k, lyap_max=lyap_max, T_lambda=T_lambda)
        data = _serialize(U[:, :k].astype(np.float32))
        data += _serialize(S[:k].astype(np.float32))
        data += _serialize(Vt[:k, :].astype(np.float32))
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        rank = metadata["rank"]

        U = _deserialize(data[: shape[0] * rank * 4]).reshape(shape[0], rank)
        pos = shape[0] * rank * 4
        S = _deserialize(data[pos : pos + rank * 4])
        pos += rank * 4
        Vt = _deserialize(data[pos : pos + rank * shape[-1] * 4]).reshape(
            rank, shape[-1]
        )

        return ((U * S) @ Vt).reshape(shape).astype(np.float32)
