from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class LateralInhibition:
    """D11. LATERAL-INHIBITION: WTA — y_i = w_ij if max_k w_ik else 0."""

    name = "lateral_inhibition"
    category = "novel_biological"

    def compress(self, tensor: np.ndarray, top_k: int = 1) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        m, n = t.shape
        k = min(top_k, n)

        top_idx = np.argpartition(t, -k, axis=1)[:, -k:]
        top_vals = np.take_along_axis(t, top_idx, axis=1)

        meta = dict(shape=t.shape, k=k)
        data = (
            _serialize(top_idx.astype(np.int16)) + top_vals.astype(np.float16).tobytes()
        )
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        k = metadata["k"]
        m, n = shape

        idx_data = data[: m * k * 2]
        val_data = data[m * k * 2 :]

        top_idx = np.frombuffer(idx_data, dtype=np.int16).astype(int).reshape(m, k)
        top_vals = (
            np.frombuffer(val_data, dtype=np.float16).astype(np.float64).reshape(m, k)
        )

        recon = np.zeros((m, n), dtype=np.float64)
        for i in range(m):
            for j in range(k):
                ci = top_idx[i, j]
                if 0 <= ci < n:
                    recon[i, ci] = top_vals[i, j]
        return recon.astype(np.float32)
