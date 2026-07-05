from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class BundleGauge:
    """C3. BUNDLE-GAUGE: connection on principal G-bundle, Coulomb gauge."""

    name = "bundle_gauge"
    category = "novel_topological"

    def compress(self, tensor: np.ndarray, rank: int = None) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        m, n = t.shape
        if rank is None:
            rank = max(1, min(m, n) // 4)

        U, S, Vt = np.linalg.svd(t, full_matrices=False)
        r = min(rank, len(S))

        A = U[:, :r] @ np.diag(np.sqrt(S[:r]))
        B = np.diag(np.sqrt(S[:r])) @ Vt[:r, :]

        A_mean = np.mean(A, axis=0, keepdims=True)
        B_mean = np.mean(B, axis=1, keepdims=True)
        A_gauge = A - A_mean
        B_gauge = B - B_mean

        meta = dict(shape=t.shape, r=r)
        data = (
            _serialize(A_gauge.astype(np.float32))
            + _serialize(B_gauge.astype(np.float32))
            + _serialize(A_mean.astype(np.float32))
            + _serialize(B_mean.astype(np.float32))
        )
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        r = metadata["r"]
        m, n = shape

        off = m * r * 4
        A_g = _deserialize(data[:off]).reshape(m, r)
        pos = off
        off2 = r * n * 4
        B_g = _deserialize(data[pos : pos + off2]).reshape(r, n)
        pos += off2
        A_mean = _deserialize(data[pos : pos + r * 4]).reshape(1, r)
        pos += r * 4
        B_mean = _deserialize(data[pos : pos + r * 4]).reshape(r, 1)

        A = A_g + A_mean
        B = B_g + B_mean
        return (A @ B).astype(np.float32)
