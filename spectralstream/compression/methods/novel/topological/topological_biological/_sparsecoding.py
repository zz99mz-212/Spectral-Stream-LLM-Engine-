from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class SparseCoding:
    """D5. SPARSE-CODING: W = D·A, min ||W-DA||² + λ||A||₁, V1 dictionaries."""

    name = "sparse_coding"
    category = "novel_biological"

    def compress(
        self, tensor: np.ndarray, dict_size: int = None, lam: float = 0.1
    ) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        m, n = t.shape

        if dict_size is None:
            dict_size = max(1, min(m, n) // 2)
        k = min(dict_size, m, n)

        D = np.random.RandomState(0).randn(m, k) * 0.1
        D = D / (np.linalg.norm(D, axis=0, keepdims=True) + 1e-30)
        A = np.zeros((k, n), dtype=np.float64)

        for i in range(n):
            x = t[:, i]
            a = np.zeros(k, dtype=np.float64)
            for _ in range(50):
                residual = x - D @ a
                grad = -D.T @ residual + lam * np.sign(a)
                a -= 0.01 * grad
                a = np.maximum(0, a)
            A[:, i] = a

        A_sparse = A.ravel()
        thr = np.percentile(np.abs(A_sparse), 70)
        mask = np.abs(A_sparse) > thr
        a_idx = np.argwhere(mask).ravel()
        a_vals = A_sparse[mask]

        meta = dict(shape=t.shape, m=m, k=k, n_vals=len(a_vals))
        data = (
            _serialize(D.astype(np.float32))
            + _serialize(a_idx.astype(np.int32))
            + a_vals.astype(np.float16).tobytes()
        )
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        m = metadata["m"]
        k = metadata["k"]
        n_vals = metadata["n_vals"]
        n = shape[1]

        pos = 0
        D = _deserialize(data[: m * k * 4]).reshape(m, k)
        pos += m * k * 4
        a_idx = _deserialize(data[pos : pos + n_vals * 4]).astype(int)
        pos += n_vals * 4
        a_vals = np.frombuffer(data[pos : pos + n_vals * 2], dtype=np.float16).astype(
            np.float64
        )

        A = np.zeros((k, n), dtype=np.float64)
        A.ravel()[a_idx[a_idx < k * n]] = a_vals[: np.sum(a_idx < k * n)]
        return (D @ A).astype(np.float32)
