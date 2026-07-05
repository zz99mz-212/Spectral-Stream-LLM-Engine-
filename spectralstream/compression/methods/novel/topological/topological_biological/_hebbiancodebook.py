from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class HebbianCodebook:
    """D4. HEBBIAN-CODEBOOK: Oja's rule Δw_j = η y_j(x - y_j w_j), online PCA."""

    name = "hebbian_codebook"
    category = "novel_biological"

    def compress(
        self, tensor: np.ndarray, n_components: int = 8, n_iter: int = 100
    ) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        m, n = t.shape
        k = min(n_components, m, n)

        w = np.random.RandomState(0).randn(k, n) * 0.01
        w = w / (np.linalg.norm(w, axis=1, keepdims=True) + 1e-30)
        eta = 0.001

        for _ in range(n_iter):
            for i in range(m):
                x = t[i, :]
                y = w @ x
                dw = eta * y[:, None] * (x[None, :] - y[:, None] * w)
                w += dw
                w = w / (np.linalg.norm(w, axis=1, keepdims=True) + 1e-30)

        hebbian_components = w.astype(np.float32)
        scores = (t.astype(np.float32) @ hebbian_components.T).astype(np.float16)

        meta = dict(shape=t.shape, k=k)
        data = _serialize(hebbian_components) + scores.tobytes()
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        k = metadata["k"]
        m, n = shape

        components = _deserialize(data[: k * n * 4]).reshape(k, n)
        scores = (
            np.frombuffer(data[k * n * 4 :], dtype=np.float16)
            .astype(np.float64)
            .reshape(m, k)
        )

        return (scores @ components).astype(np.float32)
