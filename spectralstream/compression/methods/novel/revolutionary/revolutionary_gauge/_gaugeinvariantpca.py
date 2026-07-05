from __future__ import annotations

import math
import struct
from typing import Any, Dict, Tuple

import numpy as np

from spectralstream.compression.methods.novel._common import (
    _block_int8_fallback,
    _block_int8_decompress,
    _svd_compress,
    _svd_decompress,
)


def _gauge_compress(tensor, method="svd", rank=0):
    if method == "svd":
        return _svd_compress(tensor, rank)
    return _block_int8_fallback(tensor)

def _gauge_decompress(data, meta):
    if meta.get("_svd"):
        return _svd_decompress(data, meta)
    return _block_int8_decompress(data, meta)

class GaugeInvariantPCA:
    name = "gauge_invariant_pca"
    category = "revolutionary_gauge"

    def compress(self, tensor, n_components=0, **params):
        t = tensor.astype(np.float64)
        if t.ndim < 2:
            t_2d = t.reshape(1, -1)
        else:
            t_2d = t.reshape(t.shape[0], -1)
        m, n = t_2d.shape
        mean = np.mean(t_2d, axis=0)
        centered = t_2d - mean
        cov = centered.T @ centered / m
        eigvals, eigvecs = np.linalg.eigh(cov)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        k = n_components if n_components > 0 else max(1, min(m, n) // 4)
        k = min(k, n)
        projected = centered @ eigvecs[:, :k]
        data = (
            struct.pack("<III", m, n, k)
            + mean.astype(np.float32).tobytes()
            + eigvecs[:, :k].astype(np.float32).tobytes()
            + projected.astype(np.float32).tobytes()
        )
        return data, {"_pca": True, "shape": tensor.shape, "m": m, "n": n, "k": k}

    def decompress(self, data, metadata):
        m, n, k = struct.unpack_from("<III", data, 0)
        pos = 12
        mean = np.frombuffer(data[pos : pos + n * 4], dtype=np.float32)
        pos += n * 4
        eigvecs = np.frombuffer(data[pos : pos + n * k * 4], dtype=np.float32).reshape(
            n, k
        )
        pos += n * k * 4
        projected = np.frombuffer(
            data[pos : pos + m * k * 4], dtype=np.float32
        ).reshape(m, k)
        return (
            (projected @ eigvecs.T + mean).reshape(metadata["shape"]).astype(np.float32)
        )
