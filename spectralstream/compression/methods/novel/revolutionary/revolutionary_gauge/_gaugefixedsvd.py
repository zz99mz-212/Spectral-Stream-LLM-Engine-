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

class GaugeFixedSVD:
    name = "gauge_fixed_svd"
    category = "revolutionary_gauge"

    def compress(self, tensor, rank=0, **params):
        t = tensor.astype(np.float64)
        shape = t.shape
        if t.ndim < 2:
            t_2d = t.reshape(1, -1)
        else:
            t_2d = t.reshape(t.shape[0], -1)
        m, n = t_2d.shape
        k = rank if rank > 0 else max(1, min(m, n) // 4)
        k = min(k, m, n)
        U, S, Vt = np.linalg.svd(t_2d, full_matrices=False)
        k = min(k, len(S))
        # Gauge fix: remove rotational redundancy by making U have positive first column
        for i in range(k):
            if U[0, i] < 0:
                U[:, i] *= -1
                Vt[i] *= -1
        data = (
            struct.pack("<III", m, n, k)
            + U[:, :k].astype(np.float32).tobytes()
            + S[:k].astype(np.float32).tobytes()
            + Vt[:k, :].astype(np.float32).tobytes()
        )
        return data, {
            "_svd": True,
            "shape": shape,
            "m": m,
            "n": n,
            "k": k,
            "gauge_fixed": True,
        }

    def decompress(self, data, metadata):
        return _svd_decompress(data, metadata)
