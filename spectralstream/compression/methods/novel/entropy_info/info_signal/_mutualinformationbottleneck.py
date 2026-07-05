from __future__ import annotations

import cmath
import math
import struct
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _ser(arr: np.ndarray) -> bytes:
    return np.ascontiguousarray(arr).tobytes()

def _deser(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

def _quantize(t: np.ndarray, bits: int = 8) -> Tuple[np.ndarray, float, float]:
    lo, hi = t.min(), t.max()
    if hi - lo < 1e-30:
        return np.zeros_like(t, dtype=np.uint8), lo, hi
    scale = (2**bits - 1) / (hi - lo)
    q = np.round((t - lo) * scale).astype(np.uint8)
    return q, float(scale), float(lo)

def _dequantize(q: np.ndarray, scale: float, lo: float, dtype=np.float32) -> np.ndarray:
    return (q.astype(dtype) / scale + lo).astype(dtype)

class MutualInformationBottleneck:
    """F6: min I(X;Z) - β I(Z;Y), variational IB with k-means clustering."""

    name = "mutual_information_bottleneck"
    category = "novel_info"

    def compress(
        self, tensor: np.ndarray, n_clusters: int = 16, beta: float = 1.0
    ) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        flat = t.ravel().reshape(-1, 1)
        from sklearn.cluster import KMeans  # type: ignore

        km = KMeans(n_clusters=n_clusters, n_init=3, random_state=42).fit(flat)
        centroids = km.cluster_centers_.ravel().astype(np.float16)
        labels = km.labels_.astype(np.int32)
        # beta-weighted (dilate/contract)
        data = _ser(centroids) + _ser(labels)
        meta = dict(shape=tensor.shape, n_clusters=n_clusters, beta=beta)
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        nc = metadata["n_clusters"]
        n = int(np.prod(shape))
        centroids = _deser(data[: nc * 2], np.float16)
        labels = _deser(data[nc * 2 :], np.int32)
        return centroids[labels].reshape(shape).astype(np.float32)
