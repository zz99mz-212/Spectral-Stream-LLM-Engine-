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

class MatchedField:
    """G5: B(θ) = |w^H w_model(θ)|²/||w||²||w_model||², replica correlation."""

    name = "matched_field"
    category = "novel_signal"

    def compress(self, tensor: np.ndarray, n_replicas: int = 16) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        flat = t.ravel()
        n = len(flat)
        rng = np.random.RandomState(42)
        models = rng.randn(n_replicas, n).astype(np.float64)
        models /= np.linalg.norm(models, axis=1, keepdims=True) + 1e-30
        corrs = models @ flat
        best_k = min(n_replicas, max(1, n_replicas // 4))
        best_idx = np.argsort(-np.abs(corrs))[:best_k]
        best_models = models[best_idx].astype(np.float16)
        best_corrs = corrs[best_idx].astype(np.float16)
        meta = dict(
            shape=tensor.shape, n=n, n_replicas=n_replicas, best_k=best_k, seed=42
        )
        data = _ser(best_models) + _ser(best_corrs)
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        n = metadata["n"]
        bk = metadata["best_k"]
        models = _deser(data[: bk * n * 2], np.float16).reshape(bk, n)
        corrs = _deser(data[bk * n * 2 :], np.float16)
        recon = np.zeros(n, dtype=np.float32)
        for i in range(bk):
            recon += float(corrs[i]) * models[i]
        return recon.reshape(shape)
