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

class DOFShrinkage:
    """F16: SURE = -nσ² + ||ŵ-w||² + 2σ²df(ŵ), Stein's unbiased risk estimation."""

    name = "dof_shrinkage"
    category = "novel_info"

    def compress(
        self, tensor: np.ndarray, threshold_factor: float = 1.0
    ) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        flat = t.ravel()
        n = len(flat)
        sigma2 = max(float(np.var(flat)), 1e-30)
        sigma = math.sqrt(sigma2)
        # SURE soft-threshold: df = #nonzeros after shrink
        thr = sigma * threshold_factor * math.sqrt(2 * math.log(n))
        soft = np.sign(flat) * np.maximum(np.abs(flat) - thr, 0)
        dof = float(np.sum(np.abs(soft) > 0))
        sure = -n * sigma2 + np.sum((soft - flat) ** 2) + 2 * sigma2 * dof
        idx = np.argwhere(np.abs(soft) > 0).ravel().astype(np.int32)
        vals = soft[soft != 0].astype(np.float16)
        meta = dict(shape=tensor.shape, n=n, sigma2=sigma2, sure=float(sure), dof=dof)
        data = _ser(idx) + vals.tobytes()
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        n = metadata["n"]
        n_idx = (len(data) - 4) // 6
        idx = _deser(data[: n_idx * 4], np.int32).ravel()
        vals = np.frombuffer(data[n_idx * 4 :], dtype=np.float16)
        recon = np.zeros(n, dtype=np.float32)
        for i, v in zip(idx, vals):
            if i < n:
                recon[i] = float(v)
        return recon.reshape(shape)
