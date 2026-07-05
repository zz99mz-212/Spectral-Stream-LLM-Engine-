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

class NonlinearIdentification:
    """G10: Wiener/Hammerstein block-oriented nonlinear models."""

    name = "nonlinear_identification"
    category = "novel_signal"

    def compress(
        self, tensor: np.ndarray, model_type: str = "wiener", n_hidden: int = 8
    ) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        if t.ndim < 2:
            t = t.reshape(1, -1)
        m, n = t.shape
        u = np.random.RandomState(42).randn(n)
        if model_type == "wiener":
            H = np.random.RandomState(0).randn(m, n_hidden)
            z_all = H @ np.random.RandomState(1).randn(n_hidden)
            g_all_params = np.zeros((n, 4), dtype=np.float16)
            for col in range(n):
                g_all_params[col] = np.polyfit(z_all, t[:, col], deg=3).astype(
                    np.float16
                )
            meta = dict(
                shape=tensor.shape,
                model_type=model_type,
                n_hidden=n_hidden,
                H=_ser(H.astype(np.float16)),
            )
            data = _ser(g_all_params)
        else:
            # Hammerstein: w = H·g(u)
            g_params = np.polyfit(u, t.ravel(), deg=3).astype(np.float16)
            H = np.random.RandomState(0).randn(m, n).astype(np.float16)
            meta = dict(
                shape=tensor.shape, model_type=model_type, n_hidden=n_hidden, H=_ser(H)
            )
            data = _ser(g_params)
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        model_type = metadata["model_type"]
        m, n = shape if len(shape) >= 2 else (1, shape[0])
        nh = metadata["n_hidden"]
        if model_type == "wiener":
            H = _deser(metadata["H"], np.float16).reshape(m, nh)
            g_all = _deser(data, np.float16).reshape(n, 4)
            z = H @ np.random.RandomState(1).randn(nh)
            recon = np.zeros((m, n), dtype=np.float32)
            for col in range(n):
                recon[:, col] = np.polyval(g_all[col], z)
        else:
            u = np.random.RandomState(42).randn(n)
            H = _deser(metadata["H"], np.float16).reshape(m, n)
            g_params = _deser(data, np.float16)
            g_u = np.polyval(g_params, u)
            recon = H @ g_u
        return recon.reshape(shape).astype(np.float32)
