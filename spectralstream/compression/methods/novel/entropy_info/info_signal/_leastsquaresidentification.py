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

class LeastSquaresIdentification:
    """G9: ARX system identification — w_t + a_1w_{t-1}+... = b_1u_{t-1}+..."""

    name = "least_squares_identification"
    category = "novel_signal"

    def compress(self, tensor: np.ndarray, order: int = 4) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        flat = t.ravel()
        n = len(flat)
        if n <= order:
            meta = dict(shape=tensor.shape, order=order, n=n, direct=True)
            data = _ser(t.astype(np.float16))
            return data, meta
        # Build ARX regression
        Y = flat[order:]
        A_mat = np.zeros((n - order, 2 * order))
        for k in range(order):
            A_mat[:, k] = -flat[order - 1 - k : n - 1 - k]
        u = np.random.RandomState(42).randn(n)
        for k in range(order):
            A_mat[:, order + k] = u[order - 1 - k : n - 1 - k]
        theta, _, _, _ = np.linalg.lstsq(A_mat, Y, rcond=None)
        residual = Y - A_mat @ theta
        q, scale, lo = _quantize(residual, bits=6)
        meta = dict(
            shape=tensor.shape,
            order=order,
            n=n,
            theta=_ser(theta.astype(np.float16)),
            scale=scale,
            lo=lo,
        )
        data = _ser(q)
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        order = metadata["order"]
        n = metadata["n"]
        if metadata.get("direct"):
            return _deser(data, np.float16).reshape(shape)
        theta = _deser(metadata["theta"], np.float16)
        scale = metadata["scale"]
        lo = metadata["lo"]
        q = _deser(data[: (n - order)], np.uint8)
        residual = _dequantize(q, scale, lo)
        u = np.random.RandomState(42).randn(n)
        recon = np.zeros(n, dtype=np.float32)
        for i in range(n):
            if i < order:
                recon[i] = 0.0
            else:
                pred = 0.0
                for k in range(order):
                    pred += -theta[k] * recon[i - 1 - k]
                    pred += theta[order + k] * u[i - 1 - k]
                recon[i] = pred + float(
                    residual[i - order] if (i - order) < len(residual) else 0.0
                )
        return recon.reshape(shape).astype(np.float32)
