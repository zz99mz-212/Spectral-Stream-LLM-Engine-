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

class KalmanFilter:
    """G1: State-space Kalman filter — store gains + innovations."""

    name = "kalman_filter"
    category = "novel_signal"

    def compress(self, tensor: np.ndarray, n_steps: int = 16) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        if t.ndim < 2:
            t = t.reshape(-1, 1)
        m, n = t.shape
        F = np.eye(n) * 0.99
        H = np.eye(n) * 0.5
        Q = np.eye(n) * 0.01
        R = np.eye(n) * 0.1
        x = np.zeros(n)
        P = np.eye(n)
        innovations: List[np.ndarray] = []
        gains: List[np.ndarray] = []
        for i in range(min(m, n_steps)):
            x = F @ x
            P = F @ P @ F.T + Q
            K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)
            inn = t[i] - H @ x
            x = x + K @ inn
            P = (np.eye(n) - K @ H) @ P
            innovations.append(inn)
            gains.append(K)
        in_all = np.stack(innovations).astype(np.float16)
        g_all = np.stack(gains).astype(np.float16)
        data = _ser(F.astype(np.float16)) + _ser(H.astype(np.float16))
        data += _ser(in_all) + _ser(g_all)
        meta = dict(shape=tensor.shape, n_steps=n_steps, m=m, n=n)
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        n_steps = metadata["n_steps"]
        m, n = metadata["m"], metadata["n"]
        sz = n * n * 2
        F = _deser(data[:sz], np.float16).reshape(n, n)
        H = _deser(data[sz : 2 * sz], np.float16).reshape(n, n)
        in_sz = n_steps * n * 2
        off = 2 * sz
        in_all = _deser(data[off : off + in_sz], np.float16).reshape(n_steps, n)
        off += in_sz
        g_all = _deser(data[off:], np.float16).reshape(n_steps, n, n)
        recon = np.zeros((m, n), dtype=np.float32)
        x = np.zeros(n, dtype=np.float64)
        for i in range(m):
            if i < n_steps:
                x = F @ x + g_all[i] @ in_all[i]
            recon[i] = x
        return recon.reshape(shape)
