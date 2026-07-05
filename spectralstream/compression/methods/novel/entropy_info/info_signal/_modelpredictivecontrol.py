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

class ModelPredictiveControl:
    """G2: MPC — control trajectory over horizon H."""

    name = "model_predictive_control"
    category = "novel_signal"

    def compress(
        self, tensor: np.ndarray, horizon: int = 8, bits: int = 6
    ) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        flat = t.ravel()
        n = len(flat)
        ref = float(np.mean(flat))
        traj = flat[: min(n, horizon)].astype(np.float16)
        residual = flat[horizon:] if n > horizon else np.array([], dtype=np.float64)
        if len(residual) > 0:
            q, scale, lo = _quantize(residual, bits)
        else:
            q = np.array([], dtype=np.uint8)
            scale, lo = 1.0, 0.0
        meta = dict(
            shape=tensor.shape,
            horizon=horizon,
            ref=ref,
            scale=scale,
            lo=lo,
            bits=bits,
            n_total=n,
        )
        data = traj.tobytes() + _ser(q)
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        horizon = metadata["horizon"]
        n_total = metadata["n_total"]
        scale = metadata["scale"]
        lo = metadata["lo"]
        traj = _deser(data[: horizon * 2], np.float16)
        residual = _dequantize(_deser(data[horizon * 2 :], np.uint8), scale, lo)
        n_res = n_total - horizon
        if n_res > 0 and len(residual) >= n_res:
            traj_full = np.concatenate([traj, residual[:n_res]])
        else:
            traj_full = np.pad(traj, (0, max(0, n_total - horizon)), mode="edge")
        return traj_full[:n_total].reshape(shape).astype(np.float32)
