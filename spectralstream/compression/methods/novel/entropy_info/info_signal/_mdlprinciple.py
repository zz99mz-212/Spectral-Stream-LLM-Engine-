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

class MDLPrinciple:
    """F2: L(W) = L(model) + L(residual), two-part MDL codes."""

    name = "mdl_principle"
    category = "novel_info"

    def compress(self, tensor: np.ndarray, bits: int = 6) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        best_model = 0
        best_cost = float("inf")
        best_params: dict = {}

        # Model 0: simple mean
        mu = float(t.mean())
        resid = t - mu
        cost = 32 + resid.nbytes * (bits / 64)
        if cost < best_cost:
            best_cost = cost
            best_model = 0
            best_params = {"mu": mu}

        # Model 1: SVD rank-1
        if t.ndim >= 2 and min(t.shape) > 1:
            U, S, Vt = np.linalg.svd(t.reshape(t.shape[0], -1), full_matrices=False)
            s1 = float(S[0])
            u1 = U[:, 0].astype(np.float16)
            v1 = Vt[0, :].astype(np.float16)
            recon = s1 * np.outer(u1, v1)
            resid = t - recon
            cost = 32 + u1.nbytes + v1.nbytes + resid.nbytes * (bits / 64)
            if cost < best_cost:
                best_cost = cost
                best_model = 1
                best_params = {"s1": s1, "u1": _ser(u1), "v1": _ser(v1)}

        q, scale, lo = _quantize(resid, bits)
        meta = dict(shape=tensor.shape, model=best_model, scale=scale, lo=lo, bits=bits)
        meta.update(best_params)
        data = _ser(q)
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        model = metadata["model"]
        scale = metadata["scale"]
        lo = metadata["lo"]
        bits = metadata["bits"]
        n = int(np.prod(shape))
        nq = n
        q = _deser(data[:nq], np.uint8)
        resid = _dequantize(q, scale, lo)
        if model == 0:
            return (resid + metadata["mu"]).reshape(shape).astype(np.float32)
        u1 = _deser(metadata["u1"], np.float16)
        v1 = _deser(metadata["v1"], np.float16)
        s1 = metadata["s1"]
        recon = s1 * np.outer(u1, v1)
        return (recon + resid.reshape(shape)).astype(np.float32)
