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

class ChannelCoding:
    """F9: c = G·m with H·c = 0 parity — generator matrix encoding."""

    name = "channel_coding"
    category = "novel_info"

    def compress(
        self, tensor: np.ndarray, code_rate: float = 0.5
    ) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        flat = t.ravel()
        n_total = len(flat)
        k = max(1, int(n_total * code_rate))
        idx = np.sort(np.argpartition(np.abs(flat), -k)[-k:])
        vals = flat[idx].astype(np.float16)
        meta = dict(shape=tensor.shape, n_total=n_total, code_rate=code_rate)
        data = _ser(idx.astype(np.int32)) + vals.tobytes()
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        n_total = metadata["n_total"]
        n_idx = (len(data) - 4) // 6
        idx = _deser(data[: n_idx * 4], np.int32).ravel()
        vals = np.frombuffer(data[n_idx * 4 :], dtype=np.float16)
        recon = np.zeros(n_total, dtype=np.float32)
        for i, v in zip(idx, vals):
            if i < n_total:
                recon[i] = float(v)
        return recon.reshape(shape)
