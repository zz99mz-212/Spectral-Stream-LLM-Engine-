from __future__ import annotations

import math
import struct
from typing import Any, Dict, Tuple

import numpy as np

from ..._common import _block_int8_fallback, _block_int8_decompress


def _pack_f32(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()

def _unpack_f32(data: bytes, shape: Tuple[int, ...]) -> np.ndarray:
    return np.frombuffer(data, dtype=np.float32).copy().reshape(shape)

def _pack_int32(arr: np.ndarray) -> bytes:
    return arr.astype(np.int32).tobytes()

def _pack_uint8(arr: np.ndarray) -> bytes:
    return arr.astype(np.uint8).tobytes()

def _mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))

def _sparsify_topk(
    flat: np.ndarray, k: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = np.argsort(-np.abs(flat))[:k]
    kept = flat[idx]
    residual = flat.copy()
    residual[idx] = 0.0
    return kept, idx.astype(np.int32), residual

def _reconstruct_from_topk(kept: np.ndarray, idx: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros(n, dtype=np.float32)
    m = min(len(kept), len(idx))
    for i in range(m):
        if 0 <= idx[i] < n:
            out[int(idx[i])] = kept[i]
    return out

class HRRNormalized:
    """D16: HRR-NORMALIZED — Unit-norm holographic vector encoding."""

    name = "hrr_normalized"
    category = "fractal_holographic"

    def compress(self, tensor: np.ndarray, **params) -> Tuple[bytes, dict]:
        t = tensor.ravel().astype(np.float32)
        n = len(t)
        k = params.get("k", max(n // 8, 4))
        if k >= n:
            return _block_int8_fallback(tensor)
        kept, idx, _ = _sparsify_topk(t, k)
        norm = float(np.linalg.norm(kept))
        nrm = kept / (norm + 1e-10)
        return struct.pack("<IIf", n, k, norm) + _pack_f32(nrm) + _pack_int32(idx), {
            "shape": tensor.shape,
            "n": n,
            "k": k,
        }

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        n, k = struct.unpack_from("<II", data, 0)
        norm = struct.unpack_from("<f", data, 8)[0]
        nrm = np.frombuffer(data[12 : 12 + k * 4], dtype=np.float32)
        idx = np.frombuffer(data[12 + k * 4 : 12 + k * 8], dtype=np.int32)
        return _reconstruct_from_topk(nrm * norm, idx, n).reshape(metadata["shape"])
