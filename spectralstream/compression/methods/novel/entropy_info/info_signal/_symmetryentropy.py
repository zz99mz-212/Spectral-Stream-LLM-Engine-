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

class SymmetryEntropy:
    """F10: H_G(W) = H(W) - I(W;G), symmetry group exploitation."""

    name = "symmetry_entropy"
    category = "novel_info"

    def compress(
        self, tensor: np.ndarray, sym_threshold: float = 0.01
    ) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        if t.ndim < 2:
            t = t.reshape(1, -1)
        # detect row symmetry groups
        groups: List[int] = []
        unique_rows_map: Dict[int, int] = {}
        for i in range(t.shape[0]):
            row = t[i]
            matched = False
            for j, ur in unique_rows_map.items():
                if np.max(np.abs(row - t[j])) < sym_threshold:
                    groups.append(ur)
                    matched = True
                    break
            if not matched:
                uid = len(unique_rows_map)
                unique_rows_map[i] = uid
                groups.append(uid)
        n_unique = len(unique_rows_map)
        # store unique rows + group mapping
        unique_rows = np.array(
            [t[list(unique_rows_map.keys())[i]] for i in range(n_unique)],
            dtype=np.float16,
        )
        groups_arr = np.array(groups, dtype=np.int32)
        data = _ser(unique_rows) + _ser(groups_arr)
        meta = dict(shape=tensor.shape, n_unique=n_unique, threshold=sym_threshold)
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        nu = metadata["n_unique"]
        row_len = shape[1] if len(shape) >= 2 else 1
        unique_bytes = nu * row_len * 2
        unique_rows = _deser(data[:unique_bytes], np.float16).reshape(nu, row_len)
        groups = _deser(data[unique_bytes:], np.int32)
        return unique_rows[groups].astype(np.float32)
