"""Auto-generated from _class_wrappers.py."""

from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()


def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()


def _ensure_2d(tensor: np.ndarray) -> tuple:
    orig_ndim = tensor.ndim
    orig_shape = tensor.shape
    t = tensor.astype(np.float64)
    if t.ndim < 2:
        t = t.reshape(1, -1)
    elif t.ndim > 2:
        t = t.reshape(t.shape[0], -1)
    return t, orig_shape, orig_ndim


class MHDCompression:
    """MHD wave decomposition: DCT domain thresholding."""

    name = "mhd_compression"
    category = "physics"

    def compress(
        self, tensor: np.ndarray, keep_frac: float = 0.5
    ) -> Tuple[bytes, dict]:
        from spectralstream.core.math_primitives import dct

        t, orig_shape, orig_ndim = _ensure_2d(tensor)
        m, n = t.shape
        coeffs = dct(dct(t).T).T
        flat = coeffs.ravel()
        k = max(1, int(keep_frac * flat.size))
        idx = np.argpartition(np.abs(flat), -k)[-k:]
        idx.sort()
        kept = flat[idx]
        meta = dict(shape=orig_shape, ndim=orig_ndim, m=m, n=n, total=len(flat))
        data = _serialize(idx.astype(np.int32)) + kept.astype(np.float16).tobytes()
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        from spectralstream.core.math_primitives import idct

        shape = metadata["shape"]
        ndim = metadata.get("ndim", len(shape))
        total = metadata["total"]
        bytes_per_entry = 6
        max_entries = len(data) // bytes_per_entry
        k = max_entries
        if k <= 0:
            return np.zeros(shape, dtype=np.float32)
        idx = _deserialize(data[: k * 4]).astype(int)
        vals = np.frombuffer(data[k * 4 :], dtype=np.float16).astype(np.float64)
        coeffs = np.zeros(total, dtype=np.float64)
        for i, v in zip(idx, vals):
            if i < total:
                coeffs[i] = v
        m = metadata["m"]
        n_val = metadata["n"]
        c2d = coeffs.reshape(m, n_val)
        result = idct(idct(c2d).T).T
        return result.reshape(shape).astype(np.float32)


class Gyrokinetic:
    """Gyrokinetic: Energy-based SVD compression."""

    name = "gyrokinetic"
    category = "physics"

    def compress(self, tensor: np.ndarray, rank: int = None) -> Tuple[bytes, dict]:
        from spectralstream.compression.methods.physics.quantum import DensityMatrix

        return DensityMatrix().compress(tensor, rank)

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        from spectralstream.compression.methods.physics.quantum import DensityMatrix

        return DensityMatrix().decompress(data, metadata)
