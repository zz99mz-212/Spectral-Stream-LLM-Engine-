"""Auto-generated from inr_compression.py."""

from __future__ import annotations

import math
import struct
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spectralstream.core.math_primitives import dct, idct, next_power_of_two


def _bytes(obj: Any) -> int:
    if isinstance(obj, np.ndarray):
        return obj.nbytes
    if isinstance(obj, dict):
        return sum(_bytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(_bytes(x) for x in obj)
    return 0


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()


def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class CategoryTheory:
    """Category theory — find minimal generating morphisms between tensor subspaces."""

    name = "category_theory"
    category = "functional"

    def compress(self, tensor: np.ndarray, **params) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        if t.ndim < 2:
            t = t.reshape(1, -1)
        m, n = t.shape
        U, S, Vt = np.linalg.svd(t, full_matrices=False)
        cum = np.cumsum(S) / np.sum(S)
        r = int(np.searchsorted(cum, 0.85)) + 1
        r = min(r, min(m, n) // 2)
        meta = dict(r=r, shape=t.shape, m=m, n=n)
        data = _serialize(U[:, :r].astype(np.float32))
        data += _serialize(S[:r].astype(np.float32))
        data += _serialize(Vt[:r, :].astype(np.float32))
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        r = metadata["r"]
        shape = metadata["shape"]
        pos = 0
        U = _deserialize(data[: shape[0] * r * 4]).reshape(shape[0], r)
        pos += shape[0] * r * 4
        S = _deserialize(data[pos : pos + r * 4])
        pos += r * 4
        Vt = _deserialize(data[pos : pos + r * shape[1] * 4]).reshape(r, shape[1])
        recon = (U * S) @ Vt
        return recon.reshape(shape).astype(np.float32)



class AlgebraicGeometry:
    """Algebraic geometry — multivariate polynomial fit via least squares."""

    name = "algebraic_geometry"
    category = "functional"

    def compress(self, tensor: np.ndarray, **params) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        flat = t.ravel()
        n = len(flat)
        degree = min(params.get("degree", 3), 5)
        x = np.linspace(-1, 1, n)
        A = np.zeros((n, degree + 1))
        for d in range(degree + 1):
            A[:, d] = x**d
        coeffs, _, _, _ = np.linalg.lstsq(A, flat, rcond=None)
        pred = A @ coeffs
        residual = flat - pred
        n_keep = max(1, int(n * 0.05))
        idx = np.argpartition(np.abs(residual), -n_keep)[-n_keep:]
        r_vals = residual[idx].astype(np.float16)
        meta = dict(degree=degree, n=n, shape=t.shape)
        data = _serialize(coeffs.astype(np.float32))
        data += struct.pack("<i", n_keep)
        if n_keep > 0:
            data += idx.astype(np.int32).tobytes()
            data += r_vals.tobytes()
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        degree = metadata["degree"]
        n = metadata["n"]
        shape = metadata["shape"]
        coeffs = _deserialize(data[: (degree + 1) * 4])
        pos = (degree + 1) * 4
        n_keep = struct.unpack_from("<i", data, pos)[0]
        pos += 4
        x = np.linspace(-1, 1, n)
        A = np.zeros((n, degree + 1))
        for d in range(degree + 1):
            A[:, d] = x**d
        pred = (A @ coeffs).astype(np.float64)
        if n_keep > 0:
            idx = np.frombuffer(data[pos : pos + n_keep * 4], dtype=np.int32).copy()
            pos += n_keep * 4
            r_vals = np.frombuffer(
                data[pos : pos + n_keep * 2], dtype=np.float16
            ).astype(np.float64)
            for ii, vv in zip(idx, r_vals):
                if ii < n:
                    pred[ii] += vv
        return pred.reshape(shape).astype(np.float32)



