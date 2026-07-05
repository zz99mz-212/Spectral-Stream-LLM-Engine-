from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class NormalForm:
    """Normal form: dw_c/dt = f_c(w_c), truncate resonant terms."""

    name = "normal_form"
    category = "novel_chaos"

    def compress(self, tensor: np.ndarray, n_resonant: int = 8) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        flat = t.ravel()
        n = len(flat)

        hist, edges = np.histogram(flat, bins=32, density=True)
        centers = (edges[:-1] + edges[1:]) * 0.5
        pdf = hist + 1e-30

        poly_order = min(n_resonant, 5)
        x = centers
        A = np.vander(x, poly_order + 1, increasing=True)
        coeffs = np.linalg.lstsq(A, pdf, rcond=None)[0]

        poly_vals = A @ coeffs
        residual = pdf - poly_vals
        r2 = float(1 - np.sum(residual**2) / max(np.sum(pdf**2), 1e-30))

        U, S, Vt = np.linalg.svd(
            t.reshape(-1, n) if t.ndim < 2 else t, full_matrices=False
        )
        k = min(n_resonant, len(S))

        meta = dict(shape=tensor.shape, k=k, poly_order=poly_order, r2=r2)
        data = _serialize(coeffs.astype(np.float32))
        data += _serialize(centers.astype(np.float32))
        data += _serialize(U[:, :k].astype(np.float32))
        data += _serialize(S[:k].astype(np.float32))
        data += _serialize(Vt[:k, :].astype(np.float32))
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        k = metadata["k"]
        poly_order = metadata["poly_order"]
        m, n = shape

        pos = 0
        coeffs = _deserialize(data[: (poly_order + 1) * 4])
        pos += (poly_order + 1) * 4
        centers = _deserialize(data[pos : pos + 32 * 4])
        pos += 32 * 4

        U = _deserialize(data[pos : pos + m * k * 4]).reshape(m, k)
        pos += m * k * 4
        S = _deserialize(data[pos : pos + k * 4])
        pos += k * 4
        Vt = _deserialize(data[pos : pos + k * n * 4]).reshape(k, n)

        return ((U * S) @ Vt).reshape(shape).astype(np.float32)
