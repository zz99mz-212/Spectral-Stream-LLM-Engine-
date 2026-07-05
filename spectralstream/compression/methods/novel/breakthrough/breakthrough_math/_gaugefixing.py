from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class GaugeFixing:
    """Gauge fixing: remove ALL gauge redundancy from the weight space.
    Weight matrices have gauge symmetries: W → P@W@Q^{-1} for
    invertible P, Q. Fix the gauge by imposing conditions that
    uniquely determine the representation (e.g., row-echelon form
    or SVD normalization). Store only the gauge-invariant
    observables (singular values + gauge-fixed vectors).

    Real: SVD with fixed phase convention for U and V.
    Store singular values + gauge-fixed U and V in canonical form.
    """

    name = "gauge_fixing"
    category = "breakthrough_math"

    def compress(self, tensor: np.ndarray, rank: int = 16) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64).reshape(tensor.shape[0], -1)
        m, n = t.shape
        U, S, Vt = np.linalg.svd(t, full_matrices=False)
        k = min(rank, len(S), m, n)
        # Gauge fixing: make first element of each U column positive
        U_k = U[:, :k].copy()
        for i in range(k):
            if U_k[0, i] < 0:
                U_k[:, i] *= -1
                Vt[i, :] *= -1
        # Gauge-fixed: first row of U is now positive
        S_k = S[:k].astype(np.float32)
        U_k = U_k[:, :k].astype(np.float32)
        Vt_k = Vt[:k, :].astype(np.float32)
        buf = struct.pack("<III", m, n, k)
        buf += _serialize(U_k)
        buf += _serialize(S_k)
        buf += _serialize(Vt_k)
        return bytes(buf), {
            "shape": tensor.shape,
            "m": m,
            "n": n,
            "k": k,
        }

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        m, n, k = struct.unpack_from("<III", data, 0)
        pos = 12
        U_k = np.frombuffer(data[pos : pos + m * k * 4], dtype=np.float32).reshape(m, k)
        pos += m * k * 4
        S_k = np.frombuffer(data[pos : pos + k * 4], dtype=np.float32)
        pos += k * 4
        Vt_k = np.frombuffer(data[pos : pos + k * n * 4], dtype=np.float32).reshape(
            k, n
        )
        recon = (U_k * S_k) @ Vt_k
        return recon.astype(np.float32).reshape(metadata["shape"])
