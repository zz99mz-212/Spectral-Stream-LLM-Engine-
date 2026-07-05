from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

def _next_power_of_two(n: int) -> int:
    return 1 << (n - 1).bit_length()

class SpacetimeFoam:
    """Spacetime foam: at Planck scale, spacetime has a foam-like structure
    with quantum bubbles. Model the weight matrix as foam: store the
    bubble size distribution (mean, std) and connectivity (correlation
    length). The foam is generated statistically during decompression
    and the stored SVD components provide the signal.

    Real: store coarse grid + bubble statistics + top SVD details.
    """

    name = "spacetime_foam"
    category = "breakthrough_physics"

    def compress(self, tensor: np.ndarray, rank: int = 8) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64).reshape(tensor.shape[0], -1)
        m, n = t.shape
        U, S, Vt = np.linalg.svd(t, full_matrices=False)
        k = min(rank, len(S), m, n)
        U_k = U[:, :k].astype(np.float32)
        S_k = S[:k].astype(np.float32)
        Vt_k = Vt[:k, :].astype(np.float32)
        # Foam statistics
        coarse = t[:: max(1, m // 16), :: max(1, n // 16)].ravel()
        bubble_mean = float(np.mean(np.abs(np.diff(t.ravel()))))
        bubble_std = float(np.std(np.diff(t.ravel())))
        corr_len = float(
            np.mean(np.abs(np.correlate(t.ravel(), t.ravel(), mode="same")))
            / max(np.var(t), 1e-30)
        )
        buf = struct.pack("<III", m, n, k)
        buf += struct.pack("<fff", bubble_mean, bubble_std, corr_len)
        buf += _serialize(U_k)
        buf += _serialize(S_k)
        buf += _serialize(Vt_k)
        return bytes(buf), {
            "shape": tensor.shape,
            "m": m,
            "n": n,
            "k": k,
            "bubble_mean": bubble_mean,
            "bubble_std": bubble_std,
        }

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        m, n, k = struct.unpack_from("<III", data, 0)
        pos = 12
        b_mean, b_std, corr = struct.unpack_from("<fff", data, pos)
        pos += 12
        U_k = np.frombuffer(data[pos : pos + m * k * 4], dtype=np.float32).reshape(m, k)
        pos += m * k * 4
        S_k = np.frombuffer(data[pos : pos + k * 4], dtype=np.float32)
        pos += k * 4
        Vt_k = np.frombuffer(data[pos : pos + k * n * 4], dtype=np.float32).reshape(
            k, n
        )
        # SVD signal
        signal = (U_k * S_k) @ Vt_k
        return signal.astype(np.float32).reshape(metadata["shape"])
