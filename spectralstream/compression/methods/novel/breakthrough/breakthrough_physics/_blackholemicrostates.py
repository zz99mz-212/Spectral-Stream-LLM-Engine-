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

class BlackHoleMicrostates:
    """Black hole microstates: the weight matrix is a microstate of a
    black hole with mass M (mean), charge Q (variance), and spin J
    (skewness). The microstate counting formula gives the number of
    degrees of freedom = exp(S_BH) = exp(π sqrt(M^2 - Q^2 + J^2)).

    Store: BH parameters + top SVD components (the 'hair').
    """

    name = "black_hole_microstates"
    category = "breakthrough_physics"

    def compress(self, tensor: np.ndarray, rank: int = 16) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64).reshape(tensor.shape[0], -1)
        m, n = t.shape
        U, S, Vt = np.linalg.svd(t, full_matrices=False)
        k = min(rank, len(S), m, n)
        # Black hole parameters
        M = float(np.mean(np.abs(t)))
        Q = float(np.std(t))
        flat = t.ravel()
        J = float(np.mean(((flat - np.mean(flat)) / max(np.std(flat), 1e-10)) ** 3))
        U_k = U[:, :k].astype(np.float32)
        S_k = S[:k].astype(np.float32)
        Vt_k = Vt[:k, :].astype(np.float32)
        buf = struct.pack("<III", m, n, k)
        buf += struct.pack("<fff", M, Q, J)
        buf += _serialize(U_k)
        buf += _serialize(S_k)
        buf += _serialize(Vt_k)
        return bytes(buf), {
            "shape": tensor.shape,
            "m": m,
            "n": n,
            "k": k,
            "mass": M,
            "charge": Q,
            "spin": J,
        }

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        m, n, k = struct.unpack_from("<III", data, 0)
        pos = 12
        M, Q, J = struct.unpack_from("<fff", data, pos)
        pos += 12
        U_k = np.frombuffer(data[pos : pos + m * k * 4], dtype=np.float32).reshape(m, k)
        pos += m * k * 4
        S_k = np.frombuffer(data[pos : pos + k * 4], dtype=np.float32)
        pos += k * 4
        Vt_k = np.frombuffer(data[pos : pos + k * n * 4], dtype=np.float32).reshape(
            k, n
        )
        # Microstate reconstruction: SVD + BH scaling
        recon = (U_k * (S_k * M / max(np.mean(S_k), 1e-10))) @ Vt_k
        return recon.astype(np.float32).reshape(metadata["shape"])
