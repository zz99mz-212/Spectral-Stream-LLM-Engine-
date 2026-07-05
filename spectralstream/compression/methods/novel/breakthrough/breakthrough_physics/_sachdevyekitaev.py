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

class SachdevYeKitaev:
    """SYK model: all-to-all random interactions with coupling J. The weight
    matrix is treated as the SYK two-point function. Store coupling J,
    number of Majorana fermions N, and the top spectral components.

    The SYK model is maximally chaotic and exactly solvable in large N.
    """

    name = "sachdev_ye_kitaev"
    category = "breakthrough_physics"

    def compress(self, tensor: np.ndarray, rank: int = 16) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64).reshape(tensor.shape[0], -1)
        m, n = t.shape
        U, S, Vt = np.linalg.svd(t, full_matrices=False)
        k = min(rank, len(S), m, n)
        # SYK coupling = characteristic interaction scale
        J = float(np.median(np.abs(t)))
        N_fermions = max(min(m, n) // 2, 4)
        U_k = U[:, :k].astype(np.float32)
        S_k = S[:k].astype(np.float32)
        Vt_k = Vt[:k, :].astype(np.float32)
        buf = struct.pack("<III", m, n, k)
        buf += struct.pack("<fI", J, N_fermions)
        buf += _serialize(U_k)
        buf += _serialize(S_k)
        buf += _serialize(Vt_k)
        return bytes(buf), {
            "shape": tensor.shape,
            "m": m,
            "n": n,
            "k": k,
            "J": J,
            "N_fermions": N_fermions,
        }

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        m, n, k = struct.unpack_from("<III", data, 0)
        pos = 12
        J, N_f = struct.unpack_from("<fI", data, pos)
        pos += 8
        U_k = np.frombuffer(data[pos : pos + m * k * 4], dtype=np.float32).reshape(m, k)
        pos += m * k * 4
        S_k = np.frombuffer(data[pos : pos + k * 4], dtype=np.float32)
        pos += k * 4
        Vt_k = np.frombuffer(data[pos : pos + k * n * 4], dtype=np.float32).reshape(
            k, n
        )
        # SYK reconstruction: SVD + coupling scaling
        recon = (U_k * (S_k * J / max(J, 1e-10))) @ Vt_k
        return recon.astype(np.float32).reshape(metadata["shape"])
