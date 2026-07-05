from __future__ import annotations

import math
import struct
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class TeleportationEncode:
    """Inspired by quantum teleportation: transmit weight state via
    Bell measurements + classical bits. Decompose into Bell basis, store
    measurement outcomes.
    """

    name = "teleportation_encode"
    category = "quantum_compression"

    def compress(self, tensor: np.ndarray, bell_pairs: int = 16) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64).ravel()
        n = len(t)
        p = min(bell_pairs, n // 2)
        pairs = t[: 2 * p].reshape(p, 2)
        bell_coeffs = np.zeros((p, 4), dtype=np.float64)
        for i in range(p):
            a, b = pairs[i]
            bell_coeffs[i, 0] = (a + b) / math.sqrt(2)
            bell_coeffs[i, 1] = (a - b) / math.sqrt(2)
            bell_coeffs[i, 2] = (b + a) / math.sqrt(2)
            bell_coeffs[i, 3] = (b - a) / math.sqrt(2)
        max_coeff = np.max(np.abs(bell_coeffs), axis=1)
        basis_idx = np.argmax(np.abs(bell_coeffs), axis=1)
        meas_vals = bell_coeffs[np.arange(p), basis_idx].astype(np.float32)
        residual = t[2 * p :].astype(np.float32)
        meta = dict(
            shape=tensor.shape,
            n=n,
            p=p,
            residual_n=n - 2 * p,
        )
        data = struct.pack("<II", p, n - 2 * p)
        data += basis_idx.astype(np.uint8).tobytes()
        data += _serialize(meas_vals)
        data += _serialize(residual)
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        n = metadata["n"]
        p, res_n = struct.unpack_from("<II", data, 0)
        pos = 8
        basis_idx = np.frombuffer(data[pos : pos + p], dtype=np.uint8).astype(int)
        pos += p
        meas_vals = _deserialize(data[pos : pos + p * 4])
        pos += p * 4
        residual = _deserialize(data[pos : pos + res_n * 4])
        recon = np.zeros(n, dtype=np.float32)
        for i in range(p):
            bi = basis_idx[i]
            mv = meas_vals[i]
            if bi == 0:
                recon[2 * i] = mv / math.sqrt(2)
                recon[2 * i + 1] = mv / math.sqrt(2)
            elif bi == 1:
                recon[2 * i] = mv / math.sqrt(2)
                recon[2 * i + 1] = -mv / math.sqrt(2)
            elif bi == 2:
                recon[2 * i] = mv / math.sqrt(2)
                recon[2 * i + 1] = mv / math.sqrt(2)
            else:
                recon[2 * i] = -mv / math.sqrt(2)
                recon[2 * i + 1] = mv / math.sqrt(2)
        recon[2 * p :] = residual
        return recon.reshape(shape).astype(np.float32)
