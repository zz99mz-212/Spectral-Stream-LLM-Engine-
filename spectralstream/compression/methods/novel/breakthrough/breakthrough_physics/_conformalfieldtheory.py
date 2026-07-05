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

class ConformalFieldTheory:
    """CFT: neural network weights are correlation functions of a CFT.
    Store the central charge c (singular value decay rate), conformal
    dimensions Δ_i (singular values), and OPE coefficients C_{ijk}
    (the singular vectors). The full weight structure follows from
    the CFT data via the conformal bootstrap.

    Real: truncated SVD with power-law singular value decay fitted to c.
    """

    name = "conformal_field_theory"
    category = "breakthrough_physics"

    def compress(self, tensor: np.ndarray, rank: int = 32) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64).reshape(tensor.shape[0], -1)
        m, n = t.shape
        U, S, Vt = np.linalg.svd(t, full_matrices=False)
        k = min(rank, len(S), m, n)
        # Central charge c = decay exponent of singular values
        idx = np.arange(1, len(S) + 1, dtype=np.float64)
        log_S = np.log(S + 1e-30)
        log_idx = np.log(idx + 1e-30)
        A = np.vstack([log_idx, np.ones_like(log_idx)]).T
        slope, _ = np.linalg.lstsq(A, log_S, rcond=None)[0]
        central_charge = float(-slope)  # c = -d(log S)/d(log n)
        # Conformal dimensions = stored singular values
        U_k = U[:, :k].astype(np.float32)
        S_k = S[:k].astype(np.float32)
        Vt_k = Vt[:k, :].astype(np.float32)
        buf = struct.pack("<III", m, n, k)
        buf += struct.pack("<f", central_charge)
        buf += _serialize(U_k)
        buf += _serialize(S_k)
        buf += _serialize(Vt_k)
        return bytes(buf), {
            "shape": tensor.shape,
            "m": m,
            "n": n,
            "k": k,
            "central_charge": central_charge,
        }

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        m, n, k = struct.unpack_from("<III", data, 0)
        pos = 12
        c = struct.unpack_from("<f", data, pos)[0]
        pos += 4
        U_k = np.frombuffer(data[pos : pos + m * k * 4], dtype=np.float32).reshape(m, k)
        pos += m * k * 4
        S_k = np.frombuffer(data[pos : pos + k * 4], dtype=np.float32)
        pos += k * 4
        Vt_k = np.frombuffer(data[pos : pos + k * n * 4], dtype=np.float32).reshape(
            k, n
        )
        # CFT bootstrap: S reconstructed with power-law from c
        S_recon = S_k.copy()
        if abs(c) > 1e-10:
            for i in range(k):
                S_recon[i] = S_k[0] * ((i + 1) ** (-abs(c)))
        recon = (U_k * S_recon) @ Vt_k
        return recon.astype(np.float32).reshape(metadata["shape"])
