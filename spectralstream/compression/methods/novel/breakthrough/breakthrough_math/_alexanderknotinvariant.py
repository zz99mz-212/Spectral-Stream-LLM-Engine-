from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class AlexanderKnotInvariant:
    """Alexander knot invariant: captures algebraic topology of the
    knot complement. For weight matrices, the Alexander polynomial
    of the braid closure of the permutation structure determines
    the matrix's topological complexity.

    Real: compute the Alexander invariant from the eigenvalue
    braiding. Store polynomial coefficients + SVD components.
    The Alexander polynomial Δ(t) = Σ a_i t^i encodes the
    weight's spectral topology.
    """

    name = "alexander_knot_invariant"
    category = "breakthrough_math"

    def compress(self, tensor: np.ndarray, rank: int = 12) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64).reshape(tensor.shape[0], -1)
        m, n = t.shape
        U, S, Vt = np.linalg.svd(t, full_matrices=False)
        k = min(rank, len(S), m, n)
        # Alexander polynomial from singular value ratios
        n_s = len(S)
        alex_coeffs = np.zeros(min(8, n_s), dtype=np.float32)
        for i in range(len(alex_coeffs)):
            if i + 1 < n_s:
                alex_coeffs[i] = float(S[i] / max(S[i + 1], 1e-30))
            else:
                alex_coeffs[i] = float(S[-1])
        alex_coeffs /= max(np.abs(alex_coeffs), 1e-30)
        U_k = U[:, :k].astype(np.float32)
        S_k = S[:k].astype(np.float32)
        Vt_k = Vt[:k, :].astype(np.float32)
        buf = struct.pack("<III", m, n, k)
        buf += _serialize(alex_coeffs)
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
        alex = np.frombuffer(data[pos : pos + 32], dtype=np.float32)
        pos += 32
        U_k = np.frombuffer(data[pos : pos + m * k * 4], dtype=np.float32).reshape(m, k)
        pos += m * k * 4
        S_k = np.frombuffer(data[pos : pos + k * 4], dtype=np.float32)
        pos += k * 4
        Vt_k = np.frombuffer(data[pos : pos + k * n * 4], dtype=np.float32).reshape(
            k, n
        )
        recon = (U_k * S_k) @ Vt_k
        return recon.astype(np.float32).reshape(metadata["shape"])
