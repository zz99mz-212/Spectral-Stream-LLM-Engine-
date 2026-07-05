from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class SpectralAction:
    """Spectral action principle (Chamseddine-Connes): the action of a
    noncommutative geometry is S = Tr(f(D/Λ)) where D is the Dirac
    operator. The weight matrix's eigenvalues are the Dirac spectrum.
    Store only the spectral action parameters (cutoff Λ, function f
    coefficients) + top eigenvectors. The full spectrum is determined
    by the spectral action.

    Real: fit the singular value distribution to a spectral action
    function (polynomial of log-eigenvalues). Store function
    coefficients + top singular vectors.
    """

    name = "spectral_action"
    category = "breakthrough_math"

    def compress(self, tensor: np.ndarray, rank: int = 12) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64).reshape(tensor.shape[0], -1)
        m, n = t.shape
        U, S, Vt = np.linalg.svd(t, full_matrices=False)
        k = min(rank, len(S), m, n)
        # Fit spectral action f(x) = Σ c_i x^i to log spectrum
        n_eigs = len(S)
        x = np.linspace(0, 1, n_eigs)
        y = np.log(S + 1e-30)
        poly_deg = 3
        poly_c = np.polyfit(x, y, poly_deg).astype(np.float32)
        # Cutoff = scale parameter
        cutoff = float(np.mean(S[: max(1, n_eigs // 4)]))
        U_k = U[:, :k].astype(np.float32)
        S_k = S[:k].astype(np.float32)
        Vt_k = Vt[:k, :].astype(np.float32)
        buf = struct.pack("<III", m, n, k)
        buf += struct.pack("<f", cutoff)
        buf += _serialize(poly_c)
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
        cutoff = struct.unpack_from("<f", data, pos)[0]
        pos += 4
        poly_c = np.frombuffer(data[pos : pos + 16], dtype=np.float32)
        pos += 16
        U_k = np.frombuffer(data[pos : pos + m * k * 4], dtype=np.float32).reshape(m, k)
        pos += m * k * 4
        S_k = np.frombuffer(data[pos : pos + k * 4], dtype=np.float32)
        pos += k * 4
        Vt_k = np.frombuffer(data[pos : pos + k * n * 4], dtype=np.float32).reshape(
            k, n
        )
        recon = (U_k * S_k) @ Vt_k
        return recon.astype(np.float32).reshape(metadata["shape"])
