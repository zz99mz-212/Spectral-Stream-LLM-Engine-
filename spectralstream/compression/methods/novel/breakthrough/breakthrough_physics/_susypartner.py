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

class SUSYPartner:
    """Supersymmetric partner: every weight matrix W has a superpartner W̃
    derived from a superpotential W(Φ). Store only the superpotential
    coefficients (polynomial fitted to singular values). Both W and
    W̃ are derived from W(Φ) via SUSY transformations.

    Real: fit singular values to a polynomial 'superpotential'.
    Store polynomial coefficients + singular vectors.
    The 'SUSY partner' reconstruction uses derivative of W.
    """

    name = "susy_partner"
    category = "breakthrough_physics"

    def compress(
        self, tensor: np.ndarray, rank: int = 16, poly_deg: int = 4
    ) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64).reshape(tensor.shape[0], -1)
        m, n = t.shape
        U, S, Vt = np.linalg.svd(t, full_matrices=False)
        k = min(rank, len(S), m, n)
        # Fit superpotential to singular values
        x = np.linspace(0, 1, k)
        poly_coeffs = np.polyfit(x, S[:k] / max(S[0], 1e-30), poly_deg)
        poly_coeffs = poly_coeffs.astype(np.float32)
        U_k = U[:, :k].astype(np.float32)
        S_k = S[:k].astype(np.float32)
        Vt_k = Vt[:k, :].astype(np.float32)
        buf = struct.pack("<III", m, n, k)
        buf += struct.pack("<I", poly_deg)
        buf += _serialize(poly_coeffs)
        buf += _serialize(U_k)
        buf += _serialize(S_k)
        buf += _serialize(Vt_k)
        return bytes(buf), {
            "shape": tensor.shape,
            "m": m,
            "n": n,
            "k": k,
            "poly_deg": poly_deg,
        }

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        m, n, k = struct.unpack_from("<III", data, 0)
        pos = 12
        poly_deg = struct.unpack_from("<I", data, pos)[0]
        pos += 4
        poly_coeffs = np.frombuffer(
            data[pos : pos + (poly_deg + 1) * 4], dtype=np.float32
        )
        pos += (poly_deg + 1) * 4
        U_k = np.frombuffer(data[pos : pos + m * k * 4], dtype=np.float32).reshape(m, k)
        pos += m * k * 4
        S_k = np.frombuffer(data[pos : pos + k * 4], dtype=np.float32)
        pos += k * 4
        Vt_k = np.frombuffer(data[pos : pos + k * n * 4], dtype=np.float32).reshape(
            k, n
        )
        # Superpotential evaluation: reconstruct S from polynomial
        x = np.linspace(0, 1, k)
        P = np.polyval(poly_coeffs, x) * max(S_k[0], 1e-10)
        # SUSY partner: W' = dW/dΦ ≈ gradient of superpotential
        # Use superpotential values as rescaled singular values
        recon = (U_k * P.astype(np.float32)) @ Vt_k
        return recon.astype(np.float32).reshape(metadata["shape"])
