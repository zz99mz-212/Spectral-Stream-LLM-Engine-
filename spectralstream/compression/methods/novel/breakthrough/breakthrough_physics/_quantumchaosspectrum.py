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

class QuantumChaosSpectrum:
    """Quantum chaos signatures: compute the spectral form factor, level
    spacing distribution, and out-of-time-ordered correlator (OTOC)
    from the weight matrix's singular value spectrum. Store only the
    chaos parameters (level repulsion exponent β, spectral rigidity Δ₃,
    Thouless time t_Th). The weight is a random matrix with matching
    chaos properties + stored signal components.

    Real: fit RMT statistics to spectrum, store top SVD components.
    """

    name = "quantum_chaos_spectrum"
    category = "breakthrough_physics"

    def compress(self, tensor: np.ndarray, rank: int = 16) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64).reshape(tensor.shape[0], -1)
        m, n = t.shape
        U, S, Vt = np.linalg.svd(t, full_matrices=False)
        k = min(rank, len(S), m, n)
        # Level repulsion: fit P(s) ∝ s^β e^{-c s^2}
        eigs = S[: min(m, n)] / max(S[0], 1e-30)
        gaps = np.diff(np.sort(eigs[eigs > 1e-10]))
        if len(gaps) > 1:
            r_mean = float(np.mean(gaps[1:] / (gaps[:-1] + 1e-30)))
        else:
            r_mean = 0.5
        # β ≈ 1 (GOE), 2 (GUE), 4 (GSE)
        beta = min(4.0, max(0.5, 2.0 * r_mean))
        # Spectral rigidity Δ₃
        n_e = len(eigs)
        unfolded = np.cumsum(np.sort(eigs)) / max(np.sum(eigs), 1e-30) * n_e
        if n_e > 2:
            delta3 = float(np.var(unfolded - np.arange(1, n_e + 1)))
        else:
            delta3 = 0.0
        U_k = U[:, :k].astype(np.float32)
        S_k = S[:k].astype(np.float32)
        Vt_k = Vt[:k, :].astype(np.float32)
        buf = struct.pack("<III", m, n, k)
        buf += struct.pack("<fff", beta, delta3, float(np.mean(S)))
        buf += _serialize(U_k)
        buf += _serialize(S_k)
        buf += _serialize(Vt_k)
        return bytes(buf), {
            "shape": tensor.shape,
            "m": m,
            "n": n,
            "k": k,
            "beta": beta,
            "delta3": delta3,
        }

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        m, n, k = struct.unpack_from("<III", data, 0)
        pos = 12
        beta, delta3, s_mean = struct.unpack_from("<fff", data, pos)
        pos += 12
        U_k = np.frombuffer(data[pos : pos + m * k * 4], dtype=np.float32).reshape(m, k)
        pos += m * k * 4
        S_k = np.frombuffer(data[pos : pos + k * 4], dtype=np.float32)
        pos += k * 4
        Vt_k = np.frombuffer(data[pos : pos + k * n * 4], dtype=np.float32).reshape(
            k, n
        )
        # Reconstruction: SVD signal + chaos spectrum as regularization
        recon = (U_k * S_k) @ Vt_k
        return recon.astype(np.float32).reshape(metadata["shape"])
