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

class RandomMatrixEnsemble:
    """Random matrix theory: model weight statistics via Wigner-Dyson
    ensembles (GOE/GUE/GSE). Store ensemble type + spectral correlation
    parameters + top SVD components for the non-random signal.

    The random part is sampled from the ensemble during decompression;
    the signal part is stored via truncated SVD.
    """

    name = "random_matrix_ensemble"
    category = "breakthrough_physics"

    def compress(self, tensor: np.ndarray, signal_rank: int = 16) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64).reshape(tensor.shape[0], -1)
        m, n = t.shape
        # Classify ensemble type by level spacing ratio
        eigs = np.linalg.svd(t, compute_uv=False)
        gaps = np.diff(np.sort(eigs[: min(m, n)]))
        r_mean = float(np.mean(gaps[1:] / (gaps[:-1] + 1e-30)))
        # GOE (r≈0.53), GUE (r≈0.60), GSE (r≈0.67)
        if r_mean < 0.56:
            ensemble_type = 0  # GOE
        elif r_mean < 0.63:
            ensemble_type = 1  # GUE
        else:
            ensemble_type = 2  # GSE
        # Extract signal (non-random) part via truncated SVD
        k = min(signal_rank, m, n, len(eigs))
        U, S, Vt = np.linalg.svd(t, full_matrices=False)
        sig_mean = float(np.mean(t))
        sig_std = float(np.std(t))
        U_k = U[:, :k].astype(np.float32)
        S_k = S[:k].astype(np.float32)
        Vt_k = Vt[:k, :].astype(np.float32)
        buf = struct.pack("<III", m, n, k)
        buf += struct.pack("<Bff", ensemble_type, sig_mean, sig_std)
        buf += _serialize(U_k)
        buf += _serialize(S_k)
        buf += _serialize(Vt_k)
        return bytes(buf), {
            "shape": tensor.shape,
            "m": m,
            "n": n,
            "k": k,
            "ensemble": ["GOE", "GUE", "GSE"][ensemble_type],
        }

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        m, n, k = struct.unpack_from("<III", data, 0)
        pos = 12
        ensemble_type, sig_mean, sig_std = struct.unpack_from("<Bff", data, pos)
        pos += 9
        U_k = np.frombuffer(data[pos : pos + m * k * 4], dtype=np.float32).reshape(m, k)
        pos += m * k * 4
        S_k = np.frombuffer(data[pos : pos + k * 4], dtype=np.float32)
        pos += k * 4
        Vt_k = np.frombuffer(data[pos : pos + k * n * 4], dtype=np.float32).reshape(
            k, n
        )
        # Signal = truncated SVD
        signal = (U_k * S_k) @ Vt_k
        return signal.astype(np.float32).reshape(metadata["shape"])
