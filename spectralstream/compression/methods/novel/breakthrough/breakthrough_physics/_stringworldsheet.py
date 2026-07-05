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

class StringWorldsheet:
    """Relativistic string: the weight matrix is the worldsheet of a string
    sweeping through spacetime. The singular vectors are the string's
    oscillation modes (left/right movers), singular values are the
    mode amplitudes. The string tension T sets the overall scale.

    Store: truncated SVD components + fitted string tension T.
    Reconstruct: Σ_i s_i * U_i V_i^T = string's embedding in target space.
    """

    name = "string_worldsheet"
    category = "breakthrough_physics"

    def compress(self, tensor: np.ndarray, rank: int = 32) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64).reshape(tensor.shape[0], -1)
        m, n = t.shape
        k = min(rank, m, n)
        U, S, Vt = np.linalg.svd(t, full_matrices=False)
        k = min(k, len(S))
        # String tension = characteristic scale
        T = float(np.mean(S[: max(1, k // 2)]))
        # Left/right mover decomposition
        left_movers = U[:, :k].astype(np.float32)
        right_movers = Vt[:k, :].astype(np.float32)
        mode_amplitudes = S[:k].astype(np.float32)
        buf = struct.pack("<III", m, n, k)
        buf += struct.pack("<f", T)
        buf += _serialize(left_movers)
        buf += _serialize(mode_amplitudes)
        buf += _serialize(right_movers)
        return bytes(buf), {
            "shape": tensor.shape,
            "m": m,
            "n": n,
            "k": k,
            "tension": T,
        }

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        m, n, k = struct.unpack_from("<III", data, 0)
        pos = 12
        T = struct.unpack_from("<f", data, pos)[0]
        pos += 4
        U = np.frombuffer(data[pos : pos + m * k * 4], dtype=np.float32).reshape(m, k)
        pos += m * k * 4
        S = np.frombuffer(data[pos : pos + k * 4], dtype=np.float32)
        pos += k * 4
        Vt = np.frombuffer(data[pos : pos + k * n * 4], dtype=np.float32).reshape(k, n)
        # Reconstruct: string's worldsheet coordinates
        recon = (U * (S * T / max(T, 1e-10))) @ Vt
        return recon.astype(np.float32).reshape(metadata["shape"])
