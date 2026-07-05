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

class SuperspaceCompress:
    """Superspace encoding: weights are superfields Φ(x, θ, θ̄) in
    superspace. The bosonic component emerges from θ-integration.
    Store only the superfield component fields (which are fewer
    than the full matrix due to SUSY constraints).

    Real: decompose weight into even (bosonic) and odd (fermionic)
    parts via parity splitting. The bosonic part is the main signal,
    fermionic part is the residual. Store both compactly.
    """

    name = "superspace_compress"
    category = "breakthrough_physics"

    def compress(self, tensor: np.ndarray, rank: int = 12) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64).reshape(tensor.shape[0], -1)
        m, n = t.shape
        # Bosonic component = even-index rows/cols (symmetric part)
        bosonic = (t + t.T) / 2.0 if m == n else t.copy()
        # Fermionic component = antisymmetric part
        fermionic = (t - t.T) / 2.0 if m == n else np.zeros_like(t)
        # Compress bosonic via SVD
        U, S, Vt = np.linalg.svd(bosonic, full_matrices=False)
        k = min(rank, len(S), m, n)
        U_k = U[:, :k].astype(np.float32)
        S_k = S[:k].astype(np.float32)
        Vt_k = Vt[:k, :].astype(np.float32)
        # Fermionic residual via sparsity
        f_flat = fermionic.ravel()
        f_thr = np.percentile(np.abs(f_flat), 80)
        f_mask = np.abs(f_flat) >= f_thr
        f_kept = f_flat[f_mask].astype(np.float16)
        f_idx = np.where(f_mask)[0].astype(np.int32)
        buf = struct.pack("<III", m, n, k)
        buf += struct.pack("<I", len(f_kept))
        buf += _serialize(U_k)
        buf += _serialize(S_k)
        buf += _serialize(Vt_k)
        buf += f_idx.tobytes()
        buf += f_kept.tobytes()
        return bytes(buf), {
            "shape": tensor.shape,
            "m": m,
            "n": n,
            "k": k,
            "n_fermionic": len(f_kept),
        }

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        m, n, k = struct.unpack_from("<III", data, 0)
        pos = 12
        n_f = struct.unpack_from("<I", data, pos)[0]
        pos += 4
        U_k = np.frombuffer(data[pos : pos + m * k * 4], dtype=np.float32).reshape(m, k)
        pos += m * k * 4
        S_k = np.frombuffer(data[pos : pos + k * 4], dtype=np.float32)
        pos += k * 4
        Vt_k = np.frombuffer(data[pos : pos + k * n * 4], dtype=np.float32).reshape(
            k, n
        )
        pos += k * n * 4
        bosonic = (U_k * S_k) @ Vt_k
        if n_f > 0:
            f_idx = np.frombuffer(data[pos : pos + n_f * 4], dtype=np.int32)
            pos += n_f * 4
            f_vals = np.frombuffer(data[pos : pos + n_f * 2], dtype=np.float16).astype(
                np.float64
            )
            fermionic = np.zeros(m * n, dtype=np.float64)
            for idx, val in zip(f_idx, f_vals):
                if 0 <= idx < m * n:
                    fermionic[idx] = val
            fermionic = fermionic.reshape(m, n)
            # θ-integration: bosonic + fermionic mixer
            bosonic += fermionic * 0.1
        return bosonic.astype(np.float32).reshape(metadata["shape"])
