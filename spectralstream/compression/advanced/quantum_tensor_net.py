"""Quantum-inspired tensor network compression via SVD-based MPS decomposition."""

from __future__ import annotations

import struct
from typing import Dict, List, Optional, Tuple

import numpy as np

EPS = 1e-12


def _von_neumann_entropy(rho: np.ndarray) -> float:
    evals = np.linalg.eigvalsh(rho)
    evals = np.maximum(evals, EPS)
    return float(-np.sum(evals * np.log2(evals + EPS)))


def _renyi_entropy(rho: np.ndarray, alpha: float = 2.0) -> float:
    evals = np.linalg.eigvalsh(rho)
    evals = np.maximum(evals, EPS)
    if alpha == 1.0:
        return _von_neumann_entropy(rho)
    purity = float(np.sum(evals**alpha))
    return float(np.log2(purity) / (1.0 - alpha))


class QuantumTensorNetCompressor:
    """Compress tensors via SVD-based low-rank approximation (MPS-style).

    Decomposes matrix → U * diag(S) * Vt, truncates singular values
    based on error budget, stores truncated SVD factors.
    """

    def __init__(self, max_bond_dim: int = 16, error_budget: float = 0.01):
        self.max_bond_dim = max_bond_dim
        self.error_budget = error_budget

    def compress(self, tensor: np.ndarray) -> Tuple[bytes, dict]:
        t = np.asarray(tensor, dtype=np.float64)
        orig_shape = t.shape
        flat = t.ravel()

        U, S, Vt = np.linalg.svd(t.reshape(orig_shape[0], -1), full_matrices=False)

        total_energy = float(np.sum(S**2))
        cumsum = np.cumsum(S**2)
        if total_energy > EPS:
            n_keep = (
                int(np.searchsorted(cumsum, (1.0 - self.error_budget) * total_energy))
                + 1
            )
        else:
            n_keep = len(S)
        n_keep = max(1, min(n_keep, self.max_bond_dim, len(S)))

        U = U[:, :n_keep].astype(np.float32)
        S = S[:n_keep].astype(np.float32)
        Vt = Vt[:n_keep, :].astype(np.float32)

        metadata: dict = {
            "orig_shape": list(orig_shape),
            "rank": n_keep,
            "max_bond_dim": self.max_bond_dim,
            "error_budget": self.error_budget,
            "total_energy": total_energy,
            "kept_energy": float(cumsum[n_keep - 1]) if n_keep > 0 else 0.0,
        }

        buf = bytearray()
        buf += struct.pack("<I", n_keep)
        buf += struct.pack("<II", U.shape[0], U.shape[1])
        buf += U.tobytes()
        buf += struct.pack("<I", len(S))
        buf += S.tobytes()
        buf += struct.pack("<II", Vt.shape[0], Vt.shape[1])
        buf += Vt.tobytes()

        return bytes(buf), metadata

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        orig_shape = tuple(metadata["orig_shape"])
        offset = 0
        n_keep = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        u_rows = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        u_cols = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        U = np.frombuffer(
            data[offset : offset + u_rows * u_cols * 4], dtype=np.float32
        ).reshape(u_rows, u_cols)
        offset += u_rows * u_cols * 4
        s_len = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        S = np.frombuffer(data[offset : offset + s_len * 4], dtype=np.float32)
        offset += s_len * 4
        vt_rows = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        vt_cols = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        Vt = np.frombuffer(
            data[offset : offset + vt_rows * vt_cols * 4], dtype=np.float32
        ).reshape(vt_rows, vt_cols)

        result = (U * S[np.newaxis, :]) @ Vt
        result = result.reshape(orig_shape)
        return result.astype(np.float32)
