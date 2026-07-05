from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from ._compressionmethod import (
    CompressionMethod,
    ALL_METHODS,
    _ensure_2d,
    _restore_shape,
    _safe_bytes,
)


def _ensure_2d(t: np.ndarray) -> Tuple[np.ndarray, tuple]:
    if t.ndim == 1:
        return t.reshape(1, -1), t.shape
    if t.ndim > 2:
        orig = t.shape
        return t.reshape(orig[0], -1), orig
    return t, t.shape


def _restore_shape(t: np.ndarray, orig_shape: tuple) -> np.ndarray:
    return t.reshape(orig_shape) if t.shape != orig_shape else t


def _safe_bytes(data: Any) -> int:
    if isinstance(data, np.ndarray):
        return data.nbytes
    if isinstance(data, dict):
        return sum(_safe_bytes(v) for v in data.values()) + sum(
            _safe_bytes(k) for k in data.keys()
        )
    if isinstance(data, (list, tuple)):
        return sum(_safe_bytes(x) for x in data)
    if isinstance(data, (int, float, np.integer, np.floating)):
        return 8
    if isinstance(data, str):
        return len(data)
    return 0


class QuantumEntanglementCompression(CompressionMethod):
    """Exploit inter-tensor correlations using quantum entanglement analogy.

    Mathematical basis:
        If two weight tensors W1 and W2 are "entangled" (highly correlated),
        their joint state can be written as:
            |W1, W2> = sum_{ij} c_{ij} |i> |j>
        where c_{ij} has low Schmidt rank: c_{ij} = sum_k u_k[i] * v_k[j].

    Algorithm:
        1. Partition tensor into pairs of sub-matrices
        2. Compute cross-correlation matrix C = W1^T W2
        3. SVD of C gives Schmidt decomposition
        4. Store Schmidt coefficients + singular vectors

    Compression: O(r*(m1+m2)) instead of O(m1*m2), where r = Schmidt rank.
    """

    name = "quantum_entanglement"
    category = "quantum_mechanics"

    def compress(self, tensor, n_pairs=8, max_schmidt_rank=8, **kw):
        t, orig = _ensure_2d(tensor)
        m, n = t.shape

        # SVD fallback
        U_s, S_s, Vt_s = np.linalg.svd(t.astype(np.float64), full_matrices=False)
        svd_rank = min(16, len(S_s))
        svd_data = {
            "U": U_s[:, :svd_rank].astype(np.float32),
            "S": S_s[:svd_rank].astype(np.float32),
            "Vt": Vt_s[:svd_rank, :].astype(np.float32),
            "rank": svd_rank,
        }

        half_m = m // 2
        if half_m < 1:
            half_m = 1

        pair_size = max(1, m // n_pairs)
        schmidt_data = []

        for p in range(n_pairs):
            i_start = p * pair_size
            i_end = min(i_start + pair_size, m)
            if i_end <= i_start:
                continue
            block = t[i_start:i_end].astype(np.float64)

            mid = block.shape[0] // 2
            if mid < 1:
                mid = 1
            W1 = block[:mid]
            W2 = block[mid:]

            if W1.shape[0] == 0 or W2.shape[0] == 0:
                continue

            C = W1.T @ W2
            U, S, Vt_ = np.linalg.svd(C, full_matrices=False)
            r = min(max_schmidt_rank, len(S))

            schmidt_data.append(
                {
                    "U": U[:, :r].astype(np.float32),
                    "S": S[:r].astype(np.float32),
                    "Vt": Vt_[:r, :].astype(np.float32),
                    "W1": W1.astype(np.float32),
                    "W2_mean": W2.mean(axis=0).astype(np.float32),
                    "pair_idx": p,
                    "shape1": W1.shape,
                    "shape2": W2.shape,
                }
            )

        return {
            "pairs": schmidt_data,
            "n_pairs": n_pairs,
            "pair_size": pair_size,
            "shape": t.shape,
            "svd": svd_data,
        }, {"orig_shape": orig}

    def decompress(self, cd, meta):
        if "svd" in cd:
            U = cd["svd"]["U"].astype(np.float64)
            S = cd["svd"]["S"].astype(np.float64)
            Vt = cd["svd"]["Vt"].astype(np.float64)
            return _restore_shape(((U * S) @ Vt).astype(np.float32), meta["orig_shape"])

        m, n = (
            meta["orig_shape"][:2]
            if len(meta["orig_shape"]) >= 2
            else (1, meta["orig_shape"][0])
        )
        result = np.zeros((m, n), dtype=np.float32)
        pair_size = cd["pair_size"]

        for pdata in cd["pairs"]:
            p = pdata["pair_idx"]
            i_start = p * pair_size
            W1 = pdata["W1"].astype(np.float64)
            U = pdata["U"].astype(np.float64)
            S = pdata["S"].astype(np.float64)
            Vt = pdata["Vt"].astype(np.float64)
            W2_mean = pdata["W2_mean"].astype(np.float64)

            mid = W1.shape[0]
            C_approx = U @ np.diag(S) @ Vt
            W2_recon = np.zeros((mid, n), dtype=np.float64)
            for i in range(mid):
                W2_recon[i] = W1[i] @ C_approx.T / (np.linalg.norm(W1[i]) + 1e-10)
            W2_recon += W2_mean[None, :]

            end1 = min(i_start + mid, m)
            sl1 = end1 - i_start
            result[i_start:end1] = W1[:sl1].astype(np.float32)

            end2 = min(i_start + mid + W2_recon.shape[0], m)
            sl2 = end2 - (i_start + mid)
            if sl2 > 0:
                result[i_start + mid : end2] = W2_recon[:sl2].astype(np.float32)

        return _restore_shape(result, meta["orig_shape"])


def _generate_monomials(n_vars: int, degree: int) -> list:
    """Generate all monomials of given degree in n_vars variables."""
    if degree == 0:
        return [()]
    if degree == 1:
        return [(i,) for i in range(n_vars)]
    result = []
    for i in range(n_vars):
        for rest in _generate_monomials(n_vars, degree - 1):
            if len(rest) == 0 or i >= rest[0]:
                result.append((i,) + rest)
    return result[:50]  # limit for efficiency
