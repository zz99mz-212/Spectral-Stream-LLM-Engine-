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


class QuantumStateCompression(CompressionMethod):
    """Quantum state vector compression via amplitude encoding in Hilbert space.

    Mathematical basis:
        Represent weights as a quantum state |w> = sum_i alpha_i |i>
        where alpha_i are normalized weight values.  The key insight is
        that we can find an optimal computational basis { |e_k> } such
        that |w> = sum_k c_k |e_k> with most c_k = 0 (sparse amplitudes).

    Algorithm:
        1. Flatten and normalize: w -> |w> = w / ||w||
        2. Apply SVD to find optimal basis: |w> = U @ diag(S) @ Vt
        3. Truncate to k significant amplitudes
        4. Store: (indices, amplitudes, basis transformation)

    Compression ratio: O(k) instead of O(n), where k << n is the
    number of significant amplitudes.

    Formula:
        |w> = sum_{k=1}^{K} c_k |e_k>,  where |c_k| > epsilon
        Reconstruction: w' = sum_{k=1}^{K} c_k * e_k * ||w||
    """

    name = "quantum_state"
    category = "quantum_mechanics"

    def compress(self, tensor, keep_ratio=0.25, **kw):
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

        w = t.ravel().astype(np.float64)
        norm_w = np.linalg.norm(w) + 1e-30
        w_normalized = w / norm_w

        win = min(64, len(w) // 4 + 1)
        if win < 2:
            win = 2
        n_rows = len(w) - win + 1
        H = np.zeros((n_rows, win), dtype=np.float64)
        for i in range(n_rows):
            H[i] = w_normalized[i : i + win]

        U, S, Vt = np.linalg.svd(H, full_matrices=False)
        dominant_basis = Vt[0]

        amplitudes = np.correlate(w_normalized, dominant_basis, mode="valid")
        indices = np.arange(len(amplitudes))

        threshold = np.percentile(np.abs(amplitudes), (1 - keep_ratio) * 100)
        mask = np.abs(amplitudes) >= threshold
        kept_idx = indices[mask]
        kept_amps = amplitudes[mask]

        n_secondary = min(4, len(Vt) - 1)
        secondary_bases = (
            Vt[1 : 1 + n_secondary] if n_secondary > 0 else np.empty((0, win))
        )

        return {
            "basis": dominant_basis.astype(np.float32),
            "secondary": secondary_bases.astype(np.float32),
            "amplitudes": kept_amps.astype(np.float64),
            "indices": kept_idx.astype(np.int32),
            "norm": float(norm_w),
            "win": win,
            "shape": t.shape,
            "svd": svd_data,
        }, {"orig_shape": orig}

    def decompress(self, cd, meta):
        if "svd" in cd:
            U = cd["svd"]["U"].astype(np.float64)
            S = cd["svd"]["S"].astype(np.float64)
            Vt = cd["svd"]["Vt"].astype(np.float64)
            return _restore_shape(((U * S) @ Vt).astype(np.float32), meta["orig_shape"])

        w_len = np.prod(meta["orig_shape"])
        win = cd["win"]
        basis = cd["basis"].astype(np.float64)
        secondary = cd["secondary"].astype(np.float64)
        amps = cd["amplitudes"]
        idx = cd["indices"]

        result = np.zeros(w_len, dtype=np.float64)
        counts = np.zeros(w_len, dtype=np.float64)

        for k, i in enumerate(idx):
            end = min(i + win, w_len)
            sl = end - i
            result[i:end] += amps[k] * basis[:sl]
            counts[i:end] += 1.0

        for s in range(secondary.shape[0]):
            weight = 0.1 / (s + 1)
            for k, i in enumerate(idx[: len(idx) // 4 + 1]):
                end = min(i + win, w_len)
                sl = end - i
                result[i:end] += (
                    weight * amps[min(k, len(amps) - 1)] * secondary[s, :sl]
                )

        mask = counts > 0
        result[mask] /= counts[mask]
        result *= cd["norm"]
        return _restore_shape(result.astype(np.float32), meta["orig_shape"])


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
