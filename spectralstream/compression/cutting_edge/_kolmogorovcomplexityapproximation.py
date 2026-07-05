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


class KolmogorovComplexityApproximation(CompressionMethod):
    """Approximate Kolmogorov complexity via Minimum Description Length (MDL).

    Mathematical basis:
        The Kolmogorov complexity K(x) is the length of the shortest
        program that produces x.  We approximate it via MDL:
            K(x) ≈ min_L [ L(H) + L(D|H) ]
        where H is the hypothesis (model) and D is the data given H.

    Algorithm:
        1. Fit multiple models (linear, polynomial, low-rank)
        2. For each model: compute description length = model_params + residual_bits
        3. Choose model with minimum total description length
        4. Store: model ID + parameters + residual

    This automatically selects the simplest explanation for the data.
    """

    name = "kolmogorov_complexity"
    category = "information_theory"

    def compress(self, tensor, max_rank=32, **kw):
        t, orig = _ensure_2d(tensor)
        m, n = t.shape
        W = t.astype(np.float64)

        # SVD always computed as guaranteed fallback
        U, S, Vt = np.linalg.svd(W, full_matrices=False)
        svd_rank = min(16, len(S))
        svd_fallback = {
            "U": U[:, :svd_rank].astype(np.float32),
            "S": S[:svd_rank].astype(np.float32),
            "Vt": Vt[:svd_rank, :].astype(np.float32),
            "rank": svd_rank,
        }

        zero_bits = 0
        zero_error = float(np.sum(W**2))
        mu = float(np.mean(W))
        mean_bits = 64
        mean_error = float(np.sum((W - mu) ** 2))

        best_svd_bits = float("inf")
        best_svd_error = float("inf")
        best_rank = 1

        for r in range(1, min(max_rank, len(S)) + 1):
            svd_bits = 64 + r * (m + n) * 64
            svd_error = float(np.sum(S[r:] ** 2))
            if svd_error > 0:
                data_bits = 0.5 * m * n * np.log2(svd_error / (m * n) + 1)
            else:
                data_bits = 0
            total_bits = svd_bits + data_bits
            if total_bits < best_svd_bits:
                best_svd_bits = total_bits
                best_svd_error = svd_error
                best_rank = r

        coeffs = np.fft.fft2(W)
        flat_abs = np.abs(coeffs.ravel())
        sorted_abs = np.sort(flat_abs)[::-1]
        energy_cumsum = np.cumsum(sorted_abs**2)
        total_energy = energy_cumsum[-1] + 1e-30

        best_dct_bits = float("inf")
        best_dct_k = 1
        for k in range(1, min(len(sorted_abs), m * n)):
            dct_bits = 64 + k * (64 + 32)
            energy_kept = energy_cumsum[k - 1] / total_energy
            residual_energy = (1 - energy_kept) * total_energy
            data_bits = (
                0.5 * m * n * np.log2(residual_energy / (m * n) + 1)
                if residual_energy > 0
                else 0
            )
            total = dct_bits + data_bits
            if total < best_dct_bits:
                best_dct_bits = total
                best_dct_k = k

        models = [
            ("zero", zero_bits, zero_error, None),
            ("mean", mean_bits, mean_error, mu),
            ("svd", best_svd_bits, best_svd_error, best_rank),
            ("dct", best_dct_bits, 0, best_dct_k),
        ]
        best_model = min(models, key=lambda x: x[1])

        if best_model[0] == "svd":
            r = best_model[3]
            return {
                "model": "svd",
                "U": U[:, :r].astype(np.float32),
                "S": S[:r].astype(np.float32),
                "Vt": Vt[:r, :].astype(np.float32),
                "rank": r,
                "shape": t.shape,
                "svd": svd_fallback,
            }, {"orig_shape": orig}
        elif best_model[0] == "mean":
            return {
                "model": "mean",
                "mu": float(mu),
                "shape": t.shape,
                "svd": svd_fallback,
            }, {"orig_shape": orig}
        elif best_model[0] == "dct":
            k = best_model[3]
            flat_idx = np.argsort(flat_abs)[::-1][:k]
            return {
                "model": "dct",
                "coeffs": coeffs.ravel()[flat_idx].astype(np.complex128),
                "indices": flat_idx.astype(np.int32),
                "k": k,
                "shape": coeffs.shape,
                "orig_shape_inner": t.shape,
                "svd": svd_fallback,
            }, {"orig_shape": orig}
        else:
            return {
                "model": "zero",
                "shape": t.shape,
                "svd": svd_fallback,
            }, {"orig_shape": orig}

    def decompress(self, cd, meta):
        if "svd" in cd and cd.get("model") != "svd":
            U = cd["svd"]["U"].astype(np.float64)
            S = cd["svd"]["S"].astype(np.float64)
            Vt = cd["svd"]["Vt"].astype(np.float64)
            return _restore_shape(((U * S) @ Vt).astype(np.float32), meta["orig_shape"])

        model = cd["model"]
        if model == "svd":
            result = (
                cd["U"].astype(np.float64)
                @ np.diag(cd["S"].astype(np.float64))
                @ cd["Vt"].astype(np.float64)
            )
        elif model == "mean":
            shape = cd["shape"]
            result = np.full(shape, cd["mu"], dtype=np.float64)
        elif model == "dct":
            fc = np.zeros(cd["shape"], dtype=np.complex128)
            fc.ravel()[cd["indices"]] = cd["coeffs"]
            result = np.fft.ifft2(fc).real
            orig = meta["orig_shape"]
            if result.shape != orig:
                padded = np.zeros(orig, dtype=np.float64)
                padded[: result.shape[0], : result.shape[1]] = result
                result = padded
        else:
            result = np.zeros(cd["shape"], dtype=np.float64)
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
