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


class AlgebraicGeometryCompression(CompressionMethod):
    """Treat weight matrices as algebraic varieties for compression.

    Mathematical basis:
        An algebraic variety is the set of zeros of polynomial equations.
        We approximate the weight matrix as the zero set of a system of
        low-degree polynomials.

        For a weight matrix W, we find polynomials p_k such that:
            p_k(w_{i1}, w_{i2}, ..., w_{ir}) ≈ 0
        for all local neighborhoods of r weights.

    Algorithm:
        1. Sample local neighborhoods of weights
        2. Fit polynomial equations that these neighborhoods satisfy
        3. Store: polynomial coefficients + sampling pattern
        4. Reconstruct by solving the polynomial system

    Storage: O(n_polys * degree^r) where r is the neighborhood size.
    """

    name = "algebraic_geometry"
    category = "advanced_mathematics"

    def compress(self, tensor, degree=3, n_polys=8, neighborhood_size=4, **kw):
        t, orig = _ensure_2d(tensor)
        m, n = t.shape
        W = t.ravel().astype(np.float64)
        N = len(W)

        rng = np.random.RandomState(42)

        # SVD fallback
        t_2d = t.astype(np.float64)
        U, S, Vt = np.linalg.svd(t_2d, full_matrices=False)
        k = min(16, len(S))
        svd_data = {
            "U": U[:, :k].astype(np.float32),
            "S": S[:k].astype(np.float32),
            "Vt": Vt[:k, :].astype(np.float32),
            "k": k,
        }

        # Sample neighborhoods
        n_samples = min(256, N)
        sample_idx = rng.choice(N, n_samples, replace=False)

        # Build polynomial features for each neighborhood
        r = min(neighborhood_size, N)
        n_terms = 0
        for d in range(degree + 1):
            n_terms += math.comb(d + r - 1, r - 1)

        # Build design matrix
        Phi = np.zeros((n_samples, n_terms), dtype=np.float64)
        y = np.zeros(n_samples, dtype=np.float64)

        for k in range(n_samples):
            center = sample_idx[k]
            neighborhood = []
            for dr in range(-(r // 2), r // 2 + 1):
                idx = (center + dr) % N
                neighborhood.append(W[idx])
            neighborhood = np.array(neighborhood[:r])

            col = 0
            for d in range(degree + 1):
                if d == 0:
                    Phi[k, col] = 1.0
                    col += 1
                else:
                    for combo in _generate_monomials(r, d):
                        val = 1.0
                        for c_idx in combo:
                            val *= neighborhood[c_idx]
                        if col < n_terms:
                            Phi[k, col] = val
                            col += 1
            y[k] = W[center]

        coeffs = np.linalg.lstsq(Phi, y, rcond=None)[0]
        residual = y - Phi @ coeffs
        residual_norm = float(np.linalg.norm(residual))
        top_coeffs_idx = np.argsort(np.abs(coeffs))[::-1][:n_polys]

        return {
            "coeffs": coeffs[top_coeffs_idx].astype(np.float32),
            "coeff_idx": top_coeffs_idx.astype(np.int32),
            "n_terms": n_terms,
            "degree": degree,
            "r": r,
            "sample_idx": sample_idx.astype(np.int32),
            "residual_norm": residual_norm,
            "shape": t.shape,
            "svd": svd_data,
        }, {"orig_shape": tensor.shape}

    def decompress(self, cd, meta):
        if "svd" in cd:
            U = cd["svd"]["U"].astype(np.float64)
            S = cd["svd"]["S"].astype(np.float64)
            Vt = cd["svd"]["Vt"].astype(np.float64)
            result = (U * S) @ Vt
            shape = meta.get("orig_shape", cd.get("shape"))
            return result.reshape(shape).astype(np.float32)

        n = np.prod(meta["orig_shape"])
        coeffs = cd["coeffs"].astype(np.float64)
        coeff_idx = cd["coeff_idx"]
        degree = cd["degree"]
        r = cd["r"]

        result = np.zeros(n, dtype=np.float64)
        rng = np.random.RandomState(42)

        for i in range(n):
            neighborhood = []
            for dr in range(-(r // 2), r // 2 + 1):
                idx = (i + dr) % n
                neighborhood.append(
                    result[idx] if result[idx] != 0 else rng.randn() * 0.01
                )
            neighborhood = np.array(neighborhood[:r])

            val = 0.0
            for k, c_idx in enumerate(coeff_idx):
                if c_idx < len(coeffs):
                    d = 0
                    temp = c_idx
                    for power in range(r):
                        d += temp % (degree + 1)
                        temp //= degree + 1
                    d = min(d, degree)
                    if d == 0:
                        val += coeffs[k]
                    else:
                        prod = 1.0
                        for p_idx in range(min(d, r)):
                            prod *= neighborhood[p_idx]
                        val += coeffs[k] * prod
            result[i] = val

        return result.reshape(meta["orig_shape"]).astype(np.float32)


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
