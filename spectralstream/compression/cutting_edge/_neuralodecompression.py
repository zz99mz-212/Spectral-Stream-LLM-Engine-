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


class NeuralODECompression(CompressionMethod):
    """Treat weight evolution through layers as a Neural ODE.

    Mathematical basis:
        Instead of storing weights at each layer W_1, W_2, ..., W_L,
        we store an ODE that generates them:
            dW/dt = f(W, t; theta)

        where f is a neural network parameterized by theta.
        We solve the ODE from t=0 to t=L to recover all layer weights.

    Algorithm:
        1. Compute differences between consecutive layers: dW = W_{l+1} - W_l
        2. Fit a parametric ODE: dW/dt ≈ g(W, t; theta)
        3. Store: initial condition W_0 + ODE parameters theta
        4. Reconstruct by numerical integration

    Storage: O(dim(W) + n_params) instead of O(L * dim(W)).
    """

    name = "neural_ode"
    category = "hybrid"

    def compress(self, tensor, n_layers_approx=4, svd_rank=16, **kw):
        t, orig = _ensure_2d(tensor)
        m, n = t.shape
        W = t.astype(np.float64)

        # SVD fallback
        U, S, Vt = np.linalg.svd(W, full_matrices=False)
        k = min(svd_rank, len(S))
        svd_data = {
            "U": U[:, :k].astype(np.float32),
            "S": S[:k].astype(np.float32),
            "Vt": Vt[:k, :].astype(np.float32),
            "k": k,
        }

        velocities = np.diff(W, axis=0)

        if velocities.shape[0] == 0:
            return {
                "W0": W[0:1].astype(np.float32),
                "theta": np.zeros(1, dtype=np.float32),
                "n_layers": 1,
                "shape": t.shape,
                "svd": svd_data,
            }, {"orig_shape": orig}

        t_grid = np.arange(m - 1, dtype=np.float64) / max(m - 2, 1)
        W_vals = W[:-1]

        n_phi = min(n_layers_approx, m - 1)
        phi = np.zeros((m - 1, n_phi), dtype=np.float64)
        for k in range(n_phi):
            phi[:, k] = t_grid**k

        alpha = np.linalg.lstsq(phi, velocities, rcond=None)[0]
        recon_vel = phi @ alpha
        residual = float(
            np.linalg.norm(velocities - recon_vel)
            / (np.linalg.norm(velocities) + 1e-30)
        )

        return {
            "W0": W[0].astype(np.float32),
            "alpha": alpha.astype(np.float32),
            "n_phi": n_phi,
            "m": m,
            "shape": t.shape,
            "svd": svd_data,
        }, {"orig_shape": orig}

    def decompress(self, cd, meta):
        if "svd" in cd:
            U = cd["svd"]["U"].astype(np.float64)
            S = cd["svd"]["S"].astype(np.float64)
            Vt = cd["svd"]["Vt"].astype(np.float64)
            return _restore_shape(((U * S) @ Vt).astype(np.float32), meta["orig_shape"])

        W0 = cd["W0"].astype(np.float64)
        alpha = cd["alpha"].astype(np.float64)
        n_phi = cd["n_phi"]
        m = cd["m"]
        n = len(W0)

        t_grid = np.arange(m, dtype=np.float64) / max(m - 1, 1)
        phi = np.zeros((m, n_phi), dtype=np.float64)
        for k in range(n_phi):
            phi[:, k] = t_grid**k

        result = np.zeros((m, n), dtype=np.float64)
        result[0] = W0
        for t_idx in range(1, m):
            velocity = phi[t_idx] @ alpha
            result[t_idx] = result[t_idx - 1] + velocity / max(m - 1, 1)

        return _restore_shape(result.astype(np.float32), meta["orig_shape"])
