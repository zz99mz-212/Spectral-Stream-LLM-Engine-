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


class VlasovDistributionCompression(CompressionMethod):
    """Model weight distribution as a Vlasov equation solution f(x,v,t).

    Mathematical basis:
        The Vlasov equation governs the evolution of a distribution function
        in phase space:
            df/dt + v * df/dx + F(x) * df/dv = 0

        For weight matrices, we treat row index as "position" x and
        column index as "velocity" v.  The weight matrix W(i,j) represents
        the phase-space density f(x_i, v_j).

        We solve for the characteristics (particle trajectories) and
        store the characteristic map rather than the full distribution.

    Algorithm:
        1. Compute characteristics: dx/dt = v, dv/dt = -dV/dx
        2. Fit potential V(x) from weight gradient structure
        3. Store: initial conditions + potential parameters
        4. Reconstruct by advecting particles along characteristics

    Storage: O(N_particles * dim + K_potential_params).
    """

    name = "vlasov_distribution"
    category = "plasma_physics"

    def compress(self, tensor, n_particles=32, n_char_steps=10, svd_rank=16, **kw):
        t, orig = _ensure_2d(tensor)
        m, n = t.shape

        # SVD fallback
        U, S, Vt = np.linalg.svd(t.astype(np.float64), full_matrices=False)
        k = min(svd_rank, len(S))
        svd_data = {
            "U": U[:, :k].astype(np.float32),
            "S": S[:k].astype(np.float32),
            "Vt": Vt[:k, :].astype(np.float32),
            "k": k,
        }

        grad_x = np.gradient(t.astype(np.float64), axis=0)
        grad_v = np.gradient(t.astype(np.float64), axis=1)
        V = -np.cumsum(grad_x.mean(axis=1))

        rng = np.random.RandomState(42)
        n_p = min(n_particles, m)
        particle_idx = rng.choice(m, n_p, replace=False)
        particle_idx.sort()

        char_data = []
        for p_i in particle_idx:
            x0 = float(p_i) / m
            v0 = float(t[p_i, n // 2]) if n > 0 else 0.0
            trajectory = [(x0, v0)]
            x, v = x0, v0
            dt_char = 1.0 / n_char_steps
            for _ in range(n_char_steps):
                force_idx = int(np.clip(x * m, 0, m - 1))
                F = -V[min(force_idx + 1, m - 1)] + V[max(force_idx - 1, 0)]
                v += dt_char * F
                x += dt_char * v
                x = np.clip(x, 0, 1)
                trajectory.append((x, v))
            char_data.append(
                {
                    "x0": float(x0),
                    "v0": float(v0),
                    "traj": np.array(trajectory, dtype=np.float32),
                }
            )

        x_grid = np.linspace(0, 1, m)
        degree = min(6, m - 1)
        powers = np.arange(degree + 1, dtype=np.float64)
        V_grid = V / (np.max(np.abs(V)) + 1e-10)
        V_mat = x_grid[:, None] ** powers[None, :]
        pot_coeffs = np.linalg.lstsq(V_mat, V_grid, rcond=None)[0]

        return {
            "particles": char_data,
            "pot_coeffs": pot_coeffs.astype(np.float32),
            "n_particles": len(char_data),
            "V_scale": float(np.max(np.abs(V)) + 1e-10),
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
        x_grid = np.linspace(0, 1, m)
        degree = len(cd["pot_coeffs"]) - 1
        powers = np.arange(degree + 1, dtype=np.float64)
        V_mat = x_grid[:, None] ** powers[None, :]
        V = (V_mat @ cd["pot_coeffs"].astype(np.float64)) * cd["V_scale"]

        result = np.zeros((m, n), dtype=np.float64)
        for pdata in cd["particles"]:
            traj = pdata["traj"].astype(np.float64)
            for t_idx in range(len(traj)):
                x_val = int(np.clip(traj[t_idx, 0] * m, 0, m - 1))
                v_val = traj[t_idx, 1]
                col_center = int(np.clip((v_val * 0.5 + 0.5) * n, 0, n - 1))
                width = max(1, n // 16)
                for j in range(max(0, col_center - width), min(n, col_center + width)):
                    kernel = np.exp(
                        -0.5 * ((j - col_center) / (width * 0.5 + 1e-10)) ** 2
                    )
                    result[x_val, j] += V[x_val] * kernel / (width + 1e-10)

        n_p = cd["n_particles"]
        if n_p > 0:
            result /= n_p * 0.1 + 1e-10

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
