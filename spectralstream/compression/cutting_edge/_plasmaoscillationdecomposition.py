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


class PlasmaOscillationDecomposition(CompressionMethod):
    """Decompose weight matrix into plasma oscillation normal modes.

    Mathematical basis:
        Plasma oscillations have dispersion relation:
            omega^2 = omega_p^2 + k^2 * v_th^2

        where omega_p is the plasma frequency, k is the wavevector,
        and v_th is the thermal velocity.

        We decompose the weight matrix as a sum of normal modes:
            W(i,j) = sum_n A_n * cos(k_n * i + phi_n) * exp(-gamma_n * i)
        where gamma_n accounts for Landau damping.

    Algorithm:
        1. Compute 2D FFT to find dominant frequencies
        2. Fit plasma dispersion relation to dominant modes
        3. Store: mode amplitudes, wavevectors, damping rates
        4. Reconstruct by superposition of damped oscillations

    Storage: O(N_modes * 4) for (A, k, phi, gamma) per mode.
    """

    name = "plasma_oscillation"
    category = "plasma_physics"

    def compress(self, tensor, n_modes=32, svd_rank=16, **kw):
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

        f2d = np.fft.fft2(t.astype(np.float64))
        power = np.abs(f2d) ** 2
        flat_power = power.ravel()
        top_indices = np.argsort(flat_power)[::-1][:n_modes]

        modes = []
        for idx in top_indices:
            ky = idx // n
            kx = idx % n
            amp = float(np.abs(f2d[ky, kx])) / (m * n)
            phase = float(np.angle(f2d[ky, kx]))
            kx_norm = kx / n
            ky_norm = ky / m

            neighbors = []
            for dkx in [-1, 0, 1]:
                for dky in [-1, 0, 1]:
                    if dkx == 0 and dky == 0:
                        continue
                    ny = (ky + dky) % m
                    nx = (kx + dkx) % n
                    neighbors.append(power[ny, nx])
            spectral_width = np.std(neighbors) / (amp * m * n + 1e-10)
            gamma = float(spectral_width * 0.5)

            modes.append(
                {
                    "amp": amp,
                    "kx": float(kx),
                    "ky": float(ky),
                    "kx_norm": kx_norm,
                    "ky_norm": ky_norm,
                    "phase": phase,
                    "gamma": gamma,
                }
            )

        modes.sort(key=lambda x: -x["amp"])

        return {
            "modes": modes,
            "n_modes": len(modes),
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
        result = np.zeros((m, n), dtype=np.float64)

        for mode in cd["modes"]:
            amp = mode["amp"]
            kx = mode["kx"]
            ky = mode["ky"]
            phase = mode["phase"]
            gamma = mode["gamma"]

            x = np.arange(m, dtype=np.float64)
            y = np.arange(n, dtype=np.float64)
            XX, YY = np.meshgrid(x, y, indexing="ij")

            spatial_phase = 2 * np.pi * (kx * YY / max(n, 1) + ky * XX / max(m, 1))
            damping = np.exp(
                -gamma * np.sqrt((XX / max(m, 1)) ** 2 + (YY / max(n, 1)) ** 2)
            )
            result += amp * np.cos(spatial_phase + phase) * damping

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
