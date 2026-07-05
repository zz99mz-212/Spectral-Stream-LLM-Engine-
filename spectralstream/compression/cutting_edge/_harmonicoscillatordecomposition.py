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


class HarmonicOscillatorDecomposition(CompressionMethod):
    """Decompose weight matrix as sum of 2D harmonic oscillators.

    Mathematical basis:
        W(i,j) = sum_{k=1}^{K} A_k * cos(omega_k * i + phi_k) * cos(omega_k * j + psi_k)

        This represents the weight matrix as a superposition of standing
        waves (harmonic oscillators) with different frequencies, phases,
        and amplitudes.

    Algorithm:
        1. 2D DFT to find dominant frequencies
        2. For each dominant frequency, fit amplitude and phase
        3. Store: (amplitude, frequency, phase_x, phase_y) per mode

    Storage: O(K * 5) where K = number of modes.
    """

    name = "harmonic_oscillator"
    category = "hybrid"

    def compress(self, tensor, n_modes=32, svd_rank=16, **kw):
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

        # 2D DFT
        F = np.fft.fft2(W)
        power = np.abs(F)
        flat_power = power.ravel()
        top_idx = np.argsort(flat_power)[::-1][:n_modes]

        modes = []
        for idx in top_idx:
            ky = idx // n
            kx = idx % n
            amplitude = float(np.abs(F[ky, kx])) / (m * n)
            phase = float(np.angle(F[ky, kx]))
            freq_x = kx / n
            freq_y = ky / m

            neighbors = []
            for dkx in [-1, 0, 1]:
                for dky in [-1, 0, 1]:
                    if dkx == 0 and dky == 0:
                        continue
                    ny = (ky + dky) % m
                    nx = (kx + dkx) % n
                    neighbors.append(power[ny, nx])
            spectral_width = np.std(neighbors) / (power[ky, kx] + 1e-10)
            damping = float(spectral_width * 0.5)

            modes.append(
                {
                    "amplitude": amplitude,
                    "freq_x": freq_x,
                    "freq_y": freq_y,
                    "phase": phase,
                    "damping": damping,
                }
            )

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
        x = np.arange(m, dtype=np.float64)
        y = np.arange(n, dtype=np.float64)
        XX, YY = np.meshgrid(x, y, indexing="ij")

        for mode in cd["modes"]:
            A = mode["amplitude"]
            fx = mode["freq_x"]
            fy = mode["freq_y"]
            phi = mode["phase"]
            gamma = mode["damping"]
            spatial = np.cos(2 * np.pi * fx * YY + phi) * np.cos(
                2 * np.pi * fy * XX + phi
            )
            damping = np.exp(
                -gamma * np.sqrt((XX / max(m, 1)) ** 2 + (YY / max(n, 1)) ** 2)
            )
            result += A * spatial * damping

        return _restore_shape(result.astype(np.float32), meta["orig_shape"])
