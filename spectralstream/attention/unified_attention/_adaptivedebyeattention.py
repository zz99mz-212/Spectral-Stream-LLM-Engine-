from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union

import numpy as np

from spectralstream.core.math_primitives import (
    BAND_HIGH,
    BAND_LOW,
    BAND_NORMAL,
    DCTRotator,
    HadamardRotator,
    WaveletTransform,
    apply_spectral_kernel,
    band_limit,
    dct,
    fft,
    fftfreq,
    gibbs_softmax,
    idct,
    ifft,
    next_power_of_two,
    softmax,
    spectral_entropy,
    yukawa_kernel_1d,
)


class AdaptiveDebyeAttention:
    """Self-consistent Debye-screened mean-field attention.

    Replaces the fixed screening_length lambda_D with a spatially
    varying Debye length computed self-consistently from the token
    distribution:

        lambda_D(x) = sqrt(T_eff(x) / n_eff(x))

    where:
        T_eff(x) = local temperature = variance of key power
                   (measure of kinetic energy)
        n_eff(x) = local density = charge deposition from keys
                   (measure of particle density)

    The variable-coefficient Yukawa equation:
        (laplacian - 1/lambda_D(x)^2) Phi = -rho

    is solved iteratively via Jacobi relaxation on the spectral grid.

    Physical motivation: in real plasmas, the Debye length varies
    with local temperature and density. Hot, tenuous regions have
    large Debye lengths (weak screening, long-range), while cold,
    dense regions have short Debye lengths (strong screening, local).
    """

    def __init__(
        self,
        d_model: int = 512,
        n_grid: int = 64,
        base_screening_length: float = 1.0,
        temperature: float = 1.0,
        n_jacobi_iter: int = 5,
        causal: bool = True,
        n_heads: int = 8,
    ):
        self.d_model = d_model
        self.n_grid = next_power_of_two(n_grid)
        self.base_screening_length = base_screening_length
        self.temperature = temperature
        self.n_jacobi_iter = n_jacobi_iter
        self.causal = causal
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads if n_heads > 0 else d_model

        self._kernel_cache: Optional[np.ndarray] = None

    def _compute_local_temperature(
        self,
        keys: np.ndarray,
    ) -> np.ndarray:
        """T_eff(x) = local variance of key power (kinetic energy proxy)."""
        n = keys.shape[0]
        key_power = np.sum(keys ** 2, axis=-1)

        window = max(3, n // 10)
        local_temp = np.zeros(n, dtype=np.float64)

        for i in range(n):
            lo = max(0, i - window // 2)
            hi = min(n, i + window // 2 + 1)
            local_temp[i] = float(np.var(key_power[lo:hi])) + 1e-10

        return local_temp

    def _compute_local_density(
        self,
        keys: np.ndarray,
    ) -> np.ndarray:
        """n_eff(x) = local charge density (key power deposition)."""
        n = keys.shape[0]
        key_power = np.sum(keys ** 2, axis=-1)

        rho = np.zeros(self.n_grid, dtype=np.float64)
        positions = np.linspace(0.0, 1.0 - 1.0 / max(n, 2), n)

        for i in range(n):
            xi = positions[i] * self.n_grid
            xi = np.clip(xi, 0.0, self.n_grid - 1)
            left = int(np.floor(xi))
            left = max(0, min(left, self.n_grid - 2))
            right = left + 1
            wl = 1.0 - (xi - left)
            wr = xi - left
            rho[left] += wl * key_power[i]
            rho[right] += wr * key_power[i]

        return rho

    def _compute_debye_length_grid(
        self,
        local_temp: np.ndarray,
        local_density_grid: np.ndarray,
    ) -> np.ndarray:
        """Compute lambda_D(x) = sqrt(T_eff / n_eff) on the spectral grid."""
        local_temp_grid = np.zeros(self.n_grid, dtype=np.float64)
        n = len(local_temp)
        positions = np.linspace(0.0, 1.0 - 1.0 / max(n, 2), n)

        for i in range(n):
            xi = positions[i] * self.n_grid
            xi = np.clip(xi, 0.0, self.n_grid - 1)
            left = int(np.floor(xi))
            left = max(0, min(left, self.n_grid - 2))
            right = left + 1
            wl = 1.0 - (xi - left)
            wr = xi - left
            local_temp_grid[left] += wl * local_temp[i]
            local_temp_grid[right] += wr * local_temp[i]

        safe_density = np.maximum(local_density_grid, 1e-10)
        debye_length = np.sqrt(local_temp_grid / safe_density)
        debye_length = np.clip(debye_length, 0.1 * self.base_screening_length, 10.0 * self.base_screening_length)

        return debye_length

    def _jacobi_solve(
        self,
        rho: np.ndarray,
        debye_length_grid: np.ndarray,
    ) -> np.ndarray:
        """Solve variable-coefficient Yukawa equation via Jacobi relaxation.

        (laplacian - 1/lambda_D(x)^2) Phi = -rho

        Uses finite differences on the spectral grid with Jacobi iteration.
        """
        n_grid = self.n_grid
        dx = 1.0 / n_grid

        inv_debye_sq = 1.0 / (debye_length_grid ** 2 + 1e-10)

        phi = np.zeros(n_grid, dtype=np.float64)

        for _ in range(self.n_jacobi_iter):
            phi_new = np.zeros_like(phi)
            for i in range(n_grid):
                ip = (i + 1) % n_grid
                im = (i - 1) % n_grid
                laplacian = (phi[ip] + phi[im] - 2.0 * phi[i]) / (dx * dx)
                phi_new[i] = (-rho[i] + laplacian) / (-inv_debye_sq[i] + 1e-10)
                denominator = -inv_debye_sq[i] - 2.0 / (dx * dx)
                phi_new[i] = (-rho[i] - (phi[ip] + phi[im]) / (dx * dx)) / denominator

            phi = phi_new

        return phi

    def _interpolate_grid_to_tokens(
        self,
        phi_grid: np.ndarray,
        positions: np.ndarray,
    ) -> np.ndarray:
        """Interpolate grid potential to token positions."""
        n = len(positions)
        phi = np.zeros(n, dtype=np.float64)

        for i in range(n):
            xi = positions[i] * self.n_grid
            xi = np.clip(xi, 0.0, self.n_grid - 1)
            left = int(np.floor(xi))
            left = max(0, min(left, self.n_grid - 2))
            right = left + 1
            wl = 1.0 - (xi - left)
            wr = xi - left
            phi[i] = wl * phi_grid[left] + wr * phi_grid[right]

        return phi

    def forward(
        self,
        q: np.ndarray,
        k: np.ndarray,
        v: np.ndarray,
        mask: Optional[np.ndarray] = None,
        return_debye_lengths: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Adaptive Debye attention forward pass.

        Computes spatially-varying Debye screening self-consistently
        and solves the variable-coefficient Poisson equation.
        """
        q = np.asarray(q)
        k = np.asarray(k)
        v = np.asarray(v)
        if q.ndim < 2 or k.ndim < 2 or v.ndim < 2:
            raise ValueError(f"q, k, v must be 2D arrays, got shapes {q.shape}, {k.shape}, {v.shape}")
        if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
            raise ValueError(f"q, k, v must have same length, got {q.shape[0]}, {k.shape[0]}, {v.shape[0]}")
        n = q.shape[0]
        d = v.shape[-1]

        if mask is not None:
            valid = mask.astype(np.float64)
        else:
            valid = np.ones(n, dtype=np.float64)

        local_temp = self._compute_local_temperature(k)
        local_density_grid = self._compute_local_density(k)
        debye_length_grid = self._compute_debye_length_grid(local_temp, local_density_grid)

        rho = np.zeros(self.n_grid, dtype=np.float64)
        positions = np.linspace(0.0, 1.0 - 1.0 / max(n, 2), n)

        for i in range(n):
            if valid[i] < 0.5:
                continue
            xi = positions[i] * self.n_grid
            xi = np.clip(xi, 0.0, self.n_grid - 1)
            left = int(np.floor(xi))
            left = max(0, min(left, self.n_grid - 2))
            right = left + 1
            wl = 1.0 - (xi - left)
            wr = xi - left
            power = float(np.dot(k[i], k[i])) * valid[i]
            rho[left] += wl * power
            rho[right] += wr * power

        phi_grid = self._jacobi_solve(rho, debye_length_grid)
        phi = self._interpolate_grid_to_tokens(phi_grid, positions)

        weights = gibbs_softmax(phi, temperature=self.temperature)

        output = np.zeros((n, d), dtype=np.float64)
        v_mean_global = np.average(v, weights=valid, axis=0)

        for i in range(n):
            output[i] = v[i] + weights[i] * v_mean_global

        if return_debye_lengths:
            debye_token = self._interpolate_grid_to_tokens(debye_length_grid, positions)
            return output.astype(q.dtype), debye_token

        return output.astype(q.dtype)
