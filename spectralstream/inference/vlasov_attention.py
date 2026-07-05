"""
Vlasov Mean-Field O(n) Attention for SpectralStream
=====================================================
Physics-inspired attention mechanism based on Vlasov-Poisson
plasma dynamics. Achieves O(n) complexity via mean-field
approximation instead of O(n^2) pairwise dot products.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from spectralstream.core.math_primitives import (
    fft,
    fftfreq,
    gibbs_softmax,
    ifft,
    next_power_of_two,
    softmax,
    spectral_entropy,
)

logger = logging.getLogger(__name__)


@dataclass
class VlasovConfig:
    """Configuration for Vlasov attention."""

    n_grid: int = 64
    screening_length: float = 1.0
    temperature: float = 1.0
    causal: bool = True
    use_yukawa: bool = True
    n_heads: int = 8
    n_kv_heads: Optional[int] = None
    deposition_method: str = "linear"
    max_pic_steps: int = 10
    pic_tolerance: float = 1e-4


class VlasovAttention:
    """Vlasov mean-field attention via Poisson equation on a spectral grid.

    Instead of computing O(n^2) attention weights between all token pairs,
    this method:
      1. Deposits key charge density onto a 1D spectral grid
      2. Solves Poisson in Fourier space: Phi(k) = V(k) * rho(k)
      3. Interpolates the potential back to token positions
      4. Uses the potential energy as attention weights
    """

    def __init__(self, config: Optional[VlasovConfig] = None):
        self.config = config or VlasovConfig()
        self.n_grid = next_power_of_two(self.config.n_grid)
        self.n_heads = self.config.n_heads
        self.n_kv_heads = self.config.n_kv_heads or self.n_heads
        self.head_dim = 0
        self._kernel_cache: dict[int, np.ndarray] = {}

    def _get_kernel(self, n: int) -> np.ndarray:
        if n not in self._kernel_cache:
            k = fftfreq(n, d=1.0 / n)
            if self.config.use_yukawa:
                mu = 1.0 / max(self.config.screening_length, 1e-10)
                k_sq = (2.0 * np.pi * k) ** 2
                kernel = 4.0 * np.pi / (k_sq + mu**2 + 1e-30)
                kernel[0] = 4.0 * np.pi / (mu**2)
            else:
                sigma = self.config.screening_length
                kernel = np.exp(-2.0 * (np.pi * sigma * k) ** 2).astype(np.float64)
            self._kernel_cache[n] = kernel
        return self._kernel_cache[n]

    def _deposit_charge(
        self,
        keys: np.ndarray,
        positions: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        n = keys.shape[0]
        valid = (
            mask.astype(np.float64)
            if mask is not None
            else np.ones(n, dtype=np.float64)
        )
        rho = np.zeros(self.n_grid, dtype=np.float64)
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
            power = float(np.dot(keys[i], keys[i]))
            if self.config.deposition_method == "linear":
                rho[left] += wl * power * valid[i]
                rho[right] += wr * power * valid[i]
            else:
                idx = int(np.round(xi)) % self.n_grid
                rho[idx] += power * valid[i]
        return rho

    def _solve_poisson(self, rho: np.ndarray) -> np.ndarray:
        rho_fft = fft(rho)
        kernel = self._get_kernel(self.n_grid)
        phi_fft = rho_fft * kernel
        return ifft(phi_fft).real

    def _interpolate_to_tokens(
        self,
        phi_grid: np.ndarray,
        positions: np.ndarray,
    ) -> np.ndarray:
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
    ) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64)
        k = np.asarray(k, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        if q.ndim < 2:
            raise ValueError(f"q must be 2D, got shape {q.shape}")
        if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
            raise ValueError("q, k, v must have the same length")
        n = q.shape[0]
        d = q.shape[-1]
        positions = np.linspace(0.0, 1.0 - 1.0 / max(n, 2), n)
        rho = self._deposit_charge(k, positions, mask)
        phi_grid = self._solve_poisson(rho)
        phi = self._interpolate_to_tokens(phi_grid, positions)
        weights = gibbs_softmax(phi, temperature=self.config.temperature)
        if d > 1:
            v_mean = np.mean(v, axis=0, keepdims=True)
            output = (
                v
                + weights[:, None]
                * np.sum(weights[:, None] * phi[:, None], axis=0)
                * v_mean
            )
        else:
            output = v + weights * np.sum(weights * phi) * np.mean(v)
        return output.astype(q.dtype)

    def forward_causal(
        self,
        q: np.ndarray,
        k: np.ndarray,
        v: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64)
        k = np.asarray(k, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        n = q.shape[0]
        d = q.shape[-1]
        output = np.zeros_like(q, dtype=np.float64)
        positions = np.linspace(0.0, 1.0 - 1.0 / max(n, 2), n)
        valid = (
            np.ones(n, dtype=np.float64) if mask is None else mask.astype(np.float64)
        )
        kernel = self._get_kernel(self.n_grid)
        rho_acc = np.zeros(self.n_grid, dtype=np.float64)
        for i in range(n):
            if valid[i] < 0.5 and i > 0:
                output[i] = output[i - 1]
                continue
            xi = positions[i] * self.n_grid
            xi = np.clip(xi, 0.0, self.n_grid - 1)
            left = int(np.floor(xi))
            left = max(0, min(left, self.n_grid - 2))
            right = left + 1
            wl = 1.0 - (xi - left)
            wr = xi - left
            power = float(np.dot(k[i], k[i]))
            rho_acc[left] += wl * power * valid[i]
            rho_acc[right] += wr * power * valid[i]
            rho_fft = fft(rho_acc)
            phi_fft = rho_fft * kernel
            phi_grid = ifft(phi_fft).real
            phi_i = wl * phi_grid[left] + wr * phi_grid[right]
            w_i = float(np.exp(-phi_i / max(self.config.temperature, 1e-10)))
            v_mean_i = np.mean(v[: i + 1], axis=0) if i > 0 else v[0]
            output[i] = v[i] + w_i * v_mean_i
        return output.astype(q.dtype)


@dataclass
class PICStep:
    """A single PIC simulation step."""

    step_id: int
    rho: np.ndarray
    phi: np.ndarray
    potential_energy: float
    kinetic_energy: float
    converged: bool


class VlasovPICScheduler:
    """Particle-in-Cell scheduler for batched Vlasov attention.

    Manages multiple PIC simulations (one per attention head or batch
    element), scheduling field solves and particle pushes to maximize
    throughput while maintaining accuracy.
    """

    def __init__(self, config: Optional[VlasovConfig] = None):
        self.config = config or VlasovConfig()
        self._vlasov = VlasovAttention(config)
        self._step_history: List[List[PICStep]] = []

    def _interpolate_phi(
        self,
        phi_grid: np.ndarray,
        xi: float,
        n_grid: int,
    ) -> float:
        xi_c = np.clip(xi, 0.0, n_grid - 1)
        left = int(np.floor(xi_c))
        left = max(0, min(left, n_grid - 2))
        right = left + 1
        wl = 1.0 - (xi_c - left)
        wr = xi_c - left
        return float(wl * phi_grid[left] + wr * phi_grid[right])

    def _compute_energy(
        self,
        positions: np.ndarray,
        charges: np.ndarray,
        phi_grid: np.ndarray,
        velocities: np.ndarray,
        n_grid: int,
    ) -> Tuple[float, float]:
        pe = 0.0
        for i in range(len(positions)):
            xi = positions[i] * n_grid
            phi_i = self._interpolate_phi(phi_grid, xi, n_grid)
            pe += charges[i] * phi_i
        ke = 0.5 * float(np.sum(velocities**2))
        return pe, ke

    def run_pic_simulation(
        self,
        q: np.ndarray,
        k: np.ndarray,
        v: np.ndarray,
        n_steps: Optional[int] = None,
        mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, List[PICStep]]:
        q = np.asarray(q, dtype=np.float64)
        k = np.asarray(k, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        n_steps_actual = n_steps or self.config.max_pic_steps
        n = q.shape[0]
        d = q.shape[-1]
        positions = np.linspace(0.0, 1.0 - 1.0 / max(n, 2), n)
        charges = np.linalg.norm(k, axis=1) ** 2
        velocities = np.random.RandomState(42).randn(n, d) * 0.01
        valid = (
            np.ones(n, dtype=np.float64) if mask is None else mask.astype(np.float64)
        )
        n_grid = self._vlasov.n_grid
        kernel = self._vlasov._get_kernel(n_grid)
        steps: List[PICStep] = []
        rho_acc = np.zeros(n_grid, dtype=np.float64)

        for step in range(n_steps_actual):
            rho = np.zeros(n_grid, dtype=np.float64)
            for i in range(n):
                if valid[i] < 0.5:
                    continue
                xi = positions[i] * n_grid
                xi = np.clip(xi, 0.0, n_grid - 1)
                left = int(np.floor(xi))
                left = max(0, min(left, n_grid - 2))
                right = left + 1
                wl = 1.0 - (xi - left)
                wr = xi - left
                rho[left] += wl * charges[i]
                rho[right] += wr * charges[i]

            rho_acc += rho
            rho_fft = fft(rho_acc)
            phi_fft = rho_fft * kernel
            phi_grid = ifft(phi_fft).real
            pe, ke = self._compute_energy(
                positions, charges, phi_grid, velocities, n_grid
            )

            for i in range(n):
                if valid[i] < 0.5:
                    continue
                xi = positions[i] * n_grid
                phi_i = self._interpolate_phi(phi_grid, xi, n_grid)
                force = -charges[i] * phi_i * 0.01
                velocities[i] += force * np.sign(velocities[i] + 1e-10)
                positions[i] += float(np.mean(velocities[i])) * 0.001
                positions[i] = np.clip(positions[i], 0.0, 1.0)

            converged = (
                step > 0
                and abs(pe - steps[-1].potential_energy) < self.config.pic_tolerance
            )
            steps.append(
                PICStep(
                    step_id=step,
                    rho=rho_acc.copy(),
                    phi=phi_grid.copy(),
                    potential_energy=pe,
                    kinetic_energy=ke,
                    converged=converged,
                )
            )
            if converged:
                break

        output = np.zeros_like(q, dtype=np.float64)
        phi_final = steps[-1].phi if steps else np.zeros(n_grid)
        for i in range(n):
            xi = positions[i] * n_grid
            phi_i = self._interpolate_phi(phi_final, xi, n_grid)
            w_i = float(np.exp(-phi_i / max(self.config.temperature, 1e-10)))
            output[i] = v[i] + w_i * np.mean(v, axis=0)

        return output.astype(q.dtype), steps

    def get_total_energy_history(self) -> List[float]:
        history = []
        for sim_steps in self._step_history:
            for step in sim_steps:
                history.append(step.potential_energy + step.kinetic_energy)
        return history

    def reset(self):
        self._step_history.clear()
