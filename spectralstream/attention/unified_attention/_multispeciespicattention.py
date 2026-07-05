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


class MultiSpeciesPICAttention:
    """Multi-species Particle-in-Cell attention.

    Classifies tokens into species based on their content, position,
    and attention patterns, then applies per-species PIC dynamics:

    1. Species classification:
       - Content species: determined by key vector clustering
       - Position species: determined by spatial location
       - Attention species: determined by attention pattern similarity

    2. Per-species dynamics:
       - Charge q_s and mass m_s per species s
       - Species-specific force: F_s = q_s * (E + v_s x B)
       - Current density deposition from species motion

    3. Field solve:
       - Charge density: rho = sum_s q_s * n_s(x)
       - Current density: J = sum_s q_s * n_s(x) * v_s(x)
       - Magnetic field from current: curl(B) = mu_0 * J
       - Electric field from charge: div(E) = rho / epsilon_0

    4. Lorentz force mixing:
       - The magnetic field couples different species
       - Cross-species attention via B-field mixing term
    """

    def __init__(
        self,
        d_model: int = 512,
        n_grid: int = 64,
        n_species: int = 3,
        charge_ratios: Optional[List[float]] = None,
        mass_ratios: Optional[List[float]] = None,
        temperature: float = 1.0,
        causal: bool = True,
        n_heads: int = 8,
    ):
        self.d_model = d_model
        self.n_grid = next_power_of_two(n_grid)
        self.n_species = n_species
        self.temperature = temperature
        self.causal = causal
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads if n_heads > 0 else d_model

        if charge_ratios is not None and len(charge_ratios) >= n_species:
            self.charge_ratios = np.array(charge_ratios[:n_species], dtype=np.float64)
        else:
            self.charge_ratios = np.array([1.0, -1.0, 0.5][:n_species], dtype=np.float64)

        if mass_ratios is not None and len(mass_ratios) >= n_species:
            self.mass_ratios = np.array(mass_ratios[:n_species], dtype=np.float64)
        else:
            self.mass_ratios = np.array([1.0, 0.0005, 1836.0][:n_species], dtype=np.float64)

    def _classify_species(
        self,
        q: np.ndarray,
        k: np.ndarray,
    ) -> np.ndarray:
        """Classify tokens into species via content-based clustering.

        Uses k-means-like assignment based on key vector similarity
        to species centroids.
        """
        n = k.shape[0]
        d = k.shape[-1]

        centroids = np.zeros((self.n_species, d), dtype=np.float64)
        chunk = max(1, n // self.n_species)
        for s in range(self.n_species):
            lo = s * chunk
            hi = min((s + 1) * chunk, n) if s < self.n_species - 1 else n
            if lo < hi:
                centroids[s] = np.mean(k[lo:hi], axis=0)

        species = np.zeros(n, dtype=np.int32)
        for i in range(n):
            sims = np.array([float(np.dot(k[i], centroids[s])) for s in range(self.n_species)])
            species[i] = int(np.argmax(sims))

        return species

    def _deposit_species_charge(
        self,
        keys: np.ndarray,
        species: np.ndarray,
        positions: np.ndarray,
    ) -> np.ndarray:
        """Deposit charge density per species onto the grid.

        Returns charge density array of shape (n_species, n_grid).
        """
        rho_species = np.zeros((self.n_species, self.n_grid), dtype=np.float64)

        for s in range(self.n_species):
            mask = species == s
            if not np.any(mask):
                continue
            s_indices = np.where(mask)[0]
            for i in s_indices:
                xi = positions[i] * self.n_grid
                xi = np.clip(xi, 0.0, self.n_grid - 1)
                left = int(np.floor(xi))
                left = max(0, min(left, self.n_grid - 2))
                right = left + 1
                wl = 1.0 - (xi - left)
                wr = xi - left
                power = float(np.dot(keys[i], keys[i]))
                rho_species[s, left] += wl * power
                rho_species[s, right] += wr * power

        return rho_species

    def _deposit_current(
        self,
        keys: np.ndarray,
        values: np.ndarray,
        species: np.ndarray,
        positions: np.ndarray,
    ) -> np.ndarray:
        """Deposit current density J = q * n * v per species.

        Returns current density of shape (n_species, n_grid).
        """
        J_species = np.zeros((self.n_species, self.n_grid), dtype=np.float64)

        for s in range(self.n_species):
            mask = species == s
            if not np.any(mask):
                continue
            s_indices = np.where(mask)[0]
            for i in s_indices:
                xi = positions[i] * self.n_grid
                xi = np.clip(xi, 0.0, self.n_grid - 1)
                left = int(np.floor(xi))
                left = max(0, min(left, self.n_grid - 2))
                right = left + 1
                wl = 1.0 - (xi - left)
                wr = xi - left
                velocity = float(np.mean(values[i]))
                J_species[s, left] += wl * velocity * self.charge_ratios[s]
                J_species[s, right] += wr * velocity * self.charge_ratios[s]

        return J_species

    def _solve_fields(
        self,
        rho_species: np.ndarray,
        J_species: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Solve for electric and magnetic fields.

        E = -grad(phi) where laplacian(phi) = -rho_total
        B from current: B_tilde(k) = mu_0 * J_tilde(k) / (ik)

        Returns:
            E_field: (n_grid,) electric field
            B_field: (n_grid,) magnetic field
        """
        rho_total = np.sum(rho_species * self.charge_ratios[:, np.newaxis], axis=0)
        J_total = np.sum(J_species, axis=0)

        mu = 1.0
        k_freq = fftfreq(self.n_grid, d=1.0 / self.n_grid)
        k_sq = (2.0 * np.pi * k_freq) ** 2 + mu ** 2

        rho_fft = fft(rho_total)
        phi_fft = 4.0 * np.pi * rho_fft / (k_sq + 1e-30)
        phi_grid = ifft(phi_fft).real

        E_field = np.zeros(self.n_grid, dtype=np.float64)
        for i in range(self.n_grid):
            ip = (i + 1) % self.n_grid
            im = (i - 1) % self.n_grid
            E_field[i] = -(phi_grid[ip] - phi_grid[im]) / (2.0 / self.n_grid)

        J_fft = fft(J_total)
        B_fft = J_fft / (1j * 2.0 * np.pi * k_freq + 1e-10)
        B_field = ifft(B_fft).real

        return E_field, B_field

    def _lorentz_force_mix(
        self,
        species_outputs: List[np.ndarray],
        E_field: np.ndarray,
        B_field: np.ndarray,
        species: np.ndarray,
        positions: np.ndarray,
    ) -> np.ndarray:
        """Apply Lorentz force mixing between species.

        F_s = q_s * (E + v_s x B) produces cross-species coupling
        via the shared E and B fields.
        """
        n = len(species)
        d = species_outputs[0].shape[-1]
        output = np.zeros((n, d), dtype=np.float64)

        for i in range(n):
            s = species[i]
            xi = positions[i] * self.n_grid
            xi = np.clip(xi, 0.0, self.n_grid - 1)
            left = int(np.floor(xi))
            left = max(0, min(left, self.n_grid - 2))
            right = left + 1
            wl = 1.0 - (xi - left)
            wr = xi - left

            E_i = wl * E_field[left] + wr * E_field[right]
            B_i = wl * B_field[left] + wr * B_field[right]

            v_s = species_outputs[s][i]
            q_s = self.charge_ratios[s]

            force = q_s * E_i * np.sign(v_s + 1e-10)
            force = force + q_s * B_i * np.roll(v_s, 1) * 0.01

            output[i] = v_s + force * 0.1

        return output

    def forward(
        self,
        q: np.ndarray,
        k: np.ndarray,
        v: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Multi-species PIC attention forward pass.

        Classifies tokens, deposits species-resolved charge and current,
        solves for E and B fields, and applies Lorentz force mixing.
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

        species = self._classify_species(q, k)
        positions = np.linspace(0.0, 1.0 - 1.0 / max(n, 2), n)

        rho_species = self._deposit_species_charge(k, species, positions)
        J_species = self._deposit_current(k, v, species, positions)

        E_field, B_field = self._solve_fields(rho_species, J_species)

        species_outputs: List[np.ndarray] = [np.zeros((n, d), dtype=np.float64) for _ in range(self.n_species)]

        for s in range(self.n_species):
            mask_s = species == s
            if not np.any(mask_s):
                continue

            s_indices = np.where(mask_s)[0]
            q_s = q[s_indices]
            k_s = k[s_indices]
            v_s = v[s_indices]

            scores = np.einsum("id,jd->ij", q_s, k_s) / math.sqrt(self.head_dim)
            weights = softmax(scores, temperature=self.temperature)

            if self.causal:
                causal_mask = np.triu(np.ones((len(s_indices), len(s_indices)), dtype=np.float64), k=1) * (-1e30)
                scores_causal = scores + causal_mask
                weights = softmax(scores_causal, temperature=self.temperature)

            s_out = np.einsum("ij,jd->id", weights, v_s)
            species_outputs[s][s_indices] = s_out

        output = self._lorentz_force_mix(species_outputs, E_field, B_field, species, positions)

        return output.astype(q.dtype)
