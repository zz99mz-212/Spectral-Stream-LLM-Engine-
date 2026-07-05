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


class VlasovMeanFieldAttention:
    """O(n) mean-field attention via Vlasov-Poisson theory.

    Merges the best from all 4 independent implementations:
    - spectral_kv.py: DCT compression, Lloyd-Max quantizer
    - vlasov_attention.py: Yukawa screening, causal prefix scan
    - mean_field.py: 15 spectral upgrades, Hopfield energy
    - quasar_extraction.py: DCT/spectral ops, gyrokinetic splitting

    Key insight: Instead of O(n^2) pairwise dot products, we:
    1. Deposit keys onto a spectral grid (charge density)
    2. Solve Poisson in Fourier space (O(n log n))
    3. Interpolate mean-field potential back to tokens
    4. Gibbs-softmax weights from potential energy

    Mathematical foundation — Vlasov-Poisson system:
        df/dt + v . grad_x f - grad_x Phi . grad_v f = 0
        laplacian(Phi) = -rho
        rho(k) = |K(k)|^2
        Phi(k) = V(k) * rho(k)
        V(k) = 4pi / (k^2 + mu^2)  (Yukawa screening)

    Complexity: O(n) mean-field, O(n log n) with FFT convolution.
    """

    def __init__(
        self,
        d_model: int = 512,
        n_grid: int = 64,
        screening_length: float = 1.0,
        temperature: float = 1.0,
        causal: bool = True,
        use_yukawa: bool = True,
        n_heads: int = 8,
        n_kv_heads: Optional[int] = None,
    ):
        self.d_model = d_model
        self.n_grid = next_power_of_two(n_grid)
        self.screening_length = screening_length
        self.temperature = temperature
        self.causal = causal
        self.use_yukawa = use_yukawa
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads

        self.head_dim = d_model // n_heads if n_heads > 0 else d_model
        self._gqa_factor = n_heads // self.n_kv_heads if self.n_kv_heads > 0 else 1

        self._kernel: Optional[np.ndarray] = None
        self._causal_state: Optional[dict] = None

    def _get_kernel(self, n: int) -> np.ndarray:
        if self._kernel is None or len(self._kernel) != n:
            if self.use_yukawa:
                self._kernel = yukawa_kernel_1d(
                    n, self.screening_length, dx=1.0 / n,
                )
            else:
                k = fftfreq(n, d=1.0 / n)
                sigma = self.screening_length
                self._kernel = np.exp(
                    -2.0 * (np.pi * sigma * k) ** 2
                ).astype(np.float64)
        return self._kernel

    def compute_potential(
        self,
        keys: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute the mean-field potential Phi from keys.

        Steps:
          1. Deposit keys to grid -> charge density rho(x)
          2. FFT -> rho_tilde(k)
          3. Solve Poisson in spectral: Phi_tilde(k) = V_tilde(k) * rho_tilde(k)
          4. IFFT -> Phi(x)
        """
        n = keys.shape[0]

        if mask is not None:
            valid = mask.astype(np.float64)
        else:
            valid = np.ones(n, dtype=np.float64)

        positions = np.linspace(0.0, 1.0 - 1.0 / max(n, 2), n)
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
            rho[left] += wl * power * valid[i]
            rho[right] += wr * power * valid[i]

        rho_fft = fft(rho)
        kernel = self._get_kernel(self.n_grid)
        phi_fft = rho_fft * kernel
        phi = ifft(phi_fft).real
        return phi

    def interpolate_potential(
        self,
        phi_grid: np.ndarray,
        positions: np.ndarray,
    ) -> np.ndarray:
        """Interpolate mean-field potential from grid to token positions."""
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
        return_potential: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """O(n) mean-field attention forward pass."""
        q = np.asarray(q)
        k = np.asarray(k)
        v = np.asarray(v)
        if q.ndim < 2 or k.ndim < 2 or v.ndim < 2:
            raise ValueError(f"q, k, v must be 2D arrays, got shapes {q.shape}, {k.shape}, {v.shape}")
        if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
            raise ValueError(f"q, k, v must have same length, got {q.shape[0]}, {k.shape[0]}, {v.shape[0]}")
        n = q.shape[0]
        d = q.shape[-1]

        if mask is not None:
            valid = mask.astype(np.float64)
        else:
            valid = np.ones(n, dtype=np.float64)

        positions = np.linspace(0.0, 1.0 - 1.0 / max(n, 2), n)

        if self.causal:
            return self._causal_forward(q, k, v, positions, valid, return_potential)

        phi_grid = self.compute_potential(k, mask)
        phi = self.interpolate_potential(phi_grid, positions)

        weights = gibbs_softmax(phi, temperature=self.temperature)

        if d > 1:
            v_mean = np.mean(v, axis=0, keepdims=True)
            output = v + weights[:, None] * np.sum(weights[:, None] * phi[:, None], axis=0) * v_mean
        else:
            output = v + weights * np.sum(weights * phi) * np.mean(v)

        if return_potential:
            return output.astype(q.dtype), phi
        return output.astype(q.dtype)

    def _causal_forward(
        self,
        q: np.ndarray,
        k: np.ndarray,
        v: np.ndarray,
        positions: np.ndarray,
        valid: np.ndarray,
        return_potential: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Causal variant via running accumulation (prefix scan)."""
        n = q.shape[0]
        d = q.shape[-1]
        output = np.zeros_like(q, dtype=np.float64)

        rho_acc = np.zeros(self.n_grid, dtype=np.float64)
        v_acc = np.zeros((self.n_grid, d), dtype=np.float64)
        count_acc = np.zeros(self.n_grid, dtype=np.float64)
        kernel = self._get_kernel(self.n_grid)

        phi_history = [] if return_potential else None

        for i in range(n):
            if valid[i] < 0.5 and i > 0:
                output[i] = output[i - 1]
                if return_potential and phi_history:
                    phi_history.append(phi_history[-1])
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

            for j in range(d):
                v_acc[left, j] += wl * v[i, j] * valid[i]
                v_acc[right, j] += wr * v[i, j] * valid[i]
            count_acc[left] += wl * valid[i]
            count_acc[right] += wr * valid[i]

            rho_fft = fft(rho_acc)
            phi_fft = rho_fft * kernel
            phi_grid = ifft(phi_fft).real

            phi_i = wl * phi_grid[left] + wr * phi_grid[right]

            w_i = float(np.exp(-phi_i / max(self.temperature, 1e-10)))
            safe_count = max(float(np.sum(count_acc)), 1e-10)

            v_mean_i = np.sum(v_acc, axis=0) / safe_count
            output[i] = v[i] + w_i * v_mean_i

            if return_potential and phi_history is not None:
                phi_history.append(phi_grid)

        if return_potential:
            return output.astype(q.dtype), np.array(phi_history) if phi_history else phi_grid
        return output.astype(q.dtype)

    def spectral_forward(
        self,
        q: np.ndarray,
        k: np.ndarray,
        v: np.ndarray,
        spectral_rank: Optional[int] = None,
    ) -> np.ndarray:
        """Forward pass in compressed DCT domain."""
        if spectral_rank is None:
            spectral_rank = max(8, self.d_model // 8)

        n = q.shape[0]
        d = q.shape[-1]

        k_spec = dct(k)[:, :spectral_rank]
        v_spec = dct(v)[:, :spectral_rank]

        output_spec = self.forward(q[:, :spectral_rank], k_spec, v_spec)

        expanded = np.zeros((n, d), dtype=np.float64)
        expanded[:, :spectral_rank] = output_spec
        result = idct(expanded)
        return result.astype(q.dtype)

    def reset_causal_state(self):
        self._causal_state = None
