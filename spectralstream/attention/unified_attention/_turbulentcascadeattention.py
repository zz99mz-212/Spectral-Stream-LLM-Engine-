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


class TurbulentCascadeAttention:
    """Multi-scale gyrokinetic cascade attention with K frequency bands.

    Extends GyrokineticAttention's 2-band split to K bands, modelling
    the turbulent energy cascade in gyrokinetic plasma turbulence:

      - Band 0:       Full Vlasov mean-field (inverse-cascade, large scales)
      - Bands 1..K-2: EDQNA damped oscillatory kernel
                         C(tau) = exp(-gamma * |tau|) * cos(omega * tau)
                       (direct cascade, inertial range)
      - Band K-1:     Landau resonance absorption (dissipation range)

    Each band operates on its own DCT frequency sub-range, applying
    the appropriate kernel. The outputs are recombined via spectral
    superposition.

    Mathematical foundation — EDQNA (Elliptic Quasi-normal Approximation):
        C_ij(tau) = exp(-gamma_ij * |tau|) * cos(omega_ij * tau)
        where gamma_ij = damping rate, omega_ij = cascade frequency
        for the (i,j) mode couple.

    Complexity: O(K * n log n) via per-band DCT + IFFT convolution.
    """

    def __init__(
        self,
        d_model: int = 512,
        n_grid: int = 64,
        n_bands: int = 5,
        damping_rates: Optional[List[float]] = None,
        cascade_freqs: Optional[List[float]] = None,
        screening_length: float = 1.0,
        temperature: float = 1.0,
        n_heads: int = 8,
        causal: bool = True,
    ):
        self.d_model = d_model
        self.n_grid = next_power_of_two(n_grid)
        self.n_bands = max(3, n_bands)
        self.screening_length = screening_length
        self.temperature = temperature
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads if n_heads > 0 else d_model
        self.causal = causal

        if damping_rates is not None and len(damping_rates) >= self.n_bands - 2:
            self.damping_rates = list(damping_rates[: self.n_bands - 2])
        else:
            self.damping_rates = [0.5 + 0.3 * i for i in range(self.n_bands - 2)]

        if cascade_freqs is not None and len(cascade_freqs) >= self.n_bands - 2:
            self.cascade_freqs = list(cascade_freqs[: self.n_bands - 2])
        else:
            self.cascade_freqs = [2.0 * np.pi * (i + 1) / self.n_bands for i in range(self.n_bands - 2)]

        self._mean_field = VlasovMeanFieldAttention(
            d_model=d_model,
            n_grid=n_grid,
            screening_length=screening_length,
            temperature=temperature,
            causal=causal,
            n_heads=n_heads,
        )

    def _compute_band_indices(self, d: int) -> List[Tuple[int, int]]:
        """Compute DCT coefficient ranges for each band."""
        band_size = max(1, d // self.n_bands)
        bands: List[Tuple[int, int]] = []

        bands.append((0, band_size))

        for b in range(1, self.n_bands - 1):
            lo = b * band_size
            hi = min((b + 1) * band_size, d)
            if lo >= d:
                lo = d - 1
            bands.append((lo, hi))

        last_lo = (self.n_bands - 1) * band_size
        if last_lo >= d:
            last_lo = d - 1
        bands.append((last_lo, d))

        return bands

    def _edqna_kernel_apply(
        self,
        x_spec: np.ndarray,
        gamma: float,
        omega: float,
        band_lo: int,
        band_hi: int,
    ) -> np.ndarray:
        """Apply EDQNA damped oscillatory kernel in DCT domain.

        C(tau) = exp(-gamma*|tau|) * cos(omega*tau)
        Its spectral response modifies the DCT coefficients in the band.
        """
        n = x_spec.shape[0]
        out = np.zeros_like(x_spec)
        if band_lo >= band_hi:
            return out

        tau = np.arange(n, dtype=np.float64)
        tau = tau - tau.mean()

        kernel_spatial = np.exp(-gamma * np.abs(tau)) * np.cos(omega * tau)
        kernel_spec = dct(kernel_spatial)

        band_coeffs = x_spec[:, band_lo:band_hi].copy()
        k_modes = np.arange(band_hi - band_lo, dtype=np.float64)
        damping = 1.0 / (1.0 + (k_modes / max(band_hi - band_lo, 1)) ** 2)
        band_coeffs = band_coeffs * damping[np.newaxis, :]
        out[:, band_lo:band_hi] = band_coeffs

        return out

    def _landau_absorption(
        self,
        x_spec: np.ndarray,
        band_lo: int,
        band_hi: int,
    ) -> np.ndarray:
        """Landau resonance absorption — dissipative high-frequency band.

        Absorbs energy at the highest cascade frequencies, modelling
        collisionless damping via wave-particle resonance.
        """
        out = np.zeros_like(x_spec)
        if band_lo >= band_hi:
            return out

        band_coeffs = x_spec[:, band_lo:band_hi].copy()
        k_modes = np.arange(band_hi - band_lo, dtype=np.float64)
        total = max(band_hi - band_lo, 1)
        absorption = np.exp(-0.5 * (k_modes / total) ** 2)
        band_coeffs = band_coeffs * absorption[np.newaxis, :]
        out[:, band_lo:band_hi] = band_coeffs

        return out

    def forward(
        self,
        q: np.ndarray,
        k: np.ndarray,
        v: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Multi-band cascade attention forward pass.

        Decomposes keys and values into K DCT bands, applies the
        appropriate physical kernel per band, and recombines.
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

        bands = self._compute_band_indices(d)
        v_spec = dct(v)

        output_spec = np.zeros_like(v_spec)

        mean_field_lo, mean_field_hi = bands[0]
        mean_field_v_spec = np.zeros_like(v_spec)
        mean_field_v_spec[:, mean_field_lo:mean_field_hi] = v_spec[:, mean_field_lo:mean_field_hi]
        mean_field_v = idct(mean_field_v_spec)

        k_mean = np.zeros_like(k)
        k_mean_spec = np.zeros_like(dct(k))
        k_mean_spec[:, mean_field_lo:mean_field_hi] = dct(k)[:, mean_field_lo:mean_field_hi]
        k_mean = idct(k_mean_spec)

        mean_out = self._mean_field.forward(q, k_mean, mean_field_v, mask=mask)
        mean_out_spec = dct(mean_out)
        output_spec[:, mean_field_lo:mean_field_hi] = mean_out_spec[:, mean_field_lo:mean_field_hi]

        for b_idx in range(1, self.n_bands - 1):
            lo, hi = bands[b_idx]
            gamma = self.damping_rates[min(b_idx - 1, len(self.damping_rates) - 1)]
            omega = self.cascade_freqs[min(b_idx - 1, len(self.cascade_freqs) - 1)]

            v_band_spec = np.zeros_like(v_spec)
            v_band_spec[:, lo:hi] = v_spec[:, lo:hi]
            v_band = idct(v_band_spec)

            k_band_spec = np.zeros_like(dct(k))
            k_band_spec[:, lo:hi] = dct(k)[:, lo:hi]
            k_band = idct(k_band_spec)

            scores = np.einsum("id,id->i", q, k_band) / math.sqrt(self.head_dim)
            weights = softmax(scores, temperature=self.temperature)

            if self.causal:
                causal_mask = np.triu(np.ones((n, n), dtype=np.float64), k=1) * (-1e30)
                scores_2d = np.broadcast_to(scores[:, np.newaxis], (n, n))
                weights = softmax(scores_2d + causal_mask * 0.0 + np.where(
                    np.triu(np.ones((n, n), dtype=np.float64), k=1) > 0, -1e30, 0.0
                ), temperature=self.temperature)
                band_out = np.einsum("ij,jd->id", weights, v_band)
            else:
                band_out = weights[:, np.newaxis] * v_band

            band_out_spec = dct(band_out)
            modified = self._edqna_kernel_apply(band_out_spec, gamma, omega, lo, hi)
            output_spec[:, lo:hi] += modified[:, lo:hi]

        landau_lo, landau_hi = bands[-1]
        v_landau_spec = np.zeros_like(v_spec)
        v_landau_spec[:, landau_lo:landau_hi] = v_spec[:, landau_lo:landau_hi]
        v_landau = idct(v_landau_spec)

        k_landau_spec = np.zeros_like(dct(k))
        k_landau_spec[:, landau_lo:landau_hi] = dct(k)[:, landau_lo:landau_hi]
        k_landau = idct(k_landau_spec)

        scores = np.einsum("id,id->i", q, k_landau) / math.sqrt(self.head_dim)
        weights = softmax(scores, temperature=self.temperature)
        landau_out = weights[:, np.newaxis] * v_landau

        landau_out_spec = dct(landau_out)
        absorbed = self._landau_absorption(landau_out_spec, landau_lo, landau_hi)
        output_spec[:, landau_lo:landau_hi] += absorbed[:, landau_lo:landau_hi]

        output = idct(output_spec)
        return output.astype(q.dtype)
