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


class VlasovHelmholtzDecomposition:
    """Vlasov-Helmholtz decomposition of attention.

    Any vector field can be decomposed into:
        F = -grad(Phi) + curl(A)

    where:
        -grad(Phi)  is the irrotational (curl-free, mean-field) component
        curl(A)     is the solenoidal (divergence-free, detail) component

    In attention terms:
        Irrotational: tokens pulled toward mean field (O(n) via Poisson)
        Solenoidal: tokens maintain detailed pair-wise interactions

    The decomposition is computed via spectral projection:
        F_irr(k) = (k.F(k)) * k/|k|^2     (longitudinal projection)
        F_sol(k) = F(k) - F_irr(k)         (transverse projection)
    """

    def __init__(
        self,
        d_model: int = 512,
        irrotational_weight: float = 0.7,
        solenoidal_weight: float = 0.3,
        spectral_rank: int = 32,
    ):
        self.d_model = d_model
        self.irrotational_weight = irrotational_weight
        self.solenoidal_weight = solenoidal_weight
        self.spectral_rank = spectral_rank

    def decompose(
        self,
        field: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        n, d = field.shape
        d_fft = next_power_of_two(d)

        if d_fft > d:
            field_pad = np.zeros((n, d_fft), dtype=np.float64)
            field_pad[:, :d] = field
        else:
            field_pad = field.copy()

        f_fft = fft(field_pad, axis=-1)
        k = fftfreq(d_fft).reshape(1, -1) * 2.0 * np.pi

        k_sq = k ** 2 + 1e-30
        k_dot_f = np.sum(f_fft * k, axis=-1, keepdims=True)
        f_irr_fft = k_dot_f * k / k_sq
        f_sol_fft = f_fft - f_irr_fft

        irr = ifft(f_irr_fft, axis=-1).real[:, :d]
        sol = ifft(f_sol_fft, axis=-1).real[:, :d]

        return irr, sol

    def spectral_decompose(
        self,
        field: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        d = field.shape[-1]
        rank = min(self.spectral_rank, d)

        field_spec = dct(field)
        irr_spec = np.zeros_like(field_spec)
        sol_spec = np.zeros_like(field_spec)

        irr_spec[:, :rank] = field_spec[:, :rank]
        sol_spec[:, rank:] = field_spec[:, rank:]

        irr = idct(irr_spec)
        sol = idct(sol_spec)

        return irr, sol, field_spec

    def combine(
        self,
        irrotational: np.ndarray,
        solenoidal: np.ndarray,
    ) -> np.ndarray:
        return (
            self.irrotational_weight * irrotational
            + self.solenoidal_weight * solenoidal
        )
