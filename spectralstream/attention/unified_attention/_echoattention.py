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


class EchoAttention:
    """Plasma echo attention for long-range dependency recovery.

    In plasma physics, an echo occurs when two successive pulses
    deposited in phase-space at different times produce a macroscopic
    signal at a later time, even when the individual distributions
    have phase-mixed to invisibility.

    Mechanism:
      1. First pulse:  deposit token i's key as a phase-space structure
         f_i(x,v) ~ delta(v - v_i) * phi(x - x_i)
      2. Second pulse: correlate token j's query with the deposited
         structure, producing a beaten signal
      3. Echo:         token k recovers token i's information through
         the intermediary j, even when direct attention A(k,i) ~ 0

    This enables O(n * n_echo) long-range information flow where
    n_echo << n tokens participate in the echo relay.

    Mathematical foundation — Vlasov echo:
        f_echo(x,v,t) = integral dx1 dv1 f1(x1,v1) * f2(x1,v1,t) *
                        delta(v - v_echo(x,x1,t))
        where v_echo = v + grad_x(Phi_1 + Phi_2) * t
    """

    def __init__(
        self,
        d_model: int = 512,
        n_echo: int = 64,
        n_grid: int = 64,
        temperature: float = 1.0,
        causal: bool = True,
        n_heads: int = 8,
    ):
        self.d_model = d_model
        self.n_echo = n_echo
        self.n_grid = next_power_of_two(n_grid)
        self.temperature = temperature
        self.causal = causal
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads if n_heads > 0 else d_model

        self._echo_buffer: Optional[np.ndarray] = None
        self._echo_positions: Optional[np.ndarray] = None

    def _deposit_phase_space(
        self,
        keys: np.ndarray,
        positions: np.ndarray,
    ) -> np.ndarray:
        """Deposit key vectors onto a 2D phase-space grid (x, v).

        Creates a charge distribution rho(x, v) from key vectors.
        """
        n = keys.shape[0]
        d = keys.shape[-1]
        grid_x = self.n_grid
        grid_v = min(self.n_grid, d)

        rho = np.zeros((grid_x, grid_v), dtype=np.float64)

        for i in range(n):
            xi = positions[i] * grid_x
            xi = np.clip(xi, 0.0, grid_x - 1)
            left_x = int(np.floor(xi))
            left_x = max(0, min(left_x, grid_x - 2))
            right_x = left_x + 1
            wl_x = 1.0 - (xi - left_x)
            wr_x = xi - left_x

            vi = keys[i, :grid_v]
            vi_norm = vi / (np.linalg.norm(vi) + 1e-10)

            for dv in range(grid_v):
                val = vi_norm[dv] * float(np.dot(keys[i], keys[i]))
                rho[left_x, dv] += wl_x * val
                rho[right_x, dv] += wr_x * val

        return rho

    def _echo_correlate(
        self,
        queries: np.ndarray,
        rho: np.ndarray,
        positions: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Correlate queries with deposited phase-space structure.

        Returns echo-amplified values and the echo weights.
        """
        n = queries.shape[0]
        grid_x, grid_v = rho.shape

        rho_fft_v = fft(rho, axis=1)
        echo_signal = ifft(rho_fft_v, axis=1).real

        echo_scores = np.zeros(n, dtype=np.float64)
        echo_values = np.zeros((n, grid_v), dtype=np.float64)

        for i in range(n):
            xi = positions[i] * grid_x
            xi = np.clip(xi, 0.0, grid_x - 1)
            left_x = int(np.floor(xi))
            left_x = max(0, min(left_x, grid_x - 2))
            right_x = left_x + 1
            wl_x = 1.0 - (xi - left_x)
            wr_x = xi - left_x

            local_signal = wl_x * echo_signal[left_x] + wr_x * echo_signal[right_x]
            local_rho = wl_x * rho[left_x] + wr_x * rho[right_x]

            qi = queries[i, :grid_v]
            echo_scores[i] = float(np.dot(qi, local_signal))
            echo_values[i] = local_rho * qi

        return echo_values, echo_scores

    def _reconstruct_echo(
        self,
        echo_values: np.ndarray,
        echo_scores: np.ndarray,
        v: np.ndarray,
    ) -> np.ndarray:
        """Reconstruct output from echo signals.

        Combines the echo-recovered information with direct values.
        """
        n = v.shape[0]
        d = v.shape[-1]
        grid_v = echo_values.shape[1]

        weights = softmax(echo_scores, temperature=self.temperature)

        output = np.zeros((n, d), dtype=np.float64)
        for i in range(n):
            echo_contrib = np.zeros(d, dtype=np.float64)
            echo_contrib[:grid_v] = echo_values[i] * weights[i]

            output[i] = v[i] + echo_contrib

        return output

    def forward(
        self,
        q: np.ndarray,
        k: np.ndarray,
        v: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Echo attention forward pass.

        Complexity: O(n * n_echo) where n_echo = n for full echo,
        or O(n * n_echo) with n_echo << n for sparse echo relay.
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

        positions = np.linspace(0.0, 1.0 - 1.0 / max(n, 2), n)

        n_echo_actual = min(self.n_echo, n)
        echo_idx = np.linspace(0, n - 1, n_echo_actual, dtype=int)

        k_echo = k[echo_idx]
        positions_echo = positions[echo_idx]

        rho = self._deposit_phase_space(k_echo, positions_echo)

        echo_values, echo_scores = self._echo_correlate(q, rho, positions)

        output = self._reconstruct_echo(echo_values, echo_scores, v)

        if self.causal:
            causal_weights = np.ones((n, n), dtype=np.float64)
            causal_weights = np.triu(causal_weights, k=1) * (-1e30)
            direct_scores = np.einsum("id,jd->ij", q, k) / math.sqrt(self.head_dim)
            direct_attn = softmax(direct_scores + causal_weights, temperature=self.temperature)
            direct_out = np.einsum("ij,jd->id", direct_attn, v)
            output = 0.5 * output + 0.5 * direct_out
        else:
            direct_scores = np.einsum("id,jd->ij", q, k) / math.sqrt(self.head_dim)
            direct_attn = softmax(direct_scores, temperature=self.temperature)
            direct_out = np.einsum("ij,jd->id", direct_attn, v)
            output = 0.5 * output + 0.5 * direct_out

        return output.astype(q.dtype)

    def reset_echo_state(self):
        """Reset the echo buffer."""
        self._echo_buffer = None
        self._echo_positions = None
