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


class InstabilityAttention:
    """Plasma instability-driven attention with auto-selection.

    Three instability mechanisms drive distinct attention patterns:

    1. Rayleigh-Taylor (RT): Content-dependent mixing at density
       interfaces. When a heavy fluid sits atop a light fluid,
       perturbations grow exponentially. In attention: tokens at
       high-gradient boundaries undergo enhanced mixing.

    2. Kelvin-Helmholtz (KH): Velocity-shear-dependent local attention.
       The local attention window width adapts to the local velocity
       shear — high shear → narrow window (KH billows), low shear
       → wide window.

    3. Drift Wave (DW): Gradient-driven long-range coupling.
       Density gradients drive propagating waves that couple
       distant tokens along the gradient direction.

    Per-token auto-selection: the dominant instability mechanism is
    selected based on local flow properties (gradient, shear, density).
    """

    def __init__(
        self,
        d_model: int = 512,
        growth_rate_rt: float = 1.0,
        shear_sensitivity_kh: float = 0.5,
        gradient_coupling_dw: float = 0.3,
        temperature: float = 1.0,
        causal: bool = True,
        n_heads: int = 8,
    ):
        self.d_model = d_model
        self.growth_rate_rt = growth_rate_rt
        self.shear_sensitivity_kh = shear_sensitivity_kh
        self.gradient_coupling_dw = gradient_coupling_dw
        self.temperature = temperature
        self.causal = causal
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads if n_heads > 0 else d_model

    def _compute_local_properties(
        self,
        x: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute local density gradient, velocity shear, and density.

        Returns:
            gradient: |df/dx| — density gradient magnitude
            shear: |dv/dx| — velocity shear magnitude
            density: |f| — local density (key power)
        """
        n = x.shape[0]
        gradient = np.zeros(n, dtype=np.float64)
        shear = np.zeros(n, dtype=np.float64)
        density = np.zeros(n, dtype=np.float64)

        for i in range(n):
            density[i] = float(np.sqrt(np.dot(x[i], x[i])))

            if i > 0 and i < n - 1:
                gradient[i] = float(np.linalg.norm(x[i + 1] - x[i - 1])) / 2.0
                shear[i] = float(np.linalg.norm(x[i + 1] - 2.0 * x[i] + x[i - 1]))
            elif i == 0 and n > 1:
                gradient[i] = float(np.linalg.norm(x[1] - x[0]))
                shear[i] = 0.0
            elif i == n - 1 and n > 1:
                gradient[i] = float(np.linalg.norm(x[n - 1] - x[n - 2]))
                shear[i] = 0.0

        return gradient, shear, density

    def _rayleigh_taylor_attention(
        self,
        q: np.ndarray,
        k: np.ndarray,
        v: np.ndarray,
        gradient: np.ndarray,
    ) -> np.ndarray:
        """RT instability: enhanced mixing at high-gradient interfaces.

        Attention weights are amplified proportionally to the local
        density gradient, modelling the Rayleigh-Taylor instability
        where perturbations grow at heavy-light interfaces.
        """
        n = q.shape[0]
        scores = np.einsum("id,jd->ij", q, k) / math.sqrt(self.head_dim)

        grad_product = np.outer(gradient, gradient)
        rt_boost = self.growth_rate_rt * grad_product / (np.max(grad_product) + 1e-10)
        scores = scores + rt_boost

        weights = softmax(scores, temperature=self.temperature)
        output = np.einsum("ij,jd->id", weights, v)

        if self.causal:
            causal_mask = np.triu(np.ones((n, n), dtype=np.float64), k=1) * (-1e30)
            scores_causal = scores + causal_mask
            weights = softmax(scores_causal, temperature=self.temperature)
            output = np.einsum("ij,jd->id", weights, v)

        return output

    def _kelvin_helmholtz_attention(
        self,
        q: np.ndarray,
        k: np.ndarray,
        v: np.ndarray,
        shear: np.ndarray,
    ) -> np.ndarray:
        """KH instability: adaptive window attention.

        The local attention window width is inversely proportional
        to the velocity shear. High shear → narrow, focused attention
        (KH billow rollers). Low shear → wide, diffuse attention.
        """
        n = q.shape[0]
        output = np.zeros_like(q, dtype=np.float64)

        max_shear = np.max(shear) + 1e-10
        normalized_shear = shear / max_shear

        for i in range(n):
            window_half = max(1, int(
                (1.0 - normalized_shear[i]) * self.shear_sensitivity_kh * n / 2.0
            ))
            lo = max(0, i - window_half)
            hi = min(n, i + window_half + 1)

            if self.causal:
                hi = i + 1

            if hi <= lo:
                output[i] = v[i]
                continue

            q_i = q[i : i + 1]
            k_win = k[lo:hi]
            v_win = v[lo:hi]

            score = np.dot(q_i.ravel(), k_win.T) / math.sqrt(self.head_dim)
            w = softmax(score, temperature=self.temperature)
            output[i] = np.dot(w, v_win)

        return output

    def _drift_wave_attention(
        self,
        q: np.ndarray,
        k: np.ndarray,
        v: np.ndarray,
        gradient: np.ndarray,
    ) -> np.ndarray:
        """Drift wave: gradient-driven long-range coupling.

        Density gradients drive propagating waves that couple tokens
        along the gradient direction with a phase shift proportional
        to the gradient magnitude.
        """
        n = q.shape[0]
        d = v.shape[-1]

        grad_phase = np.cumsum(gradient)
        phase_shift = np.exp(1j * 2.0 * np.pi * grad_phase / (np.max(grad_phase) + 1e-10))

        k_complex = k.astype(np.complex128) * phase_shift[:, np.newaxis]
        q_complex = q.astype(np.complex128)

        scores = np.einsum("id,jd->ij", q_complex, np.conj(k_complex)).real / math.sqrt(self.head_dim)

        gradient_mask = np.outer(gradient, gradient)
        gradient_mask = gradient_mask / (np.max(gradient_mask) + 1e-10)
        scores = scores + self.gradient_coupling_dw * gradient_mask

        weights = softmax(scores, temperature=self.temperature)
        output = np.einsum("ij,jd->id", weights, v)

        if self.causal:
            causal_mask = np.triu(np.ones((n, n), dtype=np.float64), k=1) * (-1e30)
            scores_causal = scores + causal_mask
            weights = softmax(scores_causal, temperature=self.temperature)
            output = np.einsum("ij,jd->id", weights, v)

        return output

    def _select_instability(
        self,
        gradient: np.ndarray,
        shear: np.ndarray,
    ) -> np.ndarray:
        """Per-token instability selection based on local dominance.

        Returns an integer array where:
            0 = Rayleigh-Taylor (gradient-dominated)
            1 = Kelvin-Helmholtz (shear-dominated)
            2 = Drift Wave (mixed, moderate gradient + shear)
        """
        n = len(gradient)
        selection = np.zeros(n, dtype=np.int32)

        max_grad = np.max(gradient) + 1e-10
        max_shear = np.max(shear) + 1e-10

        norm_grad = gradient / max_grad
        norm_shear = shear / max_shear

        for i in range(n):
            if norm_grad[i] > 0.6 and norm_grad[i] > norm_shear[i]:
                selection[i] = 0
            elif norm_shear[i] > 0.4:
                selection[i] = 1
            else:
                selection[i] = 2

        return selection

    def forward(
        self,
        q: np.ndarray,
        k: np.ndarray,
        v: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Instability attention forward pass with auto-selection.

        Each token uses the instability mechanism that dominates its
        local flow properties.
        """
        q = np.asarray(q)
        k = np.asarray(k)
        v = np.asarray(v)
        if q.ndim < 2 or k.ndim < 2 or v.ndim < 2:
            raise ValueError(f"q, k, v must be 2D arrays, got shapes {q.shape}, {k.shape}, {v.shape}")
        if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
            raise ValueError(f"q, k, v must have same length, got {q.shape[0]}, {k.shape[0]}, {v.shape[0]}")
        n = q.shape[0]

        gradient, shear, density = self._compute_local_properties(k)
        selection = self._select_instability(gradient, shear)

        output = np.zeros_like(q, dtype=np.float64)

        rt_mask = selection == 0
        kh_mask = selection == 1
        dw_mask = selection == 2

        if np.any(rt_mask):
            rt_indices = np.where(rt_mask)[0]
            q_rt = q[rt_indices]
            k_rt = k[rt_indices]
            v_rt = v[rt_indices]
            grad_rt = gradient[rt_indices]
            output[rt_indices] = self._rayleigh_taylor_attention(q_rt, k_rt, v_rt, grad_rt)

        if np.any(kh_mask):
            kh_indices = np.where(kh_mask)[0]
            q_kh = q[kh_indices]
            k_kh = k[kh_indices]
            v_kh = v[kh_indices]
            shear_kh = shear[kh_indices]
            output[kh_indices] = self._kelvin_helmholtz_attention(q_kh, k_kh, v_kh, shear_kh)

        if np.any(dw_mask):
            dw_indices = np.where(dw_mask)[0]
            q_dw = q[dw_indices]
            k_dw = k[dw_indices]
            v_dw = v[dw_indices]
            grad_dw = gradient[dw_indices]
            output[dw_indices] = self._drift_wave_attention(q_dw, k_dw, v_dw, grad_dw)

        return output.astype(q.dtype)
