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


class GyrokineticAttention:
    """Gyrokinetic frequency-split attention.

    Separates fast cyclotron motion (high DCT frequencies) from
    slow drift motion (low DCT frequencies):

      - Slow component: full mean-field attention (captures broad structure)
      - Fast component: simplified local kernel (captures fine detail)
    """

    def __init__(
        self,
        d_model: int = 512,
        n_grid: int = 64,
        split_fraction: float = 0.3,
        fast_kernel_width: float = 0.1,
        screening_length: float = 1.0,
        temperature: float = 1.0,
        n_heads: int = 8,
        causal: bool = True,
    ):
        self.d_model = d_model
        self.n_grid = n_grid
        self.split_fraction = split_fraction
        self.fast_kernel_width = fast_kernel_width
        self.screening_length = screening_length
        self.temperature = temperature
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.causal_flag = causal

        self.slow_attn = VlasovMeanFieldAttention(
            d_model=d_model,
            n_grid=n_grid,
            screening_length=screening_length,
            temperature=temperature,
            causal=causal,
            n_heads=n_heads,
        )

    def _gyrokinetic_split(
        self,
        k: np.ndarray,
        v: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n_slow = max(1, int(self.head_dim * self.split_fraction))

        k_spec = dct(k)
        v_spec = dct(v)

        k_slow_spec = np.zeros_like(k_spec)
        v_slow_spec = np.zeros_like(v_spec)
        k_fast_spec = np.zeros_like(k_spec)
        v_fast_spec = np.zeros_like(v_spec)

        k_slow_spec[:, :n_slow] = k_spec[:, :n_slow]
        v_slow_spec[:, :n_slow] = v_spec[:, :n_slow]
        k_fast_spec[:, n_slow:] = k_spec[:, n_slow:]
        v_fast_spec[:, n_slow:] = v_spec[:, n_slow:]

        k_slow = idct(k_slow_spec)
        v_slow = idct(v_slow_spec)
        k_fast = idct(k_fast_spec)
        v_fast = idct(v_fast_spec)

        return k_slow, v_slow, k_fast, v_fast

    def _local_fast_attention(
        self,
        q_fast: np.ndarray,
        k_fast: np.ndarray,
        v_fast: np.ndarray,
        positions: np.ndarray,
    ) -> np.ndarray:
        n = q_fast.shape[0]
        d = q_fast.shape[-1]

        if n < 2:
            return v_fast

        local_window = max(3, int(1.0 / max(self.fast_kernel_width, 0.01)))
        output = np.zeros_like(q_fast, dtype=np.float64)

        for i in range(n):
            start = max(0, i - local_window)
            end = min(n, i + local_window + 1)
            window = end - start

            if window <= 1:
                output[i] = v_fast[i]
                continue

            dist = np.abs(positions[i] - positions[start:end])
            weights = np.exp(-0.5 * (dist / max(self.fast_kernel_width, 1e-10)) ** 2)

            if self.causal_flag:
                weights[i - start + 1:] = 0.0

            w_sum = np.sum(weights) + 1e-30
            weights = weights / w_sum

            output[i] = np.tensordot(weights, v_fast[start:end], axes=1)

        return output

    def forward(
        self,
        q: np.ndarray,
        k: np.ndarray,
        v: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Gyrokinetic attention forward pass with fast/slow frequency splitting."""
        q = np.asarray(q)
        k = np.asarray(k)
        v = np.asarray(v)
        if q.ndim < 2 or k.ndim < 2 or v.ndim < 2:
            raise ValueError(f"q, k, v must be 2D arrays, got shapes {q.shape}, {k.shape}, {v.shape}")
        if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
            raise ValueError(f"q, k, v must have same length, got {q.shape[0]}, {k.shape[0]}, {v.shape[0]}")
        n = q.shape[0]

        if mask is not None:
            valid = mask.astype(np.float64)
        else:
            valid = np.ones(n, dtype=np.float64)

        k_slow, v_slow, k_fast, v_fast = self._gyrokinetic_split(k, v)

        slow_output = self.slow_attn.forward(q, k_slow, v_slow, mask=mask)

        positions = np.linspace(0.0, 1.0 - 1.0 / max(n, 2), n)
        fast_output = self._local_fast_attention(q, k_fast, v_fast, positions)

        slow_spec = dct(slow_output)
        fast_spec = dct(fast_output)

        output_spec = slow_spec + fast_spec
        output = idct(output_spec)

        return output.astype(q.dtype)
