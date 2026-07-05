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


class UnifiedAttentionSelector:
    """Automatically selects the optimal attention strategy based on
    sequence length, hardware constraints, and accuracy requirements.

    Strategy selection:
      - n < 512:     Standard Vlasov mean-field (full accuracy)
      - 512 <= n < 4096:  Vlasov with spectral compression
      - 4096 <= n < 32768: Flash Vlasov (tiled, memory-efficient)
      - n >= 32768:  Gyrokinetic (fast/slow split for speed)
    """

    def __init__(
        self,
        d_model: int = 512,
        n_grid: int = 64,
        n_heads: int = 8,
        screen_length: float = 1.0,
        temperature: float = 1.0,
    ):
        self.d_model = d_model
        self.n_grid = n_grid
        self.n_heads = n_heads
        self.screen_length = screen_length
        self.temperature = temperature

        self._standard = VlasovMeanFieldAttention(
            d_model=d_model, n_grid=n_grid,
            screening_length=screen_length, temperature=temperature,
            causal=True, n_heads=n_heads,
        )
        self._flash = VlasovFlashAttention(
            d_model=d_model, n_grid=n_grid,
            screening_length=screen_length, temperature=temperature,
            causal=True, n_heads=n_heads, block_size=2048,
        )
        self._gyrokinetic = GyrokineticAttention(
            d_model=d_model, n_grid=n_grid,
            screening_length=screen_length, temperature=temperature,
            n_heads=n_heads, causal=True,
        )

    def forward(
        self,
        q: np.ndarray,
        k: np.ndarray,
        v: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Auto-select and run optimal attention strategy."""
        q = np.asarray(q)
        k = np.asarray(k)
        v = np.asarray(v)
        if q.ndim < 2 or k.ndim < 2 or v.ndim < 2:
            raise ValueError(f"q, k, v must be 2D arrays, got shapes {q.shape}, {k.shape}, {v.shape}")
        if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
            raise ValueError(f"q, k, v must have same length, got {q.shape[0]}, {k.shape[0]}, {v.shape[0]}")
        n = q.shape[0]

        if n < 512:
            return self._standard.forward(q, k, v, mask=mask)
        elif n < 4096:
            return self._standard.spectral_forward(q, k, v)
        elif n < 32768:
            return self._flash.forward(q, k, v, mask=mask)
        else:
            return self._gyrokinetic.forward(q, k, v, mask=mask)
