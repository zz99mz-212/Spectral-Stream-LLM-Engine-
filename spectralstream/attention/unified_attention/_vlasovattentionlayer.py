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


class VlasovAttentionLayer:
    """Full transformer layer using Vlasov mean-field attention.

    Components:
      - QKV projections (spectral-domain matrix multiply)
      - Vlasov attention (mean-field or flash variant)
      - Skip connection via spectral resonance gating
      - Output projection
      - RMS pre-norm in spectral domain
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        n_grid: int = 64,
        screening_length: float = 1.0,
        temperature: float = 1.0,
        use_flash: bool = False,
        block_size: int = 1024,
        use_spectral_norm: bool = True,
        rms_eps: float = 1e-6,
        seed: int = 42,
    ):
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        self.head_dim = d_model // n_heads
        self.n_grid = n_grid
        self.use_flash = use_flash
        self.block_size = block_size
        self.use_spectral_norm = use_spectral_norm
        self.rms_eps = rms_eps

        self._gqa_factor = n_heads // self.n_kv_heads
        self._gate_bias: float = 0.0

        rng = np.random.RandomState(seed)
        scale = math.sqrt(d_model)
        self.w_q = rng.randn(d_model, d_model).astype(np.float32) / scale
        self.w_k = rng.randn(d_model, self.n_kv_heads * self.head_dim).astype(np.float32) / scale
        self.w_v = rng.randn(d_model, self.n_kv_heads * self.head_dim).astype(np.float32) / scale
        self.w_o = rng.randn(self.n_heads * self.head_dim, d_model).astype(np.float32) / scale

        if use_flash:
            self.attn = VlasovFlashAttention(
                d_model=d_model,
                n_grid=n_grid,
                block_size=block_size,
                screening_length=screening_length,
                temperature=temperature,
                causal=True,
                n_heads=n_heads,
            )
        else:
            self.attn = VlasovMeanFieldAttention(
                d_model=d_model,
                n_grid=n_grid,
                screening_length=screening_length,
                temperature=temperature,
                causal=True,
                n_heads=n_heads,
                n_kv_heads=self.n_kv_heads,
            )

    def _rms_norm(self, x: np.ndarray) -> np.ndarray:
        if self.use_spectral_norm:
            x_spec = dct(x)
            rms = np.sqrt(np.mean(x_spec ** 2, axis=-1, keepdims=True) + self.rms_eps)
            x_spec = (x_spec / rms) * math.sqrt(self.head_dim)
            return idct(x_spec)
        else:
            rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + self.rms_eps)
            return (x / rms) * math.sqrt(self.d_model)

    def _spectral_resonance_gate(
        self,
        residual: np.ndarray,
        attn_output: np.ndarray,
    ) -> np.ndarray:
        res_spec = dct(residual)
        attn_spec = dct(attn_output)

        res_flat = res_spec.ravel()
        attn_flat = attn_spec.ravel()

        cos_sim = float(np.dot(res_flat, attn_flat)) / (
            np.linalg.norm(res_flat) * np.linalg.norm(attn_flat) + 1e-30
        )
        gate = float(np.clip(cos_sim + self._gate_bias, 0.0, 1.0))

        self._gate_bias = 0.99 * self._gate_bias + 0.01 * (gate - 0.5)

        return residual + gate * attn_output

    def forward(
        self,
        x: np.ndarray,
        mask: Optional[np.ndarray] = None,
        return_attention: bool = False,
    ) -> np.ndarray:
        """Full Vlasov attention layer forward pass."""
        x = np.asarray(x)
        if x.ndim < 2:
            raise ValueError(f"x must be a 2D array, got shape {x.shape}")
        n = x.shape[0]

        x_norm = self._rms_norm(x)

        q = x_norm @ self.w_q
        k = x_norm @ self.w_k
        v = x_norm @ self.w_v

        if self.n_heads > self.n_kv_heads:
            k = k.reshape(n, self.n_kv_heads, self.head_dim)
            v = v.reshape(n, self.n_kv_heads, self.head_dim)
            factor = self.n_heads // self.n_kv_heads
            k = np.repeat(k, factor, axis=1)
            v = np.repeat(v, factor, axis=1)
            k = k.reshape(n, self.n_heads * self.head_dim)
            v = v.reshape(n, self.n_heads * self.head_dim)

        q = q.reshape(n, self.n_heads, self.head_dim)
        k = k.reshape(n, self.n_heads, self.head_dim)
        v = v.reshape(n, self.n_heads, self.head_dim)

        outputs = []
        for h in range(self.n_heads):
            out_h = self.attn.forward(q[:, h, :], k[:, h, :], v[:, h, :], mask=mask)
            outputs.append(out_h)

        attn_out = np.concatenate(outputs, axis=-1)
        attn_out = attn_out @ self.w_o

        output = self._spectral_resonance_gate(x, attn_out)

        if return_attention:
            return output, attn_out
        return output

    def prefill(self, prompt: np.ndarray):
        n = prompt.shape[0]
        k = prompt @ self.w_k
        v = prompt @ self.w_v
        k = k.reshape(n, self.n_kv_heads, self.head_dim)
        v = v.reshape(n, self.n_kv_heads, self.head_dim)

    def reset(self):
        self._gate_bias = 0.0
        if hasattr(self.attn, 'reset_causal_state'):
            self.attn.reset_causal_state()
