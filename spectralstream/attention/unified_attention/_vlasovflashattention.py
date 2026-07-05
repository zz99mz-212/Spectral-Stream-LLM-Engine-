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


class VlasovFlashAttention:
    """Tiled Vlasov attention — O(block_size * n_grid) per tile.

    Divides the input sequence into blocks, computes local mean-field
    within each block, then merges blocks via spectral interference
    (phasor addition in Fourier space).

    Memory: O(block_size * d_head + n_grid) per tile, not O(n^2).
    """

    def __init__(
        self,
        d_model: int = 512,
        n_grid: int = 64,
        block_size: int = 1024,
        screening_length: float = 1.0,
        temperature: float = 1.0,
        causal: bool = True,
        n_heads: int = 8,
        overlap: int = 0,
    ):
        self.d_model = d_model
        self.n_grid = next_power_of_two(n_grid)
        self.block_size = block_size
        self.screening_length = screening_length
        self.temperature = temperature
        self.causal = causal
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads if n_heads > 0 else d_model
        self.overlap = overlap

        self._mean_field = VlasovMeanFieldAttention(
            d_model=d_model,
            n_grid=n_grid,
            screening_length=screening_length,
            temperature=temperature,
            causal=False,
            n_heads=n_heads,
        )
        self._kernel = yukawa_kernel_1d(
            self.n_grid, screening_length, dx=1.0 / self.n_grid,
        )

    def _tile_sequence(
        self,
        q: np.ndarray,
        k: np.ndarray,
        v: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> List[VlasovBlock]:
        n = q.shape[0]
        stride = self.block_size - self.overlap
        blocks = []

        positions = np.linspace(0.0, 1.0 - 1.0 / max(n, 2), n)
        if mask is None:
            valid = np.ones(n, dtype=np.float64)
        else:
            valid = mask.astype(np.float64)

        start = 0
        while start < n:
            end = min(start + self.block_size, n)
            idx = np.arange(start, end)
            block_q = q[idx]
            block_k = k[idx]
            block_v = v[idx]
            block_pos = positions[idx]
            block_valid = valid[idx]

            phi = self._mean_field.compute_potential(
                block_k,
                block_valid.astype(bool) if mask is not None else None,
            )
            block_phi = self._mean_field.interpolate_potential(phi, block_pos)

            blocks.append(VlasovBlock(
                indices=idx,
                positions=block_pos,
                q=block_q,
                k=block_k,
                v=block_v,
                local_phi=block_phi,
                valid=block_valid,
            ))
            if end == n:
                break
            start += stride

        return blocks

    def _merge_spectral(self, blocks: List[VlasovBlock]) -> np.ndarray:
        """Merge blocks via spectral interference.

        The key insight: total charge density in Fourier space is
        the sum of densities from all blocks. Since the Poisson
        equation is linear, the potentials add.
        """
        n_total = sum(b.size for b in blocks)
        d = blocks[0].v.shape[-1]

        global_rho_fft = np.zeros(self.n_grid, dtype=np.complex128)

        for block in blocks:
            positions = np.linspace(0.0, 1.0 - 1.0 / max(block.size, 2), block.size)
            rho = np.zeros(self.n_grid, dtype=np.float64)

            for i in range(block.size):
                if block.valid[i] < 0.5:
                    continue
                xi = positions[i] * self.n_grid
                xi = np.clip(xi, 0.0, self.n_grid - 1)
                left = int(np.floor(xi))
                left = max(0, min(left, self.n_grid - 2))
                right = left + 1
                wl = 1.0 - (xi - left)
                wr = xi - left
                power = float(np.dot(block.k[i], block.k[i]))
                rho[left] += wl * power
                rho[right] += wr * power

            global_rho_fft += fft(rho)

        phi_global_fft = global_rho_fft * self._kernel
        phi_global = ifft(phi_global_fft).real

        output = np.zeros((n_total, d), dtype=np.float64)
        offset = 0

        for block in blocks:
            n_b = block.size
            positions_b = np.linspace(0.0, 1.0 - 1.0 / max(n_b, 2), n_b)
            for i in range(n_b):
                xi = positions_b[i] * self.n_grid
                xi = np.clip(xi, 0.0, self.n_grid - 1)
                left = int(np.floor(xi))
                left = max(0, min(left, self.n_grid - 2))
                right = left + 1
                wl = 1.0 - (xi - left)
                wr = xi - left
                phi_i = wl * phi_global[left] + wr * phi_global[right]

                w_i = float(np.exp(-phi_i / max(self.temperature, 1e-10)))

                if d > 1:
                    v_mean = np.mean(block.v, axis=0)
                    output[offset + i] = block.v[i] + w_i * v_mean
                else:
                    output[offset + i] = block.v[i] + w_i * float(np.mean(block.v))

            offset += n_b

        return output

    def forward(
        self,
        q: np.ndarray,
        k: np.ndarray,
        v: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Tiled Vlasov flash attention forward pass."""
        q = np.asarray(q)
        k = np.asarray(k)
        v = np.asarray(v)
        if q.ndim < 2 or k.ndim < 2 or v.ndim < 2:
            raise ValueError(f"q, k, v must be 2D arrays, got shapes {q.shape}, {k.shape}, {v.shape}")
        if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
            raise ValueError(f"q, k, v must have same length, got {q.shape[0]}, {k.shape[0]}, {v.shape[0]}")
        n = q.shape[0]
        if n <= self.block_size:
            return self._mean_field.forward(q, k, v, mask=mask)

        blocks = self._tile_sequence(q, k, v, mask)
        output = self._merge_spectral(blocks)
        return output.astype(q.dtype)
