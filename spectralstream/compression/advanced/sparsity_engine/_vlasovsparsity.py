
import math
import time
import threading
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Any, Callable, Optional, Union

import numpy as np

from spectralstream.core.math_primitives import (
    next_power_of_two as _next_power_of_two,
    softmax as _softmax,
    dct as _dct,
    idct as _idct,
    spectral_entropy as _spectral_entropy,
    fwht,
    ifwht,
)

def _csr_from_dense(dense: np.ndarray, threshold: float = 1e-10) -> tuple:
    mask = np.abs(dense) > threshold
    indices = np.where(mask)
    values = dense[indices]
    return values, indices, dense.shape

def _dense_from_csr(values: np.ndarray, indices: tuple, shape: tuple) -> np.ndarray:
    out = np.zeros(shape, dtype=np.float64)
    out[indices] = values
    return out

def _nm_mask(shape: tuple, n: int, m: int, rng_seed: Optional[int] = None) -> np.ndarray:
    rows, cols = shape
    mask = np.zeros(shape, dtype=bool)
    rng = np.random.RandomState(rng_seed)
    for i in range(rows):
        for j_block in range(0, cols, m):
            block_end = min(j_block + m, cols)
            block_size = block_end - j_block
            chosen = rng.choice(block_size, min(n, block_size), replace=False)
            for c in chosen:
                mask[i, j_block + c] = True
    return mask

def _apply_nm_pattern(weights: np.ndarray, n: int, m: int) -> np.ndarray:
    rows, cols = weights.shape
    out = weights.copy()
    for i in range(rows):
        for j_block in range(0, cols, m):
            block_end = min(j_block + m, cols)
            block = out[i, j_block:block_end]
            if len(block) > n:
                abs_vals = np.abs(block)
                threshold = np.sort(abs_vals)[-n] if n > 0 else 0
                block[abs_vals < threshold] = 0.0
    return out

def _block_mask(shape: tuple, block_h: int, block_w: int, sparsity: float,
                rng_seed: Optional[int] = None) -> np.ndarray:
    rows, cols = shape
    mask = np.ones(shape, dtype=bool)
    rng = np.random.RandomState(rng_seed)
    for i in range(0, rows, block_h):
        for j in range(0, cols, block_w):
            if rng.random() < sparsity:
                ih = min(i + block_h, rows)
                jw = min(j + block_w, cols)
                mask[i:ih, j:jw] = False
    return mask

def _energy_ratio(x: np.ndarray) -> np.ndarray:
    x_spec = _dct(x)
    power = x_spec ** 2
    total = np.sum(power)
    cum = np.cumsum(power) / (total + 1e-30)
    return cum

def _sparsity_ratio(w: np.ndarray) -> float:
    return float(np.mean(np.abs(w) < 1e-10))

def _circular_conv(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = len(a)
    A_fft = np.fft.fft(a.astype(np.complex128))
    B_fft = np.fft.fft(b.astype(np.complex128))
    return np.fft.ifft(A_fft * B_fft).real.astype(np.float64)

def _circular_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = len(a)
    A_fft = np.fft.fft(a.astype(np.complex128))
    B_fft = np.fft.fft(b.astype(np.complex128))
    return np.fft.ifft(np.conj(A_fft) * B_fft).real.astype(np.float64)

class VlasovSparsity:
    def __init__(self, n_grid: int = 64, screening_length: float = 1.0):
        self.n_grid = _next_power_of_two(n_grid)
        self.screening_length = screening_length
        self._kernel = self._yukawa_kernel(self.n_grid)

    def _yukawa_kernel(self, n: int) -> np.ndarray:
        k = np.fft.fftfreq(n)
        mu = 1.0 / max(self.screening_length, 1e-10)
        k_sq = (2.0 * np.pi * k) ** 2
        kernel = 4.0 * np.pi / (k_sq + mu ** 2 + 1e-30)
        kernel[0] = 4.0 * np.pi / (mu ** 2)
        return kernel.astype(np.float64)

    def compute_interaction_strength(self, tokens: np.ndarray) -> np.ndarray:
        n, d = tokens.shape
        positions = np.linspace(0.0, 1.0, n)
        rho = np.zeros(self.n_grid, dtype=np.float64)
        for i in range(n):
            xi = positions[i] * self.n_grid
            xi = np.clip(xi, 0.0, self.n_grid - 1)
            left = int(np.floor(xi))
            left = max(0, min(left, self.n_grid - 2))
            right = left + 1
            wl = 1.0 - (xi - left)
            wr = xi - left
            power = float(np.dot(tokens[i], tokens[i]))
            rho[left] += wl * power
            rho[right] += wr * power
        rho_fft = np.fft.fft(rho)
        phi_fft = rho_fft * self._kernel
        phi = np.fft.ifft(phi_fft).real
        interaction_strength = np.zeros((n, d), dtype=np.float64)
        for i in range(n):
            xi = positions[i] * self.n_grid
            xi = np.clip(xi, 0.0, self.n_grid - 1)
            left = int(np.floor(xi))
            left = max(0, min(left, self.n_grid - 2))
            right = left + 1
            wl = 1.0 - (xi - left)
            wr = xi - left
            phi_i = wl * phi[left] + wr * phi[right]
            interaction_strength[i] = np.abs(phi_i) / (n + 1e-10)
        return interaction_strength

    def prune_low_interaction(self, weights: np.ndarray,
                               token_states: Optional[np.ndarray] = None,
                               threshold: float = 0.05) -> np.ndarray:
        if token_states is None:
            return weights
        interaction = self.compute_interaction_strength(token_states)
        interaction_norm = interaction / (np.max(interaction) + 1e-10)
        weights_2d = weights.reshape(weights.shape[0], -1)
        if interaction_norm.shape[0] == weights_2d.shape[0]:
            mask = interaction_norm.mean(axis=1, keepdims=True) >= threshold
            out = weights * mask.reshape(weights.shape)
        else:
            scaling = np.mean(interaction_norm) / (np.mean(np.abs(weights)) + 1e-10)
            out = weights * (np.abs(weights) > threshold * scaling)
        return out
