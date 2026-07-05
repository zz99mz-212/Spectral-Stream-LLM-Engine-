
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

class HolographicSparsity:
    def __init__(self, dim: int = 4096):
        self.dim = dim
        rng = np.random.RandomState(42)
        self._base_key = rng.randn(dim).astype(np.float64)
        self._base_key = self._base_key / (np.linalg.norm(self._base_key) + 1e-10)

    def encode_sparse(self, weights: np.ndarray) -> np.ndarray:
        w_flat = weights.ravel()
        if len(w_flat) > self.dim:
            w_flat = w_flat[:self.dim]
        elif len(w_flat) < self.dim:
            w_flat = np.pad(w_flat, (0, self.dim - len(w_flat)))
        return _circular_conv(self._base_key, w_flat)

    def decode_sparse(self, encoded: np.ndarray,
                       original_shape: tuple) -> np.ndarray:
        flat_dim = min(np.prod(original_shape), self.dim)
        key = self._base_key[:flat_dim]
        recalled = _circular_corr(key, encoded[:flat_dim])
        pad_size = np.prod(original_shape) - flat_dim
        if pad_size > 0:
            recalled = np.pad(recalled, (0, pad_size))
        return recalled[:np.prod(original_shape)].reshape(original_shape).astype(np.float64)

    def represent_pattern(self, patterns: list[np.ndarray]) -> np.ndarray:
        encoded = np.zeros(self.dim, dtype=np.float64)
        for pat in patterns:
            pat_flat = pat.ravel()
            if len(pat_flat) > self.dim:
                pat_flat = pat_flat[:self.dim]
            elif len(pat_flat) < self.dim:
                pat_flat = np.pad(pat_flat, (0, self.dim - len(pat_flat)))
            key = pat_flat / (np.linalg.norm(pat_flat) + 1e-10)
            encoded += _circular_conv(key, pat_flat)
        return encoded

    def decode_patterns(self, encoded: np.ndarray, n_patterns: int,
                         original_shape: tuple) -> list[np.ndarray]:
        patterns = []
        for _ in range(n_patterns):
            recalled = _circular_corr(self._base_key, encoded)
            pat = recalled[:np.prod(original_shape)].reshape(original_shape)
            patterns.append(pat)
            key = pat.ravel() / (np.linalg.norm(pat.ravel()) + 1e-10)
            if len(key) < self.dim:
                key = np.pad(key, (0, self.dim - len(key)))
            elif len(key) > self.dim:
                key = key[:self.dim]
            encoded -= _circular_conv(key, pat.ravel())
        return patterns

    def interference_encode(self, weight_set: list[np.ndarray]) -> np.ndarray:
        combined = np.zeros(self.dim, dtype=np.complex128)
        for i, w in enumerate(weight_set):
            w_flat = w.ravel()
            if len(w_flat) > self.dim:
                w_flat = w_flat[:self.dim]
            elif len(w_flat) < self.dim:
                w_flat = np.pad(w_flat, (0, self.dim - len(w_flat)))
            phase = np.exp(2j * np.pi * i / max(len(weight_set), 1))
            combined += phase * w_flat.astype(np.complex128)
        return combined
