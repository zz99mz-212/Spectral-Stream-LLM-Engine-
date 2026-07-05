
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

class SelfOrganizingSparsity:
    def __init__(self, local_radius: int = 1, n_iterations: int = 3,
                 birth_threshold: float = 0.6, death_threshold: float = 0.2):
        self.local_radius = local_radius
        self.n_iterations = n_iterations
        self.birth_threshold = birth_threshold
        self.death_threshold = death_threshold

    def _local_density(self, mask: np.ndarray, i: int, j: int) -> float:
        r = self.local_radius
        h, w = mask.shape
        i_min = max(0, i - r)
        i_max = min(h, i + r + 1)
        j_min = max(0, j - r)
        j_max = min(w, j + r + 1)
        nbhd = mask[i_min:i_max, j_min:j_max]
        return float(np.mean(nbhd))

    def evolve(self, weights: np.ndarray, target_sparsity: float) -> np.ndarray:
        mask = np.abs(weights) >= float(np.percentile(np.abs(weights), 50))
        for _ in range(self.n_iterations):
            new_mask = mask.copy()
            for i in range(weights.shape[0]):
                for j in range(weights.shape[1]):
                    density = self._local_density(mask, i, j)
                    weight_mag = float(np.abs(weights[i, j]))
                    if mask[i, j]:
                        if weight_mag < float(np.percentile(np.abs(weights), 20)):
                            new_mask[i, j] = False
                        elif density < self.death_threshold:
                            new_mask[i, j] = False
                    else:
                        if density > self.birth_threshold and \
                                weight_mag > float(np.percentile(np.abs(weights), 70)):
                            new_mask[i, j] = True
            mask = new_mask
        current = float(np.mean(mask))
        target_density = 1.0 - target_sparsity
        if current > target_density:
            extra = np.where(mask)
            scores = np.abs(weights[extra])
            n_remove = int((current - target_density) * mask.size)
            if n_remove > 0:
                remove_idx = np.argsort(scores)[:n_remove]
                extra_arr = tuple(arr[remove_idx] for arr in extra)
                mask[extra_arr] = False
        elif current < target_density:
            missing = np.where(~mask)
            scores = np.abs(weights[missing])
            n_add = int((target_density - current) * mask.size)
            if n_add > 0:
                add_idx = np.argsort(-scores)[:n_add]
                missing_arr = tuple(arr[add_idx] for arr in missing)
                mask[missing_arr] = True
        return np.where(mask, weights, 0.0).astype(weights.dtype)
