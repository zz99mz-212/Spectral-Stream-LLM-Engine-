
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

class StructuredSparsity:
    def __init__(self, config: Optional[SparsityConfig] = None):
        self.config = config or SparsityConfig()
        self._tiled_patterns: dict[str, TiledPattern] = {}
        self._auto_patterns: dict[str, np.ndarray] = {}
        self._learning_rate: float = 0.01

    def apply_nm(self, weights: np.ndarray, n: int = 2, m: int = 4) -> np.ndarray:
        return _apply_nm_pattern(weights, n, m)

    def apply_channel(self, weights: np.ndarray, keep_fraction: Optional[float] = None
                      ) -> np.ndarray:
        if keep_fraction is None:
            keep_fraction = 1.0 - self.config.channel_prune_fraction
        row_norms = np.linalg.norm(weights.reshape(weights.shape[0], -1), axis=1)
        n_keep = max(1, int(weights.shape[0] * keep_fraction))
        keep_idx = np.argsort(-row_norms)[:n_keep]
        out = np.zeros_like(weights)
        out[keep_idx] = weights[keep_idx]
        return out

    def apply_block(self, weights: np.ndarray, block_h: int = 32, block_w: int = 32,
                     sparsity: Optional[float] = None) -> np.ndarray:
        if sparsity is None:
            sparsity = self.config.target_sparsity
        rows, cols = weights.shape
        out = weights.copy()
        for i in range(0, rows, block_h):
            ih = min(i + block_h, rows)
            for j in range(0, cols, block_w):
                jw = min(j + block_w, cols)
                block = out[i:ih, j:jw]
                block_norm = float(np.linalg.norm(block))
                if block_norm < float(np.linalg.norm(weights)) / max(weights.size, 1) * block.size:
                    out[i:ih, j:jw] = 0.0
        return out

    def apply_tiled(self, weights: np.ndarray, tile_h: int = 64, tile_w: int = 64,
                     pattern: Optional[np.ndarray] = None,
                     layer_name: str = "") -> np.ndarray:
        rows, cols = weights.shape
        out = weights.copy()
        if pattern is None:
            if layer_name in self._tiled_patterns:
                tp = self._tiled_patterns[layer_name]
                pattern = tp.mask
        if pattern is None:
            n_tiles_h = (rows + tile_h - 1) // tile_h
            n_tiles_w = (cols + tile_w - 1) // tile_w
            pattern = np.random.RandomState(42).binomial(
                1, 1.0 - self.config.target_sparsity,
                size=(n_tiles_h, n_tiles_w)
            ).astype(bool)
        n_tiles_h, n_tiles_w = pattern.shape
        for ti in range(n_tiles_h):
            for tj in range(n_tiles_w):
                if not pattern[ti, tj]:
                    i0 = ti * tile_h
                    i1 = min(i0 + tile_h, rows)
                    j0 = tj * tile_w
                    j1 = min(j0 + tile_w, cols)
                    out[i0:i1, j0:j1] = 0.0
        if layer_name:
            if layer_name not in self._tiled_patterns:
                self._tiled_patterns[layer_name] = TiledPattern(
                    tile_h=tile_h, tile_w=tile_w, mask=pattern,
                )
            else:
                self._tiled_patterns[layer_name].mask = pattern
        return out

    def learn_pattern_auto(self, weights: np.ndarray, gradient: np.ndarray,
                            layer_name: str = "", n_clusters: int = 8) -> np.ndarray:
        if layer_name not in self._auto_patterns:
            rng = np.random.RandomState(hash(layer_name) & 0x7FFFFFFF)
            self._auto_patterns[layer_name] = rng.binomial(
                1, 0.5, size=weights.shape
            ).astype(np.float64)
        pattern = self._auto_patterns[layer_name]
        importance = np.abs(weights) + np.abs(gradient) * self._learning_rate
        pattern += self._learning_rate * (importance - 0.5)
        pattern = np.clip(pattern, 0.0, 1.0)
        self._auto_patterns[layer_name] = pattern
        return weights * (pattern > 0.5)

    def search_best_pattern(self, weights: np.ndarray, n_trials: int = 8,
                              sparsity: Optional[float] = None
                              ) -> tuple[np.ndarray, np.ndarray]:
        if sparsity is None:
            sparsity = self.config.target_sparsity
        best_error = float("inf")
        best_mask = np.ones(weights.shape, dtype=bool)
        rng = np.random.RandomState(42)
        for _ in range(n_trials):
            if sparsity > 0:
                threshold = float(np.percentile(np.abs(weights), sparsity * 100))
                mask = np.abs(weights) >= threshold * rng.uniform(0.5, 1.5)
            else:
                mask = np.ones(weights.shape, dtype=bool)
            pruned = np.where(mask, weights, 0.0)
            error = float(np.sum((pruned - weights) ** 2))
            if error < best_error:
                best_error = error
                best_mask = mask
        return np.where(best_mask, weights, 0.0).astype(weights.dtype), best_mask
