from __future__ import annotations

import math
import time
import threading
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Any, Callable, Optional, Union

import numpy as np

from ._helpers import (
    _apply_nm_pattern,
    _block_mask,
    _circular_conv,
    _circular_corr,
    _csr_from_dense,
    _dense_from_csr,
    _energy_ratio,
    _nm_mask,
    _sparsity_ratio,
)
from ._pruningpattern import PruningPattern
from ._pruningresult import PruningResult
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


def _nm_mask(
    shape: tuple, n: int, m: int, rng_seed: Optional[int] = None
) -> np.ndarray:
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


def _block_mask(
    shape: tuple,
    block_h: int,
    block_w: int,
    sparsity: float,
    rng_seed: Optional[int] = None,
) -> np.ndarray:
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
    power = x_spec**2
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


class BasePruner:
    def __init__(
        self,
        pattern: PruningPattern = PruningPattern.UNSTRUCTURED,
        nm_n: int = 2,
        nm_m: int = 4,
        block_h: int = 32,
        block_w: int = 32,
    ):
        self.pattern = pattern
        self.nm_n = nm_n
        self.nm_m = nm_m
        self.block_h = block_h
        self.block_w = block_w

    def _apply_pattern(self, scores: np.ndarray, target_sparsity: float) -> np.ndarray:
        if self.pattern == PruningPattern.UNSTRUCTURED:
            threshold = float(np.percentile(np.abs(scores), target_sparsity * 100))
            return np.abs(scores) >= threshold
        elif self.pattern == PruningPattern.N_M:
            return _nm_mask(scores.shape, self.nm_n, self.nm_m)
        elif self.pattern == PruningPattern.CHANNEL:
            row_norms = np.linalg.norm(scores.reshape(scores.shape[0], -1), axis=1)
            threshold = float(np.percentile(row_norms, target_sparsity * 100))
            mask = np.ones(scores.shape, dtype=bool)
            zero_rows = row_norms < threshold
            mask[zero_rows] = False
            return mask
        elif self.pattern == PruningPattern.BLOCK:
            return ~_block_mask(
                scores.shape, self.block_h, self.block_w, 1.0 - target_sparsity
            )
        else:
            threshold = float(np.percentile(np.abs(scores), target_sparsity * 100))
            return np.abs(scores) >= threshold

    def prune(
        self,
        weights: np.ndarray,
        target_sparsity: float,
        layer_name: str = "",
        **kwargs,
    ) -> PruningResult:
        raise NotImplementedError
