from __future__ import annotations
from ._basepruner import BasePruner
from ._pruningresult import PruningResult
import math
import time
import threading
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Any, Callable, Optional, Union

import numpy as np
from ._helpers import _csr_from_dense, _dense_from_csr, _nm_mask, _block_mask, _energy_ratio, _sparsity_ratio, _circular_conv, _circular_corr, _apply_nm_pattern

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

class HolographicPruner(BasePruner):
    def prune(self, weights: np.ndarray, target_sparsity: float,
              layer_name: str = "",
              hrr_keys: Optional[list[np.ndarray]] = None,
              **kwargs) -> PruningResult:
        w = weights.astype(np.float64)
        r, c = w.shape
        dim = r + c
        if hrr_keys is not None and len(hrr_keys) > 0:
            key = hrr_keys[0]
            if len(key) < dim:
                key = np.pad(key, (0, dim - len(key)))
            elif len(key) > dim:
                key = key[:dim]
        else:
            rng = np.random.RandomState(hash(layer_name) & 0x7FFFFFFF)
            key = rng.randn(dim).astype(np.float64)
            key = key / (np.linalg.norm(key) + 1e-10)
        w_flat = w.ravel()
        flat_dim = min(len(w_flat), dim)
        w_trunc = w_flat[:flat_dim]
        key = key[:flat_dim]
        encoded = _circular_conv(key, w_trunc)
        recalled = _circular_corr(key, encoded)
        pad_size = w.size - flat_dim
        if pad_size > 0:
            recalled_padded = np.pad(recalled, (0, pad_size))[:w.size]
        else:
            recalled_padded = recalled[:w.size]
        recall_fidelity = np.abs(recalled_padded.reshape(w.shape))
        recall_importance = recall_fidelity / (np.max(recall_fidelity) + 1e-10)
        scores = np.abs(w) * (1.0 + recall_importance)
        mask = self._apply_pattern(scores, target_sparsity)
        actual = 1.0 - float(np.mean(mask))
        return PruningResult(
            mask=mask, sparsity=actual, strategy="holographic",
            layer_name=layer_name, importance_scores=scores,
        )
