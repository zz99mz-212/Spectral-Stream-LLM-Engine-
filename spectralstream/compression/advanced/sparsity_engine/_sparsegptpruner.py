from __future__ import annotations
from ._pruningpattern import PruningPattern
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

class SparseGPTPruner(BasePruner):
    def __init__(
        self,
        pattern: PruningPattern = PruningPattern.UNSTRUCTURED,
        damp: float = 0.01,
        blocksize: int = 128,
        **kwargs,
    ):
        super().__init__(pattern, **kwargs)
        self.damp = damp
        self.blocksize = blocksize

    def prune(
        self,
        weights: np.ndarray,
        target_sparsity: float,
        layer_name: str = "",
        hessian: Optional[np.ndarray] = None,
        **kwargs,
    ) -> PruningResult:
        w = weights.astype(np.float64).copy()
        rows, cols = w.shape
        if hessian is None:
            H = np.eye(cols, dtype=np.float64)
        else:
            H = np.asarray(hessian, dtype=np.float64)
            if H.shape != (cols, cols):
                H = np.eye(cols, dtype=np.float64)
        H = H + self.damp * np.eye(cols, dtype=np.float64)
        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            H_inv = np.linalg.pinv(H)

        mask = np.ones_like(w, dtype=bool)
        n_keep = max(1, int(cols * (1.0 - target_sparsity)))
        order = np.argsort(np.abs(w).sum(axis=0))[::-1]

        for i_block in range(0, cols, self.blocksize):
            block_end = min(i_block + self.blocksize, cols)
            block_cols = order[i_block:block_end]
            if len(block_cols) == 0:
                continue
            w_block = w[:, block_cols]
            H_inv_block = H_inv[np.ix_(block_cols, block_cols)]
            diag = np.diag(H_inv_block)
            err_scale = np.where(diag > 1e-10, 1.0 / diag, np.inf)
            scores = (w_block**2) * err_scale[np.newaxis, :]
            for j_local in range(len(block_cols)):
                col_scores = scores[:, j_local]
                ones_idx = np.argsort(col_scores)[: max(1, rows - n_keep)]
                mask[ones_idx, block_cols[j_local]] = False
                if len(ones_idx) > 0 and np.isfinite(err_scale[j_local]):
                    w_update = w_block[:, j_local]
                    w_block -= np.outer(w_update, H_inv_block[:, j_local]) / (
                        H_inv_block[j_local, j_local] + 1e-10
                    )

        actual = 1.0 - float(np.mean(mask))
        scores = np.abs(w)
        return PruningResult(
            mask=mask,
            sparsity=actual,
            strategy="sparsegpt",
            layer_name=layer_name,
            importance_scores=scores,
        )
