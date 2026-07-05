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

class SparsePruner:
    def __init__(self, config: Optional[SparsityConfig] = None):
        self.config = config or SparsityConfig()
        self._pruners = {
            "magnitude": MagnitudePruner(),
            "wanda": WandaPruner(),
            "sparsegpt": SparseGPTPruner(),
            "spectral": SpectralPruner(),
            "movement": MovementPruner(),
            "combined": CombinedPruner(),
            "vlasov": VlasovPruner(),
            "resonant": ResonantPruner(),
            "holographic": HolographicPruner(),
            "quantum": QuantumPruner(n_superpositions=self.config.quantum_superposition_size),
            "self_organizing": SelfOrganizingPruner(n_iterations=self.config.self_organizing_iterations),
        }
        self._history: dict[str, list[np.ndarray]] = {}
        self._results: dict[str, PruningResult] = {}

    def get_pruner(self, name: str) -> BasePruner:
        return self._pruners.get(name, self._pruners["combined"])

    def record_weight_snapshot(self, layer_name: str, weights: np.ndarray):
        if layer_name not in self._history:
            self._history[layer_name] = []
        self._history[layer_name].append(weights.copy())
        if len(self._history[layer_name]) > 10:
            self._history[layer_name].pop(0)

    def prune_layer(self, weights: np.ndarray, target_sparsity: Optional[float] = None,
                    layer_name: str = "",
                    strategy: Optional[str] = None,
                    activation_norm: Optional[np.ndarray] = None,
                    hessian: Optional[np.ndarray] = None,
                    token_interactions: Optional[np.ndarray] = None,
                    training_frequencies: Optional[np.ndarray] = None,
                    hrr_keys: Optional[list[np.ndarray]] = None,
                    **kwargs) -> PruningResult:
        target = target_sparsity if target_sparsity is not None else self.config.target_sparsity
        strategy = strategy or self.config.pruning_strategy
        pruner = self._pruners.get(strategy, self._pruners["combined"])
        extra = dict(
            activation_norm=activation_norm,
            hessian=hessian,
            weight_history=self._history.get(layer_name),
            token_interactions=token_interactions,
            training_frequencies=training_frequencies,
            hrr_keys=hrr_keys,
        )
        extra.update(kwargs)
        result = pruner.prune(weights, target, layer_name=layer_name, **{
            k: v for k, v in extra.items() if v is not None
        })
        self._results[layer_name] = result
        return result

    def apply_mask(self, weights: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return np.where(mask, weights, 0.0).astype(weights.dtype)

    def apply_pruning(self, weights: np.ndarray, layer_name: str = "",
                      **kwargs) -> tuple[np.ndarray, PruningResult]:
        result = self.prune_layer(weights, layer_name=layer_name, **kwargs)
        pruned = self.apply_mask(weights, result.mask)
        return pruned, result

    def sparsity_of(self, weights: np.ndarray) -> float:
        return float(np.mean(np.abs(weights) < 1e-10))

    def get_result(self, layer_name: str) -> Optional[PruningResult]:
        return self._results.get(layer_name)
