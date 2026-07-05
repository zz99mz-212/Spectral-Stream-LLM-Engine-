
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

class ActivationSparsity:
    def __init__(self, config: Optional[SparsityConfig] = None):
        self.config = config or SparsityConfig()
        self._threshold_mode = ActivationThreshold.DYNAMIC_PERCENTILE
        self._threshold_percentile: float = 90.0
        self._running_stats: dict[str, dict] = {}
        self._hdc_predictors: dict[str, Callable] = {}

    def set_threshold_mode(self, mode: str, percentile: float = 90.0):
        self._threshold_mode = mode
        self._threshold_percentile = percentile

    def get_threshold(self, activations: np.ndarray,
                      layer_name: str = "") -> float:
        flat = np.abs(activations.ravel())
        if self._threshold_mode == ActivationThreshold.FIXED:
            return self.config.activation_threshold
        elif self._threshold_mode == ActivationThreshold.DYNAMIC_MEAN:
            return float(np.mean(flat)) * 0.1
        elif self._threshold_mode == ActivationThreshold.DYNAMIC_PERCENTILE:
            return float(np.percentile(flat, self._threshold_percentile))
        elif self._threshold_mode == ActivationThreshold.DISTRIBUTION_ADAPTIVE:
            mean = float(np.mean(flat))
            std = float(np.std(flat)) + 1e-10
            return mean + 1.5 * std
        return self.config.activation_threshold

    def compute_sparsity(self, activations: np.ndarray,
                         threshold: Optional[float] = None,
                         layer_name: str = "") -> tuple[np.ndarray, float]:
        if threshold is None:
            threshold = self.get_threshold(activations, layer_name)
        mask = np.abs(activations) >= threshold
        sparsity = 1.0 - float(np.mean(mask))
        if layer_name:
            if layer_name not in self._running_stats:
                self._running_stats[layer_name] = {"threshold": [], "sparsity": []}
            self._running_stats[layer_name]["threshold"].append(threshold)
            self._running_stats[layer_name]["sparsity"].append(sparsity)
            max_hist = 100
            for k in self._running_stats[layer_name]:
                if len(self._running_stats[layer_name][k]) > max_hist:
                    self._running_stats[layer_name][k] = \
                        self._running_stats[layer_name][k][-max_hist:]
        return mask, sparsity

    def sparse_ffn(self, hidden: np.ndarray, w1: np.ndarray, w2: np.ndarray,
                   w3: Optional[np.ndarray] = None,
                   activation_fn: str = "silu",
                   layer_name: str = "",
                   early_exit: bool = True) -> np.ndarray:
        if early_exit:
            input_norm = np.linalg.norm(hidden, axis=-1)
            inactive = input_norm < self.config.activation_threshold * hidden.shape[-1]
            if np.all(inactive):
                return np.zeros_like(hidden)
        if activation_fn == "relu":
            gate_pre = hidden @ w1
            gate = np.maximum(gate_pre, 0.0)
        elif activation_fn == "gelu":
            gate_pre = hidden @ w1
            gate = 0.5 * gate_pre * (1.0 + np.tanh(
                np.sqrt(2.0 / np.pi) * (gate_pre + 0.044715 * gate_pre ** 3)))
        elif activation_fn == "silu":
            gate_pre = hidden @ w1
            gate = gate_pre / (1.0 + np.exp(-gate_pre))
        else:
            gate_pre = hidden @ w1
            gate = gate_pre / (1.0 + np.exp(-gate_pre))
        mask, act_sparsity = self.compute_sparsity(gate, layer_name=layer_name)
        if act_sparsity > 0.5 and np.any(mask):
            if w3 is not None:
                up = hidden @ w3
                sparse_out = gate * up
                output = sparse_out[:, mask.any(axis=0)] @ w2[mask.any(axis=0)]
            else:
                sparse_gate = gate * mask
                output = sparse_gate @ w2
        else:
            if w3 is not None:
                up = hidden @ w3
                output = (gate * up) @ w2
            else:
                output = gate @ w2
        return output

    def predict_sparsity_hdc(self, hidden: np.ndarray,
                              layer_name: str = "") -> np.ndarray:
        if layer_name in self._hdc_predictors:
            predictor = self._hdc_predictors[layer_name]
            return predictor(hidden)
        fixed_sparsity = 0.9
        n = hidden.shape[-1]
        n_keep = max(1, int(n * (1.0 - fixed_sparsity)))
        scores = np.linalg.norm(hidden, axis=0)
        top_idx = np.argsort(-scores)[:n_keep]
        mask = np.zeros(n, dtype=bool)
        mask[top_idx] = True
        return mask

    def register_hdc_predictor(self, layer_name: str,
                                predictor_fn: Callable[[np.ndarray], np.ndarray]):
        self._hdc_predictors[layer_name] = predictor_fn

    def sparse_attention_by_activation(self, query: np.ndarray,
                                        keys: np.ndarray,
                                        values: np.ndarray,
                                        threshold: Optional[float] = None) -> np.ndarray:
        q_mag = np.linalg.norm(query, axis=-1)
        if threshold is None:
            threshold = float(np.percentile(q_mag, 50))
        high_mag = q_mag >= threshold
        if not np.any(high_mag):
            return np.zeros_like(query)
        if np.all(high_mag):
            sim = query @ keys.T
            attn = _softmax(sim)
            return attn @ values
        q_high = query[high_mag]
        sim = q_high @ keys.T
        attn = _softmax(sim)
        out_high = attn @ values
        output = np.zeros_like(query)
        output[high_mag] = out_high
        return output

    def get_stats(self, layer_name: str = "") -> dict:
        if layer_name and layer_name in self._running_stats:
            s = self._running_stats[layer_name]
            return {
                "mean_threshold": float(np.mean(s["threshold"])),
                "mean_sparsity": float(np.mean(s["sparsity"])),
                "thresholds": s["threshold"][-10:],
                "sparsities": s["sparsity"][-10:],
            }
        return {}
