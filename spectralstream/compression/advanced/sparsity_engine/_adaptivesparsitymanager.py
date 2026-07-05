
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

class AdaptiveSparsityManager:
    def __init__(self, config: Optional[SparsityConfig] = None,
                 global_min_sparsity: float = 0.1,
                 global_max_sparsity: float = 0.995,
                 adaptation_rate: float = 0.05,
                 error_window: int = 128):
        self.config = config or SparsityConfig()
        self.global_min = global_min_sparsity
        self.global_max = global_max_sparsity
        self.adaptation_rate = adaptation_rate
        self.error_window = error_window
        self._layer_states: dict[str, LayerSparsityState] = {}
        self._global_quality: deque = deque(maxlen=1000)
        self._coherence_scores: deque = deque(maxlen=1000)
        self._token_difficulty: deque = deque(maxlen=1000)
        self._lock = threading.Lock()

    def register_layer(self, layer_name: str, base_sparsity: Optional[float] = None):
        with self._lock:
            self._layer_states[layer_name] = LayerSparsityState(
                base_sparsity=base_sparsity or self.config.target_sparsity,
                current_sparsity=base_sparsity or self.config.target_sparsity,
            )

    def get_sparsity(self, layer_name: str) -> float:
        state = self._layer_states.get(layer_name)
        if state is None:
            return self.config.target_sparsity
        return state.current_sparsity

    def set_sparsity(self, layer_name: str, sparsity: float):
        with self._lock:
            state = self._layer_states.get(layer_name)
            if state is None:
                self.register_layer(layer_name, sparsity)
            else:
                state.current_sparsity = np.clip(sparsity, self.global_min, self.global_max)

    def record_token(self, layer_name: str, confidence: float, entropy: float,
                     error: Optional[bool] = None):
        with self._lock:
            state = self._layer_states.get(layer_name)
            if state is None:
                self.register_layer(layer_name)
                state = self._layer_states[layer_name]
            state.token_entropies.append(entropy)
            state.confidences.append(confidence)
            state.n_tokens_seen += 1
            if error is not None:
                state.errors.append(1.0 if error else 0.0)
                state.n_errors += 1
                self._global_quality.append(0.0 if error else 1.0)

    def record_coherence(self, coherence: float):
        self._coherence_scores.append(coherence)

    def record_token_difficulty(self, difficulty: float):
        self._token_difficulty.append(difficulty)

    def adapt_sparsity(self, layer_name: str, force: bool = False) -> float:
        with self._lock:
            state = self._layer_states.get(layer_name)
            if state is None:
                return self.config.target_sparsity
            if state.n_tokens_seen < 10 and not force:
                return state.current_sparsity
            recent_entropy = np.mean(state.token_entropies) if state.token_entropies else 0.5
            recent_confidence = np.mean(state.confidences) if state.confidences else 0.5
            recent_error_rate = np.mean(state.errors) if state.errors else 0.0
            coherence = np.mean(self._coherence_scores) if self._coherence_scores else 0.5
            difficulty = np.mean(self._token_difficulty) if self._token_difficulty else 0.5
            quality = np.mean(self._global_quality) if self._global_quality else 0.95
            if recent_error_rate > 0.2:
                sparsity_delta = -self.adaptation_rate * min(1.0, recent_error_rate * 3)
            elif recent_confidence > 0.8 and recent_entropy < 0.3:
                sparsity_delta = self.adaptation_rate * 0.5 * min(1.0, recent_confidence)
            elif difficulty > 0.7:
                sparsity_delta = -self.adaptation_rate * difficulty
            elif coherence > 0.7:
                sparsity_delta = self.adaptation_rate * 0.3 * coherence
            elif quality < 0.8:
                sparsity_delta = -self.adaptation_rate * 0.5
            else:
                sparsity_delta = 0.0
            new_sparsity = state.current_sparsity + sparsity_delta
            new_sparsity = np.clip(new_sparsity, self.global_min, self.global_max)
            state.current_sparsity = new_sparsity
            return new_sparsity

    def adapt_all_layers(self) -> dict[str, float]:
        updates = {}
        with self._lock:
            for layer_name in list(self._layer_states.keys()):
                updates[layer_name] = self.adapt_sparsity(layer_name)
        return updates

    def get_layer_state(self, layer_name: str) -> Optional[dict]:
        state = self._layer_states.get(layer_name)
        if state is None:
            return None
        return {
            "base_sparsity": state.base_sparsity,
            "current_sparsity": state.current_sparsity,
            "mean_entropy": float(np.mean(state.token_entropies)) if state.token_entropies else 0.0,
            "mean_confidence": float(np.mean(state.confidences)) if state.confidences else 0.0,
            "error_rate": float(np.mean(state.errors)) if state.errors else 0.0,
            "n_tokens": state.n_tokens_seen,
            "n_errors": state.n_errors,
        }

    def get_global_quality(self) -> float:
        return float(np.mean(self._global_quality)) if self._global_quality else 1.0

    def reset(self):
        with self._lock:
            self._layer_states.clear()
            self._global_quality.clear()
            self._coherence_scores.clear()
            self._token_difficulty.clear()
