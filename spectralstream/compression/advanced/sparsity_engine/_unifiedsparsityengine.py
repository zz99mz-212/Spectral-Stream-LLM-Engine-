
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

class UnifiedSparsityEngine:
    def __init__(self, config: Optional[SparsityConfig] = None,
                 model_dim: int = 512,
                 n_layers: int = 8,
                 n_heads: int = 8):
        self.config = config or SparsityConfig()
        self.model_dim = model_dim
        self.n_layers = n_layers
        self.n_heads = n_heads

        self.pruner = SparsePruner(self.config)
        self.executor = DynamicSparseExecutor(self.config)
        self.activation = ActivationSparsity(self.config)
        self.spectral = SpectralSparsity(self.config)
        self.structured = StructuredSparsity(self.config)
        self.adaptive = AdaptiveSparsityManager(self.config)
        self.hdc_predictor = HDCSparsityPredictor(hidden_dim=model_dim)

        self.vlasov_sparsity = VlasovSparsity()
        self.resonant_sparsity = ResonantSparsity()
        self.holographic_sparsity = HolographicSparsity(dim=self.config.hrr_dim)
        self.quantum_sparsity = QuantumSparsity(n_superpositions=self.config.quantum_superposition_size)
        self.self_organizing = SelfOrganizingSparsity(n_iterations=self.config.self_organizing_iterations)

        self._stats: dict[str, Any] = {
            "total_pruned": 0,
            "total_weights": 0,
            "activation_sparsity_saved": 0,
            "spectral_compression_ratio": 1.0,
        }
        self._lock = threading.Lock()

    def prune_weights(self, weights: np.ndarray, layer_name: str = "",
                       target_sparsity: Optional[float] = None,
                       strategy: Optional[str] = None) -> tuple[np.ndarray, PruningResult]:
        target = target_sparsity if target_sparsity is not None else \
            self.adaptive.get_sparsity(layer_name)
        result = self.pruner.prune_layer(
            weights, target, layer_name=layer_name, strategy=strategy,
        )
        return self.pruner.apply_mask(weights, result.mask), result

    def sparse_forward_linear(self, weights: np.ndarray, input_vec: np.ndarray,
                               layer_name: str = "") -> np.ndarray:
        density = float(np.mean(np.abs(weights) > 1e-10))
        if density > self.config.fallback_density_threshold:
            return input_vec @ weights.T
        return self.executor.sparse_gemm(weights, input_vec)

    def sparse_forward_ffn(self, hidden: np.ndarray,
                            w1: np.ndarray, w2: np.ndarray,
                            w3: Optional[np.ndarray] = None,
                            activation: str = "silu",
                            layer_name: str = "") -> np.ndarray:
        return self.activation.sparse_ffn(
            hidden, w1, w2, w3,
            activation_fn=activation,
            layer_name=layer_name,
        )

    def sparse_attention(self, query: np.ndarray, keys: np.ndarray,
                          values: np.ndarray, layer_name: str = "") -> np.ndarray:
        sparsity = self.adaptive.get_sparsity(f"{layer_name}_attn")
        return self.executor.sparse_attention(query, keys, values, sparsity=sparsity)

    def compress_spectral(self, x: np.ndarray, layer_name: str = "",
                           keep_fraction: Optional[float] = None) -> np.ndarray:
        return self.spectral.dct_sparsify(x, keep_fraction=keep_fraction)

    def apply_structured_pruning(self, weights: np.ndarray, pattern: str = "nm",
                                  layer_name: str = "") -> np.ndarray:
        if pattern == "nm":
            return self.structured.apply_nm(weights, self.config.nm_ratio[0],
                                            self.config.nm_ratio[1])
        elif pattern == "channel":
            return self.structured.apply_channel(weights)
        elif pattern == "block":
            return self.structured.apply_block(weights, self.config.block_size,
                                               self.config.block_size)
        elif pattern == "tiled":
            return self.structured.apply_tiled(weights, layer_name=layer_name)
        return weights

    def adapt_sparsity(self, layer_name: str) -> float:
        return self.adaptive.adapt_sparsity(layer_name)

    def adapt_all(self) -> dict[str, float]:
        return self.adaptive.adapt_all_layers()

    def record_feedback(self, layer_name: str, confidence: float,
                         entropy: float, error: Optional[bool] = None):
        self.adaptive.record_token(layer_name, confidence, entropy, error)

    def get_sparsity(self, layer_name: str) -> float:
        return self.adaptive.get_sparsity(layer_name)

    def get_stats(self) -> dict:
        with self._lock:
            return dict(self._stats)

    def get_layer_info(self, layer_name: str) -> dict:
        info = self.adaptive.get_layer_state(layer_name) or {}
        result = self.pruner.get_result(layer_name)
        if result is not None:
            info["pruning_strategy"] = result.strategy
            info["pruning_sparsity"] = result.sparsity
        return info

    def export_sparsity_report(self) -> dict:
        report = {
            "config": self.config.__dict__,
            "global_quality": self.adaptive.get_global_quality(),
            "layers": {},
        }
        for layer_name in list(self.adaptive._layer_states.keys()):
            report["layers"][layer_name] = self.get_layer_info(layer_name)
        return report
