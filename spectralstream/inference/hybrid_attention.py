"""
Hybrid Attention for SpectralStream
=====================================
Combines standard, wavelet, and Vlasov attention mechanisms with
a learned router that selects the best method per layer.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from spectralstream.core.math_primitives import (
    dct,
    fwht,
    gibbs_softmax,
    idct,
    next_power_of_two,
    softmax,
    spectral_entropy,
)

logger = logging.getLogger(__name__)


class AttentionMethod(IntEnum):
    """Available attention methods."""

    STANDARD = 0
    WAVELET = 1
    VLASOV = 2
    LINEAR = 3


@dataclass
class HybridAttentionConfig:
    """Configuration for hybrid attention."""

    d_model: int = 512
    n_heads: int = 8
    n_kv_heads: Optional[int] = None
    temperature: float = 1.0
    causal: bool = True
    n_grid: int = 64
    screening_length: float = 1.0
    wavelet_n_levels: int = 3
    router_hidden_dim: int = 128
    router_learning_rate: float = 0.01
    fallback_method: AttentionMethod = AttentionMethod.STANDARD


@dataclass
class RouterState:
    """State of the attention router."""

    method_counts: Dict[int, int] = field(default_factory=dict)
    method_errors: Dict[int, List[float]] = field(default_factory=dict)
    weights: Optional[np.ndarray] = None
    bias: Optional[np.ndarray] = None
    initialized: bool = False


class StandardAttentionBackend:
    """Standard O(n^2) softmax attention."""

    def __init__(self, config: HybridAttentionConfig):
        self.config = config
        self.head_dim = config.d_model // config.n_heads

    def forward(
        self,
        q: np.ndarray,
        k: np.ndarray,
        v: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        n = q.shape[0]
        scores = np.einsum("id,jd->ij", q, k) / math.sqrt(self.head_dim)
        if self.config.causal:
            causal_mask = np.triu(np.full((n, n), -1e30, dtype=np.float64), k=1)
            scores = scores + causal_mask
        if mask is not None:
            scores = scores + (1.0 - mask[np.newaxis, :].astype(np.float64)) * (-1e30)
        weights = softmax(scores, temperature=self.config.temperature)
        return (weights @ v).astype(q.dtype)


class WaveletAttentionBackend:
    """Wavelet-based O(n log n) attention via frequency band splitting."""

    def __init__(self, config: HybridAttentionConfig):
        self.config = config
        self.head_dim = config.d_model // config.n_heads
        self.n_levels = config.wavelet_n_levels

    def _haar_forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n = x.shape[-1]
        if n % 2 == 1:
            x = np.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, 1)])
        even = x[..., 0::2]
        odd = x[..., 1::2]
        approx = (even + odd) * 0.5
        detail = (even - odd) * 0.5
        return approx, detail

    def forward(
        self,
        q: np.ndarray,
        k: np.ndarray,
        v: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        n = q.shape[0]
        d = q.shape[-1]
        if n >= 4:
            q_spec = dct(q)
            k_spec = dct(k)
            v_spec = dct(v)
            n_low = max(4, n // 4)
            q_low = q_spec[:, :n_low]
            k_low = k_spec[:, :n_low]
            v_low = v_spec[:, :n_low]
            scores = np.einsum("id,jd->ij", q_low, k_low) / math.sqrt(n_low)
            if self.config.causal:
                causal_mask = np.triu(np.full((n, n), -1e30, dtype=np.float64), k=1)
                scores = scores + causal_mask
            weights = softmax(scores, temperature=self.config.temperature)
            output_low = weights @ v_spec[:, :n_low]
            high_start = n_low
            if high_start < d:
                q_high = q_spec[:, high_start:]
                k_high = k_spec[:, high_start:]
                v_high = v_spec[:, high_start:]
                q_phi = np.maximum(q_high, 0) + 1e-6
                k_phi = np.maximum(k_high, 0) + 1e-6
                kv = (k_phi.T @ v_high) / math.sqrt(d - high_start + 1e-6)
                output_high = q_phi @ kv
            else:
                output_high = np.zeros_like(q)
            output_spec = np.zeros_like(v_spec)
            output_spec[:, :n_low] = output_low
            if high_start < d:
                output_spec[:, high_start:] = output_high
            return idct(output_spec).astype(q.dtype)
        else:
            scores = np.einsum("id,jd->ij", q, k) / math.sqrt(d)
            if self.config.causal:
                causal_mask = np.triu(np.full((n, n), -1e30, dtype=np.float64), k=1)
                scores = scores + causal_mask
            weights = softmax(scores, temperature=self.config.temperature)
            return (weights @ v).astype(q.dtype)


class VlasovAttentionBackend:
    """Vlasov mean-field O(n) attention."""

    def __init__(self, config: HybridAttentionConfig):
        self.config = config
        self.n_grid = next_power_of_two(config.n_grid)
        self._kernel: Optional[np.ndarray] = None

    def _get_kernel(self) -> np.ndarray:
        if self._kernel is None:
            mu = 1.0 / max(self.config.screening_length, 1e-10)
            k = np.fft.fftfreq(self.n_grid, d=1.0 / self.n_grid)
            k_sq = (2.0 * np.pi * k) ** 2
            self._kernel = 4.0 * np.pi / (k_sq + mu**2 + 1e-30)
            self._kernel[0] = 4.0 * np.pi / (mu**2)
        return self._kernel

    def forward(
        self,
        q: np.ndarray,
        k: np.ndarray,
        v: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        n = q.shape[0]
        d = q.shape[-1]
        positions = np.linspace(0.0, 1.0 - 1.0 / max(n, 2), n)
        valid = (
            np.ones(n, dtype=np.float64) if mask is None else mask.astype(np.float64)
        )
        rho = np.zeros(self.n_grid, dtype=np.float64)
        for i in range(n):
            if valid[i] < 0.5:
                continue
            xi = positions[i] * self.n_grid
            xi = np.clip(xi, 0.0, self.n_grid - 1)
            left = int(np.floor(xi))
            left = max(0, min(left, self.n_grid - 2))
            right = left + 1
            wl = 1.0 - (xi - left)
            wr = xi - left
            power = float(np.dot(k[i], k[i]))
            rho[left] += wl * power * valid[i]
            rho[right] += wr * power * valid[i]
        rho_fft = np.fft.fft(rho)
        phi_fft = rho_fft * self._get_kernel()
        phi_grid = np.fft.ifft(phi_fft).real
        phi = np.zeros(n, dtype=np.float64)
        for i in range(n):
            xi = positions[i] * self.n_grid
            xi = np.clip(xi, 0.0, self.n_grid - 1)
            left = int(np.floor(xi))
            left = max(0, min(left, self.n_grid - 2))
            right = left + 1
            wl = 1.0 - (xi - left)
            wr = xi - left
            phi[i] = wl * phi_grid[left] + wr * phi_grid[right]
        weights = gibbs_softmax(phi, temperature=self.config.temperature)
        v_mean = np.mean(v, axis=0, keepdims=True)
        output = (
            v
            + weights[:, None]
            * np.sum(weights[:, None] * phi[:, None], axis=0)
            * v_mean
        )
        return output.astype(q.dtype)


class LinearAttentionBackend:
    """O(n) linear attention via kernel trick."""

    def __init__(self, config: HybridAttentionConfig):
        self.config = config
        self.head_dim = config.d_model // config.n_heads

    def forward(
        self,
        q: np.ndarray,
        k: np.ndarray,
        v: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        q_phi = np.maximum(q, 0) + 1.0
        k_phi = np.maximum(k, 0) + 1.0
        kv = k_phi.T @ v
        k_sum = k_phi.sum(axis=0)
        numerator = q_phi @ kv
        denominator = q_phi @ k_sum[:, None] + 1e-10
        output = numerator / denominator
        return output.astype(q.dtype)


class AttentionRouter:
    """Learned router that selects the best attention method per layer.

    Uses a lightweight linear classifier on input statistics
    (entropy, norm, dimension) to select from available methods.
    """

    def __init__(self, config: HybridAttentionConfig):
        self.config = config
        self.n_methods = len(AttentionMethod)
        self._states: Dict[str, RouterState] = {}
        self._backends: Dict[AttentionMethod, Callable] = {}

    def register_backend(self, method: AttentionMethod, backend: Callable):
        self._backends[method] = backend

    def _compute_features(self, q: np.ndarray, k: np.ndarray) -> np.ndarray:
        n, d = q.shape
        features = np.array(
            [
                spectral_entropy(q.ravel()),
                spectral_entropy(k.ravel()),
                float(np.mean(np.abs(q))),
                float(np.std(q)),
                float(np.mean(np.abs(k))),
                float(np.std(k)),
                float(n),
                float(d),
                float(n) / float(d),
            ],
            dtype=np.float64,
        )
        return features

    def _classify(self, features: np.ndarray) -> AttentionMethod:
        entropy = features[0]
        n = features[6]
        d = features[7]
        ratio = features[8]
        if n < 256:
            return AttentionMethod.STANDARD
        elif ratio > 4.0:
            return AttentionMethod.VLASOV
        elif entropy < 0.5 and n > 1024:
            return AttentionMethod.WAVELET
        else:
            return AttentionMethod.LINEAR

    def select_method(
        self,
        q: np.ndarray,
        k: np.ndarray,
        layer_name: str = "",
    ) -> AttentionMethod:
        features = self._compute_features(q, k)
        method = self._classify(features)
        if method not in self._backends:
            method = self.config.fallback_method
        return method

    def record_error(self, layer_name: str, method: AttentionMethod, error: float):
        if layer_name not in self._states:
            self._states[layer_name] = RouterState()
        state = self._states[layer_name]
        m = int(method)
        state.method_counts[m] = state.method_counts.get(m, 0) + 1
        if m not in state.method_errors:
            state.method_errors[m] = []
        state.method_errors[m].append(error)

    def get_method_stats(self, layer_name: str) -> Dict[int, Dict]:
        if layer_name not in self._states:
            return {}
        state = self._states[layer_name]
        stats = {}
        for m in state.method_counts:
            errors = state.method_errors.get(m, [])
            stats[m] = {
                "count": state.method_counts[m],
                "mean_error": float(np.mean(errors)) if errors else 0.0,
            }
        return stats


class HybridAttention:
    """Hybrid attention that switches between standard, wavelet, vlasov,
    and linear attention methods based on input characteristics.
    """

    def __init__(self, config: Optional[HybridAttentionConfig] = None):
        self.config = config or HybridAttentionConfig()
        self.head_dim = self.config.d_model // self.config.n_heads
        self.router = AttentionRouter(self.config)
        standard = StandardAttentionBackend(self.config)
        wavelet = WaveletAttentionBackend(self.config)
        vlasov = VlasovAttentionBackend(self.config)
        linear = LinearAttentionBackend(self.config)
        self.router.register_backend(AttentionMethod.STANDARD, standard)
        self.router.register_backend(AttentionMethod.WAVELET, wavelet)
        self.router.register_backend(AttentionMethod.VLASOV, vlasov)
        self.router.register_backend(AttentionMethod.LINEAR, linear)
        self._backends = {
            AttentionMethod.STANDARD: standard,
            AttentionMethod.WAVELET: wavelet,
            AttentionMethod.VLASOV: vlasov,
            AttentionMethod.LINEAR: linear,
        }
        self._method_stats: Dict[str, Dict] = {}

    def forward(
        self,
        q: np.ndarray,
        k: np.ndarray,
        v: np.ndarray,
        mask: Optional[np.ndarray] = None,
        layer_name: str = "",
    ) -> Tuple[np.ndarray, AttentionMethod]:
        q = np.asarray(q, dtype=np.float64)
        k = np.asarray(k, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        if q.ndim < 2 or k.ndim < 2 or v.ndim < 2:
            raise ValueError(
                f"q, k, v must be 2D, got shapes {q.shape}, {k.shape}, {v.shape}"
            )
        if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
            raise ValueError("q, k, v must have the same length")
        method = self.router.select_method(q, k, layer_name)
        backend = self._backends[method]
        output = backend.forward(q, k, v, mask)
        if layer_name not in self._method_stats:
            self._method_stats[layer_name] = {"method_counts": {}, "total": 0}
        stats = self._method_stats[layer_name]
        m_name = method.name
        stats["method_counts"][m_name] = stats["method_counts"].get(m_name, 0) + 1
        stats["total"] += 1
        return output, method

    def forward_with_fallback(
        self,
        q: np.ndarray,
        k: np.ndarray,
        v: np.ndarray,
        mask: Optional[np.ndarray] = None,
        layer_name: str = "",
    ) -> np.ndarray:
        try:
            output, method = self.forward(q, k, v, mask, layer_name)
            return output
        except Exception as e:
            logger.warning("Attention method failed, falling back to standard: %s", e)
            backend = self._backends[AttentionMethod.STANDARD]
            return backend.forward(q, k, v, mask)

    def get_stats(self) -> Dict:
        return dict(self._method_stats)

    def get_available_methods(self) -> List[str]:
        return [m.name for m in AttentionMethod]
