"""
Mean-Field Attention & Spectral Operations v2
-----------------------------------------------
15 spectral/DCT upgrades inspired by
Quasar v5 Vlasov attention and plasma-physics-based inference.
"""

from __future__ import annotations

import numpy as np
from collections import deque
from typing import Callable, Optional

from spectralstream.core.math_primitives import (
    dct,
    idct,
    softmax,
    spectral_entropy,
)


class SpectralField:
    """Spectral-domain operations via DCT.

    Delegates to the canonical implementations in math_primitives.
    """

    @staticmethod
    def dct(x: np.ndarray) -> np.ndarray:
        return dct(x)

    @staticmethod
    def idct(x: np.ndarray) -> np.ndarray:
        return idct(x)

    @staticmethod
    def compress_spectral(
        x: np.ndarray,
        compression_ratio: float = 0.2,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        n = x.shape[-1]
        k = max(1, int(n * compression_ratio))
        coeffs = SpectralField.dct(x)
        magnitude = np.abs(coeffs)
        threshold = np.sort(magnitude.ravel())[-k] if k < magnitude.size else 0.0
        mask = magnitude >= threshold
        compressed = coeffs * mask
        return compressed, mask, k

    @staticmethod
    def spectral_similarity(
        a: np.ndarray,
        b: np.ndarray,
        compression_ratio: float = 0.2,
    ) -> float:
        a_spec, _, _ = SpectralField.compress_spectral(a.ravel(), compression_ratio)
        b_spec, _, _ = SpectralField.compress_spectral(b.ravel(), compression_ratio)
        a_flat = a_spec.ravel()
        b_flat = b_spec.ravel()
        sim = np.dot(a_flat, b_flat)
        norm = np.linalg.norm(a_flat) * np.linalg.norm(b_flat) + 1e-10
        return float(sim / norm)

    @staticmethod
    def band_limit(x: np.ndarray, n_keep: int) -> np.ndarray:
        coeffs = SpectralField.dct(x)
        coeffs[..., n_keep:] = 0.0
        return SpectralField.idct(coeffs)


class VlasovMeanFieldAttention:
    """O(n) mean-field attention via Vlasov plasma theory.

    Uses the screened Coulomb (Yukawa) potential as the interaction
    kernel, computed in O(n) instead of O(n^2).
    """

    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        spectral_rank: int = 64,
        sigma: float = 1.0,
        screening_length: float = 1.0,
        use_yukawa: bool = True,
        use_causal_mask: bool = True,
    ):
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.spectral_rank = min(spectral_rank, self.head_dim)
        self.sigma = sigma
        self.screening_length = screening_length
        self.use_yukawa = use_yukawa
        self.use_causal_mask = use_causal_mask
        self._causal_state: Optional[np.ndarray] = None

    def _yukawa_kernel(self, dist_sq: np.ndarray) -> np.ndarray:
        mu = self.screening_length
        r = np.sqrt(dist_sq + 1e-10)
        return np.exp(-r / mu) / (1.0 + dist_sq)

    def _gaussian_kernel(self, dist_sq: np.ndarray) -> np.ndarray:
        return np.exp(-0.5 * dist_sq / (self.sigma**2 + 1e-10))

    def mean_field_attention(
        self,
        q: np.ndarray,
        k: np.ndarray,
        v: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        n_q = q.shape[0]
        d = q.shape[-1]
        k_mean = np.mean(k, axis=0, keepdims=True)
        k_std = np.std(k, axis=0, keepdims=True) + 1e-6
        q_dev = q - k_mean
        scaled_dev = q_dev / (k_std * self.sigma)
        dist_sq = np.sum(scaled_dev**2, axis=-1, keepdims=True)
        if self.use_yukawa:
            weights = self._yukawa_kernel(dist_sq)
        else:
            weights = self._gaussian_kernel(dist_sq)
        weights = weights / (np.sum(weights, axis=0, keepdims=True) + 1e-10)
        v_mean = np.mean(v, axis=0, keepdims=True)
        v_std = np.std(v, axis=0, keepdims=True) + 1e-6
        if self.use_causal_mask:
            output = self._causal_mean_field(q, k, v, weights)
        else:
            output = weights * v_mean + (1.0 - weights) * v
        if mask is not None:
            mask_weights = mask.astype(np.float32)
            mask_weights = mask_weights / (
                np.sum(mask_weights, axis=-1, keepdims=True) + 1e-10
            )
            output = mask_weights @ v
        return output

    def _causal_mean_field(
        self,
        q: np.ndarray,
        k: np.ndarray,
        v: np.ndarray,
        weights: np.ndarray,
    ) -> np.ndarray:
        n = q.shape[0]
        d = q.shape[-1]
        output = np.zeros_like(q, dtype=np.float64)
        running_mean_k = np.zeros(d, dtype=np.float64)
        running_count = 1e-10
        for i in range(n):
            running_mean_k = (running_mean_k * running_count + k[i]) / (
                running_count + 1.0
            )
            running_count += 1.0
            std_k = np.std(k[: i + 1], axis=0) + 1e-6
            dev = q[i] - running_mean_k
            dist_sq = float(np.sum((dev / std_k) ** 2))
            if self.use_yukawa:
                w = float(self._yukawa_kernel(np.array([[dist_sq]]))[0, 0])
            else:
                w = float(self._gaussian_kernel(np.array([[dist_sq]]))[0, 0])
            v_mean_i = np.mean(v[: i + 1], axis=0)
            output[i] = w * v_mean_i + (1.0 - w) * v[i]
        return output.astype(np.float32)

    def spectral_attention(
        self,
        q: np.ndarray,
        k: np.ndarray,
        v: np.ndarray,
        compression_ratio: float = 0.1,
    ) -> np.ndarray:
        n, d = q.shape
        k_spec, _, _ = SpectralField.compress_spectral(k, compression_ratio)
        v_spec, _, _ = SpectralField.compress_spectral(v, compression_ratio)
        q_spec = SpectralField.dct(q)
        q_r = q_spec.reshape(n, self.n_heads, self.head_dim)
        k_r = k_spec.reshape(n, self.n_heads, self.head_dim)
        v_r = v_spec.reshape(n, self.n_heads, self.head_dim)
        k_mean = np.mean(k_r, axis=0, keepdims=True)
        v_mean = np.mean(v_r, axis=0, keepdims=True)
        k_std = np.std(k_r, axis=0, keepdims=True) + 1e-6
        q_dev = q_r - k_mean
        dist_sq = np.sum((q_dev / (k_std * self.sigma)) ** 2, axis=-1, keepdims=True)
        if self.use_yukawa:
            weights = self._yukawa_kernel(dist_sq)
        else:
            weights = self._gaussian_kernel(dist_sq)
        weights = weights / (np.sum(weights, axis=0, keepdims=True) + 1e-10)
        output_spec = weights * v_mean + (1.0 - weights) * v_r
        output_spec = output_spec.reshape(n, d)
        return SpectralField.idct(output_spec).astype(np.float32)


def gyrokinetic_split(
    x: np.ndarray,
    split_fraction: float = 0.3,
) -> tuple[np.ndarray, np.ndarray]:
    coeffs = SpectralField.dct(x)
    n = coeffs.shape[-1]
    n_slow = max(1, int(n * split_fraction))
    slow_coeffs = np.zeros_like(coeffs)
    fast_coeffs = np.zeros_like(coeffs)
    slow_coeffs[..., :n_slow] = coeffs[..., :n_slow]
    fast_coeffs[..., n_slow:] = coeffs[..., n_slow:]
    slow = SpectralField.idct(slow_coeffs)
    fast = SpectralField.idct(fast_coeffs)
    return slow, fast


class SpectralGate:
    """Gate computation based on spectral entropy.

    When spectral entropy is low -> coherent representation,
    can skip or reduce computation.
    """

    def __init__(self, threshold_low: float = 0.3, threshold_high: float = 0.7):
        self.threshold_low = threshold_low
        self.threshold_high = threshold_high

    def gate_value(self, x: np.ndarray) -> float:
        ent = spectral_entropy(x.ravel())
        if ent < self.threshold_low:
            return 0.0
        elif ent > self.threshold_high:
            return 1.0
        else:
            alpha = (ent - self.threshold_low) / (
                self.threshold_high - self.threshold_low + 1e-10
            )
            return float(alpha)

    def gate_ffn(self, x: np.ndarray) -> float:
        return self.gate_value(x)

    def gate_attention(self, x: np.ndarray) -> float:
        return self.gate_value(x)


class HDCSkipPredictor:
    """Hyperdimensional computing (HDC) predictor for layer skipping."""

    def __init__(self, dim: int = 512, hd_dim: int = 10000):
        self.dim = dim
        self.hd_dim = hd_dim
        rng = np.random.RandomState(42)
        self._projection = rng.randn(dim, hd_dim).astype(np.float32)
        self._projection /= (
            np.linalg.norm(self._projection, axis=0, keepdims=True) + 1e-10
        )
        self._state = np.zeros(hd_dim, dtype=np.float32)
        self._prototype_attention: Optional[np.ndarray] = None
        self._prototype_ffn: Optional[np.ndarray] = None
        self._confidence_threshold = 0.7

    def encode(self, x: np.ndarray) -> np.ndarray:
        flat = x.ravel()
        if flat.size > self.dim:
            flat = flat[: self.dim]
        elif flat.size < self.dim:
            flat = np.pad(flat, (0, self.dim - flat.size))
        return flat @ self._projection

    def update_state(self, x: np.ndarray):
        enc = self.encode(x)
        self._state = 0.9 * self._state + 0.1 * enc

    def set_prototype_attention(self, x: np.ndarray):
        self._prototype_attention = self.encode(x)

    def set_prototype_ffn(self, x: np.ndarray):
        self._prototype_ffn = self.encode(x)

    def should_skip_attention(self, x: np.ndarray) -> tuple[bool, float]:
        if self._prototype_attention is None:
            return False, 0.0
        enc = self.encode(x)
        sim = float(
            np.dot(enc, self._prototype_attention)
            / (np.linalg.norm(enc) * np.linalg.norm(self._prototype_attention) + 1e-10)
        )
        confidence = max(0.0, sim)
        return confidence > self._confidence_threshold, confidence

    def should_skip_ffn(self, x: np.ndarray) -> tuple[bool, float]:
        if self._prototype_ffn is None:
            return False, 0.0
        enc = self.encode(x)
        sim = float(
            np.dot(enc, self._prototype_ffn)
            / (np.linalg.norm(enc) * np.linalg.norm(self._prototype_ffn) + 1e-10)
        )
        confidence = max(0.0, sim)
        return confidence > self._confidence_threshold, confidence


class SymplecticIntegrator:
    """Hamiltonian leapfrog integrator per transformer layer."""

    def __init__(self, step_size: float = 0.1, n_steps: int = 1):
        self.step_size = step_size
        self.n_steps = n_steps

    def leapfrog(
        self,
        q: np.ndarray,
        p: np.ndarray,
        grad_potential: Callable[[np.ndarray], np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        eps = self.step_size
        for _ in range(self.n_steps):
            p_half = p - 0.5 * eps * grad_potential(q)
            q = q + eps * p_half
            p = p_half - 0.5 * eps * grad_potential(q)
        return q, p

    def layer_step(
        self,
        residual: np.ndarray,
        layer_fn: Callable[[np.ndarray], np.ndarray],
        momentum: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        d = residual.shape[-1]
        if momentum is None:
            momentum = np.zeros_like(residual)

        def grad_pot(x: np.ndarray) -> np.ndarray:
            return layer_fn(x)

        return self.leapfrog(residual, momentum, grad_pot)


class ResonanceRouter:
    """Route compute based on resonance score.

    High resonance -> run full compute path.
    Low resonance  -> run reduced (cheaper) compute path.
    """

    def __init__(
        self,
        window: int = 64,
        high_threshold: float = 0.7,
        low_threshold: float = 0.3,
    ):
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self._scores: deque = deque(maxlen=window)

    def record(self, score: float):
        self._scores.append(score)

    def resonance_score(self) -> float:
        if not self._scores:
            return 0.5
        return float(np.mean(list(self._scores)))

    def compute_budget(self) -> float:
        rs = self.resonance_score()
        if rs > self.high_threshold:
            return 1.0
        elif rs < self.low_threshold:
            return 0.25
        else:
            return 0.25 + 0.75 * (rs - self.low_threshold) / (
                self.high_threshold - self.low_threshold + 1e-10
            )

    def route_attention(
        self, full_output: np.ndarray, cheap_output: np.ndarray
    ) -> np.ndarray:
        budget = self.compute_budget()
        return budget * full_output + (1.0 - budget) * cheap_output


class HopfieldEnergy:
    """Modern Hopfield energy computation for attractor scoring.

    E(z) = -1/beta * logsumexp(beta * Xi^T . z)
    """

    def __init__(self, beta: float = 8.0, pattern_dim: int = 512):
        self.beta = beta
        self.pattern_dim = pattern_dim
        self.patterns: list[np.ndarray] = []
        self._pattern_matrix: Optional[np.ndarray] = None
        self._dirty = True

    def store(self, pattern: np.ndarray):
        self.patterns.append(pattern.ravel().copy())
        if len(self.patterns) > 256:
            self.patterns.pop(0)
        self._dirty = True

    def store_batch(self, patterns: np.ndarray):
        for p in patterns:
            self.store(p)

    def _rebuild(self):
        if not self.patterns:
            self._pattern_matrix = None
        else:
            self._pattern_matrix = np.stack(self.patterns, axis=0)
        self._dirty = False

    def energy(self, state: np.ndarray) -> float:
        if self._dirty:
            self._rebuild()
        if self._pattern_matrix is None:
            return 0.0
        state_flat = state.ravel()
        similarities = self._pattern_matrix @ state_flat
        max_sim = np.max(similarities)
        logsumexp = max_sim + np.log(
            np.sum(np.exp(self.beta * (similarities - max_sim))) + 1e-10
        )
        return float(-1.0 / self.beta * logsumexp)

    def score(self, state: np.ndarray, reference: Optional[np.ndarray] = None) -> float:
        if self._dirty:
            self._rebuild()
        if self._pattern_matrix is None:
            return 0.5
        e = self.energy(state)
        if reference is not None:
            e_ref = self.energy(reference)
            delta = e - e_ref
        else:
            delta = e
        return float(1.0 / (1.0 + np.exp(delta)))


class MonoidalChunk:
    """Process long context in chunks with overlap (monoidal composition)."""

    def __init__(self, chunk_size: int = 1024, overlap: int = 128):
        self.chunk_size = chunk_size
        self.overlap = overlap
        assert overlap < chunk_size, "Overlap must be less than chunk size"

    def chunk(self, seq: np.ndarray) -> list[tuple[int, int, np.ndarray]]:
        n = seq.shape[0]
        stride = self.chunk_size - self.overlap
        chunks = []
        start = 0
        while start < n:
            end = min(start + self.chunk_size, n)
            chunks.append((start, end, seq[start:end]))
            if end == n:
                break
            start += stride
        return chunks

    def merge(
        self,
        chunk_outputs: list[tuple[int, int, np.ndarray]],
        total_length: int,
    ) -> np.ndarray:
        d = chunk_outputs[0][2].shape[-1]
        result = np.zeros((total_length, d), dtype=np.float64)
        weight_sum = np.zeros((total_length, 1), dtype=np.float64)
        for start, end, out in chunk_outputs:
            n_chunk = out.shape[0]
            weights = np.ones(n_chunk, dtype=np.float64)
            if start > 0:
                ramp = np.minimum(
                    np.arange(self.overlap, dtype=np.float64) / self.overlap, 1.0
                )
                weights[: len(ramp)] = ramp
            if end < total_length:
                ramp = np.maximum(
                    1.0 - np.arange(self.overlap, dtype=np.float64) / self.overlap, 0.0
                )
                weights[-len(ramp) :] = np.minimum(weights[-len(ramp) :], ramp)
            result[start:end] += out * weights[:, None]
            weight_sum[start:end, 0] += weights
        weight_sum = np.maximum(weight_sum, 1e-10)
        return (result / weight_sum).astype(np.float32)


class AdaptiveSpectralRank:
    """Adjust DCT coefficient count per layer based on entropy."""

    def __init__(
        self,
        min_rank: int = 8,
        max_rank: int = 128,
        default_rank: int = 64,
    ):
        self.min_rank = min_rank
        self.max_rank = max_rank
        self.default_rank = default_rank

    def rank_for_layer(self, layer_input: np.ndarray) -> int:
        ent = spectral_entropy(layer_input.ravel())
        fraction = np.clip(ent, 0.0, 1.0)
        rank = int(self.min_rank + fraction * (self.max_rank - self.min_rank))
        return max(self.min_rank, min(self.max_rank, rank))

    def compress_layer(
        self,
        x: np.ndarray,
        layer_idx: int,
        prev_entropy: Optional[float] = None,
    ) -> tuple[np.ndarray, int]:
        if prev_entropy is not None:
            fraction = np.clip(prev_entropy, 0.0, 1.0)
        else:
            fraction = np.clip(spectral_entropy(x.ravel()), 0.0, 1.0)
        n_keep = int(self.min_rank + fraction * (self.max_rank - self.min_rank))
        n_keep = max(self.min_rank, min(self.max_rank, n_keep))
        return SpectralField.band_limit(x, n_keep), n_keep


class BornMachineSampler:
    """Quantum wavefunction sampling via Born's rule."""

    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature

    def sample(self, logits: np.ndarray, n_samples: int = 1) -> np.ndarray:
        logits = logits.astype(np.float64) / self.temperature
        max_l = np.max(logits, axis=-1, keepdims=True)
        probs = np.exp(logits - max_l)
        probs = probs / (np.sum(probs, axis=-1, keepdims=True) + 1e-10)
        flat = probs.reshape(-1, probs.shape[-1])
        indices = np.array(
            [np.random.choice(flat.shape[1], p=flat[i]) for i in range(flat.shape[0])]
        )
        if n_samples > 1:
            return indices.reshape(*probs.shape[:-1], n_samples)
        return indices.reshape(*probs.shape[:-1])

    def wavefunction(self, logits: np.ndarray) -> np.ndarray:
        psi = np.exp(logits.astype(np.float64) / self.temperature)
        return psi / np.sqrt(np.sum(psi**2, axis=-1, keepdims=True) + 1e-10)


class GroverAmplifier:
    """Grover-inspired amplification of high-value candidates."""

    def __init__(self, n_iterations: int = 3):
        self.n_iterations = n_iterations

    def amplify(
        self,
        amplitudes: np.ndarray,
        oracle: Callable[[np.ndarray], np.ndarray],
    ) -> np.ndarray:
        a = amplitudes.astype(np.float64).copy()
        n = len(a)
        a = a / np.sqrt(np.sum(a**2) + 1e-10)
        for _ in range(self.n_iterations):
            good = oracle(a).astype(np.float64)
            a = a * (1.0 - 2.0 * good)
            mean = np.mean(a)
            a = 2.0 * mean - a
        a = np.abs(a)
        a = a / (np.sum(a) + 1e-10)
        return a

    def amplify_candidates(
        self,
        scores: np.ndarray,
        n_keep: Optional[int] = None,
    ) -> np.ndarray:
        probs = np.maximum(scores, 0.0).astype(np.float64)
        probs = probs / (np.sum(probs) + 1e-10)
        amplitudes = np.sqrt(probs)

        def oracle(x: np.ndarray) -> np.ndarray:
            threshold = (
                np.median(scores) if n_keep is None else np.sort(scores)[-n_keep]
            )
            return (scores >= threshold).astype(np.float64)

        amplified = self.amplify(amplitudes, oracle)
        return amplified


class CrossLayerResonance:
    """Feed spectral entropy and resonance forward through layers."""

    def __init__(self, n_layers: int, momentum: float = 0.3):
        self.n_layers = n_layers
        self.momentum = momentum
        self._entropies: list[float] = []
        self._resonances: list[float] = []

    def record(self, entropy: float, resonance: float):
        self._entropies.append(entropy)
        self._resonances.append(resonance)

    def current_entropy(self) -> float:
        if not self._entropies:
            return 0.5
        smoothed = (1.0 - self.momentum) * self._entropies[-1]
        if len(self._entropies) > 1:
            smoothed += self.momentum * self._entropies[-2]
        return float(smoothed)

    def current_resonance(self) -> float:
        if not self._resonances:
            return 0.5
        return float(np.mean(self._resonances[-3:]))

    def modulation_factor(self) -> float:
        ent = self.current_entropy()
        res = self.current_resonance()
        return float(np.clip(1.0 - 0.5 * ent * res, 0.0, 1.0))

    def reset(self):
        self._entropies.clear()
        self._resonances.clear()
