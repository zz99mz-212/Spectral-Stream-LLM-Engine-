from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from ._compressionmethod import CompressionMethod, ALL_METHODS, _ensure_2d, _restore_shape, _safe_bytes


def _ensure_2d(t: np.ndarray) -> Tuple[np.ndarray, tuple]:
    if t.ndim == 1:
        return t.reshape(1, -1), t.shape
    if t.ndim > 2:
        orig = t.shape
        return t.reshape(orig[0], -1), orig
    return t, t.shape

def _restore_shape(t: np.ndarray, orig_shape: tuple) -> np.ndarray:
    return t.reshape(orig_shape) if t.shape != orig_shape else t

def _safe_bytes(data: Any) -> int:
    if isinstance(data, np.ndarray):
        return data.nbytes
    if isinstance(data, dict):
        return sum(_safe_bytes(v) for v in data.values()) + sum(_safe_bytes(k) for k in data.keys())
    if isinstance(data, (list, tuple)):
        return sum(_safe_bytes(x) for x in data)
    if isinstance(data, (int, float, np.integer, np.floating)):
        return 8
    if isinstance(data, str):
        return len(data)
    return 0

class QuantumTunnelingOptimizer(CompressionMethod):
    """Use quantum tunneling to escape local minima in compression optimization.

    Mathematical basis:
        In quantum annealing, the tunneling probability through a barrier
        of height V and width d is:
            P_tunnel ~ exp(-2 * d * sqrt(2*m*(V-E)) / hbar)

        We use this metaphor to "tunnel" through poor local optima when
        searching for the best quantization levels.

    Algorithm:
        1. Start with initial quantization (Lloyd-Max)
        2. Compute energy landscape (distortion)
        3. Apply tunneling: occasionally accept worse solutions with
           probability P = exp(-beta * delta_E)
        4. Temperature schedule: T(t) = T0 * alpha^t (simulated annealing
           with tunneling bonus)

    This finds better quantization than pure Lloyd-Max for multi-modal
    weight distributions.
    """
    name = "quantum_tunneling"
    category = "quantum_mechanics"

    def compress(self, tensor, n_bits=4, n_rounds=50, tunnel_rate=0.3, **kw):
        flat = tensor.ravel().astype(np.float64)
        n_levels = 1 << n_bits
        mu, sigma = float(np.mean(flat)), float(np.std(flat) + 1e-10)
        scale = max(abs(mu - 5 * sigma), abs(mu + 5 * sigma), 1e-8)
        normed = np.clip(flat / scale, -1.0, 1.0)

        # Initialize centroids via Lloyd-Max
        centroids = np.linspace(normed.min(), normed.max(), n_levels)
        for _ in range(20):
            bds = (centroids[1:] + centroids[:-1]) / 2.0
            idx = np.clip(np.digitize(normed, bds), 0, n_levels - 1)
            nc = np.array([normed[idx == i].mean() if np.any(idx == i) else centroids[i]
                           for i in range(n_levels)])
            if np.allclose(centroids, nc, atol=1e-6):
                break
            centroids = nc

        # Quantum tunneling optimization
        rng = np.random.RandomState(42)
        T0 = 0.5
        alpha = 0.95
        T = T0
        best_centroids = centroids.copy()
        best_distortion = float(np.mean((normed - centroids[np.clip(np.digitize(normed, bds), 0, n_levels - 1)]) ** 2))

        for round_idx in range(n_rounds):
            # Perturb centroids
            trial_centroids = centroids + rng.randn(n_levels) * T * 0.1
            trial_centroids = np.sort(trial_centroids)
            bds_t = (trial_centroids[1:] + trial_centroids[:-1]) / 2.0
            trial_idx = np.clip(np.digitize(normed, bds_t), 0, n_levels - 1)
            trial_distortion = float(np.mean((normed - trial_centroids[trial_idx]) ** 2))

            delta_E = trial_distortion - best_distortion
            if delta_E < 0:
                # Improvement — accept
                centroids = trial_centroids
                best_centroids = trial_centroids.copy()
                best_distortion = trial_distortion
            else:
                # Quantum tunneling: accept with probability
                # P = exp(-delta_E / T) * tunnel_bonus
                tunnel_bonus = np.exp(-tunnel_rate * delta_E / (T + 1e-10))
                P = np.exp(-delta_E / (T + 1e-10)) * (1.0 + tunnel_bonus)
                if rng.random() < min(P, 1.0):
                    centroids = trial_centroids
                    # Don't update best if worse

            T *= alpha

        # Final quantization with best centroids
        bds = (best_centroids[1:] + best_centroids[:-1]) / 2.0
        indices = np.clip(np.digitize(normed, bds), 0, n_levels - 1).astype(np.uint8)

        return {
            "idx": indices,
            "cb": best_centroids.astype(np.float32),
            "scale": float(scale),
            "shape": tensor.shape,
        }, {"orig_shape": tensor.shape}

    def decompress(self, cd, meta):
        return (cd["cb"][cd["idx"]] * cd["scale"]).reshape(meta["orig_shape"]).astype(np.float32)

def _generate_monomials(n_vars: int, degree: int) -> list:
    """Generate all monomials of given degree in n_vars variables."""
    if degree == 0:
        return [()]
    if degree == 1:
        return [(i,) for i in range(n_vars)]
    result = []
    for i in range(n_vars):
        for rest in _generate_monomials(n_vars, degree - 1):
            if len(rest) == 0 or i >= rest[0]:
                result.append((i,) + rest)
    return result[:50]  # limit for efficiency

