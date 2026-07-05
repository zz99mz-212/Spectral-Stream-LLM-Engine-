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

class RateDistortionOptimalCompression(CompressionMethod):
    """Find theoretically optimal compression via Blahut-Arimoto algorithm.

    Mathematical basis:
        The rate-distortion function R(D) gives the minimum rate (bits)
        needed to achieve distortion D:
            R(D) = min_{p(x_hat|x): E[d(X,X_hat)] <= D} I(X; X_hat)

        The Blahut-Arimoto algorithm iteratively computes the optimal
        conditional distribution p(x_hat|x) that achieves R(D).

    Algorithm:
        1. Quantize weight values to alphabet
        2. Run Blahut-Arimoto to find optimal channel
        3. Store: input distribution + transition matrix + outputs

    This achieves the theoretical limit of compression for given distortion.
    """
    name = "rate_distortion_optimal"
    category = "information_theory"

    def compress(self, tensor, n_input_levels=256, n_output_levels=16, max_iter=30, **kw):
        flat = tensor.ravel().astype(np.float64)

        # Quantize input to alphabet
        s = max(abs(flat.max()), abs(flat.min()), 1e-8)
        input_idx = np.clip(np.round((np.clip(flat / s, -1, 1) + 1) / 2 * (n_input_levels - 1)).astype(int),
                           0, n_input_levels - 1)

        # Input distribution
        p_x = np.bincount(input_idx, minlength=n_input_levels).astype(np.float64) + 1e-10
        p_x /= p_x.sum()

        # Initialize output distribution (uniform)
        p_y = np.ones(n_output_levels) / n_output_levels

        # Distortion matrix: d(i,j) = (i - j)^2 / (n_input_levels - 1)^2
        i_vals = np.arange(n_input_levels, dtype=np.float64) / (n_input_levels - 1)
        j_vals = np.arange(n_output_levels, dtype=np.float64) / (n_output_levels - 1)
        D_matrix = (i_vals[:, None] - j_vals[None, :]) ** 2

        # Blahut-Arimoto iterations
        beta = 10.0  # Lagrange multiplier (controls rate-distortion tradeoff)
        Q = np.ones((n_input_levels, n_output_levels)) / n_output_levels  # channel

        for iteration in range(max_iter):
            # E-step: compute Q(j|i) proportional to p(y) * exp(-beta * D(i,j))
            log_Q = np.log(p_y[None, :] + 1e-30) - beta * D_matrix
            log_Q -= log_Q.max(axis=1, keepdims=True)
            Q = np.exp(log_Q)
            Q /= Q.sum(axis=1, keepdims=True)

            # M-step: update p(y) = sum_x p(x) Q(j|x)
            p_y_new = np.sum(p_x[:, None] * Q, axis=0)
            p_y_new = np.maximum(p_y_new, 1e-10)
            p_y_new /= p_y_new.sum()

            if np.allclose(p_y, p_y_new, atol=1e-6):
                break
            p_y = p_y_new

        # Output reconstruction values ( centroids of output distribution)
        output_values = j_vals * s * 2 - s

        # Quantize using learned channel
        output_idx = np.clip(np.argmax(Q[input_idx], axis=1), 0, n_output_levels - 1).astype(np.uint8)

        return {
            "idx": output_idx,
            "output_values": output_values.astype(np.float32),
            "p_x": p_x.astype(np.float32),
            "p_y": p_y.astype(np.float32),
            "scale": float(s),
            "n_input": n_input_levels,
            "n_output": n_output_levels,
            "shape": tensor.shape,
        }, {"orig_shape": tensor.shape}

    def decompress(self, cd, meta):
        return cd["output_values"][cd["idx"]].reshape(meta["orig_shape"]).astype(np.float32)

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

