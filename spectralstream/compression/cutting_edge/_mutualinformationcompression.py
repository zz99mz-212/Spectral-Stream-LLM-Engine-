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

class MutualInformationCompression(CompressionMethod):
    """Maximize mutual information between original and compressed weights.

    Mathematical basis:
        The Information Bottleneck method finds a compressed representation
        T that maximizes I(T; Y) while minimizing I(T; X):
            min_{p(t|x)} I(T; X) - beta * I(T; Y)

        For weight compression, X = original weights, Y = task-relevant
        features (here approximated by weight structure), T = compressed.

    Algorithm:
        1. Cluster weights into "information bottleneck" clusters
        2. Optimize cluster assignments to maximize relevant information
        3. Store: cluster centroids + assignments

    The beta parameter controls the rate-distortion tradeoff.
    """
    name = "mutual_information"
    category = "information_theory"

    def compress(self, tensor, n_clusters=16, beta=1.0, n_iter=20, **kw):
        t, orig = _ensure_2d(tensor)
        m, n = t.shape

        rng = np.random.RandomState(42)
        # Initialize centroids
        centroids = t[rng.choice(m, min(n_clusters, m), replace=False)].copy().astype(np.float64)

        # Target distribution: use weight gradients as "relevant" features
        grad = np.gradient(t.astype(np.float64), axis=1)

        for iteration in range(n_iter):
            # E-step: compute soft assignments
            dists = np.linalg.norm(t[:, None, :].astype(np.float64) - centroids[None, :, :], axis=2)

            # Information-theoretic weighting: P(t|x) ~ P(t) * exp(-beta * D(x, centroid))
            log_Ptx = -beta * dists
            log_Ptx -= log_Ptx.max(axis=1, keepdims=True)
            P_tx = np.exp(log_Ptx)
            P_tx /= P_tx.sum(axis=1, keepdims=True)

            # Cluster distribution: P(t) = (1/n) sum_x P(t|x)
            P_t = P_tx.mean(axis=0) + 1e-10
            P_t /= P_t.sum()

            # M-step: update centroids weighted by assignments
            for c in range(n_clusters):
                weights = P_tx[:, c] * P_t[c]
                total = weights.sum() + 1e-10
                centroids[c] = (t.astype(np.float64).T @ weights) / total

            # Compute mutual information I(X; T) - beta * I(T; Y) approximation
            # I(X; T) = sum_{x,t} P(x,t) log(P(t|x) / P(t))
            H_T = -np.sum(P_t * np.log(P_t + 1e-30))
            H_TX = -np.sum(P_tx * np.log(P_tx + 1e-30)) / m
            IX_T = H_T - H_TX

            if iteration > 0 and abs(IX_T - prev_IX_T) < 1e-6:
                break
            prev_IX_T = IX_T

        # Hard assignments
        assignments = np.argmin(dists, axis=1).astype(np.uint8)

        return {
            "centroids": centroids.astype(np.float32),
            "assignments": assignments,
            "n_clusters": n_clusters,
            "beta": beta,
            "IX_T": float(IX_T),
            "shape": t.shape,
        }, {"orig_shape": orig}

    def decompress(self, cd, meta):
        return cd["centroids"][cd["assignments"]].reshape(meta["orig_shape"]).astype(np.float32)

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

