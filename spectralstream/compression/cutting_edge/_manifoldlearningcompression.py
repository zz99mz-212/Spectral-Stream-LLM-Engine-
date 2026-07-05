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

class ManifoldLearningCompression(CompressionMethod):
    """Treat weight matrix rows as points on a low-dimensional manifold.

    Mathematical basis:
        If weight rows lie near a low-dimensional manifold M embedded
        in R^n, then we can find an isometric embedding phi: M -> R^d
        with d << n using Isomap or Locally Linear Embedding (LLE).

        We store the d-dimensional coordinates and the reconstruction
        mapping (decoder).

    Algorithm:
        1. Build k-NN graph of weight rows
        2. Compute geodesic distances (Isomap) or local weights (LLE)
        3. Embed to d dimensions via MDS or eigen-decomposition
        4. Store: d-dimensional coordinates + decoder matrix

    Storage: O(m*d + n*d) instead of O(m*n).
    """
    name = "manifold_learning"
    category = "advanced_mathematics"

    def compress(self, tensor, n_neighbors=8, n_components=8, **kw):
        t, orig = _ensure_2d(tensor)
        m, n = t.shape
        d = min(n_components, min(m, n) - 1)
        k = min(n_neighbors, m - 1)

        W = t.astype(np.float64)

        # Step 1: Build k-NN graph
        from scipy.spatial.distance import cdist  # type: ignore
        dist_matrix = cdist(W, W)

        # k-NN adjacency
        knn_graph = np.zeros((m, m), dtype=np.float64)
        for i in range(m):
            nearest = np.argsort(dist_matrix[i])[1:k + 1]
            knn_graph[i, nearest] = dist_matrix[i, nearest]
            knn_graph[nearest, i] = dist_matrix[i, nearest]

        # Step 2: Compute geodesic distances via Floyd-Warshall
        geodesic = dist_matrix.copy()
        geodesic[knn_graph == 0] = np.inf
        np.fill_diagonal(geodesic, 0)

        for kk in range(m):
            for i in range(m):
                for j in range(m):
                    if geodesic[i, kk] + geodesic[kk, j] < geodesic[i, j]:
                        geodesic[i, j] = geodesic[i, kk] + geodesic[kk, j]

        # Replace inf with max finite distance
        max_dist = np.max(geodesic[geodesic < np.inf])
        geodesic[geodesic == np.inf] = max_dist * 2

        # Step 3: Classical MDS embedding
        geodesic_sq = geodesic ** 2
        mds_double = geodesic_sq.astype(np.float64)
        H = np.eye(m) - np.ones((m, m)) / m
        B = -0.5 * H @ mds_double @ H

        eigvals, eigvecs = np.linalg.eigh(B)
        order = np.argsort(eigvals)[::-1][:d]
        embed_coords = eigvecs[:, order] * np.sqrt(np.maximum(eigvals[order], 0))[None, :]

        # Step 4: Learn decoder (linear mapping from embedding to original space)
        # W_approx = embed_coords @ decoder^T
        # Solve: decoder^T = pinv(embed_coords) @ W
        decoder = np.linalg.lstsq(embed_coords, W, rcond=None)[0]

        return {
            "coords": embed_coords.astype(np.float32),
            "decoder": decoder.astype(np.float32),
            "d": d,
            "k": k,
            "shape": t.shape,
        }, {"orig_shape": orig}

    def decompress(self, cd, meta):
        coords = cd["coords"].astype(np.float64)
        decoder = cd["decoder"].astype(np.float64)
        result = coords @ decoder
        return _restore_shape(result.astype(np.float32), meta["orig_shape"])

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

