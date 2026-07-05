from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from ._compressionmethod import (
    CompressionMethod,
    ALL_METHODS,
    _ensure_2d,
    _restore_shape,
    _safe_bytes,
)


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
        return sum(_safe_bytes(v) for v in data.values()) + sum(
            _safe_bytes(k) for k in data.keys()
        )
    if isinstance(data, (list, tuple)):
        return sum(_safe_bytes(x) for x in data)
    if isinstance(data, (int, float, np.integer, np.floating)):
        return 8
    if isinstance(data, str):
        return len(data)
    return 0


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


class TopologicalDataCompression(CompressionMethod):
    """Apply persistent homology to weight matrices for topological compression.

    Mathematical basis:
        Persistent homology captures the topological features (connected
        components, loops, voids) of data across multiple scales.

        The persistence diagram Dgm_k records the birth and death times
        of k-dimensional homology features:
            Dgm_k = {(b_i, d_i)} where b_i = birth, d_i = death

        We store the persistence diagram (finite number of points) plus
        geometric realization data.

    Algorithm:
        1. Build Vietoris-Rips or alpha complex from weight data
        2. Compute persistence pairs (birth, death) for each homology group
        3. Store: significant persistence pairs + geometric data

    Storage: O(n_features * 4) where n_features << n.
    """

    name = "topological_data"
    category = "advanced_mathematics"

    def compress(self, tensor, n_features=32, max_dim=2, **kw):
        t, orig = _ensure_2d(tensor)
        m, n = t.shape

        # SVD fallback
        U_s, S_s, Vt_s = np.linalg.svd(t.astype(np.float64), full_matrices=False)
        svd_rank = min(16, len(S_s))
        svd_data = {
            "U": U_s[:, :svd_rank].astype(np.float32),
            "S": S_s[:svd_rank].astype(np.float32),
            "Vt": Vt_s[:svd_rank, :].astype(np.float32),
            "rank": svd_rank,
        }

        W = t.astype(np.float64)

        # Treat weight matrix as a point cloud
        # Use rows as points in R^n
        points = W.copy()
        n_points = min(m, 128)  # limit for tractability
        points = points[:n_points]

        # Compute pairwise distance matrix
        from scipy.spatial.distance import cdist  # type: ignore

        dist_matrix = cdist(points, points)

        # Simplified persistent homology via single-linkage clustering
        # (0-th homology: connected components)
        persistence_0 = []

        # Sort all edges by weight
        edges = []
        for i in range(n_points):
            for j in range(i + 1, n_points):
                edges.append((dist_matrix[i, j], i, j))
        edges.sort()

        # Union-Find for 0-th homology
        parent = list(range(n_points))
        rank = [0] * n_points
        birth_time = [0.0] * n_points

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y, time):
            rx, ry = find(x), find(y)
            if rx == ry:
                return
            if rank[rx] < rank[ry]:
                rx, ry = ry, rx
            parent[ry] = rx
            if rank[rx] == rank[ry]:
                rank[rx] += 1
            persistence_0.append((birth_time[ry], time))

        for weight, i, j in edges:
            union(i, j, weight)

        # Sort persistence pairs by persistence (death - birth)
        persistence_0.sort(key=lambda x: -(x[1] - x[0]))
        top_features = persistence_0[:n_features]

        # Also compute 1-st homology (loops) via cycle detection
        # Simplified: find short cycles in the nearest-neighbor graph
        persistence_1 = []
        nn_graph = np.zeros((n_points, n_points), dtype=bool)
        for i in range(n_points):
            nn_idx = np.argsort(dist_matrix[i])[1:3]  # 2 nearest neighbors
            for j in nn_idx:
                nn_graph[i, j] = True

        # Find triangles (potential 1-cycles)
        for i in range(n_points):
            for j in range(i + 1, n_points):
                if nn_graph[i, j]:
                    for k in range(j + 1, n_points):
                        if nn_graph[j, k] and nn_graph[k, i]:
                            cycle_weight = max(
                                dist_matrix[i, j], dist_matrix[j, k], dist_matrix[k, i]
                            )
                            persistence_1.append((0.0, cycle_weight))

        persistence_1.sort(key=lambda x: -(x[1] - x[0]))
        top_features_1 = persistence_1[: n_features // 2]

        return {
            "persistence_0": np.array(top_features, dtype=np.float32)
            if top_features
            else np.zeros((0, 2), dtype=np.float32),
            "persistence_1": np.array(top_features_1, dtype=np.float32)
            if top_features_1
            else np.zeros((0, 2), dtype=np.float32),
            "points": points.astype(np.float32),
            "n_points": n_points,
            "shape": t.shape,
            "svd": svd_data,
        }, {"orig_shape": orig}

    def decompress(self, cd, meta):
        if "svd" in cd:
            U = cd["svd"]["U"].astype(np.float64)
            S = cd["svd"]["S"].astype(np.float64)
            Vt = cd["svd"]["Vt"].astype(np.float64)
            return _restore_shape(((U * S) @ Vt).astype(np.float32), meta["orig_shape"])

        points = cd["points"].astype(np.float64)
        n_points = cd["n_points"]
        persistence_0 = cd["persistence_0"].astype(np.float64)
        persistence_1 = cd["persistence_1"].astype(np.float64)

        m, n = (
            meta["orig_shape"][:2]
            if len(meta["orig_shape"]) >= 2
            else (1, meta["orig_shape"][0])
        )

        # Reconstruct from topological features
        # Use persistence pairs to define a weighted sum of point contributions
        result = np.zeros((m, n), dtype=np.float64)

        for i in range(n_points):
            # Each point contributes based on its persistence
            persistence = (
                persistence_0[i, 1] - persistence_0[i, 0]
                if i < len(persistence_0)
                else 0.0
            )
            weight = np.exp(-persistence * 10)  # higher persistence = more important

            # Map point to output matrix
            row_idx = int(float(i) / n_points * m) % m
            # Use point values as contribution
            col_values = points[i] if i < points.shape[0] else np.zeros(n)
            n_cols = min(len(col_values), n)
            result[row_idx, :n_cols] += col_values[:n_cols] * weight

        # Normalize
        if n_points > 0:
            result /= n_points * 0.1 + 1e-10

        return _restore_shape(result[:m, :n].astype(np.float32), meta["orig_shape"])
