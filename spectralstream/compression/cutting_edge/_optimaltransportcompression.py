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


class OptimalTransportCompression(CompressionMethod):
    """Use optimal transport (Wasserstein distance) for compression.

    Mathematical basis:
        The optimal transport plan T between two distributions mu and nu
        minimizes the transport cost:
            T* = argmin_T <C, T>  subject to  T @ 1 = mu, T^T @ 1 = nu

        where C is the cost matrix and T is the coupling.

        For weight compression:
        - Source: simple distribution (e.g., uniform)
        - Target: weight distribution
        - Store the transport plan parameters instead of full weights

    Algorithm:
        1. Discretize weight distribution into histogram bins
        2. Compute optimal transport from uniform to weight histogram
        3. Store: transport plan (sparse) + bin locations

    Storage: O(n_bins^2) for the transport plan.
    """

    name = "optimal_transport"
    category = "advanced_mathematics"

    def compress(self, tensor, n_bins=16, svd_rank=16, **kw):
        t, orig = _ensure_2d(tensor)
        m, n = t.shape
        W = t.ravel().astype(np.float64)

        # SVD fallback
        U, S, Vt = np.linalg.svd(t.astype(np.float64), full_matrices=False)
        k = min(svd_rank, len(S))
        svd_data = {
            "U": U[:, :k].astype(np.float32),
            "S": S[:k].astype(np.float32),
            "Vt": Vt[:k, :].astype(np.float32),
            "k": k,
        }

        hist, bin_edges = np.histogram(W, bins=n_bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        source = np.ones(n_bins) / n_bins
        target = hist.astype(np.float64) / (len(W) + 1e-10)
        target = target / (target.sum() + 1e-10)

        C = (bin_centers[:, None] - bin_centers[None, :]) ** 2
        eps = 0.01
        K = np.exp(-C / eps)

        u = np.ones(n_bins) / n_bins
        v = np.ones(n_bins) / n_bins

        for _ in range(50):
            u = source / (K @ v + 1e-30)
            v = target / (K.T @ u + 1e-30)
            u = np.clip(u, 1e-30, 1e30)
            v = np.clip(v, 1e-30, 1e30)

        T = np.diag(u) @ K @ np.diag(v)
        threshold = np.percentile(T.ravel(), 70)
        mask = T > threshold
        sparse_T = T[mask]
        sparse_idx = np.argwhere(mask)

        return {
            "sparse_T": sparse_T.astype(np.float32),
            "sparse_idx": sparse_idx.astype(np.int32),
            "bin_centers": bin_centers.astype(np.float32),
            "source": source.astype(np.float32),
            "target": target.astype(np.float32),
            "n_bins": n_bins,
            "shape": t.shape,
            "svd": svd_data,
        }, {"orig_shape": tensor.shape}

    def decompress(self, cd, meta):
        if "svd" in cd:
            U = cd["svd"]["U"].astype(np.float64)
            S = cd["svd"]["S"].astype(np.float64)
            Vt = cd["svd"]["Vt"].astype(np.float64)
            return ((U * S) @ Vt).reshape(meta["orig_shape"]).astype(np.float32)

        n_bins = cd["n_bins"]
        bin_centers = cd["bin_centers"].astype(np.float64)
        target = cd["target"].astype(np.float64)
        n = np.prod(meta["orig_shape"])

        result = np.zeros(n, dtype=np.float64)
        for i in range(n_bins):
            n_samples = int(target[i] * n)
            if n_samples > 0:
                start = int(float(i) / n_bins * n)
                end = min(start + n_samples, n)
                result[start:end] = bin_centers[i]

        if cd["sparse_idx"].shape[0] > 0:
            for k in range(cd["sparse_idx"].shape[0]):
                i, j = cd["sparse_idx"][k]
                if i < n_bins and j < n_bins:
                    idx_start = int(float(j) / n_bins * n)
                    idx_end = min(idx_start + int(cd["sparse_T"][k] * n * 10), n)
                    result[idx_start:idx_end] += bin_centers[i] * 0.01

        return result.reshape(meta["orig_shape"]).astype(np.float32)


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
