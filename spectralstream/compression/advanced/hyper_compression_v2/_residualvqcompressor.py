from __future__ import annotations

import json
import logging
import math
import pickle
import struct
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


def _format_size(n: int) -> str:
    if n >= 1024**3:
        return f"{n / 1024**3:.1f} GB"
    if n >= 1024**2:
        return f"{n / 1024**2:.1f} MB"
    if n >= 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n} B"


class ResidualVQCompressor:
    """Residual Vector Quantization for weight compression.

    Uses multiple codebooks iteratively: each codebook quantizes the
    residual error from the previous codebook, achieving high fidelity
    at low bitrates.
    """

    def __init__(
        self,
        n_codebooks: int = 4,
        codebook_size: int = 256,
        n_bits: int = 8,
    ) -> None:
        """
        Args:
            n_codebooks: Number of residual quantization stages.
            codebook_size: Number of vectors per codebook.
            n_bits: Bits per codebook index.
        """
        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size
        self.n_bits = n_bits

    @staticmethod
    def _pairwise_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Efficient pairwise L2 distances without 3D broadcast."""
        a_norm = np.sum(a**2, axis=1, keepdims=True)
        b_norm = np.sum(b**2, axis=1, keepdims=True)
        dot = a @ b.T
        return np.sqrt(np.maximum(a_norm + b_norm.T - 2 * dot, 0))

    def _learn_codebook(self, vectors: np.ndarray, k: int) -> np.ndarray:
        n, dim = vectors.shape
        if n == 0:
            return np.zeros((k, dim), dtype=np.float64)
        k = min(k, n)
        rng = np.random.RandomState(42)
        max_train = min(10000, n)
        subset_idx = rng.choice(n, size=max_train, replace=False)
        subset = vectors[subset_idx]

        indices = rng.choice(max_train, size=k, replace=(max_train < k))
        centroids = subset[indices].copy()

        for _ in range(10):
            dists = self._pairwise_distances(subset, centroids)
            assignments = np.argmin(dists, axis=1)
            for j in range(k):
                mask = assignments == j
                if np.any(mask):
                    centroids[j] = subset[mask].mean(axis=0)
        return centroids

    def compress(self, tensor: np.ndarray) -> Tuple[bytes, Dict[str, Any]]:
        t = np.asarray(tensor, dtype=np.float64)
        orig_shape = t.shape
        flat = t.ravel()
        n = len(flat)

        dim = min(16, n)
        n_vectors = max(1, n // dim)
        truncated = flat[: n_vectors * dim].reshape(n_vectors, dim)
        residual = truncated.copy()

        codebooks: List[np.ndarray] = []
        all_indices: List[np.ndarray] = []

        for _ in range(self.n_codebooks):
            if residual.size == 0:
                break
            codebook = self._learn_codebook(residual, self.codebook_size)
            codebooks.append(codebook)
            dists = self._pairwise_distances(residual, codebook)
            indices = np.argmin(dists, axis=1)
            all_indices.append(indices)
            reconstructed = codebook[indices]
            residual = residual - reconstructed

        codebook_data = [cb.astype(np.float32).tobytes() for cb in codebooks]
        index_data = [idx.astype(np.uint8).tobytes() for idx in all_indices]

        compressed_dict = {
            "n_codebooks": len(codebooks),
            "codebook_size": self.codebook_size,
            "vector_dim": dim,
            "n_vectors": n_vectors,
            "codebooks": codebook_data,
            "indices": index_data,
        }
        data = pickle.dumps(compressed_dict)
        total_bytes = sum(len(d) for d in codebook_data) + sum(
            len(d) for d in index_data
        )
        metadata: Dict[str, Any] = {
            "orig_shape": list(orig_shape),
            "n_bytes": total_bytes,
            "type": "residual_vq",
        }
        return data, metadata

    def decompress(self, compressed: bytes, metadata: Dict[str, Any]) -> np.ndarray:
        orig_shape = tuple(metadata["orig_shape"])
        d = pickle.loads(compressed)
        n_codebooks = d["n_codebooks"]
        vector_dim = d["vector_dim"]
        n_vectors = d["n_vectors"]
        codebook_data = d.get("codebooks", [])
        index_data = d.get("indices", [])

        result = np.zeros((n_vectors, vector_dim), dtype=np.float64)
        for i in range(min(n_codebooks, len(codebook_data), len(index_data))):
            cb_raw = codebook_data[i]
            if isinstance(cb_raw, str):
                cb_raw = cb_raw.encode("latin-1")
            codebook = np.frombuffer(cb_raw, dtype=np.float32).reshape(-1, vector_dim)
            indices = np.frombuffer(index_data[i], dtype=np.uint8)
            result += codebook[indices[:n_vectors]]

        flat = result.ravel()[: int(np.prod(orig_shape))]
        if len(flat) < int(np.prod(orig_shape)):
            flat = np.pad(flat, (0, int(np.prod(orig_shape)) - len(flat)))
        return flat.reshape(orig_shape).astype(np.float32)
