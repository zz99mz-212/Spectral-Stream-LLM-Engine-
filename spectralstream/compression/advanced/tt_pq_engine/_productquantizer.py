from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from spectralstream.core.math_primitives import (
    HadamardRotator,
    LloydMaxQuantizer,
    dct,
    fwht,
    idct,
    next_power_of_two,
    softmax,
    spectral_entropy,
)


class ProductQuantizer:
    """Product Quantization with M subspaces and k-means++ initialization.

    Splits vectors into M disjoint sub-vectors and quantizes each
    independently with a local codebook, achieving exponential
    compression (e.g., M=8, 4-bit → 32x compression).

    Supports 4-bit, 6-bit, and 8-bit quantization via configurable
    codebook sizes and Lloyd-Max optimization.
    """

    def __init__(self, config: Optional[PQConfig] = None):
        self.config = config or PQConfig()
        self.n_subspaces = self.config.n_subspaces
        self.n_bits = self.config.n_bits
        self.n_clusters = min(
            1 << self.n_bits,
            self.config.n_clusters_per_subspace,
        )
        self._codebooks: Optional[List[np.ndarray]] = None
        self._subspace_dim: Optional[int] = None
        self._trained = False

    def _kmeans_plus_plus_init(
        self, data: np.ndarray, n_clusters: int, rng: np.random.RandomState
    ) -> np.ndarray:
        """K-means++ initialization for codebook centroids."""
        n_samples = data.shape[0]
        centroids = [data[rng.randint(n_samples)].copy()]

        for _ in range(1, n_clusters):
            dists = np.min(
                np.linalg.norm(data[:, np.newaxis] - np.array(centroids), axis=2),
                axis=1,
            )
            probs = dists ** 2 / (np.sum(dists ** 2) + 1e-30)
            idx = rng.choice(n_samples, p=probs)
            centroids.append(data[idx].copy())

        return np.array(centroids)

    def _lloyd_max(
        self, data: np.ndarray, codebook: np.ndarray
    ) -> np.ndarray:
        """Lloyd-Max algorithm for optimal scalar quantizer per subspace."""
        n_clusters = codebook.shape[0]
        centroids = codebook.copy()

        for _ in range(self.config.lloyd_max_iterations):
            # Assignment
            dists = np.linalg.norm(
                data[:, np.newaxis] - centroids[np.newaxis, :], axis=2
            )
            labels = np.argmin(dists, axis=1)

            # Update
            new_centroids = np.empty_like(centroids)
            for k in range(n_clusters):
                members = data[labels == k]
                if len(members) > 0:
                    new_centroids[k] = np.mean(members, axis=0)
                else:
                    new_centroids[k] = centroids[k]

            if np.allclose(centroids, new_centroids, atol=1e-6):
                break
            centroids = new_centroids

        return centroids

    def train(self, vectors: np.ndarray, seed: int = 42) -> None:
        """Train PQ codebooks on the input vectors.

        Args:
            vectors: Input matrix of shape (n_samples, dimension).
            seed: RNG seed for reproducibility.
        """
        vectors = np.asarray(vectors, dtype=np.float64)
        if vectors.ndim != 2:
            raise ValueError(f"Expected 2D input, got {vectors.ndim}D")

        n_samples, dim = vectors.shape
        rng = np.random.RandomState(seed)

        self._subspace_dim = dim // self.n_subspaces
        if self._subspace_dim < 1:
            self._subspace_dim = 1
            self.n_subspaces = dim

        codebooks = []
        for m in range(self.n_subspaces):
            start = m * self._subspace_dim
            end = min(start + self._subspace_dim, dim)
            subspace_data = vectors[:, start:end]

            codebook = self._kmeans_plus_plus_init(
                subspace_data, self.n_clusters, rng
            )
            codebook = self._lloyd_max(subspace_data, codebook)
            codebooks.append(codebook)

        self._codebooks = codebooks
        self._trained = True
        logger.info(
            "PQ trained: subspaces=%d, clusters=%d, bits=%d",
            self.n_subspaces, self.n_clusters, self.n_bits,
        )

    def encode(self, vectors: np.ndarray) -> np.ndarray:
        """Encode vectors to PQ indices.

        Args:
            vectors: Input matrix of shape (n_samples, dimension).

        Returns:
            Indices matrix of shape (n_samples, n_subspaces).
        """
        vectors = np.asarray(vectors, dtype=np.float64)
        if not self._trained:
            self.train(vectors)

        n_samples = vectors.shape[0]
        indices = np.zeros((n_samples, self.n_subspaces), dtype=np.int32)

        for m in range(self.n_subspaces):
            start = m * self._subspace_dim
            end = min(start + self._subspace_dim, vectors.shape[1])
            subspace_data = vectors[:, start:end]
            codebook = self._codebooks[m]

            dists = np.linalg.norm(
                subspace_data[:, np.newaxis] - codebook[np.newaxis, :], axis=2
            )
            indices[:, m] = np.argmin(dists, axis=1).astype(np.int32)

        return indices

    def decode(self, indices: np.ndarray) -> np.ndarray:
        """Decode PQ indices back to vectors.

        Args:
            indices: Indices matrix of shape (n_samples, n_subspaces).

        Returns:
            Reconstructed vectors of shape (n_samples, dimension).
        """
        if self._codebooks is None:
            raise ValueError("No codebooks; call train() first")

        n_samples = indices.shape[0]
        dim = self.n_subspaces * self._subspace_dim
        reconstructed = np.zeros((n_samples, dim), dtype=np.float64)

        for m in range(self.n_subspaces):
            start = m * self._subspace_dim
            end = start + self._subspace_dim
            reconstructed[:, start:end] = self._codebooks[m][indices[:, m]]

        return reconstructed

    def bits_per_vector(self) -> int:
        """Total bits per vector: n_subspaces * n_bits."""
        return self.n_subspaces * self.n_bits

    def compression_ratio(self, dim: int) -> float:
        """Compression ratio vs float32."""
        original_bits = dim * 32
        compressed_bits = self.bits_per_vector()
        return original_bits / max(compressed_bits, 1)
