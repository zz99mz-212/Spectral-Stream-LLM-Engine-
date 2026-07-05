"""Lloyd-Max quantization."""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class LloydMaxQuantizer:
    """Lloyd-Max MSE-optimal scalar quantizer with iterative centroid updates."""

    def __init__(self, n_bits: int = 8, max_iter: int = 100, tol: float = 1e-6) -> None:
        self.n_bits = n_bits
        self.n_levels = 1 << n_bits
        self.max_iter = max_iter
        self.tol = tol
        self.centroids: Optional[np.ndarray] = None
        self.boundaries: Optional[np.ndarray] = None
        self.trained: bool = False
        self.scale: float = 1.0

    def train(self, data: np.ndarray, max_iter: int = 100) -> None:
        data = np.asarray(data)
        if data.size == 0:
            raise ValueError("Cannot train quantizer on empty data")
        try:
            flat = data.ravel().astype(np.float64)
            mu, sigma = np.mean(flat), np.std(flat)
            self.scale = max(abs(mu - 4 * sigma), abs(mu + 4 * sigma), 1e-8)
            normalized = flat / self.scale
            clipped = np.clip(normalized, -1.0, 1.0)
            centroids = np.linspace(-1.0, 1.0, self.n_levels)
            for _ in range(max_iter):
                boundaries = (centroids[1:] + centroids[:-1]) / 2.0
                indices = np.digitize(clipped, boundaries)
                new_centroids = np.array(
                    [
                        np.mean(clipped[indices == i])
                        if np.any(indices == i)
                        else centroids[i]
                        for i in range(self.n_levels)
                    ]
                )
                if np.allclose(centroids, new_centroids, atol=1e-6):
                    break
                centroids = new_centroids
            self.centroids = centroids * self.scale
            self.boundaries = boundaries * self.scale
            self.trained = True
            self._n_train = len(flat)
        except (ValueError, FloatingPointError) as e:
            logger.error("LloydMaxQuantizer.train failed: %s", e)

    def fit(self, data: np.ndarray) -> LloydMaxQuantizer:
        self.train(data)
        return self

    def quantize(self, data: np.ndarray) -> np.ndarray:
        if not self.trained:
            raise RuntimeError("LloydMaxQuantizer not trained. Call fit() first.")
        try:
            flat = data.ravel().astype(np.float64)
            indices = np.digitize(flat, self.boundaries)
            indices = np.clip(indices, 0, self.n_levels - 1).astype(np.uint8)
            return self.centroids[indices].reshape(data.shape)
        except (IndexError, ValueError) as e:
            logger.error("LloydMaxQuantizer.quantize failed: %s", e)
            raise

    def compress(self, vectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        vectors = np.asarray(vectors)
        if not self.trained:
            self.train(vectors)
        try:
            flat = vectors.ravel()
            indices = np.digitize(flat, self.boundaries)
            indices = np.clip(indices, 0, self.n_levels - 1).astype(np.uint8)
            return indices, self.centroids.copy()
        except (IndexError, ValueError) as e:
            logger.error("LloydMaxQuantizer.compress failed: %s", e)
            raise

    def decompress(self, indices: np.ndarray, shape: tuple) -> np.ndarray:
        if self.centroids is None:
            raise ValueError("Quantizer not trained")
        return self.centroids[indices.ravel()].reshape(shape)


def vectorized_lloyd_max(
    data: np.ndarray, n_bits: int = 4, n_iter: int = 20
) -> Tuple[np.ndarray, np.ndarray, float]:
    quantizer = LloydMaxQuantizer(n_bits=n_bits, max_iter=n_iter)
    quantizer.train(data)
    quantized = quantizer.quantize(data)
    return quantized, quantizer.centroids, quantizer.scale
