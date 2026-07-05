from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class TopologicalQuantization:
    """Topological functional quantization.

    Constructs a simplicial complex from weight vectors, identifies
    persistent topological features, and uses them to build a
    geometrically-informed codebook.
    """

    name = "topological_quant"
    category = "physics"

    def __init__(
        self, codebook_size: int = 256, n_neighbors: int = 16, n_iter: int = 20
    ):
        self.codebook_size = codebook_size
        self.n_neighbors = n_neighbors
        self.n_iter = n_iter

    def _persistence_features(self, data: np.ndarray) -> np.ndarray:
        n = min(len(data), self.n_neighbors + 1)
        if n < 2:
            n = min(2, self.n_neighbors + 1)
            padded = np.zeros((n, data.shape[1]), dtype=np.float64)
            padded[: len(data)] = data
            subset = padded
        else:
            subset = data[:n]
        dists = np.linalg.norm(subset[:, np.newaxis] - subset[np.newaxis, :], axis=2)
        features = []
        for i in range(n):
            neighbors = np.sort(dists[i])[1:]
            birth = float(neighbors[0]) if len(neighbors) > 0 else 0
            death = float(neighbors[-1]) if len(neighbors) > 1 else birth + 1
            features.extend([birth, death, death - birth])
        out = np.array(features, dtype=np.float64)
        expected = 3 * self.n_neighbors
        if len(out) < expected:
            out = np.pad(out, (0, expected - len(out)), constant_values=0)
        return out[:expected]

    def _build_codebook(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        k = min(self.codebook_size, len(data))
        rng = np.random.RandomState(42)
        idx = rng.choice(len(data), size=k, replace=False)
        centroids = data[idx].copy()

        for _ in range(self.n_iter):
            dists = np.linalg.norm(
                data[:, np.newaxis] - centroids[np.newaxis, :], axis=2
            )
            labels = np.argmin(dists, axis=1)
            for c in range(k):
                mask = labels == c
                if np.any(mask):
                    centroids[c] = np.mean(data[mask], axis=0)
        return centroids, labels

    def compress(
        self, tensor: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        codebook_size = kwargs.get("codebook_size", self.codebook_size)
        orig_shape = tensor.shape

        mat = tensor.reshape(-1, tensor.shape[-1]).astype(np.float64)

        features = np.zeros(
            (len(mat), 3 * min(len(mat), self.n_neighbors)), dtype=np.float64
        )
        for i in range(len(mat)):
            features[i] = self._persistence_features(mat[i : i + 1])

        codebook, codes = self._build_codebook(features)

        data_out = {
            "codes": codes.astype(np.uint16),
            "codebook": codebook.astype(np.float32),
        }
        meta = {"orig_shape": orig_shape, "method": self.name}
        return data_out, meta

    def decompress(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> np.ndarray:
        codebook = data["codebook"]
        codes = data["codes"]
        reconstructed = codebook[codes.astype(int)]
        return (
            reconstructed[:, : metadata["orig_shape"][-1]]
            .reshape(metadata["orig_shape"])
            .astype(np.float32)
        )

    def estimate_ratio(self, tensor: np.ndarray, **kwargs) -> float:
        codebook_size = kwargs.get("codebook_size", self.codebook_size)
        orig = tensor.nbytes
        comp = tensor.shape[0] * 2 + codebook_size * tensor.shape[-1] * 4
        return comp / max(orig, 1)
