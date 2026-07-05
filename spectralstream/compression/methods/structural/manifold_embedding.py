from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

METHOD_NAME = "manifold_embedding"

__all__ = ["ManifoldConfig", "ManifoldEmbedding", "METHOD_NAME"]


@dataclass
class ManifoldConfig:
    embedding_dim: int = 8
    n_neighbors: int = 16
    n_iter: int = 30
    learning_rate: float = 0.1


class ManifoldEmbedding:
    METHOD_NAME = METHOD_NAME

    def __init__(self, config: Optional[ManifoldConfig] = None):
        self.config = config or ManifoldConfig()

    def _isomap_embedding(
        self, data: np.ndarray, k: int, d: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        n = len(data)
        dists = np.linalg.norm(data[:, np.newaxis] - data[np.newaxis, :], axis=2)

        knn_graph = np.full((n, n), np.inf)
        for i in range(n):
            nearest = np.argsort(dists[i])[: k + 1]
            knn_graph[i, nearest] = dists[i, nearest]

        for k_iter in range(10):
            for i in range(n):
                for j in range(n):
                    for l in range(n):
                        new_dist = knn_graph[i, l] + knn_graph[l, j]
                        if new_dist < knn_graph[i, j]:
                            knn_graph[i, j] = new_dist

        knn_graph = np.minimum(knn_graph, knn_graph.T)
        knn_graph[np.isinf(knn_graph)] = 0

        H = np.eye(n) - np.ones((n, n)) / n
        gram = -0.5 * H @ (knn_graph**2) @ H

        eigenvalues, eigenvectors = np.linalg.eigh(gram)
        idx = np.argsort(eigenvalues)[::-1][:d]
        embedding = eigenvectors[:, idx] * np.sqrt(np.maximum(eigenvalues[idx], 0))

        return embedding, knn_graph

    def compress(
        self, tensor: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        embedding_dim = kwargs.get("embedding_dim", self.config.embedding_dim)
        n_neighbors = kwargs.get("n_neighbors", self.config.n_neighbors)
        orig_shape = tensor.shape

        mat = tensor.reshape(-1, tensor.shape[-1]).astype(np.float64)
        n_rows = min(mat.shape[0], 256)
        subset = mat[:n_rows]

        embedding, dist_matrix = self._isomap_embedding(
            subset, n_neighbors, embedding_dim
        )

        recon_map = np.linalg.lstsq(embedding, subset, rcond=None)[0]

        data_out = {
            "embedding": embedding.astype(np.float32),
            "recon_map": recon_map.astype(np.float32),
        }
        meta = {"orig_shape": orig_shape, "n_rows": n_rows, "method": METHOD_NAME}
        return data_out, meta

    def decompress(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> np.ndarray:
        embedding = data["embedding"].astype(np.float64)
        recon_map = data["recon_map"].astype(np.float64)
        reconstructed = embedding @ recon_map.T
        n_rows = metadata["n_rows"]
        return (
            reconstructed[:n_rows]
            .reshape(metadata["orig_shape"][:1] + metadata["orig_shape"][1:])
            .astype(np.float32)
        )

    def estimate_ratio(self, tensor: np.ndarray, **kwargs) -> float:
        embedding_dim = kwargs.get("embedding_dim", self.config.embedding_dim)
        n_rows = min(tensor.shape[0], 256)
        orig = tensor.nbytes
        comp = n_rows * embedding_dim * 4 + embedding_dim * tensor.shape[-1] * 4
        return comp / max(orig, 1)
