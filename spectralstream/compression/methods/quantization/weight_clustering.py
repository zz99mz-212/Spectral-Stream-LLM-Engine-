"""K-means weight clustering with 8-bit indices and learned centroids."""

from __future__ import annotations

from typing import Tuple

import numpy as np


class WeightClusteringArchive:
    """K-means clustering with 8-bit indices and learned centroids via percentile init.

    For K=256: 256*4 bytes codebook + N*1 byte indices = N + 1KB.
    Ratio ≈ 4x for large N.
    """

    name = "weight_clustering_archive"
    category = "quantization"

    def compress(
        self, tensor: np.ndarray, n_clusters: int = 256, n_iter: int = 30
    ) -> Tuple[bytes, dict]:
        flat = tensor.ravel().astype(np.float64)
        n = len(flat)
        n_clusters = min(n_clusters, n)

        quantiles = np.linspace(0, 100, n_clusters + 2)[1:-1]
        centroids = np.percentile(flat, quantiles).astype(np.float64)

        for _ in range(n_iter):
            boundaries = (centroids[1:] + centroids[:-1]) / 2.0
            indices = np.clip(np.digitize(flat, boundaries), 0, n_clusters - 1)
            new_centroids = np.array(
                [
                    flat[indices == i].mean() if np.any(indices == i) else centroids[i]
                    for i in range(n_clusters)
                ]
            )
            if np.allclose(centroids, new_centroids, atol=1e-6):
                break
            centroids = new_centroids

        boundaries = (centroids[1:] + centroids[:-1]) / 2.0
        indices = np.clip(np.digitize(flat, boundaries), 0, n_clusters - 1)

        codebook_bytes = centroids.astype(np.float32).tobytes()
        indices_bytes = indices.astype(np.uint8).tobytes()

        metadata = dict(
            n_elements=n,
            n_clusters=n_clusters,
            codebook=codebook_bytes,
            shape=tensor.shape,
        )
        return indices_bytes, metadata

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        n = metadata["n_elements"]
        centroids = np.frombuffer(metadata["codebook"], dtype=np.float32).astype(
            np.float64
        )
        indices = np.frombuffer(data, dtype=np.uint8).astype(np.int32)
        return centroids[indices[:n]].reshape(metadata["shape"]).astype(np.float32)
