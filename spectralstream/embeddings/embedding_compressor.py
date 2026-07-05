"""
Embedding Matrix Compressor for SpectralStream
================================================
Compresses embedding weight matrices using Product Quantization,
Clustered PQ with k-means, and Low-rank + residual decomposition.

Embeddings are typically the largest single parameter group in a model
(vocab_size x embed_dim), making them prime targets for compression.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from spectralstream.core.math_primitives import (
    LloydMaxQuantizer,
    dct,
    idct,
    cosine_similarity,
    unit_vector,
    BAND_COMPRESSION,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# K-Means Clustering
# ═══════════════════════════════════════════════════════════════════════════


def _kmeans(
    data: np.ndarray,
    n_clusters: int,
    max_iter: int = 20,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Simple k-means clustering.

    Args:
        data: Input data of shape (n_samples, n_features).
        n_clusters: Number of clusters.
        max_iter: Maximum iterations.
        seed: Random seed.

    Returns:
        Tuple of (centroids, assignments, inertia).
    """
    n_samples, n_features = data.shape
    rng = np.random.RandomState(seed)

    centroids = np.empty((n_clusters, n_features), dtype=np.float64)
    idx = rng.randint(n_samples)
    centroids[0] = data[idx]

    for c in range(1, n_clusters):
        dists = np.min(
            np.sum((data[:, np.newaxis] - centroids[:c]) ** 2, axis=2),
            axis=1,
        )
        probs = dists / (np.sum(dists) + 1e-10)
        idx = rng.choice(n_samples, p=probs)
        centroids[c] = data[idx]

    assignments = np.zeros(n_samples, dtype=np.int32)

    for _ in range(max_iter):
        dists = np.sum(
            (data[:, np.newaxis] - centroids) ** 2,
            axis=2,
        )
        new_assignments = np.argmin(dists, axis=1).astype(np.int32)

        if np.array_equal(assignments, new_assignments):
            break
        assignments = new_assignments

        for k in range(n_clusters):
            mask = assignments == k
            if np.any(mask):
                centroids[k] = np.mean(data[mask], axis=0)

    inertia = float(
        np.sum(
            np.sum((data - centroids[assignments]) ** 2, axis=1),
        )
    )

    return centroids, assignments, inertia


# ═══════════════════════════════════════════════════════════════════════════
# Compression Result
# ═══════════════════════════════════════════════════════════════════════════


class EmbeddingCompressionResult:
    """Container for embedding compression output."""

    __slots__ = (
        "compressed_data",
        "original_shape",
        "method",
        "compression_ratio",
        "reconstruction_error",
        "metadata",
    )

    def __init__(
        self,
        compressed_data: Any,
        original_shape: Tuple[int, ...],
        method: str,
        compression_ratio: float,
        reconstruction_error: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.compressed_data = compressed_data
        self.original_shape = original_shape
        self.method = method
        self.compression_ratio = compression_ratio
        self.reconstruction_error = reconstruction_error
        self.metadata = metadata or {}

    def __repr__(self) -> str:
        return (
            f"EmbeddingCompressionResult(method={self.method!r}, "
            f"ratio={self.compression_ratio:.2f}x, "
            f"error={self.reconstruction_error:.6f})"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Embedding Compressor
# ═══════════════════════════════════════════════════════════════════════════


class EmbeddingCompressor:
    """Compresses embedding matrices using multiple strategies.

    Strategies:
        1. Product Quantization (PQ): split embeddings into sub-vectors,
           quantize each subspace independently.
        2. Clustered PQ: group similar embeddings via k-means, then
           quantize within each cluster with higher precision.
        3. Low-rank + Residual: truncate SVD, quantize the residual.

    Usage:
        compressor = EmbeddingCompressor()
        result = compressor.compress(embedding_matrix, method='pq')
        decompressed = compressor.decompress(result)
    """

    def __init__(
        self,
        n_subspaces: int = 8,
        n_centroids: int = 256,
        n_clusters: int = 16,
        low_rank_fraction: float = 0.5,
        residual_bits: int = 4,
        seed: int = 42,
    ) -> None:
        self.n_subspaces = n_subspaces
        self.n_centroids = n_centroids
        self.n_clusters = n_clusters
        self.low_rank_fraction = low_rank_fraction
        self.residual_bits = residual_bits
        self.seed = seed

        self._quantizers: Dict[int, LloydMaxQuantizer] = {}

        logger.info(
            "EmbeddingCompressor initialized: subspaces=%d, centroids=%d, "
            "clusters=%d, low_rank_frac=%.2f",
            n_subspaces,
            n_centroids,
            n_clusters,
            low_rank_fraction,
        )

    # ── Public API ────────────────────────────────────────────────────────

    def compress(
        self,
        embedding_matrix: np.ndarray,
        method: str = "pq",
        **kwargs: Any,
    ) -> EmbeddingCompressionResult:
        """Compress an embedding matrix.

        Args:
            embedding_matrix: Weight matrix of shape (vocab_size, embed_dim).
            method: Compression method ('pq', 'clustered_pq', 'low_rank_residual').
            **kwargs: Additional method-specific parameters.

        Returns:
            EmbeddingCompressionResult with compressed data.
        """
        embedding_matrix = np.asarray(embedding_matrix, dtype=np.float64)
        start = time.perf_counter()

        if method == "pq":
            result = self._compress_pq(embedding_matrix, **kwargs)
        elif method == "clustered_pq":
            result = self._compress_clustered_pq(embedding_matrix, **kwargs)
        elif method == "low_rank_residual":
            result = self._compress_low_rank_residual(embedding_matrix, **kwargs)
        else:
            raise ValueError(f"Unknown compression method: {method!r}")

        elapsed = time.perf_counter() - start
        result.metadata["compress_time_ms"] = elapsed * 1000.0

        logger.info(
            "Compressed embeddings: method=%s, shape=%s, ratio=%.2fx, error=%.6f, time=%.2fms",
            method,
            embedding_matrix.shape,
            result.compression_ratio,
            result.reconstruction_error,
            elapsed * 1000,
        )

        return result

    def decompress(self, result: EmbeddingCompressionResult) -> np.ndarray:
        """Decompress an embedding matrix.

        Args:
            result: EmbeddingCompressionResult from compress().

        Returns:
            Reconstructed embedding matrix.
        """
        method = result.method
        data = result.compressed_data

        if method == "pq":
            return self._decompress_pq(data, result.original_shape)
        elif method == "clustered_pq":
            return self._decompress_clustered_pq(data, result.original_shape)
        elif method == "low_rank_residual":
            return self._decompress_low_rank_residual(data, result.original_shape)
        else:
            raise ValueError(f"Unknown method: {method!r}")

    # ── Product Quantization ──────────────────────────────────────────────

    def _compress_pq(
        self,
        matrix: np.ndarray,
        **kwargs: Any,
    ) -> EmbeddingCompressionResult:
        """Compress via Product Quantization.

        Splits each embedding vector into `n_subspaces` sub-vectors,
        quantizes each subspace independently with a shared codebook.
        """
        vocab_size, embed_dim = matrix.shape
        n_sub = kwargs.get("n_subspaces", self.n_subspaces)
        n_cent = kwargs.get("n_centroids", self.n_centroids)

        padded_dim = ((embed_dim + n_sub - 1) // n_sub) * n_sub
        if padded_dim != embed_dim:
            matrix_padded = np.pad(matrix, ((0, 0), (0, padded_dim - embed_dim)))
        else:
            matrix_padded = matrix

        sub_dim = padded_dim // n_sub
        n_vectors = vocab_size

        codebook = np.zeros((n_sub, n_cent, sub_dim), dtype=np.float64)
        codes = np.zeros((n_vectors, n_sub), dtype=np.int32)

        for s in range(n_sub):
            sub_vectors = matrix_padded[:, s * sub_dim : (s + 1) * sub_dim]
            kmeans_centroids, assignments, _ = _kmeans(
                sub_vectors,
                min(n_cent, n_vectors),
                max_iter=15,
                seed=self.seed + s,
            )
            codebook[s, : len(kmeans_centroids)] = kmeans_centroids
            codes[:, s] = assignments

        reconstructed = np.zeros_like(matrix_padded)
        for s in range(n_sub):
            reconstructed[:, s * sub_dim : (s + 1) * sub_dim] = codebook[s, codes[:, s]]

        if padded_dim != embed_dim:
            reconstructed = reconstructed[:, :embed_dim]

        mse = float(np.mean((matrix - reconstructed) ** 2))

        original_bytes = matrix.nbytes
        compressed_bytes = codes.nbytes + codebook.nbytes
        ratio = original_bytes / max(compressed_bytes, 1)

        return EmbeddingCompressionResult(
            compressed_data={
                "codes": codes,
                "codebook": codebook,
                "n_subspaces": n_sub,
                "sub_dim": sub_dim,
                "embed_dim": embed_dim,
            },
            original_shape=matrix.shape,
            method="pq",
            compression_ratio=ratio,
            reconstruction_error=mse,
            metadata={"n_subspaces": n_sub, "n_centroids": n_cent},
        )

    def _decompress_pq(
        self,
        data: Dict[str, Any],
        shape: Tuple[int, ...],
    ) -> np.ndarray:
        """Reconstruct from PQ representation."""
        codes = data["codes"]
        codebook = data["codebook"]
        n_sub = data["n_subspaces"]
        sub_dim = data["sub_dim"]
        embed_dim = data["embed_dim"]
        vocab_size = codes.shape[0]

        reconstructed = np.zeros((vocab_size, n_sub * sub_dim), dtype=np.float64)
        for s in range(n_sub):
            reconstructed[:, s * sub_dim : (s + 1) * sub_dim] = codebook[s, codes[:, s]]

        return reconstructed[:, :embed_dim]

    # ── Clustered Product Quantization ────────────────────────────────────

    def _compress_clustered_pq(
        self,
        matrix: np.ndarray,
        **kwargs: Any,
    ) -> EmbeddingCompressionResult:
        """Compress via Clustered Product Quantization.

        Steps:
            1. Cluster embeddings into `n_clusters` groups via k-means.
            2. Within each cluster, apply PQ with a cluster-specific codebook.
            3. Store cluster assignments + per-cluster codes + codebooks.
        """
        vocab_size, embed_dim = matrix.shape
        n_clus = min(kwargs.get("n_clusters", self.n_clusters), vocab_size)
        n_sub = kwargs.get("n_subspaces", self.n_subspaces)
        n_cent = kwargs.get("n_centroids", self.n_centroids)

        cluster_centroids, cluster_assignments, _ = _kmeans(
            matrix,
            n_clus,
            max_iter=20,
            seed=self.seed,
        )

        padded_dim = ((embed_dim + n_sub - 1) // n_sub) * n_sub
        sub_dim = padded_dim // n_sub

        cluster_codebooks: List[np.ndarray] = []
        cluster_codes: List[np.ndarray] = []
        cluster_sizes: List[int] = []

        for c in range(n_clus):
            mask = cluster_assignments == c
            cluster_size = int(np.sum(mask))
            cluster_sizes.append(cluster_size)

            if cluster_size == 0:
                cluster_codebooks.append(np.zeros((n_sub, n_cent, sub_dim)))
                cluster_codes.append(np.zeros(0, dtype=np.int32))
                continue

            cluster_data = matrix[mask]
            padded = (
                np.pad(
                    cluster_data,
                    ((0, 0), (0, padded_dim - embed_dim)),
                )
                if padded_dim != embed_dim
                else cluster_data
            )

            codebook = np.zeros((n_sub, n_cent, sub_dim), dtype=np.float64)
            codes = np.zeros((cluster_size, n_sub), dtype=np.int32)

            for s in range(n_sub):
                sub_vecs = padded[:, s * sub_dim : (s + 1) * sub_dim]
                cent, assigns, _ = _kmeans(
                    sub_vecs,
                    min(n_cent, cluster_size),
                    max_iter=10,
                    seed=self.seed + c * 100 + s,
                )
                codebook[s, : len(cent)] = cent
                codes[:, s] = assigns

            cluster_codebooks.append(codebook)
            cluster_codes.append(codes)

        reconstructed = np.zeros_like(matrix)
        for c in range(n_clus):
            mask = cluster_assignments == c
            cluster_size = int(np.sum(mask))
            if cluster_size == 0:
                continue

            padded_cluster = np.zeros((cluster_size, padded_dim), dtype=np.float64)
            for s in range(n_sub):
                padded_cluster[:, s * sub_dim : (s + 1) * sub_dim] = cluster_codebooks[
                    c
                ][s, cluster_codes[c][:, s]]
            reconstructed[mask] = padded_cluster[:, :embed_dim]

        mse = float(np.mean((matrix - reconstructed) ** 2))

        original_bytes = matrix.nbytes
        compressed_bytes = (
            cluster_assignments.nbytes
            + sum(cb.nbytes for cb in cluster_codebooks)
            + sum(c.nbytes for c in cluster_codes)
        )
        ratio = original_bytes / max(compressed_bytes, 1)

        return EmbeddingCompressionResult(
            compressed_data={
                "cluster_centroids": cluster_centroids,
                "cluster_assignments": cluster_assignments,
                "cluster_codebooks": cluster_codebooks,
                "cluster_codes": cluster_codes,
                "cluster_sizes": cluster_sizes,
                "n_clusters": n_clus,
                "n_subspaces": n_sub,
                "sub_dim": sub_dim,
                "embed_dim": embed_dim,
            },
            original_shape=matrix.shape,
            method="clustered_pq",
            compression_ratio=ratio,
            reconstruction_error=mse,
            metadata={
                "n_clusters": n_clus,
                "n_subspaces": n_sub,
                "n_centroids": n_cent,
            },
        )

    def _decompress_clustered_pq(
        self,
        data: Dict[str, Any],
        shape: Tuple[int, ...],
    ) -> np.ndarray:
        """Reconstruct from clustered PQ representation."""
        cluster_assignments = data["cluster_assignments"]
        cluster_codebooks = data["cluster_codebooks"]
        cluster_codes = data["cluster_codes"]
        n_clus = data["n_clusters"]
        n_sub = data["n_subspaces"]
        sub_dim = data["sub_dim"]
        embed_dim = data["embed_dim"]
        vocab_size = len(cluster_assignments)

        reconstructed = np.zeros(
            (vocab_size, n_sub * sub_dim),
            dtype=np.float64,
        )
        for c in range(n_clus):
            mask = cluster_assignments == c
            if not np.any(mask):
                continue
            cluster_size = int(np.sum(mask))
            for s in range(n_sub):
                reconstructed[mask, s * sub_dim : (s + 1) * sub_dim] = (
                    cluster_codebooks[c][s, cluster_codes[c][:, s]]
                )

        return reconstructed[:, :embed_dim]

    # ── Low-Rank + Residual ───────────────────────────────────────────────

    def _compress_low_rank_residual(
        self,
        matrix: np.ndarray,
        **kwargs: Any,
    ) -> EmbeddingCompressionResult:
        """Compress via low-rank SVD + quantized residual.

        Steps:
            1. Compute truncated SVD: W ≈ U @ diag(s) @ Vt (rank r).
            2. Compute residual: R = W - U @ diag(s) @ Vt.
            3. Quantize residual with Lloyd-Max quantizer.
        """
        vocab_size, embed_dim = matrix.shape
        rank_frac = kwargs.get("low_rank_fraction", self.low_rank_fraction)
        res_bits = kwargs.get("residual_bits", self.residual_bits)

        U, s, Vt = np.linalg.svd(matrix, full_matrices=False)
        total_energy = float(np.sum(s**2))
        if total_energy > 1e-10:
            cum = np.cumsum(s**2) / total_energy
            rank = int(np.searchsorted(cum, rank_frac)) + 1
            rank = max(1, min(rank, len(s)))
        else:
            rank = 1

        U_r = U[:, :rank]
        s_r = s[:rank]
        Vt_r = Vt[:rank, :]

        low_rank_approx = U_r @ np.diag(s_r) @ Vt_r

        residual = matrix - low_rank_approx

        quantizer = self._get_quantizer(res_bits)
        flat_residual = residual.ravel()
        if not quantizer.trained:
            quantizer.train(flat_residual)
        res_indices, res_centroids = quantizer.compress(flat_residual)

        dequant_residual = res_centroids[res_indices].reshape(matrix.shape)
        reconstructed = low_rank_approx + dequant_residual
        mse = float(np.mean((matrix - reconstructed) ** 2))

        original_bytes = matrix.nbytes
        compressed_bytes = (
            U_r.nbytes
            + s_r.nbytes
            + Vt_r.nbytes
            + res_indices.nbytes
            + res_centroids.nbytes
        )
        ratio = original_bytes / max(compressed_bytes, 1)

        return EmbeddingCompressionResult(
            compressed_data={
                "U": U_r,
                "s": s_r,
                "Vt": Vt_r,
                "rank": rank,
                "residual_indices": res_indices,
                "residual_centroids": res_centroids,
                "residual_bits": res_bits,
            },
            original_shape=matrix.shape,
            method="low_rank_residual",
            compression_ratio=ratio,
            reconstruction_error=mse,
            metadata={"rank": rank, "residual_bits": res_bits},
        )

    def _decompress_low_rank_residual(
        self,
        data: Dict[str, Any],
        shape: Tuple[int, ...],
    ) -> np.ndarray:
        """Reconstruct from low-rank + residual representation."""
        U_r = data["U"]
        s_r = data["s"]
        Vt_r = data["Vt"]
        res_indices = data["residual_indices"]
        res_centroids = data["residual_centroids"]

        low_rank = U_r @ np.diag(s_r) @ Vt_r
        residual = res_centroids[res_indices].reshape(shape)
        return low_rank + residual

    # ── Helpers ───────────────────────────────────────────────────────────

    def _get_quantizer(self, n_bits: int) -> LloydMaxQuantizer:
        """Get or create quantizer for given bit width."""
        if n_bits not in self._quantizers:
            self._quantizers[n_bits] = LloydMaxQuantizer(n_bits=n_bits)
        return self._quantizers[n_bits]
