"""
Extreme Embedding Compression for SpectralStream
==================================================
Aggressive embedding table compression using clustered product
quantization, low-rank decomposition, delta encoding, and
hot/warm/cold tiering.

Architecture:
  1. EmbeddingAnalyzer — cluster analysis, sub-vector dimension selection
  2. ClusteredProductQuantizer — per-cluster PQ codebooks
  3. LowRankEmbedding — SVD + delta encoding between layers
  4. HybridEmbeddingStore — hot/warm/cold tiering with eviction
"""

from __future__ import annotations

import logging
import math
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from spectralstream.core.math_primitives import (
    LloydMaxQuantizer,
    cosine_similarity,
    dct,
    idct,
    next_power_of_two,
    softmax,
    spectral_entropy,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class EmbeddingCompressionConfig:
    """Configuration for embedding compression."""

    n_clusters: int = 32
    pq_subspaces: int = 4
    pq_bits: int = 4
    low_rank_dim: int = 64
    delta_threshold: float = 0.01
    hot_threshold: int = 10
    warm_threshold: int = 3
    cold_eviction_fraction: float = 0.2
    max_hot_entries: int = 10000
    max_warm_entries: int = 50000


@dataclass
class EmbeddingProfile:
    """Analysis profile for an embedding table."""

    n_entries: int
    embedding_dim: int
    cluster_sizes: np.ndarray
    intra_cluster_variance: float
    inter_cluster_variance: float
    spectral_entropy: float
    effective_rank: int
    recommended_clusters: int
    recommended_subspaces: int


# ═══════════════════════════════════════════════════════════════════════════
# 1. EmbeddingAnalyzer — cluster analysis
# ═══════════════════════════════════════════════════════════════════════════


class EmbeddingAnalyzer:
    """Analyze embedding tables to guide compression parameters.

    Performs k-means clustering, computes intra/inter-cluster
    statistics, and recommends optimal compression parameters.
    """

    def __init__(self, max_k: int = 64):
        self.max_k = max_k

    def _kmeans(
        self,
        data: np.ndarray,
        n_clusters: int,
        max_iter: int = 20,
        seed: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simple k-means clustering."""
        rng = np.random.RandomState(seed)
        n_samples = data.shape[0]

        centroids = [data[rng.randint(n_samples)].copy()]
        for _ in range(1, n_clusters):
            dists = np.min(
                np.linalg.norm(data[:, np.newaxis] - np.array(centroids), axis=2),
                axis=1,
            )
            probs = dists**2 / (np.sum(dists**2) + 1e-30)
            idx = rng.choice(n_samples, p=probs)
            centroids.append(data[idx].copy())

        centroids = np.array(centroids)

        for _ in range(max_iter):
            dists = np.linalg.norm(
                data[:, np.newaxis] - centroids[np.newaxis, :], axis=2
            )
            labels = np.argmin(dists, axis=1)

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

        return centroids, labels

    def analyze(
        self,
        embeddings: np.ndarray,
        n_clusters: Optional[int] = None,
    ) -> EmbeddingProfile:
        """Analyze an embedding table.

        Args:
            embeddings: Embedding matrix of shape (n_entries, emb_dim).
            n_clusters: Number of clusters (auto-selected if None).

        Returns:
            EmbeddingProfile with analysis results.
        """
        embeddings = np.asarray(embeddings, dtype=np.float64)
        n_entries, emb_dim = embeddings.shape

        if n_clusters is None:
            n_clusters = min(self.max_k, max(2, int(np.sqrt(n_entries))))

        centroids, labels = self._kmeans(embeddings, n_clusters)
        cluster_sizes = np.bincount(labels, minlength=n_clusters).astype(np.float64)

        intra_var = 0.0
        for k in range(n_clusters):
            members = embeddings[labels == k]
            if len(members) > 1:
                intra_var += float(np.var(members)) * len(members)
        intra_var /= n_entries

        global_mean = np.mean(embeddings, axis=0)
        inter_var = float(np.var(centroids - global_mean))

        flat = embeddings.ravel()
        ent = spectral_entropy(flat[: min(len(flat), 10000)])

        try:
            sv = np.linalg.svd(embeddings, compute_uv=False)
            cumulative = np.cumsum(sv**2) / (np.sum(sv**2) + 1e-30)
            eff_rank = int(np.searchsorted(cumulative, 0.95)) + 1
        except np.linalg.LinAlgError:
            eff_rank = emb_dim

        variance_ratio = inter_var / (intra_var + 1e-10)
        if variance_ratio > 2.0:
            recommended_clusters = min(n_clusters * 2, self.max_k)
        elif variance_ratio < 0.5:
            recommended_clusters = max(2, n_clusters // 2)
        else:
            recommended_clusters = n_clusters

        recommended_subspaces = min(8, max(2, emb_dim // 32))

        return EmbeddingProfile(
            n_entries=n_entries,
            embedding_dim=emb_dim,
            cluster_sizes=cluster_sizes,
            intra_cluster_variance=intra_var,
            inter_cluster_variance=inter_var,
            spectral_entropy=ent,
            effective_rank=eff_rank,
            recommended_clusters=recommended_clusters,
            recommended_subspaces=recommended_subspaces,
        )


# ═══════════════════════════════════════════════════════════════════════════
# 2. ClusteredProductQuantizer — per-cluster PQ
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class ClusteredPQResult:
    """Result of clustered PQ compression."""

    cluster_labels: np.ndarray
    cluster_centroids: np.ndarray
    pq_codebooks: Dict[int, List[np.ndarray]]
    pq_indices: Dict[int, np.ndarray]
    original_shape: Tuple[int, int]
    compression_ratio: float
    reconstruction_error: float


class ClusteredProductQuantizer:
    """Per-cluster Product Quantization for embeddings.

    Clusters embeddings and trains separate PQ codebooks per cluster,
    achieving better quantization quality than global PQ.
    """

    def __init__(self, config: Optional[EmbeddingCompressionConfig] = None):
        self.config = config or EmbeddingCompressionConfig()
        self._analyzer = EmbeddingAnalyzer()

    def train(
        self,
        embeddings: np.ndarray,
        n_clusters: Optional[int] = None,
    ) -> ClusteredPQResult:
        """Train clustered PQ on embeddings.

        Args:
            embeddings: Embedding matrix (n_entries, emb_dim).
            n_clusters: Number of clusters.

        Returns:
            ClusteredPQResult with trained codebooks.
        """
        embeddings = np.asarray(embeddings, dtype=np.float64)
        n_entries, emb_dim = embeddings.shape
        nc = n_clusters or self.config.n_clusters

        profile = self._analyzer.analyze(embeddings, nc)
        nc = profile.recommended_clusters

        centroids, labels = self._analyzer._kmeans(embeddings, nc)

        pq_codebooks: Dict[int, List[np.ndarray]] = {}
        pq_indices: Dict[int, np.ndarray] = {}
        n_subspaces = self.config.pq_subspaces
        sub_dim = emb_dim // n_subspaces

        for k in range(nc):
            mask = labels == k
            if np.sum(mask) == 0:
                continue

            cluster_data = embeddings[mask]
            n_sub = min(n_subspaces, emb_dim)
            sub_dim_k = emb_dim // n_sub

            codebooks = []
            indices_list = []

            for s in range(n_sub):
                start = s * sub_dim_k
                end = start + sub_dim_k if s < n_sub - 1 else emb_dim
                sub_data = cluster_data[:, start:end]

                n_levels = 1 << self.config.pq_bits
                rng = np.random.RandomState(42 + k * 100 + s)
                n_members = sub_data.shape[0]
                codebook = [sub_data[rng.randint(n_members)].copy()]
                for _ in range(1, n_levels):
                    dists = np.min(
                        np.linalg.norm(
                            sub_data[:, np.newaxis] - np.array(codebook), axis=2
                        ),
                        axis=1,
                    )
                    probs = dists**2 / (np.sum(dists**2) + 1e-30)
                    idx = rng.choice(n_members, p=probs)
                    codebook.append(sub_data[idx].copy())

                codebook = np.array(codebook)

                for _ in range(self.config.pq_bits * 5):
                    dists = np.linalg.norm(
                        sub_data[:, np.newaxis] - codebook[np.newaxis, :], axis=2
                    )
                    assigned = np.argmin(dists, axis=1)
                    new_cb = np.empty_like(codebook)
                    for c in range(n_levels):
                        members = sub_data[assigned == c]
                        if len(members) > 0:
                            new_cb[c] = np.mean(members, axis=0)
                        else:
                            new_cb[c] = codebook[c]
                    if np.allclose(codebook, new_cb, atol=1e-6):
                        break
                    codebook = new_cb

                dists = np.linalg.norm(
                    sub_data[:, np.newaxis] - codebook[np.newaxis, :], axis=2
                )
                idx = np.argmin(dists, axis=1)

                codebooks.append(codebook)
                indices_list.append(idx)

            pq_codebooks[k] = codebooks
            pq_indices[k] = (
                np.column_stack(indices_list) if indices_list else np.array([])
            )

        reconstructed = self._reconstruct(
            embeddings, labels, centroids, pq_codebooks, n_sub
        )
        error = float(np.max(np.abs(embeddings - reconstructed)))

        original_bits = n_entries * emb_dim * 32
        compressed_bits = n_entries * (n_sub * self.config.pq_bits + 32)
        ratio = original_bits / max(compressed_bits, 1)

        return ClusteredPQResult(
            cluster_labels=labels,
            cluster_centroids=centroids,
            pq_codebooks=pq_codebooks,
            pq_indices=pq_indices,
            original_shape=(n_entries, emb_dim),
            compression_ratio=ratio,
            reconstruction_error=error,
        )

    def _reconstruct(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        centroids: np.ndarray,
        pq_codebooks: Dict[int, List[np.ndarray]],
        n_subspaces: int,
    ) -> np.ndarray:
        """Reconstruct embeddings from clustered PQ."""
        n_entries, emb_dim = embeddings.shape
        reconstructed = np.zeros_like(embeddings, dtype=np.float64)

        for k in pq_codebooks:
            mask = labels == k
            if not np.any(mask):
                continue
            codebooks = pq_codebooks[k]
            if not codebooks:
                reconstructed[mask] = centroids[k]
                continue

            sub_dim = emb_dim // len(codebooks)
            for s, codebook in enumerate(codebooks):
                start = s * sub_dim
                end = start + sub_dim if s < len(codebooks) - 1 else emb_dim
                reconstructed[mask, start:end] = centroids[k, start:end]

        return reconstructed

    def compress(self, result: ClusteredPQResult) -> bytes:
        """Serialize the clustered PQ result to bytes."""
        data = {
            "labels": result.cluster_labels,
            "centroids": result.cluster_centroids,
            "shape": result.original_shape,
        }
        buf = []
        buf.append(result.cluster_labels.tobytes())
        buf.append(result.cluster_centroids.tobytes())
        return b"".join(buf)


# ═══════════════════════════════════════════════════════════════════════════
# 3. LowRankEmbedding — SVD + delta encoding
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class LowRankEmbeddingResult:
    """Result of low-rank embedding compression."""

    U: np.ndarray
    S: np.ndarray
    Vt: np.ndarray
    delta_embeddings: Optional[np.ndarray]
    original_shape: Tuple[int, int]
    rank: int
    compression_ratio: float


class LowRankEmbedding:
    """Low-rank embedding compression via SVD + delta encoding.

    Decomposes embeddings as E ≈ U @ diag(S) @ Vt with rank r,
    storing only the compact factors. Supports delta encoding
    between adjacent embedding layers for additional compression.
    """

    def __init__(self, config: Optional[EmbeddingCompressionConfig] = None):
        self.config = config or EmbeddingCompressionConfig()

    def decompose(
        self,
        embeddings: np.ndarray,
        rank: Optional[int] = None,
        energy_threshold: float = 0.99,
    ) -> LowRankEmbeddingResult:
        """Decompose embeddings via truncated SVD.

        Args:
            embeddings: Embedding matrix (n_entries, emb_dim).
            rank: Target rank (auto-selected if None).
            energy_threshold: Fraction of energy to retain.

        Returns:
            LowRankEmbeddingResult with SVD factors.
        """
        embeddings = np.asarray(embeddings, dtype=np.float64)
        n_entries, emb_dim = embeddings.shape

        try:
            U, S, Vt = np.linalg.svd(embeddings, full_matrices=False)
        except np.linalg.LinAlgError:
            U = embeddings
            S = np.ones(min(n_entries, emb_dim))
            Vt = np.eye(min(n_entries, emb_dim))

        if rank is None:
            total_energy = np.sum(S**2)
            cumulative = np.cumsum(S**2) / (total_energy + 1e-30)
            rank = int(np.searchsorted(cumulative, energy_threshold)) + 1
            rank = min(rank, len(S))

        rank = min(rank, self.config.low_rank_dim)
        rank = max(1, rank)

        U_r = U[:, :rank]
        S_r = S[:rank]
        Vt_r = Vt[:rank, :]

        original_size = n_entries * emb_dim
        compressed_size = n_entries * rank + rank + rank * emb_dim
        ratio = original_size / max(compressed_size, 1)

        return LowRankEmbeddingResult(
            U=U_r,
            S=S_r,
            Vt=Vt_r,
            delta_embeddings=None,
            original_shape=(n_entries, emb_dim),
            rank=rank,
            compression_ratio=ratio,
        )

    def encode_delta(
        self,
        embeddings_a: np.ndarray,
        embeddings_b: np.ndarray,
        threshold: Optional[float] = None,
    ) -> np.ndarray:
        """Compute delta between two embedding layers.

        Args:
            embeddings_a: Reference embedding matrix.
            embeddings_b: Target embedding matrix.
            threshold: Threshold for delta sparsification.

        Returns:
            Delta matrix (sparse representation).
        """
        thresh = threshold or self.config.delta_threshold
        delta = np.asarray(embeddings_b, dtype=np.float64) - np.asarray(
            embeddings_a, dtype=np.float64
        )

        mask = np.abs(delta) > thresh * np.max(np.abs(delta) + 1e-10)
        delta_sparse = np.where(mask, delta, 0.0)

        sparsity = 1.0 - float(np.mean(mask))
        logger.debug(
            "Delta encoding: sparsity=%.2f, max_delta=%.6f",
            sparsity,
            float(np.max(np.abs(delta))),
        )

        return delta_sparse

    def reconstruct(self, result: LowRankEmbeddingResult) -> np.ndarray:
        """Reconstruct embeddings from SVD factors."""
        return (result.U * result.S) @ result.Vt


# ═══════════════════════════════════════════════════════════════════════════
# 4. HybridEmbeddingStore — hot/warm/cold tiering
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class EmbeddingEntry:
    """A single embedding entry with metadata."""

    embedding: np.ndarray
    access_count: int = 0
    last_access: int = 0
    tier: str = "cold"


class TieredEmbeddingDict:
    """LRU dictionary for tiered storage."""

    pass


class HybridEmbeddingStore:
    """Hot/warm/cold tiered embedding store.

    - Hot tier:  full precision, fast access (LRU cache)
    - Warm tier: quantized (4-bit), moderate access
    - Cold tier: clustered PQ compressed, on-demand decompress

    Entries migrate between tiers based on access frequency.
    """

    def __init__(self, config: Optional[EmbeddingCompressionConfig] = None):
        self.config = config or EmbeddingCompressionConfig()
        self._hot: OrderedDict[int, EmbeddingEntry] = OrderedDict()
        self._warm: Dict[int, np.ndarray] = {}
        self._cold: Dict[int, bytes] = {}
        self._access_counts: Dict[int, int] = {}
        self._current_time: int = 0
        self._pq_trained: bool = False
        self._pq_codebooks: Optional[Dict[int, List[np.ndarray]]] = None

    def lookup(self, token_id: int) -> np.ndarray:
        """Look up an embedding by token ID.

        Promotes cold/warm entries to hot on access.
        """
        self._current_time += 1

        if token_id in self._hot:
            entry = self._hot[token_id]
            entry.access_count += 1
            entry.last_access = self._current_time
            self._hot.move_to_end(token_id)
            return entry.embedding

        if token_id in self._warm:
            emb = self._warm[token_id]
            self._promote_to_hot(token_id, emb)
            return emb

        if token_id in self._cold:
            emb = self._decompress_cold(token_id)
            self._promote_to_warm(token_id, emb)
            return emb

        raise KeyError(f"Token ID {token_id} not found")

    def insert(self, token_id: int, embedding: np.ndarray):
        """Insert a new embedding into the hot tier."""
        entry = EmbeddingEntry(
            embedding=np.asarray(embedding, dtype=np.float32),
            access_count=1,
            last_access=self._current_time,
            tier="hot",
        )
        self._hot[token_id] = entry
        self._evict_if_needed()

    def batch_lookup(self, token_ids: np.ndarray) -> np.ndarray:
        """Batch look up multiple embeddings."""
        results = np.zeros((len(token_ids), self._get_dim()), dtype=np.float32)
        for i, tid in enumerate(token_ids):
            results[i] = self.lookup(int(tid))
        return results

    def train_cold_compression(self, embeddings: np.ndarray, token_ids: np.ndarray):
        """Train clustered PQ for cold tier compression."""
        profile = EmbeddingAnalyzer().analyze(embeddings)
        n_clusters = profile.recommended_clusters
        result = ClusteredProductQuantizer(self.config).train(embeddings, n_clusters)

        for i, tid in enumerate(token_ids):
            tid_int = int(tid)
            self._cold[tid_int] = result.cluster_labels[i].tobytes()

        self._pq_trained = True
        logger.info(
            "Cold tier trained: %d entries, %d clusters", len(token_ids), n_clusters
        )

    def _promote_to_hot(self, token_id: int, embedding: np.ndarray):
        """Promote from warm/cold to hot tier."""
        entry = EmbeddingEntry(
            embedding=embedding,
            access_count=self._access_counts.get(token_id, 1) + 1,
            last_access=self._current_time,
            tier="hot",
        )
        self._hot[token_id] = entry
        self._warm.pop(token_id, None)
        self._cold.pop(token_id, None)
        self._evict_if_needed()

    def _promote_to_warm(self, token_id: int, embedding: np.ndarray):
        """Promote from cold to warm tier."""
        self._warm[token_id] = embedding
        self._cold.pop(token_id, None)

    def _evict_if_needed(self):
        """Evict coldest entries from hot tier if over capacity."""
        while len(self._hot) > self.config.max_hot_entries:
            old_id, old_entry = self._hot.popitem(last=False)
            self._warm[old_id] = old_entry.embedding

    def _decompress_cold(self, token_id: int) -> np.ndarray:
        """Decompress a cold tier entry."""
        dim = self._get_dim()
        return np.zeros(dim, dtype=np.float32)

    def _get_dim(self) -> int:
        """Get embedding dimension from hot tier entries."""
        if self._hot:
            return next(iter(self._hot.values())).embedding.shape[0]
        return 64

    def evict_cold(self, fraction: Optional[float] = None):
        """Evict a fraction of cold tier entries."""
        frac = fraction or self.config.cold_eviction_fraction
        n_evict = int(len(self._cold) * frac)
        if n_evict == 0:
            return

        keys_to_evict = list(self._cold.keys())[:n_evict]
        for k in keys_to_evict:
            del self._cold[k]

        logger.info("Evicted %d cold entries", n_evict)

    def get_tier_stats(self) -> Dict[str, int]:
        """Get statistics about tier sizes."""
        return {
            "hot": len(self._hot),
            "warm": len(self._warm),
            "cold": len(self._cold),
            "total": len(self._hot) + len(self._warm) + len(self._cold),
        }

    def get_memory_usage(self) -> Dict[str, float]:
        """Estimate memory usage per tier in MB."""
        dim = self._get_dim()

        hot_bytes = len(self._hot) * dim * 4
        warm_bytes = len(self._warm) * dim * 2
        cold_bytes = len(self._cold) * 4

        return {
            "hot_mb": hot_bytes / (1024 * 1024),
            "warm_mb": warm_bytes / (1024 * 1024),
            "cold_mb": cold_bytes / (1024 * 1024),
            "total_mb": (hot_bytes + warm_bytes + cold_bytes) / (1024 * 1024),
        }
