"""
SpectralStream Embeddings Package
==================================
Embedding compression, analysis, and retrieval modules migrated from v1 archive.

Modules:
    extreme_embedding      — Clustered PQ, low-rank SVD, hot/warm/cold tiering
    embedding_compressor   — PQ, clustered PQ, low-rank+residual compression
    embeddings_handler     — OpenAI-compatible embeddings endpoint (model + HDC)
"""

from __future__ import annotations

from .extreme_embedding import (
    EmbeddingAnalyzer,
    EmbeddingCompressionConfig,
    EmbeddingEntry,
    EmbeddingProfile,
    ClusteredProductQuantizer,
    ClusteredPQResult,
    LowRankEmbedding,
    LowRankEmbeddingResult,
    HybridEmbeddingStore,
    TieredEmbeddingDict,
)

from .embedding_compressor import (
    EmbeddingCompressor,
    EmbeddingCompressionResult,
)

from .embeddings_handler import (
    compute_embedding_model,
    compute_embedding_hdc,
    handle_embeddings_request,
)

__all__ = [
    # extreme_embedding
    "EmbeddingAnalyzer",
    "EmbeddingCompressionConfig",
    "EmbeddingEntry",
    "EmbeddingProfile",
    "ClusteredProductQuantizer",
    "ClusteredPQResult",
    "LowRankEmbedding",
    "LowRankEmbeddingResult",
    "HybridEmbeddingStore",
    "TieredEmbeddingDict",
    # embedding_compressor
    "EmbeddingCompressor",
    "EmbeddingCompressionResult",
    # embeddings_handler
    "compute_embedding_model",
    "compute_embedding_hdc",
    "handle_embeddings_request",
]
