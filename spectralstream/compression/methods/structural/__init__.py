"""Structural pruning and sparsity methods."""

from ._class_wrappers import (
    Einsort,
    MonarchStructured,
    ButterflyStructured,
    Circulant,
    Vandermonde,
    Cauchy,
    HSSMatrix,
    BSSMatrix,
    Structured24,
    BlockSparsity,
    UnstructuredPruning,
    SparseGPT,
    WandaPruning,
    DynamicNMSparsity,
    ChannelPruning,
    GroupLasso,
    AdaptiveSparsity,
    SparseQuantizeCombined,
    StructuredLowRank,
    OptimalTransportCompression,
    BasisSharing,
)

# Migrated archive compression methods
from .manifold_embedding import ManifoldConfig, ManifoldEmbedding

__all__ = [
    "Einsort",
    "MonarchStructured",
    "ButterflyStructured",
    "Circulant",
    "Vandermonde",
    "Cauchy",
    "HSSMatrix",
    "BSSMatrix",
    "Structured24",
    "BlockSparsity",
    "UnstructuredPruning",
    "SparseGPT",
    "WandaPruning",
    "DynamicNMSparsity",
    "ChannelPruning",
    "GroupLasso",
    "AdaptiveSparsity",
    "SparseQuantizeCombined",
    "StructuredLowRank",
    "OptimalTransportCompression",
    "BasisSharing",
    "ManifoldConfig",
    "ManifoldEmbedding",
]
