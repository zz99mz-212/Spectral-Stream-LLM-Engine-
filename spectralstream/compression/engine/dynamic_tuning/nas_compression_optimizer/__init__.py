"""
Neural Architecture Search (NAS) for automatic discovery of optimal compression
stacking patterns per tensor type.

Uses evolutionary search + Bayesian surrogate modeling to find Pareto-optimal
sequences of compression methods.  Meta-learning accelerates search on similar
tensors via cached fingerprints.
"""

from __future__ import annotations

from ._nascompressionoptimizer import (
    NASCompressionOptimizer,
    StackingPattern,
    PatternScore,
    TensorSignature,
    SynergyMatrix,
    MetaLearningCache,
)

__all__ = [
    "NASCompressionOptimizer",
    "StackingPattern",
    "PatternScore",
    "TensorSignature",
    "SynergyMatrix",
    "MetaLearningCache",
]
