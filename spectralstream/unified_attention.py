"""Backward-compat re-export — unified_attention migrated to attention."""

from __future__ import annotations
import warnings as _warnings

_warnings.warn(
    "spectralstream.unified_attention is deprecated. Use spectralstream.attention instead.",
    DeprecationWarning,
    stacklevel=2,
)

from spectralstream.attention.unified_attention import (
    VlasovMeanFieldAttention,
    VlasovBlock,
    VlasovFlashAttention,
    GyrokineticAttention,
    SymplecticAttentionIntegrator,
    VlasovHelmholtzDecomposition,
    VlasovAttentionLayer,
    UnifiedAttentionSelector,
    TurbulentCascadeAttention,
    EchoAttention,
    InstabilityAttention,
    AdaptiveDebyeAttention,
    MultiSpeciesPICAttention,
    QuantumWalkAttention,
    MPOAttention,
    WaveletLearnableAttention,
)
