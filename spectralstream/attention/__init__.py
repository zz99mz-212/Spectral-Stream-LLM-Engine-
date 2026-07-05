"""Attention package — unified attention mechanisms for SpectralStream.

Exports the key attention classes from unified_attention.py.
All are R&D implementations — use at your own risk.
"""

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

__all__ = [
    "VlasovMeanFieldAttention",
    "VlasovBlock",
    "VlasovFlashAttention",
    "GyrokineticAttention",
    "SymplecticAttentionIntegrator",
    "VlasovHelmholtzDecomposition",
    "VlasovAttentionLayer",
    "UnifiedAttentionSelector",
    "TurbulentCascadeAttention",
    "EchoAttention",
    "InstabilityAttention",
    "AdaptiveDebyeAttention",
    "MultiSpeciesPICAttention",
    "QuantumWalkAttention",
    "MPOAttention",
    "WaveletLearnableAttention",
]
from .unified_attention import *  # auto-split re-export
