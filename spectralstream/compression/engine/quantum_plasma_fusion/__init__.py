"""
Quantum-Plasma Fusion Engine v3 — quantum annealing with tunneling
and tokamak-inspired plasma cascade ordering.
"""

from ._classic import (
    QuantumTensorState,
    QuantumStatePreparer,
    QuantumMethodSelector,
    PlasmaWavePropagator,
    SpectralFusionAnalyzer,
)

from ._quantumplasmafusionengine import (
    AnnealingResult,
    CATEGORY_INDEX,
    CATEGORY_PARENT,
    COUPLING_MATRIX,
    N_CATEGORIES,
    PARENT_CATEGORIES,
    QuantumPlasmaFusionEngine,
    TOKAMAK_CASCADE_ORDER,
    TOKAMAK_PHASE_MAP,
    TunnelEvent,
    fuse_with_engine,
)

__all__ = [
    "QuantumTensorState",
    "QuantumStatePreparer",
    "QuantumMethodSelector",
    "PlasmaWavePropagator",
    "SpectralFusionAnalyzer",
    "QuantumPlasmaFusionEngine",
    "AnnealingResult",
    "TunnelEvent",
    "fuse_with_engine",
    "CATEGORY_INDEX",
    "CATEGORY_PARENT",
    "COUPLING_MATRIX",
    "N_CATEGORIES",
    "PARENT_CATEGORIES",
    "TOKAMAK_CASCADE_ORDER",
    "TOKAMAK_PHASE_MAP",
]
