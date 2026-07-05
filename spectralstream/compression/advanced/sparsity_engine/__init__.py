"""Sparsity engine — advanced pruning algorithms.

Each submodule is imported with try/except to isolate import-chain failures
from the systemic cross-module reference issues in the split package.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _safe_import(mod_name: str, cls_name: str):
    """Try to import a class from a submodule, returning None on failure."""
    try:
        mod = __import__(
            f"spectralstream.compression.advanced.sparsity_engine.{mod_name}",
            fromlist=[cls_name],
        )
        return getattr(mod, cls_name, None)
    except Exception as exc:
        logger.debug("sparsity_engine: skipping %s.%s (%s)", mod_name, cls_name, exc)
        return None


# Export everything safely
SparsityConfig = _safe_import("_sparsityconfig", "SparsityConfig")
PruningPattern = _safe_import("_pruningpattern", "PruningPattern")
PruningSignal = _safe_import("_pruningsignal", "PruningSignal")
PruningResult = _safe_import("_pruningresult", "PruningResult")
BasePruner = _safe_import("_basepruner", "BasePruner")
MagnitudePruner = _safe_import("_magnitudepruner", "MagnitudePruner")
WandaPruner = _safe_import("_wandapruner", "WandaPruner")
SparseGPTPruner = _safe_import("_sparsegptpruner", "SparseGPTPruner")
SpectralPruner = _safe_import("_spectralpruner", "SpectralPruner")
MovementPruner = _safe_import("_movementpruner", "MovementPruner")
CombinedPruner = _safe_import("_combinedpruner", "CombinedPruner")
VlasovPruner = _safe_import("_vlasovpruner", "VlasovPruner")
ResonantPruner = _safe_import("_resonantpruner", "ResonantPruner")
HolographicPruner = _safe_import("_holographicpruner", "HolographicPruner")
QuantumPruner = _safe_import("_quantumpruner", "QuantumPruner")
SelfOrganizingPruner = _safe_import("_selforganizingpruner", "SelfOrganizingPruner")
SparsePruner = _safe_import("_sparsepruner", "SparsePruner")
SparseFormat = _safe_import("_sparseformat", "SparseFormat")
DynamicSparseExecutor = _safe_import("_dynamicsparseexecutor", "DynamicSparseExecutor")
ActivationThreshold = _safe_import("_activationthreshold", "ActivationThreshold")
ActivationSparsity = _safe_import("_activationsparsity", "ActivationSparsity")
SpectralBandConfig = _safe_import("_spectralbandconfig", "SpectralBandConfig")
SpectralSparsity = _safe_import("_spectralsparsity", "SpectralSparsity")
TiledPattern = _safe_import("_tiledpattern", "TiledPattern")
StructuredSparsity = _safe_import("_structuredsparsity", "StructuredSparsity")
LayerSparsityState = _safe_import("_layersparsitystate", "LayerSparsityState")
AdaptiveSparsityManager = _safe_import(
    "_adaptivesparsitymanager", "AdaptiveSparsityManager"
)
HDCSparsityPredictor = _safe_import("_hdcsparsitypredictor", "HDCSparsityPredictor")
VlasovSparsity = _safe_import("_vlasovsparsity", "VlasovSparsity")
ResonantSparsity = _safe_import("_resonantsparsity", "ResonantSparsity")
HolographicSparsity = _safe_import("_holographicsparsity", "HolographicSparsity")
QuantumSparsity = _safe_import("_quantumsparsity", "QuantumSparsity")
SelfOrganizingSparsity = _safe_import(
    "_selforganizingsparsity", "SelfOrganizingSparsity"
)
UnifiedSparsityEngine = _safe_import("_unifiedsparsityengine", "UnifiedSparsityEngine")
