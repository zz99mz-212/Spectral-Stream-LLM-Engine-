"""Wrap all cutting_edge compression methods with standard (bytes, dict) interface.

Lazy wrappers — no classes instantiated at import time.
get_cutting_edge_methods() returns adapter references, not instances.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from ._standalone_integration import _from_bytes, _to_bytes

from spectralstream.compression.cutting_edge import (
    AlgebraicGeometryCompression,
    CategoryTheoryCompression,
    DebyeShieldingCompression,
    DensityMatrixCompression,
    EntropyRateCompression,
    FisherInformationWeighting,
    FourierNeuralOperatorCompression,
    HarmonicOscillatorDecomposition,
    KolmogorovComplexityApproximation,
    ManifoldLearningCompression,
    MHDWaveCompression,
    MutualInformationCompression,
    NeuralODECompression,
    OptimalTransportCompression,
    PlasmaOscillationDecomposition,
    PlasmaTurbulenceDecomposition,
    QuantumEntanglementCompression,
    QuantumErrorCorrectionCompression,
    QuantumStateCompression,
    QuantumTunnelingOptimizer,
    RateDistortionOptimalCompression,
    ResonanceCompression,
    TopologicalDataCompression,
    VlasovDistributionCompression,
    WaveletScatteringTransform,
)


class _CuttingEdgeAdapter:
    """Lazy adapter — wraps a cutting_edge class, instantiates only on compress/decompress."""

    def __init__(self, name: str, category: str, cls: type) -> None:
        self.name = name
        self.category = category
        self._cls = cls

    def compress(self, tensor: np.ndarray, **kw: Any) -> Tuple[bytes, dict]:
        inst = self._cls()
        result, meta = inst.compress(tensor, **kw)
        d = result if isinstance(result, dict) else {"data": result}
        data = _to_bytes(d)
        if "original_shape" not in meta:
            meta["original_shape"] = tensor.shape
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        inst = self._cls()
        obj = _from_bytes(data)
        result = inst.decompress(obj, metadata)
        shape = metadata.get("original_shape") or metadata.get("orig_shape")
        if shape is not None:
            result = np.asarray(result).reshape(shape)
        return result.astype(np.float32)


ALL_CE_CLASSES: Dict[str, Tuple[str, type]] = {
    # Unique names — no existing METHOD_CLASSES key conflicts
    "algebraic_geometry_ce": ("functional", AlgebraicGeometryCompression),
    "category_theory_ce": ("functional", CategoryTheoryCompression),
    "fisher_information_weighted_ce": ("functional", FisherInformationWeighting),
    "fourier_neural_operator_ce": ("hybrid", FourierNeuralOperatorCompression),
    "harmonic_oscillator_ce": ("hybrid", HarmonicOscillatorDecomposition),
    "manifold_learning_ce": ("functional", ManifoldLearningCompression),
    "mhd_wave_ce": ("physics", MHDWaveCompression),
    "mutual_information_ce": ("functional", MutualInformationCompression),
    "optimal_transport_ce": ("functional", OptimalTransportCompression),
    # Alternative implementations (suffixed to avoid collision with engine methods)
    "density_matrix_ce": ("physics", DensityMatrixCompression),
    "quantum_state_ce": ("physics", QuantumStateCompression),
    "quantum_entanglement_ce": ("physics", QuantumEntanglementCompression),
    "quantum_tunneling_ce": ("physics", QuantumTunnelingOptimizer),
    "quantum_error_correct_ce": ("physics", QuantumErrorCorrectionCompression),
    "vlasov_distribution_ce": ("physics", VlasovDistributionCompression),
    "plasma_oscillation_ce": ("physics", PlasmaOscillationDecomposition),
    "debye_shielding_ce": ("physics", DebyeShieldingCompression),
    "plasma_turbulence_ce": ("physics", PlasmaTurbulenceDecomposition),
    "topological_data_ce": ("novel", TopologicalDataCompression),
    "entropy_rate_ce": ("entropy", EntropyRateCompression),
    "wavelet_scattering_ce": ("spectral", WaveletScatteringTransform),
    "neural_ode_ce": ("functional", NeuralODECompression),
    "kolmogorov_complexity_ce": ("functional", KolmogorovComplexityApproximation),
    "rate_distortion_ce": ("functional", RateDistortionOptimalCompression),
    "resonance_ce": ("physics", ResonanceCompression),
}


def get_cutting_edge_methods() -> Dict[str, Tuple[str, Any]]:
    """Return dict of (name -> (category, lazy adapter)) for all cutting_edge methods.

    No instantiation or testing — memory-safe for registration.
    """
    result: Dict[str, Tuple[str, Any]] = {}
    for name, (cat, cls) in ALL_CE_CLASSES.items():
        adapter = _CuttingEdgeAdapter(name, cat, cls)
        result[name] = (cat, adapter)
    return result
