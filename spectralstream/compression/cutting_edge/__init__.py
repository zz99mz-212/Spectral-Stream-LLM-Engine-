from ._compressionmethod import CompressionMethod
from ._quantumstatecompression import QuantumStateCompression
from ._quantumentanglementcompression import QuantumEntanglementCompression
from ._quantumtunnelingoptimizer import QuantumTunnelingOptimizer
from ._densitymatrixcompression import DensityMatrixCompression
from ._quantumerrorcorrectioncompression import QuantumErrorCorrectionCompression
from ._vlasovdistributioncompression import VlasovDistributionCompression
from ._plasmaoscillationdecomposition import PlasmaOscillationDecomposition
from ._mhdwavecompression import MHDWaveCompression
from ._debyeshieldingcompression import DebyeShieldingCompression
from ._plasmaturbulencedecomposition import PlasmaTurbulenceDecomposition
from ._ratedistortionoptimalcompression import RateDistortionOptimalCompression
from ._mutualinformationcompression import MutualInformationCompression
from ._kolmogorovcomplexityapproximation import KolmogorovComplexityApproximation
from ._fisherinformationweighting import FisherInformationWeighting
from ._entropyratecompression import EntropyRateCompression
from ._manifoldlearningcompression import ManifoldLearningCompression
from ._optimaltransportcompression import OptimalTransportCompression
from ._categorytheorycompression import CategoryTheoryCompression
from ._algebraicgeometrycompression import AlgebraicGeometryCompression
from ._topologicaldatacompression import TopologicalDataCompression
from ._resonancecompression import ResonanceCompression
from ._harmonicoscillatordecomposition import HarmonicOscillatorDecomposition
from ._fourierneuraloperatorcompression import FourierNeuralOperatorCompression
from ._waveletscatteringtransform import WaveletScatteringTransform
from ._neuralodecompression import NeuralODECompression
from ._benchmarkresult import BenchmarkResult
from ._compressionbenchmark import CompressionBenchmark

from ._compressionmethod import ALL_METHODS, _ensure_2d, _restore_shape, _safe_bytes

ALL_METHODS: dict = ALL_METHODS


def get_all_methods() -> dict:
    return dict(ALL_METHODS)


def get_methods_by_category(category: str) -> dict:
    return {n: m for n, m in ALL_METHODS.items() if m.category == category}


# Register all methods
_all_classes = [
    QuantumStateCompression,
    QuantumEntanglementCompression,
    QuantumTunnelingOptimizer,
    DensityMatrixCompression,
    QuantumErrorCorrectionCompression,
    VlasovDistributionCompression,
    PlasmaOscillationDecomposition,
    MHDWaveCompression,
    DebyeShieldingCompression,
    PlasmaTurbulenceDecomposition,
    RateDistortionOptimalCompression,
    MutualInformationCompression,
    KolmogorovComplexityApproximation,
    FisherInformationWeighting,
    EntropyRateCompression,
    ManifoldLearningCompression,
    OptimalTransportCompression,
    CategoryTheoryCompression,
    AlgebraicGeometryCompression,
    TopologicalDataCompression,
    ResonanceCompression,
    HarmonicOscillatorDecomposition,
    FourierNeuralOperatorCompression,
    WaveletScatteringTransform,
    NeuralODECompression,
]
for _cls in _all_classes:
    _inst = _cls()
    ALL_METHODS[_inst.name] = _inst

# _ce aliases for integration compatibility
algebraic_geometry_ce = ALL_METHODS["algebraic_geometry"]
harmonic_oscillator_ce = ALL_METHODS["harmonic_oscillator"]
kolmogorov_complexity_ce = ALL_METHODS["kolmogorov_complexity"]
neural_ode_ce = ALL_METHODS["neural_ode"]
optimal_transport_ce = ALL_METHODS["optimal_transport"]
vlasov_distribution_ce = ALL_METHODS["vlasov_distribution"]
plasma_oscillation_ce = ALL_METHODS["plasma_oscillation"]
quantum_state_ce = ALL_METHODS["quantum_state"]
quantum_entanglement_ce = ALL_METHODS["quantum_entanglement"]
quantum_tunneling_ce = ALL_METHODS["quantum_tunneling"]
quantum_error_correct_ce = ALL_METHODS["quantum_error_correction"]
density_matrix_ce = ALL_METHODS["density_matrix"]
debye_shielding_ce = ALL_METHODS["debye_shielding"]
plasma_turbulence_ce = ALL_METHODS["plasma_turbulence"]
mhd_wave_ce = ALL_METHODS["mhd_wave"]
mutual_information_ce = ALL_METHODS["mutual_information"]
fisher_information_weighted_ce = ALL_METHODS["fisher_information_weighted"]
entropy_rate_ce = ALL_METHODS["entropy_rate"]
manifold_learning_ce = ALL_METHODS["manifold_learning"]
category_theory_ce = ALL_METHODS["category_theory"]
topological_data_ce = ALL_METHODS["topological_data"]
resonance_ce = ALL_METHODS["resonance"]
fourier_neural_operator_ce = ALL_METHODS["fourier_neural_operator"]
wavelet_scattering_ce = ALL_METHODS["wavelet_scattering"]
rate_distortion_ce = ALL_METHODS["rate_distortion_optimal"]
