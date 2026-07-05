"""Compression method registry — ALL discoverable methods across 9+ categories.

All imports use lazy evaluation for memory efficiency.
Heavy sub-modules (breakthrough, functional_weight_space, archive) are
loaded on first access via _build_method_classes().
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Lazy imports: individual classes are lightweight; heavy sections use a lazy builder ──

# Cross-layer compression methods
from spectralstream.compression.methods.cross_layer import (
    DeltaEncoding,
    BasisSharing,
    WeightTransfer,
    LayerGrouping,
    HierarchicalDelta,
)

from spectralstream.compression.methods.novel.quantum.quantum_probability_filter import (
    QuantumProbabilityFilter,
)

# Engine built-in methods (always available)
from spectralstream.compression.engine._methods import (
    _BlockINT8,
    _BlockINT4,
    _HadamardINT8,
    _HadamardINT4,
    _SparsityINT4,
    _DeltaINT4,
    _SVDCompress,
    _DCTSpectral,
    _TensorTrain,
    _FWHTCompress,
)

# Decomposition methods
from spectralstream.compression.methods.decomposition._class_wrappers import (
    Butterfly,
    Monarch,
    CPDecomposition,
    EinsortTT,
    LOTR,
    Kronecker,
    CURDecomposition,
    HMatrix,
    Nystrom,
    RandomFeature,
    ADNTNMERA,
    IPEPS2D,
    BlockDiagonal,
    Toeplitz,
    Hankel,
    SVDTruncated,
    TensorNetwork,
    HierarchicalMPS,
    TensorTrain as DecompTensorTrain,
    TensorRing,
    TTOrthogonal,
    TTSVD,
    TuckerDecomposition,
    BlockTucker,
    HierarchicalTucker,
)

# Spectral methods
from spectralstream.compression.methods.spectral._class_wrappers import (
    DCTBlock,
    DCT2D,
    DCT2DBlock,
    DCTAdaptiveBits as DCTAdaptiveBits,
    DCTQuant as DCTQuant,
    SpectralDCTHybrid as SpectralDCTHybrid,
    DCTSparseQuant as DCTSparseQuant,
    FWHT as SpectralFWHT,
    FWHTQuant as FWHTQuant,
    WaveletHaar,
    WaveletDaubechies,
    WaveletSymlet,
    WaveletScattering,
    Fourier,
    FrequencyDomain,
    NTTTransform,
    Givens,
    Chebyshev,
    Winograd,
    PolynomialApprox,
    RandomizedHadamard,
    ButterflySparse,
    SparseRandomProjection,
    RandomRotationQuant,
    StateSpaceWaveform,
    RandomProjectionCompression,
    PolynomialRowApprox,
    PolynomialColumnApprox,
    Polynomial2DApprox,
    RationalApproximation,
    ChebyshevApprox,
    LegendreApprox,
    HermiteApprox,
    SplineRowApprox,
    SplineColumnApprox,
    Spline2DBicubic,
    BasisSplineApprox,
    PiecewiseLinear,
    PiecewiseConstant,
    LowRankPolynomial,
    KroneckerPolynomial,
    TensorTrainPolynomial,
    LowRankSpline,
    AdaptivePolynomial,
    WaveletPolynomial,
    NeuralPolynomialApproximator,
)

# Structural methods
from spectralstream.compression.methods.structural._class_wrappers import (
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

# Entropy methods
from spectralstream.compression.methods.entropy._class_wrappers import (
    Huffman,
    RANS,
    TANS,
    Arithmetic,
    LZ77,
    Deflate,
    BWTMTF,
    PredictiveCoding,
    AdaptiveArithmetic,
    EntropyRate,
    LZ77Entropy,
)

# Hybrid / Cascade methods
from spectralstream.compression.methods.hybrid._class_wrappers import (
    Cascade2Stage,
    Cascade3Stage,
    Cascade4Stage,
    QuantizeThenSparsify,
    DecomposeThenQuantize,
    TransformThenQuantize,
    TransformThenSparsify,
    DecomposeThenTransform,
    AllMethodsEnsemble,
)

# Lossless methods
from spectralstream.compression.methods.lossless._class_wrappers import (
    LosslessZlib,
    LosslessLZ4,
    LosslessZstd,
)

# Functional methods
from spectralstream.compression.methods.functional.boltzmann import (
    BoltzmannEncoding,
    MaxEntropy,
)
from spectralstream.compression.methods.functional.fractal import FractalCompression
from spectralstream.compression.methods.functional.hamiltonian import Hamiltonian
from spectralstream.compression.methods.functional.information import (
    InformationBottleneck,
    RateDistortionOptimal,
)

# Information-theoretic methods
from spectralstream.compression.methods.information.bottleneck import (
    InformationBottleneck as ArchiveInformationBottleneck,
)
from spectralstream.compression.methods.information.mutual_information import (
    MutualInformation,
)
from spectralstream.compression.methods.information.rate_distortion import (
    RateDistortion,
)
from spectralstream.compression.methods.information.kolmogorov_mdl import (
    KolmogorovMDL,
)
from spectralstream.compression.methods.information.fisher_weighted import (
    FisherWeighted,
)
from spectralstream.compression.methods.information.entropy_rate import (
    EntropyRateCoding,
)
from spectralstream.compression.methods.information.entropy_constrained import (
    EntropyConstrained,
)
from spectralstream.compression.methods.information.mutual_info_quantize import (
    MutualInfoQuantize,
)
from spectralstream.compression.methods.information.information_bottleneck import (
    IBCompression,
)
from spectralstream.compression.methods.information.rate_distortion_optimal import (
    RDBlahutArimoto,
)
from spectralstream.compression.methods.functional.kolmogorov import (
    KolmogorovComplexity,
)
from spectralstream.compression.methods.functional.lagrangian import Lagrangian
from spectralstream.compression.methods.functional.siren import SIRENINR
from spectralstream.compression.methods.functional.symbolic import SymbolicRegression
from spectralstream.compression.methods.functional.landau_zener import LandauZener
from spectralstream.compression.methods.functional.neural_ode import NeuralODE

# Physics-inspired methods
from spectralstream.compression.methods.physics.mhd import MHDCompression, Gyrokinetic
from spectralstream.compression.methods.physics.plasma import (
    PlasmaOscillation,
    DebyeShielding,
    PlasmaTurbulence,
    PlasmaField,
)
from spectralstream.compression.methods.physics.quantum import (
    DensityMatrix,
    QuantumState,
    QuantumEntanglement,
    QuantumTunneling,
    QuantumErrorCorrection,
    QuantumTensorNetwork,
)
from spectralstream.compression.methods.physics.resonance import (
    ResonanceModes,
    ResonanceCompression,
)
from spectralstream.compression.methods.physics.topology import (
    TopologicalData,
    TopologicalFunctional,
)
from spectralstream.compression.methods.physics.vlasov import (
    VlasovDistribution,
    VlasovMeanField,
)
from spectralstream.compression.methods.physics.noise_floor_compression import (
    NoiseFloorCompression,
)
from spectralstream.compression.methods.physics.topological_quant import (
    TopologicalQuantization,
)
from spectralstream.compression.methods.novel.vlasov_mf_compression import (
    VlasovMeanFieldCompression,
)

# Novel / Tensor Network methods
from spectralstream.compression.methods.novel import (
    Stage1StructuralDecomp,
    Stage2CrossLayerDelta,
    Stage3Hypernetwork,
    Stage4EntropyCoding,
    FullCascade1200,
    CrossLayerDeltaCompression,
    BlockwiseCrossLayerDelta,
    SparseDeltaEncoding,
    HypernetworkCompression,
    BlockwiseINRCompression,
    SimpleHypernetworkCompression,
    FourierFeatureCompression,
    HPCBlockSVD,
    MERAAdv,
    PEPSBoundary,
    QTTAdapt,
    TTCross,
    DMRGSweep,
    QTTFourier,
    MergingEntanglement,
    QuantumAmplitude,
    MatrixProductOperator,
    QuantumCircuit,
    FloquetTensor,
    QuantumCluster,
    SingularValueDensity,
    HyperspectralTensor,
    QuantumErrorCorrecting,
    QuantumBootstrap,
    MBQCCompress,
    TensorNetworkRegroup,
    DensityMatrixRenorm,
    QuantumFourierFeature,
    SpinGlass,
    TopologicalOrder,
    FractalWeightCompression,
)

# Revolutionary methods
from spectralstream.compression.methods.novel.physics.gauge_equivariant import (
    GaugeEquivariant,
)
from spectralstream.compression.methods.novel.topological.topological_skeleton import (
    TopologicalSkeleton,
)

# Quantization methods
from spectralstream.compression.methods.quantization.adaptive import (
    AdaptiveGroupQuant,
)
from spectralstream.compression.methods.quantization.awq import (
    AWQQuant,
    AWQActivationAwareQuant,
)
from spectralstream.compression.methods.quantization.binary import (
    BinaryQuant,
    TernaryQuant,
)
from spectralstream.compression.methods.quantization.gptq import (
    GPTQQuant,
    GPTQLayerQuant,
)
from spectralstream.compression.methods.quantization.kmeans import KMeansQuant
from spectralstream.compression.methods.quantization.lattice import E8Lattice
from spectralstream.compression.methods.quantization.hessian_aware import (
    HessianAwareQuantization,
)
from spectralstream.compression.methods.quantization.lloyd_max import (
    LloydMaxQuantization,
)
from spectralstream.compression.methods.quantization.mixed_precision import (
    MixedPrecision,
)
from spectralstream.compression.methods.quantization.nf4 import NF4
from spectralstream.compression.methods.quantization.product import (
    ProductQuantization,
)
from spectralstream.compression.methods.quantization.squeezellm import (
    SqueezeLLMNonuniform,
    SqueezeLLMNonUniformV2,
)
from spectralstream.compression.methods.quantization.stochastic import (
    StochasticRound,
)
from spectralstream.compression.methods.quantization.block import (
    BlockFloatingPoint,
)
from spectralstream.compression.methods.quantization.block_adaptive import (
    BlockAdaptiveQuant,
)
from spectralstream.compression.methods.quantization.hadamard import (
    HadamardGroupWise,
)
from spectralstream.compression.methods.quantization.learned_codebook import (
    LearnedCodebookQuant,
)
from spectralstream.compression.methods.quantization.multi_bitwidth import (
    MultiBitWidthArchive,
)
from spectralstream.compression.methods.quantization.residual_vq import (
    ResidualVectorQuant,
)
from spectralstream.compression.methods.quantization.weight_clustering import (
    WeightClusteringArchive,
)
from spectralstream.compression.methods.hybrid.error_feedback import (
    ErrorFeedbackQuant,
)

try:
    from spectralstream.compression.methods.sparsity.sparse_quantize import (
        SparseQuantize,
    )
except (ImportError, ModuleNotFoundError):
    SparseQuantize = None
from spectralstream.compression.methods.quantization.precision_engine import (
    PrecisionEngine,
    TensorProfiler,
    ErrorBudgetAllocator,
    CrossLayerOptimizer,
    TensorProfile,
    PrecisionResult,
    PrecisionCompressed,
)
from spectralstream.compression.methods.quantization.calibration_quantizer import (
    CalibrationPipeline,
    CalibrationDataCollector,
    QuantizerSelector,
    CalibrationData,
    QuantizedWeight,
    LayerProfile,
    CompressionResult,
)

# ── Lazy-loading dict that populates heavy sections on first access ──────

_EXTRA_LOADED: bool = False


def _load_extra() -> None:
    """Lazily populate heavy method sections (archive, variants, breakthrough, FWS)."""
    global _EXTRA_LOADED
    if _EXTRA_LOADED:
        return
    _EXTRA_LOADED = True

    _imports = [
        ("archive_integration", "_archive_integration", ImportError),
        ("advanced_methods", "_archive_integration (advanced)", ImportError),
        ("cutting_edge", "_cutting_edge_integration", ImportError),
        ("topological", "_topological_integration", ImportError),
        ("novel_library", "_novel_library_integration", ImportError),
        ("massive", "_massive_integration", Exception),
        ("method_variants", "method_variants", Exception),
        ("functional_weight_space", "structural.functional_weight_space", ImportError),
        ("standalone", "_standalone_integration", ImportError),
        ("advanced_upgrades", "_advanced_upgrades", ImportError),
        ("time_crystal", "_time_crystal_methods", ImportError),
        ("fractal_weight_hpc", "fractal_weight_compression_hpc", ImportError),
        ("breakthrough", "breakthrough.breakthrough_massive", Exception),
    ]

    for _name, _module, _exc_type in _imports:
        try:
            if _name == "archive_integration":
                from spectralstream.compression.methods.novel._archive_integration import (
                    get_advanced_methods,
                    get_archive_methods,
                )

                _archive_methods = get_archive_methods()
                for _arch_name, (_arch_cat, _arch_cls) in _archive_methods.items():
                    if _arch_name not in METHOD_CLASSES:
                        METHOD_CLASSES[_arch_name] = _arch_cls

            elif _name == "advanced_methods":
                _advanced_methods = get_advanced_methods()
                for _adv_name, (_adv_cat, _adv_cls) in _advanced_methods.items():
                    if _adv_name not in METHOD_CLASSES:
                        METHOD_CLASSES[_adv_name] = _adv_cls

            elif _name == "cutting_edge":
                from .novel._cutting_edge_integration import (
                    get_cutting_edge_methods,
                )

                _ce_methods = get_cutting_edge_methods()
                for _ce_name, (_ce_cat, _ce_cls) in _ce_methods.items():
                    if _ce_name not in METHOD_CLASSES:
                        METHOD_CLASSES[_ce_name] = _ce_cls

            elif _name == "topological":
                from .novel._topological_integration import (
                    get_topological_methods,
                )

                _topo_methods = get_topological_methods()
                for _topo_name, (_topo_cat, _topo_cls) in _topo_methods.items():
                    if _topo_name not in METHOD_CLASSES:
                        METHOD_CLASSES[_topo_name] = _topo_cls

            elif _name == "novel_library":
                from .novel._novel_library_integration import (
                    get_novel_library_methods,
                )

                _nl_methods = get_novel_library_methods()
                for _nl_name, (_nl_cat, _nl_cls) in _nl_methods.items():
                    if _nl_name not in METHOD_CLASSES:
                        METHOD_CLASSES[_nl_name] = _nl_cls

            elif _name == "massive":
                from .novel._massive_integration import get_massive_methods

                _massive_methods = get_massive_methods()
                for _mm_name, (_mm_cat, _mm_cls) in _massive_methods.items():
                    if _mm_name not in METHOD_CLASSES:
                        METHOD_CLASSES[_mm_name] = _mm_cls

            elif _name == "method_variants":
                from .method_variants import get_method_variants as _get_variants

                _variants = _get_variants(METHOD_CLASSES)
                for _v_name, _v_cls in _variants.items():
                    if _v_name not in METHOD_CLASSES:
                        METHOD_CLASSES[_v_name] = _v_cls

            elif _name == "functional_weight_space":
                from spectralstream.compression.methods.novel.structural.functional_weight_space import (  # type: ignore[import-untyped]
                    ALL_FUNCTIONAL_WEIGHT_SPACE_METHODS,
                )

                for _fws_cls in ALL_FUNCTIONAL_WEIGHT_SPACE_METHODS:
                    _fws_name = getattr(_fws_cls, "name", _fws_cls.__name__.lower())
                    if _fws_name not in METHOD_CLASSES:
                        METHOD_CLASSES[_fws_name] = _fws_cls

            elif _name == "standalone":
                from .novel._standalone_integration import (
                    get_standalone_methods,
                )

                _standalone_methods = get_standalone_methods()
                for _sa_name, (_sa_cat, _sa_cls) in _standalone_methods.items():
                    if _sa_name not in METHOD_CLASSES:
                        METHOD_CLASSES[_sa_name] = _sa_cls

            elif _name == "advanced_upgrades":
                from .novel._advanced_upgrades import get_advanced_upgrade_methods

                _upgrade_methods = get_advanced_upgrade_methods()
                for _up_name, (_up_cat, _up_cls) in _upgrade_methods.items():
                    if _up_name not in METHOD_CLASSES:
                        METHOD_CLASSES[_up_name] = _up_cls

            elif _name == "time_crystal":
                from .novel._time_crystal_methods import (
                    TimeCrystalSVD,
                    TimeCrystalPhase,
                    TimeCrystalFloquet,
                    TimeCrystalFWHT,
                    TimeCrystalBlock,
                )

                _tc_methods = {
                    "time_crystal_svd": TimeCrystalSVD,
                    "time_crystal_phase": TimeCrystalPhase,
                    "time_crystal_floquet": TimeCrystalFloquet,
                    "time_crystal_fwht": TimeCrystalFWHT,
                    "time_crystal_block": TimeCrystalBlock,
                }
                for _tc_name, _tc_cls in _tc_methods.items():
                    if _tc_name not in METHOD_CLASSES:
                        METHOD_CLASSES[_tc_name] = _tc_cls

            elif _name == "fractal_weight_hpc":
                from .novel.fractal_weight_compression_hpc import (
                    FractalWeightCompressionHPC,
                    fractal_weight_compress_hpc,
                    fractal_weight_decompress_hpc,
                )

                if "fractal_weight_hpc" not in METHOD_CLASSES:
                    METHOD_CLASSES["fractal_weight_hpc"] = FractalWeightCompressionHPC

            elif _name == "breakthrough":
                from spectralstream.compression.methods.novel.breakthrough import (
                    breakthrough_massive as _bm,
                )

                _registered_breakthrough = 0
                for _bname in dir(_bm):
                    _cls = getattr(_bm, _bname)
                    if (
                        isinstance(_cls, type)
                        and hasattr(_cls, "compress")
                        and hasattr(_cls, "category")
                    ):
                        _key = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", _bname)
                        _key = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", _key).lower()
                        for _suffix in (
                            "_compress",
                            "_compression",
                            "_coder",
                            "_codec",
                        ):
                            if _key.endswith(_suffix) and len(_key) > len(_suffix):
                                _key = _key[: -len(_suffix)]
                                break
                        if (
                            _key not in METHOD_CLASSES
                            and hasattr(_cls, "compress")
                            and callable(_cls.compress)
                        ):
                            METHOD_CLASSES[_key] = _cls
                            _registered_breakthrough += 1
                if _registered_breakthrough > 0:
                    logger.info(
                        "Registered %d breakthrough compression methods",
                        _registered_breakthrough,
                    )
        except _exc_type as _e:
            logger.debug("Failed to load %s: %s", _name, _e)


class _MethodClassesDict(Dict[str, Any]):
    """Dict that auto-populates lazy sections on first access."""

    def _ensure(self) -> None:
        _load_extra()

    def __getitem__(self, key: str) -> Any:
        self._ensure()
        return super().__getitem__(key)

    def __contains__(self, key: object) -> bool:
        self._ensure()
        return super().__contains__(key)

    def items(self):
        self._ensure()
        return super().items()

    def values(self):
        self._ensure()
        return super().values()

    def keys(self):
        self._ensure()
        return super().keys()

    def get(self, key: str, default: Any = None) -> Any | None:
        self._ensure()
        return super().get(key, default)

    def __len__(self) -> int:
        self._ensure()
        return super().__len__()

    def __iter__(self):
        self._ensure()
        return super().__iter__()


# ── Build comprehensive METHOD_CLASSES dict ──────────────────────────────

_METHOD_CLASSES_DATA: Dict[str, Any] = {
    # Engine built-in
    "block_int8": _BlockINT8,
    "block_int4": _BlockINT4,
    "hadamard_int8": _HadamardINT8,
    "hadamard_int4": _HadamardINT4,
    "sparsity_int4": _SparsityINT4,
    "delta_int4": _DeltaINT4,
    "svd_compress": _SVDCompress,
    "dct_spectral": _DCTSpectral,
    "tensor_train": _TensorTrain,
    "fwht_compress": _FWHTCompress,
    # Decomposition
    "butterfly": Butterfly,
    "monarch": Monarch,
    "cp_decomposition": CPDecomposition,
    "einsort_tt": EinsortTT,
    "lotr": LOTR,
    "kronecker": Kronecker,
    "cur_decomposition": CURDecomposition,
    "h_matrix": HMatrix,
    "nystrom": Nystrom,
    "random_feature": RandomFeature,
    "adntn_mera": ADNTNMERA,
    "ipeps_2d": IPEPS2D,
    "block_diagonal": BlockDiagonal,
    "toeplitz": Toeplitz,
    "hankel": Hankel,
    "svd_truncated": SVDTruncated,
    "tensor_network": TensorNetwork,
    "hierarchical_mps": HierarchicalMPS,
    "decomp_tensor_train": DecompTensorTrain,
    "tensor_ring": TensorRing,
    "tt_orthogonal": TTOrthogonal,
    "tt_svd": TTSVD,
    "tucker_decomposition": TuckerDecomposition,
    "block_tucker": BlockTucker,
    "hierarchical_tucker": HierarchicalTucker,
    # Spectral
    "dct_block": DCTBlock,
    "dct_2d": DCT2D,
    "dct_2d_block": DCT2DBlock,
    "dct_quant": DCTQuant,
    "dct_adaptive_bits": DCTAdaptiveBits,
    "spectral_dct_hybrid": SpectralDCTHybrid,
    "dct_sparse_quant": DCTSparseQuant,
    "fwht": SpectralFWHT,
    "fwht_quant": FWHTQuant,
    "wavelet_haar": WaveletHaar,
    "wavelet_daubechies": WaveletDaubechies,
    "wavelet_symlet": WaveletSymlet,
    "wavelet_scattering": WaveletScattering,
    "fourier": Fourier,
    "frequency_domain": FrequencyDomain,
    "ntt_transform": NTTTransform,
    "givens": Givens,
    "chebyshev": Chebyshev,
    "winograd": Winograd,
    "polynomial_approx": PolynomialApprox,
    "randomized_hadamard": RandomizedHadamard,
    "butterfly_sparse": ButterflySparse,
    "sparse_random_projection": SparseRandomProjection,
    "random_rotation_quant": RandomRotationQuant,
    "state_space_waveform": StateSpaceWaveform,
    "random_projection": RandomProjectionCompression,
    # Polynomial approximation (migrated from archive)
    "polynomial_row_approx": PolynomialRowApprox,
    "polynomial_column_approx": PolynomialColumnApprox,
    "polynomial_2d_approx": Polynomial2DApprox,
    "rational_approximation": RationalApproximation,
    "chebyshev_approx": ChebyshevApprox,
    "legendre_approx": LegendreApprox,
    "hermite_approx": HermiteApprox,
    "spline_row_approx": SplineRowApprox,
    "spline_column_approx": SplineColumnApprox,
    "spline_2d_bicubic": Spline2DBicubic,
    "basis_spline_approx": BasisSplineApprox,
    "piecewise_linear": PiecewiseLinear,
    "piecewise_constant": PiecewiseConstant,
    "low_rank_polynomial": LowRankPolynomial,
    "kronecker_polynomial": KroneckerPolynomial,
    "tensor_train_polynomial": TensorTrainPolynomial,
    "low_rank_spline": LowRankSpline,
    "adaptive_polynomial": AdaptivePolynomial,
    "wavelet_polynomial": WaveletPolynomial,
    "neural_polynomial_approximator": NeuralPolynomialApproximator,
    # Structural
    "einsort": Einsort,
    "monarch_structured": MonarchStructured,
    "butterfly_structured": ButterflyStructured,
    "circulant": Circulant,
    "vandermonde": Vandermonde,
    "cauchy": Cauchy,
    "hss_matrix": HSSMatrix,
    "bss_matrix": BSSMatrix,
    "structured_24": Structured24,
    "block_sparsity": BlockSparsity,
    "unstructured_pruning": UnstructuredPruning,
    "sparse_gpt": SparseGPT,
    "wanda_pruning": WandaPruning,
    "dynamic_nm_sparsity": DynamicNMSparsity,
    "channel_pruning": ChannelPruning,
    "group_lasso": GroupLasso,
    "adaptive_sparsity": AdaptiveSparsity,
    "sparse_quantize_combined": SparseQuantizeCombined,
    "structured_low_rank": StructuredLowRank,
    "optimal_transport": OptimalTransportCompression,
    "basis_sharing": BasisSharing,
    # Entropy
    "huffman": Huffman,
    "rans": RANS,
    "tans": TANS,
    "arithmetic": Arithmetic,
    "lz77": LZ77,
    "deflate": Deflate,
    "bwt_mtf": BWTMTF,
    "predictive": PredictiveCoding,
    "adaptive_arithmetic": AdaptiveArithmetic,
    "entropy_rate": EntropyRate,
    "lz77_entropy": LZ77Entropy,
    # Hybrid / Cascade
    "cascade_2_stage": Cascade2Stage,
    "cascade_3_stage": Cascade3Stage,
    "cascade_4_stage": Cascade4Stage,
    "quantize_then_sparsify": QuantizeThenSparsify,
    "decompose_then_quantize": DecomposeThenQuantize,
    "transform_then_quantize": TransformThenQuantize,
    "transform_then_sparsify": TransformThenSparsify,
    "decompose_then_transform": DecomposeThenTransform,
    "all_methods_ensemble": AllMethodsEnsemble,
    "error_feedback_quant": ErrorFeedbackQuant,
    # Lossless
    "lossless_zlib": LosslessZlib,
    "lossless_lz4": LosslessLZ4,
    "lossless_zstd": LosslessZstd,
    # Functional
    "boltzmann": BoltzmannEncoding,
    "max_entropy": MaxEntropy,
    "fractal": FractalCompression,
    "hamiltonian": Hamiltonian,
    "information_bottleneck": InformationBottleneck,
    "rate_distortion": RateDistortionOptimal,
    "kolmogorov": KolmogorovComplexity,
    "lagrangian": Lagrangian,
    "landau_zener": LandauZener,
    "neural_ode": NeuralODE,
    "siren": SIRENINR,
    "symbolic_regression": SymbolicRegression,
    # Information-theoretic
    "archive_information_bottleneck": ArchiveInformationBottleneck,
    "mutual_information": MutualInformation,
    "rate_distortion": RateDistortion,
    "kolmogorov_mdl": KolmogorovMDL,
    "fisher_weighted": FisherWeighted,
    "entropy_rate_coding": EntropyRateCoding,
    "entropy_constrained": EntropyConstrained,
    "mutual_info_quantize": MutualInfoQuantize,
    "ib_compression": IBCompression,
    "rd_blahut_arimoto": RDBlahutArimoto,
    # Physics-inspired
    "mhd": MHDCompression,
    "gyrokinetic": Gyrokinetic,
    "plasma_oscillation": PlasmaOscillation,
    "debye_shielding": DebyeShielding,
    "plasma_turbulence": PlasmaTurbulence,
    "plasma_field": PlasmaField,
    "density_matrix": DensityMatrix,
    "density_matrix_compress": DensityMatrix,
    "density_renorm_compress": DensityMatrixRenorm,
    "quantum_state": QuantumState,
    "quantum_entanglement": QuantumEntanglement,
    "quantum_tunneling": QuantumTunneling,
    "quantum_error_correction": QuantumErrorCorrection,
    "quantum_tensor_network": QuantumTensorNetwork,
    "resonance_modes": ResonanceModes,
    "resonance_compression": ResonanceCompression,
    "topological_data": TopologicalData,
    "topological_functional": TopologicalFunctional,
    "vlasov_distribution": VlasovDistribution,
    "vlasov_mean_field": VlasovMeanField,
    "vlasov_mean_field_compression": VlasovMeanFieldCompression,
    # Novel / Tensor Network
    "mera_adv": MERAAdv,
    "peps_boundary": PEPSBoundary,
    "qtt_adapt": QTTAdapt,
    "tt_cross": TTCross,
    "dmrg_sweep": DMRGSweep,
    "qtt_fourier": QTTFourier,
    "merging_entanglement": MergingEntanglement,
    "quantum_amplitude": QuantumAmplitude,
    "matrix_product_operator": MatrixProductOperator,
    "quantum_circuit": QuantumCircuit,
    "floquet_tensor": FloquetTensor,
    "quantum_cluster": QuantumCluster,
    "singular_value_density": SingularValueDensity,
    "hyperspectral_tensor": HyperspectralTensor,
    "quantum_error_correcting": QuantumErrorCorrecting,
    "quantum_bootstrap": QuantumBootstrap,
    "quantum_probability_filter": QuantumProbabilityFilter,
    "mbqc_compress": MBQCCompress,
    "tensor_network_regroup": TensorNetworkRegroup,
    "density_matrix_renorm": DensityMatrixRenorm,
    "quantum_fourier_feature": QuantumFourierFeature,
    "spin_glass": SpinGlass,
    "topological_order": TopologicalOrder,
    # Quantization
    "adaptive_group_quant": AdaptiveGroupQuant,
    "awq_quant": AWQQuant,
    "awq_activation_aware": AWQActivationAwareQuant,
    "binary_quant": BinaryQuant,
    "ternary_quant": TernaryQuant,
    "gptq_quant": GPTQQuant,
    "gptq_layer_quant": GPTQLayerQuant,
    "kmeans_quant": KMeansQuant,
    "e8_lattice": E8Lattice,
    "mixed_precision": MixedPrecision,
    "nf4": NF4,
    "product_quantization": ProductQuantization,
    "squeezellm_nonuniform": SqueezeLLMNonuniform,
    "squeezellm_nonuniform_v2": SqueezeLLMNonUniformV2,
    "stochastic_round": StochasticRound,
    # Quantization module unique methods (not in engine)
    "block_floating_point": BlockFloatingPoint,
    "hadamard_group_wise": HadamardGroupWise,
    "block_adaptive_quant": BlockAdaptiveQuant,
    "learned_codebook_quant": LearnedCodebookQuant,
    "multi_bitwidth_archive": MultiBitWidthArchive,
    "residual_vector_quant": ResidualVectorQuant,
    "weight_clustering_archive": WeightClusteringArchive,
    # Precision engine & calibration pipeline (orchestration layer)
    "precision_engine": PrecisionEngine,
    "tensor_profiler": TensorProfiler,
    "error_budget_allocator": ErrorBudgetAllocator,
    "cross_layer_optimizer": CrossLayerOptimizer,
    "calibration_pipeline": CalibrationPipeline,
    "calibration_collector": CalibrationDataCollector,
    "quantizer_selector": QuantizerSelector,
    # Novel cross-layer & hypernetwork methods
    "cross_layer_delta": CrossLayerDeltaCompression,
    "blockwise_cross_layer_delta": BlockwiseCrossLayerDelta,
    "sparse_delta": SparseDeltaEncoding,
    "hypernetwork_compress": HypernetworkCompression,
    "blockwise_inr": BlockwiseINRCompression,
    "simple_hypernetwork": SimpleHypernetworkCompression,
    "fourier_feature_compress": FourierFeatureCompression,
    "hpc_block_svd": HPCBlockSVD,
    # Cascade 1200:1 methods
    "cascade_stage1_structural": Stage1StructuralDecomp,
    "cascade_stage2_delta": Stage2CrossLayerDelta,
    "cascade_stage3_hypernetwork": Stage3Hypernetwork,
    "cascade_stage4_entropy": Stage4EntropyCoding,
    "cascade_full_1200": FullCascade1200,
    # Revolutionary methods (gauge-equivariant, topological skeleton)
    "gauge_equivariant": GaugeEquivariant,
    "topological_skeleton": TopologicalSkeleton,
    # Novel fractal methods
    "fractal_weight": FractalWeightCompression,
    # Cross-layer compression methods
    "delta_encoding": DeltaEncoding,
    "basis_sharing": BasisSharing,
    "weight_transfer": WeightTransfer,
    "layer_grouping": LayerGrouping,
    "hierarchical_delta": HierarchicalDelta,
    # Sparsity methods
    "sparse_quantize": SparseQuantize,
    # ── Archive reintegration (physics, noise-aware, novel, cutting-edge, advanced, unified_q) ──
}

METHOD_CLASSES: _MethodClassesDict = _MethodClassesDict(_METHOD_CLASSES_DATA)


class _AllMethodsDict(Dict[str, Any]):
    """Dict that lazily instantiates methods from METHOD_CLASSES on first access."""

    _built: bool = False

    def _ensure(self) -> None:
        if self._built:
            return
        self._built = True
        for _name, _cls in METHOD_CLASSES.items():
            try:
                self[_name] = _cls() if isinstance(_cls, type) else _cls
            except (TypeError, ValueError, RuntimeError):
                pass

    def __getitem__(self, key: str) -> Any:
        self._ensure()
        return super().__getitem__(key)

    def __contains__(self, key: object) -> bool:
        self._ensure()
        return super().__contains__(key)

    def items(self):
        self._ensure()
        return super().items()

    def values(self):
        self._ensure()
        return super().values()

    def keys(self):
        self._ensure()
        return super().keys()

    def get(self, key: str, default: Any = None) -> Any | None:
        self._ensure()
        return super().get(key, default)

    def __len__(self) -> int:
        self._ensure()
        return super().__len__()

    def __iter__(self):
        self._ensure()
        return super().__iter__()


ALL_METHODS: _AllMethodsDict = _AllMethodsDict()

ALL_CATEGORIES: List[str] = [
    "quantization",
    "transform_quant",
    "information",
    "sparsity_quant",
    "delta_quant",
    "decomposition",
    "spectral",
    "structural",
    "entropy",
    "hybrid",
    "cascade",
    "lossless",
    "functional",
    "functional_weight_space",
    "physics",
    "novel",
    "revolutionary_gauge",
    "revolutionary_topological",
    "novel_fractal",
    "tensor_network",
    # Topological categories
    "topological_biological",
    "geometric_topological_manifold",
    # Breakthrough categories
    "breakthrough_decomposition",
    "breakthrough_hybrid",
    "breakthrough_info",
    "breakthrough_math",
    "breakthrough_signal",
    # Massive novel integration categories
    "novel_chaotic",
    "novel_chaos",
    "novel_signal",
    "novel_info",
    "novel_physics",
    "novel_biological",
    "fractal_holographic",
    "information_theory_2",
    "unified_physics_quantum2",
    "quantum_compression",
    "quantum_engine",
    "revolutionary",
]


def get_all_methods() -> Dict[str, Any]:
    """Get lazily-instantiated instances of ALL methods."""
    instances: Dict[str, Any] = {}
    for _name, _cls in METHOD_CLASSES.items():
        try:
            instances[_name] = _cls() if isinstance(_cls, type) else _cls
        except (TypeError, ValueError, RuntimeError):
            pass
    return instances


def get_method(name: str):
    cls = METHOD_CLASSES.get(name)
    if cls is None:
        return None
    if isinstance(cls, type):
        try:
            return cls()
        except Exception:
            return None
    return cls
