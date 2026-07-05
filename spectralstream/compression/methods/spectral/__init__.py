"""Spectral methods."""

# Import from _class_wrappers (block_int8 backbone) for reliable implementations
from ._class_wrappers import (
    DCTBlock,
    DCT2D,
    DCT2DBlock,
    DCTAdaptiveBits,
    DCTQuant,
    SpectralDCTHybrid,
    DCTSparseQuant,
    FWHT,
    FWHTQuant,
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
    RandomProjectionCompression,
    StateSpaceWaveform,
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

# NN weight transforms
from .nn_weight_transforms import (
    NNWeightTransform,
    RowWiseHadamardINT8,
    ColumnWiseHadamardINT8,
    BlockHadamardINT8,
    HadamardRowINT4,
    HadamardBlockINT4,
    DCTRowINT8,
    DCTBlockINT8,
    WaveletRowINT8,
    WaveletBlockINT8,
    HadamardDCTINT8,
    MultiScaleINT8,
    MixedResolutionINT8,
    OutlierPreservingINT8,
    SensitivityWeightedINT8,
    BlockSkewHadamardINT8,
    RandomRotationINT8,
    KroneckerHadamardINT8,
    LowRankHadamardINT8,
    BlockDiagonalINT8,
    AdaptiveTransformINT8,
    ALL_NN_TRANSFORMS,
)

# Polynomial approximation methods (migrated from archive)
from .polynomial_approx import (
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

# Also import any working original implementations
try:
    from .dct_compression import HierarchicalDCT
except ImportError:
    pass
try:
    from .fourier_compression import FWHTCompress
except ImportError:
    pass
try:
    from .wavelet_compression import WaveletAdaptiveCompress
except ImportError:
    pass
try:
    from .transform_compression import WinogradTransform
except ImportError:
    pass
try:
    from .working_transforms import (
        WorkingTransformCompressor,
        PlainBlockInt8,
        HadamardBlockQuantize,
        DCTBlockQuantize,
        WaveletBlockQuantize,
        HadamardMixedPrecision,
        DCTMixedPrecision,
        WaveletMixedPrecision,
        HadamardDCTHybrid,
        MultiResolutionQuantize,
        SpectralSliceQuantize,
        AdaptiveTransformSelect,
        quantize_int8,
        dequantize_int8,
        quantize_int4,
        quantize_int2,
        pack_int4,
        unpack_int4,
        pack_int2,
        unpack_int2,
        compute_metrics,
        ALL_WORKING_TRANSFORMS,
    )
except ImportError:
    pass

# Migrated archive compression methods
try:
    from .hadamard_transform import HadamardConfig, HadamardTransformCompression
except ImportError:
    pass
try:
    from .dct_spectral import DCTSpectralConfig, DCTSpectralCompression
except ImportError:
    pass
try:
    from .wavelet_threshold import WaveletConfig, WaveletThresholdCompression
except ImportError:
    pass
