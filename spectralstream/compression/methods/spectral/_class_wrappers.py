"""Backward-compatible wrappers delegating to real implementations."""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np

from spectralstream.compression.methods.spectral.dct import (
    DCTBlock,
    DCT2D,
    DCT2DBlock,
    DCTAdaptiveBits,
    DCTQuant,
    SpectralDCTHybrid,
    DCTSparseQuant,
)
from spectralstream.compression.methods.spectral.fwht import (
    FWHT,
    FWHTQuant,
    RandomizedHadamard,
    RandomRotationQuant,
)
from spectralstream.compression.methods.spectral.wavelet import (
    WaveletHaar,
    WaveletDaubechies,
    WaveletSymlet,
    WaveletScattering,
)
from spectralstream.compression.methods.spectral.fourier import Fourier, FrequencyDomain
from spectralstream.compression.methods.spectral.transforms import (
    NTTTransform,
    Givens,
    Chebyshev,
    Winograd,
    PolynomialApprox,
)
from spectralstream.compression.methods.spectral.sparse_transform import (
    ButterflySparse,
    SparseRandomProjection,
)
from spectralstream.compression.methods.spectral.random_projection import (
    RandomProjectionCompression,
)
from spectralstream.compression.methods.spectral.state_space_waveform import (
    StateSpaceWaveform,
)
from spectralstream.compression.methods.spectral.polynomial_approx import (
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
