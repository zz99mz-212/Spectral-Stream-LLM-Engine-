"""
Unified Core Mathematical Primitives for SpectralStream
=======================================================
Single canonical source of truth for ALL mathematical primitives used across
the SpectralStream compression system.

Refactored into submodules:
  prng.py, fft.py, transforms.py, spectral.py, numerical.py,
  coherence.py, quantization.py, kernels.py, rotators.py,
  hd_vectors.py, wavelets.py, ntt.py, compressed_sensing.py,
  decomposition.py, metrics.py, bfloat16.py, dtype_detection.py
"""

from __future__ import annotations

# BF16 utilities
from .bfloat16 import (
    bfloat16_to_float32,
    compression_ratio_adjustment,
    dtype_is_bf16,
    dtype_is_float,
    ensure_float32,
    float32_to_bfloat16,
    is_bfloat16,
    maybe_contract_to_uint16,
)

# Dtype detection
from .dtype_detection import (
    SAFETENSORS_DTYPE_MAP,
    analyze_model_dtypes,
    denormalize_from_compression,
    detect_native_dtype,
    dtype_is_bf16,
    dtype_is_float,
    get_dtype_size,
    get_precision_bits,
    ndarray_dtype_to_safetensors,
    normalize_dtype,
    normalize_for_compression,
    safetensors_dtype_to_str,
    scan_safetensors_header,
)

# PRNG
from .prng import next_power_of_two, splitmix64

# FFT
from .fft import fft, fftfreq, ifft, irfft, rfft

# Transforms (DCT, FWHT, zigzag, effective_rank)
from .transforms import (
    dct,
    dct_2d,
    effective_rank,
    fwht,
    idct,
    idct_2d,
    ifwht,
    zigzag_indices,
)

# Spectral analysis
from .spectral import (
    auto_keep_fraction,
    band_limit,
    energy_concentration,
    spectral_entropy,
    spectral_power_density,
)

# Numerical utilities
from .numerical import (
    cosine_similarity,
    gibbs_softmax,
    logsumexp,
    softmax,
    unit_vector,
)

# Coherence
from .coherence import cascade_eviction_score, landau_zener_coherence

# Quantization
from .quantization import LloydMaxQuantizer, vectorized_lloyd_max

# Kernels
from .kernels import (
    BAND_COMPRESSION,
    BAND_HIGH,
    BAND_LOW,
    BAND_NORMAL,
    apply_spectral_kernel,
    yukawa_kernel_1d,
)

# Rotators
from .rotators import DCTRotator, HadamardRotator

# HD Vectors & HRR
from .hd_vectors import (
    generate_random_complex_vector,
    generate_random_hd_vector,
    hrr_bind,
    hrr_bundle,
    hrr_unbind,
)

# Wavelets
from .wavelets import WaveletTransform

# NTT
from .ntt import NTT

# Compressed Sensing
from .compressed_sensing import CompressedSensing

# Decomposition
from .decomposition import SymAntiSymDecomposition, truncated_svd

# Legacy ops
from .legacy_ops import (
    attention,
    attention_tiled,
    mean_field_attention,
    min_p_sampling,
    rms_norm,
    rope,
    swiglu,
    top_k_sampling,
)

# Metrics (all individual metric functions)
from .metrics import (
    compression_quality,
    compute_all_metrics,
    compute_bit_error_rate,
    compute_correlation_coefficient,
    compute_cosine_similarity,
    compute_effective_rank_ratio,
    compute_histogram_overlap,
    compute_kld,
    compute_kolmogorov_smirnov,
    compute_mae,
    compute_max_abs_error,
    compute_mse,
    compute_nmse,
    compute_psnr,
    compute_relative_error,
    compute_rmse,
    compute_snr,
    compute_spectral_angle,
    compute_ssim,
    compute_wasserstein_distance,
)

# Quality (dataclass + assessor)
from .quality import CompressionQuality, QualityAssessor

# Dashboard
from .metrics_dashboard import (
    format_comparison_table,
    format_metrics_summary,
    format_rate_distortion_table,
)

__all__ = [
    # BF16
    "bfloat16_to_float32",
    "float32_to_bfloat16",
    "is_bfloat16",
    "ensure_float32",
    "maybe_contract_to_uint16",
    "compression_ratio_adjustment",
    "dtype_is_bf16",
    "dtype_is_float",
    # Dtype detection
    "SAFETENSORS_DTYPE_MAP",
    "analyze_model_dtypes",
    "denormalize_from_compression",
    "detect_native_dtype",
    "dtype_is_bf16",
    "dtype_is_float",
    "get_dtype_size",
    "get_precision_bits",
    "ndarray_dtype_to_safetensors",
    "normalize_dtype",
    "normalize_for_compression",
    "safetensors_dtype_to_str",
    "scan_safetensors_header",
    "splitmix64",
    "next_power_of_two",
    "fft",
    "ifft",
    "rfft",
    "irfft",
    "fftfreq",
    "dct",
    "idct",
    "dct_2d",
    "idct_2d",
    "zigzag_indices",
    "fwht",
    "ifwht",
    "effective_rank",
    "spectral_entropy",
    "spectral_power_density",
    "band_limit",
    "auto_keep_fraction",
    "energy_concentration",
    "softmax",
    "logsumexp",
    "gibbs_softmax",
    "unit_vector",
    "cosine_similarity",
    "landau_zener_coherence",
    "cascade_eviction_score",
    "LloydMaxQuantizer",
    "vectorized_lloyd_max",
    "yukawa_kernel_1d",
    "apply_spectral_kernel",
    "BAND_HIGH",
    "BAND_NORMAL",
    "BAND_LOW",
    "BAND_COMPRESSION",
    "HadamardRotator",
    "DCTRotator",
    "generate_random_hd_vector",
    "generate_random_complex_vector",
    "hrr_bind",
    "hrr_unbind",
    "hrr_bundle",
    "WaveletTransform",
    "NTT",
    "CompressedSensing",
    "SymAntiSymDecomposition",
    "truncated_svd",
    # Metrics
    "compute_mse",
    "compute_rmse",
    "compute_mae",
    "compute_nmse",
    "compute_snr",
    "compute_psnr",
    "compute_relative_error",
    "compute_cosine_similarity",
    "compute_max_abs_error",
    "compute_ssim",
    "compute_spectral_angle",
    "compute_histogram_overlap",
    "compute_kld",
    "compute_wasserstein_distance",
    "compute_kolmogorov_smirnov",
    "compute_correlation_coefficient",
    "compute_effective_rank_ratio",
    "compute_bit_error_rate",
    "compute_all_metrics",
    "compression_quality",
    # Quality dataclass + assessor
    "CompressionQuality",
    "QualityAssessor",
    # Dashboard
    "format_metrics_summary",
    "format_comparison_table",
    "format_rate_distortion_table",
    # Legacy ops
    "attention",
    "attention_tiled",
    "mean_field_attention",
    "min_p_sampling",
    "rms_norm",
    "rope",
    "swiglu",
    "top_k_sampling",
]
