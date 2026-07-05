"""Automatic registration of ALL compression methods from METHOD_CLASSES.

This file discovers and registers every method with honest metadata.
Uses METHOD_CLASSES from methods/__init__.py as the source of truth.
"""

import logging
import re
from typing import Any, Dict, Tuple

from spectralstream.compression.registry.enum import CompressionMethod
from spectralstream.compression.registry.metadata import MethodMetadata
from spectralstream.compression.registry.registry import MethodRegistry

logger = logging.getLogger(__name__)

# ── Category → default metadata ranges ─────────────────────────────────────
CATEGORY_META: Dict[str, Tuple[Tuple[float, float], Tuple[float, float], str]] = {
    "quantization": ((3.0, 16.0), (0.003, 0.15), "Reduce bit precision per block"),
    "transform_quant": ((3.0, 16.0), (0.005, 0.15), "Transform-domain quantization"),
    "sparsity_quant": ((2.0, 12.0), (0.01, 0.2), "Sparsity-aware quantization"),
    "delta_quant": ((2.0, 8.0), (0.005, 0.1), "Delta encoding quantization"),
    "decomposition": (
        (5.0, 100.0),
        (0.01, 0.3),
        "Factorize tensor into low-rank components",
    ),
    "spectral": (
        (2.0, 20.0),
        (0.005, 0.08),
        "Transform-domain coefficient thresholding",
    ),
    "entropy": ((1.5, 4.0), (0.0, 0.0), "Entropy coding of quantized symbols"),
    "structural": ((2.0, 10.0), (0.01, 0.3), "Exploit sparsity or structure"),
    "functional": (
        (5.0, 50.0),
        (0.05, 0.3),
        "Functional approximation of weight space",
    ),
    "physics": ((5.0, 100.0), (0.05, 0.5), "Physics-inspired compression"),
    "lossless": ((1.5, 4.0), (0.0, 0.0), "Lossless byte-level compression"),
    "hybrid": ((10.0, 200.0), (0.005, 0.05), "Multi-stage cascade compression"),
    "tensor_network": (
        (10.0, 500.0),
        (0.005, 0.15),
        "Tensor network & quantum-inspired",
    ),
    "novel": ((5.0, 100.0), (0.01, 0.15), "Novel compression approaches"),
}

CATEGORY_NAMES: Dict[str, str] = {cat: cat for cat in CATEGORY_META}


def _register_all() -> None:
    """Register every compression method from METHOD_CLASSES and engine built-ins.

    Scans METHOD_CLASSES and registers each one with appropriate metadata.
    Also registers engine built-in methods that may not be in METHOD_CLASSES.
    """
    from spectralstream.compression.methods import METHOD_CLASSES
    from spectralstream.compression.engine.method_tiers import get_tier
    from spectralstream.compression.engine._methods import (
        METHOD_REGISTRY as ENGINE_METHODS,
    )

    count = 0
    tier_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

    # Collect all methods: METHOD_CLASSES + engine built-ins
    all_classes = dict(METHOD_CLASSES)

    # Add engine built-in methods that may not be in METHOD_CLASSES
    for name, inst in ENGINE_METHODS.items():
        if name not in all_classes:
            all_classes[name] = type(inst)

    for method_name, method_cls in all_classes.items():
        try:
            if isinstance(method_cls, type):
                cat = getattr(method_cls, "category", "quantization")
                description = (method_cls.__doc__ or "").strip().split("\n")[0][:120]
            else:
                cat = getattr(method_cls, "category", "quantization")
                description = ""
            tier = get_tier(method_name, cat)
            tier_counts[tier.value] = tier_counts.get(tier.value, 0) + 1

            ratio_range, error_range, default_desc = CATEGORY_META.get(
                cat, ((2.0, 10.0), (0.01, 0.1), "")
            )
            is_lossless = cat in ("entropy", "lossless")
            needs_cal = cat in (
                "decomposition",
                "functional",
                "physics",
                "hybrid",
                "tensor_network",
                "novel",
            )

            method_id = _find_enum_for_name(method_name)
            if method_id is None:
                logger.debug("Skipping %s (no enum entry)", method_name)
                continue

            meta = MethodMetadata(
                method_id=method_id,
                name=method_name,
                category=cat,
                description=description or default_desc or f"{method_name} compression",
                compression_ratio_range=ratio_range,
                expected_error_range=error_range,
                is_lossless=is_lossless,
                requires_calibration=needs_cal,
            )
            MethodRegistry.register(method_id, meta)
            count += 1
        except Exception as e:
            logger.debug("Failed to register %s: %s", method_name, e)

    logger.info(
        "Registered %d methods from METHOD_CLASSES "
        "(Tier1=%d Tier2=%d Tier3=%d Tier4=%d Tier5=%d)",
        count,
        tier_counts.get(1, 0),
        tier_counts.get(2, 0),
        tier_counts.get(3, 0),
        tier_counts.get(4, 0),
        tier_counts.get(5, 0),
    )


# ── Comprehensive name → enum mapping ──────────────────────────────────────
_NAME_TO_ENUM: Dict[str, CompressionMethod] = {
    # Core quantization
    "block_int8": CompressionMethod.BLOCK_INT8,
    "block_int4": CompressionMethod.BLOCK_INT4,
    "hadamard_int8": CompressionMethod.HADAMARD_INT8,
    "hadamard_int4": CompressionMethod.HADAMARD_INT4,
    "sparsity_int4": CompressionMethod.SPARSITY_INT4,
    "delta_int4": CompressionMethod.DELTA_INT4,
    # Engine built-in
    "svd_compress": CompressionMethod.SVD_COMPRESS,
    "dct_spectral": CompressionMethod.DCT_SPECTRAL,
    "tensor_train": CompressionMethod.TENSOR_TRAIN,
    "fwht_compress": CompressionMethod.FWHT_COMPRESS,
    # Cascade engine
    "svd_truncated": CompressionMethod.SVD_TRUNCATED,
    "dct_2d": CompressionMethod.DCT_2D,
    "hadamard_quant": CompressionMethod.HADAMARD_QUANT,
    "lossless_zstd": CompressionMethod.LOSSLESS_ZSTD,
    "lossless_rans": CompressionMethod.LOSSLESS_RANS,
    # Decomposition
    "cp_decomposition": CompressionMethod.CP_DECOMPOSITION,
    "kronecker": CompressionMethod.KRONECKER,
    "cur_decomposition": CompressionMethod.CUR_DECOMPOSITION,
    "tucker_decomposition": CompressionMethod.TUCKER_DECOMPOSITION,
    "block_tucker": CompressionMethod.BLOCK_TUCKER,
    "hierarchical_tucker": CompressionMethod.HIERARCHICAL_TUCKER,
    "tensor_ring": CompressionMethod.TENSOR_RING,
    "tt_orthogonal": CompressionMethod.TT_ORTHOGONAL,
    "tt_svd": CompressionMethod.TT_SVD,
    "tensor_network": CompressionMethod.TENSOR_NETWORK,
    "hierarchical_mps": CompressionMethod.HIERARCHICAL_MPS,
    "butterfly": CompressionMethod.BUTTERFLY,
    "monarch": CompressionMethod.MONARCH,
    "einsort_tt": CompressionMethod.EINSORT_TT,
    "lotr": CompressionMethod.LOTR,
    "h_matrix": CompressionMethod.H_MATRIX,
    "nystrom": CompressionMethod.NYSTROM,
    "random_feature": CompressionMethod.RANDOM_FEATURE,
    "block_diagonal": CompressionMethod.BLOCK_DIAGONAL,
    "toeplitz": CompressionMethod.TOEPLITZ,
    "hankel": CompressionMethod.HANKEL,
    "adntn_mera": CompressionMethod.ADNTN_MERA,
    "ipeps_2d": CompressionMethod.IPEPS_2D,
    # Spectral
    "dct_block": CompressionMethod.DCT_BLOCK,
    "dct_2d_block": CompressionMethod.DCT_2D_BLOCK,
    "fwht": CompressionMethod.FWHT,
    "wavelet_haar": CompressionMethod.WAVELET_HAAR,
    "wavelet_daubechies": CompressionMethod.WAVELET_DAUBECHIES,
    "wavelet_symlet": CompressionMethod.WAVELET_SYMLET,
    "wavelet_scattering": CompressionMethod.WAVELET_SCATTERING,
    "fourier": CompressionMethod.FOURIER,
    "frequency_domain": CompressionMethod.FREQUENCY_DOMAIN,
    "ntt_transform": CompressionMethod.NTT_TRANSFORM,
    "givens": CompressionMethod.GIVENS,
    "chebyshev": CompressionMethod.CHEBYSHEV,
    "winograd": CompressionMethod.WINOGRAD,
    "polynomial_approx": CompressionMethod.POLYNOMIAL_APPROX,
    "polynomial_row_approx": CompressionMethod.POLYNOMIAL_ROW_APPROX,
    "polynomial_column_approx": CompressionMethod.POLYNOMIAL_COLUMN_APPROX,
    "polynomial_2d_approx": CompressionMethod.POLYNOMIAL_2D_APPROX,
    "rational_approximation": CompressionMethod.RATIONAL_APPROXIMATION,
    "chebyshev_approx": CompressionMethod.CHEBYSHEV_APPROX,
    "legendre_approx": CompressionMethod.LEGENDRE_APPROX,
    "hermite_approx": CompressionMethod.HERMITE_APPROX,
    "spline_row_approx": CompressionMethod.SPLINE_ROW_APPROX,
    "spline_column_approx": CompressionMethod.SPLINE_COLUMN_APPROX,
    "spline_2d_bicubic": CompressionMethod.SPLINE_2D_BICUBIC,
    "basis_spline_approx": CompressionMethod.BASIS_SPLINE_APPROX,
    "piecewise_linear": CompressionMethod.PIECEWISE_LINEAR,
    "piecewise_constant": CompressionMethod.PIECEWISE_CONSTANT,
    "low_rank_polynomial": CompressionMethod.LOW_RANK_POLYNOMIAL,
    "kronecker_polynomial": CompressionMethod.KRONECKER_POLYNOMIAL,
    "tensor_train_polynomial": CompressionMethod.TENSOR_TRAIN_POLYNOMIAL,
    "low_rank_spline": CompressionMethod.LOW_RANK_SPLINE,
    "adaptive_polynomial": CompressionMethod.ADAPTIVE_POLYNOMIAL,
    "wavelet_polynomial": CompressionMethod.WAVELET_POLYNOMIAL,
    "neural_polynomial_approximator": CompressionMethod.NEURAL_POLYNOMIAL_APPROXIMATOR,
    "randomized_hadamard": CompressionMethod.RANDOMIZED_HADAMARD,
    "butterfly_sparse": CompressionMethod.BUTTERFLY_SPARSE,
    "sparse_random_projection": CompressionMethod.SPARSE_RANDOM_PROJECTION,
    "random_rotation_quant": CompressionMethod.RANDOM_ROTATION_QUANT,
    # Structural
    "einsort": CompressionMethod.EINSORT,
    "monarch_structured": CompressionMethod.MONARCH_STRUCTURED,
    "butterfly_structured": CompressionMethod.BUTTERFLY_STRUCTURED,
    "circulant": CompressionMethod.CIRCULANT,
    "vandermonde": CompressionMethod.VANDERMONDE,
    "cauchy": CompressionMethod.CAUCHY,
    "hss_matrix": CompressionMethod.HSS_MATRIX,
    "bss_matrix": CompressionMethod.BSS_MATRIX,
    "structured_24": CompressionMethod.STRUCTURED_24,
    "block_sparsity": CompressionMethod.BLOCK_SPARSITY,
    "unstructured_pruning": CompressionMethod.UNSTRUCTURED_PRUNING,
    "sparse_gpt": CompressionMethod.SPARSE_GPT,
    "wanda_pruning": CompressionMethod.WANDA_PRUNING,
    "dynamic_nm_sparsity": CompressionMethod.DYNAMIC_NM_SPARSITY,
    "channel_pruning": CompressionMethod.CHANNEL_PRUNING,
    "group_lasso": CompressionMethod.GROUP_LASSO,
    "adaptive_sparsity": CompressionMethod.ADAPTIVE_SPARSITY,
    "sparse_quantize_combined": CompressionMethod.SPARSE_QUANTIZE_COMBINED,
    # Entropy
    "huffman": CompressionMethod.HUFFMAN,
    "rans": CompressionMethod.RANS,
    "tans": CompressionMethod.TANS,
    "arithmetic": CompressionMethod.ARITHMETIC,
    "lz77": CompressionMethod.LZ77,
    "deflate": CompressionMethod.DEFLATE,
    "bwt_mtf": CompressionMethod.BWT_MTF,
    "predictive": CompressionMethod.PREDICTIVE,
    # Hybrid
    "cascade_2_stage": CompressionMethod.CASCADE_2_STAGE,
    "cascade_3_stage": CompressionMethod.CASCADE_3_STAGE,
    "cascade_4_stage": CompressionMethod.CASCADE_4_STAGE,
    "quantize_then_sparsify": CompressionMethod.QUANTIZE_THEN_SPARSIFY,
    "decompose_then_quantize": CompressionMethod.DECOMPOSE_THEN_QUANTIZE,
    "transform_then_quantize": CompressionMethod.TRANSFORM_THEN_QUANTIZE,
    "transform_then_sparsify": CompressionMethod.TRANSFORM_THEN_SPARSIFY,
    "decompose_then_transform": CompressionMethod.DECOMPOSE_THEN_TRANSFORM,
    "all_methods_ensemble": CompressionMethod.ALL_METHODS_ENSEMBLE,
    # Lossless
    "lossless_zlib": CompressionMethod.ZLIB,
    "lossless_lz4": CompressionMethod.LZ4,
    "lossless_zstd": CompressionMethod.ZSTD,
    # Functional
    "boltzmann": CompressionMethod.BOLTZMANN,
    "fractal": CompressionMethod.FRACTAL,
    "hamiltonian": CompressionMethod.HAMILTONIAN,
    "information_bottleneck": CompressionMethod.INFORMATION,
    "kolmogorov": CompressionMethod.KOLMOGOROV,
    "lagrangian": CompressionMethod.LAGRANGIAN,
    "landau_zener": CompressionMethod.LANDAU_ZENER,
    "neural_ode": CompressionMethod.NEURAL_ODE,
    "siren": CompressionMethod.SIREN,
    "symbolic_regression": CompressionMethod.SYMBOLIC,
    # Physics
    "mhd": CompressionMethod.MHD,
    "vlasov": CompressionMethod.VLASOV,
    "density_matrix": CompressionMethod.DENSITY_MATRIX,
    "plasma_oscillation": CompressionMethod.PLASMA_OSCILLATION,
    "quantum": CompressionMethod.QUANTUM,
    # Novel / Tensor Network
    "mera_adv": CompressionMethod.MERA_ADV,
    "peps_boundary": CompressionMethod.PEPS_BOUNDARY,
    "qtt_adapt": CompressionMethod.QTT_ADAPT,
    "tt_cross": CompressionMethod.TT_CROSS,
    "dmrg_sweep": CompressionMethod.DMRG_SWEEP,
    "qtt_fourier": CompressionMethod.QTT_FOURIER,
    "merging_entanglement": CompressionMethod.MERGING_ENTANGLEMENT,
    "quantum_amplitude": CompressionMethod.QUANTUM_AMPLITUDE,
    "matrix_product_operator": CompressionMethod.MATRIX_PRODUCT_OPERATOR,
    "quantum_circuit": CompressionMethod.QUANTUM_CIRCUIT,
    "floquet_tensor": CompressionMethod.FLOQUET_TENSOR,
    "quantum_cluster": CompressionMethod.QUANTUM_CLUSTER,
    "singular_value_density": CompressionMethod.SINGULAR_VALUE_DENSITY,
    "hyperspectral_tensor": CompressionMethod.HYPERSPECTRAL_TENSOR,
    "quantum_error_correcting": CompressionMethod.QUANTUM_ERROR_CORRECTING,
    "quantum_bootstrap": CompressionMethod.QUANTUM_BOOTSTRAP,
    "mbqc_compress": CompressionMethod.MBQC_COMPRESS,
    "tensor_network_regroup": CompressionMethod.TENSOR_NETWORK_REGROUP,
    "density_matrix_renorm": CompressionMethod.DENSITY_MATRIX_RENORM,
    "quantum_fourier_feature": CompressionMethod.QUANTUM_FOURIER_FEATURE,
    "spin_glass": CompressionMethod.SPIN_GLASS,
    "topological_order": CompressionMethod.TOPOLOGICAL_ORDER,
    # Revolutionary
    "gauge_equivariant": CompressionMethod.GAUGE_EQUIVARIANT,
    "topological_skeleton": CompressionMethod.TOPOLOGICAL_SKELETON,
    # ── Cutting-edge methods ──
    "algebraic_geometry": CompressionMethod.ALGEBRAIC_GEOMETRY,
    "category_theory_ce": CompressionMethod.CATEGORY_THEORY_CE,
    "fisher_information_weighted": CompressionMethod.FISHER_INFORMATION_WEIGHTED,
    "fourier_neural_operator": CompressionMethod.FOURIER_NEURAL_OPERATOR,
    "harmonic_oscillator": CompressionMethod.HARMONIC_OSCILLATOR,
    "manifold_learning": CompressionMethod.MANIFOLD_LEARNING,
    "mhd_wave": CompressionMethod.MHD_WAVE,
    "mutual_information_ce": CompressionMethod.MUTUAL_INFORMATION_CE,
    "optimal_transport": CompressionMethod.OPTIMAL_TRANSPORT,
    "density_matrix_ce": CompressionMethod.DENSITY_MATRIX_CE,
    "quantum_state_ce": CompressionMethod.QUANTUM_STATE_CE,
    "quantum_entanglement_ce": CompressionMethod.QUANTUM_ENTANGLEMENT_CE,
    "quantum_tunneling_ce": CompressionMethod.QUANTUM_TUNNELING_CE,
    "quantum_error_correct_ce": CompressionMethod.QUANTUM_ERROR_CORRECT_CE,
    "vlasov_distribution_ce": CompressionMethod.VLASOV_DISTRIBUTION_CE,
    "plasma_oscillation_ce": CompressionMethod.PLASMA_OSCILLATION_CE,
    "debye_shielding_ce": CompressionMethod.DEBYE_SHIELDING_CE,
    "plasma_turbulence_ce": CompressionMethod.PLASMA_TURBULENCE_CE,
    "topological_data_ce": CompressionMethod.TOPOLOGICAL_DATA_CE,
    "entropy_rate_ce": CompressionMethod.ENTROPY_RATE_CE,
    "wavelet_scattering_ce": CompressionMethod.WAVELET_SCATTERING_CE,
    "neural_ode_ce": CompressionMethod.NEURAL_ODE_CE,
    "kolmogorov_complexity_ce": CompressionMethod.KOLMOGOROV_COMPLEXITY_CE,
    "rate_distortion_ce": CompressionMethod.RATE_DISTORTION_CE,
    "resonance_ce": CompressionMethod.RESONANCE_CE,
    # ── Novel library methods ──
    "cp_als": CompressionMethod.CP_ALS,
    "tucker_svd_nl": CompressionMethod.TUCKER_SVD_NL,
    "block_tucker_nl": CompressionMethod.BLOCK_TUCKER_NL,
    "htucker_nl": CompressionMethod.HTUCKER_NL,
    "tensor_network_nl": CompressionMethod.TENSOR_NETWORK_NL,
    "butterfly_factor_nl": CompressionMethod.BUTTERFLY_FACTOR_NL,
    "kronecker_product_nl": CompressionMethod.KRONECKER_PRODUCT_NL,
    "block_diag_nl": CompressionMethod.BLOCK_DIAG_NL,
    "circulant_approx_nl": CompressionMethod.CIRCULANT_APPROX_NL,
    "toeplitz_approx_nl": CompressionMethod.TOEPLITZ_APPROX_NL,
    "hankel_approx_nl": CompressionMethod.HANKEL_APPROX_NL,
    "lotr_nl": CompressionMethod.LOTR_NL,
    "tt_svd_nl": CompressionMethod.TT_SVD_NL,
    "tt_orth_nl": CompressionMethod.TT_ORTH_NL,
    "tr_svd_nl": CompressionMethod.TR_SVD_NL,
    "lloyd_max": CompressionMethod.LLOYD_MAX,
    "adaptive_scalar": CompressionMethod.ADAPTIVE_SCALAR,
    "rvq": CompressionMethod.RVQ,
    "additive_codebook": CompressionMethod.ADDITIVE_CODEBOOK,
    "hessian_aware": CompressionMethod.HESSIAN_AWARE,
    "fisher_info": CompressionMethod.FISHER_INFO,
    "hierarchical_pq": CompressionMethod.HIERARCHICAL_PQ,
    "lattice_anchored": CompressionMethod.LATTICE_ANCHORED,
    "e8_lattice_nl": CompressionMethod.E8_LATTICE_NL,
    "mixed_precision_nl": CompressionMethod.MIXED_PRECISION_NL,
    "gptq_layer_quant_nl": CompressionMethod.GPTQ_LAYER_QUANT_NL,
    "awq_activation_nl": CompressionMethod.AWQ_ACTIVATION_NL,
    "binary_quant_nl": CompressionMethod.BINARY_QUANT_NL,
    "ternary_quant_nl": CompressionMethod.TERNARY_QUANT_NL,
    "nf4_nl": CompressionMethod.NF4_NL,
    "product_quant_nl": CompressionMethod.PRODUCT_QUANT_NL,
    "ntt": CompressionMethod.NTT,
    "dct_spectral_nl": CompressionMethod.DCT_SPECTRAL_NL,
    "dct_2d_block_nl": CompressionMethod.DCT_2D_BLOCK_NL,
    "wavelet_threshold": CompressionMethod.WAVELET_THRESHOLD,
    "fwht_nl": CompressionMethod.FWHT_NL,
    "random_hadamard_nl": CompressionMethod.RANDOM_HADAMARD_NL,
    "winograd_nl": CompressionMethod.WINOGRAD_NL,
    "random_rot_quant_nl": CompressionMethod.RANDOM_ROT_QUANT_NL,
    "butterfly_sparse_nl": CompressionMethod.BUTTERFLY_SPARSE_NL,
    "sparse_random_proj": CompressionMethod.SPARSE_RANDOM_PROJ,
    "structured_sparsity_nl": CompressionMethod.STRUCTURED_SPARSITY_NL,
    "block_sparsity_nl": CompressionMethod.BLOCK_SPARSITY_NL,
    "unstruct_pruning_nl": CompressionMethod.UNSTRUCT_PRUNING_NL,
    "sparse_gpt_nl": CompressionMethod.SPARSE_GPT_NL,
    "wanda_pruning_nl": CompressionMethod.WANDA_PRUNING_NL,
    "dynamic_nm_nl": CompressionMethod.DYNAMIC_NM_NL,
    "channel_pruning_nl": CompressionMethod.CHANNEL_PRUNING_NL,
    "group_lasso_nl": CompressionMethod.GROUP_LASSO_NL,
    "adaptive_sparsity_nl": CompressionMethod.ADAPTIVE_SPARSITY_NL,
    "sparse_quant_nl": CompressionMethod.SPARSE_QUANT_NL,
    "huffman_nl": CompressionMethod.HUFFMAN_NL,
    "rans_nl": CompressionMethod.RANS_NL,
    "tans_nl": CompressionMethod.TANS_NL,
    "arithmetic_nl": CompressionMethod.ARITHMETIC_NL,
    "lz77_entropy_nl": CompressionMethod.LZ77_ENTROPY_NL,
    "vlasov_mf_nl": CompressionMethod.VLASOV_MF_NL,
    "holographic_phase": CompressionMethod.HOLOGRAPHIC_PHASE,
    "qtensor_net_nl": CompressionMethod.QTENSOR_NET_NL,
    "timecrystal": CompressionMethod.TIMECRYSTAL,
    "plasma_field_nl": CompressionMethod.PLASMA_FIELD_NL,
    "spectral_density": CompressionMethod.SPECTRAL_DENSITY,
    "fisher_rao": CompressionMethod.FISHER_RAO,
    "symplectic": CompressionMethod.SYMPLECTIC,
    "info_bottleneck": CompressionMethod.INFO_BOTTLENECK,
    "rd_optimal_nl": CompressionMethod.RD_OPTIMAL_NL,
    "landau_zener_nl": CompressionMethod.LANDAU_ZENER_NL,
    "boltzmann_nl": CompressionMethod.BOLTZMANN_NL,
    "max_entropy_nl": CompressionMethod.MAX_ENTROPY_NL,
    "cross_layer_delta_nl": CompressionMethod.CROSS_LAYER_DELTA_NL,
    # ── Advanced / Archive canonical names ──
    "turbo_quant_codec": CompressionMethod.TURBO_QUANT_CODEC,
    "freq_domain_compressor": CompressionMethod.FREQ_DOMAIN_COMPRESSOR,
    "tensor_ring_compressor": CompressionMethod.TENSOR_RING_COMPRESSOR,
    "tensor_train_compressor": CompressionMethod.TENSOR_TRAIN_COMPRESSOR,
    "amplitude_phase_compressor": CompressionMethod.AMPLITUDE_PHASE_COMPRESSOR,
    "holographic_weight_encoder": CompressionMethod.HOLOGRAPHIC_WEIGHT_ENCODER,
    "residual_vq_compressor": CompressionMethod.RESIDUAL_VQ_COMPRESSOR,
    "quantum_tensor_net_compressor": CompressionMethod.QUANTUM_TENSOR_NET_COMPRESSOR,
    "unified_quantizer": CompressionMethod.UNIFIED_QUANTIZER,
    "tt_pq_pipeline": CompressionMethod.TT_PQ_PIPELINE,
    # ── Standalone wrapper names ──
    "spectra_quantizer": CompressionMethod.UNIFIED_QUANTIZER,
    "tt_compressor_adv": CompressionMethod.TENSOR_TRAIN_COMPRESSOR,
    "tt_pq_advanced": CompressionMethod.TT_PQ_PIPELINE,
    # ── Enum-name to class-name mappings for missing 79 entries ──
    "svd_compress": CompressionMethod.SVD_COMPRESS,
    "lossless_rans": CompressionMethod.LOSSLESS_RANS,
    "hierarchical_dct": CompressionMethod.HIERARCHICAL_DCT,
    "fwht_compress_alt": CompressionMethod.FWHT_COMPRESS_ALT,
    "wavelet_adaptive": CompressionMethod.WAVELET_ADAPTIVE,
    "winograd_conv": CompressionMethod.WINOGRAD_TRANSFORM,
    "ans_core": CompressionMethod.ANS_CORE,
    "ans_table": CompressionMethod.ANS_TABLE,
    "zlib": CompressionMethod.ZLIB,
    "lz4": CompressionMethod.LZ4,
    "zstd": CompressionMethod.ZSTD,
    "huffman_codec": CompressionMethod.HUFFMAN_CODEC,
    "category_theory": CompressionMethod.CATEGORY_THEORY,
    "fisher": CompressionMethod.FISHER,
    "hierarchical": CompressionMethod.HIERARCHICAL,
    "information": CompressionMethod.INFORMATION,
    "symbolic": CompressionMethod.SYMBOLIC,
    "vlasov": CompressionMethod.VLASOV,
    "resonance": CompressionMethod.RESONANCE,
    "topology": CompressionMethod.TOPOLOGY,
    # ── Breakthrough massive damaged-name mappings ──
    "abelianvariety": CompressionMethod.ABELIAN_VARIETY_COMPRESS,
    "adaptivecrossapprox": CompressionMethod.ADAPTIVE_CROSS_APPROX,
    "adaptivefilter": CompressionMethod.ADAPTIVE_FILTER_COMPRESS,
    "adaptivelineenhancer": CompressionMethod.ADAPTIVE_LINE_ENHANCER,
    "adaptivelowrank": CompressionMethod.ADAPTIVE_LOW_RANK,
    "additivequanthybrid": CompressionMethod.ADDITIVE_CODEBOOK_QUANT,
    "algorithmicinfo": CompressionMethod.ALGORITHMIC_INFO,
    "analyticnumbertheory": CompressionMethod.ANALYTIC_NUMBER_THEORY,
    "apafilter": CompressionMethod.APA_FILTER_COMPRESS,
    "arnoldiiteration": CompressionMethod.ARNOLDI_ITERATION,
    "asymptoticquant": CompressionMethod.ASYMPTOTIC_QUANT,
    "autoencoderhybrid": CompressionMethod.BREAKTHROUGH_HYBRID,
    "automorphicform": CompressionMethod.AUTOMORPHIC_FORM,
    "blahutarimoto": CompressionMethod.BLAHUT_ARIMOTO,
    "blindequalizer": CompressionMethod.BLIND_EQUALIZER_COMPRESS,
    "blocklanczos": CompressionMethod.BLOCK_LANCZOS,
    "cmaequalizer": CompressionMethod.CMA_EQUALIZER_COMPRESS,
    "communitydetectionhybrid": CompressionMethod.BREAKTHROUGH_HYBRID,
    "compositequanthybrid": CompressionMethod.BREAKTHROUGH_HYBRID,
    "coveringnumquant": CompressionMethod.COVERING_NUM_QUANT,
    "crossapproximation": CompressionMethod.CROSS_APPROXIMATION,
    "crossentropyquant": CompressionMethod.CROSS_ENTROPY_QUANT,
    "dctkmeanshybrid": CompressionMethod.DCT_KMEANS_HYBRID,
    "decisionfeedbackeq": CompressionMethod.DECISION_FEEDBACK_EQ,
    "dictionaryhybrid": CompressionMethod.BREAKTHROUGH_HYBRID,
    "directionalmultipole": CompressionMethod.DIRECTIONAL_MULTIPOLE,
    "divideconquersvd": CompressionMethod.DIVIDE_CONQUER_SVD,
    "ellipticcurve": CompressionMethod.ELLIPTIC_CURVE_COMPRESS,
    "entropyconstrainedvq": CompressionMethod.ENTROPY_CONSTRAINED_VQ,
    "etalecohomology": CompressionMethod.ETALE_COHOMOLOGY_COMPRESS,
    "extendedkalmanfilter": CompressionMethod.EXTENDED_KALMAN_FILTER,
    "fieldtheory": CompressionMethod.FIELD_THEORY_COMPRESS,
    "fractionalspaceeq": CompressionMethod.FRACTIONAL_SPACE_EQ,
    "galoistheory": CompressionMethod.GALOIS_THEORY_COMPRESS,
    "gershobound": CompressionMethod.GERSHO_BOUND,
    "grouptheory": CompressionMethod.GROUP_THEORY_COMPRESS,
    "h2matrix": CompressionMethod.H2_MATRIX,
    "hinfinityfilter": CompressionMethod.H_INFINITY_FILTER,
    "highratequant": CompressionMethod.HIGH_RATE_QUANT,
    "hmatrixdecomp": CompressionMethod.H_MATRIX_DECOMP,
    "hierarchicalsvd": CompressionMethod.HIERARCHICAL_SVD,
    "infobottleneckquant": CompressionMethod.INFO_BOTTLENECK_QUANT,
    "jacobidavidson": CompressionMethod.JACOBI_DAVIDSON,
    "jacobianvariety": CompressionMethod.JACOBIAN_VARIETY_COMPRESS,
    "jensenshannonquant": CompressionMethod.JENSEN_SHANNON_QUANT,
    "kalmanfilter": CompressionMethod.KALMAN_FILTER_COMPRESS,
    "kolmogorovinfo": CompressionMethod.KOLMOGOROV_INFO,
    "krylovsubspace": CompressionMethod.KRYLOV_SUBSPACE,
    "kullbackleiblerquant": CompressionMethod.KULLBACK_LEIBLER_QUANT,
    "lfunction": CompressionMethod.L_FUNCTION_COMPRESS,
    "lefschetzfixedpoint": CompressionMethod.LEFSCHETZ_FIXED_POINT,
    "lmsfilter": CompressionMethod.LMS_FILTER_COMPRESS,
    "matchedfilter": CompressionMethod.MATCHED_FILTER_COMPRESS,
    "maxrelevancequant": CompressionMethod.MAX_RELEVANCE_QUANT,
    "minredundancyquant": CompressionMethod.MIN_REDUNDANCY_QUANT,
    "modularcurve": CompressionMethod.MODULAR_CURVE_COMPRESS,
    "modularform": CompressionMethod.MODULAR_FORM_COMPRESS,
    "motivic": CompressionMethod.MOTIVIC_COMPRESS,
    "mutualinfomaxquant": CompressionMethod.MUTUAL_INFO_MAX_QUANT,
    "nlmsfilter": CompressionMethod.NLMS_FILTER_COMPRESS,
    "numbertheory": CompressionMethod.NUMBER_THEORY_COMPRESS,
    "packingnumquant": CompressionMethod.PACKING_NUM_QUANT,
    "particlefilter": CompressionMethod.PARTICLE_FILTER_COMPRESS,
    "proxypoint": CompressionMethod.PROXY_POINT,
    "randomizedsvd": CompressionMethod.RANDOMIZED_SVD,
    "ratedistortionfunc": CompressionMethod.RATE_DISTORTION_FUNC,
    "ratedistortionvq": CompressionMethod.RATE_DISTORTION_VQ,
    "ringtheory": CompressionMethod.RING_THEORY_COMPRESS,
    "rlsfilter": CompressionMethod.RLS_FILTER_COMPRESS,
    "shimuravariety": CompressionMethod.SHIMURA_VARIETY_COMPRESS,
    "spherepackingquant": CompressionMethod.SPHERE_PACKING_QUANT,
    "streamingsvd": CompressionMethod.STREAMING_SVD,
    "svdwavelethybrid": CompressionMethod.SVD_WAVELET_HYBRID,
    "tensorringdecomp": CompressionMethod.TENSOR_RING_DECOMP,
    "thetafunction": CompressionMethod.THETA_FUNCTION_COMPRESS,
    "tuckerdecomp": CompressionMethod.TUCKER_DECOMP,
    "turboequalizer": CompressionMethod.TURBO_EQUALIZER_COMPRESS,
    "unscentedkalmanfilter": CompressionMethod.UNSCENTED_KALMAN_FILTER,
    "weilconjecture": CompressionMethod.WEIL_CONJECTURE_COMPRESS,
    "wienerfilter": CompressionMethod.WIENER_FILTER_COMPRESS,
    "zadorbound": CompressionMethod.ZADOR_BOUND,
    "zetafunction": CompressionMethod.ZETA_FUNCTION_COMPRESS,
}


def _find_enum_for_name(name: str) -> CompressionMethod | None:
    """Find the CompressionMethod enum value for a given method name."""
    # 1. Check explicit mapping first
    if name in _NAME_TO_ENUM:
        return _NAME_TO_ENUM[name]

    # 2. Try uppercase match (triggers _missing_() for auto-creation)
    enum_name = name.upper().replace("-", "_")
    try:
        return CompressionMethod[enum_name]
    except KeyError:
        pass

    # 3. Try lowercase comparison
    for member in CompressionMethod:
        mname = member.name.lower().lstrip("_")
        tname = name.lower().replace("-", "_")
        if mname == tname:
            return member

    # 4. Stripped-name matching: remove common suffixes and recompare
    suffixes = (
        "_compress",
        "_coder",
        "_codec",
        "_quant",
        "_pruning",
        "_transform",
        "_decomposition",
        "_compression",
        "_engine",
        "_encoder",
        "_decoder",
    )
    base = name.lower()
    for suffix in suffixes:
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break

    for member in _STRIPPED_ENUM_CACHE:
        if _STRIPPED_ENUM_CACHE[member] == base:
            return member

    # 5. CamelCase → snake_case conversion for flattened names
    snake = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    snake = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", snake).lower().replace("-", "_")
    if snake != name.lower():
        try:
            enum_name = snake.upper()
            return CompressionMethod[enum_name]
        except KeyError:
            pass
        # Strip suffixes from snake_case version too
        for suffix in suffixes:
            if snake.endswith(suffix):
                snake_base = snake[: -len(suffix)]
                break
        else:
            snake_base = snake
        for member in _STRIPPED_ENUM_CACHE:
            if _STRIPPED_ENUM_CACHE[member] == snake_base:
                return member

    # 6. Try direct _missing_ hook as final fallback (auto-creates member)
    result = CompressionMethod._missing_(name)
    if result is not None and isinstance(result, CompressionMethod):
        cached = getattr(result, "_name_", None)
        if cached == name.upper().replace("-", "_"):
            return result
        # Only return if it's a legit auto-created member
        if result.value >= 10000:
            return result

    return None


# Precomputed stripped enum names for fast fuzzy matching
_STRIPPED_ENUM_CACHE: dict = {}
_suffixes_tuple = (
    "_compress",
    "_coder",
    "_codec",
    "_quant",
    "_pruning",
    "_transform",
    "_decomposition",
    "_compression",
    "_engine",
    "_encoder",
    "_decoder",
)
for member in CompressionMethod:
    mname = member.name.lower().lstrip("_")
    for suffix in _suffixes_tuple:
        if mname.endswith(suffix):
            mname = mname[: -len(suffix)]
            break
    _STRIPPED_ENUM_CACHE[member] = mname
