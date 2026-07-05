"""Module extracted from quantization_tuning.py — tunenf4."""

from __future__ import annotations

import math
from typing import Optional, Tuple

from . import FP32_BYTES, TunedParams


def _block_quant_bytes_per_elem(bits: int, block_size: int) -> float:
    """Average bytes per element for uniform block quantization.
    Overhead: one float32 scale per block => 4/block_size bytes/element.
    """
    return bits / 8.0 + 4.0 / block_size


def _block_quant_ratio(bits: int, block_size: int) -> float:
    """Compression ratio for block quantization (float32 input => quantized)."""
    bpe = _block_quant_bytes_per_elem(bits, block_size)
    return FP32_BYTES / bpe if bpe > 0 else float("inf")


def _uniform_quant_mse(bits: int, sigma_sq: float = 1.0) -> float:
    """High-rate uniform scalar quantisation MSE: Delta^2/12.
    Delta = 2 * range / 2^bits, range approx= 6 * sqrt(sigma^2).
    """
    sigma = math.sqrt(max(sigma_sq, 1e-30))
    delta = 6.0 * sigma / (1 << bits)
    return delta * delta / 12.0


def _search_block_quant(
    target_ratio: float,
    n_elements: int,
    bits_candidates: Tuple[int, ...] = (2, 3, 4, 5, 6, 8),
    block_size_candidates: Tuple[int, ...] = (16, 32, 64, 128, 256, 512),
    sigma_sq: float = 1.0,
) -> TunedParams:
    """Search (bits, block_size) for uniform block quantisation. Always returns a result."""
    best: Optional[TunedParams] = None
    best_err = float("inf")
    for bits in bits_candidates:
        for bs in block_size_candidates:
            if bs > n_elements:
                continue
            ratio = _block_quant_ratio(bits, bs)
            mse = _uniform_quant_mse(bits, sigma_sq)
            err = abs(ratio - target_ratio)
            if err < best_err:
                best_err = err
                best = TunedParams(
                    method_name="block_quant",
                    params={"bits": bits, "block_size": bs},
                    estimated_ratio=ratio,
                    estimated_mse=mse,
                )
    if best is not None:
        return best
    bs = max(16, 1 << int(math.floor(math.log2(n_elements))))
    ratio = _block_quant_ratio(bits_candidates[0], bs)
    mse = _uniform_quant_mse(bits_candidates[0], sigma_sq)
    return TunedParams(
        method_name="block_quant",
        params={"bits": int(bits_candidates[0]), "block_size": bs},
        estimated_ratio=ratio,
        estimated_mse=mse,
    )


def _exact_block_quant_params(
    target_ratio: float,
    bits: int,
    n_elements: int,
) -> TunedParams:
    """Solve block_size analytically for a fixed bit-width. Always returns a result."""
    if target_ratio <= 0:
        target_ratio = 1.0
    denom = 4.0 / target_ratio - bits / 8.0
    if denom <= 0:
        bs = 16
    else:
        bs = int(round(4.0 / denom))
        bs = max(16, min(bs, 4096))
        bs = 1 << int(round(math.log2(bs)))
    if bs > n_elements:
        bs = 1 << int(math.floor(math.log2(n_elements)))
    bs = max(16, bs)
    ratio = _block_quant_ratio(bits, bs)
    mse = _uniform_quant_mse(bits)
    return TunedParams(
        method_name="block_quant",
        params={"bits": bits, "block_size": bs},
        estimated_ratio=ratio,
        estimated_mse=mse,
    )


def tune_block_int8(
    target_ratio: float, n_elements: int, sigma_sq: float = 1.0
) -> TunedParams:
    """Tune block_int8 to target ratio via block_size."""
    result = _exact_block_quant_params(target_ratio, 8, n_elements)
    result.method_name = "block_int8"
    return result


def tune_block_int4(
    target_ratio: float, n_elements: int, sigma_sq: float = 1.0
) -> TunedParams:
    """Tune block_int4 to target ratio via block_size."""
    result = _exact_block_quant_params(target_ratio, 4, n_elements)
    result.method_name = "block_int4"
    return result


def tune_hadamard_int8(
    target_ratio: float, n_elements: int, sigma_sq: float = 1.0
) -> TunedParams:
    """Tune hadamard_int8 to target ratio via block_size."""
    result = tune_block_int8(target_ratio, n_elements, sigma_sq)
    result.method_name = "hadamard_int8"
    return result


def tune_hadamard_int4(
    target_ratio: float, n_elements: int, sigma_sq: float = 1.0
) -> TunedParams:
    """Tune hadamard_int4 to target ratio via block_size."""
    result = tune_block_int4(target_ratio, n_elements, sigma_sq)
    result.method_name = "hadamard_int4"
    return result


def tune_sparsity_int4(
    target_ratio: float, n_elements: int, sigma_sq: float = 1.0
) -> TunedParams:
    """Tune sparsity_int4: 50% sparsity x int4, adjusted via group_size."""
    best = min(
        (abs(4.0 / (0.5 + 4.0 / gs) - target_ratio), gs)
        for gs in (16, 32, 64, 128, 256)
    )
    group_size = best[1]
    ratio = 4.0 / (0.5 + 4.0 / group_size)
    mse = _uniform_quant_mse(4, sigma_sq) * 0.5
    return TunedParams(
        method_name="sparsity_int4",
        params={"group_size": group_size},
        estimated_ratio=ratio,
        estimated_mse=mse,
    )


def tune_delta_int4(
    target_ratio: float, n_elements: int, sigma_sq: float = 1.0
) -> TunedParams:
    """Tune delta_int4: same as block_int4 but delta encoding halves variance."""
    result = tune_block_int4(target_ratio, n_elements, sigma_sq * 0.5)
    result.method_name = "delta_int4"
    return result


def _bits_from_ratio(target_ratio: float, block_size: int = 64) -> int:
    """Find the smallest bit-width that approximately meets target_ratio."""
    for bits in (8, 6, 5, 4, 3, 2):
        if _block_quant_ratio(bits, block_size) >= target_ratio:
            return bits
    return 2


def tune_group_wise_quant(
    target_ratio: float, n_elements: int, sigma_sq: float = 1.0
) -> TunedParams:
    """Tune GroupWiseQuant (block_size, bits) to target ratio."""
    best = _search_block_quant(
        target_ratio,
        n_elements,
        bits_candidates=(2, 3, 4, 5, 6, 8),
        sigma_sq=sigma_sq,
    )
    best.method_name = "group_wise_quant"
    return best


def tune_asymmetric_quant(
    target_ratio: float, n_elements: int, sigma_sq: float = 1.0
) -> TunedParams:
    """Tune AsymmetricQuant (block_size, bits) to target ratio."""
    result = tune_group_wise_quant(target_ratio, n_elements, sigma_sq)
    result.method_name = "asymmetric_quant"
    return result


def tune_nf4(
    target_ratio: float, n_elements: int, sigma_sq: float = 1.0
) -> TunedParams:
    """Tune NF4 (block_size) to target ratio. Always 4-bit with block scaling."""
    result = _exact_block_quant_params(target_ratio, 4, n_elements)
    result.method_name = "nf4"
    result.estimated_mse = 0.003 * sigma_sq
    return result


def tune_binary_quant(
    target_ratio: float, n_elements: int, sigma_sq: float = 1.0
) -> TunedParams:
    """Tune BinaryQuant (block_size). 1-bit + scale."""
    result = _exact_block_quant_params(target_ratio, 1, n_elements)
    result.method_name = "binary_quant"
    result.estimated_mse = sigma_sq
    return result


def tune_ternary_quant(
    target_ratio: float, n_elements: int, sigma_sq: float = 1.0
) -> TunedParams:
    """Tune TernaryQuant (block_size). 2-bit equivalent + scale."""
    result = _exact_block_quant_params(target_ratio, 2, n_elements)
    result.method_name = "ternary_quant"
    result.estimated_mse = 0.5 * sigma_sq
    return result


def tune_bqq_binary_quadratic(
    target_ratio: float, n_elements: int, sigma_sq: float = 1.0
) -> TunedParams:
    """Tune BQQ (block_size). 1-bit normalized +/-1."""
    result = _exact_block_quant_params(target_ratio, 1, n_elements)
    result.method_name = "bqq_binary_quadratic"
    result.estimated_mse = sigma_sq
    return result


def tune_adaptive_group_quant(
    target_ratio: float, n_elements: int, sigma_sq: float = 1.0
) -> TunedParams:
    """Tune AdaptiveGroupQuant (bits, n_groups)."""
    best: Optional[TunedParams] = None
    best_err = float("inf")
    for bits in (2, 3, 4, 5, 6, 8):
        for n_groups in (2, 4, 8, 16, 32):
            if n_groups > n_elements:
                continue
            bpe = bits / 8.0 + n_groups / n_elements
            ratio = FP32_BYTES / bpe if bpe > 0 else float("inf")
            mse = _uniform_quant_mse(bits, sigma_sq)
            err = abs(ratio - target_ratio)
            if err < best_err:
                best_err = err
                best = TunedParams(
                    method_name="adaptive_group_quant",
                    params={"bits": bits, "n_groups": n_groups},
                    estimated_ratio=ratio,
                    estimated_mse=mse,
                )
    assert best is not None
    return best


def tune_outlier_aware_quant(
    target_ratio: float, n_elements: int, sigma_sq: float = 1.0
) -> TunedParams:
    """Tune OutlierAwareQuant (bits). 5% outliers at fp16, rest quantized."""
    best: Optional[TunedParams] = None
    best_err = float("inf")
    for bits in (2, 3, 4, 5, 6, 8):
        outlier_bytes = 0.05 * n_elements * 2
        inlier_bytes = 0.95 * n_elements * (bits / 8.0 + 4.0 / 128)
        bpe = (outlier_bytes + inlier_bytes) / n_elements
        ratio = FP32_BYTES / bpe
        mse = _uniform_quant_mse(bits, sigma_sq) * 0.95
        err = abs(ratio - target_ratio)
        if err < best_err:
            best_err = err
            best = TunedParams(
                method_name="outlier_aware_quant",
                params={"bits": bits, "outlier_threshold": 3.0},
                estimated_ratio=ratio,
                estimated_mse=mse,
            )
    assert best is not None
    return best


def tune_sensitivity_aware_quant(
    target_ratio: float, n_elements: int, sigma_sq: float = 1.0
) -> TunedParams:
    """Tune SensitivityAwareQuant (sensitivity). Variable bit-width 2-8 per block."""
    best: Optional[TunedParams] = None
    best_err = float("inf")
    for sens in (0.1, 0.25, 0.5, 0.75, 1.0):
        avg_bits = 2 + sens * (8 - 2)
        bs = 64
        bpe = avg_bits / 8.0 + 4.0 / bs
        ratio = FP32_BYTES / bpe
        mse = _uniform_quant_mse(int(avg_bits), sigma_sq)
        err = abs(ratio - target_ratio)
        if err < best_err:
            best_err = err
            best = TunedParams(
                method_name="sensitivity_aware_quant",
                params={"sensitivity": sens},
                estimated_ratio=ratio,
                estimated_mse=mse,
            )
    assert best is not None
    return best


def tune_hadamard_group_wise(
    target_ratio: float, n_elements: int, sigma_sq: float = 1.0
) -> TunedParams:
    """Tune HadamardGroupWise (block_size, bits)."""
    best = _search_block_quant(
        target_ratio,
        n_elements,
        bits_candidates=(2, 3, 4, 5, 6, 8),
        sigma_sq=sigma_sq,
    )
    best.method_name = "hadamard_group_wise"
    return best


def tune_stochastic_round(
    target_ratio: float, n_elements: int, sigma_sq: float = 1.0
) -> TunedParams:
    """Tune StochasticRound (bits). Block_size fixed at 256."""
    best: Optional[TunedParams] = None
    best_err = float("inf")
    for bits in (4, 5, 6, 8, 10, 12, 16):
        bs = 256
        bpe = bits / 8.0 + 4.0 / bs
        ratio = FP32_BYTES / bpe
        mse = _uniform_quant_mse(bits, sigma_sq)
        err = abs(ratio - target_ratio)
        if err < best_err:
            best_err = err
            best = TunedParams(
                method_name="stochastic_round",
                params={"bits": bits},
                estimated_ratio=ratio,
                estimated_mse=mse,
            )
    assert best is not None
    return best


def tune_residual_quant(
    target_ratio: float, n_elements: int, sigma_sq: float = 1.0
) -> TunedParams:
    """Tune ResidualQuant (n_stages, bits). Multi-stage cascade."""
    best: Optional[TunedParams] = None
    best_err = float("inf")
    for n_stages in (1, 2, 3, 4):
        bits = _bits_from_ratio(target_ratio * (n_stages**0.5), 64)
        bits = max(4, bits)
        bpe = n_stages * (bits / 8.0 + 4.0 / n_elements)
        ratio = FP32_BYTES / bpe
        mse = _uniform_quant_mse(bits, sigma_sq) / max(n_stages, 1)
        err = abs(ratio - target_ratio)
        if err < best_err:
            best_err = err
            best = TunedParams(
                method_name="residual_quant",
                params={"n_stages": n_stages, "bits": bits},
                estimated_ratio=ratio,
                estimated_mse=mse,
            )
    assert best is not None
    return best


def tune_error_feedback_quant(
    target_ratio: float, n_elements: int, sigma_sq: float = 1.0
) -> TunedParams:
    """Tune ErrorFeedbackQuant (block_size, bits) with error propagation."""
    best = _search_block_quant(
        target_ratio,
        n_elements,
        bits_candidates=(2, 3, 4, 5, 6, 8),
        sigma_sq=sigma_sq * 0.3,
    )
    best.method_name = "error_feedback_quant"
    return best


def tune_lloyd_max_quant(
    target_ratio: float, n_elements: int, sigma_sq: float = 1.0
) -> TunedParams:
    """Tune LloydMaxQuant (n_bits). Optimal scalar quantisation."""
    best: Optional[TunedParams] = None
    best_err = float("inf")
    for n_bits in (2, 3, 4, 5, 6, 8):
        n_levels = 1 << n_bits
        bpe = n_bits / 8.0 + n_levels * 4.0 / n_elements
        ratio = FP32_BYTES / bpe
        mse = _uniform_quant_mse(n_bits, sigma_sq) * 0.5
        err = abs(ratio - target_ratio)
        if err < best_err:
            best_err = err
            best = TunedParams(
                method_name="lloyd_max_quant",
                params={"n_bits": n_bits},
                estimated_ratio=ratio,
                estimated_mse=mse,
            )
    assert best is not None
    return best


def tune_octopus_quant(
    target_ratio: float, n_elements: int, sigma_sq: float = 1.0
) -> TunedParams:
    """Tune OctopusQuant (n_bits). Hadamard + Lloyd-Max."""
    result = tune_lloyd_max_quant(target_ratio, n_elements, sigma_sq)
    result.method_name = "octopus_quant"
    return result


def tune_squeezellm_nonuniform(
    target_ratio: float, n_elements: int, sigma_sq: float = 1.0
) -> TunedParams:
    """Tune SqueezeLLMNonuniform (n_bits). 0.5% outliers + non-uniform codebook."""
    best: Optional[TunedParams] = None
    best_err = float("inf")
    for n_bits in (2, 3, 4, 5, 6, 8):
        n_levels = 1 << n_bits
        outlier_frac = 0.005
        outlier_bytes = outlier_frac * n_elements * 2
        inlier_bytes = (1 - outlier_frac) * n_elements * (n_bits / 8.0)
        codebook = n_levels * 4.0 / n_elements
        bpe = (outlier_bytes + inlier_bytes) / n_elements + codebook
        ratio = FP32_BYTES / bpe
        mse = _uniform_quant_mse(n_bits, sigma_sq) * 0.95
        err = abs(ratio - target_ratio)
        if err < best_err:
            best_err = err
            best = TunedParams(
                method_name="squeezellm_nonuniform",
                params={"n_bits": n_bits},
                estimated_ratio=ratio,
                estimated_mse=mse,
            )
    assert best is not None
    return best


def tune_awq_quant(
    target_ratio: float, n_elements: int, sigma_sq: float = 1.0
) -> TunedParams:
    """Tune AWQQuant (bits, block_size). Activation-aware weight quantisation."""
    best = _search_block_quant(
        target_ratio,
        n_elements,
        bits_candidates=(2, 3, 4, 5, 6, 8),
        sigma_sq=sigma_sq * 0.8,
    )
    best.method_name = "awq_quant"
    return best


def tune_gptq_quant(
    target_ratio: float, n_elements: int, sigma_sq: float = 1.0
) -> TunedParams:
    """Tune GPTQQuant (bits, block_size). Column-wise error compensation."""
    best = _search_block_quant(
        target_ratio,
        n_elements,
        bits_candidates=(2, 3, 4, 5, 6, 8),
        sigma_sq=sigma_sq * 0.3,
    )
    best.method_name = "gptq_quant"
    return best


def tune_mixed_precision(
    target_ratio: float, n_elements: int, sigma_sq: float = 1.0
) -> TunedParams:
    """Tune MixedPrecision (block_size). Per-block adaptive bit-width (2/4/8)."""
    best: Optional[TunedParams] = None
    best_err = float("inf")
    for block_size in (16, 32, 64, 128, 256):
        if block_size > n_elements:
            continue
        avg_bits = 0.33 * 2 + 0.34 * 4 + 0.33 * 8
        bpe = avg_bits / 8.0 + 4.0 / block_size + 1.0 / (block_size * 4)
        ratio = FP32_BYTES / bpe
        mse = _uniform_quant_mse(4, sigma_sq) * 0.6
        err = abs(ratio - target_ratio)
        if err < best_err:
            best_err = err
            best = TunedParams(
                method_name="mixed_precision",
                params={
                    "block_size": block_size,
                    "low_var_bits": 2,
                    "med_var_bits": 4,
                    "high_var_bits": 8,
                },
                estimated_ratio=ratio,
                estimated_mse=mse,
            )
    assert best is not None
    return best


def tune_dynamic_bitwidth(
    target_ratio: float, n_elements: int, sigma_sq: float = 1.0
) -> TunedParams:
    """Tune DynamicBitwidth (block_size). 2/4/8 bit per block by variance."""
    result = tune_mixed_precision(target_ratio, n_elements, sigma_sq)
    result.method_name = "dynamic_bitwidth"
    return result


def tune_multi_bitwidth(
    target_ratio: float, n_elements: int, sigma_sq: float = 1.0
) -> TunedParams:
    """Tune MultiBitwidth. Best block size for int4."""
    result = _exact_block_quant_params(target_ratio, 4, n_elements)
    result.method_name = "multi_bitwidth"
    return result


def tune_mixed_bitwidth_quant(
    target_ratio: float, n_elements: int, sigma_sq: float = 1.0
) -> TunedParams:
    """Tune MixedBitwidthQuant (block_size). 1/2/4/8 bit per block."""
    best: Optional[TunedParams] = None
    best_err = float("inf")
    for block_size in (16, 32, 64, 128, 256):
        if block_size > n_elements:
            continue
        avg_bits = 0.25 * 1 + 0.25 * 2 + 0.25 * 4 + 0.25 * 8
        bpe = avg_bits / 8.0 + 4.0 / block_size
        ratio = FP32_BYTES / bpe
        mse = _uniform_quant_mse(4, sigma_sq) * 0.5
        err = abs(ratio - target_ratio)
        if err < best_err:
            best_err = err
            best = TunedParams(
                method_name="mixed_bitwidth_quant",
                params={"block_size": block_size},
                estimated_ratio=ratio,
                estimated_mse=mse,
            )
    assert best is not None
    return best


def tune_quantum_tunneling(
    target_ratio: float, n_elements: int, sigma_sq: float = 1.0
) -> TunedParams:
    """Tune QuantumTunneling (bits, block_size). Wraps NF4."""
    result = tune_nf4(target_ratio, n_elements, sigma_sq)
    result.method_name = "quantum_tunneling"
    return result


def tune_quantum_error_correction(
    target_ratio: float, n_elements: int, sigma_sq: float = 1.0
) -> TunedParams:
    """Tune QuantumErrorCorrection (n_bits). Uniform scalar quantisation."""
    best: Optional[TunedParams] = None
    best_err = float("inf")
    for n_bits in (2, 3, 4, 5, 6, 8):
        n_levels = 1 << n_bits
        bpe = n_bits / 8.0 + n_levels * 4.0 / n_elements
        ratio = FP32_BYTES / bpe
        mse = _uniform_quant_mse(n_bits, sigma_sq)
        err = abs(ratio - target_ratio)
        if err < best_err:
            best_err = err
            best = TunedParams(
                method_name="quantum_error_correction",
                params={"n_bits": n_bits},
                estimated_ratio=ratio,
                estimated_mse=mse,
            )
    assert best is not None
    return best
