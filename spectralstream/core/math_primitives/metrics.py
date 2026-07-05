"""Comprehensive compression quality metrics — single authoritative source.

Consolidates metric implementations from:
  - core/math_primitives/metrics.py    (original MSE, SNR, PSNR, rel error)
  - compression/engine/_helpers.py     (_compute_metrics)
  - compression/cutting_edge/_compressionmethod.py  (estimate_error)
  - compression/unified_quantizer.py   (get_quality_metrics)
  - core/validation.py                 (advanced metrics, now DEPRECATED)
  - kv_cache/core.py                   (QualityMetrics dataclass)

Every function uses NumPy vectorized operations, handles shape mismatch,
and follows a consistent convention for edge cases (epsilon protection).
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np


# ── Internal helpers ───────────────────────────────────────────────────


def _sanitize(*arrays: np.ndarray) -> List[np.ndarray]:
    """Flatten, convert to float64, match to smallest size, and replace non-finite values."""
    result = []
    for a in arrays:
        arr = np.asarray(a, dtype=np.float64).ravel()
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        result.append(arr)
    if len(result) > 1:
        min_len = min(len(r) for r in result)
        result = [r[:min_len] for r in result]
    return result


def _match_shape(
    original: np.ndarray, reconstructed: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Ensure both arrays have the same number of elements (trim if needed)."""
    n = min(original.size, reconstructed.size)
    return original.ravel()[:n], reconstructed.ravel()[:n]


# ── Primary error metrics ──────────────────────────────────────────────


def compute_mse(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Mean squared error between original and reconstructed.

    Parameters
    ----------
    original : np.ndarray
        Original (reference) tensor.
    reconstructed : np.ndarray
        Reconstructed (compressed/decompressed) tensor.

    Returns
    -------
    float
        Mean squared error. Lower is better. 0 = perfect reconstruction.
    """
    o, r = _match_shape(original, reconstructed)
    o = o.astype(np.float64)
    r = r.astype(np.float64)
    return float(np.mean((o - r) ** 2))


def compute_rmse(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Root mean squared error — interpretable in original units.

    Parameters
    ----------
    original : np.ndarray
        Original tensor.
    reconstructed : np.ndarray
        Reconstructed tensor.

    Returns
    -------
    float
        RMSE = sqrt(MSE). Lower is better.
    """
    return float(math.sqrt(compute_mse(original, reconstructed)))


def compute_mae(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Mean absolute error — robust to outliers, in original units.

    Parameters
    ----------
    original : np.ndarray
        Original tensor.
    reconstructed : np.ndarray
        Reconstructed tensor.

    Returns
    -------
    float
        Mean absolute error. Lower is better.
    """
    o, r = _match_shape(original, reconstructed)
    o = o.astype(np.float64)
    r = r.astype(np.float64)
    return float(np.mean(np.abs(o - r)))


def compute_nmse(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Normalized mean squared error — MSE divided by variance of original.

    NMSE = MSE / var(original). 1.0 means error ≈ signal variance (poor).
    0.0 = perfect reconstruction. > 1 = worse than guessing the mean.

    Parameters
    ----------
    original : np.ndarray
        Original tensor.
    reconstructed : np.ndarray
        Reconstructed tensor.

    Returns
    -------
    float
        Normalized MSE (dimensionless). Lower is better.
    """
    o, r = _match_shape(original, reconstructed)
    o = o.astype(np.float64)
    r = r.astype(np.float64)
    mse = float(np.mean((o - r) ** 2))
    var_o = float(np.var(o)) + 1e-30
    return float(mse / var_o)


# ── Signal-quality metrics ─────────────────────────────────────────────


def compute_snr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Signal-to-noise ratio in decibels.

    SNR_db = 10 * log10(signal_power / noise_power).
    Higher is better. Typical range: 0-100 dB.

    Parameters
    ----------
    original : np.ndarray
        Original tensor.
    reconstructed : np.ndarray
        Reconstructed tensor.

    Returns
    -------
    float
        SNR in dB. Higher is better. +inf if noise is zero.
    """
    o, r = _match_shape(original, reconstructed)
    o = o.astype(np.float64)
    r = r.astype(np.float64)
    signal = float(np.mean(o**2))
    noise = float(np.mean((o - r) ** 2))
    if noise < 1e-30:
        return float("inf")
    return 10.0 * math.log10(signal / noise)


def compute_psnr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Peak signal-to-noise ratio in decibels.

    PSNR_db = 10 * log10(max_val² / mse). Higher is better.
    Uses the 10*log10 convention (equivalent to 20*log10(peak/sqrt(mse))).

    Parameters
    ----------
    original : np.ndarray
        Original tensor.
    reconstructed : np.ndarray
        Reconstructed tensor.

    Returns
    -------
    float
        PSNR in dB. Higher is better. +inf if MSE is zero.
    """
    o, r = _match_shape(original, reconstructed)
    o = o.astype(np.float64)
    r = r.astype(np.float64)
    mse = float(np.mean((o - r) ** 2))
    max_val = float(np.max(np.abs(o)))
    if max_val < 1e-30:
        return float("inf") if mse < 1e-30 else 0.0
    if mse < 1e-30:
        return float("inf")
    return 10.0 * math.log10(max_val**2 / mse)


# ── Error-norm metrics ─────────────────────────────────────────────────


def compute_relative_error(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Relative L2 error: ||o - r|| / ||o||.

    Also known as normalized RMSE (NRMSE). 0 = perfect, 1.0 = error as large
    as the signal itself, > 1 = catastrophic.

    Parameters
    ----------
    original : np.ndarray
        Original tensor.
    reconstructed : np.ndarray
        Reconstructed tensor.

    Returns
    -------
    float
        Relative L2 error (dimensionless). Lower is better.
    """
    o, r = _match_shape(original, reconstructed)
    o = o.astype(np.float64)
    r = r.astype(np.float64)
    o_norm = float(np.linalg.norm(o))
    if o_norm < 1e-30:
        return 0.0
    return float(np.linalg.norm(o - r) / o_norm)


def compute_cosine_similarity(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Cosine similarity between original and reconstructed vectors.

    Range: [-1, 1]. 1.0 = identical direction (perfect), 0 = orthogonal,
    -1 = opposite direction.

    Parameters
    ----------
    original : np.ndarray
        Original tensor.
    reconstructed : np.ndarray
        Reconstructed tensor.

    Returns
    -------
    float
        Cosine similarity. Higher is better. Clamped to [-1, 1].
    """
    o, r = _match_shape(original, reconstructed)
    o = o.astype(np.float64)
    r = r.astype(np.float64)
    dot = float(np.dot(o, r))
    norm_o = float(np.linalg.norm(o))
    norm_r = float(np.linalg.norm(r))
    if norm_o < 1e-30 or norm_r < 1e-30:
        return 1.0 if norm_o == norm_r else 0.0
    return float(np.clip(dot / (norm_o * norm_r), -1.0, 1.0))


def compute_max_abs_error(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Maximum absolute error (Chebyshev / L-infinity norm).

    Worst-case per-element error. Useful for detecting outliers.

    Parameters
    ----------
    original : np.ndarray
        Original tensor.
    reconstructed : np.ndarray
        Reconstructed tensor.

    Returns
    -------
    float
        Max |original - reconstructed|. Lower is better.
    """
    o, r = _match_shape(original, reconstructed)
    o = o.astype(np.float64)
    r = r.astype(np.float64)
    return float(np.max(np.abs(o - r)))


# ── Structural similarity ──────────────────────────────────────────────


def _block_ssim(
    block_o: np.ndarray, block_r: np.ndarray, C1: float, C2: float
) -> float:
    """SSIM for a single block / full image."""
    mu_o = float(np.mean(block_o))
    mu_r = float(np.mean(block_r))
    sigma_o_sq = float(np.var(block_o))
    sigma_r_sq = float(np.var(block_r))
    sigma_or = float(np.mean((block_o - mu_o) * (block_r - mu_r)))
    numerator = (2.0 * mu_o * mu_r + C1) * (2.0 * sigma_or + C2)
    denominator = (mu_o**2 + mu_r**2 + C1) * (sigma_o_sq + sigma_r_sq + C2)
    return numerator / (denominator + 1e-15)


def compute_ssim(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Structural similarity index (simplified block-based).

    Uses 11×11 non-overlapping blocks with K1=0.01, K2=0.03.
    For non-2D tensors, the array is flattened and reshaped to a square.
    For tensors smaller than 11×11, falls back to global SSIM.

    Range: [-1, 1]. 1.0 = identical structure.

    Parameters
    ----------
    original : np.ndarray
        Original tensor.
    reconstructed : np.ndarray
        Reconstructed tensor.

    Returns
    -------
    float
        Mean SSIM across all blocks. Higher is better.
    """
    o = original.astype(np.float64)
    r = reconstructed.astype(np.float64)

    # Match elements first
    n_elems = min(o.size, r.size)
    o = o.ravel()[:n_elems]
    r = r.ravel()[:n_elems]

    # Reshape to 2D square
    side = int(np.ceil(np.sqrt(n_elems)))
    o = np.resize(o, (side, side))
    r = np.resize(r, (side, side))

    K1, K2 = 0.01, 0.03
    window_size = 11
    data_range = float(np.max(o) - np.min(o))
    if data_range < 1e-10:
        data_range = 1.0
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    m = n = side
    blocks_m = m // window_size
    blocks_n = n // window_size

    if blocks_m == 0 or blocks_n == 0:
        return _block_ssim(o, r, C1, C2)

    ssim_sum = 0.0
    count = 0
    for i in range(blocks_m):
        for j in range(blocks_n):
            bo = o[
                i * window_size : (i + 1) * window_size,
                j * window_size : (j + 1) * window_size,
            ]
            br = r[
                i * window_size : (i + 1) * window_size,
                j * window_size : (j + 1) * window_size,
            ]
            ssim_sum += _block_ssim(bo, br, C1, C2)
            count += 1

    return float(ssim_sum / max(count, 1))


# ── Distributional metrics ─────────────────────────────────────────────


def compute_spectral_angle(
    original: np.ndarray, reconstructed: np.ndarray, max_coeffs: int = 4096
) -> float:
    """Spectral angle between original and reconstructed in DCT domain.

    Uses subsampling to prevent OOM: DCT of full flattened tensor creates an
    N×N matrix (512 GiB for 512×512). Limited to *max_coeffs* coefficients.

    Parameters
    ----------
    original : np.ndarray
        Original tensor.
    reconstructed : np.ndarray
        Reconstructed tensor.
    max_coeffs : int
        Max DCT coefficients (prevents OOM). Default 4096.

    Returns
    -------
    float
        Spectral angle in radians [0, pi/2]. Lower is better.
    """
    from .numerical import cosine_similarity as _cos_sim

    o, r = _sanitize(original, reconstructed)
    # Subsample to prevent OOM from full DCT matrix
    if len(o) > max_coeffs:
        idx = np.linspace(0, len(o) - 1, max_coeffs, dtype=np.int64)
        o = o[idx]
        r = r[idx]
    # Use block-based DCT via dct_2d on reshaped chunks
    chunk_size = 2048
    n_chunks = max(1, len(o) // chunk_size)
    dct_o_parts = []
    dct_r_parts = []
    for i in range(n_chunks):
        chunk_o = o[i * chunk_size : (i + 1) * chunk_size]
        chunk_r = r[i * chunk_size : (i + 1) * chunk_size]
        # Pad to square for dct_2d
        side = int(np.ceil(np.sqrt(len(chunk_o))))
        pad_o = np.pad(chunk_o, (0, side * side - len(chunk_o)))
        pad_r = np.pad(chunk_r, (0, side * side - len(chunk_r)))
        from .transforms import dct_2d as _dct_2d

        dct_o_parts.append(_dct_2d(pad_o.reshape(side, side)).ravel())
        dct_r_parts.append(_dct_2d(pad_r.reshape(side, side)).ravel())
    dct_o = np.concatenate(dct_o_parts)[:max_coeffs]
    dct_r = np.concatenate(dct_r_parts)[:max_coeffs]
    sim = _cos_sim(dct_o, dct_r)
    sim = float(np.clip(sim, -1.0, 1.0))
    return float(math.acos(sim))


def compute_histogram_overlap(
    original: np.ndarray, reconstructed: np.ndarray, bins: int = 100
) -> float:
    """Histogram intersection overlap between two distributions.

    Range: [0, 1]. 1.0 = identical value distributions.

    Parameters
    ----------
    original : np.ndarray
        Original tensor.
    reconstructed : np.ndarray
        Reconstructed tensor.
    bins : int
        Number of histogram bins (default 100).

    Returns
    -------
    float
        Histogram overlap score [0, 1]. Higher is better.
    """
    o, r = _sanitize(original, reconstructed)
    combined = np.concatenate([o, r])
    lo, hi = float(np.min(combined)), float(np.max(combined))
    if hi - lo < 1e-30:
        return 1.0
    n = len(o)
    actual_bins = min(bins, max(2, n // 2))
    h1, _ = np.histogram(o, bins=actual_bins, range=(lo, hi), density=False)
    h2, _ = np.histogram(r, bins=actual_bins, range=(lo, hi), density=False)
    intersection = float(np.sum(np.minimum(h1, h2)))
    denom = float(max(np.sum(h1), np.sum(h2)))
    return intersection / denom if denom > 0 else 1.0


def compute_kld(
    original: np.ndarray, reconstructed: np.ndarray, bins: int = 100, eps: float = 1e-10
) -> float:
    """Kullback-Leibler divergence D_KL(P||Q).

    Measures information loss when using reconstructed distribution to
    approximate original. 0 = identical distributions.

    Parameters
    ----------
    original : np.ndarray
        Original tensor (reference distribution P).
    reconstructed : np.ndarray
        Reconstructed tensor (approximate distribution Q).
    bins : int
        Number of histogram bins (default 100).
    eps : float
        Smoothing epsilon (default 1e-10).

    Returns
    -------
    float
        KL divergence in nats. Lower is better. 0 = identical.
    """
    o, r = _sanitize(original, reconstructed)
    combined = np.concatenate([o, r])
    lo, hi = float(np.min(combined)), float(np.max(combined))
    if hi - lo < 1e-30:
        return 0.0
    n = len(o)
    actual_bins = min(bins, max(2, n // 2))
    h_p, _ = np.histogram(o, bins=actual_bins, range=(lo, hi), density=True)
    h_q, _ = np.histogram(r, bins=actual_bins, range=(lo, hi), density=True)
    p = np.clip(h_p, eps, None)
    q = np.clip(h_q, eps, None)
    p = p / np.sum(p)
    q = q / np.sum(q)
    return float(np.sum(p * np.log(p / q)))


def compute_wasserstein_distance(
    original: np.ndarray, reconstructed: np.ndarray
) -> float:
    """Wasserstein-1 distance (Earth Mover's Distance).

    Measures the minimum cost to transform one distribution into another.
    0 = identical distributions. Higher = more different.

    Uses sorted-array approximation for 1D: W = mean(|sorted(o) - sorted(r)|).

    Parameters
    ----------
    original : np.ndarray
        Original tensor.
    reconstructed : np.ndarray
        Reconstructed tensor.

    Returns
    -------
    float
        Wasserstein-1 distance. Lower is better. 0 = identical.
    """
    o, r = _sanitize(original, reconstructed)
    o.sort()
    r.sort()
    n = min(len(o), len(r))
    if n < 2:
        return 0.0
    o_resampled = np.interp(
        np.linspace(0, 1, n),
        np.linspace(0, 1, len(o)),
        o,
    )
    r_resampled = np.interp(
        np.linspace(0, 1, n),
        np.linspace(0, 1, len(r)),
        r,
    )
    return float(np.mean(np.abs(o_resampled - r_resampled)))


def compute_kolmogorov_smirnov(
    original: np.ndarray, reconstructed: np.ndarray
) -> Tuple[float, float]:
    """Kolmogorov-Smirnov test: statistic and approximate p-value.

    KS statistic measures the maximum vertical distance between two
    empirical CDFs. p-value estimates the probability that they are
    drawn from the same distribution.

    Parameters
    ----------
    original : np.ndarray
        Original tensor.
    reconstructed : np.ndarray
        Reconstructed tensor.

    Returns
    -------
    Tuple[float, float]
        (KS statistic [0, 1], approximate p-value [0, 1]).
        Lower KS statistic = more similar distributions.
        Higher p-value = more likely same distribution.
    """
    o, r = _sanitize(original, reconstructed)
    if len(o) < 2 or len(r) < 2:
        return 1.0, 0.0
    combined = np.concatenate([o, r])
    combined.sort()
    n1, n2 = len(o), len(r)
    cdf1 = np.searchsorted(o, combined, side="right") / n1
    cdf2 = np.searchsorted(r, combined, side="right") / n2
    d_stat = float(np.max(np.abs(cdf1 - cdf2)))
    ne = n1 * n2 / (n1 + n2)
    try:
        p_value = 2.0 * math.exp(-2.0 * d_stat**2 * ne)
    except (OverflowError, ValueError):
        p_value = 0.0
    return d_stat, float(min(max(p_value, 0.0), 1.0))


def compute_correlation_coefficient(
    original: np.ndarray, reconstructed: np.ndarray
) -> float:
    """Pearson correlation coefficient between original and reconstructed.

    Range: [-1, 1]. 1.0 = perfect linear correlation, 0 = no correlation,
    -1 = perfect inverse correlation.

    Parameters
    ----------
    original : np.ndarray
        Original tensor.
    reconstructed : np.ndarray
        Reconstructed tensor.

    Returns
    -------
    float
        Pearson correlation coefficient. Higher absolute value = stronger
        linear relationship. Clamped to [-1, 1].
    """
    o, r = _sanitize(original, reconstructed)
    if len(o) < 2:
        return 1.0
    o_mean = o - np.mean(o)
    r_mean = r - np.mean(r)
    num = float(np.dot(o_mean, r_mean))
    denom = float(np.linalg.norm(o_mean) * np.linalg.norm(r_mean))
    if denom < 1e-30:
        return 0.0
    return float(np.clip(num / denom, -1.0, 1.0))


def compute_effective_rank_ratio(
    original: np.ndarray, reconstructed: np.ndarray
) -> float:
    """Ratio of effective rank preserved after compression.

    Measures how well the spectral complexity (effective rank) of the
    original tensor is maintained. 1.0 = full rank preservation, 0 = none.

    Reshapes to 2D for SVD-based effective rank computation.

    Parameters
    ----------
    original : np.ndarray
        Original tensor.
    reconstructed : np.ndarray
        Reconstructed tensor.

    Returns
    -------
    float
        Effective rank ratio [0, 1]. Higher is better.
    """
    from .transforms import effective_rank as _eff_rank

    o = np.asarray(original, dtype=np.float64)
    r = np.asarray(reconstructed, dtype=np.float64)

    n_elems = min(o.size, r.size)
    o = o.ravel()[:n_elems]
    r = r.ravel()[:n_elems]

    side = int(np.ceil(np.sqrt(n_elems)))
    o = np.resize(o, (side, side))
    r = np.resize(r, (side, side))

    n = min(side, 256)
    o_2d = o[:n, :n]
    r_2d = r[:n, :n]

    orig_rank = _eff_rank(o_2d)
    if orig_rank <= 1e-10:
        return 1.0
    recon_rank = _eff_rank(r_2d)
    return float(min(recon_rank / orig_rank, 1.0))


# ── Bit-level metrics ──────────────────────────────────────────────────


def compute_bit_error_rate(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Bit error rate (BER) — fraction of bits that differ.

    Converts both tensors to binary representation and computes the ratio
    of mismatched bits. Useful for assessing bit-exact reconstruction.

    Parameters
    ----------
    original : np.ndarray
        Original tensor.
    reconstructed : np.ndarray
        Reconstructed tensor.

    Returns
    -------
    float
        Bit error rate [0, 1]. 0 = bit-exact match, 0.5 = random.
    """
    n = min(original.size, reconstructed.size)
    o = original.ravel()[:n]
    r = reconstructed.ravel()[:n]

    if o.dtype == r.dtype:
        bits_o = np.unpackbits(o.view(np.uint8))
        bits_r = np.unpackbits(r.view(np.uint8))
    else:
        o_bytes = o.astype(np.float64).view(np.uint8)
        r_bytes = r.astype(np.float64).view(np.uint8)
        bits_o = np.unpackbits(o_bytes)
        bits_r = np.unpackbits(r_bytes)

    n_bits = min(len(bits_o), len(bits_r))
    if n_bits == 0:
        return 0.0
    return float(np.sum(bits_o[:n_bits] != bits_r[:n_bits]) / n_bits)


# ── Batch / consolidation ──────────────────────────────────────────────


def compute_all_metrics(
    original: np.ndarray, reconstructed: np.ndarray
) -> Dict[str, float]:
    """Compute ALL supported metrics in a single call.

    Parameters
    ----------
    original : np.ndarray
        Original tensor.
    reconstructed : np.ndarray
        Reconstructed tensor.

    Returns
    -------
    Dict[str, float]
        Dictionary with every metric name → value.
    """
    ks_stat, ks_p = compute_kolmogorov_smirnov(original, reconstructed)
    return {
        "mse": compute_mse(original, reconstructed),
        "rmse": compute_rmse(original, reconstructed),
        "mae": compute_mae(original, reconstructed),
        "nmse": compute_nmse(original, reconstructed),
        "snr_db": compute_snr(original, reconstructed),
        "psnr_db": compute_psnr(original, reconstructed),
        "relative_error": compute_relative_error(original, reconstructed),
        "cosine_similarity": compute_cosine_similarity(original, reconstructed),
        "max_abs_error": compute_max_abs_error(original, reconstructed),
        "ssim": compute_ssim(original, reconstructed),
        "spectral_angle": compute_spectral_angle(original, reconstructed),
        "histogram_overlap": compute_histogram_overlap(original, reconstructed),
        "kld_divergence": compute_kld(original, reconstructed),
        "wasserstein_distance": compute_wasserstein_distance(original, reconstructed),
        "ks_statistic": ks_stat,
        "ks_p_value": ks_p,
        "correlation_coefficient": compute_correlation_coefficient(
            original, reconstructed
        ),
        "effective_rank_ratio": compute_effective_rank_ratio(original, reconstructed),
        "bit_error_rate": compute_bit_error_rate(original, reconstructed),
    }


# ── Legacy alias (backward compatible) ─────────────────────────────────


def compression_quality(
    original: np.ndarray,
    reconstructed: np.ndarray,
    original_nbytes: int,
    compressed_nbytes: int,
) -> Dict[str, float]:
    """Legacy compression quality wrapper: metrics + compression ratio.

    Parameters
    ----------
    original : np.ndarray
        Original tensor.
    reconstructed : np.ndarray
        Reconstructed tensor.
    original_nbytes : int
        Original byte count.
    compressed_nbytes : int
        Compressed byte count.

    Returns
    -------
    Dict[str, float]
        Metrics dict with 'mse', 'snr_db', 'psnr_db', 'relative_error',
        'cosine_similarity', and 'compression_ratio'.
    """
    metrics = compute_all_metrics(original, reconstructed)
    metrics["compression_ratio"] = original_nbytes / max(compressed_nbytes, 1)
    return {
        "mse": metrics["mse"],
        "snr_db": metrics["snr_db"],
        "psnr_db": metrics["psnr_db"],
        "relative_error": metrics["relative_error"],
        "cosine_similarity": metrics["cosine_similarity"],
        "compression_ratio": metrics.get("compression_ratio", 1.0),
    }
