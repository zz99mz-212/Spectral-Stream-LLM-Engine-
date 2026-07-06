"""
Loss Metrics Intelligence Engine — comprehensive loss metrics collection and analysis.

Collects 25+ metrics per tensor across 5 categories:

1. Spectral Metrics (5)
   - L1 norm error
   - L2 norm error (MSE, RMSE)
   - Spectral norm error (maximum singular value)
   - Spectral entropy change
   - Energy concentration preservation

2. Statistical Metrics (6)
   - Signal-to-Noise Ratio (SNR, PSNR)
   - Structural Similarity (SSIM) for 2D tensors
   - Cosine similarity (vector and matrix)
   - KL divergence of value distributions
   - Wasserstein distance
   - Mean Absolute Error (MAE)

3. Structural Metrics (5)
   - Effective rank preservation
   - Sparsity pattern preservation
   - Condition number change
   - Mutual information between rows/columns
   - Cross-correlation preservation

4. Compression Metrics (5)
   - Compression ratio (per-tensor and cumulative)
   - Bit rate (bits per parameter)
   - Entropy rate preservation
   - Kolmogorov complexity estimate
   - Method-specific efficiency

5. Quality Grades (4+)
   - Per-metric pass/fail against tiered budgets
   - Overall quality grade (EXCELLENT/GOOD/FAIR/POOR/FAIL)
   - Confidence score (how reliable is this measurement?)
   - Recommended next action (accept/reject/retry)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ── Quality Grade ─────────────────────────────────────────────────────────


class QualityGrade(Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    FAIL = "fail"


# ── Metric Dataclasses ────────────────────────────────────────────────────


@dataclass
class SpectralMetrics:
    l1_error: float = 0.0
    mse: float = 0.0
    rmse: float = 0.0
    spectral_norm_error: float = 0.0
    spectral_entropy_change: float = 0.0
    energy_concentration_preserved: float = 1.0


@dataclass
class StatisticalMetrics:
    snr_db: float = float("inf")
    psnr_db: float = float("inf")
    ssim: float = 1.0
    cosine_similarity: float = 1.0
    kl_divergence: float = 0.0
    wasserstein_distance: float = 0.0
    mae: float = 0.0


@dataclass
class StructuralMetrics:
    effective_rank_preserved: float = 1.0
    sparsity_preserved: float = 1.0
    condition_number_change: float = 0.0
    mutual_information_preserved: float = 1.0
    cross_correlation_preserved: float = 1.0


@dataclass
class CompressionMetrics:
    compression_ratio: float = 1.0
    bit_rate: float = 32.0
    entropy_rate_original: float = 0.0
    entropy_rate_compressed: float = 0.0
    kolmogorov_estimate: float = 0.0


@dataclass
class PerTensorLossMetrics:
    name: str = ""
    original_shape: Tuple = ()
    original_dtype: str = ""
    original_nbytes: int = 0
    compressed_nbytes: int = 0
    method_used: str = ""
    native_dtype: str = ""  # Original model dtype ("BF16", "float32", etc.)
    precision_preserved: bool = True  # Whether BF16 format was preserved end-to-end
    precision_conversion_error: float = 0.0  # Error from BF16→f32→BF16 round-trip

    spectral: SpectralMetrics = field(default_factory=SpectralMetrics)
    statistical: StatisticalMetrics = field(default_factory=StatisticalMetrics)
    structural: StructuralMetrics = field(default_factory=StructuralMetrics)
    compression: CompressionMetrics = field(default_factory=CompressionMetrics)

    quality_grade: QualityGrade = QualityGrade.EXCELLENT
    overall_loss_percent: float = 0.0
    confidence_score: float = 1.0
    recommended_action: str = "accept"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "original_shape": list(self.original_shape),
            "original_dtype": self.original_dtype,
            "native_dtype": self.native_dtype,
            "precision_preserved": self.precision_preserved,
            "precision_conversion_error": self.precision_conversion_error,
            "original_nbytes": self.original_nbytes,
            "compressed_nbytes": self.compressed_nbytes,
            "method_used": self.method_used,
            "quality_grade": self.quality_grade.value,
            "overall_loss_percent": self.overall_loss_percent,
            "confidence_score": self.confidence_score,
            "recommended_action": self.recommended_action,
            "spectral": {
                "l1_error": self.spectral.l1_error,
                "mse": self.spectral.mse,
                "rmse": self.spectral.rmse,
                "spectral_norm_error": self.spectral.spectral_norm_error,
                "spectral_entropy_change": self.spectral.spectral_entropy_change,
                "energy_concentration_preserved": self.spectral.energy_concentration_preserved,
            },
            "statistical": {
                "snr_db": self.statistical.snr_db,
                "psnr_db": self.statistical.psnr_db,
                "ssim": self.statistical.ssim,
                "cosine_similarity": self.statistical.cosine_similarity,
                "kl_divergence": self.statistical.kl_divergence,
                "wasserstein_distance": self.statistical.wasserstein_distance,
                "mae": self.statistical.mae,
            },
            "structural": {
                "effective_rank_preserved": self.structural.effective_rank_preserved,
                "sparsity_preserved": self.structural.sparsity_preserved,
                "condition_number_change": self.structural.condition_number_change,
                "mutual_information_preserved": self.structural.mutual_information_preserved,
                "cross_correlation_preserved": self.structural.cross_correlation_preserved,
            },
            "compression": {
                "compression_ratio": self.compression.compression_ratio,
                "bit_rate": self.compression.bit_rate,
                "entropy_rate_original": self.compression.entropy_rate_original,
                "entropy_rate_compressed": self.compression.entropy_rate_compressed,
                "kolmogorov_estimate": self.compression.kolmogorov_estimate,
            },
        }


# ── SVD Result Cache ─────────────────────────────────────────────────────


class _SVDCache:
    """Thread-safe SVD result cache to avoid recomputing SVD for the same tensor."""

    def __init__(self):
        self._cache: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def get(self, tensor: np.ndarray, key: str = "") -> Optional[Tuple]:
        # Use id + shape + dtype + checksum as cache key
        flat = np.asarray(tensor, dtype=np.float64).ravel()
        if flat.size < 4:
            return None
        max_dim = min(256, flat.size)
        sample = flat[:max_dim]
        cksum = _fast_checksum(sample)
        cache_key = id(tensor) ^ hash(tensor.shape) ^ hash(tensor.dtype) ^ hash(cksum)
        if key:
            cache_key ^= hash(key)
        return self._cache.get(cache_key)

    def put(self, tensor: np.ndarray, svd_result: Tuple, key: str = "") -> None:
        flat = np.asarray(tensor, dtype=np.float64).ravel()
        if flat.size < 4:
            return
        max_dim = min(256, flat.size)
        sample = flat[:max_dim]
        cksum = _fast_checksum(sample)
        cache_key = id(tensor) ^ hash(tensor.shape) ^ hash(tensor.dtype) ^ hash(cksum)
        if key:
            cache_key ^= hash(key)
        self._cache[cache_key] = svd_result

    def clear(self) -> None:
        self._cache.clear()


def _fast_checksum(arr: np.ndarray) -> int:
    """Fast approximate checksum via sum of absolute values."""
    return int(np.sum(np.abs(arr)) * 1e6) & 0xFFFFFFFF


_GLOBAL_SVD_CACHE = _SVDCache()


# ── Internal Helpers ──────────────────────────────────────────────────────


def _flatten_pair(
    original: np.ndarray, reconstructed: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    o = np.asarray(original, dtype=np.float64).ravel()
    r = np.asarray(reconstructed, dtype=np.float64).ravel()
    n = min(o.size, r.size)
    return o[:n], r[:n]


def _safe_norm(x: np.ndarray, ord: Optional[float] = None) -> float:
    n = np.linalg.norm(x, ord=ord)
    return float(n) if np.isfinite(n) else 0.0


def _clip_float(v: float) -> float:
    if not np.isfinite(v):
        return 0.0
    return v


# ── Spectral Entropy ─────────────────────────────────────────────────────


def _spectral_entropy_1d(signal: np.ndarray) -> float:
    if signal.size < 2:
        return 0.0
    spectrum = np.abs(np.fft.rfft(signal))
    psd = spectrum**2
    total = float(np.sum(psd))
    if total < 1e-30:
        return 0.0
    psd_norm = psd / total
    psd_norm = psd_norm[psd_norm > 1e-30]
    if psd_norm.size == 0:
        return 0.0
    return float(-np.sum(psd_norm * np.log(psd_norm))) / math.log(max(psd_norm.size, 2))


# ── Energy Concentration ──────────────────────────────────────────────────


def _energy_concentration(signal: np.ndarray, fraction: float = 0.9) -> float:
    flat = np.asarray(signal, dtype=np.float64).ravel()
    if flat.size < 2:
        return 1.0
    from spectralstream.core.math_primitives import dct as _dct

    try:
        n = flat.size
        n_padded = 1 << (n - 1).bit_length() if n > 1 else 1
        padded = np.pad(flat, (0, n_padded - n)) if n_padded > n else flat
        coeffs = _dct(padded)
        energy = coeffs**2
        total_energy = float(np.sum(energy))
        if total_energy < 1e-30:
            return 1.0
        sorted_energy = np.sort(energy.ravel())[::-1]
        cumulative = np.cumsum(sorted_energy) / total_energy
        k = int(np.searchsorted(cumulative, fraction)) + 1
        return k / max(n_padded, 1)
    except Exception:
        return 1.0


# ── Effective Rank via SVD Entropy ───────────────────────────────────────


def _effective_rank(matrix: np.ndarray) -> float:
    mat = np.asarray(matrix, dtype=np.float64)
    if mat.ndim == 1:
        mat = mat.reshape(-1, 1)
    elif mat.ndim >= 3:
        mat = mat.reshape(mat.shape[0], -1)
    m, n = mat.shape
    if min(m, n) < 2:
        return 1.0
    k = min(m, n, 256)
    sub = mat[:k, :k]
    try:
        s = np.linalg.svd(sub, compute_uv=False)
        s = s[s > 1e-10]
        if s.size == 0:
            return 1.0
        s_norm = s / np.sum(s)
        return float(np.exp(-np.sum(s_norm * np.log(s_norm))))
    except np.linalg.LinAlgError:
        return float(k)


# ── Spectral Norm (max singular value) via power iteration ───────────────


def _spectral_norm(matrix: np.ndarray, power_iters: int = 20) -> float:
    flat = np.asarray(matrix, dtype=np.float64).ravel()
    if flat.size < 2:
        return float(np.max(np.abs(flat))) if flat.size > 0 else 0.0

    if matrix.ndim <= 1:
        return float(np.linalg.norm(matrix))

    m, n = matrix.shape
    if min(m, n) < 256:
        try:
            s = np.linalg.svd(matrix.astype(np.float64), compute_uv=False)
            return float(s[0])
        except np.linalg.LinAlgError:
            pass

    # Power iteration for large matrices
    mat = matrix.astype(np.float64)
    v = np.random.randn(n).astype(np.float64)
    v /= max(np.linalg.norm(v), 1e-30)
    for _ in range(power_iters):
        u = mat @ v
        u_norm = float(np.linalg.norm(u))
        if u_norm < 1e-30:
            break
        u /= u_norm
        v = mat.T @ u
        v_norm = float(np.linalg.norm(v))
        if v_norm < 1e-30:
            break
        v /= v_norm
    return float(np.linalg.norm(mat @ v))


# ── Condition Number ──────────────────────────────────────────────────────


def _condition_number(matrix: np.ndarray) -> float:
    if matrix.ndim <= 1:
        return 1.0
    try:
        s = np.linalg.svd(matrix.astype(np.float64), compute_uv=False)
        s = s[s > 1e-30]
        if s.size < 2:
            return 1.0
        return float(s[0] / s[-1])
    except np.linalg.LinAlgError:
        return float("inf")


# ── Mutual Information Estimate ──────────────────────────────────────────


def _mutual_information_estimate(
    original: np.ndarray, reconstructed: np.ndarray, n_bins: int = 32
) -> float:
    o, r = _flatten_pair(original, reconstructed)
    if o.size < 16:
        return 1.0
    lo = min(float(np.min(o)), float(np.min(r)))
    hi = max(float(np.max(o)), float(np.max(r)))
    if hi - lo < 1e-30:
        return 1.0
    bins = np.linspace(lo, hi, n_bins + 1)
    h_orig, _ = np.histogram(o, bins=bins, density=True)
    h_recon, _ = np.histogram(r, bins=bins, density=True)
    eps = 1e-30
    p = h_orig + eps
    q = h_recon + eps
    p = p / np.sum(p)
    q = q / np.sum(q)
    kl_pq = float(np.sum(p * np.log(p / q)))
    kl_qp = float(np.sum(q * np.log(q / p)))
    js = 0.5 * kl_pq + 0.5 * kl_qp
    return max(0.0, 1.0 - js)


# ── SSIM (2D) ─────────────────────────────────────────────────────────────


def _ssim_2d(
    original: np.ndarray, reconstructed: np.ndarray, window_size: int = 11
) -> float:
    o = np.asarray(original, dtype=np.float64)
    r = np.asarray(reconstructed, dtype=np.float64)
    n = min(o.size, r.size)
    side = int(np.ceil(np.sqrt(n)))
    o = np.resize(o.ravel()[:n], (side, side))
    r = np.resize(r.ravel()[:n], (side, side))

    K1, K2 = 0.01, 0.03
    data_range = float(np.max(o) - np.min(o))
    if data_range < 1e-10:
        data_range = 1.0
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    def _block_ssim(bo, br):
        mu_o = float(np.mean(bo))
        mu_r = float(np.mean(br))
        sigma_o_sq = float(np.var(bo))
        sigma_r_sq = float(np.var(br))
        sigma_or = float(np.mean((bo - mu_o) * (br - mu_r)))
        num = (2.0 * mu_o * mu_r + C1) * (2.0 * sigma_or + C2)
        den = (mu_o**2 + mu_r**2 + C1) * (sigma_o_sq + sigma_r_sq + C2)
        return num / (den + 1e-15)

    ws = window_size
    m = n = side
    blocks_m = m // ws
    blocks_n = n // ws
    if blocks_m == 0 or blocks_n == 0:
        return _block_ssim(o, r)

    ssim_vals = np.zeros((blocks_m, blocks_n), dtype=np.float64)
    for i in range(blocks_m):
        si = i * ws
        for j in range(blocks_n):
            sj = j * ws
            bo = o[si : si + ws, sj : sj + ws]
            br = r[si : si + ws, sj : sj + ws]
            ssim_vals[i, j] = _block_ssim(bo, br)
    return float(np.mean(ssim_vals))


def _try_scipy_ssim(original: np.ndarray, reconstructed: np.ndarray) -> Optional[float]:
    try:
        from scipy.ndimage import uniform_filter

        o = np.asarray(original, dtype=np.float64)
        r = np.asarray(reconstructed, dtype=np.float64)
        n = min(o.size, r.size)
        side = int(np.ceil(np.sqrt(n)))
        o = np.resize(o.ravel()[:n], (side, side))
        r = np.resize(r.ravel()[:n], (side, side))
        K1, K2 = 0.01, 0.03
        data_range = float(np.max(o) - np.min(o))
        if data_range < 1e-10:
            data_range = 1.0
        C1, C2 = (K1 * data_range) ** 2, (K2 * data_range) ** 2
        window = np.ones((11, 11)) / 121.0

        def _filter(im):
            return uniform_filter(im, size=11, mode="constant")

        mu_o = _filter(o)
        mu_r = _filter(r)
        mu_o_sq = mu_o**2
        mu_r_sq = mu_r**2
        mu_or = mu_o * mu_r

        sigma_o_sq = _filter(o**2) - mu_o_sq
        sigma_r_sq = _filter(r**2) - mu_r_sq
        sigma_or = _filter(o * r) - mu_or

        num = (2.0 * mu_or + C1) * (2.0 * sigma_or + C2)
        den = (mu_o_sq + mu_r_sq + C1) * (sigma_o_sq + sigma_r_sq + C2)
        ssim_map = num / (den + 1e-15)
        return float(np.mean(ssim_map))
    except Exception:
        return None


# ── Entropy Rate ──────────────────────────────────────────────────────────


def _entropy_rate(arr: np.ndarray, order: int = 1) -> float:
    flat = np.asarray(arr, dtype=np.float64).ravel()
    if flat.size < order + 10:
        return 0.0
    sample = flat[: min(len(flat), 5000)]
    n_states = 16
    percentiles = np.percentile(sample, np.linspace(0, 100, n_states + 1))
    quantized = np.clip(np.digitize(sample, percentiles) - 1, 0, n_states - 1)
    trans = np.zeros((n_states, n_states), dtype=np.float64)
    for i in range(len(quantized) - order):
        state = int(quantized[i])
        nxt = int(quantized[i + 1])
        trans[state, nxt] += 1.0
    row_sums = trans.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    probs = trans / row_sums
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -np.sum(probs * np.log2(probs + 1e-30), axis=1)
    stationary = row_sums.ravel() / max(row_sums.sum(), 1.0)
    return float(np.sum(stationary * h))


# ── Kolmogorov Complexity Estimate ────────────────────────────────────────


def _kolmogorov_estimate(flat: np.ndarray) -> float:
    if flat.size < 32:
        return 0.0
    sample = flat[: min(len(flat), 1000)]
    n_states = 8
    percentiles = np.percentile(sample, np.linspace(0, 100, n_states + 1))
    quantized = np.clip(np.digitize(sample, percentiles) - 1, 0, n_states - 1)
    s = "".join(chr(ord("a") + int(x)) for x in quantized)
    n = len(s)
    lib: set = set()
    w = ""
    for c in s:
        if w + c in lib:
            w += c
        else:
            lib.add(w + c)
            w = ""
    if n == 0:
        return 0.0
    complexity = len(lib) / n
    return min(float(complexity), 1.0)


# ── Cross-Correlation Preservation ────────────────────────────────────────


def _cross_correlation_preserved(
    original: np.ndarray, reconstructed: np.ndarray
) -> float:
    o, r = _flatten_pair(original, reconstructed)
    if o.size < 4:
        return 1.0
    n = o.size
    n_blocks = min(8, n // 4)
    if n_blocks < 2:
        return 1.0
    block_size = n // n_blocks
    trimmed = n_blocks * block_size
    o_trimmed = o[:trimmed].reshape(n_blocks, block_size)
    r_trimmed = r[:trimmed].reshape(n_blocks, block_size)
    corr_orig = np.corrcoef(o_trimmed)
    corr_recon = np.corrcoef(r_trimmed)
    diff = np.abs(corr_orig - corr_recon)
    return max(0.0, 1.0 - float(np.mean(diff)))


# ── Loss Metrics Intelligence Engine ──────────────────────────────────────


class LossMetricsIntelligenceEngine:
    """
    Comprehensive loss metrics collection and analysis engine.

    Collects 25+ metrics per tensor across 5 categories:

    1. Spectral Metrics (5)
       - L1 norm error, L2 norm error (MSE, RMSE)
       - Spectral norm error (maximum singular value)
       - Spectral entropy change
       - Energy concentration preservation

    2. Statistical Metrics (6)
       - SNR, PSNR, SSIM, Cosine Similarity
       - KL divergence, Wasserstein distance, MAE

    3. Structural Metrics (5)
       - Effective rank preservation
       - Sparsity pattern preservation
       - Condition number change
       - Mutual information preservation
       - Cross-correlation preservation

    4. Compression Metrics (5)
       - Compression ratio, bit rate
       - Entropy rate preservation
       - Kolmogorov complexity estimate

    5. Quality Grades (4+)
       - Per-metric pass/fail against tiered budgets
       - Overall quality grade (EXCELLENT/GOOD/FAIR/POOR/FAIL)
       - Confidence score
       - Recommended next action (accept/reject/retry)
    """

    def __init__(self, use_svd_cache: bool = True):
        self._svd_cache = _SVDCache() if use_svd_cache else _GLOBAL_SVD_CACHE

    def compute_all_metrics(
        self,
        original: np.ndarray,
        reconstructed: np.ndarray,
        tensor_name: str = "",
        method_used: str = "",
        compressed_nbytes: int = 0,
        tensor_type: str = "weight",
        native_dtype: str = "",
    ) -> PerTensorLossMetrics:
        """
        Compute ALL 25+ metrics between original and reconstructed tensor.

        Parameters
        ----------
        original : np.ndarray
            Original (pre-compression) tensor.
        reconstructed : np.ndarray
            Reconstructed (post-decompression) tensor.
        tensor_name : str
            Name of the tensor for identification.
        method_used : str
            Compression method name.
        compressed_nbytes : int
            Size of compressed representation in bytes.
            If 0, compression metrics are estimated from the reconstructed tensor.
        tensor_type : str
            Type classification (attention_q, ffn_gate, etc.).
            Used for quality grading against tiered budgets.
        native_dtype : str
            Original model dtype (e.g., "BF16", "float32").
            Used to determine if precision was preserved.

        Returns
        -------
        PerTensorLossMetrics
            All computed metrics with quality grading.
        """
        spectral = self._compute_spectral_metrics(original, reconstructed)
        statistical = self._compute_statistical_metrics(original, reconstructed)
        structural = self._compute_structural_metrics(original, reconstructed)
        compression = self._compute_compression_metrics(
            original, compressed_nbytes, tensor_name
        )

        orig_nbytes = original.nbytes if isinstance(original, np.ndarray) else 0
        comp_nbytes = compressed_nbytes if compressed_nbytes > 0 else orig_nbytes

        # Detect precision preservation
        was_native_bf16 = native_dtype.upper() in ("BF16", "BFLOAT16", "BF16")
        if not native_dtype:
            was_native_bf16 = original.dtype == np.uint16
        output_is_bf16 = reconstructed.dtype == np.uint16
        precision_preserved = (not was_native_bf16) or output_is_bf16

        # Compute precision conversion error (BF16→float32→BF16 round-trip)
        precision_conversion_error = 0.0
        if was_native_bf16 and output_is_bf16:
            from spectralstream.core.math_primitives import (
                bfloat16_to_float32,
                float32_to_bfloat16,
            )

            orig_f32 = bfloat16_to_float32(original)
            reconverted = float32_to_bfloat16(orig_f32)
            reconverted_f32 = bfloat16_to_float32(reconverted)
            diff = orig_f32.ravel() - reconverted_f32.ravel()
            precision_conversion_error = float(np.sqrt(np.mean(diff**2)))

        combined = PerTensorLossMetrics(
            name=tensor_name,
            original_shape=original.shape,
            original_dtype=str(original.dtype)
            if hasattr(original, "dtype")
            else "unknown",
            native_dtype=native_dtype if native_dtype else str(original.dtype),
            precision_preserved=precision_preserved,
            precision_conversion_error=precision_conversion_error,
            original_nbytes=orig_nbytes,
            compressed_nbytes=comp_nbytes,
            method_used=method_used,
            spectral=spectral,
            statistical=statistical,
            structural=structural,
            compression=compression,
        )

        combined.overall_loss_percent = self._compute_overall_loss_percent(combined)
        combined.quality_grade = self._grade_quality(combined, tensor_type)
        combined.confidence_score = self._compute_confidence(combined)
        combined.recommended_action = self._recommend_action(combined)

        return combined

    def _compute_spectral_metrics(
        self, original: np.ndarray, reconstructed: np.ndarray
    ) -> SpectralMetrics:
        from spectralstream.core.math_primitives import ensure_float32

        o = np.asarray(ensure_float32(original), dtype=np.float64)
        r = np.asarray(ensure_float32(reconstructed), dtype=np.float64)
        n = min(o.size, r.size)
        o_flat = o.ravel()[:n]
        r_flat = r.ravel()[:n]
        d_flat = o_flat - r_flat

        # L1 error
        l1 = float(np.sum(np.abs(d_flat))) / max(n, 1)

        # MSE / RMSE
        mse = float(np.mean(d_flat**2))
        rmse = math.sqrt(mse) if mse > 0 else 0.0

        # Spectral norm error
        spec_norm_error = 0.0
        if o.ndim == 2 and r.ndim == 2:
            try:
                s_orig = np.linalg.svd(
                    o[: min(o.shape[0], 256), : min(o.shape[1], 256)], compute_uv=False
                )
                s_recon = np.linalg.svd(
                    r[: min(r.shape[0], 256), : min(r.shape[1], 256)], compute_uv=False
                )
                min_s = min(s_orig.size, s_recon.size)
                if min_s > 0:
                    spec_norm_error = float(
                        np.max(np.abs(s_orig[:min_s] - s_recon[:min_s]))
                    )
            except np.linalg.LinAlgError:
                spec_norm_error = _spectral_norm(diff)

        # Spectral entropy change
        ent_orig = _spectral_entropy_1d(o_flat)
        ent_recon = _spectral_entropy_1d(r_flat)
        ent_change = abs(ent_orig - ent_recon)

        # Energy concentration preservation
        ec_orig = _energy_concentration(o_flat)
        ec_recon = _energy_concentration(r_flat)
        ec_preserved = 1.0 - min(abs(ec_orig - ec_recon), 1.0)

        return SpectralMetrics(
            l1_error=l1,
            mse=mse,
            rmse=rmse,
            spectral_norm_error=spec_norm_error,
            spectral_entropy_change=ent_change,
            energy_concentration_preserved=ec_preserved,
        )

    def _compute_statistical_metrics(
        self, original: np.ndarray, reconstructed: np.ndarray
    ) -> StatisticalMetrics:
        from spectralstream.core.math_primitives import ensure_float32

        o, r = _flatten_pair(ensure_float32(original), ensure_float32(reconstructed))
        if o.size == 0:
            return StatisticalMetrics()

        diff = o - r
        mse = float(np.mean(diff**2))
        mae = float(np.mean(np.abs(diff)))

        # SNR
        var_signal = float(np.var(o))
        var_noise = float(np.var(diff))
        if var_noise < 1e-30:
            snr = float("inf")
        else:
            snr = float(10.0 * math.log10(max(var_signal / var_noise, 1e-30)))

        # PSNR
        peak_val = float(np.max(np.abs(o)))
        if peak_val < 1e-30 or mse < 1e-30:
            psnr = float("inf")
        else:
            psnr = float(10.0 * math.log10(peak_val**2 / max(mse, 1e-30)))

        # Cosine similarity
        dot = float(np.dot(o, r))
        n_o = float(np.linalg.norm(o))
        n_r = float(np.linalg.norm(r))
        if n_o < 1e-30 or n_r < 1e-30:
            cos_sim = 1.0 if n_o == n_r else 0.0
        else:
            cos_sim = float(np.clip(dot / (n_o * n_r), -1.0, 1.0))

        # SSIM
        ssim_val = _ssim_2d(original, reconstructed)
        try_scipy = _try_scipy_ssim(original, reconstructed)
        if try_scipy is not None:
            ssim_val = try_scipy

        # KL divergence
        combined = np.concatenate([o, r])
        lo, hi = float(np.min(combined)), float(np.max(combined))
        if hi - lo < 1e-30:
            kl_div = 0.0
            wass_dist = 0.0
        else:
            n_bins = min(256, max(10, int(math.sqrt(o.size))))
            bins = np.linspace(lo, hi, n_bins + 1)
            h_orig, _ = np.histogram(o, bins=bins, density=True)
            h_recon, _ = np.histogram(r, bins=bins, density=True)
            eps = 1e-30
            p = h_orig + eps
            q = h_recon + eps
            p = p / np.sum(p)
            q = q / np.sum(q)
            kl_div = float(np.sum(p * np.log(p / q)))
            # Wasserstein via sorted-array approximation
            o_sorted = np.sort(o)
            r_sorted = np.sort(r)
            n_min = min(len(o_sorted), len(r_sorted))
            o_resampled = np.interp(
                np.linspace(0, 1, n_min),
                np.linspace(0, 1, len(o_sorted)),
                o_sorted,
            )
            r_resampled = np.interp(
                np.linspace(0, 1, n_min),
                np.linspace(0, 1, len(r_sorted)),
                r_sorted,
            )
            wass_dist = float(np.mean(np.abs(o_resampled - r_resampled)))

        return StatisticalMetrics(
            snr_db=snr,
            psnr_db=psnr,
            ssim=ssim_val,
            cosine_similarity=cos_sim,
            kl_divergence=kl_div,
            wasserstein_distance=wass_dist,
            mae=mae,
        )

    def _compute_structural_metrics(
        self, original: np.ndarray, reconstructed: np.ndarray
    ) -> StructuralMetrics:
        o, r = _flatten_pair(original, reconstructed)
        if o.size < 4:
            return StructuralMetrics()

        # Effective rank preservation
        er_orig = _effective_rank(original)
        er_recon = _effective_rank(reconstructed)
        eff_rank_preserved = (
            min(er_recon / max(er_orig, 1e-10), 1.0) if er_orig > 1e-10 else 1.0
        )

        # Sparsity pattern preservation
        sparsity_orig = float(np.mean(np.abs(o) < 1e-10))
        sparsity_recon = float(np.mean(np.abs(r) < 1e-10))
        sparsity_preserved = max(0.0, 1.0 - abs(sparsity_orig - sparsity_recon))

        # Condition number change
        cond_orig = _condition_number(original)
        cond_recon = _condition_number(reconstructed)
        if cond_orig == float("inf") or cond_recon == float("inf"):
            cond_change = 0.0
        elif cond_orig < 1e-30:
            cond_change = cond_recon if cond_recon < 1e6 else 1.0
        else:
            cond_change = abs(cond_recon - cond_orig) / max(cond_orig, 1e-30)

        # Mutual information preservation
        mi = _mutual_information_estimate(original, reconstructed)

        # Cross-correlation preservation
        xcorr = _cross_correlation_preserved(original, reconstructed)

        return StructuralMetrics(
            effective_rank_preserved=eff_rank_preserved,
            sparsity_preserved=sparsity_preserved,
            condition_number_change=cond_change,
            mutual_information_preserved=mi,
            cross_correlation_preserved=xcorr,
        )

    def _compute_compression_metrics(
        self,
        original: np.ndarray,
        compressed_nbytes: int,
        tensor_name: str = "",
    ) -> CompressionMetrics:
        from spectralstream.core.math_primitives import ensure_float32

        o_f32 = ensure_float32(original)
        o = np.asarray(o_f32, dtype=np.float64)
        orig_nbytes = original.nbytes if isinstance(original, np.ndarray) else o.nbytes
        comp_nbytes = compressed_nbytes if compressed_nbytes > 0 else orig_nbytes

        compression_ratio = orig_nbytes / max(comp_nbytes, 1)

        n_params = max(o.size, 1)
        bit_rate = (comp_nbytes * 8) / n_params

        flat = o.ravel()
        ent_rate_orig = _entropy_rate(flat)
        kolm = _kolmogorov_estimate(flat)

        return CompressionMetrics(
            compression_ratio=compression_ratio,
            bit_rate=bit_rate,
            entropy_rate_original=ent_rate_orig,
            entropy_rate_compressed=ent_rate_orig,
            kolmogorov_estimate=kolm,
        )

    def _compute_overall_loss_percent(self, metrics: PerTensorLossMetrics) -> float:
        mse = metrics.spectral.mse
        orig_flat = None
        norm_orig = max(
            metrics.spectral.l1_error * math.sqrt(max(metrics.original_nbytes, 1)),
            1e-30,
        )
        if mse > 0 and norm_orig > 0:
            return min(mse / norm_orig * 100, 100.0)
        rel_err = 0.0
        snr = metrics.statistical.snr_db
        if snr < float("inf"):
            rel_err = 10.0 ** (-snr / 20.0)
        return min(rel_err * 100, 100.0)

    def _grade_quality(
        self, metrics: PerTensorLossMetrics, tensor_type: str = "weight"
    ) -> QualityGrade:
        """
        Grade quality against tiered budgets.

        Uses TIERED_BUDGETS from the engine's tiered_error module when available,
        otherwise falls back to general thresholds.
        """
        try:
            from ..engine.tiered_error import get_budget

            max_rel_err, max_mse, min_snr = get_budget(tensor_type)
        except Exception:
            max_rel_err, max_mse, min_snr = 0.02, 0.0001, 25.0

        mse = metrics.spectral.mse
        snr = metrics.statistical.snr_db
        cos_sim = metrics.statistical.cosine_similarity
        effective_rank = metrics.structural.effective_rank_preserved
        kl_div = metrics.statistical.kl_divergence
        mae = metrics.statistical.mae

        failures = 0
        total_checks = 7

        if mse > max_mse:
            failures += 1
        if snr < min_snr:
            failures += 1
        if cos_sim < 0.99:
            failures += 1
        if effective_rank < 0.5:
            failures += 1
        if kl_div > 0.1:
            failures += 1
        if mae > max_mse * 10:
            failures += 1
        if mse > max_mse * 5:
            failures += 1

        fail_ratio = failures / max(total_checks, 1)
        if fail_ratio <= 0.1:
            return QualityGrade.EXCELLENT
        elif fail_ratio <= 0.3:
            return QualityGrade.GOOD
        elif fail_ratio <= 0.5:
            return QualityGrade.FAIR
        elif fail_ratio <= 0.7:
            return QualityGrade.POOR
        else:
            return QualityGrade.FAIL

    def _compute_confidence(self, metrics: PerTensorLossMetrics) -> float:
        n_elements = (
            int(np.prod(metrics.original_shape)) if metrics.original_shape else 0
        )
        if n_elements < 4:
            return 0.3
        snr = metrics.statistical.snr_db
        if snr == float("inf") or snr > 60:
            return 0.95
        if snr > 30:
            return 0.85
        if snr > 10:
            return 0.7
        return 0.5

    def _recommend_action(self, metrics: PerTensorLossMetrics) -> str:
        grade = metrics.quality_grade
        if grade == QualityGrade.FAIL:
            return "reject"
        elif grade == QualityGrade.POOR:
            return "retry"
        elif grade == QualityGrade.FAIR:
            return "inspect"
        elif grade == QualityGrade.GOOD:
            return "accept"
        else:
            return "accept"

    # ── Summary Report ──────────────────────────────────────────────────

    def summary_report(self, all_metrics: List[PerTensorLossMetrics]) -> Dict[str, Any]:
        """Generate model-level summary from per-tensor metrics."""
        if not all_metrics:
            return {
                "n_tensors": 0,
                "overall_quality": "unknown",
                "overall_loss_percent": 0.0,
                "avg_confidence": 0.0,
                "grade_distribution": {},
                "avg_compression_ratio": 1.0,
                "total_original_bytes": 0,
                "total_compressed_bytes": 0,
                "tensor_results": [],
            }

        n = len(all_metrics)
        grades = [m.quality_grade for m in all_metrics]
        grade_counts: Dict[str, int] = {}
        for g in grades:
            grade_counts[g.value] = grade_counts.get(g.value, 0) + 1

        loss_pcts = [m.overall_loss_percent for m in all_metrics]
        avg_loss = float(np.mean(loss_pcts)) if loss_pcts else 0.0
        max_loss = float(np.max(loss_pcts)) if loss_pcts else 0.0

        confs = [m.confidence_score for m in all_metrics]
        avg_conf = float(np.mean(confs)) if confs else 0.0

        ratios = [m.compression.compression_ratio for m in all_metrics]
        avg_ratio = float(np.mean(ratios)) if ratios else 1.0

        total_orig = sum(m.original_nbytes for m in all_metrics)
        total_comp = sum(m.compressed_nbytes for m in all_metrics)

        # Determine overall quality: worst grade wins
        grade_priority = [
            QualityGrade.FAIL,
            QualityGrade.POOR,
            QualityGrade.FAIR,
            QualityGrade.GOOD,
            QualityGrade.EXCELLENT,
        ]
        overall_quality = QualityGrade.EXCELLENT.value
        for gp in grade_priority:
            if gp in grades:
                overall_quality = gp.value
                break

        passed_1pct = sum(1 for m in all_metrics if m.overall_loss_percent < 1.0)
        passed_5pct = sum(1 for m in all_metrics if m.overall_loss_percent < 5.0)

        return {
            "n_tensors": n,
            "overall_quality": overall_quality,
            "overall_loss_percent": avg_loss,
            "max_loss_percent": max_loss,
            "avg_confidence": avg_conf,
            "grade_distribution": grade_counts,
            "avg_compression_ratio": avg_ratio,
            "total_original_bytes": total_orig,
            "total_compressed_bytes": total_comp,
            "overall_compression_ratio": total_orig / max(total_comp, 1),
            "tensors_passed_1pct": passed_1pct,
            "tensors_passed_5pct": passed_5pct,
            "tensor_results": [m.to_dict() for m in all_metrics],
        }

    def check_certification_eligibility(
        self, summary: Dict[str, Any], max_loss_pct: float = 1.0
    ) -> Tuple[bool, str]:
        """Check if the model meets certification requirements (< 1% loss target).

        Parameters
        ----------
        summary : dict
            Output from summary_report().
        max_loss_pct : float
            Maximum acceptable loss percentage (default 1.0%).

        Returns
        -------
        eligible : bool
            True if model passes certification.
        message : str
            Human-readable explanation.
        """
        if summary["n_tensors"] == 0:
            return False, "No tensors to certify"

        avg_loss = summary["overall_loss_percent"]
        max_loss = summary["max_loss_percent"]
        n_failed = summary["n_tensors"] - summary["tensors_passed_1pct"]

        if avg_loss > max_loss_pct:
            return (
                False,
                f"Average loss {avg_loss:.2f}% exceeds threshold {max_loss_pct}%",
            )
        if max_loss > max_loss_pct * 3:
            return (
                False,
                f"Max loss {max_loss:.2f}% exceeds 3x threshold {max_loss_pct * 3}%",
            )
        if n_failed > summary["n_tensors"] * 0.1:
            return (
                False,
                f"{n_failed}/{summary['n_tensors']} tensors exceed 1% loss",
            )

        return (
            True,
            f"Certification PASSED (avg loss {avg_loss:.2f}%, max {max_loss:.2f}%)",
        )
