"""
Comprehensive loss metrics for the compression engine.

Provides per-tensor quality metrics so the engine can auto-select the
right compression aggressiveness for each tensor type.

Key Components
--------------
- ``TensorLossMetrics`` : dataclass holding 20+ loss metrics per tensor
- ``LossMetricsTracker`` : tracks running quality, checks budgets, detects when
  recompression is needed
- ``compute_tiered_error_budget`` : helper to get error budgets by tensor type
- ``compute_tiered_error_budget`` : helper to get error budgets by tensor type
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Quality grade thresholds ──────────────────────────────────────────────
QUALITY_EXCELLENT_SNR = 40.0
QUALITY_GOOD_SNR = 30.0
QUALITY_FAIR_SNR = 20.0
QUALITY_POOR_SNR = 10.0

QUALITY_EXCELLENT_MSE = 1e-5
QUALITY_GOOD_MSE = 1e-4
QUALITY_FAIR_MSE = 1e-3
QUALITY_POOR_MSE = 1e-2

QUALITY_EXCELLENT_COSINE = 0.999
QUALITY_GOOD_COSINE = 0.99
QUALITY_FAIR_COSINE = 0.95
QUALITY_POOR_COSINE = 0.9


@dataclass
class TensorLossMetrics:
    """Comprehensive loss metrics for a compressed tensor.

    Attributes
    ----------
    name : str
        Tensor name for identification.
    original_shape : tuple of int
        Shape of the original tensor.
    original_dtype : str
        Data type of the original tensor.
    original_size : int
        Size of the original tensor in bytes.
    compressed_size : int
        Size of the compressed tensor in bytes.
    compression_ratio : float
        Ratio of original size to compressed size.

    Core errors
    -----------
    mse : float
        Mean squared error.
    mae : float
        Mean absolute error.
    max_ae : float
        Maximum absolute error.
    rmse : float
        Root mean squared error.
    relative_error_l2 : float
        ||x - x_hat||_2 / ||x||_2, relative L2 error.
    relative_error_linf : float
        ||x - x_hat||_inf / ||x||_inf, relative infinity-norm error.

    Signal-based
    ------------
    snr_db : float
        Signal-to-noise ratio in dB.
    psnr_db : float
        Peak signal-to-noise ratio in dB.
    cosine_similarity : float
        Cosine similarity between original and reconstructed [-1, 1].

    Statistical
    -----------
    kl_divergence : float
        KL divergence between distributions (clipped, numerical stable).
    wasserstein_distance : float
        Earth mover's distance between distributions.
    ks_statistic : float
        Kolmogorov-Smirnov statistic.
    js_divergence : float
        Jensen-Shannon divergence.

    Distribution
    ------------
    mean_bias : float
        mean(original) - mean(reconstructed).
    std_shift : float
        std(original) - std(reconstructed).
    skewness_shift : float
        Skewness difference.
    kurtosis_shift : float
        Kurtosis difference.

    Outlier preservation
    --------------------
    outlier_preservation_rate : float
        Fraction of 3-sigma outliers preserved.
    top_1_preservation : float
        Fraction of top 1% values preserved.
    bottom_1_preservation : float
        Fraction of bottom 1% preserved.

    Quality grade
    -------------
    quality_grade : str
        One of EXCELLENT, GOOD, FAIR, POOR, UNACCEPTABLE.
    is_acceptable : bool
        Whether quality is acceptable (at least FAIR).
    """

    # Identifiers
    name: str = ""
    original_shape: Tuple[int, ...] = (0,)
    original_dtype: str = "float32"
    original_size: int = 0
    compressed_size: int = 0
    compression_ratio: float = 1.0

    # Core errors
    mse: float = 0.0
    mae: float = 0.0
    max_ae: float = 0.0
    rmse: float = 0.0
    relative_error_l2: float = 0.0
    relative_error_linf: float = 0.0

    # Signal-based
    snr_db: float = float("inf")
    psnr_db: float = float("inf")
    cosine_similarity: float = 1.0

    # Statistical
    kl_divergence: float = 0.0
    wasserstein_distance: float = 0.0
    ks_statistic: float = 0.0
    js_divergence: float = 0.0

    # Distribution
    mean_bias: float = 0.0
    std_shift: float = 0.0
    skewness_shift: float = 0.0
    kurtosis_shift: float = 0.0

    # Outlier preservation
    outlier_preservation_rate: float = 1.0
    top_1_preservation: float = 1.0
    bottom_1_preservation: float = 1.0

    # Quality grade
    quality_grade: str = "EXCELLENT"
    is_acceptable: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """All metrics as a flat dictionary (for JSON serialization / CLI output).

        Returns
        -------
        dict
            All metric fields as a flat dictionary with string keys.
        """
        return {
            "name": self.name,
            "original_shape": list(self.original_shape),
            "original_dtype": self.original_dtype,
            "original_size": self.original_size,
            "compressed_size": self.compressed_size,
            "compression_ratio": self.compression_ratio,
            "mse": self.mse,
            "mae": self.mae,
            "max_ae": self.max_ae,
            "rmse": self.rmse,
            "relative_error_l2": self.relative_error_l2,
            "relative_error_linf": self.relative_error_linf,
            "snr_db": self.snr_db,
            "psnr_db": self.psnr_db,
            "cosine_similarity": self.cosine_similarity,
            "kl_divergence": self.kl_divergence,
            "wasserstein_distance": self.wasserstein_distance,
            "ks_statistic": self.ks_statistic,
            "js_divergence": self.js_divergence,
            "mean_bias": self.mean_bias,
            "std_shift": self.std_shift,
            "skewness_shift": self.skewness_shift,
            "kurtosis_shift": self.kurtosis_shift,
            "outlier_preservation_rate": self.outlier_preservation_rate,
            "top_1_preservation": self.top_1_preservation,
            "bottom_1_preservation": self.bottom_1_preservation,
            "quality_grade": self.quality_grade,
            "is_acceptable": self.is_acceptable,
        }

    @staticmethod
    def compute(
        original: np.ndarray,
        reconstructed: np.ndarray,
        name: str = "",
        compressed_size: Optional[int] = None,
    ) -> TensorLossMetrics:
        """Compute ALL loss metrics between original and reconstructed tensors.

        Handles:
        - Flattening for multi-dim tensors
        - Numerical stability (avoid div by zero)
        - Speed: O(n) for basic metrics, O(n log n) for distributional

        Parameters
        ----------
        original : np.ndarray
            Original (pre-compression) tensor.
        reconstructed : np.ndarray
            Reconstructed (post-decompression) tensor.
        name : str
            Optional tensor name for identification.
        compressed_size : int, optional
            Size of the compressed representation in bytes.  If None,
            compression-related fields are set to defaults.

        Returns
        -------
        TensorLossMetrics
            Fully populated metrics instance.
        """
        # Flatten to 1D for distributional comparisons
        orig_flat = original.ravel().astype(np.float64)
        recon_flat = reconstructed.ravel().astype(np.float64)

        n = orig_flat.size

        # ── Basic sizes ────────────────────────────────────────────────
        orig_size = original.nbytes
        comp_size = compressed_size if compressed_size is not None else orig_size
        comp_ratio = float(orig_size / max(comp_size, 1))

        # ── Core errors (O(n)) ─────────────────────────────────────────
        diff = orig_flat - recon_flat
        abs_diff = np.abs(diff)

        mse = float(np.mean(diff**2))
        mae = float(np.mean(abs_diff))
        max_ae = float(np.max(abs_diff))
        rmse = float(math.sqrt(mse))

        # Relative L2: ||x - x_hat||_2 / ||x||_2
        norm_orig_l2 = max(np.linalg.norm(orig_flat), 1e-30)
        relative_error_l2 = float(np.linalg.norm(diff) / norm_orig_l2)

        # Relative Linf: ||x - x_hat||_inf / ||x||_inf
        norm_orig_linf = max(np.max(np.abs(orig_flat)), 1e-30)
        relative_error_linf = float(np.max(abs_diff) / norm_orig_linf)

        # ── Signal-based (O(n)) ────────────────────────────────────────
        var_signal = float(np.var(orig_flat))
        var_noise = float(np.var(diff))

        if var_noise < 1e-30:
            snr_db = float("inf")
        else:
            snr_db = float(10.0 * math.log10(max(var_signal / var_noise, 1e-30)))

        peak_val = float(np.max(np.abs(orig_flat)))
        if peak_val < 1e-30 or mse < 1e-30:
            psnr_db = float("inf")
        else:
            psnr_db = float(10.0 * math.log10(peak_val**2 / max(mse, 1e-30)))

        # Cosine similarity
        dot_product = float(np.dot(orig_flat, recon_flat))
        norm_product = max(
            np.linalg.norm(orig_flat) * np.linalg.norm(recon_flat), 1e-30
        )
        cosine_similarity = float(np.clip(dot_product / norm_product, -1.0, 1.0))

        # ── Distribution statistics (O(n)) ────────────────────────────
        mean_orig = float(np.mean(orig_flat))
        mean_recon = float(np.mean(recon_flat))
        std_orig = float(np.std(orig_flat))
        std_recon = float(np.std(recon_flat))
        mean_bias = mean_orig - mean_recon
        std_shift = std_orig - std_recon

        # Skewness: third standardized moment
        def _skewness(arr: np.ndarray) -> float:
            s = float(np.std(arr))
            if s < 1e-30:
                return 0.0
            return float(np.mean(((arr - np.mean(arr)) / s) ** 3))

        # Kurtosis: fourth standardized moment (excess)
        def _kurtosis(arr: np.ndarray) -> float:
            s = float(np.std(arr))
            if s < 1e-30:
                return 0.0
            return float(np.mean(((arr - np.mean(arr)) / s) ** 4)) - 3.0

        skewness_orig = _skewness(orig_flat)
        skewness_recon = _skewness(recon_flat)
        kurtosis_orig = _kurtosis(orig_flat)
        kurtosis_recon = _kurtosis(recon_flat)

        skewness_shift = skewness_orig - skewness_recon
        kurtosis_shift = kurtosis_orig - kurtosis_recon

        # ── Distributional divergences (O(n log n)) ───────────────────
        # Bin both signals into histograms for KL / JS / Wasserstein
        n_bins = min(256, max(10, int(math.sqrt(n))))
        combined_min = float(min(np.min(orig_flat), np.min(recon_flat)))
        combined_max = float(max(np.max(orig_flat), np.max(recon_flat)))

        if combined_max - combined_min < 1e-30:
            # Identical or constant signals
            kl_div = 0.0
            js_div = 0.0
            wass_dist = 0.0
            ks_stat = 0.0
        else:
            bins = np.linspace(combined_min, combined_max, n_bins + 1)
            hist_orig, _ = np.histogram(orig_flat, bins=bins, density=True)
            hist_recon, _ = np.histogram(recon_flat, bins=bins, density=True)

            # Add tiny epsilon to avoid log(0)
            eps = 1e-30
            p = hist_orig + eps
            q = hist_recon + eps
            # Normalize to valid probability distributions
            p = p / np.sum(p)
            q = q / np.sum(q)

            # KL(P || Q)
            kl_div = float(np.sum(p * np.log(p / q)))

            # JS(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M), M = (P+Q)/2
            m = 0.5 * (p + q)
            js_div = float(
                0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m))
            )

            # Wasserstein-1 (Earth mover's distance) via CDF difference
            # |CDF_P - CDF_Q| integrated over bins
            cdf_p = np.cumsum(p)
            cdf_q = np.cumsum(q)
            wass_dist = float(
                np.sum(np.abs(cdf_p - cdf_q)) * (combined_max - combined_min) / n_bins
            )

            # Kolmogorov-Smirnov statistic = max |CDF_P - CDF_Q|
            ks_stat = float(np.max(np.abs(cdf_p - cdf_q)))

        # ── Outlier preservation (O(n)) ───────────────────────────────
        # 3-sigma outliers
        threshold_outlier = (
            3.0 * std_orig + abs(mean_orig) if std_orig > 1e-30 else float("inf")
        )
        if threshold_outlier == float("inf"):
            outlier_preservation_rate = 1.0
        else:
            orig_outliers = np.abs(orig_flat) > threshold_outlier
            if np.any(orig_outliers):
                recon_outliers = np.abs(recon_flat) > threshold_outlier
                outlier_preservation_rate = float(
                    np.sum(orig_outliers & recon_outliers) / np.sum(orig_outliers)
                )
            else:
                outlier_preservation_rate = 1.0

        # Top 1% and bottom 1% preservation
        n_top = max(1, n // 100)
        top_indices = np.argsort(orig_flat)[-n_top:]
        bottom_indices = np.argsort(orig_flat)[:n_top]

        # Check how many top/bottom values remain in the top/bottom
        # of the reconstructed tensor
        recon_top_indices = set(np.argsort(recon_flat)[-n_top:])
        recon_bottom_indices = set(np.argsort(recon_flat)[:n_top])

        top_1_preservation = float(len(set(top_indices) & recon_top_indices) / n_top)
        bottom_1_preservation = float(
            len(set(bottom_indices) & recon_bottom_indices) / n_top
        )

        # ── Quality grading ───────────────────────────────────────────
        grade, acceptable = _grade_quality(
            mse=mse,
            snr_db=snr_db,
            cosine_similarity=cosine_similarity,
            kl_divergence=kl_div,
            js_divergence=js_div,
            max_ae=max_ae,
            relative_error_l2=relative_error_l2,
        )

        return TensorLossMetrics(
            name=name,
            original_shape=original.shape,
            original_dtype=str(original.dtype),
            original_size=orig_size,
            compressed_size=comp_size,
            compression_ratio=comp_ratio,
            mse=mse,
            mae=mae,
            max_ae=max_ae,
            rmse=rmse,
            relative_error_l2=relative_error_l2,
            relative_error_linf=relative_error_linf,
            snr_db=snr_db,
            psnr_db=psnr_db,
            cosine_similarity=cosine_similarity,
            kl_divergence=kl_div,
            wasserstein_distance=wass_dist,
            ks_statistic=ks_stat,
            js_divergence=js_div,
            mean_bias=mean_bias,
            std_shift=std_shift,
            skewness_shift=skewness_shift,
            kurtosis_shift=kurtosis_shift,
            outlier_preservation_rate=outlier_preservation_rate,
            top_1_preservation=top_1_preservation,
            bottom_1_preservation=bottom_1_preservation,
            quality_grade=grade,
            is_acceptable=acceptable,
        )

    def summary(self) -> str:
        """Single-line quality summary string."""
        return (
            f"[{self.quality_grade}] {self.name}: "
            f"MSE={self.mse:.2e} MAE={self.mae:.2e} "
            f"SNR={self.snr_db:.1f}dB Cos={self.cosine_similarity:.4f} "
            f"Ratio={self.compression_ratio:.1f}x "
            f"KL={self.kl_divergence:.4f} KS={self.ks_statistic:.4f}"
        )

    def check_budget(self, budget: Dict[str, float]) -> Tuple[bool, Dict[str, str]]:
        """Check if this tensor's metrics satisfy a given error budget.

        Parameters
        ----------
        budget : dict
            Error budget with optional keys ``max_mse``, ``max_mae``,
            ``min_snr``, ``max_relative_error_l2``, ``max_kl``,
            ``min_cosine``.

        Returns
        -------
        passed : bool
            True if all budget constraints are met.
        violations : dict
            Map of budget key -> human-readable violation message.
        """
        checks: Dict[str, Tuple[float, float, str]] = {
            "max_mse": (self.mse, budget.get("max_mse", float("inf")), "≤"),
            "max_mae": (self.mae, budget.get("max_mae", float("inf")), "≤"),
            "min_snr": (self.snr_db, budget.get("min_snr", 0.0), "≥"),
            "max_relative_error_l2": (
                self.relative_error_l2,
                budget.get("max_relative_error_l2", float("inf")),
                "≤",
            ),
            "max_kl": (
                self.kl_divergence,
                budget.get("max_kl", float("inf")),
                "≤",
            ),
            "min_cosine": (
                self.cosine_similarity,
                budget.get("min_cosine", 0.0),
                "≥",
            ),
        }

        violations: Dict[str, str] = {}
        for key, (value, limit, direction) in checks.items():
            if key not in budget:
                continue
            if limit is None or limit == float("inf"):
                continue
            if direction == "≤" and value > limit:
                violations[key] = f"{value:.2e} exceeds max {limit:.2e}"
            elif direction == "≥" and value < limit:
                violations[key] = f"{value:.2f} below min {limit:.2f}"

        return len(violations) == 0, violations


def _grade_quality(
    mse: float,
    snr_db: float,
    cosine_similarity: float,
    kl_divergence: float,
    js_divergence: float,
    max_ae: float,
    relative_error_l2: float,
) -> Tuple[str, bool]:
    """Determine quality grade from a set of metrics.

    Returns
    -------
    grade : str
        One of ``EXCELLENT``, ``GOOD``, ``FAIR``, ``POOR``, ``UNACCEPTABLE``.
    acceptable : bool
        ``True`` if grade is at least ``FAIR``.
    """
    # Count how many metrics pass each threshold level
    excellent = 0
    good = 0
    fair = 0
    poor = 0

    # MSE thresholds
    if mse <= QUALITY_EXCELLENT_MSE:
        excellent += 1
    elif mse <= QUALITY_GOOD_MSE:
        good += 1
    elif mse <= QUALITY_FAIR_MSE:
        fair += 1
    elif mse <= QUALITY_POOR_MSE:
        poor += 1
    else:
        pass  # unacceptable

    # SNR thresholds
    if snr_db >= QUALITY_EXCELLENT_SNR:
        excellent += 1
    elif snr_db >= QUALITY_GOOD_SNR:
        good += 1
    elif snr_db >= QUALITY_FAIR_SNR:
        fair += 1
    elif snr_db >= QUALITY_POOR_SNR:
        poor += 1
    else:
        pass

    # Cosine thresholds
    if cosine_similarity >= QUALITY_EXCELLENT_COSINE:
        excellent += 1
    elif cosine_similarity >= QUALITY_GOOD_COSINE:
        good += 1
    elif cosine_similarity >= QUALITY_FAIR_COSINE:
        fair += 1
    elif cosine_similarity >= QUALITY_POOR_COSINE:
        poor += 1
    else:
        pass

    # Relative L2
    if relative_error_l2 <= 0.001:
        excellent += 1
    elif relative_error_l2 <= 0.01:
        good += 1
    elif relative_error_l2 <= 0.05:
        fair += 1
    elif relative_error_l2 <= 0.1:
        poor += 1
    else:
        pass

    # KL divergence
    if kl_divergence <= 0.001:
        excellent += 1
    elif kl_divergence <= 0.01:
        good += 1
    elif kl_divergence <= 0.1:
        fair += 1
    elif kl_divergence <= 0.5:
        poor += 1
    else:
        pass

    # JS divergence
    if js_divergence <= 0.001:
        excellent += 1
    elif js_divergence <= 0.01:
        good += 1
    elif js_divergence <= 0.05:
        fair += 1
    elif js_divergence <= 0.2:
        poor += 1
    else:
        pass

    # Max absolute error
    if max_ae <= 0.001:
        excellent += 1
    elif max_ae <= 0.01:
        good += 1
    elif max_ae <= 0.1:
        fair += 1
    elif max_ae <= 1.0:
        poor += 1
    else:
        pass

    # Determine overall grade by majority vote, with excellence bias
    if excellent >= 4:
        grade = "EXCELLENT"
    elif good + excellent >= 4:
        grade = "GOOD"
    elif fair + good + excellent >= 4:
        grade = "FAIR"
    elif poor + fair + good + excellent >= 4:
        grade = "POOR"
    else:
        grade = "UNACCEPTABLE"

    acceptable = grade in ("EXCELLENT", "GOOD", "FAIR")
    return grade, acceptable


# ── Loss Metrics Tracker ──────────────────────────────────────────────────


class LossMetricsTracker:
    """Tracks running loss metrics and makes quality-aware decisions.

    Features
    --------
    - Checks metrics against tiered error budgets per tensor type
    - Tracks running quality across model (per-tensor-type sliding window)
    - Detects when recompression is needed
    - Provides summary statistics for reporting

    Does NOT replace ``LossMetricsTracker`` in ``loss_aware_compressor.py``
    (which handles error budget allocation and cascade pattern selection).
    This tracker focuses on the *feedback side* — measuring and validating
    compression quality.

    Examples
    --------
    >>> tracker = LossMetricsTracker()
    >>> budget = tracker.get_budget("attention_q")
    >>> metrics = TensorLossMetrics.compute(orig, recon)
    >>> passed, violations = metrics.check_budget(budget)
    >>> should_redo, reason = tracker.should_recompress("attention_q", metrics)
    """

    # Error budgets by tensor type (tighter for sensitive tensors)
    TIERED_ERROR_BUDGETS: Dict[str, Dict[str, float]] = {
        "attention_q": {
            "max_mse": 0.0001,
            "max_mae": 0.005,
            "min_snr": 40.0,
            "max_relative_error_l2": 0.01,
            "min_cosine": 0.999,
            "max_kl": 0.01,
        },
        "attention_k": {
            "max_mse": 0.0001,
            "max_mae": 0.005,
            "min_snr": 40.0,
            "max_relative_error_l2": 0.01,
            "min_cosine": 0.999,
            "max_kl": 0.01,
        },
        "attention_v": {
            "max_mse": 0.0005,
            "max_mae": 0.01,
            "min_snr": 30.0,
            "max_relative_error_l2": 0.02,
            "min_cosine": 0.995,
            "max_kl": 0.05,
        },
        "attention_o": {
            "max_mse": 0.0005,
            "max_mae": 0.01,
            "min_snr": 30.0,
            "max_relative_error_l2": 0.02,
            "min_cosine": 0.995,
            "max_kl": 0.05,
        },
        "qkv_fused": {
            "max_mse": 0.0002,
            "max_mae": 0.008,
            "min_snr": 35.0,
            "max_relative_error_l2": 0.015,
            "min_cosine": 0.997,
            "max_kl": 0.03,
        },
        "ffn_gate": {
            "max_mse": 0.001,
            "max_mae": 0.02,
            "min_snr": 25.0,
            "max_relative_error_l2": 0.05,
            "min_cosine": 0.99,
            "max_kl": 0.1,
        },
        "ffn_up": {
            "max_mse": 0.001,
            "max_mae": 0.02,
            "min_snr": 25.0,
            "max_relative_error_l2": 0.05,
            "min_cosine": 0.99,
            "max_kl": 0.1,
        },
        "ffn_down": {
            "max_mse": 0.001,
            "max_mae": 0.02,
            "min_snr": 25.0,
            "max_relative_error_l2": 0.05,
            "min_cosine": 0.99,
            "max_kl": 0.1,
        },
        "embedding": {
            "max_mse": 0.0005,
            "max_mae": 0.01,
            "min_snr": 30.0,
            "max_relative_error_l2": 0.02,
            "min_cosine": 0.995,
            "max_kl": 0.05,
        },
        "norm": {
            "max_mse": 0.01,
            "max_mae": 0.05,
            "min_snr": 15.0,
            "max_relative_error_l2": 0.1,
            "min_cosine": 0.95,
            "max_kl": 0.5,
        },
        "output": {
            "max_mse": 0.0005,
            "max_mae": 0.01,
            "min_snr": 30.0,
            "max_relative_error_l2": 0.02,
            "min_cosine": 0.995,
            "max_kl": 0.05,
        },
        "weight": {
            "max_mse": 0.001,
            "max_mae": 0.02,
            "min_snr": 25.0,
            "max_relative_error_l2": 0.05,
            "min_cosine": 0.99,
            "max_kl": 0.1,
        },
    }

    # Pattern aggressiveness by tensor type
    PATTERN_MAP: Dict[str, str] = {
        "attention_q": "balanced",
        "attention_k": "balanced",
        "attention_v": "aggressive",
        "attention_o": "aggressive",
        "qkv_fused": "balanced",
        "ffn_gate": "aggressive",
        "ffn_up": "aggressive",
        "ffn_down": "aggressive",
        "embedding": "balanced",
        "norm": "lightning",
        "output": "balanced",
        "weight": "aggressive",
    }

    def __init__(self, default_aggressiveness: str = "balanced"):
        self._default_aggressiveness = default_aggressiveness
        self._running_quality: Dict[str, List[float]] = {}
        self._tensor_metrics: Dict[str, TensorLossMetrics] = {}

    def get_budget(self, tensor_type: str) -> Dict[str, float]:
        """Get error budget for a tensor type, with sensible defaults.

        Parameters
        ----------
        tensor_type : str
            Tensor type string (e.g. ``"attention_q"``, ``"ffn_gate"``).

        Returns
        -------
        dict
            Error budget with keys like ``max_mse``, ``max_mae``, ``min_snr``.
        """
        return compute_tiered_error_budget(tensor_type)

    def select_pattern(
        self,
        tensor: np.ndarray,
        tensor_type: str = "weight",
        target_ratio: Optional[float] = None,
    ) -> str:
        """Select compression pattern based on tensor type and loss tolerance.

        Sensitive tensors (attention_q/k) → conservative pattern.
        Robust tensors (ffn) → aggressive pattern.
        Norms/biases → lightweight pattern.

        Parameters
        ----------
        tensor : np.ndarray
            The tensor to compress.
        tensor_type : str
            Type classification (e.g. ``"attention_q"``, ``"ffn_gate"``).
        target_ratio : float, optional
            Desired compression ratio. Higher values may push toward
            more aggressive patterns.

        Returns
        -------
        str
            Cascade pattern name.
        """
        # Start with type-based default
        pattern = self.PATTERN_MAP.get(tensor_type, self._default_aggressiveness)

        # Adjust for tensor size
        n_elements = tensor.size
        if n_elements >= 10_000_000:
            # Very large tensors can tolerate more aggression
            if pattern in ("balanced",):
                pattern = "aggressive"
        elif n_elements < 100_000:
            # Small tensors need conservative treatment
            if pattern == "extreme":
                pattern = "aggressive"
            elif pattern == "aggressive":
                pattern = "balanced"

        # Adjust for target ratio
        if target_ratio is not None:
            if target_ratio >= 500:
                pattern = "extreme"
            elif target_ratio >= 200:
                if pattern != "lightning":
                    pattern = "aggressive" if pattern == "balanced" else pattern
                pattern = "deep_svd_dct_fwht_wavelet"
            elif target_ratio >= 100:
                if pattern == "lightning":
                    pattern = "balanced"

        # If we have historical quality data, adjust based on that
        type_history = self._running_quality.get(tensor_type, [])
        if len(type_history) >= 3:
            avg_mse = sum(type_history[-3:]) / 3
            if avg_mse > 0.01 and pattern in ("extreme", "aggressive"):
                # Previous compressions had high error — dial back
                if pattern == "extreme":
                    pattern = "deep_svd_dct_fwht_wavelet"
                elif pattern == "aggressive":
                    pattern = "balanced"
                logger.debug(
                    "Dialing back pattern for %s to %s (avg_mse=%.4f)",
                    tensor_type,
                    pattern,
                    avg_mse,
                )

        return pattern

    def record_metrics(
        self,
        tensor_type: str,
        metrics: TensorLossMetrics,
    ) -> None:
        """Record metrics for a tensor type to track running quality.

        Parameters
        ----------
        tensor_type : str
            Type classification.
        metrics : TensorLossMetrics
            Computed loss metrics.
        """
        if tensor_type not in self._running_quality:
            self._running_quality[tensor_type] = []
        self._running_quality[tensor_type].append(metrics.mse)

        # Keep sliding window of last 20
        if len(self._running_quality[tensor_type]) > 20:
            self._running_quality[tensor_type] = self._running_quality[tensor_type][
                -20:
            ]

        # Store under composite key if name available
        key = f"{tensor_type}:{metrics.name}" if metrics.name else tensor_type
        self._tensor_metrics[key] = metrics

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics across all recorded tensor types.

        Returns
        -------
        dict
            Per-type and overall statistics.
        """
        per_type: Dict[str, Dict[str, Any]] = {}
        for ttype, mses in self._running_quality.items():
            if mses:
                per_type[ttype] = {
                    "count": len(mses),
                    "avg_mse": float(np.mean(mses)),
                    "max_mse": float(np.max(mses)),
                    "min_mse": float(np.min(mses)),
                    "avg_snr": float(
                        np.mean(
                            [
                                self._tensor_metrics.get(
                                    f"{ttype}:{n}",
                                    self._tensor_metrics.get(
                                        ttype, TensorLossMetrics()
                                    ),
                                ).snr_db
                                for n in ["_"]
                            ]
                        )
                    )
                    if self._tensor_metrics
                    else 0.0,
                }

        return {
            "per_type": per_type,
            "total_tensors_tracked": len(self._tensor_metrics),
        }

    def should_recompress(
        self,
        tensor_type: str,
        metrics: TensorLossMetrics,
    ) -> Tuple[bool, str]:
        """Determine if a tensor should be re-compressed with different settings.

        Parameters
        ----------
        tensor_type : str
            Type classification.
        metrics : TensorLossMetrics
            Computed loss metrics.

        Returns
        -------
        should_recompress : bool
            True if the tensor should be re-compressed.
        reason : str
            Human-readable explanation.
        """
        budget = self.get_budget(tensor_type)
        passed, violations = metrics.check_budget(budget)

        if passed:
            return False, ""

        # Check severity — how many budget thresholds were violated
        n_violations = len(violations)
        if n_violations <= 1:
            return True, f"Budget violation: {list(violations.values())[0]}"
        return True, f"{n_violations} budget violations: {list(violations.keys())}"


def compute_tiered_error_budget(tensor_type: str) -> Dict[str, float]:
    """Get error budget for a tensor type, with sensible defaults.

    Parameters
    ----------
    tensor_type : str
        Tensor type (e.g. ``"attention_q"``, ``"ffn_gate"``, ``"norm"``).

    Returns
    -------
    dict
        Budget dict with keys ``max_mse``, ``max_mae``, ``min_snr``,
        ``max_relative_error_l2``, ``min_cosine``, ``max_kl``.
    """
    defaults: Dict[str, float] = {
        "max_mse": 0.001,
        "max_mae": 0.01,
        "min_snr": 20.0,
        "max_relative_error_l2": 0.05,
        "min_cosine": 0.99,
        "max_kl": 0.1,
    }
    tiered = LossMetricsTracker.TIERED_ERROR_BUDGETS.get(tensor_type, {})
    result = defaults.copy()
    result.update(tiered)
    return result
