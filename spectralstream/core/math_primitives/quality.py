"""Compression quality dataclass and comprehensive assessor.

Single authoritative source for:
  - CompressionQuality dataclass (all 20 metrics)
  - QualityAssessor (orchestrated assessment with configurable thresholds)
  - Grade computation (S/A/B/C/D/F)
  - Composite score (weighted combination of all metrics)
  - Rate-distortion curve generation
"""

import math
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .metrics import (
    compute_all_metrics,
)


# ── Grade thresholds (matches compression/engine/_helpers.py:_grade_error) ──

GRADE_THRESHOLDS: Dict[str, float] = {
    "S": 0.0002,
    "A": 0.001,
    "B": 0.005,
    "C": 0.01,
    "D": 0.05,
    "F": float("inf"),
}

GRADE_ORDER = ["S", "A", "B", "C", "D", "F"]


def _grade_from_error(relative_error: float) -> str:
    """Map relative error to letter grade."""
    for grade, threshold in GRADE_THRESHOLDS.items():
        if relative_error < threshold:
            return grade
    return "F"


# ── CompressionQuality dataclass ───────────────────────────────────────


@dataclass
class CompressionQuality:
    """Single source of truth for tensor compression quality assessment.

    Contains ALL metrics computed between original and reconstructed tensors.
    Each field is a float; default 0.0 indicates "not computed" for that metric.

    Use :meth:`grade` for S/A/B/C/D/F letter grade based on relative_error.
    Use :meth:`composite_score` for a [0, 1] weighted summary of all metrics.
    """

    # Primary error metrics
    mse: float = 0.0
    rmse: float = 0.0
    mae: float = 0.0
    nmse: float = 0.0

    # Signal-quality metrics
    snr_db: float = 0.0
    psnr_db: float = 0.0
    relative_error: float = 0.0
    cosine_similarity: float = 0.0
    max_abs_error: float = 0.0

    # Structural / spectral
    ssim: float = 0.0
    spectral_angle: float = 0.0

    # Distributional
    histogram_overlap: float = 0.0
    kld_divergence: float = 0.0
    wasserstein_distance: float = 0.0
    ks_statistic: float = 0.0
    ks_p_value: float = 0.0
    correlation_coefficient: float = 0.0
    effective_rank_ratio: float = 0.0

    # Bit-level
    bit_error_rate: float = 0.0

    # ── Derivation methods ─────────────────────────────────────────────

    def grade(self) -> str:
        """S/A/B/C/D/F grade based on relative_error thresholds.

        Thresholds:
            S: relative_error < 0.0002 (0.02%)
            A: relative_error < 0.001  (0.1%)
            B: relative_error < 0.005  (0.5%)
            C: relative_error < 0.01   (1%)
            D: relative_error < 0.05   (5%)
            F: relative_error >= 0.05  (5%+)

        Returns
        -------
        str
            Letter grade S, A, B, C, D, or F.
        """
        return _grade_from_error(self.relative_error)

    def composite_score(self) -> float:
        """Weighted composite quality score in [0, 1].

        Combines all 20 metrics into a single interpretable score.
        1.0 = perfect reconstruction, 0.0 = complete failure.

        Weights prioritize: relative_error > cosine_similarity > snr > ssim
        > correlation > rmse > mae > effective_rank > histogram_overlap > kld
        > spectral_angle > wasserstein > ks > bit_error_rate > psnr.

        Returns
        -------
        float
            Composite quality score [0, 1]. Higher is better.
        """
        # Normalise every metric to [0, 1] where 1 = perfect
        rel = max(0.0, 1.0 - min(self.relative_error * 10, 10.0))
        snr = min(1.0, max(0.0, self.snr_db / 60.0))
        psnr = min(1.0, max(0.0, self.psnr_db / 100.0))
        cos = max(0.0, (self.cosine_similarity + 1.0) / 2.0)
        ssim = max(0.0, (self.ssim + 1.0) / 2.0)
        rmse = max(0.0, 1.0 - min(self.rmse * 10, 10.0))
        mae = max(0.0, 1.0 - min(self.mae * 10, 10.0))
        hist = min(1.0, max(0.0, self.histogram_overlap))
        kld = max(0.0, 1.0 - min(self.kld_divergence / 5.0, 1.0))
        corr = max(0.0, (self.correlation_coefficient + 1.0) / 2.0)
        rank = min(1.0, max(0.0, self.effective_rank_ratio))
        angle = max(0.0, 1.0 - min(self.spectral_angle / (math.pi / 2.0), 1.0))
        ber = max(0.0, 1.0 - min(self.bit_error_rate * 1000, 1.0))
        wass = max(0.0, 1.0 - min(self.wasserstein_distance * 10, 1.0))
        ks = max(0.0, 1.0 - self.ks_statistic)

        return float(
            rel * 0.15
            + cos * 0.12
            + snr * 0.10
            + ssim * 0.10
            + corr * 0.08
            + rmse * 0.06
            + mae * 0.06
            + rank * 0.06
            + hist * 0.05
            + kld * 0.04
            + angle * 0.04
            + wass * 0.03
            + ks * 0.03
            + ber * 0.03
            + psnr * 0.05
        )

    def passes_threshold(self, max_error: float = 0.01) -> bool:
        """Check if relative_error is within the maximum allowed error.

        Parameters
        ----------
        max_error : float
            Maximum acceptable relative error (default 0.01 = 1%).

        Returns
        -------
        bool
            True if relative_error <= max_error.
        """
        return self.relative_error <= max_error

    def to_dict(self) -> Dict[str, float]:
        """All metrics as a plain dictionary."""
        return asdict(self)


# ── QualityAssessor ────────────────────────────────────────────────────

DEFAULT_THRESHOLDS: Dict[str, float] = {
    "S": 0.0002,
    "A": 0.001,
    "B": 0.005,
    "C": 0.01,
    "D": 0.05,
    "F": float("inf"),
}


class QualityAssessor:
    """Comprehensive quality assessment with configurable grading thresholds.

    Parameters
    ----------
    thresholds : Dict[str, float] or None
        Grade thresholds as {grade: max_relative_error}.
        Defaults to S=0.02%, A=0.1%, B=0.5%, C=1%, D=5%, F=5%+.
    """

    def __init__(self, thresholds: Optional[Dict[str, float]] = None):
        self.thresholds = thresholds or DEFAULT_THRESHOLDS

    def assess(
        self, original: np.ndarray, reconstructed: np.ndarray
    ) -> CompressionQuality:
        """Compute ALL metrics between original and reconstructed.

        Parameters
        ----------
        original : np.ndarray
            Original (reference) tensor.
        reconstructed : np.ndarray
            Reconstructed (compressed/decompressed) tensor.

        Returns
        -------
        CompressionQuality
            Fully populated quality dataclass with every metric.
        """
        m = compute_all_metrics(original, reconstructed)
        return CompressionQuality(
            mse=m["mse"],
            rmse=m["rmse"],
            mae=m["mae"],
            nmse=m["nmse"],
            snr_db=m["snr_db"],
            psnr_db=m["psnr_db"],
            relative_error=m["relative_error"],
            cosine_similarity=m["cosine_similarity"],
            max_abs_error=m["max_abs_error"],
            ssim=m["ssim"],
            spectral_angle=m["spectral_angle"],
            histogram_overlap=m["histogram_overlap"],
            kld_divergence=m["kld_divergence"],
            wasserstein_distance=m["wasserstein_distance"],
            ks_statistic=m["ks_statistic"],
            ks_p_value=m["ks_p_value"],
            correlation_coefficient=m["correlation_coefficient"],
            effective_rank_ratio=m["effective_rank_ratio"],
            bit_error_rate=m["bit_error_rate"],
        )

    def assess_batch(
        self,
        pairs: List[Tuple[str, np.ndarray, np.ndarray]],
    ) -> Dict[str, CompressionQuality]:
        """Assess multiple tensor pairs at once.

        Parameters
        ----------
        pairs : List[Tuple[str, np.ndarray, np.ndarray]]
            List of (name, original, reconstructed) tuples.

        Returns
        -------
        Dict[str, CompressionQuality]
            Name → CompressionQuality mapping.
        """
        return {name: self.assess(orig, recon) for name, orig, recon in pairs}

    def compare(self, results: List[CompressionQuality]) -> Dict[str, Any]:
        """Compare multiple compression results, ranked by composite score.

        Parameters
        ----------
        results : List[CompressionQuality]
            List of CompressionQuality instances to compare.

        Returns
        -------
        Dict[str, Any]
            Comparison summary with 'rankings', 'best', 'worst', 'average',
            'std_dev' of composite scores, and 'summary_table'.
        """
        if not results:
            return {
                "rankings": [],
                "best": None,
                "worst": None,
                "average": 0.0,
                "std_dev": 0.0,
                "summary_table": "",
            }

        scored = [(q.composite_score(), q.grade(), q) for q in results]
        scored.sort(key=lambda x: -x[0])

        scores = [s[0] for s in scored]
        avg = float(np.mean(scores)) if scores else 0.0
        std = float(np.std(scores)) if scores else 0.0

        lines = [
            f"{'Rank':<6} {'Grade':<6} {'Score':<8} {'MSE':<12} {'SNR':<10} {'RelErr':<10} {'SSIM':<8}",
            f"{'-' * 6:6} {'-' * 6:6} {'-' * 8:8} {'-' * 12:12} {'-' * 10:10} {'-' * 10:10} {'-' * 8:8}",
        ]
        for rank, (score, grade, q) in enumerate(scored, 1):
            lines.append(
                f"{rank:<6} {grade:<6} {score:<8.4f} {q.mse:<12.6e} {q.snr_db:<10.2f} {q.relative_error:<10.6f} {q.ssim:<8.4f}"
            )

        return {
            "rankings": [(q.grade(), q.composite_score(), q) for _, _, q in scored],
            "best": scored[0][2] if scored else None,
            "worst": scored[-1][2] if scored else None,
            "average": avg,
            "std_dev": std,
            "summary_table": "\n".join(lines),
        }

    def rate_distortion_curve(
        self,
        original: np.ndarray,
        methods: List[Callable],
        ratios: List[float],
    ) -> Dict[str, Any]:
        """Generate full rate-distortion data for plotting.

        Each method is called as ``method(tensor, ratio)`` and must return
        a reconstructed tensor. Distortion is measured as MSE (or 1/SNR).

        Parameters
        ----------
        original : np.ndarray
            Original tensor.
        methods : List[Callable]
            List of compression callables: ``reconstructed = method(original, ratio)``.
        ratios : List[float]
            Target compression ratios to test.

        Returns
        -------
        Dict[str, Any]
            Nested dict: ``{method_name: [(ratio, mse, snr_db, quality), ...]}``
        """
        curves: Dict[str, List[Tuple[float, float, float, CompressionQuality]]] = {}
        for method_fn in methods:
            name = getattr(method_fn, "__name__", str(method_fn))
            points: List[Tuple[float, float, float, CompressionQuality]] = []
            for ratio in ratios:
                try:
                    recon = method_fn(original, ratio)
                    q = self.assess(original, recon)
                    points.append((ratio, q.mse, q.snr_db, q))
                except Exception:
                    points.append(
                        (ratio, float("nan"), float("nan"), CompressionQuality())
                    )
            curves[name] = points
        return curves
