"""Metrics dashboard — ASCII / rich-formatted output for terminal display.

Provides human-readable summaries, comparison tables, and simple
rate-distortion visualisation for CompressionQuality data.
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .quality import CompressionQuality, QualityAssessor


def _bar(value: float, width: int = 20) -> str:
    """Render a horizontal bar proportional to *value* in [0, 1]."""
    filled = max(0, min(int(value * width), width))
    return "█" * filled + "░" * (width - filled)


def _grade_color(grade: str) -> str:
    """ANSI colour code for a grade."""
    colors = {"S": "92", "A": "92", "B": "93", "C": "93", "D": "91", "F": "91"}
    return colors.get(grade, "0")


def format_metrics_summary(quality: CompressionQuality) -> str:
    """Return a colourised, fixed-width summary of all metrics.

    Parameters
    ----------
    quality : CompressionQuality
        Quality assessment to format.

    Returns
    -------
    str
        Multi-line ASCII table with every metric.
    """
    grade = quality.grade()
    score = quality.composite_score()
    passed = "✓ PASS" if quality.passes_threshold() else "✗ FAIL"

    lines = [
        "╔══════════════════════════════════════════════════════════════════════════╗",
        "║                     Compression Quality Report                          ║",
        "╚══════════════════════════════════════════════════════════════════════════╝",
        f"  Grade: \033[{_grade_color(grade)}m{grade}\033[0m  "
        f"  Score: {score:.4f}  Status: {passed}",
        "",
        "  ── Error Metrics ────────────────────────────────────────────────",
        f"    MSE:              {quality.mse:>16.8e}",
        f"    RMSE:             {quality.rmse:>16.8e}",
        f"    MAE:              {quality.mae:>16.8e}",
        f"    NMSE:             {quality.nmse:>16.8e}",
        f"    Relative Error:   {quality.relative_error:>16.6%}",
        f"    Max Abs Error:    {quality.max_abs_error:>16.8e}",
        "",
        "  ── Signal Quality ───────────────────────────────────────────────",
        f"    SNR:              {quality.snr_db:>10.2f} dB   {_bar(min(quality.snr_db / 60, 1.0))}",
        f"    PSNR:             {quality.psnr_db:>10.2f} dB   {_bar(min(quality.psnr_db / 100, 1.0))}",
        f"    Cosine Sim:       {quality.cosine_similarity:>10.6f}   {_bar(max(0, (quality.cosine_similarity + 1) / 2))}",
        f"    SSIM:             {quality.ssim:>10.6f}   {_bar(max(0, (quality.ssim + 1) / 2))}",
        f"    Spect Angle:      {quality.spectral_angle:>10.4f} rad",
        "",
        "  ── Distributional ───────────────────────────────────────────────",
        f"    Histogram Overlap: {quality.histogram_overlap:>9.4f}   {_bar(quality.histogram_overlap)}",
        f"    KL Divergence:    {quality.kld_divergence:>12.6e}",
        f"    Wasserstein Dist: {quality.wasserstein_distance:>12.6e}",
        f"    KS Statistic:     {quality.ks_statistic:>12.6f}",
        f"    KS p-value:       {quality.ks_p_value:>12.6f}",
        f"    Correlation:      {quality.correlation_coefficient:>10.6f}   {_bar(max(0, (quality.correlation_coefficient + 1) / 2))}",
        f"    Eff. Rank Ratio:  {quality.effective_rank_ratio:>10.6f}   {_bar(quality.effective_rank_ratio)}",
        "",
        "  ── Bit-Level ───────────────────────────────────────────────────",
        f"    Bit Error Rate:   {quality.bit_error_rate:>12.6e}   {_bar(max(0, 1.0 - quality.bit_error_rate * 1000))}",
        "",
        f"  \033[{_grade_color(grade)}m{'━' * 66}\033[0m",
        f"  Overall Grade: \033[{_grade_color(grade)}m{grade}\033[0m  ·  "
        f"Composite Score: {score:.4f} / 1.0",
    ]
    return "\n".join(lines)


def format_comparison_table(results: List[Tuple[str, CompressionQuality]]) -> str:
    """Show side-by-side method comparison as a formatted table.

    Parameters
    ----------
    results : List[Tuple[str, CompressionQuality]]
        List of (method_name, quality) pairs.

    Returns
    -------
    str
        ASCII table comparing methods across key metrics.
    """
    if not results:
        return "(no results to compare)"

    # Determine column widths
    name_w = max(len(n) for n, _ in results) + 2
    name_w = max(name_w, 12)

    header = (
        f"{'Method':<{name_w}} {'Grade':<6} {'Score':<8} "
        f"{'MSE':<12} {'SNR':<8} {'RelErr':<10} {'SSIM':<8} {'CosSim':<8} {'BER':<10}"
    )
    sep = "-" * len(header)
    lines = [header, sep]

    for name, q in results:
        grade = q.grade()
        score = q.composite_score()
        ber = q.bit_error_rate
        lines.append(
            f"{name:<{name_w}} "
            f"\033[{_grade_color(grade)}m{grade:<6}\033[0m "
            f"{score:<8.4f} "
            f"{q.mse:<12.6e} "
            f"{q.snr_db:<8.2f} "
            f"{q.relative_error:<10.6f} "
            f"{q.ssim:<8.4f} "
            f"{q.cosine_similarity:<8.4f} "
            f"{ber:<10.6e}"
        )

    lines.append(sep)
    return "\n".join(lines)


def format_rate_distortion_table(
    curves: Dict[str, List[Tuple[float, float, float, CompressionQuality]]],
) -> str:
    """Format rate-distortion data as an ASCII table.

    Parameters
    ----------
    curves : Dict[str, List[Tuple[float, float, float, CompressionQuality]]]
        Output of :meth:`QualityAssessor.rate_distortion_curve`.

    Returns
    -------
    str
        ASCII table with columns: Ratio, Method, MSE, SNR, Grade, Score.
    """
    if not curves:
        return "(no rate-distortion data)"

    # Collect all data points
    rows: List[Tuple[float, str, float, float, str, float]] = []
    for method_name, points in curves.items():
        for ratio, mse, snr, quality in points:
            if math.isnan(mse):
                continue
            rows.append(
                (
                    ratio,
                    method_name,
                    mse,
                    snr,
                    quality.grade(),
                    quality.composite_score(),
                )
            )

    rows.sort(key=lambda r: (r[0], -r[5]))

    lines = [
        f"{'Ratio':<8} {'Method':<28} {'MSE':<14} {'SNR':<10} {'Grade':<6} {'Score':<8}",
        "-" * 74,
    ]
    for ratio, method, mse, snr, grade, score in rows:
        lines.append(
            f"{ratio:<8.1f}x "
            f"{method:<28} "
            f"{mse:<14.6e} "
            f"{snr:<10.2f} "
            f"\033[{_grade_color(grade)}m{grade:<6}\033[0m "
            f"{score:<8.4f}"
        )
    return "\n".join(lines)
