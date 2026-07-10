# --- _certificatebuilder.py ---
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field, asdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np

from spectralstream.compression.literature_estimates import (
    LITERATURE_ESTIMATES,
    LITERATURE_DISCLAIMER,
)


class CertificateBuilder:
    """Builds CompressionCertificate from engine results."""

    @staticmethod
    def from_compression_report(
        report: "CompressionReport | Dict[str, Any]",
        tensor_names: Optional[List[str]] = None,
        model_name: str = "Unknown Model",
        model_architecture: str = "unknown",
        model_params: str = "unknown",
    ) -> CompressionCertificate:
        """Build a certificate from a CompressionReport or dict.

        Accepts either a :class:`CompressionReport` instance or a plain dict
        (as returned by :meth:`CompressionIntelligenceEngine.compress_dict`).
        Uses :class:`QualityAssessor` to compute ALL metrics for each tensor.
        Falls back gracefully if the original tensor data is unavailable.
        """
        from spectralstream.core.math_primitives.quality import QualityAssessor

        qa = QualityAssessor()

        # ── Normalise dict → report-like access ──────────────────────────
        if isinstance(report, dict):
            tensors: list = report.get("tensors", [])
            is_dict = True
        else:
            tensors = list(report.tensors)
            is_dict = False

        if tensor_names is None:
            if is_dict:
                tmethods = report.get("tensor_methods", {})
                tensor_names = list(tmethods.keys())
            else:
                tensor_names = list(report.tensor_methods.keys())
            if not tensor_names:
                tensor_names = [f"tensor_{i}" for i in range(len(tensors))]

        certificates = []
        for i, ct in enumerate(tensors):
            name = tensor_names[i] if i < len(tensor_names) else f"tensor_{i}"
            mse = ct.relative_error**2

            try:
                quality = qa.assess(
                    ct.original_tensor
                    if hasattr(ct, "original_tensor") and ct.original_tensor is not None
                    else np.array([0.0]),
                    ct.data
                    if hasattr(ct, "data") and isinstance(ct.data, np.ndarray)
                    else np.array([0.0]),
                )
                if not hasattr(ct, "original_tensor") or ct.original_tensor is None:
                    quality.mse = mse
                    quality.snr_db = ct.snr_db
                    quality.psnr_db = ct.psnr_db
                    quality.cosine_similarity = ct.cosine_similarity
                    quality.relative_error = ct.relative_error
                has_full = (
                    hasattr(ct, "original_tensor") and ct.original_tensor is not None
                )
            except Exception:
                quality = None
                has_full = False

            if quality is not None and has_full:
                cert = TensorCertificate(
                    name=name,
                    shape=ct.original_shape,
                    original_dtype=ct.original_dtype,
                    original_bytes=int(ct.compression_ratio * len(ct.data)),
                    compressed_bytes=len(ct.data),
                    compression_ratio=ct.compression_ratio,
                    method=ct.method,
                    method_category="",
                    relative_error=quality.relative_error,
                    snr_db=quality.snr_db,
                    psnr_db=quality.psnr_db,
                    cosine_similarity=quality.cosine_similarity,
                    mse=quality.mse,
                    compression_time_ms=ct.computation_time * 1000,
                    decompression_time_ms=0.0,
                    quality_grade=quality.grade(),
                    rmse=quality.rmse,
                    mae=quality.mae,
                    nmse=quality.nmse,
                    max_abs_error=quality.max_abs_error,
                    ssim=quality.ssim,
                    spectral_angle=quality.spectral_angle,
                    histogram_overlap=quality.histogram_overlap,
                    kld_divergence=quality.kld_divergence,
                    wasserstein_distance=quality.wasserstein_distance,
                    ks_statistic=quality.ks_statistic,
                    ks_p_value=quality.ks_p_value,
                    correlation_coefficient=quality.correlation_coefficient,
                    effective_rank_ratio=quality.effective_rank_ratio,
                    bit_error_rate=quality.bit_error_rate,
                    composite_score=quality.composite_score(),
                )
            else:
                cert = TensorCertificate(
                    name=name,
                    shape=ct.original_shape,
                    original_dtype=ct.original_dtype,
                    original_bytes=int(ct.compression_ratio * len(ct.data)),
                    compressed_bytes=len(ct.data),
                    compression_ratio=ct.compression_ratio,
                    method=ct.method,
                    method_category="",
                    relative_error=ct.relative_error,
                    snr_db=ct.snr_db,
                    psnr_db=ct.psnr_db,
                    cosine_similarity=ct.cosine_similarity,
                    mse=mse,
                    compression_time_ms=ct.computation_time * 1000,
                    decompression_time_ms=0.0,
                    quality_grade=ct.quality_grade,
                )
            certificates.append(cert)

        errors = [c.relative_error for c in certificates]
        snrs = [c.snr_db for c in certificates if c.snr_db != float("inf")]

        # ── Extract aggregate fields ──────────────────────────────────────
        if is_dict:
            r: Dict[str, Any] = report
            total_orig_bytes = int(r.get("total_orig_bytes", 0))
            total_comp_bytes = int(r.get("total_compressed_bytes", 0))
            overall_ratio = float(r.get("overall_ratio", 1.0))
            n_tensors = int(r.get("num_tensors", len(tensors)))
            time_sec = float(r.get("time_seconds", 0.0))
            weighted = float(r.get("weighted_error", 0.0))
            avg = float(r.get("avg_error", 0.0))
            mx = float(r.get("max_error", 0.0))
            md = dict(r.get("method_distribution", {}))
        else:
            total_orig_bytes = report.total_original_bytes
            total_comp_bytes = report.total_compressed_bytes
            overall_ratio = report.overall_ratio
            n_tensors = len(report.tensors)
            time_sec = report.time_seconds
            weighted = report.weighted_error
            avg = report.avg_error
            mx = report.max_error
            md = dict(report.method_distribution)

        return CompressionCertificate(
            model_name=model_name,
            model_path="",
            model_architecture=model_architecture,
            model_params=model_params,
            total_original_bytes=total_orig_bytes,
            total_compressed_bytes=total_comp_bytes,
            overall_ratio=overall_ratio,
            total_tensors=n_tensors,
            compression_time_seconds=time_sec,
            weighted_error=weighted,
            avg_error=avg,
            max_error=mx,
            min_error=min(errors) if errors else 0,
            avg_snr_db=float(np.mean(snrs)) if snrs else 0,
            tensor_certificates=certificates,
            method_distribution=md,
        )

    @staticmethod
    def from_compressed_tensors(
        tensors: List[Tuple[str, "CompressedTensor"]],
        model_name: str = "Unknown",
        compression_time: float = 0.0,
    ) -> CompressionCertificate:
        """Build certificate from individual compressed tensors with names.

        Attempts to compute full metrics via QualityAssessor if the
        compressed tensor carries an ``original_tensor`` attribute.
        """
        from spectralstream.core.math_primitives.quality import QualityAssessor

        qa = QualityAssessor()

        certificates = []
        total_orig = 0
        total_comp = 0
        method_dist: Dict[str, int] = {}

        for tensor_name, ct in tensors:
            orig_bytes = int(ct.compression_ratio * len(ct.data))
            comp_bytes = len(ct.data)
            mse = ct.relative_error**2

            try:
                orig_t = getattr(ct, "original_tensor", None)
                if orig_t is not None:
                    quality = qa.assess(orig_t, ct.data)
                    has_full = True
                else:
                    quality = None
                    has_full = False
            except Exception:
                quality = None
                has_full = False

            if quality is not None and has_full:
                cert = TensorCertificate(
                    name=tensor_name,
                    shape=ct.original_shape,
                    original_dtype=ct.original_dtype,
                    original_bytes=orig_bytes,
                    compressed_bytes=comp_bytes,
                    compression_ratio=ct.compression_ratio,
                    method=ct.method,
                    method_category="",
                    relative_error=quality.relative_error,
                    snr_db=quality.snr_db,
                    psnr_db=quality.psnr_db,
                    cosine_similarity=quality.cosine_similarity,
                    mse=quality.mse,
                    compression_time_ms=ct.computation_time * 1000,
                    decompression_time_ms=0.0,
                    quality_grade=quality.grade(),
                    rmse=quality.rmse,
                    mae=quality.mae,
                    nmse=quality.nmse,
                    max_abs_error=quality.max_abs_error,
                    ssim=quality.ssim,
                    spectral_angle=quality.spectral_angle,
                    histogram_overlap=quality.histogram_overlap,
                    kld_divergence=quality.kld_divergence,
                    wasserstein_distance=quality.wasserstein_distance,
                    ks_statistic=quality.ks_statistic,
                    ks_p_value=quality.ks_p_value,
                    correlation_coefficient=quality.correlation_coefficient,
                    effective_rank_ratio=quality.effective_rank_ratio,
                    bit_error_rate=quality.bit_error_rate,
                    composite_score=quality.composite_score(),
                )
            else:
                cert = TensorCertificate(
                    name=tensor_name,
                    shape=ct.original_shape,
                    original_dtype=ct.original_dtype,
                    original_bytes=orig_bytes,
                    compressed_bytes=comp_bytes,
                    compression_ratio=ct.compression_ratio,
                    method=ct.method,
                    method_category="",
                    relative_error=ct.relative_error,
                    snr_db=ct.snr_db,
                    psnr_db=ct.psnr_db,
                    cosine_similarity=ct.cosine_similarity,
                    mse=mse,
                    compression_time_ms=ct.computation_time * 1000,
                    decompression_time_ms=0.0,
                    quality_grade=ct.quality_grade,
                )
            certificates.append(cert)
            total_orig += orig_bytes
            total_comp += comp_bytes
            method_dist[ct.method] = method_dist.get(ct.method, 0) + 1

        errors = [c.relative_error for c in certificates]
        snrs = [c.snr_db for c in certificates if c.snr_db != float("inf")]

        return CompressionCertificate(
            model_name=model_name,
            model_path="",
            model_architecture="auto-detected",
            model_params="unknown",
            total_original_bytes=total_orig,
            total_compressed_bytes=total_comp,
            overall_ratio=total_orig / max(total_comp, 1),
            total_tensors=len(tensors),
            compression_time_seconds=compression_time,
            weighted_error=float(np.mean(errors)) if errors else 0,
            avg_error=float(np.mean(errors)) if errors else 0,
            max_error=max(errors) if errors else 0,
            min_error=min(errors) if errors else 0,
            avg_snr_db=float(np.mean(snrs)) if snrs else 0,
            tensor_certificates=certificates,
            method_distribution=method_dist,
        )


# --- _compressioncertificate.py ---

import json
import math
import time
from dataclasses import dataclass, field, asdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np


@dataclass
class CompressionCertificate:
    """Full compression certificate for an entire model."""

    # Model info
    model_name: str
    model_path: str
    model_architecture: str
    model_params: str

    # Overall stats
    total_original_bytes: int
    total_compressed_bytes: int
    overall_ratio: float
    total_tensors: int
    compression_time_seconds: float

    # Quality
    weighted_error: float
    avg_error: float
    max_error: float
    min_error: float
    avg_snr_db: float

    # Per-tensor certificates
    tensor_certificates: List[TensorCertificate] = field(default_factory=list)

    # Method distribution
    method_distribution: Dict[str, int] = field(default_factory=dict)

    # Grade distribution
    grade_distribution: Dict[str, int] = field(
        default_factory=lambda: {"S": 0, "A": 0, "B": 0, "C": 0, "D": 0, "F": 0}
    )

    # Per-method-grade breakdown
    method_grade_breakdown: Dict[str, Dict[str, int]] = field(default_factory=dict)

    # Industry comparison
    industry_comparison: Dict[str, Any] = field(default_factory=dict)

    # Marketing metrics
    marketing_highlights: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        self._compute_distributions()
        self._compute_marketing_highlights()

    @property
    def avg_ssim(self) -> float:
        ssims = [
            c.ssim
            for c in self.tensor_certificates
            if c.ssim != 0.0 or not all(v == 0.0 for v in [c.mse, c.relative_error])
        ]
        return float(np.mean(ssims)) if ssims else 0.0

    @property
    def avg_composite_score(self) -> float:
        scores = [
            c.composite_score
            for c in self.tensor_certificates
            if c.composite_score > 0.0
        ]
        return float(np.mean(scores)) if scores else 0.0

    def _compute_distributions(self):
        for cert in self.tensor_certificates:
            self.grade_distribution[cert.quality_grade] = (
                self.grade_distribution.get(cert.quality_grade, 0) + 1
            )
            method = cert.method
            self.method_distribution[method] = (
                self.method_distribution.get(method, 0) + 1
            )
        for cert in self.tensor_certificates:
            method = cert.method
            grade = cert.quality_grade
            if method not in self.method_grade_breakdown:
                self.method_grade_breakdown[method] = {
                    "S": 0,
                    "A": 0,
                    "B": 0,
                    "C": 0,
                    "D": 0,
                    "F": 0,
                }
            self.method_grade_breakdown[method][grade] = (
                self.method_grade_breakdown[method].get(grade, 0) + 1
            )

    def _compute_marketing_highlights(self):
        """Generate marketing-worthy statistics."""
        ratios = [
            c.compression_ratio
            for c in self.tensor_certificates
            if c.compression_ratio > 0
        ]
        errors = [c.relative_error for c in self.tensor_certificates]

        self.marketing_highlights = {
            "compression_power": f"{self.overall_ratio:.1f}x",
            "original_size_gb": f"{self.total_original_bytes / 1e9:.2f}",
            "compressed_size_gb": f"{self.total_compressed_bytes / 1e9:.2f}",
            "space_saved_gb": f"{(self.total_original_bytes - self.total_compressed_bytes) / 1e9:.2f}",
            "accuracy_preserved": f"{100 - self.avg_error * 100:.4f}%",
            "signal_quality": f"{self.avg_snr_db:.1f} dB",
            "s_grade_tensors": f"{self.grade_distribution.get('S', 0)}",
            "a_grade_or_better": f"{self.grade_distribution.get('S', 0) + self.grade_distribution.get('A', 0)}",
            "fastest_method": (
                max(self.method_distribution, key=lambda k: self.method_distribution[k])
                if self.method_distribution
                else "N/A"
            ),
            "methods_used": str(len(self.method_distribution)),
            "time_saved": self._estimate_download_time_saved(),
        }
        self._compute_industry_comparison()

    def _compute_industry_comparison(self):
        """Compare compression power against known methods."""
        ratio = self.overall_ratio
        # Build comparisons from the literature_estimates module + current run
        comparisons = list(LITERATURE_ESTIMATES) + [
            ("SpectralStream (current)", round(ratio, 1), "This run", "hybrid")
        ]
        better_count = sum(1 for _, r, _, _ in comparisons if r < ratio and r != ratio)
        total_known = sum(1 for _, r, _, _ in comparisons if r != ratio)
        rank = sum(1 for _, r, _, _ in comparisons if r >= ratio)
        self.industry_comparison = {
            "comparisons": [
                {
                    "name": n,
                    "ratio": r,
                    "description": d,
                    "type": t,
                    "beats": ratio > r if r != ratio else None,
                }
                for n, r, d, t in comparisons
            ],
            "beats_standard_quant": ratio > 4.0,
            "beats_int4": ratio > 8.0,
            "rank": f"{rank}/{total_known}",
            "better_than_count": better_count,
            "total_compared": total_known,
            "disclaimer": LITERATURE_DISCLAIMER,
        }

    def _estimate_download_time_saved(self) -> str:
        saved_bytes = self.total_original_bytes - self.total_compressed_bytes
        mb_saved = saved_bytes / 1e6
        seconds_saved = mb_saved * 8 / 100
        if seconds_saved > 3600:
            return f"{seconds_saved / 3600:.1f} hours"
        elif seconds_saved > 60:
            return f"{seconds_saved / 60:.1f} minutes"
        return f"{seconds_saved:.0f} seconds"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": {
                "name": self.model_name,
                "architecture": self.model_architecture,
                "parameters": self.model_params,
                "original_size_gb": round(self.total_original_bytes / 1e9, 2),
                "compressed_size_gb": round(self.total_compressed_bytes / 1e9, 2),
            },
            "compression": {
                "ratio": round(self.overall_ratio, 2),
                "time_seconds": round(self.compression_time_seconds, 2),
                "methods_used": len(self.method_distribution),
            },
            "quality": {
                "avg_error_percent": round(self.avg_error * 100, 4),
                "max_error_percent": round(self.max_error * 100, 4),
                "min_error_percent": round(self.min_error * 100, 4),
                "avg_snr_db": round(self.avg_snr_db, 2),
                "avg_ssim": round(self.avg_ssim, 4),
                "avg_composite_score": round(self.avg_composite_score, 4),
                "grade_distribution": self.grade_distribution,
            },
            "marketing": self.marketing_highlights,
            "method_grade_breakdown": self.method_grade_breakdown,
            "industry_comparison": self.industry_comparison,
            "tensors": [c.to_dict() for c in self.tensor_certificates],
            "method_distribution": self.method_distribution,
        }

    def to_json(self) -> str:
        """Return JSON string of the certificate."""
        return json.dumps(self.to_dict(), indent=2, default=str)

    def to_html(self) -> str:
        """Generate a beautiful HTML certificate page."""
        highlights = self.marketing_highlights

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Compression Certificate — {self.model_name}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
          max-width: 1200px; margin: 0 auto; padding: 20px; background: #0a0a0f; color: #e0e0e0; }}
  h1, h2, h3 {{ color: #ffffff; }}
  .header {{ text-align: center; padding: 40px;
             background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
             border-radius: 16px; margin-bottom: 30px; }}
  .header h1 {{ font-size: 2.5em; margin: 0; }}
  .header .subtitle {{ color: #8888ff; font-size: 1.2em; }}
  .badge-container {{ display: flex; justify-content: center; gap: 20px; flex-wrap: wrap; margin: 20px 0; }}
  .badge {{ background: #1a1a2e; border: 1px solid #333; border-radius: 12px; padding: 20px 30px;
            text-align: center; min-width: 150px; }}
  .badge .value {{ font-size: 2em; font-weight: bold; color: #00ff88; }}
  .badge .label {{ font-size: 0.85em; color: #8888ff; margin-top: 5px; }}
  .badge.gold .value {{ color: #ffd700; }}
  .badge.purple .value {{ color: #b388ff; }}
  .section {{ background: #1a1a2e; border-radius: 12px; padding: 25px; margin: 20px 0; }}
  table {{ width: 100%; border-collapse: collapse; }}
  th, td {{ padding: 10px 15px; text-align: left; border-bottom: 1px solid #333; }}
  th {{ color: #8888ff; font-weight: 600; text-transform: uppercase; font-size: 0.85em; }}
  .grade-S {{ color: #00ff88; }}
  .grade-A {{ color: #00cc66; }}
  .grade-B {{ color: #ffd700; }}
  .grade-C {{ color: #ff8c00; }}
  .grade-D {{ color: #ff4444; }}
  .grade-F {{ color: #ff0000; }}
  .method-dist {{ display: flex; gap: 10px; flex-wrap: wrap; }}
  .method-tag {{ background: #2a2a4e; padding: 5px 12px; border-radius: 20px; font-size: 0.85em; }}
  .footer {{ text-align: center; color: #666; margin-top: 40px; padding: 20px; }}
  .progress-bar {{ background: #333; border-radius: 10px; height: 20px; overflow: hidden; margin: 5px 0; }}
  .progress-fill {{ height: 100%; transition: width 0.5s; background: linear-gradient(90deg, #00ff88, #00cc66); }}
</style>
</head>
<body>

<div class="header">
  <h1>🔬 Compression Certificate</h1>
  <div class="subtitle">SpectralStream Intelligence Engine</div>
  <p style="color: #888;">{self.model_name} · {self.model_architecture}</p>
</div>

<div class="badge-container">
  <div class="badge">
    <div class="value">{highlights["compression_power"]}</div>
    <div class="label">Compression Ratio</div>
  </div>
  <div class="badge gold">
    <div class="value">{highlights["accuracy_preserved"]}</div>
    <div class="label">Accuracy Preserved</div>
  </div>
  <div class="badge purple">
    <div class="value">{highlights["signal_quality"]}</div>
    <div class="label">Signal Quality (SNR)</div>
  </div>
  <div class="badge">
    <div class="value">{highlights["space_saved_gb"]} GB</div>
    <div class="label">Space Saved</div>
  </div>
</div>

<div class="section">
  <h2>📊 Overall Statistics</h2>
  <table>
    <tr><td>Original Size</td><td>{self.total_original_bytes / 1e9:.2f} GB</td></tr>
    <tr><td>Compressed Size</td><td>{self.total_compressed_bytes / 1e9:.2f} GB</td></tr>
    <tr><td>Compression Ratio</td><td><strong>{self.overall_ratio:.2f}x</strong></td></tr>
    <tr><td>Number of Tensors</td><td>{self.total_tensors}</td></tr>
    <tr><td>Average Error</td><td>{self.avg_error * 100:.4f}%</td></tr>
    <tr><td>Maximum Error</td><td>{self.max_error * 100:.4f}%</td></tr>
    <tr><td>Average SNR</td><td>{self.avg_snr_db:.2f} dB</td></tr>
    <tr><td>Average SSIM</td><td>{self.avg_ssim:.4f}</td></tr>
    <tr><td>Composite Score</td><td>{self.avg_composite_score:.4f}</td></tr>
    <tr><td>Compression Time</td><td>{self.compression_time_seconds:.1f}s</td></tr>
    <tr><td>Methods Used</td><td>{len(self.method_distribution)}</td></tr>
  </table>
</div>

<div class="section">
  <h2>🏆 Grade Distribution</h2>
  <table>
    <tr><th>Grade</th><th>Count</th><th>Distribution</th></tr>
"""
        total = sum(self.grade_distribution.values()) or 1
        for grade in ["S", "A", "B", "C", "D", "F"]:
            count = self.grade_distribution.get(grade, 0)
            pct = count / total * 100
            html += f"""
    <tr>
      <td class="grade-{grade}"><strong>{grade}</strong></td>
      <td>{count}</td>
      <td><div class="progress-bar"><div class="progress-fill" style="width:{pct}%"></div></div></td>
    </tr>"""

        html += """
  </table>
</div>

<div class="section">
  <h2>🔧 Method Distribution</h2>
  <div class="method-dist">
"""
        for method, count in sorted(
            self.method_distribution.items(), key=lambda x: -x[1]
        ):
            html += f'    <span class="method-tag">{method} ({count})</span>\n'

        html += """
  </div>
</div>

<div class="section">
  <h2>📊 Per-Method Grade Breakdown</h2>
  <table>
    <tr><th>Method</th><th>S</th><th>A</th><th>B</th><th>C</th><th>D</th><th>F</th></tr>
"""
        for method, grades in sorted(self.method_grade_breakdown.items()):
            s = grades.get("S", 0)
            a = grades.get("A", 0)
            b = grades.get("B", 0)
            c = grades.get("C", 0)
            d = grades.get("D", 0)
            f = grades.get("F", 0)
            html += f"""
    <tr>
      <td>{method}</td>
      <td class="grade-S">{s}</td>
      <td class="grade-A">{a}</td>
      <td class="grade-B">{b}</td>
      <td class="grade-C">{c}</td>
      <td class="grade-D">{d}</td>
      <td class="grade-F">{f}</td>
    </tr>"""
        html += """
  </table>
</div>

<div class="section">
  <h2>📋 Per-Tensor Report</h2>
  <table>
    <tr><th>Tensor</th><th>Shape</th><th>Method</th><th>Ratio</th><th>Error</th><th>SNR</th><th>SSIM</th><th>Score</th><th>Grade</th></tr>
"""
        for cert in self.tensor_certificates:
            shape_str = "×".join(str(s) for s in cert.shape[:3])
            if len(cert.shape) > 3:
                shape_str += "…"
            html += f"""
    <tr>
      <td style="font-size:0.85em">{cert.name[:50]}</td>
      <td>{shape_str}</td>
      <td>{cert.method}</td>
      <td>{cert.compression_ratio:.2f}x</td>
      <td class="grade-{cert.quality_grade}">{cert.relative_error * 100:.4f}%</td>
      <td>{cert.snr_db:.1f}</td>
      <td>{cert.ssim:.4f}</td>
      <td>{cert.composite_score:.4f}</td>
      <td class="grade-{cert.quality_grade}"><strong>{cert.quality_grade}</strong></td>
    </tr>"""

        html += """
  </table>
</div>

<div class="section">
  <h2>🏆 Industry Comparison</h2>
  <table>
    <tr><th>Method</th><th>Ratio</th><th>Description</th><th>Comparison</th></tr>
"""
        for c in self.industry_comparison.get("comparisons", []):
            if c["ratio"] == round(self.overall_ratio, 1):
                continue
            comparison_icon = (
                "🟢 Beats"
                if c["beats"]
                else ("🔴 Loses to" if c["beats"] is False else "—")
            )
            html += f"""
    <tr>
      <td>{c["name"]}</td>
      <td>{c["ratio"]}x</td>
      <td style="font-size:0.85em">{c["description"]}</td>
      <td class="{"ok" if c.get("beats") else ("fail" if c.get("beats") is False else "")}">{comparison_icon}</td>
    </tr>"""
        html += (
            """
  </table>
  <p style="margin-top:10px;color:#888;">Rank: <strong>"""
            + self.industry_comparison.get("rank", "?")
            + """</strong> · Beats """
            + str(self.industry_comparison.get("better_than_count", 0))
            + """ of """
            + str(self.industry_comparison.get("total_compared", 0))
            + """ standard methods</p>
</div>

<div class="footer">
  <p>Generated by SpectralStream Intelligence Engine</p>
  <p style="font-size:0.85em">SpectralStream — Pure-Python LLM Inference · Advanced Compression Technology</p>
</div>

</body>
</html>"""
        )
        return html

    def to_markdown(self) -> str:
        """Generate a markdown report for READMEs."""
        lines = [
            f"# Compression Certificate — {self.model_name}",
            f"",
            f"## Overview",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Compression Ratio | **{self.overall_ratio:.2f}x** |",
            f"| Original Size | {self.total_original_bytes / 1e9:.2f} GB |",
            f"| Compressed Size | {self.total_compressed_bytes / 1e9:.2f} GB |",
            f"| Space Saved | {(self.total_original_bytes - self.total_compressed_bytes) / 1e9:.2f} GB |",
            f"| Average Error | {self.avg_error * 100:.4f}% |",
            f"| Maximum Error | {self.max_error * 100:.4f}% |",
            f"| Average SNR | {self.avg_snr_db:.2f} dB |",
            f"| Average SSIM | {self.avg_ssim:.4f} |",
            f"| Composite Score | {self.avg_composite_score:.4f} |",
            f"| Tensors | {self.total_tensors} |",
            f"| Methods Used | {len(self.method_distribution)} |",
            f"| Compression Time | {self.compression_time_seconds:.1f}s |",
            f"",
            f"## Quality Grades",
            f"| Grade | Count |",
            f"|-------|-------|",
        ]
        for grade in ["S", "A", "B", "C", "D", "F"]:
            lines.append(f"| {grade} | {self.grade_distribution.get(grade, 0)} |")

        lines.extend(
            [
                f"",
                f"## Method Distribution",
            ]
        )
        for method, count in sorted(
            self.method_distribution.items(), key=lambda x: -x[1]
        ):
            lines.append(f"- {method}: {count}")

        lines.extend(
            [
                f"",
                f"## Industry Comparison",
                f"| Method | Ratio | Result |",
                f"|--------|-------|--------|",
            ]
        )
        for c in self.industry_comparison.get("comparisons", []):
            if c["ratio"] == round(self.overall_ratio, 1):
                continue
            result = (
                "🟢 Beats"
                if c["beats"]
                else ("🔴 Loses to" if c["beats"] is False else "—")
            )
            lines.append(f"| {c['name']} | {c['ratio']}x | {result} |")
        lines.append(f"")
        lines.append(f"**Rank:** {self.industry_comparison.get('rank', '?')}")

        # Surface the anti-fabrication disclaimer
        disclaimer = self.industry_comparison.get("disclaimer", "")
        if disclaimer:
            lines.append("")
            lines.append(f"⚠️ {disclaimer}")

        return "\n".join(lines)

    def to_text(self) -> str:
        """Generate a terminal-friendly text report."""
        lines = [
            "╔══════════════════════════════════════════════════════════════╗",
            "║            SpectralStream Compression Certificate           ║",
            "╚══════════════════════════════════════════════════════════════╝",
            f"  Model: {self.model_name}",
            f"  Architecture: {self.model_architecture}",
            f"",
            f"  📊 Overall Statistics",
            f"  {'Original Size:':<25} {self.total_original_bytes / 1e9:.2f} GB",
            f"  {'Compressed Size:':<25} {self.total_compressed_bytes / 1e9:.2f} GB",
            f"  {'Compression Ratio:':<25} \033[92m{self.overall_ratio:.2f}x\033[0m",
            f"  {'Space Saved:':<25} {(self.total_original_bytes - self.total_compressed_bytes) / 1e9:.2f} GB",
            f"",
            f"  📈 Quality Metrics",
            f"  {'Average Error:':<25} {self.avg_error * 100:.4f}%",
            f"  {'Maximum Error:':<25} {self.max_error * 100:.4f}%",
            f"  {'Average SNR:':<25} {self.avg_snr_db:.2f} dB",
            f"  {'Average SSIM:':<25} {self.avg_ssim:.4f}",
            f"  {'Composite Score:':<25} {self.avg_composite_score:.4f}",
            f"  {'Tensors:':<25} {self.total_tensors}",
            f"  {'Methods Used:':<25} {len(self.method_distribution)}",
            f"",
            f"  🏆 Grade Distribution",
        ]
        for grade in ["S", "A", "B", "C", "D", "F"]:
            count = self.grade_distribution.get(grade, 0)
            bar = "█" * count + "░" * max(0, 20 - count)
            lines.append(
                f"  \033[9{'2' if grade in 'SA' else '3' if grade in 'BC' else '1'}m{grade}\033[0m {bar} {count}"
            )

        lines.append("")
        lines.append("  📊 Industry Comparison:")
        lines.append(f"  {'Rank:':<20} {self.industry_comparison.get('rank', '?')}")
        lines.append(f"  {'Beats Standard Quant:':<20} {self.industry_comparison.get('beats_standard_quant', False)}")
        lines.append(f"  {'Beats INT4:':<20} {self.industry_comparison.get('beats_int4', False)}")
        lines.append("")
        for c in self.industry_comparison.get("comparisons", []):
            if c["ratio"] == round(self.overall_ratio, 1):
                continue
            result = (
                "🟢 Beats"
                if c["beats"]
                else ("🔴 Loses to" if c["beats"] is False else "—")
            )
            lines.append(f"  • {c['name']:<25} {c['ratio']}x  {result}")
        lines.append("")
        lines.append("  ⚠️ " + self.industry_comparison.get("disclaimer", ""))

        lines.append("")
        lines.append("  📋 Per-Tensor Summary:")
        for cert in self.tensor_certificates:
            lines.append(f"  {cert.summary_line()}")

        return "\n".join(lines)

    def save(self, output_path: str, formats: Optional[List[str]] = None) -> None:
        """Save certificate in multiple formats."""
        if formats is None:
            formats = ["json", "html", "md", "txt"]
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)

        if "json" in formats:
            (p.parent / f"{p.stem}.json").write_text(
                json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8"
            )
        if "html" in formats:
            (p.parent / f"{p.stem}.html").write_text(self.to_html(), encoding="utf-8")
        if "md" in formats:
            (p.parent / f"{p.stem}.md").write_text(self.to_markdown(), encoding="utf-8")
        if "txt" in formats:
            (p.parent / f"{p.stem}.txt").write_text(self.to_text(), encoding="utf-8")


# --- _tensorcertificate.py ---


@dataclass
class TensorCertificate:
    """Certificate for a single tensor's compression."""

    name: str
    shape: Tuple[int, ...]
    original_dtype: str
    original_bytes: int
    compressed_bytes: int
    compression_ratio: float
    method: str
    method_category: str

    # Quality metrics (legacy — always populated)
    relative_error: float
    snr_db: float
    psnr_db: float
    cosine_similarity: float
    mse: float

    # Performance
    compression_time_ms: float
    decompression_time_ms: float

    # Grade
    quality_grade: str  # S, A, B, C, D, F

    # ── New comprehensive metrics (default 0.0 = not computed) ─────────
    rmse: float = 0.0
    mae: float = 0.0
    nmse: float = 0.0
    max_abs_error: float = 0.0
    ssim: float = 0.0
    spectral_angle: float = 0.0
    histogram_overlap: float = 0.0
    kld_divergence: float = 0.0
    wasserstein_distance: float = 0.0
    ks_statistic: float = 0.0
    ks_p_value: float = 0.0
    correlation_coefficient: float = 0.0
    effective_rank_ratio: float = 0.0
    bit_error_rate: float = 0.0
    composite_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def summary_line(self) -> str:
        grade_colors = {
            "S": "🟢",
            "A": "🟢",
            "B": "🟡",
            "C": "🟠",
            "D": "🔴",
            "F": "⛔",
        }
        emoji = grade_colors.get(self.quality_grade, "⚪")
        pct_error = self.relative_error * 100
        score = self.composite_score
        return (
            f"{emoji} {self.name:<48} "
            f"| {self.compression_ratio:>8.2f}x "
            f"| {pct_error:>7.4f}% err "
            f"| {self.snr_db:>6.2f} dB "
            f"| SSIM {self.ssim:<.4f} "
            f"| {score:<.4f} "
            f"| {self.quality_grade} "
            f"| {self.method:<20}"
        )


# --- _validationcertificate.py ---


@dataclass
class ValidationCertificate:
    """Certificate for SSF file validation."""

    file_path: str
    file_size: int
    n_tensors: int
    header_ok: bool
    checksum_ok: bool
    index_ok: bool
    errors: List[str] = field(default_factory=list)

    # Validation results
    tensors_validated: int = 0
    tensors_failed: int = 0
    tensor_results: List[ValidationResult] = field(default_factory=list)

    # Overall quality
    overall_ratio: float = 1.0
    avg_relative_error: float = 0.0
    max_relative_error: float = 0.0
    avg_snr_db: float = 0.0
    grade_distribution: Dict[str, int] = field(
        default_factory=lambda: {"S": 0, "A": 0, "B": 0, "C": 0, "D": 0, "F": 0}
    )
    method_distribution: Dict[str, int] = field(default_factory=dict)

    def is_valid(self) -> bool:
        return (
            self.header_ok
            and self.checksum_ok
            and self.index_ok
            and self.tensors_failed == 0
        )

    def overall_grade(self) -> str:
        s = self.grade_distribution.get("S", 0)
        a = self.grade_distribution.get("A", 0)
        total = max(sum(self.grade_distribution.values()), 1)
        s_pct = s / total
        a_pct = (s + a) / total
        if s_pct >= 0.9 and self.is_valid():
            return "S"
        if a_pct >= 0.85 and self.is_valid():
            return "A"
        if self.is_valid():
            return "B"
        if self.tensors_failed < self.n_tensors // 10:
            return "C"
        return "F"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file": self.file_path,
            "file_size": self.file_size,
            "n_tensors": self.n_tensors,
            "valid": self.is_valid(),
            "overall_grade": self.overall_grade(),
            "structural": {
                "header_ok": self.header_ok,
                "checksum_ok": self.checksum_ok,
                "index_ok": self.index_ok,
                "errors": self.errors,
            },
            "quality": {
                "overall_ratio": round(self.overall_ratio, 2),
                "avg_relative_error": round(self.avg_relative_error, 6),
                "max_relative_error": round(self.max_relative_error, 6),
                "avg_snr_db": round(self.avg_snr_db, 2),
                "grade_distribution": self.grade_distribution,
                "method_distribution": self.method_distribution,
            },
            "tensors_validated": self.tensors_validated,
            "tensors_failed": self.tensors_failed,
            "tensors": [r.__dict__ for r in self.tensor_results],
        }

    def to_html(self) -> str:
        grade_colors = {
            "S": "#00ff88",
            "A": "#00cc66",
            "B": "#ffd700",
            "C": "#ff8c00",
            "D": "#ff4444",
            "F": "#ff0000",
        }
        highlights = {
            "valid": "✓ Valid" if self.is_valid() else "✗ Invalid",
            "grade": self.overall_grade(),
            "ratio": f"{self.overall_ratio:.1f}x",
            "avg_err": f"{self.avg_relative_error * 100:.4f}%",
        }
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Validation Certificate — {self.file_path}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
          max-width: 1200px; margin: 0 auto; padding: 20px; background: #0a0a0f; color: #e0e0e0; }}
  h1, h2, h3 {{ color: #ffffff; }}
  .header {{ text-align: center; padding: 40px;
             background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
             border-radius: 16px; margin-bottom: 30px; }}
  .header h1 {{ font-size: 2.5em; margin: 0; }}
  .badge-container {{ display: flex; justify-content: center; gap: 20px; flex-wrap: wrap; margin: 20px 0; }}
  .badge {{ background: #1a1a2e; border: 1px solid #333; border-radius: 12px; padding: 20px 30px;
            text-align: center; min-width: 150px; }}
  .badge .value {{ font-size: 2em; font-weight: bold; color: #00ff88; }}
  .badge .label {{ font-size: 0.85em; color: #8888ff; margin-top: 5px; }}
  .section {{ background: #1a1a2e; border-radius: 12px; padding: 25px; margin: 20px 0; }}
  table {{ width: 100%; border-collapse: collapse; }}
  th, td {{ padding: 10px 15px; text-align: left; border-bottom: 1px solid #333; }}
  th {{ color: #8888ff; font-weight: 600; text-transform: uppercase; font-size: 0.85em; }}
  .ok {{ color: #00ff88; }}
  .fail {{ color: #ff4444; }}
  .grade-S {{ color: #00ff88; }} .grade-A {{ color: #00cc66; }} .grade-B {{ color: #ffd700; }}
  .grade-C {{ color: #ff8c00; }} .grade-D {{ color: #ff4444; }} .grade-F {{ color: #ff0000; }}
  .progress-bar {{ background: #333; border-radius: 10px; height: 20px; overflow: hidden; margin: 5px 0; }}
  .progress-fill {{ height: 100%; transition: width 0.5s; background: linear-gradient(90deg, #00ff88, #00cc66); }}
</style>
</head>
<body>
<div class="header">
  <h1>🔍 Validation Certificate</h1>
  <div class="subtitle">SpectralStream SSF Integrity & Quality Report</div>
  <p style="color: #888;">{self.file_path}</p>
</div>
<div class="badge-container">
  <div class="badge"><div class="value">{highlights["valid"]}</div><div class="label">Status</div></div>
  <div class="badge"><div class="value" style="color:{grade_colors.get(highlights["grade"], "#fff")}">{highlights["grade"]}</div><div class="label">Overall Grade</div></div>
  <div class="badge"><div class="value">{highlights["ratio"]}</div><div class="label">Compression Ratio</div></div>
  <div class="badge"><div class="value">{highlights["avg_err"]}</div><div class="label">Avg Error</div></div>
</div>
<div class="section">
  <h2>📊 Structural Integrity</h2>
  <table>
    <tr><td>Header</td><td class="{"ok" if self.header_ok else "fail"}">{"✓ Valid" if self.header_ok else "✗ Invalid"}</td></tr>
    <tr><td>File Checksum</td><td class="{"ok" if self.checksum_ok else "fail"}">{"✓ Valid" if self.checksum_ok else "✗ Invalid"}</td></tr>
    <tr><td>Index</td><td class="{"ok" if self.index_ok else "fail"}">{"✓ Valid" if self.index_ok else "✗ Invalid"}</td></tr>
    <tr><td>Tensors Validated</td><td>{self.tensors_validated}</td></tr>
    <tr><td>Tensors Failed</td><td>{self.tensors_failed}</td></tr>
    <tr><td>File Size</td><td>{self.file_size / 1e6:.1f} MB</td></tr>
  </table>
</div>
<div class="section">
  <h2>🏆 Quality Grades</h2>
  <table>
    <tr><th>Grade</th><th>Count</th><th>Distribution</th></tr>
"""
        total = sum(self.grade_distribution.values()) or 1
        for grade in ["S", "A", "B", "C", "D", "F"]:
            count = self.grade_distribution.get(grade, 0)
            pct = count / total * 100
            html += f"""
    <tr>
      <td class="grade-{grade}"><strong>{grade}</strong></td>
      <td>{count}</td>
      <td><div class="progress-bar"><div class="progress-fill" style="width:{pct}%"></div></div></td>
    </tr>"""
        html += """
  </table>
</div>
<div class="section">
  <h2>🔧 Method Distribution</h2>
"""
        for method, count in sorted(
            self.method_distribution.items(), key=lambda x: -x[1]
        ):
            html += f'    <span class="badge" style="display:inline-block;padding:5px 12px;margin:3px">{method}: {count}</span>\n'
        html += """
</div>
<div class="section">
  <h2>📋 Per-Tensor Validation</h2>
  <table>
    <tr><th>Tensor</th><th>Method</th><th>Ratio</th><th>Error</th><th>SNR</th><th>SSIM</th><th>Grade</th><th>Decompress</th><th>Checksum</th></tr>
"""
        for r in self.tensor_results:
            html += f"""
    <tr>
      <td style="font-size:0.85em">{r.name[:50]}</td>
      <td>{r.method}</td>
      <td>{r.compression_ratio:.1f}x</td>
      <td class="grade-{r.quality_grade}">{r.relative_error * 100:.4f}%</td>
      <td>{r.snr_db:.1f}</td>
      <td>{r.ssim:.4f}</td>
      <td class="grade-{r.quality_grade}"><strong>{r.quality_grade}</strong></td>
      <td class="{"ok" if r.decompression_ok else "fail"}">{"✓" if r.decompression_ok else "✗"}</td>
      <td class="{"ok" if r.checksum_ok else "fail"}">{"✓" if r.checksum_ok else "✗"}</td>
    </tr>"""
        html += """
  </table>
</div>
<div class="footer" style="text-align:center;color:#666;margin-top:40px;padding:20px;">
  <p>Generated by SpectralStream Intelligence Engine</p>
  <p style="font-size:0.85em">SpectralStream — Pure-Python LLM Inference · Advanced Compression Technology</p>
</div>
</body>
</html>"""
        return html

    def to_markdown(self) -> str:
        lines = [
            f"# Validation Certificate — {self.file_path}",
            f"",
            f"## Status: {'✓ VALID' if self.is_valid() else '✗ INVALID'} · Overall Grade: {self.overall_grade()}",
            f"",
            f"## Structural Integrity",
            f"| Check | Status |",
            f"|-------|--------|",
            f"| Header | {'✓' if self.header_ok else '✗'} |",
            f"| File Checksum | {'✓' if self.checksum_ok else '✗'} |",
            f"| Index | {'✓' if self.index_ok else '✗'} |",
            f"| Tensors | {self.tensors_validated} validated, {self.tensors_failed} failed |",
            f"| File Size | {self.file_size / 1e6:.1f} MB |",
            f"",
            f"## Quality",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Overall Ratio | {self.overall_ratio:.1f}x |",
            f"| Avg Error | {self.avg_relative_error * 100:.4f}% |",
            f"| Max Error | {self.max_relative_error * 100:.4f}% |",
            f"| Avg SNR | {self.avg_snr_db:.1f} dB |",
            f"",
            f"## Grade Distribution",
            f"| Grade | Count |",
            f"|-------|-------|",
        ]
        for grade in ["S", "A", "B", "C", "D", "F"]:
            lines.append(f"| {grade} | {self.grade_distribution.get(grade, 0)} |")

        lines.extend(
            [
                f"",
                f"## Method Distribution",
                f"| Method | Count |",
                f"|--------|-------|",
            ]
        )
        for method, count in sorted(
            self.method_distribution.items(), key=lambda x: -x[1]
        ):
            lines.append(f"| {method} | {count} |")

        lines.extend(
            [
                f"",
                f"## Per-Tensor Validation",
                f"| Tensor | Method | Ratio | Error | SNR | Grade | Decompress | Checksum |",
                f"|--------|--------|-------|-------|-----|-------|------------|----------|",
            ]
        )
        for r in self.tensor_results:
            decomp_ok = "✓" if r.decompression_ok else "✗"
            csum_ok = "✓" if r.checksum_ok else "✗"
            lines.append(
                f"| {r.name[:50]} | {r.method} | {r.compression_ratio:.1f}x "
                f"| {r.relative_error * 100:.4f}% | {r.snr_db:.1f} "
                f"| {r.quality_grade} | {decomp_ok} | {csum_ok} |"
            )

        return "\n".join(lines)

    def to_text(self) -> str:
        grade_colors_display = {
            "S": "🟢",
            "A": "🟢",
            "B": "🟡",
            "C": "🟠",
            "D": "🔴",
            "F": "⛔",
        }
        lines = [
            "╔══════════════════════════════════════════════════════════════╗",
            "║          SpectralStream SSF Validation Certificate          ║",
            "╚══════════════════════════════════════════════════════════════╝",
            f"  File: {self.file_path}",
            f"  Status: {'✓ VALID' if self.is_valid() else '✗ INVALID'}",
            f"  Overall Grade: {self.overall_grade()}",
            f"",
            f"  📊 Structural",
            f"  {'Header:':<20} {'✓' if self.header_ok else '✗'}",
            f"  {'Checksum:':<20} {'✓' if self.checksum_ok else '✗'}",
            f"  {'Index:':<20} {'✓' if self.index_ok else '✗'}",
            f"  {'File Size:':<20} {self.file_size / 1e6:.1f} MB",
            f"",
            f"  📈 Quality",
            f"  {'Ratio:':<20} {self.overall_ratio:.1f}x",
            f"  {'Avg Error:':<20} {self.avg_relative_error * 100:.4f}%",
            f"  {'Max Error:':<20} {self.max_relative_error * 100:.4f}%",
            f"  {'Avg SNR:':<20} {self.avg_snr_db:.1f} dB",
            f"",
            f"  🏆 Grade Distribution",
        ]
        for grade in ["S", "A", "B", "C", "D", "F"]:
            count = self.grade_distribution.get(grade, 0)
            bar = "█" * count + "░" * max(0, 20 - count)
            lines.append(
                f"  {grade_colors_display.get(grade, '')} {grade} {bar} {count}"
            )
        return "\n".join(lines)

    def save(self, output_path: str, formats: Optional[List[str]] = None) -> None:
        if formats is None:
            formats = ["json", "html", "md", "txt"]
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if "json" in formats:
            (p.parent / f"{p.stem}.json").write_text(
                json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8"
            )
        if "html" in formats:
            (p.parent / f"{p.stem}.html").write_text(self.to_html(), encoding="utf-8")
        if "md" in formats:
            (p.parent / f"{p.stem}.md").write_text(self.to_markdown(), encoding="utf-8")
        if "txt" in formats:
            (p.parent / f"{p.stem}.txt").write_text(self.to_text(), encoding="utf-8")


# --- _validationresult.py ---


@dataclass
class ValidationResult:
    """Result of validating a single tensor."""

    name: str
    shape: Tuple[int, ...]
    method: str
    original_size: int
    compressed_size: int
    compression_ratio: float
    relative_error: float
    snr_db: float
    psnr_db: float
    cosine_similarity: float
    mse: float
    quality_grade: str
    checksum_ok: bool
    decompression_ok: bool

    # Extended metrics
    rmse: float = 0.0
    mae: float = 0.0
    nmse: float = 0.0
    max_abs_error: float = 0.0
    ssim: float = 0.0
    spectral_angle: float = 0.0
    histogram_overlap: float = 0.0
    kld_divergence: float = 0.0
    wasserstein_distance: float = 0.0
    ks_statistic: float = 0.0
    ks_p_value: float = 0.0
    correlation_coefficient: float = 0.0
    effective_rank_ratio: float = 0.0
    bit_error_rate: float = 0.0
    composite_score: float = 0.0
