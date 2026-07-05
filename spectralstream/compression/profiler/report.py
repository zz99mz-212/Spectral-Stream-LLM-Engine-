from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from spectralstream.compression.profiler.scanner import ModelScanResult
from spectralstream.compression.profiler.analyzer import (
    TensorProfile,
    SensitivityHeatmap,
)
from spectralstream.compression.profiler.allocator import BitAllocation


@dataclass
class ProfilingReport:
    model_path: str
    model_format: str
    total_original_bytes: int = 0
    tensor_count: int = 0
    tensor_profiles: Dict[str, TensorProfile] = field(default_factory=dict)
    sensitivity_heatmap: Optional[SensitivityHeatmap] = None
    bit_allocations: List[BitAllocation] = field(default_factory=list)
    estimated_ratio_vs_quality: Dict[float, Dict[str, float]] = field(
        default_factory=dict
    )
    bottleneck_tensors: List[str] = field(default_factory=list)
    time_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_path": self.model_path,
            "model_format": self.model_format,
            "total_original_bytes": self.total_original_bytes,
            "tensor_count": self.tensor_count,
            "tensor_profiles": {
                name: {
                    "name": p.name,
                    "shape": list(p.shape),
                    "dtype": p.dtype,
                    "n_elements": p.n_elements,
                    "nbytes": p.nbytes,
                    "mean": p.mean,
                    "std": p.std,
                    "min_val": p.min_val,
                    "max_val": p.max_val,
                    "dynamic_range": p.dynamic_range,
                    "outlier_ratio": p.outlier_ratio,
                    "effective_rank": p.effective_rank,
                    "spectral_decay_rate": p.spectral_decay_rate,
                    "energy_concentration": p.energy_concentration,
                    "spectral_entropy": p.spectral_entropy,
                    "sensitivity": p.sensitivity,
                    "sensitivity_category": p.sensitivity_category,
                    "tensor_type": p.tensor_type,
                    "recommended_method": p.recommended_method,
                    "recommended_bits": p.recommended_bits,
                    "compression_difficulty": p.compression_difficulty,
                }
                for name, p in self.tensor_profiles.items()
            },
            "bit_allocations": [
                {
                    "tensor": a.tensor,
                    "bits": a.bits,
                    "method": a.method,
                    "expected_error": a.expected_error,
                    "expected_ratio": a.expected_ratio,
                    "allocation_weight": a.allocation_weight,
                }
                for a in self.bit_allocations
            ],
            "estimated_ratio_vs_quality": self.estimated_ratio_vs_quality,
            "bottleneck_tensors": self.bottleneck_tensors,
            "time_seconds": self.time_seconds,
        }

    def summary(self) -> str:
        lines = [
            "Compression Profiling Report",
            f"  Model: {self.model_path}",
            f"  Format: {self.model_format}",
            f"  Tensors: {self.tensor_count}",
            f"  Total Size: {self.total_original_bytes:,} bytes ({self.total_original_bytes / 1024 / 1024:.1f} MB)",
            f"  Time: {self.time_seconds:.2f}s",
            "",
            "  Sensitivity Distribution:",
        ]
        if self.sensitivity_heatmap:
            cats = {}
            for cat, sens in zip(
                self.sensitivity_heatmap.categories,
                self.sensitivity_heatmap.sensitivities,
            ):
                cats.setdefault(cat, []).append(sens)
            for cat, sens_list in sorted(cats.items()):
                lines.append(
                    f"    {cat}: {len(sens_list)} tensors, mean sens={np.mean(sens_list):.3f}"
                )
        lines.append("")
        lines.append("  Bottleneck Tensors (hardest to compress):")
        for bt in self.bottleneck_tensors[:10]:
            lines.append(f"    {bt}")
        if self.estimated_ratio_vs_quality:
            lines.append("")
            lines.append("  Estimated Quality vs Ratio:")
            for ratio, metrics in sorted(self.estimated_ratio_vs_quality.items()):
                err = metrics.get("expected_error", 0)
                lines.append(f"    Ratio {ratio:.0f}x: error ~ {err:.4%}")
        return "\n".join(lines)


class ReportBuilder:
    @staticmethod
    def build(
        scan: ModelScanResult,
        profiles: Dict[str, TensorProfile],
        allocations: List[BitAllocation],
        heatmap: Optional[SensitivityHeatmap],
        time_seconds: float,
    ) -> ProfilingReport:
        report = ProfilingReport(
            model_path=scan.path,
            model_format=scan.format,
            total_original_bytes=scan.total_bytes,
            tensor_count=scan.tensor_count,
            tensor_profiles=profiles,
            sensitivity_heatmap=heatmap,
            bit_allocations=allocations,
            time_seconds=time_seconds,
        )
        report.estimated_ratio_vs_quality = ReportBuilder._estimate_quality_curve(
            profiles, allocations
        )
        report.bottleneck_tensors = ReportBuilder._find_bottlenecks(profiles)
        return report

    @staticmethod
    def _estimate_quality_curve(
        profiles: Dict[str, TensorProfile], allocations: List[BitAllocation]
    ) -> Dict[float, Dict[str, float]]:
        ratios = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
        curve: Dict[float, Dict[str, float]] = {}
        for ratio in ratios:
            if not allocations:
                curve[ratio] = {"expected_error": 0.0, "expected_ratio": float(ratio)}
                continue
            avg_bits = np.mean([a.bits for a in allocations])
            error_scale = max(1.0, avg_bits / 8.0)
            expected_error = min(0.5, 0.01 * (ratio / 100.0) ** 0.7 / error_scale)
            curve[float(ratio)] = {
                "expected_error": expected_error,
                "expected_ratio": float(ratio),
            }
        return curve

    @staticmethod
    def _find_bottlenecks(
        profiles: Dict[str, TensorProfile], top_n: int = 20
    ) -> List[str]:
        scored = sorted(profiles.items(), key=lambda x: -x[1].compression_difficulty)
        return [name for name, _ in scored[:top_n]]
