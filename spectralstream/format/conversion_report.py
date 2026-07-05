"""
Conversion Report — Detailed per-tensor and per-layer conversion metrics
========================================================================
Provides ConversionReport, LayerReport, and TensorReport dataclasses
for tracking model conversion quality, compression ratios, and error metrics.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class TensorReport:
    """Per-tensor compression report."""

    name: str = ""
    shape: tuple = ()
    method: str = "raw"
    original_bytes: int = 0
    compressed_bytes: int = 0
    ratio: float = 1.0
    snr: float = 0.0
    rel_error: float = 0.0
    cos_sim: float = 1.0
    psnr: float = 0.0
    mse: float = 0.0
    layer_id: int = -1
    time_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "method": self.method,
            "original_bytes": self.original_bytes,
            "compressed_bytes": self.compressed_bytes,
            "ratio": round(self.ratio, 2),
            "snr": round(self.snr, 2),
            "rel_error": round(self.rel_error, 6),
            "cos_sim": round(self.cos_sim, 6),
            "psnr": round(self.psnr, 2),
            "mse": round(self.mse, 8),
            "layer_id": self.layer_id,
            "time_seconds": round(self.time_seconds, 4),
        }


@dataclass
class LayerReport:
    """Per-layer compression summary."""

    layer_id: int = 0
    num_tensors: int = 0
    original_bytes: int = 0
    compressed_bytes: int = 0
    ratio: float = 1.0
    avg_snr: float = 0.0
    avg_rel_error: float = 0.0
    avg_cos_sim: float = 1.0
    tensors: list[TensorReport] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "layer_id": self.layer_id,
            "num_tensors": self.num_tensors,
            "original_bytes": self.original_bytes,
            "compressed_bytes": self.compressed_bytes,
            "ratio": round(self.ratio, 2),
            "avg_snr": round(self.avg_snr, 2),
            "avg_rel_error": round(self.avg_rel_error, 6),
            "avg_cos_sim": round(self.avg_cos_sim, 6),
            "tensor_count": len(self.tensors),
        }


@dataclass
class ConversionReport:
    """Full conversion report with per-layer and per-tensor detail."""

    input_path: str = ""
    output_path: str = ""
    total_tensors: int = 0
    total_layers: int = 0
    original_size: int = 0
    compressed_size: int = 0
    ratio: float = 1.0
    max_error: float = 0.0
    avg_error: float = 0.0
    per_layer: list[LayerReport] = field(default_factory=list)
    per_tensor: list[TensorReport] = field(default_factory=list)
    time_seconds: float = 0.0
    meets_target: bool = False
    target_ratio: float = 5000.0
    target_max_error: float = 0.0002
    architecture: str = "unknown"
    model_name: str = ""
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add_tensor(self, report: TensorReport):
        self.per_tensor.append(report)
        self.total_tensors += 1
        self.original_size += report.original_bytes
        self.compressed_size += report.compressed_bytes

    def finalize(self):
        self.ratio = self.original_size / max(self.compressed_size, 1)
        if self.per_tensor:
            self.avg_error = sum(t.rel_error for t in self.per_tensor) / len(
                self.per_tensor
            )
            self.max_error = max((t.rel_error for t in self.per_tensor), default=0.0)
        self.meets_target = (
            self.ratio >= self.target_ratio and self.max_error <= self.target_max_error
        )
        self._build_layer_reports()

    def _build_layer_reports(self):
        layer_map: dict[int, list[TensorReport]] = {}
        for t in self.per_tensor:
            lid = t.layer_id
            if lid not in layer_map:
                layer_map[lid] = []
            layer_map[lid].append(t)
        self.per_layer = []
        for lid in sorted(layer_map.keys()):
            tensors = layer_map[lid]
            lr = LayerReport(
                layer_id=lid,
                num_tensors=len(tensors),
                original_bytes=sum(t.original_bytes for t in tensors),
                compressed_bytes=sum(t.compressed_bytes for t in tensors),
                tensors=tensors,
            )
            lr.ratio = lr.original_bytes / max(lr.compressed_bytes, 1)
            if tensors:
                lr.avg_snr = sum(t.snr for t in tensors) / len(tensors)
                lr.avg_rel_error = sum(t.rel_error for t in tensors) / len(tensors)
                lr.avg_cos_sim = sum(t.cos_sim for t in tensors) / len(tensors)
            self.per_layer.append(lr)
        self.total_layers = len(self.per_layer)

    def summary(self) -> str:
        lines = [
            f"Conversion Report",
            f"  Input:       {self.input_path}",
            f"  Output:      {self.output_path}",
            f"  Architecture:{self.architecture}",
            f"  Tensors:     {self.total_tensors}",
            f"  Layers:      {self.total_layers}",
            f"  Original:    {self._fmt(self.original_size)}",
            f"  Compressed:  {self._fmt(self.compressed_size)}",
            f"  Ratio:       {self.ratio:.1f}:1",
            f"  Max Error:   {self.max_error:.6f}",
            f"  Avg Error:   {self.avg_error:.6f}",
            f"  Time:        {self.time_seconds:.1f}s",
            f"  Target:      {self.target_ratio:.0f}:1 (max_err={self.target_max_error})",
            f"  Meets:       {'YES' if self.meets_target else 'NO'}",
        ]
        if self.errors:
            lines.append(f"  Errors:      {len(self.errors)}")
            for e in self.errors[:5]:
                lines.append(f"    - {e}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "input_path": self.input_path,
            "output_path": self.output_path,
            "total_tensors": self.total_tensors,
            "total_layers": self.total_layers,
            "original_size": self.original_size,
            "compressed_size": self.compressed_size,
            "ratio": round(self.ratio, 2),
            "max_error": round(self.max_error, 6),
            "avg_error": round(self.avg_error, 6),
            "time_seconds": round(self.time_seconds, 2),
            "meets_target": self.meets_target,
            "target_ratio": self.target_ratio,
            "target_max_error": self.target_max_error,
            "architecture": self.architecture,
            "model_name": self.model_name,
            "per_layer": [lr.to_dict() for lr in self.per_layer],
            "per_tensor": [tr.to_dict() for tr in self.per_tensor],
        }

    def save_json(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @staticmethod
    def _fmt(n: int) -> str:
        if n >= 1024**3:
            return f"{n / 1024**3:.2f} GB"
        if n >= 1024**2:
            return f"{n / 1024**2:.1f} MB"
        if n >= 1024:
            return f"{n / 1024:.1f} KB"
        return f"{n} B"
