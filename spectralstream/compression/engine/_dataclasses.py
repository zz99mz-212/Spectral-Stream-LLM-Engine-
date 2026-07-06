"""Data classes for the compression engine."""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ._constants import QUALITY_GRADE_THRESHOLDS


@dataclass(slots=True)
class CompressionConfig:
    target_ratio: float = 5000.0
    max_error: float = 0.0002
    min_ratio: float = 500.0
    streaming: bool = True
    cache_size_gb: float = 2.0
    num_workers: int = 4
    max_memory_gb: float = 4.0
    memory_budget_mb: int = 256
    precision: str = "float32"
    quality_safety_margin: float = 1.5
    enable_bootstrap: bool = True
    enable_calibration: bool = False
    checkpoint_interval: int = 50
    max_candidate_methods: int = 10
    hybrid_search_depth: int = 3
    sensitivity_adjustment_power: float = 1.5
    min_error_budget: float = 0.0001
    max_error_budget: float = 0.05
    ratio_weight: float = 1.0
    quality_weight: float = 2.0
    error_weight: float = 3.0

    def __sizeof__(self) -> int:
        return 192


@dataclass(slots=True)
class TensorProfile:
    name: str = ""
    shape: Tuple[int, ...] = (0,)
    dtype: str = ""
    native_dtype: str = ""  # Original dtype from safetensors (e.g., "BF16")
    n_elements: int = 0
    nbytes: int = 0
    tensor_type: str = "generic"
    sensitivity: float = 0.5
    sensitivity_category: str = "UNKNOWN"
    mean: float = 0.0
    std: float = 0.0
    min_val: float = 0.0
    max_val: float = 0.0
    kurtosis: float = 0.0
    skewness: float = 0.0
    dynamic_range: float = 0.0
    outlier_ratio: float = 0.0
    effective_rank: float = 0.0
    spectral_decay_rate: float = 0.0
    energy_concentration: float = 0.0
    spectral_entropy: float = 0.0
    toeplitz_score: float = 0.0
    circulant_score: float = 0.0
    block_diagonal_score: float = 0.0
    hierarchical_score: float = 0.0
    entropy_rate: float = 0.0
    mutual_information: float = 0.0
    kolmogorov_complexity: float = 0.0
    noise_floor: float = 0.0
    nm_sparsity_score: float = 0.0
    block_sparsity_score: float = 0.0
    unstructured_sparsity_score: float = 0.0
    sparsity_details: Dict[str, float] = field(default_factory=dict)
    name_sensitivity: float = 0.5
    gradient_sensitivity: Optional[float] = None
    hessian_sensitivity: Optional[float] = None
    recommended_bits: int = 8
    recommended_methods: List[str] = field(default_factory=list)
    optimal_bits: int = 8

    def __sizeof__(self) -> int:
        base = 512
        dict_overhead = 64 + 8 * len(self.sparsity_details)
        list_overhead = 56 + 8 * len(self.recommended_methods)
        str_overhead = (
            49
            + len(self.name)
            + len(self.dtype)
            + len(self.sensitivity_category)
            + len(self.tensor_type)
        )
        return base + dict_overhead + list_overhead + str_overhead


@dataclass(slots=True)
class CompressedTensor:
    _data: bytes = field(repr=False)
    method: str = ""
    params: dict = field(default_factory=dict, repr=False)
    original_shape: Tuple[int, ...] = (0,)
    original_dtype: str = ""
    compression_ratio: float = 0.0
    relative_error: float = 0.0
    snr_db: float = 0.0
    psnr_db: float = 0.0
    cosine_similarity: float = 0.0
    computation_time: float = 0.0
    method_attempts: int = 1

    def __sizeof__(self) -> int:
        base = 192
        data_overhead = 33 + len(self._data)
        params_overhead = 240 + 8 * len(self.params) if self.params else 56
        return base + data_overhead + params_overhead

    @property
    def data(self) -> bytes:
        return self._data

    def get_data_copy(self) -> bytes:
        return bytes(self._data)

    def get_data_size(self) -> int:
        return len(self._data)

    @property
    def quality_grade(self) -> str:
        if self.relative_error < QUALITY_GRADE_THRESHOLDS["S"]:
            return "S"
        if self.relative_error < QUALITY_GRADE_THRESHOLDS["A"]:
            return "A"
        if self.relative_error < QUALITY_GRADE_THRESHOLDS["B"]:
            return "B"
        if self.relative_error < QUALITY_GRADE_THRESHOLDS["C"]:
            return "C"
        if self.relative_error < QUALITY_GRADE_THRESHOLDS["D"]:
            return "D"
        return "F"


@dataclass
class CompressionReport:
    tensors: List[CompressedTensor] = field(default_factory=list, repr=False)
    total_original_bytes: int = 0
    total_compressed_bytes: int = 0
    overall_ratio: float = 1.0
    previous_best_ratio: Optional[float] = None
    average_ratio: float = 1.0
    weighted_error: float = 0.0
    avg_error: float = 0.0
    max_error: float = 0.0
    min_error: float = 0.0
    per_layer_error: Dict[str, float] = field(default_factory=dict, repr=False)
    method_distribution: Dict[str, int] = field(default_factory=dict, repr=False)
    method_error_stats: Dict[str, Dict[str, float]] = field(
        default_factory=dict, repr=False
    )
    time_seconds: float = 0.0
    profile_time: float = 0.0
    compress_time: float = 0.0
    failures: List[str] = field(default_factory=list, repr=False)
    tensor_errors: Dict[str, float] = field(default_factory=dict, repr=False)
    tensor_ratios: Dict[str, float] = field(default_factory=dict, repr=False)
    tensor_methods: Dict[str, str] = field(default_factory=dict, repr=False)
    telemetry: Dict[str, Any] = field(default_factory=dict, repr=False)
    memory_peak_mb: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "num_tensors": len(self.tensors),
            "total_original_bytes": self.total_original_bytes,
            "total_compressed_bytes": self.total_compressed_bytes,
            "overall_ratio": self.overall_ratio,
            "average_ratio": self.average_ratio,
            "weighted_error": self.weighted_error,
            "avg_error": self.avg_error,
            "max_error": self.max_error,
            "min_error": self.min_error,
            "method_distribution": dict(self.method_distribution),
            "time_seconds": self.time_seconds,
            "profile_time": self.profile_time,
            "compress_time": self.compress_time,
            "failures": list(self.failures),
        }
        if self.previous_best_ratio is not None:
            d["previous_best_ratio"] = self.previous_best_ratio
        return d

    def summary(self) -> str:
        lines = [
            "Compression Intelligence Report",
            f"  Tensors: {len(self.tensors)}",
            f"  Original: {self.total_original_bytes:,} bytes ({self.total_original_bytes / 1024 / 1024:.1f} MB)",
            f"  Compressed: {self.total_compressed_bytes:,} bytes ({self.total_compressed_bytes / 1024 / 1024:.1f} MB)",
            f"  Ratio: {self.overall_ratio:.2f}x",
            f"  Avg Error: {self.avg_error:.4%}",
            f"  Max Error: {self.max_error:.4%}",
            f"  Min Error: {self.min_error:.4%}",
            f"  Weighted Error: {self.weighted_error:.4%}",
            f"  Time: {self.time_seconds:.2f}s (profile={self.profile_time:.2f}s, compress={self.compress_time:.2f}s)",
            f"  Methods: {dict(self.method_distribution)}",
        ]
        if self.failures:
            lines.append(f"  Failures: {len(self.failures)} - {self.failures[:5]}")
        return "\n".join(lines)

    def save_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def save_gz(self, path: str) -> None:
        with gzip.open(path, "wt", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


@dataclass(slots=True)
class CompressionTelemetry:
    timestamps: Dict[str, float] = field(default_factory=dict, repr=False)
    per_tensor_stats: Dict[str, Dict[str, float]] = field(
        default_factory=dict, repr=False
    )
    method_success_rates: Dict[str, Dict[str, float]] = field(
        default_factory=dict, repr=False
    )
    memory_trace: List[Dict[str, float]] = field(default_factory=list, repr=False)


@dataclass
class CalibrationData:
    inputs: Optional[np.ndarray] = None
    activations: Optional[Dict[str, np.ndarray]] = field(default=None, repr=False)
    gradients: Optional[Dict[str, np.ndarray]] = field(default=None, repr=False)
    hessians: Optional[Dict[str, np.ndarray]] = field(default=None, repr=False)
    fisher_info: Optional[Dict[str, np.ndarray]] = field(default=None, repr=False)

    def __sizeof__(self) -> int:
        total = 64
        for arr in (self.inputs,):
            if arr is not None:
                total += arr.nbytes
        for d in (self.activations, self.gradients, self.hessians, self.fisher_info):
            if d is not None:
                total += 240
                for v in d.values():
                    total += v.nbytes
        return total

    @classmethod
    def from_random(
        cls, n_samples: int = 256, vocab_size: int = 256000, seq_len: int = 128
    ) -> CalibrationData:
        inputs = np.random.randint(0, vocab_size, size=(n_samples, seq_len)).astype(
            np.int64
        )
        return cls(inputs=inputs)

    @classmethod
    def from_numpy(cls, path: str) -> CalibrationData:
        data = np.load(path, allow_pickle=False)
        return cls(
            inputs=data.get("inputs"),
            activations=data.get("activations", None),
            gradients=data.get("gradients", None),
            hessians=data.get("hessians", None),
        )

    @classmethod
    def from_json(cls, path: str) -> CalibrationData:
        with open(path) as f:
            raw = json.load(f)
        if "inputs" in raw:
            return cls(inputs=np.array(raw["inputs"], dtype=np.int64))
        return cls()

    def compute_fisher(self, activations: Dict[str, np.ndarray]) -> CalibrationData:
        fisher: Dict[str, np.ndarray] = {}
        for name, act in activations.items():
            fisher[name] = act.astype(np.float32) ** 2
        return CalibrationData(
            inputs=self.inputs, activations=activations, fisher_info=fisher
        )
