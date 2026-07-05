"""WorldModelCompressor — high-level model compression using world model intelligence.

Loads all tensors from a safetensors file, builds a unified model graph via
TensorWorldModel, then compresses each tensor using the CompressionRouter with
world model guidance. Returns per-tensor results + model-level stats.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .._dataclasses import CompressionConfig
from .._io import _SafetensorsIO
from .._orchestrator import CompressionIntelligenceEngine
from .tensor_world_model import TensorWorldModel
from .compression_router import CompressionRouter

logger = logging.getLogger(__name__)


def _human_size(n: int) -> str:
    nf = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if nf < 1024:
            return f"{nf:.1f}{unit}"
        nf /= 1024
    return f"{nf:.1f}TB"


@dataclass
class ModelCompressionStats:
    """Model-level compression statistics."""

    total_tensors: int = 0
    total_original_bytes: int = 0
    total_compressed_bytes: int = 0
    overall_ratio: float = 1.0
    avg_error: float = 0.0
    avg_ratio: float = 0.0
    elapsed_seconds: float = 0.0
    failures: int = 0
    method_distribution: Dict[str, int] = field(default_factory=dict)
    type_distribution: Dict[str, int] = field(default_factory=dict)
    per_tensor_types: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def summary_lines(self) -> List[str]:
        lines = [
            "=" * 70,
            "World Model Compression Complete:",
            f"  Tensors:           {self.total_tensors}",
            f"  Original:          {_human_size(self.total_original_bytes)} ({self.total_original_bytes} bytes)",
            f"  Compressed:        {_human_size(self.total_compressed_bytes)} ({self.total_compressed_bytes} bytes)",
            f"  Overall Ratio:     {self.overall_ratio:.1f}x",
            f"  Avg Ratio:         {self.avg_ratio:.1f}x",
            f"  Avg Error:         {self.avg_error:.6f}",
            f"  Time:              {self.elapsed_seconds:.2f}s",
            f"  Failures:          {self.failures}",
        ]
        if self.method_distribution:
            lines.append(f"  Method Distribution:")
            for method, count in sorted(
                self.method_distribution.items(), key=lambda x: -x[1]
            ):
                lines.append(f"    {method:<30} {count}")
        if self.per_tensor_types:
            lines.append(f"  Per-Type Stats:")
            for ttype, stats in sorted(self.per_tensor_types.items()):
                lines.append(
                    f"    {ttype:<20} count={stats['count']:>3d}  "
                    f"ratio={stats['avg_ratio']:>8.1f}x  "
                    f"error={stats['avg_error']:.6f}"
                )
        lines.append("=" * 70)
        return lines


class WorldModelCompressor:
    """High-level model compressor using world model intelligence.

    Loads all tensors from a safetensors file, builds a unified model graph
    via TensorWorldModel, then compresses each tensor using the
    CompressionRouter's intelligent path with world model guidance.
    """

    def __init__(
        self,
        engine: Optional[CompressionIntelligenceEngine] = None,
        config: Optional[CompressionConfig] = None,
        world_model: Optional[TensorWorldModel] = None,
        max_workers: int = 4,
    ):
        if engine is None:
            engine = CompressionIntelligenceEngine(config=config or CompressionConfig())
        self._engine = engine
        self._config = config or getattr(engine, "_config", CompressionConfig())
        self._world_model = world_model or TensorWorldModel(max_workers=max_workers)
        self._router = CompressionRouter(
            engine=self._engine,
            world_model=self._world_model,
        )

    def compress_model(
        self,
        model_path: str,
        output_path: Optional[str] = None,
        target_ratio: float = 0.0,
        max_error: float = 0.0,
        progress_callback: Any = None,
    ) -> Tuple[Dict[str, Any], ModelCompressionStats]:
        """Compress an entire model using world model intelligence.

        Parameters
        ----------
        model_path : str
            Path to the .safetensors model file.
        output_path : str, optional
            Path to write compressed output (.ssf).
        target_ratio : float
            Target compression ratio (0 = auto-detect).
        max_error : float
            Maximum relative error (0 = auto-detect).
        progress_callback : callable, optional
            Progress callback (current, total, name).

        Returns
        -------
        results : dict
            Per-tensor compression results {name: {data, metadata, ratio, error}}.
        stats : ModelCompressionStats
            Model-level statistics.
        """
        t_start = time.perf_counter()

        # Resolve auto parameters
        effective_ratio = target_ratio if target_ratio > 0 else 5000.0
        effective_error = max_error if max_error > 0 else 0.01

        # --- Phase 1: Load all tensors ---
        safetensors_io = _SafetensorsIO(use_mmap=True)
        tensor_info = safetensors_io.scan(model_path)
        total = len(tensor_info)
        if total == 0:
            raise ValueError(f"No tensors found in {model_path}")

        logger.info(
            "WorldModelCompressor: loading %d tensors (%.1f MB total)",
            total,
            sum(t[3] for t in tensor_info.values()) / 1e6,
        )

        tensors: Dict[str, np.ndarray] = {}
        for name, (shape, dtype_str, offset, nbytes) in tensor_info.items():
            tensors[name] = safetensors_io.read(
                model_path, shape, dtype_str, offset, nbytes
            )

        # --- Phase 2: World model scan ---
        logger.info("WorldModelCompressor: scanning model with TensorWorldModel...")
        model_profile = self._world_model.scan_from_dict(tensors)
        logger.info(
            "World model profile: %d tensors, %d layers, %.2f GB estimated",
            model_profile.graph.n_tensors,
            model_profile.layer_count,
            model_profile.estimated_model_size_gb,
        )

        # --- Phase 3: Compress via router ---
        logger.info(
            "WorldModelCompressor: compressing %d tensors (ratio=%.0f, error=%.6f)",
            total,
            effective_ratio,
            effective_error,
        )

        results: Dict[str, Any] = {}
        total_orig = 0
        total_comp = 0
        failures = 0
        method_dist: Dict[str, int] = {}
        type_stats: Dict[str, List[float]] = {}
        error_list: List[float] = []
        ratio_list: List[float] = []

        for idx, (name, tensor) in enumerate(tensors.items()):
            if progress_callback:
                progress_callback(idx + 1, total, name)

            t0 = time.perf_counter()
            try:
                data, meta, ratio_val, error_val = self._router.compress(
                    tensor,
                    target_ratio=effective_ratio,
                    max_error=effective_error,
                    name=name,
                    use_world_model=True,
                )
                dt = time.perf_counter() - t0

                total_orig += tensor.nbytes
                comp_size = len(data) if isinstance(data, (bytes, bytearray)) else 0
                total_comp += comp_size

                method = meta.get("method", "unknown")
                method_dist[method] = method_dist.get(method, 0) + 1
                error_list.append(error_val)
                ratio_list.append(ratio_val)

                # Per-type stats
                tensor_type = "unknown"
                node = model_profile.graph.get(name)
                if node is not None:
                    tensor_type = node.tensor_type
                type_stats.setdefault(tensor_type, []).append(ratio_val)

                results[name] = {
                    "data": data,
                    "metadata": meta,
                    "ratio": ratio_val,
                    "error": error_val,
                    "time": dt,
                    "method": method,
                    "original_bytes": tensor.nbytes,
                    "compressed_bytes": comp_size,
                    "tensor_type": tensor_type,
                }

                logger.info(
                    "  [%d/%d] %-50s %-20s ratio=%8.1fx  error=%.6f  time=%.3fs",
                    idx + 1,
                    total,
                    name[-50:],
                    method,
                    ratio_val,
                    error_val,
                    dt,
                )

            except Exception as e:
                failures += 1
                logger.error(
                    "  [%d/%d] %-50s FAILED: %s", idx + 1, total, name[-50:], e
                )
                results[name] = {
                    "data": b"",
                    "metadata": {"method": "failed", "error": str(e)},
                    "ratio": 1.0,
                    "error": 1.0,
                    "time": 0.0,
                    "method": "failed",
                    "original_bytes": tensor.nbytes,
                    "compressed_bytes": 0,
                    "tensor_type": "unknown",
                }

        elapsed = time.perf_counter() - t_start
        overall_ratio = total_orig / max(total_comp, 1)

        # Build per-type summary
        per_type_dict: Dict[str, Dict[str, float]] = {}
        for ttype, ratios in type_stats.items():
            per_type_dict[ttype] = {
                "count": len(ratios),
                "avg_ratio": float(np.mean(ratios)),
                "avg_error": float(np.mean(error_list)) if error_list else 0.0,
            }

        stats = ModelCompressionStats(
            total_tensors=total,
            total_original_bytes=total_orig,
            total_compressed_bytes=total_comp,
            overall_ratio=overall_ratio,
            avg_error=float(np.mean(error_list)) if error_list else 0.0,
            avg_ratio=float(np.mean(ratio_list)) if ratio_list else 0.0,
            elapsed_seconds=elapsed,
            failures=failures,
            method_distribution=method_dist,
            type_distribution={t: len(ns) for t, ns in type_stats.items()},
            per_tensor_types=per_type_dict,
        )

        return results, stats

    def close(self) -> None:
        """Clean up resources."""
        pass
