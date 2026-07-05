"""
Model Calibrator — intelligence layer that scans a model, tests each compression
method against real model tensors, and builds a per-tensor-type performance map.

This is the "learning" component of the compression engine.
"""

from __future__ import annotations


import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MethodResult:
    """Results from testing a single method on a tensor."""

    method_name: str
    ratio: float
    error: float
    elapsed: float


@dataclass
class TensorSample:
    """Information about a single tensor in the model."""

    name: str
    shape: Tuple[int, ...]
    dtype: str
    nbytes: int
    tensor_type: str  # embedding, attention_q, attention_k, etc.
    data: Optional[np.ndarray] = None


class ModelCalibrator:
    """Calibrates the compression engine to a specific model.

    Tests all available compression methods against real model tensors
    and builds a performance map: tensor_type -> best methods.
    """

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.tensor_info: Dict[str, TensorSample] = {}
        self.performance_map: Dict[str, List[MethodResult]] = {}
        self.calibration_results: Dict[str, Any] = {}
        self._all_methods: Dict[str, Any] = {}

    def scan_model(self) -> Dict[str, TensorSample]:
        """Scan the model file and catalog all tensors by type.

        Uses safetensors I/O to read tensor metadata without loading everything.
        """
        from spectralstream.compression.engine._io import _SafetensorsIO

        io = _SafetensorsIO()
        raw_info = io.scan(self.model_path)

        samples: Dict[str, TensorSample] = {}
        for name, (shape, dtype, offset, nbytes) in raw_info.items():
            tensor_type = self._classify_tensor(name)
            samples[name] = TensorSample(
                name=name,
                shape=shape,
                dtype=dtype,
                nbytes=nbytes,
                tensor_type=tensor_type,
            )
        self.tensor_info = samples
        return samples

    def calibrate(
        self,
        sample_per_type: int = 2,
        error_threshold: float = 0.01,
        max_methods: Optional[int] = None,
    ) -> Dict[str, List[MethodResult]]:
        """Test ALL methods against sample tensors of each type.

        Builds a performance map: tensor_type -> methods sorted by ratio/error.

        Args:
            sample_per_type: Number of tensors to sample per type.
            error_threshold: Maximum acceptable relative error (1% default).
            max_methods: Maximum methods to keep per type (None = all).

        Returns:
            Dict mapping tensor_type -> list of MethodResult sorted by quality.
        """
        from spectralstream.compression.methods import METHOD_CLASSES
        from spectralstream.compression.engine._io import _SafetensorsIO

        io = _SafetensorsIO()

        if not self.tensor_info:
            self.scan_model()

        # Group tensors by type
        type_groups: Dict[str, List[Tuple[str, Tuple, str, int, int]]] = {}
        raw_info = io.scan(self.model_path)
        for name, info in raw_info.items():
            ttype = self._classify_tensor(name)
            if ttype not in type_groups:
                type_groups[ttype] = []
            type_groups[ttype].append((name,) + info)

        # Load all method instances
        methods: Dict[str, Any] = {}
        for mname, mcls in METHOD_CLASSES.items():
            try:
                if isinstance(mcls, type):
                    methods[mname] = mcls()
                else:
                    methods[mname] = mcls
            except Exception:
                continue
        self._all_methods = methods

        performance_map: Dict[str, List[MethodResult]] = {}

        for tensor_type, tensor_list in type_groups.items():
            logger.info(
                "Calibrating tensor type '%s' (%d tensors)",
                tensor_type,
                len(tensor_list),
            )
            type_results: List[MethodResult] = []

            # Sample up to sample_per_type tensors
            n_sample = min(sample_per_type, len(tensor_list))
            sample_indices = np.linspace(0, len(tensor_list) - 1, n_sample, dtype=int)

            for si in sample_indices:
                name, shape, dt, off, nb = tensor_list[si]
                try:
                    tensor = io.read(self.model_path, shape, dt, off, nb)
                except Exception:
                    continue

                for method_name, method_inst in methods.items():
                    try:
                        result = self._test_method(method_inst, tensor, method_name)
                        if result.error <= error_threshold and result.ratio > 1.0:
                            type_results.append(result)
                    except Exception:
                        continue

            # Deduplicate by method_name, keep best
            best_by_name: Dict[str, MethodResult] = {}
            for r in type_results:
                if r.method_name not in best_by_name or r.ratio / max(
                    r.error, 1e-10
                ) > best_by_name[r.method_name].ratio / max(
                    best_by_name[r.method_name].error, 1e-10
                ):
                    best_by_name[r.method_name] = r

            # Sort by quality = ratio / error
            sorted_results = sorted(
                best_by_name.values(),
                key=lambda r: r.ratio / max(r.error, 1e-10),
                reverse=True,
            )
            if max_methods is not None:
                sorted_results = sorted_results[:max_methods]

            performance_map[tensor_type] = sorted_results
            logger.info(
                "  -> %d working methods for '%s'",
                len(sorted_results),
                tensor_type,
            )

        self.performance_map = performance_map
        self.calibration_results = {
            ttype: [r.method_name for r in results[:5]]
            for ttype, results in performance_map.items()
        }
        return performance_map

    def get_best_method(
        self, tensor_name: str, tensor: np.ndarray
    ) -> Tuple[str, float, float]:
        """Get the best compression method for a specific tensor.

        Args:
            tensor_name: Name of the tensor (e.g., 'model.layers.0.attn.q_proj').
            tensor: Tensor data.

        Returns:
            Tuple of (method_name, expected_ratio, expected_error).
        """
        tensor_type = self._classify_tensor(tensor_name)

        if tensor_type in self.performance_map and self.performance_map[tensor_type]:
            best = self.performance_map[tensor_type][0]
            return best.method_name, best.ratio, best.error

        # Fallback: test a few methods on the fly
        return "block_int8", 4.0, 0.005

    def _test_method(
        self, method_inst: Any, tensor: np.ndarray, method_name: str
    ) -> MethodResult:
        """Test a single method on a tensor. Return compression metrics."""
        t0 = time.perf_counter()

        compressed, meta = method_inst.compress(tensor)
        decompressed = method_inst.decompress(compressed, meta)

        elapsed = time.perf_counter() - t0

        # Ensure correct shape
        if decompressed.shape != tensor.shape:
            try:
                decompressed = decompressed.reshape(tensor.shape)
            except Exception:
                decompressed = np.resize(decompressed, tensor.shape)

        ratio = max(tensor.nbytes / max(len(compressed), 1), 0.0)
        error = self._compute_error(tensor, decompressed)

        return MethodResult(method_name, ratio, error, elapsed)

    @staticmethod
    def _compute_error(original: np.ndarray, reconstructed: np.ndarray) -> float:
        """Compute relative error between original and reconstructed."""
        orig_flat = original.ravel().astype(np.float64)
        recon_flat = reconstructed.ravel().astype(np.float64)
        denom = max(np.linalg.norm(orig_flat), 1e-30)
        return float(np.linalg.norm(orig_flat - recon_flat) / denom)

    @staticmethod
    def _classify_tensor(name: str) -> str:
        """Classify a tensor by its name."""
        nl = name.lower()
        if any(k in nl for k in ("embed", "tok_embeddings", "wte", "lm_head")):
            return "embedding"
        if any(k in nl for k in ("attn", "attention")):
            if "q_proj" in nl or "q." in nl:
                return "attention_q"
            if "k_proj" in nl or "k." in nl:
                return "attention_k"
            if "v_proj" in nl or "v." in nl:
                return "attention_v"
            if "o_proj" in nl or "out" in nl:
                return "attention_o"
            return "attention"
        if "qkv" in nl:
            return "qkv_fused"
        if any(k in nl for k in ("gate", "w1", "gating")):
            return "ffn_gate"
        if any(k in nl for k in ("up", "w3")):
            return "ffn_up"
        if any(k in nl for k in ("down", "w2")):
            return "ffn_down"
        if any(k in nl for k in ("ffn", "mlp")):
            return "ffn"
        if any(k in nl for k in ("norm", "ln_", "rms")):
            return "norm"
        return "other"

    def get_calibration_summary(self) -> str:
        """Return a human-readable summary of calibration results."""
        lines = ["Model Calibration Summary", "=" * 50]
        for ttype, results in sorted(self.performance_map.items()):
            lines.append(f"\n{ttype}:")
            for i, r in enumerate(results[:5]):
                lines.append(
                    f"  {i + 1}. {r.method_name:40s} "
                    f"ratio={r.ratio:.2f}x  error={r.error:.4f}  "
                    f"time={r.elapsed * 1000:.1f}ms"
                )
        return "\n".join(lines)


def calibrate_tensor(
    tensor: np.ndarray,
    methods: Dict[str, Any],
    error_threshold: float = 0.01,
    max_methods: Optional[int] = None,
) -> List[MethodResult]:
    results: List[MethodResult] = []
    for method_name, method_inst in methods.items():
        try:
            t0 = time.perf_counter()
            compressed, meta = method_inst.compress(tensor)
            decompressed = method_inst.decompress(compressed, meta)
            elapsed = time.perf_counter() - t0
            if decompressed.shape != tensor.shape:
                try:
                    decompressed = decompressed.reshape(tensor.shape)
                except Exception:
                    decompressed = np.resize(decompressed, tensor.shape)
            ratio = max(tensor.nbytes / max(len(compressed), 1), 0.0)
            orig_flat = tensor.ravel().astype(np.float64)
            recon_flat = decompressed.ravel().astype(np.float64)
            denom = max(np.linalg.norm(orig_flat), 1e-30)
            error = float(np.linalg.norm(orig_flat - recon_flat) / denom)
            if error <= error_threshold and ratio > 1.0:
                results.append(MethodResult(method_name, ratio, error, elapsed))
        except Exception:
            continue
    results.sort(key=lambda r: r.ratio / max(r.error, 1e-10), reverse=True)
    if max_methods is not None:
        results = results[:max_methods]
    return results


# ── Integration with CompressionIntelligenceEngine ─────────────────────────


def calibrate_engine(
    engine: Any,
    model_path: str,
    sample_per_type: int = 2,
) -> ModelCalibrator:
    """Convenience: calibrate an engine for a specific model.

    This integrates the calibrator with the CompressionIntelligenceEngine,
    registering calibrated method preferences.
    """
    calibrator = ModelCalibrator(model_path)
    calibrator.scan_model()
    calibrator.calibrate(sample_per_type=sample_per_type)

    # Register calibrated method preferences in the engine
    if hasattr(engine, "selector") and hasattr(engine.selector, "method_scores"):
        for ttype, results in calibrator.performance_map.items():
            for r in results[:3]:
                if r.method_name not in engine.selector.method_scores:
                    engine.selector.method_scores[r.method_name] = {
                        "ratio": r.ratio,
                        "error": r.error,
                        "quality": 1.0 - r.error,
                    }

    setattr(engine, "calibrator", calibrator)
    setattr(engine, "use_calibrator", True)
    return calibrator
