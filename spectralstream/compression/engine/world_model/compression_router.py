"""Compression Router — routes between in-memory and streaming modes,
fast path and intelligent path, single-tensor and model-level compression.
Monitors memory budget and adjusts strategy dynamically.
"""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .._dataclasses import TensorProfile, CompressionConfig
from .._orchestrator import CompressionIntelligenceEngine
from ..memory_mapped_engine import MemoryMappedTensorEngine
from ..streaming_pipeline import StreamingCompressionPipeline
from .tensor_world_model import TensorWorldModel, UnifiedModelProfile
from .method_oracle import MethodOracle, RankedMethod
from .cascade_oracle import CascadeOracle, CascadePlan

logger = logging.getLogger(__name__)


class RouteMode(Enum):
    FAST_PATH = auto()
    INTELLIGENT_PATH = auto()
    STREAMING = auto()
    CASCADE = auto()
    MODEL_LEVEL = auto()


@dataclass
class RouteDecision:
    """The router's decision about how to compress."""

    mode: RouteMode = RouteMode.FAST_PATH
    use_cascade: bool = False
    use_world_model: bool = False
    use_streaming: bool = False
    use_model_level: bool = False
    chunk_size_mb: int = 256
    memory_budget_mb: int = 256
    reason: str = "default"


class CompressionRouter:
    """Routes compression requests to the optimal execution path.

    Decision tree:
    1. If tensor is a dict (multiple tensors) → model-level compression
    2. If total memory > budget → streaming (memory-mapped)
    3. If target_ratio > 500 → intelligent path (world model + oracle)
    4. If use_world_model=True → full world model path
    5. Otherwise → fast path

    Monitors memory and adjusts strategy dynamically.
    """

    def __init__(
        self,
        engine: CompressionIntelligenceEngine,
        world_model: Optional[TensorWorldModel] = None,
        method_oracle: Optional[MethodOracle] = None,
        cascade_oracle: Optional[CascadeOracle] = None,
        memory_budget_mb: int = 256,
    ):
        self._engine = engine
        self._world_model = world_model or TensorWorldModel()
        self._method_oracle = method_oracle or MethodOracle(engine)
        self._cascade_oracle = cascade_oracle or CascadeOracle(engine)
        self._memory_budget_mb = memory_budget_mb
        self._config = getattr(engine, "_config", CompressionConfig())

        # Streaming pipeline (lazy)
        self._streaming_pipeline: Optional[StreamingCompressionPipeline] = None

    @property
    def streaming_pipeline(self) -> StreamingCompressionPipeline:
        if self._streaming_pipeline is None:
            self._streaming_pipeline = StreamingCompressionPipeline(
                self._engine, self._config
            )
        return self._streaming_pipeline

    # ═══════════════════════════════════════════════════════════════════
    #  PUBLIC API
    # ═══════════════════════════════════════════════════════════════════

    def route(
        self,
        tensor_or_dict: Any,
        target_ratio: float = 5000.0,
        max_error: float = 0.01,
        name: str = "",
        use_world_model: bool = False,
        use_cascade: bool = False,
        use_streaming: bool = False,
    ) -> RouteDecision:
        """Determine the optimal compression route.

        Parameters
        ----------
        tensor_or_dict : np.ndarray or dict
            Single tensor or dict of {name: tensor}.
        target_ratio : float
            Desired compression ratio.
        max_error : float
            Maximum acceptable error.
        name : str
            Tensor name (for single-tensor mode).
        use_world_model : bool
            Force world model path.
        use_cascade : bool
            Force cascade compression.
        use_streaming : bool
            Force streaming compression.

        Returns
        -------
        RouteDecision
            The routing decision with mode and parameters.
        """
        is_dict = isinstance(tensor_or_dict, dict)

        if is_dict:
            return self._route_model_level(
                tensor_or_dict,
                target_ratio=target_ratio,
                max_error=max_error,
                use_world_model=use_world_model,
            )

        if use_streaming:
            return RouteDecision(
                mode=RouteMode.STREAMING,
                use_streaming=True,
                reason="streaming forced",
            )

        if use_world_model:
            return RouteDecision(
                mode=RouteMode.INTELLIGENT_PATH,
                use_world_model=True,
                reason="world model forced",
            )

        if use_cascade or target_ratio > 500:
            return RouteDecision(
                mode=RouteMode.CASCADE,
                use_cascade=True,
                use_world_model=True,
                reason=f"cascade for target_ratio={target_ratio}",
            )

        nbytes = tensor_or_dict.nbytes if hasattr(tensor_or_dict, "nbytes") else 0
        budget_bytes = self._memory_budget_mb * 1024 * 1024

        return RouteDecision(
            mode=RouteMode.CASCADE,
            use_cascade=True,
            use_world_model=True,
            reason=f"cascade for all sizes ({nbytes / 1024**2:.0f} MB)",
        )

    def compress(
        self,
        tensor_or_dict: Any,
        target_ratio: float = 5000.0,
        max_error: float = 0.01,
        name: str = "",
        use_world_model: bool = False,
        use_cascade: bool = False,
        use_streaming: bool = False,
        progress_callback: Any = None,
    ) -> Any:
        """Compress using the optimal route determined by the router.

        Parameters
        ----------
        tensor_or_dict : np.ndarray or dict
            Single tensor or dict of {name: tensor}.
        target_ratio : float
            Desired compression ratio.
        max_error : float
            Maximum acceptable error.
        name : str
            Tensor name (single-tensor mode).
        use_world_model : bool
            Force world-model path.
        use_cascade : bool
            Force cascade compression.
        use_streaming : bool
            Force streaming compression.
        progress_callback : callable, optional
            Progress callback for model-level compression.

        Returns
        -------
        tuple or dict
            Single tensor: (compressed, metadata, ratio, error).
            Dict of tensors: {name: (compressed, metadata, ratio, error)} or report.
        """
        decision = self.route(
            tensor_or_dict,
            target_ratio=target_ratio,
            max_error=max_error,
            name=name,
            use_world_model=use_world_model,
            use_cascade=use_cascade,
            use_streaming=use_streaming,
        )

        if decision.mode == RouteMode.MODEL_LEVEL:
            return self._compress_model_level(
                tensor_or_dict,
                target_ratio=target_ratio,
                max_error=max_error,
                use_world_model=decision.use_world_model,
                progress_callback=progress_callback,
            )

        tensor = tensor_or_dict
        if decision.mode == RouteMode.STREAMING:
            return self._compress_streaming(
                tensor,
                target_ratio=target_ratio,
                max_error=max_error,
                name=name,
            )

        if decision.mode == RouteMode.CASCADE or decision.use_world_model:
            return self._compress_intelligent(
                tensor,
                target_ratio=target_ratio,
                max_error=max_error,
                name=name,
            )

        return self._compress_fast(
            tensor,
            target_ratio=target_ratio,
            max_error=max_error,
            name=name,
        )

    # ═══════════════════════════════════════════════════════════════════
    #  INTERNAL ROUTING
    # ═══════════════════════════════════════════════════════════════════

    def _route_model_level(
        self,
        tensors: Dict[str, np.ndarray],
        target_ratio: float,
        max_error: float,
        use_world_model: bool,
    ) -> RouteDecision:
        """Route decision for model-level (dict of tensors)."""
        total_bytes = sum(t.nbytes for t in tensors.values())
        budget_bytes = self._memory_budget_mb * 1024 * 1024

        if total_bytes > budget_bytes:
            return RouteDecision(
                mode=RouteMode.STREAMING,
                use_streaming=True,
                use_world_model=True,
                memory_budget_mb=self._memory_budget_mb,
                reason=f"model too large ({total_bytes / 1024**2:.0f} MB > budget)",
            )

        if use_world_model or target_ratio > 500:
            return RouteDecision(
                mode=RouteMode.MODEL_LEVEL,
                use_model_level=True,
                use_world_model=True,
                reason=f"model-level with world model ({len(tensors)} tensors)",
            )

        return RouteDecision(
            mode=RouteMode.MODEL_LEVEL,
            use_model_level=True,
            use_world_model=False,
            reason=f"model-level fast path ({len(tensors)} tensors)",
        )

    # ═══════════════════════════════════════════════════════════════════
    #  COMPRESSION EXECUTION PATHS
    # ═══════════════════════════════════════════════════════════════════

    def _compress_fast(
        self,
        tensor: np.ndarray,
        target_ratio: float,
        max_error: float,
        name: str,
    ) -> Tuple[bytes, dict, float, float]:
        """Fast path: standard engine compress_fast."""
        logger.debug("Router: fast path for '%s'", name)
        return self._engine.compress_fast(
            tensor,
            name=name,
            target_ratio=target_ratio,
            max_error=max_error,
        )

    def _compress_intelligent(
        self,
        tensor: np.ndarray,
        target_ratio: float,
        max_error: float,
        name: str,
    ) -> Tuple[bytes, dict, float, float]:
        """Intelligent path: world model + oracle + cascade."""
        logger.debug("Router: intelligent path for '%s'", name)

        # 1. Profile tensor through world model
        profile = self._engine.profile_tensor(tensor, name)
        tensor_type = getattr(profile, "tensor_type", "weight")

        # 2. Method oracle selects best methods
        ranked = self._method_oracle.select(
            profile=profile,
            tensor_type=tensor_type,
            target_ratio=target_ratio,
            max_error=max_error,
            max_results=15,
        )

        # 3. Cascade oracle plans multi-stage cascade
        plan = self._cascade_oracle.plan(
            tensor_type=tensor_type,
            tensor=tensor,
            profile=profile,
            target_ratio=target_ratio,
            max_error=max_error,
            name=name,
        )

        # 4. Execute via cascade if viable, otherwise via method testing
        if plan is not None and plan.n_stages >= 2:
            try:
                return self._execute_cascade_plan(
                    tensor, plan, target_ratio, max_error, name
                )
            except Exception as exc:
                logger.debug("Cascade execution failed: %s", exc)

        # Fallback: use oracle-ranked methods via standard validation
        method_list = []
        for rm in ranked[:10]:
            if rm.instance is not None:
                method_list.append(
                    {
                        "instance": rm.instance,
                        "params": rm.params,
                        "name": rm.name,
                    }
                )

        if not method_list:
            return self._compress_fast(tensor, target_ratio, max_error, name)

        error_budget = max_error / max(target_ratio, 1.0)
        from .._helpers import compress_tensor_with_validation

        return compress_tensor_with_validation(
            tensor, profile, method_list, error_budget
        )

    def _execute_cascade_plan(
        self,
        tensor: np.ndarray,
        plan: CascadePlan,
        target_ratio: float,
        max_error: float,
        name: str,
    ) -> Tuple[bytes, dict, float, float]:
        """Execute a cascade plan by applying stages sequentially."""
        stacked_data: List[bytes] = []
        stacked_meta: List[dict] = []
        current = tensor.copy()
        total_ratio = 1.0
        total_error = 0.0

        for stage in plan.stages:
            inst = None
            if self._engine is not None and hasattr(self._engine, "_methods"):
                inst = self._engine._methods.get(stage.method_name)
            if inst is None:
                continue
            try:
                data, meta = inst.compress(current, **stage.params)
                recon = inst.decompress(data, meta)
                if recon.shape != current.shape:
                    recon = recon.reshape(current.shape)

                stage_ratio = current.nbytes / max(len(data), 1)
                total_ratio *= stage_ratio
                var = float(np.var(current))
                mse = float(np.mean((current.ravel() - recon.ravel()) ** 2))
                stage_error = mse / var if var > 0 else float(mse)
                total_error += stage_error

                stacked_data.append(data)
                stacked_meta.append(
                    {
                        "method": stage.method_name,
                        "params": meta,
                        "ratio": stage_ratio,
                    }
                )

                residual = current.astype(np.float32) - recon.astype(np.float32)
                current = residual
            except Exception as exc:
                logger.debug("Stage '%s' failed: %s", stage.method_name, exc)
                continue

            if total_ratio >= target_ratio or total_error >= max_error:
                break

        if not stacked_data:
            raise RuntimeError("No cascade stages succeeded")

        import struct

        packed = bytearray()
        stage_lengths = []
        for sd in stacked_data:
            stage_lengths.append(len(sd))
            packed += struct.pack("<I", len(sd))
            packed += sd

        metadata = {
            "cascade": True,
            "n_stages": len(stacked_data),
            "stages": stacked_meta,
            "stage_lengths": stage_lengths,
            "total_ratio": total_ratio,
            "total_error": min(total_error, 1.0),
            "original_shape": list(tensor.shape),
            "method": "cascade",
            "oracle": True,
        }
        return bytes(packed), metadata, total_ratio, min(total_error, 1.0)

    def _compress_streaming(
        self,
        tensor: np.ndarray,
        target_ratio: float,
        max_error: float,
        name: str,
    ) -> Tuple[bytes, dict, float, float]:
        """Streaming path: chunked compression for large tensors."""
        logger.debug("Router: streaming path for '%s'", name)
        if hasattr(self._engine, "_chunked_compress"):
            return self._engine._chunked_compress(tensor, target_ratio, max_error, name)
        from ..chunked_compressor import ChunkedCompressor

        compressor = ChunkedCompressor(self._engine)
        return compressor.compress_chunked(name, tensor, target_ratio, max_error)

    def _compress_model_level(
        self,
        tensors: Dict[str, np.ndarray],
        target_ratio: float,
        max_error: float,
        use_world_model: bool,
        progress_callback: Any = None,
    ) -> Dict[str, Any]:
        """Model-level compression with optional world model scanning."""
        logger.debug(
            "Router: model-level compression of %d tensors (world_model=%s)",
            len(tensors),
            use_world_model,
        )

        if use_world_model:
            try:
                model_profile = self._world_model.scan_from_dict(tensors)
                logger.debug(
                    "World model: %d tensors, %.2f GB, %d layers",
                    model_profile.graph.n_tensors,
                    model_profile.estimated_model_size_gb,
                    model_profile.layer_count,
                )
            except Exception as exc:
                logger.debug("World model scan failed: %s", exc)

        results: Dict[str, Any] = {}
        for idx, (name, tensor) in enumerate(tensors.items()):
            if progress_callback:
                progress_callback(idx + 1, len(tensors), name)
            try:
                data, meta, ratio, error = self.compress(
                    tensor,
                    target_ratio=target_ratio,
                    max_error=max_error,
                    name=name,
                    use_world_model=use_world_model,
                )
                results[name] = {
                    "data": data,
                    "metadata": meta,
                    "ratio": ratio,
                    "error": error,
                }
            except Exception as exc:
                logger.warning("Compression failed for '%s': %s", name, exc)
                results[name] = {
                    "data": b"",
                    "metadata": {
                        "method": "failed",
                        "original_shape": list(tensor.shape),
                    },
                    "ratio": 1.0,
                    "error": 1.0,
                }
            del tensor
            gc.collect()

        return results

    @property
    def memory_budget_mb(self) -> int:
        return self._memory_budget_mb

    @memory_budget_mb.setter
    def memory_budget_mb(self, value: int) -> None:
        self._memory_budget_mb = max(value, 64)
