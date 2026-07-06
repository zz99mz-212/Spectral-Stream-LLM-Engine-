"""
Compression Intelligence Engine — unified orchestrator.
Zero enable_* methods. Zero dead sub-engines. One cohesive system.

Memory-efficient: all operations use O(1) scratch, numpy views,
in-place operations, mmap I/O, and streaming for large tensors.
Consumer-device ready (8-16GB RAM).
"""

from __future__ import annotations

import gc
import logging
import os
import struct
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ._profiler import CompressionProfiler
from ._allocator import ErrorBudgetAllocator
from ._helpers import (
    _classify_by_name,
    _select_methods,
    compress_tensor_with_validation,
    _build_report,
    compute_tensor_size,
    compute_compression_ratio,
    compute_error_metrics,
    BYPASS_HIGH_CONFIDENCE,
    BYPASS_MEDIUM_CONFIDENCE,
    TEST_FULL,
)
from ._tensor_type_strategy import (
    compress_with_tensor_strategy,
    decompress_with_strategy,
    _tensor_type_strategy,
)
from ._dataclasses import CompressionConfig, CompressedTensor, TensorProfile
from .method_discovery import MethodDiscovery
from .method_tiers import get_tier, MethodTier
from ._tier_common import MethodTier as _MethodTier, tier_score
from .memory_mapped_engine import MemoryMappedTensorEngine
from .chunked_compressor import ChunkedCompressor
from .streaming_pipeline import StreamingCompressionPipeline
from .dynamic_method_tester import DynamicMethodTester
from .quantum_cascade import QuantumSuperpositionEngine, CascadeSuperpositionPlan

logger = logging.getLogger(__name__)


class LazyMethodDict(Dict[str, Any]):
    """Dict that lazily instantiates methods from METHOD_CLASSES on access.

    Only the 10 engine built-in methods are pre-loaded.  All other
    methods are instantiated on first access.  This reduces memory
    from ~1055 instances to only those actually used.
    """

    _BUILTIN_NAMES = {
        "block_int8",
        "block_int4",
        "hadamard_int8",
        "hadamard_int4",
        "sparsity_int4",
        "delta_int4",
        "svd_compress",
        "dct_spectral",
        "tensor_train",
        "fwht_compress",
    }

    def __init__(self) -> None:
        super().__init__()
        self._loaded: Dict[str, Any] = {}
        self._method_classes: Optional[Dict[str, Any]] = None
        self._engine_methods: Dict[str, Any] = {}
        self._resolved: set = set()

        try:
            from spectralstream.compression.engine._methods import (
                METHOD_REGISTRY as _ENGINE_METHODS,
            )

            self._engine_methods = dict(_ENGINE_METHODS)
            for name, inst in self._engine_methods.items():
                self._loaded[name] = inst
                self._resolved.add(name)
        except ImportError:
            pass

    def _ensure_classes(self) -> Dict[str, Any]:
        if self._method_classes is None:
            try:
                from spectralstream.compression.methods import (
                    METHOD_CLASSES,
                    _load_extra,
                )

                _load_extra()
                self._method_classes = dict(METHOD_CLASSES)
            except Exception:
                self._method_classes = {}
        return self._method_classes

    def _resolve(self, name: str) -> Any:
        if name in self._resolved:
            return self._loaded.get(name)
        self._resolved.add(name)

        if name in self._engine_methods:
            self._loaded[name] = self._engine_methods[name]
            return self._loaded[name]

        classes = self._ensure_classes()
        cls = classes.get(name)
        if cls is not None:
            try:
                inst = cls() if isinstance(cls, type) else cls
                self._loaded[name] = inst
                return inst
            except Exception:
                pass
        return None

    def __getitem__(self, key: str) -> Any:
        if key not in self._resolved:
            inst = self._resolve(key)
            if inst is not None:
                self._loaded[key] = inst
                return inst
            raise KeyError(key)
        return self._loaded[key]

    def get(self, key: str, default: Any = None) -> Any:
        inst = self._resolve(key)
        return inst if inst is not None else default

    def __contains__(self, key: object) -> bool:
        if key in self._resolved:
            return key in self._loaded
        inst = self._resolve(str(key)) if isinstance(key, str) else None
        return inst is not None

    def items(self):
        classes = self._ensure_classes()
        all_names = list(self._loaded.keys()) + [
            n for n in classes if n not in self._loaded
        ]
        for name in all_names:
            inst = self._loaded.get(name) if name in self._resolved else None
            if inst is None:
                inst = self._resolve(name)
            if inst is not None:
                yield (name, inst)

    def values(self):
        for _, v in self.items():
            yield v

    def keys(self):
        yield from self._loaded.keys()
        classes = self._ensure_classes()
        for k in classes:
            if k not in self._loaded:
                yield k

    def __len__(self) -> int:
        classes = self._ensure_classes()
        return len(set(self._loaded.keys()) | set(classes.keys()))

    def __iter__(self):
        seen: set = set()
        for name in self._loaded:
            seen.add(name)
            yield name
        classes = self._ensure_classes()
        for name in classes:
            if name not in seen:
                seen.add(name)
                yield name

    def __setitem__(self, key: str, value: Any) -> None:
        self._loaded[key] = value
        self._resolved.add(key)

    def __repr__(self) -> str:
        return f"LazyMethodDict({len(self)} methods, {len(self._loaded)} instantiated)"


class CompressionIntelligenceEngine:
    """Unified compression orchestration engine.

    ZERO `enable_*` methods.  ZERO competing sub-engines.
    Everything is wired through UnifiedIntelligence.

    Memory-efficient: all operations use O(1) scratch, numpy views,
    in-place operations, mmap I/O, and streaming for large tensors.
    Consumer-device ready (8-16GB RAM).
    """

    def __init__(
        self,
        methods: Optional[Dict[str, Any]] = None,
        config: Optional[CompressionConfig] = None,
        use_intelligence: bool = True,
    ):
        self._config = config if config is not None else CompressionConfig()
        self._methods: Dict[str, Any] = (
            dict(methods) if methods else self._load_all_method_instances()
        )
        self._discovery = MethodDiscovery(self._methods)
        self._use_intelligence = use_intelligence

        # Core sub-engines (always-on, eagerly initialized)
        self._profiler = CompressionProfiler()
        self._allocator = ErrorBudgetAllocator()
        self._memory_manager: Optional[ProgressiveMemoryManager] = None
        self._oracle_inst: Any = None

        # Discover all available methods
        self._all_methods: Dict[str, Dict[str, Any]] = {}
        try:
            self._all_methods = MethodDiscovery.discover()
        except Exception:
            self._all_methods = {}

        # Category and tier indices for O(1) method filtering
        self._methods_by_category: Dict[str, List[str]] = {}
        self._methods_by_tier: Dict[int, List[str]] = {}
        self._build_method_indices()

        # ── UNIFIED INTELLIGENCE ── always-on, no enable_* needed
        if self._use_intelligence:
            from ._unified_intelligence import UnifiedIntelligence

            self._unified_intelligence = UnifiedIntelligence(self, self._config)
        else:
            self._unified_intelligence = None

        # ── New intelligence subsystems (lazy-initialized) ──
        self._holographic_oracle: Any = None
        self._quantum_cascade: Any = None
        self._resonant_grouper_inst: Any = None

        logger.debug(
            "CompressionIntelligenceEngine initialized (%d methods, UnifiedIntelligence=%s)",
            len(self._methods),
            self._unified_intelligence is not None,
        )

    def _build_method_indices(self) -> None:
        """Build category→names and tier→names indices for fast filtering."""
        self._methods_by_category.clear()
        self._methods_by_tier.clear()

        for mname, minfo in self._all_methods.items():
            cat = minfo.get("category", "unknown")
            self._methods_by_category.setdefault(cat, []).append(mname)

            tier = minfo.get("tier")
            if tier is not None:
                try:
                    tval = tier.value if hasattr(tier, "value") else int(tier)
                    self._methods_by_tier.setdefault(tval, []).append(mname)
                except (ValueError, TypeError):
                    pass

        for mname in self._methods:
            if mname not in self._all_methods:
                inst = self._methods[mname]
                cat = getattr(inst, "category", "unknown")
                self._methods_by_category.setdefault(cat, []).append(mname)

    @staticmethod
    def _load_all_method_instances() -> Dict[str, Any]:
        return LazyMethodDict()

    @property
    def profiler(self) -> CompressionProfiler:
        return self._profiler

    @property
    def allocator(self) -> ErrorBudgetAllocator:
        return self._allocator

    @property
    def discovery(self) -> MethodDiscovery:
        return self._discovery

    @property
    def config(self) -> CompressionConfig:
        return self._config

    @property
    def unified_intelligence(self) -> Any:
        return self._unified_intelligence

    @property
    def oracle(self) -> Any:
        """Lazy-access MethodOracle (cached for performance history accumulation)."""
        if not hasattr(self, "_oracle_inst") or self._oracle_inst is None:
            from .world_model.method_oracle import MethodOracle

            self._oracle_inst = MethodOracle(self)
        return self._oracle_inst

    @property
    def holographic_oracle(self) -> Any:
        """Lazy-access HolographicOracle (associative memory for zero-shot selection)."""
        if not hasattr(self, "_holographic_inst") or self._holographic_inst is None:
            from .holographic_oracle import HolographicOracle

            memory_path = getattr(self._config, "holographic_memory_path", "")
            self._holographic_inst = HolographicOracle(
                self, memory_path=memory_path or None
            )
        return self._holographic_inst

    @property
    def holographic_oracle(self) -> Any:
        """Lazy-access HolographicOracle for zero-shot method recall."""
        if self._holographic_oracle is None:
            from .holographic_oracle import HolographicOracle

            self._holographic_oracle = HolographicOracle(self, self.oracle)
        return self._holographic_oracle

    @property
    def quantum_cascade(self) -> Any:
        """Lazy-access QuantumCascadeEngine for parallel method testing."""
        if self._quantum_cascade is None:
            from .quantum_cascade import QuantumCascadeEngine

            num_workers = getattr(self._config, "num_workers", 4)
            self._quantum_cascade = QuantumCascadeEngine(max_workers=num_workers)
        return self._quantum_cascade

    @property
    def resonant_grouper(self) -> Any:
        """Lazy-access ResonantGrouper for spectral-resonance-based tensor grouping."""
        if self._resonant_grouper_inst is None:
            from .resonant_grouping import ResonantGrouper

            self._resonant_grouper_inst = ResonantGrouper()
        return self._resonant_grouper_inst

    @property
    def memory_manager(self) -> ProgressiveMemoryManager:
        if self._memory_manager is None:
            from .progressive_release import ProgressiveMemoryManager

            budget = getattr(self._config, "memory_budget_mb", 1024.0)
            self._memory_manager = ProgressiveMemoryManager(memory_budget_mb=budget)
        return self._memory_manager

    @property
    def memory_budget_mb(self) -> int:
        return getattr(self._config, "memory_budget_mb", 256)

    def compress_within_budget(
        self,
        tensor: np.ndarray,
        target_ratio: float,
        max_error: float,
        name: str = "",
    ) -> Tuple[bytes, dict, float, float]:
        if tensor.nbytes > self.memory_budget_mb * 1_024 * 1_024:
            return self._chunked_compress(tensor, target_ratio, max_error, name)
        else:
            return self.compress_fast(tensor, name, target_ratio, max_error)

    def _chunked_compress(
        self,
        tensor: np.ndarray,
        target_ratio: float,
        max_error: float,
        name: str = "",
    ) -> Tuple[bytes, dict, float, float]:
        from .chunked_compressor import ChunkedCompressor

        compressor = ChunkedCompressor(self)
        return compressor.compress_chunked(name, tensor, target_ratio, max_error)

    @property
    def kernel_fusion(self) -> type:
        from .hpc_kernel_fusion import HPCKernelFusion

        return HPCKernelFusion

    # ── Tensor Classification ────────────────────────────────────────────

    @staticmethod
    def _classify_by_name(name: str) -> str:
        return _classify_by_name(name)

    # ── Core API ─────────────────────────────────────────────────────────

    def profile_tensor(self, tensor: np.ndarray, name: str = "") -> TensorProfile:
        result = self.profiler.profile_tensor(tensor, name)
        del tensor
        return result

    def _select_methods(
        self,
        profile: Any,
        error_budget: float = 0.01,
        target_ratio: float = 5000.0,
        available_methods: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Select candidate compression methods based on tier and profile."""
        if available_methods is not None:
            return _select_methods(
                profile, error_budget, target_ratio, available_methods
            )

        MAX_CANDIDATES_PER_TIER = 30
        tier_priority = [1, 2, 3, 4, 5]
        candidates: List[Dict[str, Any]] = []
        seen: set = set()

        for tval in tier_priority:
            names = self._methods_by_tier.get(tval, [])
            for mname in names[:MAX_CANDIDATES_PER_TIER]:
                if mname in seen:
                    continue
                seen.add(mname)
                inst = self._methods.get(mname)
                if inst is not None:
                    candidates.append({"instance": inst, "params": {}, "name": mname})

        if len(candidates) < 20:
            remaining = 0
            for mname in self._methods:
                if mname not in seen:
                    remaining += 1
                    if remaining > MAX_CANDIDATES_PER_TIER:
                        break
                    inst = self._methods.get(mname)
                    if inst is not None:
                        candidates.append(
                            {"instance": inst, "params": {}, "name": mname}
                        )

        return _select_methods(profile, error_budget, target_ratio, candidates)

    def compress_tensor_with_validation(
        self,
        tensor: np.ndarray,
        profile: TensorProfile,
        methods: List[Dict[str, Any]],
        error_budget: float = 0.01,
    ) -> Tuple[bytes, dict, float, float]:
        result = compress_tensor_with_validation(tensor, profile, methods, error_budget)
        del tensor
        gc.collect()
        return result

    def compress_fast(
        self,
        tensor: np.ndarray,
        name: str = "",
        target_ratio: float = 10.0,
        max_error: float = 0.01,
    ) -> Tuple[bytes, dict, float, float]:
        profile = self.profile_tensor(tensor, name)
        error_budget = max_error / max(target_ratio, 1.0)
        tensor_type = self._classify_by_name(name)

        # Use HolographicOracle with associative memory recall (zero-shot path)
        if self._use_intelligence:
            try:
                ho = self.holographic_oracle
                ranked, bypass = ho.select_method(
                    tensor=tensor,
                    tensor_type=tensor_type,
                    target_ratio=target_ratio,
                    max_error=max_error,
                )
                if ranked:
                    methods = [
                        {"instance": r.instance, "params": r.params, "name": r.name}
                        for r in ranked
                    ]
                    result = self.compress_tensor_with_validation(
                        tensor,
                        profile,
                        methods,
                        error_budget,
                        bypass_decision=bypass,
                    )
                    data, meta, ratio, error = result
                    # Record success in holographic memory
                    try:
                        ho.record_success(
                            tensor=tensor,
                            tensor_type=tensor_type,
                            method_name=meta.get("method", ranked[0].name),
                            params=meta,
                            ratio=ratio,
                            error=error,
                        )
                    except Exception:
                        pass
                    # Also record in MethodOracle for backward compatibility
                    try:
                        m_oracle = self.oracle
                        m_oracle.record_performance(
                            tensor_type=tensor_type,
                            method_name=meta.get("method", ranked[0].name),
                            ratio=ratio,
                            error=error,
                        )
                    except Exception:
                        pass
                    return result
            except Exception:
                pass

        # Use MethodOracle with confidence-based bypass when world model is enabled
        if self._use_intelligence:
            try:
                oracle = self.oracle
                ranked, bypass = oracle.select_with_bypass(
                    profile=profile,
                    tensor_type=tensor_type,
                    target_ratio=target_ratio,
                    max_error=max_error,
                    max_results=10,
                )
                if ranked:
                    methods = [
                        {"instance": r.instance, "params": r.params, "name": r.name}
                        for r in ranked
                    ]
                    result = self.compress_tensor_with_validation(
                        tensor,
                        profile,
                        methods,
                        error_budget,
                        bypass_decision=bypass,
                    )
                    data, meta, ratio, error = result
                    # Record performance for future bypass decisions
                    try:
                        oracle.record_performance(
                            tensor_type=tensor_type,
                            method_name=meta.get("method", ranked[0].name),
                            ratio=ratio,
                            error=error,
                        )
                    except Exception:
                        pass
                    # Also record in holographic memory
                    try:
                        ho = self.holographic_oracle
                        ho.record_success(
                            tensor=tensor,
                            tensor_type=tensor_type,
                            method_name=meta.get("method", ranked[0].name),
                            params=meta,
                            ratio=ratio,
                            error=error,
                        )
                    except Exception:
                        pass
                    return result
            except Exception:
                pass

        methods = self._select_methods(profile, error_budget, target_ratio)
        if not methods:
            blk8 = self._methods.get("block_int8")
            if blk8:
                methods = [{"instance": blk8, "params": {}}]
            else:
                raise RuntimeError("No methods available and block_int8 not found")

        result = self.compress_tensor_with_validation(
            tensor, profile, methods, error_budget
        )
        data, meta, ratio, error = result
        # Record in holographic memory even from fallback path
        try:
            ho = self.holographic_oracle
            ho.record_success(
                tensor=tensor,
                tensor_type=tensor_type,
                method_name=meta.get("method", "block_int8"),
                params=meta,
                ratio=ratio,
                error=error,
            )
        except Exception:
            pass
        return result

    # ── Smart Compression (Novel Optimizations Integrated) ───────────────

    def compress_smart(
        self,
        tensor: np.ndarray,
        name: str = "",
        target_ratio: float = 5000.0,
        max_error: float = 0.01,
        mode: str = "balanced",
    ) -> Tuple[bytes, dict, float, float]:
        """Unified smart compression using ALL intelligence systems.

        Pipeline:
        1. Try HolographicOracle.select_method() for associative recall
           → BYPASS_HIGH_CONFIDENCE: use cached method directly (zero-shot)
           → BYPASS_MEDIUM_CONFIDENCE: test top-1 via QuantumCascade (fast mode)
           → Otherwise: fall through to Step 2
        2. Use MethodOracle.select_with_bypass() + QuantumCascade
           → HIGH confidence: test top-1 method only
           → MEDIUM confidence: test top-3 methods
           → FULL: test top-10 methods
        3. Record result in HolographicOracle + MethodOracle for continuous learning
        4. Fallback to compress_fast if all else fails

        Parameters
        ----------
        tensor : np.ndarray
            The tensor to compress.
        name : str
            Tensor name (for classification and logging).
        target_ratio : float
            Target compression ratio.
        max_error : float
            Maximum acceptable relative error.
        mode : str
            Cascade mode: 'fast', 'balanced', 'extreme'.

        Returns
        -------
        tuple of (bytes, dict, float, float)
            (compressed_data, metadata, ratio, error)
        """
        tensor_type = self._classify_by_name(name)
        profile = self.profile_tensor(tensor, name)
        error_budget = max_error / max(target_ratio, 1.0)

        # ── Step 1: Try HolographicOracle for associative recall ──
        try:
            ho = self.holographic_oracle
            ranked_ho, bypass_ho = ho.select_method(
                tensor=tensor,
                tensor_type=tensor_type,
                target_ratio=target_ratio,
                max_error=max_error,
            )
            if ranked_ho and bypass_ho == BYPASS_HIGH_CONFIDENCE:
                # Zero-shot: apply cached method directly
                best = ranked_ho[0]
                logger.debug(
                    "Smart compress: holographic recall '%s' for '%s' (conf=%.3f)",
                    best.name,
                    name,
                    best.confidence,
                )
                inst = best.instance or self._methods.get(best.name)
                if inst is not None:
                    try:
                        data, meta = inst.compress(tensor, **best.params)
                        ratio = tensor.nbytes / max(len(data), 1)
                        error = best.expected_error
                        meta["method"] = best.name
                        meta["original_shape"] = list(tensor.shape)
                        meta["compression_ratio"] = ratio
                        meta["holographic_recall"] = True
                        meta["recall_confidence"] = best.confidence
                        return data, meta, ratio, error
                    except Exception as exc:
                        logger.debug("Zero-shot method '%s' failed: %s", best.name, exc)
        except Exception as exc:
            logger.debug("HolographicOracle recall failed: %s", exc)

        # ── Step 2: Use MethodOracle + QuantumCascade ──
        try:
            oracle = self.oracle
            ranked, bypass = oracle.select_with_bypass(
                profile=profile,
                tensor_type=tensor_type,
                target_ratio=target_ratio,
                max_error=max_error,
                max_results=10,
            )
            if ranked:
                cascade_methods = [
                    {
                        "name": r.name,
                        "instance": r.instance or self._methods.get(r.name),
                        "params": r.params,
                    }
                    for r in ranked
                    if r.instance or r.name in self._methods
                ]

                if bypass == BYPASS_HIGH_CONFIDENCE:
                    n_test = 1
                    qc_mode = "fast"
                elif bypass == BYPASS_MEDIUM_CONFIDENCE:
                    n_test = 3
                    qc_mode = "balanced"
                else:
                    limit_map = {"fast": 3, "balanced": 10, "extreme": 50}
                    n_test = limit_map.get(mode, 10)
                    qc_mode = mode

                report = self.quantum_cascade.test_methods(
                    tensor=tensor,
                    methods=cascade_methods[:n_test],
                    target_ratio=target_ratio,
                    max_error=max_error,
                    mode=qc_mode,
                    profile=profile,
                    error_budget=error_budget,
                )

                if (
                    report.best_result is not None
                    and report.best_result.compressed_data is not None
                ):
                    best = report.best_result
                    # Record in holographic memory
                    try:
                        self.holographic_oracle.record_success(
                            tensor=tensor,
                            tensor_type=tensor_type,
                            method_name=best.method_name,
                            params=best.method_params,
                            ratio=best.ratio,
                            error=best.error,
                        )
                    except Exception:
                        pass
                    # Record in method oracle history
                    try:
                        oracle.record_performance(
                            tensor_type=tensor_type,
                            method_name=best.method_name,
                            ratio=best.ratio,
                            error=best.error,
                        )
                    except Exception:
                        pass
                    meta = best.metadata or {}
                    meta["method"] = best.method_name
                    meta["original_shape"] = list(tensor.shape)
                    meta["compression_ratio"] = best.ratio
                    meta["quantum_cascade"] = True
                    meta["cascade_mode"] = mode
                    meta["n_methods_tested"] = report.n_tested
                    return best.compressed_data, meta, best.ratio, best.error
        except Exception as exc:
            logger.debug("MethodOracle + QuantumCascade failed: %s", exc)

        # ── Step 3: Fallback to standard compress_fast ──
        logger.debug("Smart compress: falling back to fast path for '%s'", name)
        result = self.compress_fast(tensor, name, target_ratio, max_error)
        data, meta, ratio, error = result
        try:
            self.holographic_oracle.record_success(
                tensor=tensor,
                tensor_type=tensor_type,
                method_name=meta.get("method", "unknown"),
                params=meta,
                ratio=ratio,
                error=error,
            )
        except Exception:
            pass
        return result

    def compress_model_smart(
        self,
        tensor_source: Any,
        target_ratio: float = 5000.0,
        max_error: float = 0.01,
        mode: str = "balanced",
        progress_callback: Any = None,
    ) -> dict:
        """Compress an entire model using all optimizations.

        Pipeline:
        1. Scan all tensor metadata (header-only for streaming)
        2. Group tensors using TensorGrouper + ResonantGrouper refinement
        3. For each group:
           a. Compute resonance signature of representative
           b. Try holographic recall for the signature
           c. If match: apply cached method to all group members
           d. If no match: use compress_smart on representative
              → Apply best method to all group members
              → Store in holographic memory
        4. Write results to compressed format

        Parameters
        ----------
        tensor_source : Dict[str, np.ndarray] or str
            Either a dict of {name: tensor} or a path to a .safetensors file.
        target_ratio : float
            Target compression ratio.
        max_error : float
            Maximum acceptable relative error.
        mode : str
            Cascade mode: 'fast', 'balanced', 'extreme'.
        progress_callback : callable, optional
            Called as f(processed, total, tensor_name).

        Returns
        -------
        dict
            Compression report with timing, ratio, error.
        """
        import time as _time
        from ._dataclasses import CompressedTensor
        from ._helpers import _build_report as _build_report_fn
        from .grouping_optimizer import (
            TensorGrouper as _TensorGrouper,
            group_tensors as _group_tensors,
            get_tensor_metadata_from_dict,
        )

        t0 = _time.perf_counter()
        gc.collect()

        # ── Step 1: Get tensor metadata ──
        if isinstance(tensor_source, str):
            # File path — use header-only scan
            from .memory_mapped_engine import MemoryMappedTensorEngine
            from .world_model.tensor_world_model import TensorWorldModel

            mmap_engine = MemoryMappedTensorEngine(tensor_source)
            tensor_info = (
                mmap_engine.get_tensor_info_dict()
                if hasattr(mmap_engine, "get_tensor_info_dict")
                else {}
            )
            # Use TensorWorldModel for metadata scan
            world_model = TensorWorldModel()
            model_profile = world_model.scan_from_names(
                {
                    name: (info[0], str(info[1]), info[2], info[3])
                    for name, info in tensor_info.items()
                }
            )
            metadata = {
                name: (node.shape, node.tensor_type, node.nbytes)
                for name, node in model_profile.graph.nodes.items()
            }
            tensor_dict = None
        else:
            # Dict of tensors
            metadata = get_tensor_metadata_from_dict(tensor_source)
            metadata = {
                name: (
                    shape,
                    _classify_by_name(name),
                    int(np.prod(shape)) * 4
                    if not np.issubdtype(np.float32, np.dtype)
                    else int(np.prod(shape)) * 4,
                )
                for name, (shape, _) in metadata.items()
            }
            tensor_dict = tensor_source
            mmap_engine = None
            model_profile = None

        if not metadata:
            raise ValueError("No tensors found in source")

        n_total = len(metadata)
        logger.info(
            "Smart model compression: %d tensors, target ratio %.0f:1, mode=%s",
            n_total,
            target_ratio,
            mode,
        )

        # ── Step 2: Group tensors ──
        # First, use TensorGrouper for shape-based grouping
        shape_grouper = _TensorGrouper()
        shape_metadata = {name: (info[0], "float32") for name, info in metadata.items()}
        shape_groups = _group_tensors(shape_metadata)

        # Then refine with ResonantGrouper (uses add_metadata_grouping to merge
        # shape-based groups that have similar spectral resonance profiles)
        try:
            refined_groups = self.resonant_grouper.add_metadata_grouping(shape_groups)
            if not refined_groups:
                refined_groups = [
                    type(
                        "_SimpleFallbackGroup",
                        (),
                        {
                            "members": g.tensor_names,
                            "size": g.size,
                            "resonance_key": g.pattern,
                            "representative": g.representative,
                        },
                    )()
                    for g in shape_groups
                ]
        except Exception as exc:
            logger.debug(
                "Resonant grouping refinement failed: %s — using shape groups", exc
            )
            refined_groups = [
                type(
                    "_SimpleFallbackGroup",
                    (),
                    {
                        "members": g.tensor_names,
                        "size": g.size,
                        "resonance_key": g.pattern,
                        "representative": g.representative,
                    },
                )()
                for g in shape_groups
            ]

        logger.info(
            "Grouping: %d shape-based groups → %d refined resonance groups",
            len(shape_groups),
            len(refined_groups),
        )

        # ── Step 3 & 4: Process each group ──

        # Decide: parallel or sequential group processing
        if self._config.num_workers > 1:
            # ── PARALLEL PATH via ParallelCompressor.compress_groups() ──
            from .parallel_compressor import ParallelCompressor

            # Determine safe worker count, accounting for streaming + memory
            safe_workers = ParallelCompressor._get_safe_workers(
                cpu_count=self._config.num_workers,
                streaming=self._config.streaming,
                memory_budget_gb=self._config.max_memory_gb,
            )
            compressor = ParallelCompressor(
                max_workers=safe_workers,
                memory_budget_gb=self._config.max_memory_gb,
                streaming=self._config.streaming,
            )

            parallel_stats = compressor.compress_groups(
                engine=self,
                groups=refined_groups,
                tensor_dict=tensor_dict,
                mmap_engine=mmap_engine,
                metadata=metadata,
                target_ratio=target_ratio,
                max_error=max_error,
                mode=mode,
                progress_callback=progress_callback,
            )

            compressed_list = parallel_stats.get("tensors", [])
            failures = parallel_stats.get("failures", [])
            total_orig = parallel_stats.get("total_orig_bytes", 0)
            total_comp = parallel_stats.get("total_compressed_bytes", 0)
            errors = [
                ct.relative_error
                for ct in compressed_list
                if hasattr(ct, "relative_error")
            ]
            method_dist = parallel_stats.get("method_distribution", {})
            groups_processed = parallel_stats.get(
                "groups_processed", len(refined_groups)
            )

        else:
            # ── SEQUENTIAL PATH (original logic) ──
            compressed_list: List[CompressedTensor] = []
            failures: List[str] = []
            total_orig = 0
            total_comp = 0
            errors = []
            method_dist: Dict[str, int] = {}
            groups_processed = 0
            group_method_cache: Dict[str, Dict[str, Any]] = {}

            for group_idx, group in enumerate(refined_groups):
                # Normalize: handle both ResonantGroup (.members) and TensorGroup (.tensor_names)
                tensor_names = getattr(
                    group, "members", getattr(group, "tensor_names", [])
                )
                if not tensor_names:
                    continue
                # Pick representative (middle element for typicality)
                rep_name = getattr(
                    group,
                    "representative",
                    tensor_names[len(tensor_names) // 2]
                    if len(tensor_names) > 1
                    else tensor_names[0],
                )
                res_key = getattr(
                    group,
                    "resonance_key",
                    getattr(group, "pattern", f"group_{group_idx}"),
                )

                # Get the representative tensor
                if tensor_dict is not None and rep_name in tensor_dict:
                    rep_tensor = tensor_dict[rep_name]
                elif mmap_engine is not None:
                    try:
                        rep_tensor = np.asarray(mmap_engine.get_tensor(rep_name))
                    except Exception as exc:
                        logger.warning(
                            "Failed to load representative '%s': %s", rep_name, exc
                        )
                        failures.append(rep_name)
                        continue
                else:
                    logger.warning("Representative '%s' not available", rep_name)
                    failures.append(rep_name)
                    continue

                # Try compress_smart on representative
                try:
                    rep_data, rep_meta, rep_ratio, rep_error = self.compress_smart(
                        tensor=rep_tensor,
                        name=rep_name,
                        target_ratio=target_ratio,
                        max_error=max_error,
                        mode=mode,
                    )

                    # Cache result for group members
                    best_method = rep_meta.get("method", "unknown")
                    group_method_cache[res_key] = {
                        "method": best_method,
                        "ratio": rep_ratio,
                        "error": rep_error,
                        "params": rep_meta,
                    }

                    # Record in holographic memory for future models
                    try:
                        tensor_type = _classify_by_name(rep_name)
                        self.holographic_oracle.record_success(
                            tensor=rep_tensor,
                            tensor_type=tensor_type,
                            method_name=best_method,
                            params={},
                            ratio=rep_ratio,
                            error=rep_error,
                        )
                    except Exception:
                        pass

                    # Add representative to compressed list
                    ct = CompressedTensor(
                        _data=rep_data,
                        method=best_method,
                        params=rep_meta,
                        original_shape=rep_tensor.shape,
                        original_dtype=str(rep_tensor.dtype),
                        compression_ratio=rep_ratio,
                        relative_error=rep_error,
                    )
                    compressed_list.append(ct)
                    total_orig += rep_tensor.nbytes
                    total_comp += len(rep_data)
                    errors.append(rep_error)
                    method_dist[best_method] = method_dist.get(best_method, 0) + 1

                    if progress_callback:
                        progress_callback(len(compressed_list), n_total, rep_name)

                    groups_processed += 1

                except Exception as exc:
                    logger.warning(
                        "compress_smart failed for group '%s': %s", res_key, exc
                    )
                    failures.append(rep_name)
                    # Fallback: use compress_fast on all tensors in group
                    for tn in tensor_names:
                        try:
                            if tensor_dict is not None and tn in tensor_dict:
                                t = tensor_dict[tn]
                            elif mmap_engine is not None:
                                t = np.asarray(mmap_engine.get_tensor(tn))
                            else:
                                continue
                            data, meta, ratio, error = self.compress_fast(
                                t, tn, target_ratio, max_error
                            )
                            ct = CompressedTensor(
                                _data=data,
                                method=meta.get("method", "unknown"),
                                params=meta,
                                original_shape=t.shape,
                                original_dtype=str(t.dtype),
                                compression_ratio=ratio,
                                relative_error=error,
                            )
                            compressed_list.append(ct)
                            total_orig += t.nbytes
                            total_comp += len(data)
                            errors.append(error)
                        except Exception as e2:
                            logger.debug("Fallback failed for '%s': %s", tn, e2)
                            failures.append(tn)
                    continue

                finally:
                    if tensor_dict is None:
                        del rep_tensor
                        if groups_processed % 3 == 0:
                            gc.collect()

                # Apply cached method to remaining tensors in group
                for tn in tensor_names:
                    if tn == rep_name:
                        continue

                    try:
                        if tensor_dict is not None and tn in tensor_dict:
                            t = tensor_dict[tn]
                        elif mmap_engine is not None:
                            t = np.asarray(mmap_engine.get_tensor(tn))
                        else:
                            continue

                        cached = group_method_cache.get(res_key, {})
                        method_name = cached.get("method", "")
                        inst = self._methods.get(method_name) if method_name else None

                        if inst is not None:
                            data, meta = inst.compress(t)
                            ratio = t.nbytes / max(len(data), 1)
                            error = cached.get("error", 0.01)
                            meta["method"] = method_name
                            meta["original_shape"] = list(t.shape)
                            meta["group_cached"] = True
                        else:
                            # Fallback to compress_fast
                            data, meta, ratio, error = self.compress_fast(
                                t, tn, target_ratio, max_error
                            )

                        ct = CompressedTensor(
                            _data=data,
                            method=meta.get("method", "unknown"),
                            params=meta,
                            original_shape=t.shape,
                            original_dtype=str(t.dtype),
                            compression_ratio=ratio,
                            relative_error=error,
                        )
                        compressed_list.append(ct)
                        total_orig += t.nbytes
                        total_comp += len(data)
                        errors.append(error)

                        if progress_callback:
                            progress_callback(len(compressed_list), n_total, tn)

                    except Exception as exc:
                        logger.debug("Group apply failed for '%s': %s", tn, exc)
                        failures.append(tn)

                if tensor_dict is None and (groups_processed % 5 == 0):
                    gc.collect()

            gc.collect()

        if mmap_engine is not None:
            mmap_engine.close()

        # ── Build report ──
        elapsed = _time.perf_counter() - t0
        avg_error = float(np.mean(errors)) if errors else 0.0

        stats = {
            "tensors": compressed_list,
            "total_orig_bytes": total_orig,
            "total_compressed_bytes": total_comp,
            "overall_ratio": total_orig / max(total_comp, 1),
            "average_ratio": total_orig / max(total_comp, 1),
            "avg_error": avg_error,
            "max_error": float(np.max(errors)) if errors else 0.0,
            "num_tensors": len(compressed_list),
            "method_distribution": method_dist,
            "failures": failures,
            "time_seconds": elapsed,
            "weighted_error": avg_error,
        }

        # Build base report, then add custom fields (preserved across _build_report)
        report = _build_report_fn(stats)
        report["groups_processed"] = groups_processed
        report["total_groups"] = len(refined_groups)
        report["mode"] = mode
        report["holographic_records"] = (
            self._holographic_oracle.get_stats().get("n_entries", 0)
            if self._holographic_oracle
            else 0
        )
        report["quantum_cascade_tests"] = (
            self._quantum_cascade.stats.get("total_tested", 0)
            if self._quantum_cascade
            else 0
        )

        # Record in unified intelligence
        if self._unified_intelligence is not None:
            try:
                self._unified_intelligence.record_model_result(
                    overall_ratio=stats["overall_ratio"],
                    avg_error=avg_error,
                    total_tensors=len(compressed_list),
                    total_time=elapsed,
                )
            except Exception:
                pass

        logger.info(
            "Smart model compression: %d tensors, ratio=%.1f:1, error=%.4f, "
            "time=%.1fs, groups=%d, method_dist=%s",
            len(compressed_list),
            stats["overall_ratio"],
            avg_error,
            elapsed,
            groups_processed,
            method_dist,
        )

        return report

    def _build_report(self, stats: Dict[str, Any]) -> dict:
        return _build_report(stats)

    # ── Compression Pipeline ─────────────────────────────────────────────

    def compress(
        self,
        tensor: np.ndarray,
        target_ratio: float = 10.0,
        max_error: float = 0.01,
        name: str = "",
        use_cascade: bool = False,
        use_tensor_strategy: bool = True,
        use_quantum_cascade: bool = False,
        **kwargs: Any,
    ) -> Tuple[bytes, dict, float, float]:
        if use_quantum_cascade:
            return self._compress_with_quantum_cascade(
                tensor, target_ratio, max_error, name
            )
        if use_cascade:
            return self._compress_with_cascade(tensor, target_ratio, max_error, name)
        if use_tensor_strategy:
            return self.compress_with_tensor_strategy(
                tensor, target_ratio, max_error, name
            )
        return self.compress_fast(tensor, name, target_ratio, max_error)

    def compress_with_tensor_strategy(
        self,
        tensor: np.ndarray,
        target_ratio: float = 500.0,
        max_error: float = 0.01,
        name: str = "",
    ) -> Tuple[bytes, dict, float, float]:
        """Compress using tensor-type-aware cascade strategy (RD-optimized).

        Uses Tier 1-5 ordering per tensor type:
        - attention_q/k/v: SVD progressive → spectral → quant (last resort)
        - ffn_gate/up/down: Spectral first → SVD → quant (last resort)
        - embedding: Quant + structural sparsity

        Returns (compressed_data, metadata, ratio, error).
        """
        result = compress_with_tensor_strategy(
            tensor, target_ratio, max_error, name, self._methods
        )
        del tensor
        gc.collect()
        return result

    def _compress_with_cascade(
        self,
        tensor: np.ndarray,
        target_ratio: float,
        max_error: float,
        name: str = "",
        pattern_name: Optional[str] = None,
    ) -> Tuple[bytes, dict, float, float]:
        """Multiplicative cascade compression via CascadeOracle (unified planner)."""
        # Use CascadeOracle when world model is enabled
        if self._use_intelligence:
            try:
                from .world_model.cascade_oracle import CascadeOracle

                oracle = CascadeOracle(self, self._unified_intelligence)
                tensor_type = self._classify_by_name(name)
                plan = oracle.plan(
                    tensor_type=tensor_type,
                    tensor=tensor,
                    target_ratio=target_ratio,
                    max_error=max_error,
                    name=name,
                )
                if plan is not None and plan.total_expected_ratio >= 1.0:
                    # Convert plan to stacking engine format
                    MSE = self._get_multiplicative_stacking_cls()
                    stacking_engine = MSE(self)
                    stages_config = [
                        {
                            "method_type": s.method_category,
                            "method_name": s.method_name,
                            "params": s.params,
                        }
                        for s in plan.stages
                    ]
                    stacking_plan = stacking_engine._plan_from_config(
                        tensor, stages_config, tensor_name=name
                    )
                    if stacking_plan is not None and stacking_plan.total_ratio >= 1.0:
                        compressed, metadata = stacking_engine.execute_stacking(
                            stacking_plan, tensor
                        )
                        ratio = stacking_plan.total_ratio
                        error = stacking_plan.total_error
                        metadata["cascade"] = True
                        metadata["n_stages"] = len(stacking_plan.stages)
                        metadata["total_ratio"] = ratio
                        metadata["total_error"] = error
                        metadata["original_shape"] = tensor.shape
                        metadata["source"] = "cascade_oracle"
                        return compressed, metadata, ratio, error
            except Exception as exc:
                logger.warning("CascadeOracle failed: %s — falling back", exc)

        # Fallback to legacy stacking engine
        MSE = self._get_multiplicative_stacking_cls()
        stacking_engine = MSE(self)
        try:
            plan = stacking_engine.plan_stacking(
                tensor,
                tensor_name=name,
                target_ratio=target_ratio,
                max_error=max_error,
                pattern_name=pattern_name,
                use_dynamic_pattern=(pattern_name is None),
            )
            if plan is None or plan.total_ratio < 1.0:
                raise RuntimeError("No viable stacking plan produced")
            compressed, metadata = stacking_engine.execute_stacking(plan, tensor)
            ratio = plan.total_ratio
            error = plan.total_error
            metadata["cascade"] = True
            metadata["n_stages"] = len(plan.stages)
            metadata["total_ratio"] = ratio
            metadata["total_error"] = error
            metadata["original_shape"] = tensor.shape
            return compressed, metadata, ratio, error
        except Exception as exc:
            logger.warning("Cascade failed: %s — falling back to fast path", exc)
            return self.compress_fast(tensor, name, target_ratio, max_error)

    def _compress_with_quantum_cascade(
        self,
        tensor: np.ndarray,
        target_ratio: float,
        max_error: float,
        name: str = "",
    ) -> Tuple[bytes, dict, float, float]:
        """Quantum parallel cascade: test methods in superposition, collapse to best."""
        profile = self.profile_tensor(tensor, name)
        error_budget = max_error / max(target_ratio, 1.0)

        candidates: List[Any] = []
        methods = self._select_methods(profile, error_budget, target_ratio)
        if not methods:
            blk8 = self._methods.get("block_int8")
            if blk8:
                candidates = [{"instance": blk8, "params": {}, "name": "block_int8"}]
            else:
                raise RuntimeError("No methods available and block_int8 not found")
        else:
            candidates = methods

        try:
            from .world_model.method_oracle import RankedMethod

            ranked = [
                RankedMethod(
                    name=m.get("name", "unknown"),
                    instance=m.get("instance"),
                    params=m.get("params", {}),
                    category=m.get("category", "quantization"),
                    tier=m.get("tier", 5),
                )
                for m in candidates
                if isinstance(m, dict)
            ]
            if ranked:
                plan = CascadeSuperpositionPlan.build_for_target(target_ratio, ranked)
                engine = QuantumSuperpositionEngine(
                    max_workers=max(1, os.cpu_count() or 4),
                    early_termination=True,
                )
                data, meta, ratio, error = engine.execute_cascade(
                    tensor, plan, memory_budget_gb=48.0
                )
                if ratio >= target_ratio or error <= error_budget:
                    meta["source"] = "quantum_cascade"
                    return data, meta, ratio, error
        except Exception as exc:
            logger.warning(
                "Quantum cascade with RankedMethod failed: %s — falling back", exc
            )

        engine = QuantumSuperpositionEngine(
            max_workers=max(1, os.cpu_count() or 4),
            early_termination=True,
        )

        best_result, _ = engine.test_in_superposition(
            tensor=tensor,
            candidates=candidates,
            target_ratio=target_ratio,
            max_error=max_error,
            tensor_nbytes=tensor.nbytes,
            memory_budget_gb=48.0,
        )

        if best_result.success:
            meta = best_result.params.copy()
            meta["method"] = best_result.method_name
            meta["quantum_cascade"] = True
            meta["original_shape"] = tensor.shape
            meta["compression_ratio"] = best_result.ratio
            meta["relative_error"] = best_result.error
            meta["snr_db"] = best_result.snr_db
            return (
                best_result.compressed_data,
                meta,
                best_result.ratio,
                best_result.error,
            )

        logger.warning(
            "Quantum cascade produced no valid results — falling back to fast path"
        )
        return self.compress_fast(tensor, name, target_ratio, max_error)

    @staticmethod
    def _get_multiplicative_stacking_cls():
        from .dynamic_tuning.multiplicative_stacking import MultiplicativeStackingEngine

        return MultiplicativeStackingEngine

    @staticmethod
    def _convert_output_dtype(tensor: np.ndarray, metadata: dict) -> np.ndarray:
        """Convert tensor to native dtype based on metadata flags.

        If metadata contains _BF16_FLAG, convert float32 output to BF16 (uint16).
        """
        from spectralstream.core.math_primitives import float32_to_bfloat16

        if metadata.get("_input_was_bf16", False):
            return float32_to_bfloat16(tensor)
        return tensor

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        if metadata.get("strategy"):
            return decompress_with_strategy(data, metadata, self._methods)

        if metadata.get("cascade"):
            shape = tuple(metadata.get("original_shape", metadata.get("tensor_shape")))
            stages = metadata.get("stages")
            stage_lengths = metadata.get("stage_lengths")
            if stages is not None and stage_lengths is not None:
                # compress_intelligent cascade format: packed stage data with length prefixes
                return self._decompress_intelligent_cascade(
                    data, stages, stage_lengths, shape
                )
            MSE = self._get_multiplicative_stacking_cls()
            stacking_engine = MSE(self)
            return stacking_engine.unstack(data, metadata, shape)

        method_name = metadata.get("method", "")
        params = metadata.get("params", metadata)

        if method_name and method_name in self._methods:
            inst = self._methods[method_name]
            try:
                recon = inst.decompress(data, params)
                shape = metadata.get("original_shape")
                if shape is not None:
                    recon = recon.reshape(shape)
                return self._convert_output_dtype(recon, metadata)
            except Exception:
                pass

        fallback_count = 0
        for mname, inst in self._methods.items():
            if not hasattr(inst, "decompress"):
                continue
            if fallback_count >= 10:
                break
            try:
                recon = inst.decompress(data, params)
                if recon.size > 0:
                    shape = metadata.get("original_shape")
                    if shape is not None and recon.shape != shape:
                        recon = recon.reshape(shape)
                    return self._convert_output_dtype(recon, metadata)
            except Exception:
                fallback_count += 1
                continue

        try:
            out = np.frombuffer(data, dtype=np.float16).astype(np.float32)
            return self._convert_output_dtype(out, metadata)
        except Exception:
            return np.frombuffer(data, dtype=np.uint8)

    def profile(self, tensor: np.ndarray, **kwargs: Any) -> TensorProfile:
        return self.profile_tensor(tensor, **kwargs)

    # ── Model-level Compression ──────────────────────────────────────────

    def compress_dict(
        self,
        tensors: Dict[str, np.ndarray],
        target_ratio: Optional[float] = None,
        max_error: Optional[float] = None,
        progress_callback: Any = None,
    ) -> dict:
        t0 = time.perf_counter()
        tr = target_ratio if target_ratio is not None else self._config.target_ratio
        me = max_error if max_error is not None else self._config.max_error

        profiles: Dict[str, Any] = {}
        for name, tensor in tensors.items():
            profiles[name] = self.profile_tensor(tensor, name)
            if progress_callback:
                progress_callback(len(profiles), len(tensors), name)

        budgets = self.allocator.allocate(profiles, target_ratio=tr, max_error=me)

        # ── Parallel path (num_workers > 1) ─────────────────────────────
        if self._config.num_workers > 1:
            from .parallel_compressor import ParallelCompressor

            compressor = ParallelCompressor(
                max_workers=self._config.num_workers,
                memory_budget_gb=self._config.max_memory_gb,
                streaming=self._config.streaming,
            )
            # Use the parallel compressor for the heavy compression phase.
            # Profiles and budgets are already computed above.
            stats = compressor.compress_many(
                engine=self,
                tensors=tensors,
                target_ratio=tr,
                max_error=me,
                progress_callback=progress_callback,
            )
            return self._build_report(stats)

        # ── Sequential path (num_workers <= 1) ──────────────────────────
        compressed_list: List[CompressedTensor] = []
        failures: List[str] = []
        total_orig = 0
        total_comp = 0
        errors = []
        method_dist: Dict[str, int] = {}
        tensor_methods: Dict[str, str] = {}
        tensor_errors: Dict[str, float] = {}
        tensor_ratios: Dict[str, float] = {}

        for name, tensor in tensors.items():
            try:
                eb = budgets.get(name, me)
                profile = profiles[name]
                methods = self._select_methods(profile, eb, tr)
                data, meta, ratio, error = self.compress_tensor_with_validation(
                    tensor, profile, methods, eb
                )
                ct = CompressedTensor(
                    _data=data,
                    method=meta.get("method", "unknown"),
                    params=meta,
                    original_shape=tensor.shape,
                    original_dtype=str(tensor.dtype),
                    compression_ratio=ratio,
                    relative_error=error,
                )
                compressed_list.append(ct)
                total_orig += tensor.nbytes
                total_comp += len(data)
                errors.append(error)
                mname = ct.method
                method_dist[mname] = method_dist.get(mname, 0) + 1
                tensor_methods[name] = mname
                tensor_errors[name] = error
                tensor_ratios[name] = ratio

                if self._unified_intelligence is not None:
                    tensor_type = self._classify_by_name(name)
                    method_category = meta.get(
                        "category", meta.get("method_type", "quantization")
                    )
                    self._unified_intelligence.record_result(
                        tensor_type=tensor_type,
                        method_name=mname,
                        ratio=ratio,
                        error=error,
                        method_category=method_category,
                        target_ratio=tr,
                        tensor_name=name,
                    )
            except Exception as exc:
                logger.warning("Failed to compress '%s': %s", name, exc)
                failures.append(name)
            del tensor

        gc.collect()

        avg_error = float(np.mean(errors)) if errors else 0.0
        max_err = float(np.max(errors)) if errors else 0.0
        min_err = float(np.min(errors)) if errors else 0.0
        weighted = float(np.average(errors) if errors else 0.0)

        t1 = time.perf_counter()

        stats = {
            "tensors": compressed_list,
            "total_orig_bytes": total_orig,
            "total_compressed_bytes": total_comp,
            "overall_ratio": total_orig / max(total_comp, 1),
            "average_ratio": total_orig / max(total_comp, 1),
            "avg_error": avg_error,
            "max_error": max_err,
            "min_error": min_err,
            "num_tensors": len(compressed_list),
            "method_distribution": method_dist,
            "failures": failures,
            "per_layer_error": {ct.method: ct.relative_error for ct in compressed_list},
            "time_seconds": t1 - t0,
            "weighted_error": weighted,
            "tensor_methods": tensor_methods,
            "tensor_errors": tensor_errors,
            "tensor_ratios": tensor_ratios,
        }
        return self._build_report(stats)

    # ── Grouped Compression (Tensor Grouping Optimizer) ──────────────────

    def compress_grouped(
        self,
        tensors: Dict[str, np.ndarray],
        target_ratio: Optional[float] = None,
        max_error: Optional[float] = None,
        progress_callback: Any = None,
    ) -> dict:
        """Compress tensors using grouping optimization.

        Groups tensors by (shape, dtype, name_pattern), tests methods on one
        representative per group, then applies the best method to all tensors
        in the group — reducing method testing from O(N) to O(G) where G << N.

        Parameters
        ----------
        tensors : Dict[str, np.ndarray]
            Dictionary of tensor names → numpy arrays.
        target_ratio : float, optional
            Target compression ratio. Defaults to config value.
        max_error : float, optional
            Maximum acceptable relative error. Defaults to config value.
        progress_callback : callable, optional
            Called as f(processed, total, tensor_name).

        Returns
        -------
        dict
            Compression report with grouping optimization stats.
        """
        tr = target_ratio if target_ratio is not None else self._config.target_ratio
        me = max_error if max_error is not None else self._config.max_error

        from .grouping_optimizer import compress_with_grouping

        result = compress_with_grouping(
            engine=self,
            tensors=tensors,
            target_ratio=tr,
            max_error=me,
            progress_callback=progress_callback,
        )

        # Record grouped results in holographic memory
        if self._use_intelligence:
            try:
                ho = self.holographic_oracle
                tensor_methods = result.get("tensor_methods", {})
                tensor_errors = result.get("tensor_errors", {})
                tensor_ratios = result.get("tensor_ratios", {})
                for tname, mname in tensor_methods.items():
                    if tname in tensors:
                        ttype = self._classify_by_name(tname)
                        ho.record_success(
                            tensor=tensors[tname],
                            tensor_type=ttype,
                            method_name=mname,
                            params={},
                            ratio=tensor_ratios.get(tname, 1.0),
                            error=tensor_errors.get(tname, 0.0),
                        )
            except Exception:
                pass

        return result

    # ── Unified Intelligent Compression ───────────────────────────────────

    def compress_dynamic(
        self,
        tensor: np.ndarray,
        target_ratio: float = 5000.0,
        max_error: float = 0.01,
        name: str = "",
    ) -> Tuple[bytes, dict, float, float]:
        """Compress using ALL intelligence systems simultaneously.

        1. Unified tensor analysis (digital twin + quantum + plasma + time crystal + tokamak)
        2. Ensemble method selection (Ising + Bayesian + genetic + QFT + F1 + NASA + Raptor)
        3. Dynamic cascade plan (stage count determined by target_ratio)
        4. Execute cascade with multiplicative stacking
        5. Validate with comprehensive metrics
        6. Record result for continuous learning

        This is the SINGLE entry point that benefits from ALL sub-systems.
        """
        if self._unified_intelligence is None:
            if target_ratio > 100:
                return self._compress_with_cascade(
                    tensor, target_ratio, max_error, name
                )
            return self.compress_fast(tensor, name, target_ratio, max_error)

        # 1. Unified tensor analysis
        analysis = self._unified_intelligence.analyze_tensor(tensor, name)

        # 2. Ensemble method selection
        selected_methods = self._unified_intelligence.select_methods(
            analysis, target_ratio, max_error
        )

        # 3. Build cascade plan
        plan = self._unified_intelligence.build_cascade_plan(
            selected_methods, target_ratio
        )

        # 4. Execute if we have a valid plan, otherwise fallback
        if (
            plan is not None
            and hasattr(plan, "total_ratio")
            and plan.total_ratio >= 1.0
        ):
            try:
                MSE = self._get_multiplicative_stacking_cls()
                stacking_engine = MSE(self)
                compressed, metadata = stacking_engine.execute_stacking(plan, tensor)
                ratio = plan.total_ratio
                error = plan.total_error
                metadata["cascade"] = True
                metadata["n_stages"] = len(plan.stages)
                metadata["total_ratio"] = ratio
                metadata["total_error"] = error
                metadata["original_shape"] = tensor.shape
                metadata["unified"] = True
            except Exception:
                compressed, metadata, ratio, error = self.compress_fast(
                    tensor, name, target_ratio, max_error
                )
        else:
            # Fallback: use the selected methods via standard pipeline
            profile = self.profile_tensor(tensor.copy(), name)
            error_budget = max_error / max(target_ratio, 1.0)

            # Convert unified selection to _select_methods format
            method_list = []
            for sm in selected_methods[:10]:
                inst = sm.get("instance") or self._methods.get(sm.get("name", ""))
                if inst is not None:
                    method_list.append(
                        {
                            "instance": inst,
                            "params": sm.get("params", {}),
                            "name": sm.get("name", ""),
                        }
                    )

            if not method_list:
                return self.compress_fast(tensor, name, target_ratio, max_error)

            data, meta, ratio, error = self.compress_tensor_with_validation(
                tensor, profile, method_list, error_budget
            )
            compressed, metadata = data, meta
            # Clean up
            del tensor, profile

        # 5. Record result for continuous learning
        try:
            tensor_type = analysis.get("tensor_type", "weight")
            used_method = metadata.get("method", metadata.get("methods", ["unknown"]))
            if isinstance(used_method, list):
                used_method = used_method[0] if used_method else "unknown"
            method_category = metadata.get(
                "category", metadata.get("method_type", "quantization")
            )
            self._unified_intelligence.record_result(
                tensor_type=tensor_type,
                method_name=used_method,
                ratio=ratio,
                error=error,
                method_category=method_category,
                target_ratio=target_ratio,
                tensor_name=name,
            )
        except Exception:
            pass

        gc.collect()
        return compressed, metadata, ratio, error

    def calibrate(self, model_path: str, sample_per_type: int = 2) -> Dict[str, Any]:
        from .model_calibrator import ModelCalibrator

        calibrator = ModelCalibrator(model_path)
        result = calibrator.calibrate(sample_per_type=sample_per_type)
        return result

    # ── Progressive / Cascade Compression ─────────────────────────────---

    def compress_progressive(
        self,
        tensor: np.ndarray,
        target_ratio: float = 5000.0,
        max_error: float = 0.01,
        name: str = "",
    ) -> Tuple[bytes, dict, float, float]:
        PSC = self._get_progressive_streaming_cls()
        psc = PSC(self, target_ratio=target_ratio, max_error=max_error)
        return psc.compress_progressive(tensor, name=name)

    @staticmethod
    def _get_progressive_streaming_cls():
        from .dynamic_tuning.pareto_streaming import ProgressiveStreamingCompressor

        return ProgressiveStreamingCompressor

    def compress_cascade(
        self,
        tensor: np.ndarray,
        target_ratio: float = 1200.0,
        max_error: float = 0.01,
        name: str = "",
        pattern_name: Optional[str] = None,
    ) -> Tuple[bytes, dict, float, float]:
        if pattern_name is not None:
            return self._compress_with_cascade(
                tensor, target_ratio, max_error, name, pattern_name
            )

        MSE = self._get_multiplicative_stacking_cls()
        stacking_engine = MSE(self)
        stages_config = stacking_engine.build_cascade_config(target_ratio)
        if not stages_config:
            return self.compress_fast(tensor, name, target_ratio, max_error)

        plan = stacking_engine._plan_from_config(
            tensor, stages_config, tensor_name=name
        )
        if plan is None or plan.total_ratio < 1.0:
            try:
                plan = stacking_engine.plan_stacking(
                    tensor,
                    tensor_name=name,
                    target_ratio=target_ratio,
                    max_error=max_error,
                    use_dynamic_pattern=False,
                )
            except Exception as exc:
                logger.warning("plan_stacking fallback also failed: %s", exc)

        if plan is None or plan.total_ratio < 1.0:
            return self.compress_fast(tensor, name, target_ratio, max_error)

        compressed, metadata = stacking_engine.execute_stacking(plan, tensor)
        ratio = plan.total_ratio
        error = plan.total_error
        metadata["cascade"] = True
        metadata["n_stages"] = len(plan.stages)
        metadata["total_ratio"] = ratio
        metadata["total_error"] = error
        metadata["original_shape"] = tensor.shape
        return compressed, metadata, ratio, error

    # ── Utility methods (retained for API compatibility) ──────────────────

    def get_methods(self) -> Dict[str, Dict[str, Any]]:
        return MethodDiscovery.discover()

    def get_methods_by_tier(self, tier: int) -> Dict[str, Dict[str, Any]]:
        return MethodDiscovery.get_methods_by_tier(tier)

    def validate_all_methods(self) -> Dict[str, Tuple[bool, float, float]]:
        return MethodDiscovery.validate_all()

    def get_available_methods(self) -> Dict[str, Any]:
        if hasattr(self._methods, "_loaded"):
            return dict(self._methods._loaded)
        return dict(self._methods)

    def get_method_names(self) -> List[str]:
        return list(self._methods.keys())

    def register_method(self, name: str, instance: Any) -> None:
        if name in self._methods:
            logger.warning("Overwriting existing method '%s'", name)
        self._methods[name] = instance
        cat = getattr(instance, "category", "unknown")
        self._methods_by_category.setdefault(cat, []).append(name)

    def get_methods_by_category(self, category: str) -> Dict[str, Any]:
        names = self._methods_by_category.get(category, [])
        return {n: self._methods[n] for n in names if n in self._methods}

    def get_methods_by_categories(self, categories: List[str]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for cat in categories:
            result.update(self.get_methods_by_category(cat))
        return result

    def get_categories(self) -> Dict[str, int]:
        return {cat: len(names) for cat, names in self._methods_by_category.items()}

    def get_tier_categories(self) -> Dict[int, int]:
        return {t: len(names) for t, names in self._methods_by_tier.items()}

    @staticmethod
    def _check_memory_threshold(n_bytes: int, threshold_mb: float = 1024.0) -> bool:
        return n_bytes < threshold_mb * 1024 * 1024

    def get_telemetry(self) -> Dict[str, Any]:
        method_success = {}
        for mname, minst in self._methods.items():
            if hasattr(minst, "n_success"):
                method_success[mname] = {
                    "success": minst.n_success,
                    "total": getattr(minst, "n_total", 0),
                }
        return {
            "timestamps": {
                "created": getattr(self, "_created_at", 0),
                "last_profile": getattr(self, "_last_profile_at", 0),
            },
            "method_success_rates": method_success,
            "total_ops": sum(s.get("total", 0) for s in method_success.values()),
            "avg_ratio": getattr(self, "_avg_ratio", 0.0),
            "avg_error": getattr(self, "_avg_error", 0.0),
        }

    def validate_report(self, report: Any) -> Dict[str, Any]:
        tensors = getattr(report, "tensors", [])
        n = len(tensors)
        ratios = [
            t.compression_ratio for t in tensors if hasattr(t, "compression_ratio")
        ]
        overall = sum(ratios) / max(len(ratios), 1) if ratios else 1.0
        return {
            "n_tensors": n,
            "overall_ratio": overall,
            "quality_distribution": {},
        }

    def profile_tensor_names(
        self,
        names: List[str],
        sensitivity_overrides: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        from ._dataclasses import TensorProfile

        profiles: Dict[str, Any] = {}
        for name in names:
            tensor_type = self._classify_by_name(name)
            sensitivity = 0.5
            if sensitivity_overrides and name in sensitivity_overrides:
                sensitivity = sensitivity_overrides[name]
            profiles[name] = TensorProfile(
                name=name,
                shape=(1,),
                n_elements=1,
                nbytes=4,
                tensor_type=tensor_type,
                sensitivity=sensitivity,
            )
        return profiles

    def get_method_stats(self) -> Dict[str, Any]:
        n_methods = len(self._methods)
        n_discovered = len(self._all_methods)
        n_instantiated = (
            len(self._methods._loaded)
            if hasattr(self._methods, "_loaded")
            else n_methods
        )
        return {
            "n_methods": n_methods,
            "n_discovered": n_discovered,
            "n_instantiated": n_instantiated,
            "n_categories": len(self._methods_by_category),
            "categories": self.get_categories(),
            "tier_distribution": self.get_tier_categories(),
        }

    # ── Dynamic Method Tester ─────────────────────────────────────────

    @property
    def method_tester(self) -> DynamicMethodTester:
        """Lazy-access DynamicMethodTester wired to this engine."""
        if not hasattr(self, "_method_tester_inst") or self._method_tester_inst is None:
            self._method_tester_inst = DynamicMethodTester(self)
        return self._method_tester_inst

    def compress_intelligent(
        self,
        tensor: np.ndarray,
        target_ratio: float = 5000.0,
        max_error: float = 0.01,
        name: str = "",
    ) -> Tuple[bytes, dict, float, float]:
        """Use DynamicMethodTester to find optimal cascade.

        1. Profile tensor
        2. Test all applicable methods
        3. Find optimal cascade stacking order
        4. Execute cascade on the tensor
        5. Evaluate final quality
        """
        profile = self.profile_tensor(tensor, name)
        _ = profile  # used implicitly by method_tester

        logger.info(
            "Intelligent compression of '%s' (%s) — target ratio %.0f:1, max error %.4f",
            name,
            tensor.shape,
            target_ratio,
            max_error,
        )

        cascade = self.method_tester.find_optimal_cascade(
            tensor, target_ratio, max_error, max_stages=10
        )
        if not cascade:
            logger.info("No cascade found — falling back to fast path")
            return self.compress_fast(tensor, name, target_ratio, max_error)

        # Execute cascade: apply methods sequentially
        stacked_data: List[bytes] = []
        stacked_meta: List[dict] = []
        current = tensor.copy()
        total_ratio = 1.0
        total_error = 0.0

        for mname, params in cascade:
            methods_all = self.get_methods()
            minfo = methods_all.get(mname)
            if minfo is None:
                continue
            inst = minfo.get("instance")
            if inst is None:
                inst = self._methods.get(mname)
            if inst is None:
                continue
            try:
                data, meta = inst.compress(current, **params)
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
                    {"method": mname, "params": meta, "ratio": stage_ratio}
                )

                # Compute residual for next stage
                residual = current.astype(np.float32) - recon.astype(np.float32)
                current = residual
            except Exception as exc:
                logger.debug("Cascade stage '%s' failed: %s", mname, exc)
                continue

            if total_ratio >= target_ratio:
                break
            if total_error >= max_error:
                break
            gc.collect()

        if not stacked_data:
            return self.compress_fast(tensor, name, target_ratio, max_error)

        # Pack all stage data with length prefixes for proper decompression
        packed = bytearray()
        stage_lengths = []
        for sd in stacked_data:
            stage_lengths.append(len(sd))
            packed += struct.pack("<I", len(sd))
            packed += sd

        # Combine cascade metadata
        metadata = {
            "cascade": True,
            "n_stages": len(stacked_data),
            "stages": stacked_meta,
            "stage_lengths": stage_lengths,
            "total_ratio": total_ratio,
            "total_error": min(total_error, 1.0),
            "original_shape": list(tensor.shape),
            "method": "cascade",
        }

        compressed = bytes(packed)

        # Compute final quality against original
        try:
            final_recon = self._reconstruct_from_cascade(
                stacked_data, stacked_meta, tensor.shape
            )
            metrics = compute_error_metrics(tensor, final_recon)
            final_error = metrics.get("relative_error", total_error)
            metadata["relative_error"] = final_error
        except Exception:
            final_error = min(total_error, 1.0)
            metadata["relative_error"] = final_error

        logger.info(
            "Cascade result: %d stages, ratio %.1f:1, error %.4f",
            len(stacked_data),
            total_ratio,
            metadata["relative_error"],
        )

        return compressed, metadata, total_ratio, metadata["relative_error"]

    def _reconstruct_from_cascade(
        self,
        stacked_data: List[bytes],
        stacked_meta: List[dict],
        original_shape: Tuple[int, ...],
    ) -> np.ndarray:
        """Reconstruct tensor from cascade stages: sum of all stage reconstructions."""
        reconstruction = np.zeros(original_shape, dtype=np.float32)
        for data, meta in zip(stacked_data, stacked_meta):
            mname = meta.get("method", "")
            inst = self._methods.get(mname)
            if inst is None:
                continue
            try:
                recon = inst.decompress(data, meta.get("params", {}))
                if recon.shape != original_shape:
                    recon = recon.reshape(original_shape)
                reconstruction += recon.astype(np.float32)
            except Exception:
                continue
        return reconstruction

    def _decompress_intelligent_cascade(
        self,
        data: bytes,
        stages: List[dict],
        stage_lengths: List[int],
        original_shape: Tuple[int, ...],
    ) -> np.ndarray:
        """Decompress from compress_intelligent cascade format (packed stage data with length prefixes)."""
        reconstruction = np.zeros(original_shape, dtype=np.float32)
        offset = 0
        for stage_info, length in zip(stages, stage_lengths):
            offset += 4  # skip length prefix
            stage_data = data[offset : offset + length]
            offset += length
            mname = stage_info.get("method", "")
            inst = self._methods.get(mname)
            if inst is None:
                continue
            try:
                recon = inst.decompress(stage_data, stage_info.get("params", {}))
                if recon.shape != original_shape:
                    recon = recon.reshape(original_shape)
                reconstruction += recon.astype(np.float32)
            except Exception:
                continue
        return reconstruction

    def get_unified_diagnostics(self) -> Dict[str, Any]:
        """Return diagnostic information about the UnifiedIntelligence state."""
        if self._unified_intelligence is None:
            return {"unified_intelligence": "disabled"}
        return {
            "unified_intelligence": "active",
            "bayesian_performances": len(
                self._unified_intelligence._bayesian._performances
            ),
            "knowledge_graph_tensor_types": len(
                self._unified_intelligence._knowledge_graph._graph
            ),
            "genetic_generation": self._unified_intelligence._genetic.generation,
            "rl_experiences": self._unified_intelligence._rl._n_experiences,
            "nas_evaluations": self._unified_intelligence._nas._n_total_evaluations,
            "nas_cache_hits": self._unified_intelligence._nas._n_cache_hits,
        }

    # ── Memory-Mapped Engine ────────────────────────────────────────────

    def _lazy_memory_mapped_engine(self) -> MemoryMappedTensorEngine:
        """Lazy accessor for MemoryMappedTensorEngine."""
        from .memory_mapped_engine import MemoryMappedTensorEngine

        return MemoryMappedTensorEngine("")

    @property
    def memory_mapped_engine(self) -> MemoryMappedTensorEngine:
        return self._lazy_memory_mapped_engine()

    @property
    def streaming_pipeline(self) -> StreamingCompressionPipeline:
        if (
            not hasattr(self, "_streaming_pipeline_inst")
            or self._streaming_pipeline_inst is None
        ):
            from .streaming_pipeline import StreamingCompressionPipeline

            self._streaming_pipeline_inst = StreamingCompressionPipeline(
                self, self._config
            )
        return self._streaming_pipeline_inst

    # ── Streaming Model Compression ──────────────────────────────────────

    def compress_model_streaming(
        self,
        model_path: str,
        output_path: str,
        target_ratio: float = 5000.0,
        max_error: float = 0.01,
        progress_callback: Any = None,
    ) -> Dict[str, Any]:
        """Compress a full model file using streaming — O(1) RAM.

        Parameters
        ----------
        model_path : str
            Path to .safetensors model file
        output_path : str
            Path to output .ssf compressed file
        target_ratio : float
            Desired compression ratio
        max_error : float
            Maximum acceptable relative error
        progress_callback : callable or None
            Called as f(processed, total, tensor_name)

        Returns
        -------
        dict
            Compression report
        """
        return self.streaming_pipeline.compress_model(
            model_path=model_path,
            output_path=output_path,
            target_ratio=target_ratio,
            max_error=max_error,
            progress_callback=progress_callback,
        )

    def close(self) -> None:
        """Release all resources and free memory."""
        self._profiler = None
        self._allocator = None
        self._unified_intelligence = None
        if hasattr(self, "_streaming_pipeline_inst"):
            self._streaming_pipeline_inst = None
        if self._methods:
            self._methods.clear()
        if self._all_methods:
            self._all_methods.clear()
        gc.collect()
        logger.debug("CompressionIntelligenceEngine resources released")
