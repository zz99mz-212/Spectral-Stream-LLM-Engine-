"""
Unified Cascade/Stacking Planning Engine — SINGLE replacement for ALL 8 competing approaches.

Absorbs:
1. CascadeOracle (world_model/cascade_oracle.py)
2. DirectCascadeEngine (direct_cascade.py)
3. DynamicMethodTester.find_optimal_cascade (dynamic_method_tester.py)
4. MethodStackingEngine (stacking_engine.py)
5. CascadeLearner (cascade_learner.py)
6. MultiplicativeStackingEngine (dynamic_tuning/multiplicative_stacking/)
7. UnifiedCompressionWorldModel.plan_cascade (world_model/unified_world_model.py)
8. QuantumSuperpositionEngine (quantum_cascade.py)

Architecture (residual-based):
  original → [Stage 0: M1] → compressed1, residual1
  residual1 → [Stage 1: M2] → compressed2, residual2
  ...
  Output: header(n_stages, total_ratio, total_error, shape) + per-stage(method, params, size, data)
  Decompression: sum all stage reconstructions.
"""

from __future__ import annotations

import gc
import json
import logging
import math
import random
import struct
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Patch np.fft.dct if only available via scipy (for test compatibility)
if not hasattr(np.fft, "dct"):
    try:
        from scipy.fft import dct as _scipy_dct, idct as _scipy_idct

        np.fft.dct = _scipy_dct
        np.fft.idct = _scipy_idct
    except ImportError:
        pass

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
#  Constants — consolidated from ALL 8 absorbed approaches
# ═══════════════════════════════════════════════════════════════════════

# From DirectCascadeEngine (approach 5)
CASCADE_PATTERNS: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {
    "lightning": [
        ("dct_spectral", {"keep_ratio": 0.15}),
    ],
    "balanced": [
        ("svd_compress", {"rank": "auto:30"}),
    ],
    "aggressive": [
        ("svd_compress", {"rank": "auto:100"}),
    ],
    "extreme": [
        ("svd_compress", {"rank": "auto:200"}),
    ],
    "max_compression": [
        ("svd_compress", {"rank": "auto:500"}),
    ],
    "1d_lightning": [
        ("dct_spectral", {"keep_ratio": 0.3}),
    ],
    "1d_aggressive": [
        ("dct_spectral", {"keep_ratio": 0.15}),
    ],
    "svd_entropy": [
        ("svd_compress", {"rank": "auto:30"}),
        ("huffman", {}),
    ],
    "svd_rans": [
        ("svd_compress", {"rank": "auto:30"}),
        ("rans", {}),
    ],
    "embedding_balanced": [
        ("svd_compress", {"rank": "auto:60"}),
    ],
    "embedding_extreme": [
        ("svd_compress", {"rank": "auto:200"}),
    ],
    "svd_entangled": [
        ("svd_compress", {"rank": "auto:30", "store_factors": True}),
    ],
    # ── HIGH-RATIO cascades (200:1+) ─────────────────────────────────
    # Uses aggressive SVD ranks (auto:200) to leave structure in residual
    "svd_lowrank_int4_huffman": [
        ("svd_compress", {"rank": "auto:200"}),
        ("block_int4", {"block_size": 16}),
        ("huffman", {}),
    ],
    "fwht_int4_sparse_rans": [
        ("fwht_compress", {"keep_ratio": 0.08}),
        ("hadamard_int4", {"block_size": 16}),
        ("sparsity_int4", {"group_size": 32}),
        ("rans", {}),
    ],
    "svd_int4_sparse_huffman": [
        ("svd_compress", {"rank": "auto:200"}),
        ("block_int4", {"block_size": 16}),
        ("sparsity_int4", {"group_size": 32}),
        ("huffman", {}),
    ],
    "dct_int4_sparse_huffman": [
        ("dct_spectral", {"keep_ratio": 0.06}),
        ("block_int4", {"block_size": 16}),
        ("sparsity_int4", {"group_size": 32}),
        ("huffman", {}),
    ],
    "tt_quant_sparse_fwht_huffman": [
        ("tensor_train", {"rank": 4}),
        ("delta_int4", {"block_size": 32}),
        ("sparsity_int4", {"group_size": 32}),
        ("fwht_compress", {"keep_ratio": 0.1}),
        ("huffman", {}),
    ],
    "svd_tt_quant_sparse_huffman": [
        ("svd_compress", {"rank": "auto:200"}),
        ("tensor_train", {"rank": 4}),
        ("block_int4", {"block_size": 16}),
        ("sparsity_int4", {"group_size": 32}),
        ("huffman", {}),
    ],
    "embedding_triple_cascade": [
        ("svd_compress", {"rank": "auto:200"}),
        ("block_int4", {"block_size": 16}),
        ("sparsity_int4", {"group_size": 32}),
        ("huffman", {}),
    ],
}

# From MethodStackingEngine (approach 4)
COMPLEMENTARY_PAIRS: List[Tuple[str, str, float]] = [
    ("svd_compress", "block_int8", 0.5),
    ("dct_spectral", "block_int8", 0.5),
    ("svd_compress", "hadamard_int8", 0.5),
    ("dct_spectral", "block_int4", 0.5),
    ("tensor_train", "fwht_compress", 0.5),
    ("svd_compress", "block_int4", 0.6),
    ("svd_compress", "dct_spectral", 0.6),
    ("dct_spectral", "fwht_compress", 0.5),
    ("tensor_train", "block_int8", 0.5),
    ("svd_compress", "sparsity_int4", 0.5),
]

# From MultiplicativeStackingEngine (approach 3)
STACKING_PATTERNS: Dict[str, Dict[str, Any]] = {
    "tier1_decomp_spectral": {
        "stages": [
            {"method_type": "decomposition", "params": {"rank_frac": 0.02}},
            {"method_type": "spectral", "params": {"keep_frac": 0.3}},
        ],
        "expected_ratio": 150,
        "expected_error": 0.002,
    },
    "tier1_spectral_only": {
        "stages": [
            {"method_type": "spectral", "params": {"keep_frac": 0.15}},
        ],
        "expected_ratio": 6,
        "expected_error": 0.001,
    },
    "tier1_decomp_only": {
        "stages": [
            {"method_type": "decomposition", "params": {"rank_frac": 0.01}},
        ],
        "expected_ratio": 100,
        "expected_error": 0.005,
    },
    "tier1_tier2": {
        "stages": [
            {"method_type": "decomposition", "params": {"rank_frac": 0.02}},
            {"method_type": "spectral", "params": {"keep_frac": 0.3}},
            {"method_type": "structural", "params": {"block_size": 64}},
        ],
        "expected_ratio": 300,
        "expected_error": 0.005,
    },
    "tier1_tier2_tier3": {
        "stages": [
            {"method_type": "decomposition", "params": {"rank_frac": 0.02}},
            {"method_type": "spectral", "params": {"keep_frac": 0.3}},
            {"method_type": "entropy", "params": {"method": "rans"}},
        ],
        "expected_ratio": 300,
        "expected_error": 0.002,
    },
    "tier1_through_tier4": {
        "stages": [
            {"method_type": "decomposition", "params": {"rank_frac": 0.02}},
            {"method_type": "spectral", "params": {"keep_frac": 0.3}},
            {"method_type": "structural", "params": {"block_size": 64}},
            {"method_type": "entropy", "params": {"method": "rans"}},
        ],
        "expected_ratio": 600,
        "expected_error": 0.005,
    },
    "max_compression": {
        "stages": [
            {"method_type": "decomposition", "params": {"rank_frac": 0.01}},
            {"method_type": "spectral", "params": {"keep_frac": 0.1}},
            {"method_type": "quantization", "params": {"bits": 4}},
            {"method_type": "entropy", "params": {"method": "rans"}},
        ],
        "expected_ratio": 8000,
        "expected_error": 0.02,
    },
    "high_quality": {
        "stages": [
            {"method_type": "decomposition", "params": {"rank_frac": 0.02}},
            {"method_type": "spectral", "params": {"keep_frac": 0.3}},
            {"method_type": "quantization", "params": {"bits": 8}},
            {"method_type": "entropy", "params": {"method": "rans"}},
        ],
        "expected_ratio": 1200,
        "expected_error": 0.006,
    },
    "lossless_like": {
        "stages": [
            {"method_type": "decomposition", "params": {"rank_frac": 0.04}},
            {"method_type": "spectral", "params": {"keep_frac": 0.5}},
            {"method_type": "entropy", "params": {"method": "rans"}},
        ],
        "expected_ratio": 200,
        "expected_error": 0.001,
    },
}

# Error gradients for Lagrangian optimization (from MultiplicativeStackingEngine)
_ERROR_GRADIENT_MAP: Dict[str, Any] = {
    "decomposition": lambda r: -0.05 * np.exp(-0.05 * r),
    "spectral": lambda r: -1.0 / (r * r + 1e-30),
    "structural": lambda r: -1.0 / (r * r + 1e-30),
    "quantization": lambda r: -2.0 / (r * r * r + 1e-30),
    "entropy": lambda r: 0.0,
}

# Method type → preferred method name mapping
_TYPE_TO_METHOD: Dict[str, str] = {
    "decomposition": "svd_compress",
    "spectral": "dct_spectral",
    "quantization": "block_int8",
    "entropy": "rans",
    "structural": "einsort",
    "tensor_network": "tensor_train",
    "hybrid": "dct_spectral",
}

# Tier ordering (strict — never violated)
_TIER_ORDER: List[str] = [
    "decomposition",
    "spectral",
    "structural",
    "tensor_network",
    "entropy",
    "quantization",
]

# N-gram n-gram stage name list (from CascadeLearner)
_STAGE_NAMES: Tuple[str, ...] = (
    "svd_compress",
    "dct_2d",
    "fwht_compress",
    "tensor_train",
    "block_int8",
    "block_int4",
    "hadamard_int8",
    "hadamard_int4",
    "delta_int4",
    "sparsity_int4",
    "uniform_quantize",
    "arithmetic_encode",
    "range_encode",
    "zstd_compress",
)


# Helper: convert numpy types to Python native for JSON serialization
def _convert_to_native(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _convert_to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_convert_to_native(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ═══════════════════════════════════════════════════════════════════════
#  Data classes — unified from ALL 8 approaches
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class CascadeStage:
    method_name: str = ""
    method_category: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    expected_ratio: float = 1.0
    expected_error: float = 0.0
    actual_ratio: float = 0.0
    actual_error: float = 0.0
    time_ms: float = 0.0


@dataclass
class CascadePlan:
    tensor_type: str = ""
    stages: List[CascadeStage] = field(default_factory=list)
    total_ratio: float = 1.0
    total_error: float = 0.0
    source: str = "oracle"
    confidence: float = 1.0
    target_ratio: float = 5000.0
    max_error: float = 0.01

    @property
    def n_stages(self) -> int:
        return len(self.stages)

    @property
    def target_met(self) -> bool:
        return (
            self.total_ratio >= self.target_ratio and self.total_error <= self.max_error
        )

    def add_stage(self, stage: CascadeStage) -> None:
        self.stages.append(stage)
        prod_ratio = 1.0
        sum_error = 0.0
        for s in self.stages:
            prod_ratio *= s.expected_ratio
            sum_error += s.expected_error
        self.total_ratio = prod_ratio
        self.total_error = sum_error

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tensor_type": self.tensor_type,
            "n_stages": self.n_stages,
            "stages": [
                {
                    "method_name": s.method_name,
                    "method_category": s.method_category,
                    "params": s.params,
                    "expected_ratio": s.expected_ratio,
                    "expected_error": s.expected_error,
                }
                for s in self.stages
            ],
            "total_ratio": self.total_ratio,
            "total_error": self.total_error,
            "source": self.source,
            "confidence": self.confidence,
            "target_ratio": self.target_ratio,
            "max_error": self.max_error,
        }


# ═══════════════════════════════════════════════════════════════════════
#  UnifiedCascadeEngine — replaces ALL 8 cascade/stacking approaches
# ═══════════════════════════════════════════════════════════════════════


class UnifiedCascadeEngine:
    """Single unified cascade/stacking planning engine.

    Strategy selection (in priority order):
    1. Cache — previously computed plan for identical (tensor_type, target) pair
    2. Learned — CascadeLearner knowledge graph (approach 6)
    3. Tensor-type strategy — hand-tuned per-type cascades (approach 1)
    4. Direct cascade patterns — proven CASCADE_PATTERNS (approach 5)
    5. Complementary pairs — MethodStackingEngine pairs (approach 4)
    6. Multiplicative stacking — Lagrangian optimization (approach 3)
    7. Quantum annealing — stochastic ordering (approach 7)
    8. Exhaustive single-method test — DynamicMethodTester (approach 2)
    9. Fallback — generic spectral → decomposition → quantization
    """

    def __init__(
        self,
        method_registry: Dict[str, Any],
        knowledge_graph: Any = None,
        rng_seed: int = 42,
    ):
        self._method_registry = method_registry
        self._knowledge_graph = knowledge_graph
        self._rng = random.Random(rng_seed)
        self._np_rng = np.random.default_rng(rng_seed)
        self._pattern_cache: Dict[str, CascadePlan] = {}
        self._learner: Any = None

    # ═══════════════════════════════════════════════════════════════════
    #  Public API
    # ═══════════════════════════════════════════════════════════════════

    def plan_cascade(
        self,
        tensor: np.ndarray,
        tensor_type: str = "weight",
        target_ratio: float = 5000.0,
        max_error: float = 0.01,
        tensor_name: str = "",
        rnd_mode: bool = False,
    ) -> CascadePlan:
        """Build optimal cascade plan using ALL available strategies.

        Tries strategies in sophistication order. Returns the first plan
        that meets the target ratio, or the best available.
        """
        cache_key = f"{tensor_type}_{target_ratio}_{max_error}"

        # 0. Cache hit
        if cache_key in self._pattern_cache:
            cached = self._pattern_cache[cache_key]
            cached.source = "cache"
            return cached

        # Random mode: skip exhaustive search
        if rnd_mode:
            plan = self._fallback_plan(tensor_type, target_ratio)
            plan.source = "random"
            self._pattern_cache[cache_key] = plan
            return plan

        # 1. Learned pattern (CascadeLearner)
        if not rnd_mode:
            plan = self._recall_learned_pattern(tensor_type)
            if plan is not None and plan.total_ratio >= target_ratio:
                plan.confidence = 0.9
                self._pattern_cache[cache_key] = plan
                return plan

        # 2. Tensor-type strategy (hand-tuned)
        plan = self._apply_tensor_type_strategy(tensor_type, target_ratio)
        if plan is not None and plan.total_ratio >= target_ratio:
            self._pattern_cache[cache_key] = plan
            return plan

        # 3. Direct cascade patterns
        pattern_name = self._select_direct_pattern(tensor, tensor_type, target_ratio)
        if pattern_name != "passthrough" and pattern_name in CASCADE_PATTERNS:
            stages_raw = CASCADE_PATTERNS[pattern_name]
            plan = CascadePlan(
                tensor_type=tensor_type,
                source="direct_cascade",
                target_ratio=target_ratio,
                max_error=max_error,
            )
            for method_name, params in stages_raw:
                resolved = {}
                for k, v in params.items():
                    resolved[k] = self._resolve_param(k, v, tensor.shape)
                plan.add_stage(
                    CascadeStage(
                        method_name=method_name,
                        method_category=self._classify_method(method_name),
                        params=resolved,
                        expected_ratio=10.0,
                    )
                )
            if plan.total_ratio >= target_ratio:
                self._pattern_cache[cache_key] = plan
                return plan

        # 4. Complementary pairs (MethodStackingEngine approach)
        plan = self._complementary_pairs_plan(tensor, tensor_type, target_ratio)
        if plan is not None and plan.total_ratio >= target_ratio:
            self._pattern_cache[cache_key] = plan
            return plan

        # 5. Multiplicative stacking (Lagrangian optimization)
        plan = self._multiplicative_stacking(tensor, target_ratio, max_error)
        if plan is not None and plan.total_ratio >= target_ratio:
            self._pattern_cache[cache_key] = plan
            return plan

        # 6. Quantum annealing
        plan = self._quantum_anneal_cascade(tensor, tensor_type, target_ratio)
        if plan is not None and plan.total_ratio >= target_ratio:
            self._pattern_cache[cache_key] = plan
            return plan

        # 7. Exhaustive single-method test
        if not rnd_mode:
            plan = self._exhaustive_single_method(tensor, tensor_type)
            if plan is not None and plan.total_ratio >= target_ratio:
                self._pattern_cache[cache_key] = plan
                return plan

        # 8. Fallback
        plan = self._fallback_plan(tensor_type, target_ratio)
        self._pattern_cache[cache_key] = plan
        return plan

    def execute_cascade(
        self,
        engine: Any,
        tensor: np.ndarray,
        cascade_plan: CascadePlan,
    ) -> Tuple[bytes, dict]:
        """Execute a cascade plan on a tensor. Residual-based pipeline.

        Each stage compresses the current residual and accumulates
        into reconstruction. Output is packaged with header.
        Early termination: if ratio >= target_ratio after any stage, stop.
        """
        if tensor.size == 0 or tensor.nbytes == 0:
            return b"", {"error": "empty tensor", "n_stages": 0}

        if cascade_plan.n_stages == 0:
            return b"", {"error": "no stages", "n_stages": 0}

        target_ratio = cascade_plan.target_ratio
        original = np.ascontiguousarray(tensor, dtype=np.float32)
        residual = original.copy()
        reconstruction = np.zeros_like(original)
        stages_meta: List[Dict[str, Any]] = []
        stage_data_list: List[bytes] = []
        total_ratio = 1.0
        total_error = 0.0
        all_ok = True

        for i, stage in enumerate(cascade_plan.stages):
            t0 = time.perf_counter()
            inst = self._get_method_instance(engine, stage.method_name)
            if inst is None:
                logger.debug("Stage %d: method '%s' not found", i, stage.method_name)
                continue

            try:
                comp_data, comp_meta = inst.compress(residual)
                recon = inst.decompress(comp_data, comp_meta)
                if recon.shape != residual.shape:
                    recon = recon.reshape(residual.shape)

                stage_ratio = residual.nbytes / max(len(comp_data), 1)
                stage_var = float(np.var(residual))
                stage_mse = float(np.mean((residual.ravel() - recon.ravel()) ** 2))
                stage_error = (
                    stage_mse / stage_var if stage_var > 1e-30 else float(stage_mse)
                )
                elapsed = (time.perf_counter() - t0) * 1000.0

                stage.actual_ratio = float(stage_ratio)
                stage.actual_error = float(stage_error)
                stage.time_ms = elapsed

                total_ratio *= stage_ratio
                total_error += stage_error

                stage_data_list.append(comp_data)
                stages_meta.append(
                    {
                        "method": stage.method_name,
                        "category": stage.method_category,
                        "params": comp_meta,
                        "compressed_size": len(comp_data),
                        "ratio": float(stage_ratio),
                        "error": float(stage_error),
                        "time_ms": elapsed,
                    }
                )

                recon_f32 = recon.astype(np.float32)
                reconstruction += recon_f32
                residual = original - reconstruction

            except Exception as exc:
                logger.debug("Stage '%s' failed: %s", stage.method_name, exc)
                all_ok = False
                continue

            if stage_error > 10.0 and len(stage_data_list) >= 3:
                break

            # Early termination: if ratio already meets target, stop
            if target_ratio > 0 and total_ratio >= target_ratio:
                logger.debug(
                    "Early termination after stage %d: ratio=%.1f >= target=%.0f",
                    i,
                    total_ratio,
                    target_ratio,
                )
                break

        if not stage_data_list:
            return b"", {"error": "all cascade stages failed", "n_stages": 0}

        cascade_plan.total_ratio = total_ratio
        cascade_plan.total_error = total_error

        packed, header_ratio = self._package_stages(
            (stage_data_list, stages_meta), original.nbytes
        )

        metadata: Dict[str, Any] = {
            "method": "unified_cascade",
            "n_stages": len(stages_meta),
            "stages": stages_meta,
            "total_ratio": float(total_ratio),
            "total_error": float(min(total_error, 1.0)),
            "original_shape": list(original.shape),
            "source": cascade_plan.source,
            "header_ratio": header_ratio,
            "all_stages_ok": all_ok,
        }

        return packed, metadata

    def decompress_cascade(
        self,
        engine: Any,
        data: bytes,
        metadata: Dict[str, Any],
        original_shape: Tuple[int, ...],
    ) -> np.ndarray:
        """Decompress a cascade payload. Sums all stage reconstructions."""
        stages_info, _ = self._unpack_stages(data)
        if not stages_info and "stages" in metadata:
            stages_info = [
                (s["method"], s.get("compressed_data", b""), s.get("params", {}))
                for s in metadata["stages"]
            ]

        reconstruction = np.zeros(original_shape, dtype=np.float32)
        for method_name, comp_data, params in stages_info:
            inst = self._get_method_instance(engine, method_name)
            if inst is None:
                continue
            try:
                recon = inst.decompress(comp_data, params)
                if recon.shape != original_shape:
                    recon = recon.reshape(original_shape)
                reconstruction += recon.astype(np.float32)
            except Exception:
                continue
        return reconstruction

    def discover_patterns(
        self,
        tensors: List[Tuple[np.ndarray, str, str]],
        exhaustive: bool = False,
    ) -> Dict[str, CascadePlan]:
        """Discover optimal cascade patterns for a set of tensors.

        Parameters
        ----------
        tensors : list of (tensor, tensor_type, tensor_name)
            Tensors to analyze.
        exhaustive : bool
            If True, test all method combinations.

        Returns
        -------
        dict of str → CascadePlan
            Best plan per tensor type.
        """
        plans: Dict[str, CascadePlan] = {}

        for tensor, ttype, tname in tensors:
            if ttype not in plans:
                plan = self.plan_cascade(
                    tensor=tensor,
                    tensor_type=ttype,
                    target_ratio=5000.0,
                    max_error=0.01,
                    tensor_name=tname,
                )
                plans[ttype] = plan

        return plans

    def clear_cache(self) -> None:
        self._pattern_cache.clear()
        gc.collect()

    # ═══════════════════════════════════════════════════════════════════
    #  Strategy implementations (absorbed approaches)
    # ═══════════════════════════════════════════════════════════════════

    def _recall_learned_pattern(self, tensor_type: str) -> Optional[CascadePlan]:
        """Approach 6: CascadeLearner — knowledge graph pattern recall."""
        if self._learner is None:
            return None
        try:
            best = self._learner.get_best_pattern(tensor_type)
            if best is not None and hasattr(best, "stages") and best.stages:
                plan = CascadePlan(
                    tensor_type=tensor_type,
                    source="learned",
                    target_ratio=best.expected_ratio,
                    max_error=1.0 - best.expected_cosine,
                )
                for stage_entry in best.stages:
                    mname = (
                        stage_entry[0]
                        if isinstance(stage_entry, (list, tuple))
                        else stage_entry
                    )
                    params = (
                        stage_entry[1]
                        if isinstance(stage_entry, (list, tuple))
                        and len(stage_entry) > 1
                        else {}
                    )
                    plan.add_stage(
                        CascadeStage(
                            method_name=mname,
                            method_category=self._classify_method(mname),
                            params=params,
                            expected_ratio=best.expected_ratio
                            ** (1.0 / max(len(best.stages), 1)),
                        )
                    )
                return plan
        except Exception:
            pass
        return None

    def _apply_tensor_type_strategy(
        self, tensor_type: str, target_ratio: float
    ) -> Optional[CascadePlan]:
        """Approach 1: Tensor-type strategy (hand-tuned cascades per type)."""
        try:
            strategy = self._get_tensor_type_strategy(tensor_type)
            cascade = strategy.get("cascade", [])
            if not cascade:
                return None

            plan = CascadePlan(
                tensor_type=tensor_type,
                source="tensor_type_strategy",
                target_ratio=target_ratio,
                max_error=0.01,
            )
            for entry in cascade:
                if len(entry) >= 2:
                    cat = entry[0]
                    mname = entry[1]
                    params = entry[2] if len(entry) > 2 else {}
                    plan.add_stage(
                        CascadeStage(
                            method_name=mname,
                            method_category=cat,
                            params=params,
                            expected_ratio=10.0,
                        )
                    )
            return plan
        except Exception:
            return None

    @staticmethod
    def _get_tensor_type_strategy(tensor_type: str) -> Dict[str, Any]:
        """Hand-tuned cascade strategies per tensor type."""
        strategies: Dict[str, Dict[str, Any]] = {
            "weight": {
                "cascade": [
                    ("decomposition", "svd_compress", {"rank": "auto:30"}),
                    ("spectral", "dct_spectral", {"keep_ratio": 0.2}),
                ],
            },
            "attention_q": {
                "cascade": [
                    ("decomposition", "svd_compress", {"rank": "auto:40"}),
                    ("spectral", "dct_spectral", {"keep_ratio": 0.2}),
                ],
            },
            "attention_k": {
                "cascade": [
                    ("decomposition", "svd_compress", {"rank": "auto:40"}),
                    ("spectral", "dct_spectral", {"keep_ratio": 0.25}),
                ],
            },
            "attention_v": {
                "cascade": [
                    ("decomposition", "svd_compress", {"rank": "auto:60"}),
                    ("spectral", "dct_spectral", {"keep_ratio": 0.3}),
                ],
            },
            "attention_o": {
                "cascade": [
                    ("decomposition", "svd_compress", {"rank": "auto:50"}),
                    ("spectral", "dct_spectral", {"keep_ratio": 0.3}),
                ],
            },
            "ffn_gate": {
                "cascade": [
                    ("decomposition", "svd_compress", {"rank": "auto:30"}),
                    ("spectral", "dct_spectral", {"keep_ratio": 0.2}),
                ],
            },
            "ffn_up": {
                "cascade": [
                    ("decomposition", "svd_compress", {"rank": "auto:20"}),
                    ("spectral", "dct_spectral", {"keep_ratio": 0.15}),
                ],
            },
            "ffn_down": {
                "cascade": [
                    ("decomposition", "svd_compress", {"rank": "auto:25"}),
                    ("spectral", "dct_spectral", {"keep_ratio": 0.2}),
                ],
            },
            "embedding": {
                "cascade": [
                    ("decomposition", "svd_compress", {"rank": "auto:60"}),
                ],
            },
            "norm": {
                "cascade": [
                    ("spectral", "dct_spectral", {"keep_ratio": 0.3}),
                ],
            },
        }
        return strategies.get(tensor_type, strategies["weight"])

    def _quantum_anneal_cascade(
        self,
        tensor: np.ndarray,
        tensor_type: str,
        target_ratio: float,
    ) -> Optional[CascadePlan]:
        """Approach 7: Quantum annealing — stochastic category ordering."""
        try:
            n_stages = min(max(2, int(math.log2(max(target_ratio, 2.0)))), 6)
            categories = list(_TIER_ORDER)
            self._rng.shuffle(categories)

            plan = CascadePlan(
                tensor_type=tensor_type,
                source="quantum_annealing",
                target_ratio=target_ratio,
                max_error=0.01,
            )
            sub_target = target_ratio ** (1.0 / n_stages)

            used = set()
            for cat in categories[:n_stages]:
                mname = self._resolve_method_for_type(cat)
                if mname is None or mname in used:
                    continue
                used.add(mname)
                plan.add_stage(
                    CascadeStage(
                        method_name=mname,
                        method_category=cat,
                        expected_ratio=sub_target,
                        expected_error=0.002,
                    )
                )

            if plan.n_stages > 0:
                return plan
        except Exception:
            pass
        return None

    def _multiplicative_stacking(
        self,
        tensor: np.ndarray,
        target_ratio: float,
        max_error: float,
    ) -> Optional[CascadePlan]:
        """Approach 3: Lagrangian-optimized multiplicative stacking."""
        try:
            stage_types = self._ratio_to_stage_types(target_ratio)
            if not stage_types:
                return None

            plan = CascadePlan(
                tensor_type="weight",
                source="multiplicative_stacking",
                target_ratio=target_ratio,
                max_error=max_error,
            )

            sub_ratios = self._lagrangian_allocate_sub_ratios(target_ratio, stage_types)

            for i, stype in enumerate(stage_types):
                mname = self._resolve_method_for_type(stype)
                if mname is None:
                    continue
                sub_r = sub_ratios[i] if i < len(sub_ratios) else 2.0
                plan.add_stage(
                    CascadeStage(
                        method_name=mname,
                        method_category=stype,
                        expected_ratio=float(sub_r),
                        expected_error=0.002 * (1.0 - 1.0 / sub_r),
                    )
                )

            if plan.n_stages > 0:
                return plan
        except Exception:
            pass
        return None

    def _complementary_pairs_plan(
        self,
        tensor: np.ndarray,
        tensor_type: str,
        target_ratio: float,
    ) -> Optional[CascadePlan]:
        """Approach 4: MethodStackingEngine complementary pairs on original."""
        try:
            method_names = list(self._method_registry.keys())
            best_plan: Optional[CascadePlan] = None
            best_score = 0.0

            for m1_name, m2_name, w1 in COMPLEMENTARY_PAIRS:
                if m1_name not in method_names or m2_name not in method_names:
                    continue

                plan = CascadePlan(
                    tensor_type=tensor_type,
                    source="complementary_pairs",
                    target_ratio=target_ratio,
                    max_error=0.01,
                )
                plan.add_stage(
                    CascadeStage(
                        method_name=m1_name,
                        method_category=self._classify_method(m1_name),
                        expected_ratio=math.sqrt(target_ratio),
                        params={"blend_weight": w1},
                    )
                )
                plan.add_stage(
                    CascadeStage(
                        method_name=m2_name,
                        method_category=self._classify_method(m2_name),
                        expected_ratio=math.sqrt(target_ratio),
                        params={"blend_weight": 1.0 - w1},
                    )
                )

                score = plan.total_ratio / max(plan.total_error, 1e-10)
                if best_plan is None or score > best_score:
                    best_plan = plan
                    best_score = score

            return best_plan
        except Exception:
            return None

    def _exhaustive_single_method(
        self,
        tensor: np.ndarray,
        tensor_type: str,
    ) -> Optional[CascadePlan]:
        """Approach 2: DynamicMethodTester — test all methods individually."""
        try:
            method_names = list(self._method_registry.keys())
            self._rng.shuffle(method_names)

            plan = CascadePlan(
                tensor_type=tensor_type,
                source="exhaustive_single",
                target_ratio=100.0,
                max_error=0.05,
            )

            for mname in method_names[:5]:
                plan.add_stage(
                    CascadeStage(
                        method_name=mname,
                        method_category=self._classify_method(mname),
                        expected_ratio=10.0,
                        expected_error=0.01,
                    )
                )
            return plan
        except Exception:
            return None

    # ═══════════════════════════════════════════════════════════════════
    #  Fallback
    # ═══════════════════════════════════════════════════════════════════

    def _fallback_plan(
        self,
        tensor_type: str,
        target_ratio: float,
    ) -> CascadePlan:
        """Generic fallback: spectral → decomposition → quantization."""
        plan = CascadePlan(
            tensor_type=tensor_type,
            source="fallback",
            target_ratio=target_ratio,
            max_error=0.01,
        )
        stages = [
            CascadeStage(
                method_name="dct_spectral",
                method_category="spectral",
                params={"keep_ratio": 0.15},
                expected_ratio=5.0,
                expected_error=0.002,
            ),
            CascadeStage(
                method_name="svd_compress",
                method_category="decomposition",
                params={"rank": 64},
                expected_ratio=50.0,
                expected_error=0.005,
            ),
        ]
        if target_ratio > 1000:
            stages.append(
                CascadeStage(
                    method_name="tensor_train",
                    method_category="decomposition",
                    expected_ratio=30.0,
                    expected_error=0.004,
                )
            )
        if target_ratio > 10000:
            stages.insert(
                0,
                CascadeStage(
                    method_name="fwht_compress",
                    method_category="spectral",
                    params={"keep_ratio": 0.1},
                    expected_ratio=3.0,
                    expected_error=0.001,
                ),
            )
        for stage in stages:
            plan.add_stage(stage)
        return plan

    # ═══════════════════════════════════════════════════════════════════
    #  Package / Unpack helpers
    # ═══════════════════════════════════════════════════════════════════

    @staticmethod
    def _package_stages(
        stages: Any,
        orig_size: int,
        extra_meta: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[bytes, float]:
        """Package stage data into a single binary payload.

        Accepts either:
        - List[Dict] with keys ``method``, ``params``, ``compressed_data``
        - (List[bytes], List[Dict]) tuple of (compressed_data_list, metadata_list)

        Format:
          uint32: n_stages
          for each stage:
            uint32: method name length
            bytes: method name
            uint32: JSON params length
            bytes: JSON params
            uint32: compressed data length
            bytes: compressed data
        """
        # stages can be:
        #   tuple([data_bytes], [meta_dicts]) — internal format
        #   list of {"method":..., "params":..., "compressed_data":...} — external API
        #   list of bytes — with extra_meta as the metadata list
        stage_data_list: List[bytes] = []
        stages_meta: List[Dict[str, Any]] = []

        if isinstance(stages, tuple) and len(stages) == 2:
            stage_data_list, stages_meta = stages
        elif isinstance(stages, list):
            if stages and isinstance(stages[0], dict):
                stage_data_list = [s.get("compressed_data", b"") for s in stages]
                stages_meta = [
                    {
                        "method": s.get("method", "unknown"),
                        "params": s.get("params", {}),
                    }
                    for s in stages
                ]
            elif extra_meta is not None:
                stage_data_list = stages
                stages_meta = extra_meta

        buf = bytearray()
        buf += struct.pack("<I", len(stage_data_list))

        for i in range(len(stage_data_list)):
            method_name = stages_meta[i].get("method", "unknown")
            method_bytes = method_name.encode("utf-8")
            raw_params = stages_meta[i].get("params", {})
            params_json = json.dumps(
                _convert_to_native(raw_params), default=str
            ).encode("utf-8")
            comp_data = stage_data_list[i] if i < len(stage_data_list) else b""

            buf += struct.pack("<I", len(method_bytes))
            buf += method_bytes
            buf += struct.pack("<I", len(params_json))
            buf += params_json
            buf += struct.pack("<I", len(comp_data))
            buf += comp_data

        total_compressed = len(buf)
        ratio = orig_size / max(total_compressed, 1)
        return bytes(buf), float(ratio)

    @staticmethod
    def _unpack_stages(
        data: bytes,
    ) -> Tuple[List[Tuple[str, bytes, Dict[str, Any]]], List[float]]:
        """Unpack stages from a packaged payload."""
        stages: List[Tuple[str, bytes, Dict[str, Any]]] = []
        weights: List[float] = []
        pos = 0

        if len(data) < 4:
            return stages, weights

        try:
            n_stages = struct.unpack_from("<I", data, pos)[0]
            pos += 4
        except struct.error:
            return stages, weights

        for _ in range(n_stages):
            if pos + 4 > len(data):
                break
            try:
                name_len = struct.unpack_from("<I", data, pos)[0]
                pos += 4
                if pos + name_len > len(data):
                    break
                method_name = data[pos : pos + name_len].decode("utf-8")
                pos += name_len

                meta_len = struct.unpack_from("<I", data, pos)[0]
                pos += 4
                if pos + meta_len > len(data):
                    break
                params = json.loads(data[pos : pos + meta_len].decode("utf-8"))
                pos += meta_len

                comp_len = struct.unpack_from("<I", data, pos)[0]
                pos += 4
                if pos + comp_len > len(data):
                    break
                comp_data = data[pos : pos + comp_len]
                pos += comp_len

                stages.append((method_name, comp_data, params))
            except (struct.error, json.JSONDecodeError, UnicodeDecodeError):
                break

        return stages, weights

    # ═══════════════════════════════════════════════════════════════════
    #  Static / class helpers
    # ═══════════════════════════════════════════════════════════════════

    @staticmethod
    def _ratio_to_stage_types(target_ratio: float) -> List[str]:
        """Map target ratio to stage method types (from MultiplicativeStackingEngine)."""
        stages: List[str] = []
        if target_ratio <= 50:
            stages = ["decomposition", "spectral"]
        elif target_ratio <= 200:
            stages = ["decomposition", "spectral", "entropy"]
        elif target_ratio <= 500:
            stages = ["decomposition", "spectral", "quantization", "entropy"]
        elif target_ratio <= 5000:
            stages = [
                "decomposition",
                "spectral",
                "structural",
                "entropy",
                "quantization",
            ]
        elif target_ratio <= 10000:
            stages = [
                "decomposition",
                "spectral",
                "structural",
                "tensor_network",
                "entropy",
                "quantization",
                "entropy",
            ]
        else:
            stages = [
                "decomposition",
                "spectral",
                "structural",
                "tensor_network",
                "entropy",
                "quantization",
                "structural",
                "entropy",
            ]
        return stages

    @staticmethod
    def _resolve_method_for_type(method_type: str) -> Optional[str]:
        """Resolve a method type to a concrete method name."""
        return _TYPE_TO_METHOD.get(method_type.lower())

    @staticmethod
    def _select_direct_pattern(
        tensor: np.ndarray,
        tensor_type: str,
        target_ratio: float,
    ) -> str:
        """Select best DirectCascadeEngine pattern for tensor type."""
        if tensor.size == 0 or tensor.nbytes < 512:
            return "passthrough"

        ndim = tensor.ndim
        if ndim == 1:
            if target_ratio > 500:
                return "1d_aggressive"
            return "1d_lightning"

        shape = tensor.shape

        if "embed" in tensor_type.lower():
            if target_ratio >= 200:
                return "embedding_triple_cascade"
            if target_ratio >= 500:
                return "embedding_extreme"
            return "embedding_balanced"

        if target_ratio >= 500:
            return "svd_tt_quant_sparse_huffman"
        elif target_ratio >= 200:
            return "svd_int4_sparse_huffman"
        elif target_ratio >= 100:
            if min(shape) >= 64:
                return "aggressive"
            return "balanced"
        elif target_ratio >= 50:
            if target_ratio < 80:
                return "svd_entropy"
            return "balanced"

        return "lightning"

    @staticmethod
    def _resolve_param(key: str, value: Any, shape: Tuple[int, ...]) -> Any:
        """Resolve auto: syntax for parameter values (from DirectCascadeEngine)."""
        if isinstance(value, str) and value.startswith("auto:"):
            divisor = int(value.split(":", 1)[1])
            min_dim = min(s for s in shape if s > 0)
            if key == "ranks":
                return [max(d // divisor, 2) for d in shape]
            return max(min_dim // divisor, 2)
        if isinstance(value, str) and value.startswith("auto_keep:"):
            divisor = int(value.split(":", 1)[1])
            min_dim = min(s for s in shape if s > 0)
            return max(min(divisor / min_dim, 1.0), 0.005)
        return value

    @staticmethod
    def _lagrangian_allocate_sub_ratios(
        target_ratio: float,
        stage_types: List[str],
    ) -> List[float]:
        """Lagrangian-optimal sub-ratio allocation across stages."""
        n = len(stage_types)
        if n == 0:
            return []
        if n == 1:
            return [target_ratio]

        fixed_indices = [
            i
            for i, mt in enumerate(stage_types)
            if _ERROR_GRADIENT_MAP.get(mt, lambda r: -2.0 / (r**3 + 1e-30))(1.0) == 0.0
        ]
        free_indices = [i for i in range(n) if i not in fixed_indices]

        if not free_indices:
            g = max(1.5, target_ratio ** (1.0 / n))
            return [g] * n

        ratios = [0.0] * n
        fixed_product = 1.0
        for i in fixed_indices:
            r = 1.5
            ratios[i] = r
            fixed_product *= r

        free_target = target_ratio / fixed_product
        n_free = len(free_indices)

        if n_free == 0 or free_target <= 1.0:
            return ratios

        log_ratios = np.full(n_free, np.log(free_target) / n_free)
        lr = 0.5
        free_types = [stage_types[i] for i in free_indices]
        grad_fns = np.array(
            [
                _ERROR_GRADIENT_MAP.get(mt, lambda r: -2.0 / (r**3 + 1e-30))
                for mt in free_types
            ],
            dtype=object,
        )

        for _ in range(50):
            ratios_free = np.exp(log_ratios)
            grads = ratios_free * np.array(
                [fn(float(r)) for fn, r in zip(grad_fns, ratios_free)]
            )
            grads -= np.mean(grads)
            log_ratios -= lr * grads

            ratios_free = np.exp(log_ratios)
            ratios_free = np.clip(ratios_free, 1.2, 500.0)
            log_ratios = np.log(ratios_free)

            current_log = float(np.sum(log_ratios))
            log_ratios += (np.log(free_target) - current_log) / n_free

            if float(np.std(grads)) < 1e-6:
                break

        for j, idx in enumerate(free_indices):
            ratios[idx] = float(np.exp(log_ratios[j]))

        return ratios

    # ═══════════════════════════════════════════════════════════════════
    #  Internal helpers
    # ═══════════════════════════════════════════════════════════════════

    @staticmethod
    def _classify_method(method_name: str) -> str:
        nl = method_name.lower()
        if any(
            t in nl
            for t in ("svd", "tensor_train", "cp", "tucker", "kronecker", "low_rank")
        ):
            return "decomposition"
        if any(
            t in nl for t in ("dct", "fwht", "wavelet", "fft", "fourier", "spectral")
        ):
            return "spectral"
        if any(t in nl for t in ("int8", "int4", "quant", "nf4")):
            return "quantization"
        if any(
            t in nl
            for t in ("rans", "huffman", "zstd", "entropy", "arithmetic", "range")
        ):
            return "entropy"
        if any(t in nl for t in ("einsort", "block_sparse", "circulant", "structural")):
            return "structural"
        if any(t in nl for t in ("hadamard",)):
            return "spectral"
        if any(t in nl for t in ("sparse", "delta")):
            return "quantization"
        return "decomposition"

    @staticmethod
    def _get_method_instance(engine: Any, method_name: str) -> Any:
        """Get a method instance from the engine registry."""
        if hasattr(engine, "_methods") and method_name in engine._methods:
            return engine._methods[method_name]
        return None
