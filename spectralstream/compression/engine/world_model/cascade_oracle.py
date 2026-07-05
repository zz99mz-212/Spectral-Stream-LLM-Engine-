"""Cascade Oracle — single unified cascade planner.

Replaces DynamicMethodTester.find_optimal_cascade(),
_tensor_type_strategy.py cascades, and
MultiplicativeStackingEngine.build_cascade_config().

Integrates:
- Lagrangian optimization from MultiplicativeStackingEngine
- Pareto frontier from NASCompressionOptimizer
- Knowledge graph from SelfEvolvingIntelligence
- Tokamak cascade ordering from QuantumPlasmaFusionEngine
- Per-tensor-type strategies from _tensor_type_strategy
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .._dataclasses import TensorProfile
from .._tensor_type_strategy import _tensor_type_strategy
from .._unified_intelligence import UnifiedIntelligence
from .._helpers import _classify_by_name

logger = logging.getLogger(__name__)


@dataclass
class CascadeStage:
    """A single stage in a compression cascade."""

    method_name: str
    method_category: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    expected_ratio: float = 1.0
    expected_error: float = 0.0


@dataclass
class CascadePlan:
    """Optimal cascade plan for a tensor type."""

    tensor_type: str = ""
    stages: List[CascadeStage] = field(default_factory=list)
    total_expected_ratio: float = 1.0
    total_expected_error: float = 0.0
    n_stages: int = 0
    source: str = "oracle"

    def add_stage(self, stage: CascadeStage) -> None:
        self.stages.append(stage)
        self.n_stages = len(self.stages)
        self.total_expected_ratio *= stage.expected_ratio
        self.total_expected_error += stage.expected_error


class CascadeOracle:
    """Unified cascade planner — one planner to rule them all.

    Strategy:
    1. Try knowledge graph first (historical best cascade for tensor type)
    2. Try tensor-type strategy (_tensor_type_strategy cascades)
    3. Try Lagrangian optimization (MultiplicativeStackingEngine)
    4. Try tokamak cascade ordering (QuantumPlasmaFusion)
    5. Fall back to simple 3-stage spectral → decomp → quant
    """

    def __init__(
        self,
        engine: Any,
        unified_intelligence: Optional[UnifiedIntelligence] = None,
    ):
        self._engine = engine
        self._unified = unified_intelligence

    def plan(
        self,
        tensor_type: str = "weight",
        tensor: Optional[np.ndarray] = None,
        profile: Optional[TensorProfile] = None,
        target_ratio: float = 5000.0,
        max_error: float = 0.01,
        name: str = "",
    ) -> CascadePlan:
        """Build optimal cascade plan for a tensor type.

        Tries multiple strategies in order of sophistication:
        1. Knowledge graph best pattern
        2. Tensor-type strategy (hand-tuned)
        3. NAS Pareto frontier
        4. Quantum annealing cascade
        5. Multiplicative stacking Lagrangian
        6. Fallback generic cascade
        """
        # 1. Knowledge graph best pattern
        plan = self._from_knowledge_graph(tensor_type, target_ratio)
        if plan is not None and plan.total_expected_ratio >= target_ratio:
            return plan

        # 2. Tensor-type strategy (hand-tuned per type)
        plan = self._from_tensor_type_strategy(tensor_type)
        if plan is not None and plan.total_expected_ratio >= 10.0:
            return plan

        # 3. NAS Pareto frontier (if we have tensor data)
        if tensor is not None or profile is not None:
            plan = self._from_nas_pareto(
                tensor=tensor,
                profile=profile,
                tensor_type=tensor_type,
                target_ratio=target_ratio,
                max_error=max_error,
            )
            if plan is not None and plan.total_expected_ratio >= target_ratio:
                return plan

        # 4. Quantum annealing cascade ordering
        if profile is not None:
            plan = self._from_quantum_annealing(
                profile=profile,
                tensor_type=tensor_type,
                target_ratio=target_ratio,
            )
            if plan is not None and plan.total_expected_ratio >= target_ratio:
                return plan

        # 5. Multiplicative stacking (Lagrangian optimization)
        if tensor is not None:
            plan = self._from_multiplicative_stacking(
                tensor=tensor,
                name=name,
                target_ratio=target_ratio,
                max_error=max_error,
            )
            if plan is not None and plan.total_expected_ratio >= target_ratio:
                return plan

        # 6. Fallback
        return self._fallback_plan(tensor_type, target_ratio)

    def _from_knowledge_graph(
        self,
        tensor_type: str,
        target_ratio: float,
    ) -> Optional[CascadePlan]:
        """Query the knowledge graph for the best cascade pattern."""
        try:
            if self._unified is not None:
                from ..cascade_learner import CascadeLearner

                learner = CascadeLearner()
                best = learner.get_best_pattern(tensor_type)
                if best is not None and hasattr(best, "stages"):
                    plan = CascadePlan(
                        tensor_type=tensor_type,
                        source="knowledge_graph",
                    )
                    for stage_entry in best.stages:
                        mname = (
                            stage_entry[0]
                            if isinstance(stage_entry, (list, tuple))
                            else stage_entry
                        )
                        plan.add_stage(
                            CascadeStage(
                                method_name=mname,
                                method_category=_classify_by_name(mname)
                                if "." not in mname
                                else mname,
                            )
                        )
                    if plan.n_stages > 0:
                        return plan
        except Exception:
            pass
        return None

    def _from_tensor_type_strategy(self, tensor_type: str) -> Optional[CascadePlan]:
        """Build cascade from hand-tuned per-tensor-type strategies."""
        try:
            strategy = _tensor_type_strategy(tensor_type)
            cascade = strategy.get("cascade", [])
            if not cascade:
                return None

            plan = CascadePlan(
                tensor_type=tensor_type,
                source="tensor_type_strategy",
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

    def _from_nas_pareto(
        self,
        tensor: Optional[np.ndarray],
        profile: Optional[TensorProfile],
        tensor_type: str,
        target_ratio: float,
        max_error: float,
    ) -> Optional[CascadePlan]:
        """Use NASCompressionOptimizer Pareto frontier to find optimal cascade."""
        try:
            from ..dynamic_tuning.nas_compression_optimizer import (
                NASCompressionOptimizer,
            )

            nas = NASCompressionOptimizer(self._engine)
            if profile is not None:
                result = nas.recommend(
                    profile,
                    target_ratio=target_ratio,
                    max_error=max_error,
                    max_search_time=1.0,
                )
            elif tensor is not None:
                from .._profiler import CompressionProfiler

                p = CompressionProfiler().profile_tensor(tensor, tensor_type)
                result = nas.recommend(
                    p,
                    target_ratio=target_ratio,
                    max_error=max_error,
                    max_search_time=1.0,
                )
            else:
                return None

            stages_raw = result.get("stages", [])
            if not stages_raw:
                return None

            plan = CascadePlan(
                tensor_type=tensor_type,
                source="nas_pareto",
            )
            for stage in stages_raw:
                mname = (
                    stage[0]
                    if isinstance(stage, (list, tuple))
                    else stage.get("method_name", "")
                )
                if mname:
                    plan.add_stage(CascadeStage(method_name=mname))
            return plan
        except Exception:
            return None

    def _from_quantum_annealing(
        self,
        profile: TensorProfile,
        tensor_type: str,
        target_ratio: float,
    ) -> Optional[CascadePlan]:
        """Use quantum annealing to find optimal category sequence (tokamak order)."""
        try:
            if self._unified is not None:
                qpf = self._unified._quantum_plasma
            else:
                from ..quantum_plasma_fusion import QuantumPlasmaFusionEngine as QPF

                qpf = QPF()
                qpf.fuse_with_engine(self._engine)

            seqs = qpf.suggest_sequences(
                profile, target_ratio=target_ratio, n_sequences=1
            )
            if not seqs:
                return None

            best = seqs[0]
            methods = best.get("methods", [])
            categories = best.get("categories", [])
            if not methods and not categories:
                return None

            plan = CascadePlan(
                tensor_type=tensor_type,
                source="quantum_annealing",
                total_expected_ratio=best.get("expected_ratio", target_ratio),
                total_expected_error=best.get("expected_error", 0.01),
            )
            for i, mname in enumerate(methods):
                cat = categories[i] if i < len(categories) else ""
                plan.add_stage(
                    CascadeStage(
                        method_name=mname,
                        method_category=cat,
                        expected_ratio=10.0,
                    )
                )
            return plan
        except Exception:
            return None

    def _from_multiplicative_stacking(
        self,
        tensor: np.ndarray,
        name: str,
        target_ratio: float,
        max_error: float,
    ) -> Optional[CascadePlan]:
        """Use MultiplicativeStackingEngine Lagrangian optimization."""
        try:
            from ..dynamic_tuning.multiplicative_stacking import (
                MultiplicativeStackingEngine,
            )

            mse = MultiplicativeStackingEngine(self._engine)
            plan_result = mse.plan_stacking(
                tensor,
                tensor_name=name,
                target_ratio=target_ratio,
                max_error=max_error,
                use_dynamic_pattern=True,
            )
            if plan_result is None or plan_result.total_ratio < 1.0:
                return None

            plan = CascadePlan(
                tensor_type=name,
                source="multiplicative_stacking",
                total_expected_ratio=plan_result.total_ratio,
                total_expected_error=plan_result.total_error,
            )
            if hasattr(plan_result, "stages"):
                for stage in plan_result.stages:
                    if hasattr(stage, "method_name"):
                        plan.add_stage(
                            CascadeStage(
                                method_name=stage.method_name,
                                method_category=getattr(stage, "category", ""),
                                params=getattr(stage, "params", {}),
                                expected_ratio=getattr(stage, "ratio", 1.0),
                                expected_error=getattr(stage, "error", 0.0),
                            )
                        )
            return plan
        except Exception:
            return None

    @staticmethod
    def _fallback_plan(tensor_type: str, target_ratio: float) -> CascadePlan:
        """Generic fallback cascade: spectral → decomposition → quantization."""
        plan = CascadePlan(
            tensor_type=tensor_type,
            source="fallback",
        )
        stages = [
            CascadeStage(
                method_name="dct_spectral",
                method_category="spectral",
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
            CascadeStage(
                method_name="block_int8",
                method_category="quantization",
                params={"block_size": 128},
                expected_ratio=4.0,
                expected_error=0.01,
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

        for stage in stages:
            plan.add_stage(stage)
        return plan
