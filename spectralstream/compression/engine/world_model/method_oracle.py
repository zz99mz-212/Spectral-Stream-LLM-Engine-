"""Method Oracle — single unified method selector.

Replaces _select_methods() in _helpers.py, UnifiedIntelligence.select_methods()
in _unified_intelligence.py, and DynamicMethodTester.get_applicable_methods().

Uses Ising model quantum annealing, Bayesian posterior, tensor fingerprinting,
and ensemble voting to produce ranked method recommendations.

Supports confidence-based bypass for zero-shot prediction:
  BYPASS_HIGH_CONFIDENCE  — use top-1 method directly, skip compress/decompress testing
  BYPASS_MEDIUM_CONFIDENCE — test only top-3 methods
  TEST_FULL               — test all candidates (original behavior)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── Bypass decision constants ──────────────────────────────────────────────
BYPASS_HIGH_CONFIDENCE = "bypass_high_confidence"
BYPASS_MEDIUM_CONFIDENCE = "bypass_medium_confidence"
TEST_FULL = "test_full"

from .._dataclasses import TensorProfile
from .._tier_common import get_tier, tier_score, MethodTier
from .._unified_intelligence import UnifiedIntelligence
from ..quantum_plasma_fusion import (
    QuantumPlasmaFusionEngine as QuantumPlasmaFusionEngine_v3,
)
from ..self_evolving_intelligence import (
    BayesianPerformanceTracker,
    CompressionKnowledgeGraph,
)
from ..dynamic_tuning.nas_compression_optimizer import NASCompressionOptimizer
from ..dynamic_tuning.multiplicative_stacking import MultiplicativeStackingEngine
from .._helpers import _method_compatibility_score

logger = logging.getLogger(__name__)


@dataclass
class RankedMethod:
    """A compression method with its oracle-assigned scores."""

    name: str
    instance: Any = None
    params: Dict[str, Any] = field(default_factory=dict)
    category: str = "quantization"
    tier: int = 5
    expected_ratio: float = 1.0
    expected_error: float = 0.01
    confidence: float = 0.0
    vote_score: float = 0.0

    @property
    def score(self) -> float:
        return self.vote_score


class MethodOracle:
    """Unified method selector that replaces 3 competing systems.

    Integrates:
    - Ising model quantum annealing (QuantumPlasmaFusionEngine._anneal)
    - Bayesian posterior (BayesianPerformanceTracker)
    - Tensor fingerprinting (NASCompressionOptimizer)
    - Tier-based scoring (get_tier / tier_score)
    - Profile compatibility (method_compatibility_score)
    - Ensemble voting across all sub-systems

    Supports: category filter, tier filter, tensor type filter, ratio target.
    """

    def __init__(
        self,
        engine: Any,
        unified_intelligence: Optional[UnifiedIntelligence] = None,
    ):
        self._engine = engine
        self._unified = unified_intelligence

        # Lazy-init sub-systems
        self._bayesian: Optional[BayesianPerformanceTracker] = None
        self._knowledge_graph: Optional[CompressionKnowledgeGraph] = None
        self._quantum_plasma: Optional[QuantumPlasmaFusionEngine_v3] = None
        self._nas: Optional[NASCompressionOptimizer] = None

        self._methods_cache: Optional[Dict[str, Any]] = None

        # Performance history for confidence-based bypass
        # {tensor_type: {method_name: {n_tests, avg_error, avg_ratio, confidence}}}
        self._performance_history: Dict[str, Dict[str, Dict[str, float]]] = {}

    @property
    def bayesian(self) -> BayesianPerformanceTracker:
        if self._bayesian is None:
            from ..self_evolving_intelligence import BayesianPerformanceTracker

            self._bayesian = BayesianPerformanceTracker()
        return self._bayesian

    @property
    def knowledge_graph(self) -> CompressionKnowledgeGraph:
        if self._knowledge_graph is None:
            from ..self_evolving_intelligence import CompressionKnowledgeGraph

            self._knowledge_graph = CompressionKnowledgeGraph()
        return self._knowledge_graph

    @property
    def quantum_plasma(self) -> QuantumPlasmaFusionEngine_v3:
        if self._quantum_plasma is None:
            self._quantum_plasma = QuantumPlasmaFusionEngine_v3()
            if self._engine is not None:
                self._quantum_plasma.fuse_with_engine(self._engine)
        return self._quantum_plasma

    @property
    def nas(self) -> NASCompressionOptimizer:
        if self._nas is None:
            self._nas = NASCompressionOptimizer(self._engine)
        return self._nas

    def select(
        self,
        profile: Optional[TensorProfile] = None,
        tensor: Optional[np.ndarray] = None,
        tensor_type: str = "weight",
        target_ratio: float = 5000.0,
        max_error: float = 0.01,
        category_filter: Optional[List[str]] = None,
        tier_filter: Optional[List[int]] = None,
        tensor_type_filter: Optional[str] = None,
        max_results: int = 25,
    ) -> List[RankedMethod]:
        """Select and rank compression methods using ALL available intelligence.

        Parameters
        ----------
        profile : TensorProfile, optional
            Pre-computed tensor profile.
        tensor : np.ndarray, optional
            Raw tensor (profiled lazily if profile not given).
        tensor_type : str
            Type of tensor (weight, attention_q, etc.).
        target_ratio : float
            Desired compression ratio.
        max_error : float
            Maximum acceptable error.
        category_filter : list of str, optional
            Only methods in these categories.
        tier_filter : list of int, optional
            Only methods in these tiers.
        tensor_type_filter : str, optional
            Only methods compatible with this tensor type.
        max_results : int
            Maximum number of ranked results.

        Returns
        -------
        list of RankedMethod
            Ranked by vote_score descending.
        """
        candidates = self._gather_candidates(
            profile=profile,
            tensor=tensor,
            tensor_type=tensor_type,
            target_ratio=target_ratio,
            max_error=max_error,
            category_filter=category_filter,
            tier_filter=tier_filter,
            tensor_type_filter=tensor_type_filter,
        )
        if not candidates:
            logger.debug("MethodOracle: no candidates — falling back to tier scan")
            candidates = self._tier_fallback(
                category_filter=category_filter,
                tier_filter=tier_filter,
            )

        ranked = self._ensemble_vote(
            candidates=candidates,
            profile=profile,
            tensor_type=tensor_type,
            target_ratio=target_ratio,
            max_error=max_error,
        )
        ranked.sort(key=lambda r: -r.vote_score)
        return ranked[:max_results]

    def record_performance(
        self,
        tensor_type: str,
        method_name: str,
        ratio: float,
        error: float,
    ) -> None:
        """Track method performance for confidence-based bypass decisions.

        Updates a running history of how well each method performs on each
        tensor type.  Confidence increases as more tests accumulate and
        decreases when observed error is high.
        """
        if tensor_type not in self._performance_history:
            self._performance_history[tensor_type] = {}
        if method_name not in self._performance_history[tensor_type]:
            self._performance_history[tensor_type][method_name] = {
                "n_tests": 0,
                "avg_error": 0.0,
                "avg_ratio": 0.0,
                "confidence": 0.0,
            }
        h = self._performance_history[tensor_type][method_name]
        n = h["n_tests"]
        # Running average
        h["avg_error"] = (h["avg_error"] * n + error) / (n + 1)
        h["avg_ratio"] = (h["avg_ratio"] * n + ratio) / (n + 1)
        h["n_tests"] = n + 1
        # Confidence: saturates at ~10 tests, penalised by high error
        h["confidence"] = min(1.0, (n + 1) / 10.0) * max(
            0.0, 1.0 - h["avg_error"] * 10.0
        )

    def _compute_bypass_decision(
        self,
        ranked: List[RankedMethod],
        tensor_type: str,
    ) -> str:
        """Determine whether to bypass full method testing.

        Uses a composite of:
        1. Ensemble vote confidence (from _ensemble_vote)
        2. Bayesian posterior confidence (from BayesianPerformanceTracker)
        3. Historical performance confidence (from _performance_history)

        Returns one of BYPASS_HIGH_CONFIDENCE, BYPASS_MEDIUM_CONFIDENCE, TEST_FULL.
        """
        if not ranked:
            return TEST_FULL

        top = ranked[0]

        # 1. Ensemble vote confidence (from _ensemble_vote normalisation)
        ensemble_conf = top.confidence

        # 2. Bayesian posterior confidence
        bayesian_conf = 0.0
        try:
            perf = self.bayesian.predict(top.name, tensor_type)
            bayesian_conf = getattr(perf, "confidence", 0.0)
        except Exception:
            pass

        # 3. Historical performance confidence
        history_conf = 0.0
        history = self._performance_history.get(tensor_type, {}).get(top.name, {})
        n_tests = history.get("n_tests", 0)
        if n_tests >= 3:
            history_conf = history.get("confidence", 0.0)

        # Composite: weight ensemble most heavily, then bayesian, then history
        composite = ensemble_conf * 0.4 + bayesian_conf * 0.3 + history_conf * 0.3

        if composite >= 0.9:
            return BYPASS_HIGH_CONFIDENCE
        elif composite >= 0.6:
            return BYPASS_MEDIUM_CONFIDENCE
        else:
            return TEST_FULL

    def select_with_bypass(
        self,
        profile: Optional[TensorProfile] = None,
        tensor: Optional[np.ndarray] = None,
        tensor_type: str = "weight",
        target_ratio: float = 5000.0,
        max_error: float = 0.01,
        category_filter: Optional[List[str]] = None,
        tier_filter: Optional[List[int]] = None,
        tensor_type_filter: Optional[str] = None,
        max_results: int = 25,
        skip_testing: bool = False,
        min_confidence: float = 0.9,
    ) -> Tuple[List[RankedMethod], str]:
        """Select methods with confidence-based bypass decision.

        Parameters
        ----------
        profile : TensorProfile, optional
            Pre-computed tensor profile.
        tensor : np.ndarray, optional
            Raw tensor (profiled lazily if profile not given).
        tensor_type : str
            Type of tensor (weight, attention_q, etc.).
        target_ratio : float
            Desired compression ratio.
        max_error : float
            Maximum acceptable error.
        category_filter : list of str, optional
            Only methods in these categories.
        tier_filter : list of int, optional
            Only methods in these tiers.
        tensor_type_filter : str, optional
            Only methods compatible with this tensor type.
        max_results : int
            Maximum number of ranked results.
        skip_testing : bool
            If True, force BYPASS_HIGH_CONFIDENCE when candidates exist
            (overrides computed confidence).
        min_confidence : float
            Confidence threshold for BYPASS_HIGH_CONFIDENCE (default 0.9).

        Returns
        -------
        (ranked, bypass_decision)
            ranked : list of RankedMethod
                Methods sorted by vote_score descending.
            bypass_decision : str
                One of BYPASS_HIGH_CONFIDENCE, BYPASS_MEDIUM_CONFIDENCE, TEST_FULL.
        """
        ranked = self.select(
            profile=profile,
            tensor=tensor,
            tensor_type=tensor_type,
            target_ratio=target_ratio,
            max_error=max_error,
            category_filter=category_filter,
            tier_filter=tier_filter,
            tensor_type_filter=tensor_type_filter,
            max_results=max_results,
        )

        if not ranked:
            return ranked, TEST_FULL

        if skip_testing:
            return ranked, BYPASS_HIGH_CONFIDENCE

        bypass = self._compute_bypass_decision(ranked, tensor_type)
        return ranked, bypass

    def select_from_analysis(
        self,
        analysis: Dict[str, Any],
        target_ratio: float = 5000.0,
        max_error: float = 0.01,
        max_results: int = 25,
    ) -> List[RankedMethod]:
        """Select methods from a UnifiedIntelligence-style analysis dict."""
        tensor_type = analysis.get("tensor_type", "weight")
        profile = None
        try:
            profile = self._reconstruct_profile(analysis)
        except Exception:
            pass
        return self.select(
            profile=profile,
            tensor_type=tensor_type,
            target_ratio=target_ratio,
            max_error=max_error,
            max_results=max_results,
        )

    def _gather_candidates(
        self,
        profile: Optional[TensorProfile],
        tensor: Optional[np.ndarray],
        tensor_type: str,
        target_ratio: float,
        max_error: float,
        category_filter: Optional[List[str]],
        tier_filter: Optional[List[int]],
        tensor_type_filter: Optional[str],
    ) -> List[RankedMethod]:
        """Gather candidate methods from all available sources."""
        all_methods = self._get_all_methods()
        if not all_methods:
            return []

        candidates: List[RankedMethod] = []
        seen: set = set()

        for name, minfo in all_methods.items():
            if name in seen:
                continue

            inst = minfo.get("instance")
            if inst is None:
                try:
                    cls = minfo.get("class")
                    if cls is not None:
                        inst = cls() if isinstance(cls, type) else cls
                        minfo["instance"] = inst
                except Exception:
                    continue
            if inst is None:
                continue

            cat = minfo.get("category", "quantization")
            if category_filter is not None and cat not in category_filter:
                continue

            try:
                tier_val = get_tier(name, cat)
                tval = tier_val.value if hasattr(tier_val, "value") else int(tier_val)
            except (ValueError, TypeError):
                tval = 5
            if tier_filter is not None and tval not in tier_filter:
                continue

            if profile is not None:
                compat = _method_compatibility_score(name, profile)
                if compat <= 0.0:
                    continue

            seen.add(name)
            candidates.append(
                RankedMethod(
                    name=name,
                    instance=inst,
                    category=cat,
                    tier=tval,
                )
            )

        return candidates

    def _ensemble_vote(
        self,
        candidates: List[RankedMethod],
        profile: Optional[TensorProfile],
        tensor_type: str,
        target_ratio: float,
        max_error: float,
    ) -> List[RankedMethod]:
        """Run ensemble voting across ALL intelligence sub-systems."""
        if not candidates:
            return candidates

        method_votes: Dict[str, float] = {c.name: 0.0 for c in candidates}
        method_params: Dict[str, Dict] = {}

        def _vote(mname: str, score: float, params: Optional[Dict] = None) -> None:
            if mname in method_votes:
                method_votes[mname] = method_votes.get(mname, 0.0) + score
                if params:
                    method_params[mname] = params

        # 1. Tier-based baseline score (always available)
        for c in candidates:
            ts = tier_score(get_tier(c.name, c.category))
            _vote(c.name, ts * 0.4)

        # 2. Profile compatibility score
        if profile is not None:
            for c in candidates:
                compat = _method_compatibility_score(c.name, profile)
                _vote(c.name, compat * 0.3)

        # 3. Quantum annealing (Ising model)
        if profile is not None:
            try:
                qpf_seqs = self.quantum_plasma.suggest_sequences(
                    profile, target_ratio=target_ratio, n_sequences=3
                )
                for seq in qpf_seqs:
                    energy_weight = 1.0 / max(abs(seq.get("energy", 1.0)), 0.01)
                    for method_name in seq.get("methods", []):
                        _vote(method_name, energy_weight * 0.5)
            except Exception:
                pass

        # 4. Bayesian posterior (historical performance)
        for c in candidates:
            try:
                perf = self.bayesian.predict(c.name, tensor_type)
                _vote(c.name, perf.score * 0.2)
            except Exception:
                pass

        # 5. Knowledge graph (cross-method synergy)
        try:
            best_cat = self.knowledge_graph.get_best_category(tensor_type)
            if best_cat:
                for c in candidates:
                    if c.category == best_cat:
                        _vote(c.name, 0.15)
        except Exception:
            pass

        # 6. NAS synergy-optimized patterns
        if profile is not None:
            try:
                nas_result = self.nas.recommend(
                    profile,
                    target_ratio=target_ratio,
                    max_error=max_error,
                    max_search_time=0.5,
                )
                synergy = nas_result.get("synergy_score", 0.0)
                for stage in nas_result.get("stages", []):
                    mname = (
                        stage[0]
                        if isinstance(stage, (list, tuple))
                        else stage.get("method_name", "")
                    )
                    _vote(mname, float(synergy) * 0.3 + 0.1)
            except Exception:
                pass

        # Apply votes and compute confidence
        for c in candidates:
            c.vote_score = method_votes.get(c.name, 0.0)
            max_vote = max(method_votes.values()) if method_votes else 1e-10
            c.confidence = min(1.0, c.vote_score / max(max_vote, 1e-10))
            if c.name in method_params:
                c.params = method_params[c.name]

        return candidates

    def _tier_fallback(
        self,
        category_filter: Optional[List[str]] = None,
        tier_filter: Optional[List[int]] = None,
    ) -> List[RankedMethod]:
        """Fallback: select methods by tier priority alone."""
        all_methods = self._get_all_methods()
        if not all_methods:
            return []

        candidates: List[RankedMethod] = []
        seen: set = set()
        tier_priority = tier_filter or [1, 2, 3, 4, 5]
        max_per_tier = 15

        for tval in tier_priority:
            count = 0
            for name, minfo in all_methods.items():
                if name in seen:
                    continue
                cat = minfo.get("category", "quantization")
                if category_filter is not None and cat not in category_filter:
                    continue
                try:
                    tier_val = get_tier(name, cat)
                    tv = tier_val.value if hasattr(tier_val, "value") else int(tier_val)
                except Exception:
                    tv = 5
                if tv != tval:
                    continue
                count += 1
                if count > max_per_tier:
                    break
                seen.add(name)
                inst = minfo.get("instance")
                candidates.append(
                    RankedMethod(
                        name=name,
                        instance=inst,
                        category=cat,
                        tier=tv,
                    )
                )

        return candidates

    def _get_all_methods(self) -> Dict[str, Dict[str, Any]]:
        """Get all available methods from the engine."""
        if self._methods_cache is not None:
            return self._methods_cache
        try:
            from ..method_discovery import MethodDiscovery

            self._methods_cache = MethodDiscovery.discover()
            return self._methods_cache
        except Exception:
            pass
        if self._engine is not None and hasattr(self._engine, "get_methods"):
            try:
                self._methods_cache = self._engine.get_methods()
            except Exception:
                pass
        return self._methods_cache or {}

    @staticmethod
    def _reconstruct_profile(analysis: Dict[str, Any]) -> Optional[TensorProfile]:
        """Build TensorProfile from an analysis dict (from UnifiedIntelligence)."""
        try:
            shape = analysis.get("shape", (1,))
            if isinstance(shape, tuple):
                n_elements = int(np.prod(shape))
            else:
                n_elements = analysis.get("n_elements", 1)
                shape = (n_elements,)
            return TensorProfile(
                name=analysis.get("name", ""),
                shape=shape,
                n_elements=n_elements,
                nbytes=analysis.get("nbytes", n_elements * 4),
                tensor_type=analysis.get("tensor_type", "weight"),
                sensitivity=analysis.get("sensitivity", 0.5),
                effective_rank=analysis.get("effective_rank", 0.5),
                spectral_decay_rate=analysis.get("spectral_decay_rate", 0.5),
                entropy_rate=analysis.get("entropy", 4.0),
                noise_floor=min(analysis.get("outlier_ratio_3sigma", 0.01), 0.5),
                mean=analysis.get("mean", 0.0),
                std=analysis.get("std", 1.0),
            )
        except Exception:
            return None
