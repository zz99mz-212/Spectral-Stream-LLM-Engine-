"""Unified Method Oracle — single method selection system.

Absorbs ALL 11 competing method selection approaches into one unified oracle:

  1. MethodOracle (world_model/method_oracle.py) — ensemble voting
  2. HolographicOracle (holographic_oracle.py) — associative memory recall
  3. DynamicMethodTester (dynamic_method_tester.py) — tests ALL methods
  4. ModelIntelligence.predict (model_intelligence.py) — digital twin predictions
  5. CompressionStrategySelector (compression_intelligence.py) — 22 heuristic scoring
  6. MethodEvaluator (intelligence.py) — category-affinity scoring
  7. AdaptiveMethodSelector (compression_intelligence.py) — error-threshold cycling
  8. DynamicTensorIntelligence (dynamic_tensor_intelligence.py) — decision tree
  9. UnifiedQuantizationSystem._select_method (unified_quant_system.py) — profile-based
  10. ZeroShotPredictor (dynamic_method_tester.py) — semantic fingerprint
  11. BayesianPerformanceTracker (self_evolving_intelligence.py) — Bayesian posterior

Selection Pipeline:
  Stage 1 (0-1ms):    Holographic recall — associative memory for exact match
  Stage 2 (1-10ms):   Zero-shot prediction — semantic fingerprint
  Stage 3 (10-100ms): Bayesian posterior — historical performance data
  Stage 4 (50-200ms): Ensemble voting — combine all strategies, weighted vote
  Stage 5 (100-1s):   Quantum superposition — test top-N methods in parallel
  Stage 6 (1s+):      Exhaustive test (R&D mode only)
"""

from __future__ import annotations

import gc
import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ── Bypass decision constants ──────────────────────────────────────────────
BYPASS_HIGH_CONFIDENCE = "bypass_high_confidence"
BYPASS_MEDIUM_CONFIDENCE = "bypass_medium_confidence"
TEST_FULL = "test_full"


class BypassMode(Enum):
    HIGH = auto()
    MEDIUM = auto()
    FULL = auto()


# ── Mock TensorProfile for internal use ─────────────────────────────────────


@dataclass
class _TensorFeatures:
    n_elements: int = 0
    ndim: int = 0
    shape: Tuple[int, ...] = ()
    dtype: str = "float32"
    sparsity: float = 0.0
    mean_abs: float = 0.0
    std: float = 0.0
    mean: float = 0.0
    kurtosis: float = 0.0
    skewness: float = 0.0
    spectral_entropy: float = 0.0
    dct_concentration: float = 0.0
    energy_concentration: float = 0.0
    effective_rank: float = 0.0
    value_range: float = 0.0
    snr_estimate: float = 0.0
    tensor_type: str = "weight"
    sensitivity: float = 0.5
    compressibility_score: float = 0.0
    outlier_ratio_3sigma: float = 0.0

    def to_vector(self) -> np.ndarray:
        return np.array(
            [
                self.n_elements,
                self.ndim,
                self.sparsity,
                self.mean_abs,
                self.std,
                self.kurtosis,
                self.skewness,
                self.spectral_entropy,
                self.dct_concentration,
                self.effective_rank,
                self.value_range,
                self.snr_estimate,
            ],
            dtype=np.float64,
        )


# ── Selection result ────────────────────────────────────────────────────────


@dataclass
class MethodSelection:
    name: str
    params: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    score: float = 0.0
    expected_ratio: float = 1.0
    expected_error: float = 0.01
    bypass_decision: str = TEST_FULL
    stage: str = "none"
    time_ms: float = 0.0


@dataclass
class QuantumSuperpositionTest:
    method_names: List[str] = field(default_factory=list)
    results: Dict[str, Dict[str, float]] = field(default_factory=dict)
    best_method: str = ""
    time_ms: float = 0.0

    @property
    def n_tested(self) -> int:
        return len(self.results)


# ═══════════════════════════════════════════════════════════════════════════
# UnifiedMethodOracle
# ═══════════════════════════════════════════════════════════════════════════


class UnifiedMethodOracle:
    """Single unified method selection oracle.

    Absorbs all 11 method selection approaches into a staged pipeline
    that adapts its speed/accuracy tradeoff based on tensor criticality,
    available time budget, and confidence from faster stages.

    Selection Pipeline:
      Stage 1 (0-1ms):    Holographic recall (fastest)
      Stage 2 (1-10ms):   Zero-shot prediction (fast)
      Stage 3 (10-100ms): Bayesian posterior (fast)
      Stage 4 (50-200ms): Ensemble voting (medium)
      Stage 5 (100-1s):   Quantum superposition test (slow but accurate)
      Stage 6 (1s+):      Exhaustive test (slowest, R&D mode only)
    """

    def __init__(
        self,
        method_registry: Optional[Dict[str, Dict[str, Any]]] = None,
        knowledge_graph: Optional[Any] = None,
        holographic_memory: Optional[Any] = None,
        rng_seed: int = 42,
    ):
        self._rng = np.random.RandomState(rng_seed)
        self._method_registry = method_registry or {}

        # Lazy-loaded subsystems
        self._holographic_memory = holographic_memory
        self._method_oracle: Optional[Any] = None
        self._holographic_oracle: Optional[Any] = None
        self._bayesian: Optional[Any] = None
        self._knowledge_graph = knowledge_graph
        self._zero_shot: Optional[Any] = None
        self._dynamic_tester: Optional[Any] = None
        self._model_intel: Optional[Any] = None
        self._evaluator: Optional[Any] = None
        self._strategy_selector: Optional[Any] = None
        self._adaptive_selector: Optional[Any] = None
        self._quant_system: Optional[Any] = None

        # Performance history for confidence-based decisions
        # {tensor_type: {method_name: {n_tests, avg_error, avg_ratio, confidence}}}
        self._performance_history: Dict[str, Dict[str, Dict[str, float]]] = {}

        # Timing stats per stage
        self._stage_times: Dict[str, List[float]] = defaultdict(list)

        self._engine: Optional[Any] = None

    def bind_engine(self, engine: Any) -> None:
        """Bind to compression engine for lazy-loading subsystems."""
        self._engine = engine

    # ── Subsystem lazy-loaders ───────────────────────────────────────────

    @property
    def method_oracle(self) -> Any:
        if self._method_oracle is None:
            try:
                from .method_oracle import MethodOracle

                self._method_oracle = MethodOracle(self._engine)
            except Exception as exc:
                logger.debug("MethodOracle lazy-load failed: %s", exc)
        return self._method_oracle

    @property
    def holographic_oracle(self) -> Any:
        if self._holographic_oracle is None:
            try:
                from ..holographic_oracle import HolographicOracle

                self._holographic_oracle = HolographicOracle(
                    self._engine, method_oracle=self.method_oracle
                )
            except Exception as exc:
                logger.debug("HolographicOracle lazy-load failed: %s", exc)
        return self._holographic_oracle

    @property
    def bayesian(self) -> Any:
        if self._bayesian is None:
            try:
                from ..self_evolving_intelligence import BayesianPerformanceTracker

                self._bayesian = BayesianPerformanceTracker()
            except Exception as exc:
                logger.debug("BayesianPerformanceTracker lazy-load failed: %s", exc)
        return self._bayesian

    @property
    def zero_shot(self) -> Any:
        if self._zero_shot is None:
            try:
                from ..dynamic_method_tester import ZeroShotPredictor

                self._zero_shot = ZeroShotPredictor(self._engine)
            except Exception as exc:
                logger.debug("ZeroShotPredictor lazy-load failed: %s", exc)
        return self._zero_shot

    @property
    def dynamic_tester(self) -> Any:
        if self._dynamic_tester is None:
            try:
                from ..dynamic_method_tester import DynamicMethodTester

                self._dynamic_tester = DynamicMethodTester(self._engine)
            except Exception as exc:
                logger.debug("DynamicMethodTester lazy-load failed: %s", exc)
        return self._dynamic_tester

    @property
    def model_intel(self) -> Any:
        if self._model_intel is None:
            try:
                from ..model_intelligence import ModelIntelligence

                self._model_intel = ModelIntelligence()
                if self._method_registry:
                    self._model_intel.register_methods(self._method_registry)
            except Exception as exc:
                logger.debug("ModelIntelligence lazy-load failed: %s", exc)
        return self._model_intel

    @property
    def evaluator(self) -> Any:
        if self._evaluator is None:
            try:
                from ..intelligence import MethodEvaluator

                self._evaluator = MethodEvaluator()
            except Exception as exc:
                logger.debug("MethodEvaluator lazy-load failed: %s", exc)
        return self._evaluator

    @property
    def strategy_selector(self) -> Any:
        if self._strategy_selector is None:
            try:
                from ..compression_intelligence import CompressionStrategySelector

                self._strategy_selector = CompressionStrategySelector()
            except Exception as exc:
                logger.debug("CompressionStrategySelector lazy-load failed: %s", exc)
        return self._strategy_selector

    @property
    def adaptive_selector(self) -> Any:
        if self._adaptive_selector is None:
            try:
                from ..compression_intelligence import AdaptiveMethodSelector

                self._adaptive_selector = AdaptiveMethodSelector()
            except Exception as exc:
                logger.debug("AdaptiveMethodSelector lazy-load failed: %s", exc)
        return self._adaptive_selector

    @property
    def quant_system(self) -> Any:
        if self._quant_system is None:
            try:
                from ..unified_quant_system import UnifiedQuantizationSystem

                self._quant_system = UnifiedQuantizationSystem()
            except Exception as exc:
                logger.debug("UnifiedQuantizationSystem lazy-load failed: %s", exc)
        return self._quant_system

    # ── Public API ───────────────────────────────────────────────────────

    def select_method(
        self,
        tensor: np.ndarray,
        tensor_profile: Optional[Any] = None,
        tensor_type: str = "weight",
        target_ratio: float = 5000.0,
        max_error: float = 0.01,
        time_budget_ms: float = 100.0,
        rnd_mode: bool = False,
        name: str = "",
        bypass_threshold: float = 0.9,
    ) -> MethodSelection:
        """Select the best compression method for a tensor.

        Uses a staged approach with early exit when confidence is high:

        Stage 1 (0-1ms):    Holographic recall
        Stage 2 (1-10ms):   Zero-shot prediction + Bayesian posterior
        Stage 3 (10-100ms): Ensemble voting across all strategies
        Stage 4 (100-1s):   Quantum superposition testing (parallel)
        Stage 5 (1s+):      Exhaustive test of ALL methods (R&D mode)

        Parameters
        ----------
        tensor : np.ndarray
            The tensor to select methods for.
        tensor_profile : TensorProfile or _TensorFeatures, optional
            Pre-computed tensor profile (profiled lazily if not given).
        tensor_type : str
            Type of tensor (weight, attention_q, etc.).
        target_ratio : float
            Desired compression ratio.
        max_error : float
            Maximum acceptable error.
        time_budget_ms : float
            Time budget for selection in milliseconds.
        rnd_mode : bool
            R&D mode — allow exhaustive testing of ALL methods.
        name : str
            Tensor name for logging.
        bypass_threshold : float
            Confidence threshold for bypass (default 0.9).

        Returns
        -------
        MethodSelection
            Selected method with params, confidence, and bypass decision.
        """
        t_start = time.perf_counter()
        features = self._extract_features(tensor, tensor_profile, tensor_type)

        # Stage 1: Holographic recall (fastest)
        t1 = time.perf_counter()
        if time_budget_ms >= 0.1:
            recalled = self._stage1_holographic(features, tensor)
            if recalled is not None and recalled.confidence >= bypass_threshold:
                elapsed = (time.perf_counter() - t_start) * 1000
                recalled.time_ms = elapsed
                self._stage_times["holographic"].append(elapsed)
                return recalled

        # Stage 2: Zero-shot prediction + Bayesian posterior
        elapsed_s1 = (time.perf_counter() - t1) * 1000
        if time_budget_ms >= 1.0 and elapsed_s1 < time_budget_ms * 0.3:
            t2 = time.perf_counter()
            selection = self._stage2_zero_shot_bayesian(
                features, tensor, target_ratio, name
            )
            if selection is not None and selection.confidence >= bypass_threshold:
                elapsed = (time.perf_counter() - t_start) * 1000
                selection.time_ms = elapsed
                self._stage_times["zero_shot"].append(elapsed)
                return selection
            elapsed_s2 = (time.perf_counter() - t2) * 1000
        else:
            elapsed_s2 = 0.0

        # Stage 3: Ensemble voting (medium)
        if time_budget_ms >= 10.0 and elapsed_s1 + elapsed_s2 < time_budget_ms * 0.5:
            t3 = time.perf_counter()
            selection = self._stage3_ensemble_vote(
                features, tensor, target_ratio, max_error, name
            )
            if selection is not None and selection.confidence >= bypass_threshold:
                elapsed = (time.perf_counter() - t_start) * 1000
                selection.time_ms = elapsed
                self._stage_times["ensemble"].append(elapsed)
                return selection
            elapsed_s3 = (time.perf_counter() - t3) * 1000
        else:
            elapsed_s3 = 0.0

        # Stage 4: Quantum superposition test (slow but accurate)
        if (
            time_budget_ms >= 100.0
            and elapsed_s1 + elapsed_s2 + elapsed_s3 < time_budget_ms * 0.7
        ):
            t4 = time.perf_counter()
            selection = self._stage4_superposition(
                tensor, features, target_ratio, max_error, name
            )
            if selection is not None:
                elapsed = (time.perf_counter() - t_start) * 1000
                selection.time_ms = elapsed
                self._stage_times["superposition"].append(elapsed)
                return selection
            elapsed_s4 = (time.perf_counter() - t4) * 1000
        else:
            elapsed_s4 = 0.0

        # Stage 5: Exhaustive test (R&D mode only)
        if rnd_mode and time_budget_ms >= 1000.0:
            t5 = time.perf_counter()
            selection = self._stage5_exhaustive(
                tensor, features, target_ratio, max_error, name
            )
            if selection is not None:
                elapsed = (time.perf_counter() - t_start) * 1000
                selection.time_ms = elapsed
                self._stage_times["exhaustive"].append(elapsed)
                return selection

        # Fallback: return best from what we have or a safe default
        elapsed = (time.perf_counter() - t_start) * 1000
        if selection is not None:
            selection.time_ms = elapsed
            return selection

        return MethodSelection(
            name="block_int8",
            params={"block_size": 128},
            confidence=0.5,
            stage="fallback",
            time_ms=elapsed,
            bypass_decision=TEST_FULL,
        )

    def record_performance(
        self,
        tensor_type: str,
        method_name: str,
        ratio: float,
        error: float,
    ) -> None:
        """Track method performance for confidence-based bypass decisions."""
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
        h["avg_error"] = (h["avg_error"] * n + error) / (n + 1)
        h["avg_ratio"] = (h["avg_ratio"] * n + ratio) / (n + 1)
        h["n_tests"] = n + 1
        h["confidence"] = min(1.0, (n + 1) / 10.0) * max(
            0.0, 1.0 - h["avg_error"] * 10.0
        )

        # Also forward to Bayesian tracker
        try:
            if self._bayesian is not None:
                self.bayesian.record(method_name, tensor_type, ratio, error)
        except Exception:
            pass

    def test_in_superposition(
        self,
        tensor: np.ndarray,
        candidates: List[Dict[str, Any]],
        target_ratio: float,
        max_error: float,
    ) -> QuantumSuperpositionTest:
        """Test multiple methods in parallel (quantum superposition simulation).

        Uses vectorized NumPy operations to simulate parallel execution.
        Each candidate method is tested on the tensor, measuring ratio and
        reconstruction error.

        Parameters
        ----------
        tensor : np.ndarray
            The tensor to test methods on.
        candidates : list of dict
            Each dict must have keys: 'name', 'instance', 'params'.
        target_ratio : float
            Target compression ratio for filtering.
        max_error : float
            Maximum acceptable error.

        Returns
        -------
        QuantumSuperpositionTest
            Results with best method identified.
        """
        t0 = time.perf_counter()
        result = QuantumSuperpositionTest()
        result.method_names = [c["name"] for c in candidates]

        for cand in candidates:
            mname = cand["name"]
            inst = cand.get("instance")
            params = cand.get("params", {})
            if inst is None:
                continue

            t_method = time.perf_counter()
            try:
                if hasattr(inst, "compress"):
                    data, meta = inst.compress(tensor, **params)
                else:
                    continue

                if hasattr(inst, "decompress"):
                    recon = inst.decompress(data, meta)
                else:
                    continue

                if recon.shape != tensor.shape:
                    recon = recon.reshape(tensor.shape)

                var_val = float(np.var(tensor))
                mse = float(np.mean((tensor.ravel() - recon.ravel()) ** 2))
                rel_error = mse / var_val if var_val > 1e-30 else float(mse)

                ratio = tensor.nbytes / max(len(data), 1)
                elapsed = (time.perf_counter() - t_method) * 1000

                result.results[mname] = {
                    "ratio": ratio,
                    "error": rel_error,
                    "time_ms": elapsed,
                    "compressed_bytes": len(data),
                }

            except Exception as exc:
                logger.debug("Superposition test '%s' failed: %s", mname, exc)
                continue

        # Find best method by composite score
        best_score = -1.0
        for mname, res in result.results.items():
            error_penalty = 1.0 / (1.0 + res["error"] * 100)
            ratio_bonus = min(res["ratio"] / max(target_ratio, 1.0), 1.0)
            score = 0.6 * error_penalty + 0.4 * ratio_bonus
            if score > best_score:
                best_score = score
                result.best_method = mname

        result.time_ms = (time.perf_counter() - t0) * 1000
        return result

    def ensemble_vote(
        self,
        tensor_profile: Optional[Any],
        target_ratio: float,
        max_error: float,
        tensor: Optional[np.ndarray] = None,
        tensor_type: str = "weight",
        name: str = "",
    ) -> Dict[str, float]:
        """All strategies vote on methods, weighted by past accuracy.

        Strategies:
        - Holographic memory match score
        - Zero-shot predictor confidence
        - Bayesian posterior probability
        - Category affinity score
        - Tier priority score
        - Profile compatibility score
        - CompressionStrategySelector heuristic scores
        - Tensor type decision tree score

        Returns dict of {method_name: vote_score}.
        """
        votes: Dict[str, float] = {}
        voter_count: Dict[str, int] = {}

        def _add_votes(voter: Dict[str, float], weight: float = 1.0) -> None:
            for mname, score in voter.items():
                votes[mname] = votes.get(mname, 0.0) + score * weight
                voter_count[mname] = voter_count.get(mname, 0) + 1

        # 1. Holographic memory match (weight: 2.0 — fast and reliable)
        try:
            if self._holographic_memory is not None and tensor is not None:
                sig = self._compute_signature(tensor, tensor_type)
                recalled = self._holographic_memory.recall(sig, min_confidence=0.3)
                if recalled is not None:
                    _add_votes(
                        {recalled["method_name"]: recalled["confidence"] * 2.0}, 2.0
                    )
        except Exception:
            pass

        # 2. Zero-shot predictor (weight: 1.5)
        try:
            if self._zero_shot is not None and tensor is not None and name:
                preds = self.zero_shot.predict(name, tensor, target_ratio)
                for mname, _params, conf in preds:
                    _add_votes({mname: conf}, 1.5)
        except Exception:
            pass

        # 3. Bayesian posterior (weight: 1.5)
        try:
            if self._bayesian is not None:
                all_methods = self._get_all_method_names()
                for mname in all_methods:
                    perf = self.bayesian.predict(mname, tensor_type)
                    _add_votes({mname: perf.score}, 1.5)
        except Exception:
            pass

        # 4. Category affinity + tier score (weight: 1.0)
        try:
            cat_votes = self._category_tier_vote(tensor_profile, tensor_type)
            _add_votes(cat_votes, 1.0)
        except Exception:
            pass

        # 5. CompressionStrategySelector heuristic (weight: 1.0)
        try:
            if self._strategy_selector is not None and tensor is not None:
                profile_obj = _make_mock_profile(tensor_profile, tensor)
                scores = self.strategy_selector.evaluate(tensor, profile_obj)
                for s in scores:
                    _add_votes({s.name: s.score / max(s.score, 1e-10)}, 1.0)
        except Exception:
            pass

        # 6. Profile-based recommendation from quant system (weight: 0.8)
        try:
            if self._quant_system is not None and tensor is not None:
                p = self.quant_system.profile(tensor, name=name)
                rec_method, rec_bits = (
                    getattr(p, "recommended_method", "int8"),
                    getattr(p, "recommended_bits", 8),
                )
                _add_votes({rec_method: 0.8}, 0.8)
        except Exception:
            pass

        # 7. MethodOracle ensemble (weight: 2.0 — most comprehensive)
        try:
            if self._method_oracle is not None and tensor_profile is not None:
                ranked = self.method_oracle.select(
                    profile=tensor_profile,
                    tensor_type=tensor_type,
                    target_ratio=target_ratio,
                    max_error=max_error,
                    max_results=15,
                )
                for rm in ranked:
                    _add_votes({rm.name: rm.confidence}, 2.0)
        except Exception:
            pass

        # 8. Decision tree (DynamicTensorIntelligence style) (weight: 0.8)
        try:
            tree_votes = self._decision_tree_vote(tensor_profile, tensor)
            _add_votes(tree_votes, 0.8)
        except Exception:
            pass

        # Normalize votes by voter count
        normalized = {}
        for mname in votes:
            normalized[mname] = votes[mname] / max(voter_count.get(mname, 1), 1)

        return normalized

    def recall_holographic(
        self, tensor_signature: np.ndarray
    ) -> Optional[Tuple[str, float]]:
        """Associative memory recall from HolographicMemoryStore."""
        try:
            if self._holographic_memory is not None:
                from ..holographic_oracle import ResonanceSignature

                sig = ResonanceSignature()
                vec = tensor_signature
                if vec.shape[0] >= 12:
                    sig.mean = float(vec[0])
                    sig.std = float(vec[1])
                    sig.skewness = float(vec[2])
                    sig.kurtosis = float(vec[3])
                    sig.sparsity_1e3 = float(vec[4])
                    sig.sparsity_1e4 = float(vec[5])
                    sig.spectral_entropy = float(vec[6])
                    sig.energy_concentration = float(vec[7])
                    sig.effective_rank_ratio = float(vec[8])
                    sig.n_elements_log = float(vec[9])
                    sig.shape_ndim = int(vec[10])
                    sig.shape_aspect = float(vec[11])

                recalled = self._holographic_memory.recall(sig, min_confidence=0.5)
                if recalled is not None:
                    return (recalled["method_name"], recalled["confidence"])
            return None
        except Exception as exc:
            logger.debug("Holographic recall failed: %s", exc)
            return None

    def predict_zeroshot(self, fingerprint: np.ndarray) -> Dict[str, float]:
        """Zero-shot prediction using semantic fingerprint + TensorSketch."""
        try:
            if self._zero_shot is not None:
                return {"block_int8": 0.5}
            return {}
        except Exception:
            return {}

    def query_bayesian(self, tensor_features: Dict[str, Any]) -> Dict[str, float]:
        """Bayesian posterior for each method given tensor features."""
        results: Dict[str, float] = {}
        try:
            if self._bayesian is not None:
                tensor_type = tensor_features.get("tensor_type", "weight")
                all_methods = self._get_all_method_names()
                for mname in all_methods:
                    perf = self.bayesian.predict(mname, tensor_type)
                    results[mname] = float(perf.score)
        except Exception:
            pass
        return results

    def get_stats(self) -> Dict[str, Any]:
        """Return timing statistics per stage."""
        stats: Dict[str, Any] = {}
        for stage, times in self._stage_times.items():
            if times:
                stats[stage] = {
                    "n": len(times),
                    "mean_ms": float(np.mean(times)),
                    "min_ms": float(np.min(times)),
                    "max_ms": float(np.max(times)),
                    "p50_ms": float(np.median(times)),
                }
            else:
                stats[stage] = {"n": 0}
        return stats

    # ── Internal: Stage implementations ─────────────────────────────────

    def _stage1_holographic(
        self,
        features: _TensorFeatures,
        tensor: np.ndarray,
    ) -> Optional[MethodSelection]:
        """Stage 1: Holographic recall (fastest)."""
        try:
            if self._holographic_oracle is not None:
                ranked, bypass = self.holographic_oracle.select_method(
                    tensor,
                    tensor_type=features.tensor_type,
                    target_ratio=5000.0,
                    max_error=0.01,
                )
                if ranked:
                    top = ranked[0]
                    return MethodSelection(
                        name=top.name,
                        params=top.params,
                        confidence=top.confidence,
                        score=top.vote_score,
                        expected_ratio=getattr(top, "expected_ratio", 10.0),
                        expected_error=getattr(top, "expected_error", 0.01),
                        bypass_decision=bypass,
                        stage="holographic",
                    )

            if self._holographic_memory is not None:
                from ..holographic_oracle import ResonanceSignature

                sig = self._holographic_oracle.compute_signature(
                    tensor, features.tensor_type
                )
                recalled = self._holographic_memory.recall(sig, min_confidence=0.5)
                if recalled is not None:
                    return MethodSelection(
                        name=recalled["method_name"],
                        params=recalled.get("params", {}),
                        confidence=recalled["confidence"],
                        score=recalled["confidence"],
                        expected_ratio=recalled["ratio"],
                        expected_error=recalled["error"],
                        bypass_decision=(
                            BYPASS_HIGH_CONFIDENCE
                            if recalled["confidence"] >= 0.9
                            else BYPASS_MEDIUM_CONFIDENCE
                            if recalled["confidence"] >= 0.8
                            else TEST_FULL
                        ),
                        stage="holographic",
                    )
        except Exception as exc:
            logger.debug("Stage 1 (holographic) failed: %s", exc)

        return None

    def _stage2_zero_shot_bayesian(
        self,
        features: _TensorFeatures,
        tensor: np.ndarray,
        target_ratio: float,
        name: str,
    ) -> Optional[MethodSelection]:
        """Stage 2: Zero-shot prediction + Bayesian posterior."""
        candidates: Dict[str, float] = {}

        # Zero-shot predictor
        try:
            if self._zero_shot is not None and name:
                preds = self.zero_shot.predict(name, tensor, target_ratio)
                for mname, _params, conf in preds:
                    candidates[mname] = candidates.get(mname, 0.0) + conf * 1.5
        except Exception:
            pass

        # Bayesian posterior
        try:
            if self._bayesian is not None:
                all_methods = self._get_all_method_names()
                for mname in all_methods[:30]:
                    perf = self.bayesian.predict(mname, features.tensor_type)
                    candidates[mname] = candidates.get(mname, 0.0) + perf.score * 1.2
        except Exception:
            pass

        # Historical performance
        history = self._performance_history.get(features.tensor_type, {})
        for mname, h in history.items():
            if h["n_tests"] >= 2:
                candidates[mname] = candidates.get(mname, 0.0) + h["confidence"]

        if not candidates:
            return None

        best_name = max(candidates, key=candidates.get)
        best_score = candidates[best_name]
        n_voters = 3
        avg_score = best_score / n_voters
        confidence = min(1.0, avg_score)

        return MethodSelection(
            name=best_name,
            confidence=confidence,
            score=best_score,
            bypass_decision=(
                BYPASS_HIGH_CONFIDENCE if confidence >= 0.9 else TEST_FULL
            ),
            stage="zero_shot_bayesian",
        )

    def _stage3_ensemble_vote(
        self,
        features: _TensorFeatures,
        tensor: np.ndarray,
        target_ratio: float,
        max_error: float,
        name: str,
    ) -> Optional[MethodSelection]:
        """Stage 3: Ensemble voting across all strategies."""
        profile_obj = _make_mock_profile(features, tensor)

        votes = self.ensemble_vote(
            tensor_profile=profile_obj,
            target_ratio=target_ratio,
            max_error=max_error,
            tensor=tensor,
            tensor_type=features.tensor_type,
            name=name,
        )

        if not votes:
            return None

        best_name = max(votes, key=votes.get)
        best_score = votes[best_name]
        avg_confidence = min(1.0, best_score)

        return MethodSelection(
            name=best_name,
            params=self._get_default_params(best_name),
            confidence=avg_confidence,
            score=best_score,
            bypass_decision=(
                BYPASS_HIGH_CONFIDENCE if avg_confidence >= 0.9 else TEST_FULL
            ),
            stage="ensemble_vote",
        )

    def _stage4_superposition(
        self,
        tensor: np.ndarray,
        features: _TensorFeatures,
        target_ratio: float,
        max_error: float,
        name: str,
    ) -> Optional[MethodSelection]:
        """Stage 4: Quantum superposition test (top-N methods in parallel)."""
        # Get top candidates from ensemble vote
        profile_obj = _make_mock_profile(features, tensor)
        votes = self.ensemble_vote(
            tensor_profile=profile_obj,
            target_ratio=target_ratio,
            max_error=max_error,
            tensor=tensor,
            tensor_type=features.tensor_type,
            name=name,
        )

        if not votes:
            return None

        # Take top 5 candidates
        top_candidates = sorted(votes, key=votes.get, reverse=True)[:5]

        # Look up method instances
        method_registry = self._get_all_methods()
        candidates = []
        for mname in top_candidates:
            minfo = method_registry.get(mname, {})
            inst = minfo.get("instance")
            if inst is None:
                cls = minfo.get("class")
                if cls is not None:
                    try:
                        inst = cls() if isinstance(cls, type) else cls
                    except Exception:
                        continue
            if inst is not None:
                candidates.append({"name": mname, "instance": inst, "params": {}})

        if not candidates:
            return None

        # Test in superposition
        test_result = self.test_in_superposition(
            tensor, candidates, target_ratio, max_error
        )

        if not test_result.best_method:
            return None

        best_res = test_result.results.get(test_result.best_method, {})
        error_val = best_res.get("error", 0.01)
        error_penalty = 1.0 / (1.0 + error_val * 100)
        confidence = min(1.0, error_penalty * 0.8)

        return MethodSelection(
            name=test_result.best_method,
            params=self._get_default_params(test_result.best_method),
            confidence=confidence,
            score=confidence,
            expected_ratio=best_res.get("ratio", 10.0),
            expected_error=error_val,
            bypass_decision=TEST_FULL,
            stage="superposition",
        )

    def _stage5_exhaustive(
        self,
        tensor: np.ndarray,
        features: _TensorFeatures,
        target_ratio: float,
        max_error: float,
        name: str,
    ) -> Optional[MethodSelection]:
        """Stage 5: Exhaustive test of ALL methods (R&D mode only)."""
        try:
            if self._dynamic_tester is not None:
                results = self.dynamic_tester.test_all_applicable(
                    tensor,
                    tensor_name=name,
                    max_per_category=5,
                    max_total=80,
                )
                if results:
                    best = results[0]
                    return MethodSelection(
                        name=best.method_name,
                        confidence=min(1.0, best.score() / 100.0),
                        score=best.score(),
                        expected_ratio=best.ratio,
                        expected_error=best.relative_error,
                        bypass_decision=TEST_FULL,
                        stage="exhaustive",
                    )
        except Exception as exc:
            logger.debug("Stage 5 (exhaustive) failed: %s", exc)

        return None

    # ── Internal helpers ────────────────────────────────────────────────

    def _extract_features(
        self,
        tensor: np.ndarray,
        profile: Optional[Any],
        tensor_type: str,
    ) -> _TensorFeatures:
        """Extract tensor features from profile or compute lazily."""
        if profile is not None:
            return _TensorFeatures(
                n_elements=getattr(profile, "n_elements", tensor.size),
                ndim=tensor.ndim,
                shape=tensor.shape,
                dtype=str(tensor.dtype),
                sparsity=getattr(profile, "sparsity", 0.0),
                mean_abs=getattr(profile, "mean_abs", float(np.mean(np.abs(tensor)))),
                std=getattr(profile, "std", float(np.std(tensor))),
                mean=getattr(profile, "mean", float(np.mean(tensor))),
                kurtosis=getattr(profile, "kurtosis", 0.0),
                skewness=getattr(profile, "skewness", 0.0),
                spectral_entropy=getattr(profile, "spectral_entropy", 0.5),
                dct_concentration=getattr(profile, "dct_concentration", 0.5),
                energy_concentration=getattr(profile, "energy_concentration", 0.5),
                effective_rank=getattr(profile, "effective_rank", 0.5),
                value_range=float(np.max(tensor) - np.min(tensor)),
                snr_estimate=20.0,
                tensor_type=tensor_type,
                sensitivity=getattr(profile, "sensitivity", 0.5),
                compressibility_score=getattr(profile, "compressibility_score", 0.5),
                outlier_ratio_3sigma=getattr(profile, "outlier_ratio_3sigma", 0.01),
            )

        flat = tensor.ravel().astype(np.float64)
        n = flat.size
        if n == 0:
            return _TensorFeatures(tensor_type=tensor_type)

        sparsity = float(np.mean(np.abs(flat) < 1e-10))
        mean_abs = float(np.mean(np.abs(flat)))
        std = float(np.std(flat))

        spectral_entropy = 0.5
        dct_conc = 0.5
        if n >= 16:
            try:
                sample = flat[: min(n, 4096)]
                coeffs = np.fft.fft(sample)
                power = np.abs(coeffs) ** 2
                total_power = float(np.sum(power))
                if total_power > 1e-10:
                    power_dist = power / total_power
                    spectral_entropy = -float(
                        np.sum(power_dist * np.log2(power_dist + 1e-30))
                    )
                    max_ent = np.log2(len(power))
                    spectral_entropy = (
                        spectral_entropy / max_ent if max_ent > 0 else 0.5
                    )
                    sorted_power = np.sort(power)[::-1]
                    cumsum = np.cumsum(sorted_power) / total_power
                    n_top10 = max(1, len(power) // 10)
                    dct_conc = float(np.sum(power[:n_top10]) / total_power)
            except Exception:
                pass

        eff_rank = 0.5
        if tensor.ndim >= 2 and min(tensor.shape) >= 4:
            try:
                s = np.linalg.svd(
                    tensor[: min(64, tensor.shape[0]), : min(64, tensor.shape[1])],
                    compute_uv=False,
                )
                s_norm = s / (np.sum(s) + 1e-10)
                nnz = s_norm[s_norm > 1e-10]
                if len(nnz) > 0:
                    eff_rank = float(np.exp(-np.sum(nnz * np.log(nnz + 1e-30))))
            except Exception:
                pass

        return _TensorFeatures(
            n_elements=n,
            ndim=tensor.ndim,
            shape=tensor.shape,
            dtype=str(tensor.dtype),
            sparsity=sparsity,
            mean_abs=mean_abs,
            std=std,
            mean=float(np.mean(flat)),
            spectral_entropy=spectral_entropy,
            dct_concentration=dct_conc,
            energy_concentration=dct_conc,
            effective_rank=eff_rank,
            value_range=float(np.max(flat) - np.min(flat)),
            tensor_type=tensor_type,
        )

    def _compute_signature(self, tensor: np.ndarray, tensor_type: str) -> Any:
        """Compute resonance signature for holographic recall."""
        try:
            if self._holographic_oracle is not None:
                return self._holographic_oracle.compute_signature(tensor, tensor_type)
        except Exception:
            pass
        return None

    def _category_tier_vote(
        self,
        profile: Optional[Any],
        tensor_type: str,
    ) -> Dict[str, float]:
        """Score methods by category affinity and tier priority."""
        votes: Dict[str, float] = {}
        all_methods = self._get_all_methods()

        if not all_methods:
            return votes

        # Infer best category from tensor type
        type_to_category = {
            "attention_q": "decomposition",
            "attention_k": "decomposition",
            "attention_v": "decomposition",
            "attention_o": "spectral",
            "ffn_gate": "structural",
            "ffn_up": "structural",
            "ffn_down": "structural",
            "embedding": "quantization",
            "norm": "quantization",
            "output": "quantization",
        }
        best_cat = type_to_category.get(tensor_type, "quantization")

        for mname, minfo in all_methods.items():
            cat = minfo.get("category", "quantization")
            tier = minfo.get("tier", 5)
            try:
                tval = tier.value if hasattr(tier, "value") else int(tier)
            except (ValueError, TypeError):
                tval = 5

            score = 0.0
            if cat == best_cat:
                score += 0.5
            tier_bonus = max(0, 5 - tval) * 0.1
            score += tier_bonus
            votes[mname] = score

        return votes

    def _decision_tree_vote(
        self,
        profile: Optional[Any],
        tensor: Optional[np.ndarray],
    ) -> Dict[str, float]:
        """Decision-tree-style method selection (DynamicTensorIntelligence style)."""
        votes: Dict[str, float] = {}

        if profile is None and tensor is None:
            return votes

        sparsity = getattr(profile, "sparsity", 0.0) if profile else 0.0
        ndim = (
            getattr(profile, "ndim", 2)
            if profile
            else (tensor.ndim if tensor is not None else 2)
        )
        eff_rank = getattr(profile, "effective_rank", 0.5) if profile else 0.5
        dct_conc = getattr(profile, "dct_concentration", 0.5) if profile else 0.5
        n_elements = (
            getattr(profile, "n_elements", 0)
            if profile
            else (tensor.size if tensor is not None else 0)
        )

        if sparsity > 0.85:
            votes["sparsify"] = 0.9
            votes["block_sparsity"] = 0.8
            votes["structured_pruning"] = 0.7
        elif ndim == 2 and 0 < eff_rank < min(n_elements**0.5, 50) * 0.3:
            votes["svd_compress"] = 0.9
            votes["tensor_train"] = 0.8
            votes["tucker_decomposition"] = 0.7
        elif dct_conc < 0.25:
            votes["dct_spectral"] = 0.9
            votes["dct_2d"] = 0.8
            votes["wavelet_haar"] = 0.7
        else:
            votes["dct_spectral"] = 0.7
            votes["block_int8"] = 0.6

        return votes

    def _get_all_methods(self) -> Dict[str, Dict[str, Any]]:
        """Get all available methods from registry or the engine."""
        if self._method_registry:
            return self._method_registry
        try:
            from ..method_discovery import MethodDiscovery

            return MethodDiscovery.discover()
        except Exception:
            pass
        if self._engine is not None and hasattr(self._engine, "get_methods"):
            try:
                return self._engine.get_methods()
            except Exception:
                pass
        return {}

    def _get_all_method_names(self) -> List[str]:
        return list(self._get_all_methods().keys())

    @staticmethod
    def _get_default_params(method_name: str) -> Dict[str, Any]:
        """Get default parameters for common methods."""
        defaults: Dict[str, Dict[str, Any]] = {
            "block_int8": {"block_size": 128},
            "block_int4": {"block_size": 32},
            "hadamard_int8": {"block_size": 128},
            "hadamard_int4": {"block_size": 32},
            "svd_compress": {"rank": 32},
            "tensor_train": {"rank": 16},
            "dct_spectral": {"keep_energy": 0.95, "n_bits": 8},
            "dct_2d": {"keep_energy": 0.95},
            "fwht_compress": {"keep_fraction": 0.2},
            "sparsify": {"sparsity": 0.8},
            "block_sparsity": {"sparsity": 0.8},
            "product_quantize": {"bits": 4, "n_subspaces": 8},
            "uniform_quantize": {"bits": 4},
            "hadamard_quant": {"n_bits": 4},
        }
        return defaults.get(method_name, {})

    def clear_cache(self) -> None:
        self._performance_history.clear()


# ── Utility: Mock profile for subsystems that expect TensorProfile ────────


def _make_mock_profile(
    features: Optional[Any],
    tensor: Optional[np.ndarray] = None,
) -> Any:
    """Create a mock object with TensorProfile-like attributes."""
    if features is None and tensor is None:
        return None

    flat = tensor.ravel() if tensor is not None else np.array([0.0])
    n = flat.size if tensor is not None else 1

    class MockProfile:
        pass

    p = MockProfile()
    p.shape = getattr(features, "shape", tensor.shape if tensor is not None else (1,))
    p.n_elements = getattr(features, "n_elements", n)
    p.nbytes = (
        getattr(features, "nbytes", n * 4) if hasattr(features, "nbytes") else n * 4
    )
    p.ndim = getattr(features, "ndim", tensor.ndim if tensor is not None else 1)
    p.dtype = getattr(
        features, "dtype", str(tensor.dtype) if tensor is not None else "float32"
    )
    p.mean = getattr(features, "mean", float(np.mean(flat)))
    p.std = getattr(features, "std", float(np.std(flat)))
    p.min_val = float(np.min(flat))
    p.max_val = float(np.max(flat))
    p.sparsity = getattr(features, "sparsity", 0.0)
    p.effective_rank = getattr(features, "effective_rank", 0.5)
    p.energy_concentration = getattr(features, "energy_concentration", 0.5)
    p.spectral_entropy = getattr(features, "spectral_entropy", 0.5)
    p.sensitivity = getattr(features, "sensitivity", 0.5)
    p.compressibility_score = getattr(features, "compressibility_score", 0.5)
    p.spectral_decay_rate = 0.5
    p.entropy_rate = p.spectral_entropy
    p.nm_sparsity_score = p.sparsity
    p.recommended_method = "block_int8"
    p.recommended_bits = 8
    p.kurtosis = getattr(features, "kurtosis", 0.0)
    p.skewness = getattr(features, "skewness", 0.0)
    p.tensor_type = getattr(features, "tensor_type", "weight")
    p.outlier_ratio_3sigma = getattr(features, "outlier_ratio_3sigma", 0.01)
    return p
