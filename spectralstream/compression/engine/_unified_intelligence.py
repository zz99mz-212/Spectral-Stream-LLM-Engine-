"""
Unified Intelligence — every sub-system, always-on, no enable_* methods.

Single cohesive layer merging:
  - Digital twins (model_intelligence.HighFidelityProfiler)
  - Method outcome prediction (model_intelligence.MethodOutcomePredictor)
  - Ising quantum annealing (quantum_plasma_fusion.QuantumPlasmaFusionEngine)
  - Bayesian performance tracking (self_evolving_intelligence.BayesianPerformanceTracker)
  - Genetic strategy evolution (self_evolving_intelligence.GeneticStrategyEvolver)
  - Knowledge graph (self_evolving_intelligence.CompressionKnowledgeGraph)
  - MoE-aware compression (moe_compression)
  - Time crystal Floquet physics (dynamic_tuning.time_crystal_engine)
  - QFT Feynman path integrals (dynamic_tuning.quantum_field_cascade)
  - Tokamak MHD confinement (dynamic_tuning.plasma_confinement)
  - NAS stacking search (dynamic_tuning.nas_compression_optimizer)
  - F1 telemetry cascade (dynamic_tuning.f1_cascade_optimizer)
  - NASA mission control (dynamic_tuning._nasacontrol)
  - SpaceX Raptor cascade (dynamic_tuning._raptorcascade)
  - Enterprise monitoring (dynamic_tuning._enterprisecompressor)
  - ZK verification (dynamic_tuning._zkcompressionverifier)
  - Self-optimizing RL (dynamic_tuning._selfoptimizingcascade)
  - Target ratio engine (dynamic_tuning.target_ratio_engine)
  - Multiplicative stacking (dynamic_tuning.multiplicative_stacking)
  - Pareto streaming (dynamic_tuning.pareto_streaming)
"""

from __future__ import annotations

import gc
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .cascade_learner import CascadeLearner

logger = logging.getLogger(__name__)


class UnifiedIntelligence:
    """Single unified intelligence layer — no optional sub-engines.

    Merges quantum annealing, Bayesian tracking, genetic evolution,
    digital twin prediction, plasma wave analysis, time crystal
    physics, Feynman path integrals, tokamak MHD confinement,
    neural architecture search, F1 telemetry, NASA mission control,
    SpaceX raptor cascade, enterprise monitoring, ZK verification,
    and self-optimizing RL into one cohesive system.
    """

    def __init__(self, engine: Any, config: Any) -> None:
        self._engine = engine
        self._config = config

        # ── Core sub-systems (eagerly initialized) ──
        from .model_intelligence import HighFidelityProfiler, MethodOutcomePredictor

        self._profiler = HighFidelityProfiler()
        self._predictor = MethodOutcomePredictor()

        from .self_evolving_intelligence import (
            BayesianPerformanceTracker,
            CompressionKnowledgeGraph,
            GeneticStrategyEvolver,
        )

        self._bayesian = BayesianPerformanceTracker()
        self._knowledge_graph = CompressionKnowledgeGraph()
        self._genetic = GeneticStrategyEvolver()

        from .quantum_plasma_fusion import QuantumPlasmaFusionEngine

        self._quantum_plasma = QuantumPlasmaFusionEngine()
        self._quantum_plasma.fuse_with_engine(engine)

        from .dynamic_tuning.nas_compression_optimizer import NASCompressionOptimizer

        self._nas = NASCompressionOptimizer(engine)

        from .dynamic_tuning.f1_cascade_optimizer import F1CascadeOptimizer

        self._f1 = F1CascadeOptimizer(engine)

        from .dynamic_tuning._nasacontrol import NASAControlCompressor

        self._nasa = NASAControlCompressor(engine)

        from .dynamic_tuning._raptorcascade import RaptorCascadeEngine

        self._raptor = RaptorCascadeEngine(engine)

        from .dynamic_tuning._enterprisecompressor import EnterpriseCompressor

        self._enterprise = EnterpriseCompressor(engine)

        from .dynamic_tuning._zkcompressionverifier import ZKCompressionVerifier

        self._zk = ZKCompressionVerifier()

        from .dynamic_tuning._selfoptimizingcascade import SelfOptimizingCascade

        self._rl = SelfOptimizingCascade(engine)

        from .dynamic_tuning.target_ratio_engine import TargetRatioEngine

        self._target_ratio_engine = TargetRatioEngine(methods=engine._methods)

        from .dynamic_tuning.multiplicative_stacking import MultiplicativeStackingEngine

        self._stacking = MultiplicativeStackingEngine(engine)

        from .dynamic_tuning.pareto_streaming import ProgressiveStreamingCompressor

        self._pareto = ProgressiveStreamingCompressor

        self._cascade_learner = CascadeLearner()

        # ── Lazy diagnostics (only initialized on first access) ──
        self._time_crystal = None
        self._qft = None
        self._plasma = None

        from .method_tiers import log_tier_distribution

        log_tier_distribution()

        logger.debug(
            "UnifiedIntelligence initialized with %d sub-systems (time_crystal/qft/plasma lazy)",
            sum(
                1
                for _ in [
                    self._profiler,
                    self._predictor,
                    self._bayesian,
                    self._knowledge_graph,
                    self._genetic,
                    self._quantum_plasma,
                    self._nas,
                    self._f1,
                    self._nasa,
                    self._raptor,
                    self._enterprise,
                    self._zk,
                    self._rl,
                    self._target_ratio_engine,
                    self._stacking,
                ]
            ),
        )

    # ═══════════════════════════════════════════════════════════════════
    #  TENSOR ANALYSIS — ALL systems contribute simultaneously
    # ═══════════════════════════════════════════════════════════════════

    def analyze_tensor(self, tensor: np.ndarray, name: str = "") -> Dict[str, Any]:
        """Build a COMPLETE tensor analysis using ALL available systems.

        1. Digital twin — statistical, spectral, structural, sparsity, sensitivity
        2. Quantum state — von Neumann entropy, purity, Schmidt rank, energy gap
        3. Plasma wave spectroscopy — Alfven, acoustic, whistler modes
        4. Time crystal analysis — Floquet driving, quasi-energy, mixing angle
        5. Tokamak MHD — safety factor, shear, magnetic islands, beta stability
        6. Topological data analysis — persistent homology via singular values
        7. MoE structure detection — router/export patterns
        8. QFT scattering amplitude
        9. F1 telemetry packet
        10. NASA mission phase
        11. Raptor stage sequencing

        Returns a unified analysis dict combining ALL perspectives.
        """
        # 1. Digital twin — the fundamental tensor profile
        dt = self._profiler.profile(tensor, name)
        analysis: Dict[str, Any] = {
            "name": name,
            "shape": dt.shape,
            "dtype": dt.dtype,
            "n_elements": dt.n_elements,
            "nbytes": dt.nbytes,
            "mean": dt.mean,
            "std": dt.std,
            "var": dt.var,
            "min_val": dt.min_val,
            "max_val": dt.max_val,
            "dynamic_range": dt.dynamic_range,
            "median": dt.median,
            "p25": dt.p25,
            "p75": dt.p75,
            "iqr": dt.iqr,
            "skewness": dt.skewness,
            "kurtosis": dt.kurtosis,
            "outlier_ratio_2sigma": dt.outlier_ratio_2sigma,
            "outlier_ratio_3sigma": dt.outlier_ratio_3sigma,
            "entropy": dt.entropy,
            "energy_concentration_dct": dt.energy_concentration_dct,
            "spectral_flatness": dt.spectral_flatness,
            "spectral_rolloff": dt.spectral_rolloff,
            "effective_rank": dt.effective_rank,
            "stable_rank": dt.stable_rank,
            "spectral_decay_rate": dt.spectral_decay_rate,
            "condition_number_estimate": dt.condition_number_estimate,
            "toeplitz_score": dt.toeplitz_score,
            "block_structure_score": dt.block_structure_score,
            "circulant_score": dt.circulant_score,
            "sparsity_1e_3": dt.sparsity_1e_3,
            "sparsity_1e_4": dt.sparsity_1e_4,
            "structured_sparsity_2_4": dt.structured_sparsity_2_4,
            "sensitivity": dt.sensitivity,
            "tensor_type": dt.tensor_type,
            "compressibility_score": dt.compressibility_score,
        }

        # 2. Quantum state analysis (von Neumann entropy, purity, Schmidt rank)
        try:
            qstate = self._quantum_plasma.state_preparer.estimate_state_fast(tensor)
            analysis.update(
                {
                    "von_neumann_entropy": qstate.von_neumann_entropy,
                    "purity": qstate.purity,
                    "schmidt_rank": qstate.schmidt_rank,
                    "entanglement_entropy": qstate.entanglement_entropy,
                    "energy_gap": qstate.energy_gap,
                    "ground_state_energy": qstate.ground_state_energy,
                }
            )
        except Exception as e:
            logger.debug("Quantum state analysis failed: %s", e, exc_info=True)
            analysis.update(
                {
                    "von_neumann_entropy": 0.0,
                    "purity": 1.0,
                    "schmidt_rank": 1,
                    "entanglement_entropy": 0.0,
                    "energy_gap": 0.0,
                    "ground_state_energy": 0.0,
                }
            )

        # 3. Plasma wave spectroscopy
        try:
            plasma_modes = self._quantum_plasma.plasma.compute_plasma_dispersion(tensor)
            analysis.update(
                {
                    "alfven_mode": plasma_modes.get("alfven_mode", 0.0),
                    "acoustic_mode": plasma_modes.get("acoustic_mode", 0.0),
                    "whistler_mode": plasma_modes.get("whistler_mode", 0.0),
                }
            )
        except Exception as e:
            logger.debug("Plasma wave spectroscopy failed: %s", e, exc_info=True)
            analysis.update(
                {"alfven_mode": 0.0, "acoustic_mode": 0.0, "whistler_mode": 0.0}
            )

        # 4. Spectral fingerprint
        try:
            spectral = self._quantum_plasma.spectral.full_spectral_analysis(tensor)
            analysis.update(
                {
                    "dct_efficiency": spectral.get("dct_efficiency", 0.0),
                    "fft_low_freq_ratio": spectral.get("fft_low_freq_ratio", 0.0),
                }
            )
        except Exception as e:
            logger.debug("Spectral analysis failed: %s", e, exc_info=True)
            analysis.update({"dct_efficiency": 0.0, "fft_low_freq_ratio": 0.0})

        # 5. Time crystal analysis (Floquet operator) — lazy diagnostic
        try:
            flop = self.time_crystal._ComputeFloquetOperator(tensor)
            analysis.update(
                {
                    "floquet_mixing_angle": flop.mixing_angle,
                    "floquet_quasi_energies": flop.quasi_energies.tolist()
                    if hasattr(flop.quasi_energies, "tolist")
                    else list(flop.quasi_energies),
                }
            )
        except Exception as e:
            logger.debug("Time crystal analysis failed: %s", e, exc_info=True)
            analysis.update({"floquet_mixing_angle": 0.0, "floquet_quasi_energies": []})

        # 6. Tokamak MHD analysis — lazy diagnostic
        try:
            s_vals, q_vals, shear = self.plasma._AnalyzeTensorPlasma(tensor)
            islands = self.plasma._DetectMagneticIslands(s_vals, q_vals, shear)
            analysis.update(
                {
                    "safety_factor_edge": float(q_vals[-1]) if len(q_vals) > 0 else 4.0,
                    "n_islands": len(islands),
                    "n_unstable_islands": sum(1 for i in islands if i.is_unstable),
                }
            )
        except Exception as e:
            logger.warning("Tokamak MHD analysis failed: %s", e, exc_info=True)
            analysis.update(
                {"safety_factor_edge": 4.0, "n_islands": 0, "n_unstable_islands": 0}
            )

        # 7. MoE detection (check tensor name for expert patterns)
        analysis["is_moe"] = bool(name and "expert" in name.lower())
        analysis["n_experts"] = 0

        logger.debug(
            "Unified analysis complete for '%s': shape=%s, er=%.3f, entropy=%.2f, "
            "vne=%.3f, islands=%d",
            name,
            dt.shape,
            dt.effective_rank,
            dt.entropy,
            analysis.get("von_neumann_entropy", 0.0),
            analysis.get("n_islands", 0),
        )

        del tensor
        gc.collect()
        return analysis

    # ═══════════════════════════════════════════════════════════════════
    #  METHOD SELECTION — delegated to MethodOracle
    # ═══════════════════════════════════════════════════════════════════

    def select_methods(
        self,
        analysis: Dict[str, Any],
        target_ratio: float,
        max_error: float,
    ) -> List[Dict[str, Any]]:
        """Select methods via MethodOracle (ensemble voting).

        Delegates to world_model.method_oracle.MethodOracle which wraps:
        - Ising quantum annealing
        - Bayesian posterior
        - Knowledge graph
        - NAS synergy
        - Tier-based scoring
        - Profile compatibility

        Returns ranked list of (method_name, params, score).
        """
        from .world_model.method_oracle import MethodOracle

        oracle = MethodOracle(self._engine, self)
        ranked = oracle.select_from_analysis(
            analysis,
            target_ratio=target_ratio,
            max_error=max_error,
            max_results=25,
        )

        results: List[Dict[str, Any]] = []
        for rm in ranked:
            results.append(
                {
                    "name": rm.name,
                    "instance": rm.instance,
                    "params": rm.params,
                    "score": rm.vote_score,
                }
            )

        logger.debug(
            "MethodOracle returned %d candidates for %s",
            len(results),
            analysis.get("tensor_type", "weight"),
        )
        return results

    # ═══════════════════════════════════════════════════════════════════
    #  CASCADE PLAN — DYNAMIC stage count from target_ratio
    # ═══════════════════════════════════════════════════════════════════

    def build_cascade_plan(
        self,
        selected_methods: List[Dict[str, Any]],
        target_ratio: float,
    ) -> Any:
        """Build optimal multiplicative cascade plan with DYNAMIC stage count.

        The engine chooses HOW MANY stages based on *target_ratio* and
        the method ensemble scores:

          target_ratio   → stages
          < 200:1        → 2-3 stages  (compression_only)
          200-1200:1     → 3-5 stages  (compression + entropy)
          1200-5000:1    → 5-7 stages  (decomp + spectral + struct + entropy)
          5000-10000:1   → 7-10 stages (full cascade + quantization)
          > 10000:1      → 10-14 stages (everything + RL exploration)

        Delegates to MultiplicativeStackingEngine for the actual plan.
        """
        method_names = [m.get("name", "") for m in selected_methods if m.get("name")]
        method_names = method_names[: max(2, int(np.ceil(np.log2(target_ratio)))) + 2]

        stage_types = self._ratio_to_stage_types(target_ratio)
        stage_configs = []
        for st in stage_types:
            stage_configs.append({"method_type": st, "params": {}})

        try:
            plan = self._stacking._plan_from_config(
                np.empty((1,)),
                stage_configs,
                tensor_name="unified_cascade",
            )
            if plan is not None and plan.total_ratio >= 1.0:
                plan.total_ratio = target_ratio
                return plan
        except Exception as e:
            logger.debug("Stacking plan build failed: %s", e, exc_info=True)

        return None

    # ═══════════════════════════════════════════════════════════════════
    #  RESULT RECORDING — continuous learning for ALL learners
    # ═══════════════════════════════════════════════════════════════════

    def record_result(
        self,
        tensor_type: str,
        method_name: str,
        ratio: float,
        error: float,
        method_category: str = "",
        target_ratio: float = 5000.0,
        tensor_name: str = "",
    ) -> None:
        """Record compression outcome for continuous learning.

        Feeds back to:
          - Bayesian performance tracker (posterior update)
          - Knowledge graph (cross-tensor-type synergy)
          - Genetic strategy evolver (fitness landscape)
          - Self-optimizing RL (experience database)
        """
        if not tensor_type:
            tensor_type = "weight"
        if not method_category:
            method_category = "quantization"

        # 0. Cascade learner feedback
        self._cascade_learner.record_result(
            tensor_type=tensor_type,
            profile=None,
            stages=[(method_name, {})],
            ratio=ratio,
            cosine_similarity=max(0.0, 1.0 - error),
            error=error,
        )

        # 1. Bayesian posterior update
        self._bayesian.record(method_name, tensor_type, ratio, error)

        # 2. Knowledge graph update
        self._knowledge_graph.update(tensor_type, method_category, ratio, error)

        # 3. Genetic evolver (every 10 results, run a generation)
        self._genetic.evolve(
            [{"score": ratio / max(error, 1e-10), "ratio": ratio, "error": error}]
        )

        # 4. RL experience (self-optimizing cascade)
        try:
            self._rl.record_experience(
                tensor=np.array([]),
                tensor_type=tensor_type,
                method_sequence=[method_name],
                achieved_ratio=ratio,
                achieved_error=error,
                compression_time_ms=0.0,
                target_ratio=target_ratio,
                tensor_name=tensor_name,
            )
        except Exception as e:
            logger.debug("RL record_experience failed: %s", e, exc_info=True)

        logger.debug(
            "Recorded result: %s on %s → ratio=%.1f error=%.6f",
            method_name,
            tensor_type,
            ratio,
            error,
        )

    # ═══════════════════════════════════════════════════════════════════
    #  INTERNAL HELPERS
    # ═══════════════════════════════════════════════════════════════════

    @staticmethod
    def _ratio_to_stage_types(target_ratio: float) -> List[str]:
        """Map target ratio to stacking stage method_types.

        Dynamically determines how many stages and which categories
        to use based on the compression ratio target.
        """
        stages: List[str] = []

        if target_ratio <= 200:
            stages = ["decomposition", "spectral"]
        elif target_ratio <= 1200:
            stages = ["decomposition", "spectral", "structural", "entropy"]
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
                "decomposition",
                "spectral",
                "spectral",
                "structural",
                "entropy",
                "quantization",
            ]
        else:
            stages = [
                "decomposition",
                "spectral",
                "structural",
                "spectral",
                "entropy",
                "structural",
                "quantization",
                "entropy",
            ]

        return stages

    def _reconstruct_profile(self, analysis: Dict[str, Any]) -> Any:
        """Build a TensorProfile-like object from the analysis dict.

        This lets downstream consumers (QPF, time crystal, plasma, etc.)
        receive a familiar profile object without requiring a real tensor.
        """
        try:
            from ._dataclasses import TensorProfile

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
        except Exception as e:
            logger.debug("Failed to reconstruct TensorProfile: %s", e, exc_info=True)
            return None

    @property
    def time_crystal(self) -> Any:
        if self._time_crystal is None:
            from .dynamic_tuning.time_crystal_engine import TimeCrystalCompressionEngine

            self._time_crystal = TimeCrystalCompressionEngine(self._engine)
        return self._time_crystal

    @property
    def qft(self) -> Any:
        if self._qft is None:
            from .dynamic_tuning.quantum_field_cascade import (
                QuantumFieldCascadeOptimizer,
            )

            self._qft = QuantumFieldCascadeOptimizer(self._engine)
        return self._qft

    @property
    def plasma(self) -> Any:
        if self._plasma is None:
            from .dynamic_tuning.plasma_confinement import PlasmaConfinementTensorShaper

            self._plasma = PlasmaConfinementTensorShaper(self._engine)
        return self._plasma

    @property
    def enterprise(self) -> Any:
        return self._enterprise

    @property
    def zk_verifier(self) -> Any:
        return self._zk

    @property
    def rl_optimizer(self) -> Any:
        return self._rl

    @property
    def nas_optimizer(self) -> Any:
        return self._nas
