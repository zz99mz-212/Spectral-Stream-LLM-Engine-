"""
Unified Compression World Model — the single source of truth.

Replaces ALL competing intelligence systems:
  CompressionIntelligenceEngine, UnifiedIntelligence,
  MethodOracle, CascadeOracle, CompressionRouter, HolographicOracle,
  ModelIntelligence, SelfEvolvingIntelligence, WorldModelCompressor,
  DynamicMethodTester, DirectCascadeEngine, MethodStackingEngine,
  MultiplicativeStackingEngine, ZeroShotPredictor, and every profiling,
  selection, cascade, and certification system.

Architecture
------------
1. TensorWorldModelBuilder — parallel model scan → TensorGraph, sensitivity, redundancy
2. UnifiedOracle — quantum superposition + holographic memory + bayesian tracking
   + genetic evolution + ensemble voting → single decision
3. CascadePlanner — residual stacking with auto-discovery, Lagrangian optimization,
   tokamak ordering, knowledge graph patterns
4. StreamingPipeline — memory-mapped streaming for models up to 365GB
5. R&DBench — all-method testing, dial-in optimization, certification
6. LossMetricsCollector — 20+ metrics per tensor (spectral, statistical, structural)
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import math
import os
import struct
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────

BYPASS_HIGH_CONFIDENCE = "bypass_high_confidence"
BYPASS_MEDIUM_CONFIDENCE = "bypass_medium_confidence"
TEST_FULL = "test_full"

DEFAULT_MAX_WORKERS = 8
DEFAULT_MEMORY_BUDGET_MB = 4096
DEFAULT_TARGET_RATIO = 5000.0
DEFAULT_MAX_ERROR = 0.01


# ──────────────────────────────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────────────────────────────


class CompressionMode(Enum):
    FAST_PATH = auto()
    INTELLIGENT_PATH = auto()
    STREAMING = auto()
    CASCADE = auto()
    MODEL_LEVEL = auto()
    RESEARCH = auto()


class Tier(Enum):
    SVD = 1
    TENSOR_TRAIN = 2
    SPECTRAL = 3
    QUANTIZATION = 4
    ENTROPY = 5


# ──────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────


@dataclass
class ResonanceSignature:
    """Compact fingerprint of a tensor's compression-relevant properties."""

    mean: float = 0.0
    std: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0
    sparsity_1e3: float = 0.0
    sparsity_1e4: float = 0.0
    spectral_entropy: float = 0.0
    energy_concentration: float = 0.0
    effective_rank_ratio: float = 0.0
    n_elements_log: float = 0.0
    shape_ndim: int = 0
    shape_aspect: float = 0.0
    tensor_type: str = "weight"

    _tensor_name: str = ""
    _tensor_shape: Tuple[int, ...] = ()

    @staticmethod
    def n_features() -> int:
        return 12

    def to_vector(self) -> np.ndarray:
        return np.array(
            [
                self.mean,
                self.std,
                self.skewness,
                self.kurtosis,
                self.sparsity_1e3,
                self.sparsity_1e4,
                self.spectral_entropy,
                self.energy_concentration,
                self.effective_rank_ratio,
                self.n_elements_log,
                float(self.shape_ndim),
                self.shape_aspect,
            ],
            dtype=np.float64,
        )

    def to_hash(self) -> str:
        vec = self.to_vector()
        rounded = np.round(vec, decimals=4)
        key = self.tensor_type + "|" + ",".join(f"{v:.4f}" for v in rounded)
        return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


@dataclass
class MemoryEntry:
    signature_hash: str = ""
    signature_vector: np.ndarray = field(default_factory=lambda: np.zeros(12))
    method_name: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    ratio: float = 1.0
    error: float = 0.0
    n_success: int = 1
    timestamp: float = 0.0


@dataclass
class RankedMethod:
    name: str = ""
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


@dataclass
class CascadeStage:
    method_name: str = ""
    method_category: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    expected_ratio: float = 1.0
    expected_error: float = 0.0


@dataclass
class CascadePlan:
    tensor_type: str = ""
    stages: List[CascadeStage] = field(default_factory=list)
    total_expected_ratio: float = 1.0
    total_expected_error: float = 0.0
    n_stages: int = 0
    source: str = "unified_oracle"

    def add_stage(self, stage: CascadeStage) -> None:
        self.stages.append(stage)
        self.n_stages = len(self.stages)
        self.total_expected_ratio *= stage.expected_ratio
        self.total_expected_error += stage.expected_error


@dataclass
class MethodTestResult:
    method_name: str = ""
    category: str = ""
    tier: int = 5
    ratio: float = 1.0
    cosine_similarity: float = 1.0
    snr_db: float = float("inf")
    relative_error: float = 0.0
    compressed_bytes: int = 0
    elapsed: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def score(self) -> float:
        return (
            self.ratio
            * (1.0 - self.relative_error)
            * (0.5 + 0.5 * self.cosine_similarity)
        )


@dataclass
class TensorLossMetrics:
    name: str = ""
    original_shape: Tuple[int, ...] = (0,)
    compression_ratio: float = 1.0
    mse: float = 0.0
    mae: float = 0.0
    max_ae: float = 0.0
    rmse: float = 0.0
    relative_error_l2: float = 0.0
    relative_error_linf: float = 0.0
    snr_db: float = float("inf")
    psnr_db: float = float("inf")
    cosine_similarity: float = 1.0
    kl_divergence: float = 0.0
    wasserstein_distance: float = 0.0
    ks_statistic: float = 0.0
    js_divergence: float = 0.0
    mean_bias: float = 0.0
    std_shift: float = 0.0
    skewness_shift: float = 0.0
    kurtosis_shift: float = 0.0
    outlier_preservation_rate: float = 1.0
    spectral_norm_error: float = 0.0
    effective_rank_error: float = 0.0
    quality_grade: str = "EXCELLENT"
    is_acceptable: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ──────────────────────────────────────────────────────────────────────
# Tensor World Model Data Structures
# ──────────────────────────────────────────────────────────────────────


@dataclass
class TensorNode:
    name: str = ""
    shape: Tuple[int, ...] = ()
    dtype: str = "float32"
    n_elements: int = 0
    nbytes: int = 0
    tensor_type: str = "weight"
    layer_idx: int = -1
    sensitivity: float = 0.5
    effective_rank: float = 0.0
    spectral_decay_rate: float = 0.0
    entropy: float = 0.0
    compressibility_score: float = 0.0
    signature: Optional[ResonanceSignature] = None
    profile: Optional[Dict[str, Any]] = None


@dataclass
class ModelWorldGraph:
    nodes: Dict[str, TensorNode] = field(default_factory=dict)
    by_type: Dict[str, List[str]] = field(default_factory=dict)
    by_layer: Dict[int, List[str]] = field(default_factory=dict)
    total_params: int = 0
    total_bytes: int = 0
    n_tensors: int = 0


@dataclass
class ModelWorldProfile:
    graph: ModelWorldGraph = field(default_factory=ModelWorldGraph)
    layer_count: int = 0
    embedding_size: int = 0
    hidden_size: int = 0
    num_heads: int = 0
    estimated_model_size_gb: float = 0.0
    sensitivity_tiers: Dict[str, int] = field(default_factory=dict)
    redundancy_pairs: List[Tuple[str, str]] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────
# UnifiedCompressionWorldModel
# ──────────────────────────────────────────────────────────────────────


class UnifiedCompressionWorldModel:
    """
    Unified World Model for Compression Intelligence.

    This is the SINGLE source of truth for all compression decisions.
    It replaces 23 competing systems with one cohesive architecture.

    Architecture
    ------------
    1. scan_model()     — parallel model scan → TensorWorldModelBuilder
    2. select_method()   — unified oracle (quantum + holographic + bayesian + genetic)
    3. plan_cascade()    — cascade planner (residual stacking with auto-discovery)
    4. compress()        — main entry (fast, intelligent, streaming, or cascade)
    5. compress_streaming() — memory-mapped streaming for 365GB models
    6. benchmark_mode()  — R&D bench (all-method testing, dial-in, certification)
    7. certify()         — produce compression certificates
    8. compute_loss_metrics() — 20+ metrics per tensor

    Sub-system integration
    ----------------------
    - Quantum:       Ising model annealing via QuantumPlasmaFusionEngine
    - Holographic:   Resonance signature → associative memory recall
    - Bayesian:      Gaussian posterior tracking per (method, tensor_type)
    - Genetic:       Strategy genome evolution via crossover + mutation
    - Knowledge:     CompressionKnowledgeGraph for cross-type transfer
    - Digital Twin:  HighFidelityProfiler + MethodOutcomePredictor
    - Tokamak:       PlasmaConfinementTensorShaper cascade ordering
    - Time Crystal:  Floquet operator for periodic method cycling
    - Feynman:       QuantumFieldCascadeOptimizer scattering amplitudes
    - Cascade:       DirectCascadeEngine residual stacking
    - NAS:           NASCompressionOptimizer Pareto frontier search
    """

    def __init__(
        self,
        config: Optional[Any] = None,
        memory_path: Optional[str] = None,
        knowledge_path: Optional[str] = None,
        max_workers: int = DEFAULT_MAX_WORKERS,
        memory_budget_mb: int = DEFAULT_MEMORY_BUDGET_MB,
    ):
        self._config = config
        self._memory_budget_mb = memory_budget_mb
        self._max_workers = max_workers
        self._knowledge_path = knowledge_path

        # ── Sub-system instances (lazy) ──
        self._engine = None
        self._methods_cache: Optional[Dict[str, Dict[str, Any]]] = None

        # Oracle sub-systems (UnifiedMethodOracle is primary)
        self._unified_oracle = None
        self._method_oracle = None
        self._holographic_oracle = None
        self._cascade_oracle = None
        self._quantum_plasma = None
        self._bayesian_tracker = None
        self._knowledge_graph = None
        self._genetic_evolver = None
        self._nas_optimizer = None
        self._direct_cascade = None
        self._multiplicative_stacking = None
        self._plasma_shaper = None
        self._time_crystal = None
        self._qft_optimizer = None
        self._f1_cascade = None
        self._target_ratio_engine = None

        # Profiling
        self._profiler = None
        self._high_fidelity_profiler = None
        self._method_predictor = None
        self._digital_twins: Dict[str, Any] = {}

        # Memory management
        self._holo_memory = None

        # Statistics
        self._n_compressions = 0
        self._oracle_hits = 0
        self._oracle_misses = 0
        self._compression_history: List[Dict[str, Any]] = []

        # Holographic memory persistence
        self._memory_path = memory_path

    # ═══════════════════════════════════════════════════════════════════
    #  PROPERTIES: Lazy sub-system initialization
    # ═══════════════════════════════════════════════════════════════════

    @property
    def engine(self) -> Any:
        """Lazy-access underlying CompressionIntelligenceEngine."""
        if self._engine is None:
            from .._orchestrator import CompressionIntelligenceEngine
            from .._dataclasses import CompressionConfig

            cfg = (
                self._config
                if isinstance(self._config, CompressionConfig)
                else (
                    self._config if hasattr(self._config, "memory_budget_mb") else None
                )
            )
            if cfg is None:
                cfg = CompressionConfig(memory_budget_mb=self._memory_budget_mb)
            if hasattr(cfg, "memory_budget_mb") and not hasattr(cfg, "max_workers"):
                pass
            self._engine = CompressionIntelligenceEngine(config=cfg)
        return self._engine

    @property
    def unified_oracle(self) -> Any:
        """Lazy-access UnifiedMethodOracle — the single method selector."""
        if self._unified_oracle is None:
            from .unified_method_oracle import UnifiedMethodOracle

            self._unified_oracle = UnifiedMethodOracle(
                method_registry=self.methods_cache,
                holographic_memory=self.holo_memory,
                rng_seed=42,
            )
            self._unified_oracle.bind_engine(self.engine)
        return self._unified_oracle

    @property
    def methods_cache(self) -> Dict[str, Dict[str, Any]]:
        if self._methods_cache is None:
            from ..method_discovery import MethodDiscovery

            self._methods_cache = MethodDiscovery.discover()
        return self._methods_cache

    @property
    def profiler(self) -> Any:
        if self._profiler is None:
            from .._profiler import CompressionProfiler

            self._profiler = CompressionProfiler()
        return self._profiler

    @property
    def high_fidelity_profiler(self) -> Any:
        if self._high_fidelity_profiler is None:
            from ..model_intelligence import HighFidelityProfiler

            self._high_fidelity_profiler = HighFidelityProfiler()
        return self._high_fidelity_profiler

    @property
    def method_predictor(self) -> Any:
        if self._method_predictor is None:
            from ..model_intelligence import MethodOutcomePredictor

            self._method_predictor = MethodOutcomePredictor()
        return self._method_predictor

    @property
    def quantum_plasma(self) -> Any:
        if self._quantum_plasma is None:
            from ..quantum_plasma_fusion import QuantumPlasmaFusionEngine as QPF

            self._quantum_plasma = QPF()
            try:
                self._quantum_plasma.fuse_with_engine(self.engine)
            except Exception:
                pass
        return self._quantum_plasma

    @property
    def bayesian_tracker(self) -> Any:
        if self._bayesian_tracker is None:
            from ..self_evolving_intelligence import BayesianPerformanceTracker

            self._bayesian_tracker = BayesianPerformanceTracker()
        return self._bayesian_tracker

    @property
    def knowledge_graph(self) -> Any:
        if self._knowledge_graph is None:
            from ..self_evolving_intelligence import CompressionKnowledgeGraph

            self._knowledge_graph = CompressionKnowledgeGraph()
        return self._knowledge_graph

    @property
    def genetic_evolver(self) -> Any:
        if self._genetic_evolver is None:
            from ..self_evolving_intelligence import GeneticStrategyEvolver

            self._genetic_evolver = GeneticStrategyEvolver()
        return self._genetic_evolver

    @property
    def nas_optimizer(self) -> Any:
        if self._nas_optimizer is None:
            from ..dynamic_tuning.nas_compression_optimizer import (
                NASCompressionOptimizer,
            )

            self._nas_optimizer = NASCompressionOptimizer(self.engine)
        return self._nas_optimizer

    @property
    def direct_cascade(self) -> Any:
        if self._direct_cascade is None:
            from ..direct_cascade import DirectCascadeEngine

            self._direct_cascade = DirectCascadeEngine(store_all_stages=True)
        return self._direct_cascade

    @property
    def multiplicative_stacking(self) -> Any:
        if self._multiplicative_stacking is None:
            from ..dynamic_tuning.multiplicative_stacking import (
                MultiplicativeStackingEngine,
            )

            self._multiplicative_stacking = MultiplicativeStackingEngine(self.engine)
        return self._multiplicative_stacking

    @property
    def method_oracle(self) -> Any:
        if self._method_oracle is None:
            from .method_oracle import MethodOracle

            self._method_oracle = MethodOracle(self.engine)
        return self._method_oracle

    @property
    def holo_memory(self) -> Any:
        if self._holo_memory is None:
            from ..holographic_oracle import HolographicMemoryStore

            self._holo_memory = HolographicMemoryStore(memory_path=self._memory_path)
        return self._holo_memory

    # ═══════════════════════════════════════════════════════════════════
    #  1. TENSOR WORLD MODEL — Parallel model scan
    # ═══════════════════════════════════════════════════════════════════

    def scan_model(
        self,
        tensors: Dict[str, np.ndarray],
        max_workers: Optional[int] = None,
    ) -> ModelWorldProfile:
        """Scan ALL tensor metadata in parallel → build a complete world model.

        Parameters
        ----------
        tensors : dict of str → np.ndarray
            All tensors from the model.
        max_workers : int, optional
            Number of parallel profiling workers.

        Returns
        -------
        ModelWorldProfile
            Graph, sensitivity tiers, redundancy pairs, model metrics.
        """
        nw = max_workers or self._max_workers
        graph = self._build_graph(tensors)

        # Parallel profiling
        self._parallel_profile(tensors, graph, max_workers=nw)

        # Sensitivity tiers
        sensitivity_tiers = self._compute_sensitivity_tiers(graph)

        # Redundancy detection
        redundancy_pairs = self._find_redundancy(graph)

        # Model metrics
        metrics = self._extract_model_metrics(graph)

        return ModelWorldProfile(
            graph=graph,
            layer_count=metrics.get("layer_count", 0),
            embedding_size=metrics.get("embedding_size", 0),
            hidden_size=metrics.get("hidden_size", 0),
            num_heads=metrics.get("num_heads", 0),
            estimated_model_size_gb=graph.total_bytes / (1024**3),
            sensitivity_tiers=sensitivity_tiers,
            redundancy_pairs=redundancy_pairs,
        )

    def scan_model_from_metadata(
        self,
        tensor_infos: Dict[str, Tuple[tuple, str, int, int]],
    ) -> ModelWorldProfile:
        """Scan from metadata only (names/shapes/dtypes/sizes).

        Useful for memory-mapped loading where tensor data isn't yet in RAM.
        """
        graph = ModelWorldGraph()
        for name, (shape, dtype_str, offset, nbytes) in tensor_infos.items():
            n_elements = int(np.prod(shape)) if shape else 0
            tensor_type = self._classify_by_name(name)
            layer_idx = self._extract_layer_idx(name)
            node = TensorNode(
                name=name,
                shape=shape,
                dtype=dtype_str,
                n_elements=n_elements,
                nbytes=nbytes,
                tensor_type=tensor_type,
                layer_idx=layer_idx,
            )
            graph.nodes[name] = node
            graph.by_type.setdefault(tensor_type, []).append(name)
            graph.by_layer.setdefault(layer_idx, []).append(name)
            graph.total_params += n_elements
            graph.total_bytes += nbytes

        graph.n_tensors = len(graph.nodes)
        sensitivity_tiers = self._compute_sensitivity_tiers(graph)

        return ModelWorldProfile(
            graph=graph,
            estimated_model_size_gb=graph.total_bytes / (1024**3),
            sensitivity_tiers=sensitivity_tiers,
        )

    def _build_graph(self, tensors: Dict[str, np.ndarray]) -> ModelWorldGraph:
        graph = ModelWorldGraph()
        for name, tensor in tensors.items():
            tensor_type = self._classify_by_name(name)
            layer_idx = self._extract_layer_idx(name)
            node = TensorNode(
                name=name,
                shape=tensor.shape,
                dtype=str(tensor.dtype),
                n_elements=tensor.size,
                nbytes=tensor.nbytes,
                tensor_type=tensor_type,
                layer_idx=layer_idx,
            )
            graph.nodes[name] = node
            graph.by_type.setdefault(tensor_type, []).append(name)
            graph.by_layer.setdefault(layer_idx, []).append(name)
            graph.total_params += tensor.size
            graph.total_bytes += tensor.nbytes
        graph.n_tensors = len(graph.nodes)
        return graph

    def _parallel_profile(
        self,
        tensors: Dict[str, np.ndarray],
        graph: ModelWorldGraph,
        max_workers: int = 4,
    ) -> None:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {}
            for name, tensor in tensors.items():
                node = graph.nodes.get(name)
                if node is None:
                    continue
                future = pool.submit(self._profile_single, tensor, name)
                futures[future] = name

            for future in as_completed(futures):
                name = futures[future]
                try:
                    profile, signature = future.result()
                    node = graph.nodes.get(name)
                    if node is not None and profile is not None:
                        node.effective_rank = profile.get("effective_rank", 0.0)
                        node.spectral_decay_rate = profile.get(
                            "spectral_decay_rate", 0.0
                        )
                        node.entropy = profile.get("entropy", 0.0)
                        node.sensitivity = profile.get("sensitivity", 0.5)
                        node.compressibility_score = self._compressibility_from_profile(
                            profile
                        )
                        node.signature = signature
                        node.profile = profile
                except Exception as exc:
                    logger.debug("Parallel profile failed for '%s': %s", name, exc)

    def _profile_single(
        self,
        tensor: np.ndarray,
        name: str,
    ) -> Tuple[Dict[str, Any], Optional[ResonanceSignature]]:
        try:
            base = self.profiler.profile_tensor(tensor, name)
            profile: Dict[str, Any] = {
                "shape": base.shape,
                "dtype": base.dtype,
                "n_elements": base.n_elements,
                "nbytes": base.nbytes,
                "mean": base.mean,
                "std": base.std,
                "effective_rank": getattr(base, "effective_rank", 0.0),
                "spectral_decay_rate": getattr(base, "spectral_decay_rate", 0.0),
                "entropy": getattr(base, "entropy_rate", 0.0),
                "sensitivity": getattr(base, "sensitivity", 0.5),
                "energy_concentration": getattr(base, "energy_concentration", 0.0),
                "noise_floor": getattr(base, "noise_floor", 0.0),
            }
            signature = self._compute_signature(tensor, name)
            return profile, signature
        except Exception:
            return {}, None

    def _compute_signature(
        self, tensor: np.ndarray, name: str = ""
    ) -> ResonanceSignature:
        flat = self._sample_flat(tensor, max_samples=10000)
        n = len(flat)
        mean = float(np.mean(flat))
        std = float(np.std(flat))
        skewness = 0.0
        kurtosis = 0.0
        if std > 1e-30:
            z = (flat - mean) / std
            skewness = float(np.mean(z**3))
            kurtosis = float(np.mean(z**4)) - 3.0
        sparsity_1e3 = float(np.mean(np.abs(flat) < 0.001))
        sparsity_1e4 = float(np.mean(np.abs(flat) < 0.0001))

        spectral_entropy = 0.0
        energy_concentration = 0.0
        if n >= 16:
            try:
                dct_input = flat[: min(1024, n)]
                dct_coeffs = self._lightweight_dct(dct_input)
                dct_energy = dct_coeffs**2
                total_energy = float(np.sum(dct_energy))
                if total_energy > 1e-30:
                    dist = dct_energy / (total_energy + 1e-30)
                    spectral_entropy = -float(np.sum(dist * np.log2(dist + 1e-30)))
                    max_ent = np.log2(len(dct_coeffs))
                    spectral_entropy = (
                        spectral_entropy / max_ent if max_ent > 0 else 0.0
                    )
                    n_top = max(1, len(dct_coeffs) // 10)
                    top_energy = float(np.sum(np.sort(dct_energy)[-n_top:]))
                    energy_concentration = top_energy / (total_energy + 1e-30)
            except Exception:
                pass

        shape_aspect = 0.0
        if tensor.ndim >= 2:
            shape_aspect = max(tensor.shape) / max(min(tensor.shape), 1)

        effective_rank_ratio = 0.0
        if tensor.ndim >= 2 and min(tensor.shape) >= 4:
            try:
                sv_sample = tensor[
                    : min(64, tensor.shape[0]), : min(64, tensor.shape[1])
                ]
                s = np.linalg.svd(sv_sample, compute_uv=False)
                s_sum = float(np.sum(s))
                if s_sum > 1e-30:
                    s_norm = s / s_sum
                    eff_rank = float(np.exp(-np.sum(s_norm * np.log(s_norm + 1e-30))))
                    effective_rank_ratio = eff_rank / min(sv_sample.shape)
            except Exception:
                pass

        tensor_type = self._classify_by_name(name)
        return ResonanceSignature(
            mean=mean,
            std=std,
            skewness=skewness,
            kurtosis=kurtosis,
            sparsity_1e3=sparsity_1e3,
            sparsity_1e4=sparsity_1e4,
            spectral_entropy=spectral_entropy,
            energy_concentration=energy_concentration,
            effective_rank_ratio=effective_rank_ratio,
            n_elements_log=np.log10(max(n_elements(tensor), 1)),
            shape_ndim=tensor.ndim,
            shape_aspect=shape_aspect,
            tensor_type=tensor_type,
            _tensor_name=name,
            _tensor_shape=tensor.shape,
        )

    def _compute_sensitivity_tiers(self, graph: ModelWorldGraph) -> Dict[str, int]:
        tiers: Dict[str, int] = {}
        for name, node in graph.nodes.items():
            nl = name.lower()
            tier = 2
            if any(
                k in nl
                for k in (
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "wq",
                    "wk",
                    "wv",
                    "wo",
                )
            ):
                tier = 1
            elif any(
                k in nl for k in ("gate_proj", "up_proj", "down_proj", "w1", "w2", "w3")
            ):
                tier = 2
            elif any(k in nl for k in ("embed", "wte", "tok_emb")):
                tier = 3
            elif any(k in nl for k in ("norm", "rms", "ln_", "bias")):
                tier = 4
            elif any(k in nl for k in ("head", "lm_head")):
                tier = 3
            tiers[name] = tier
        return tiers

    def _find_redundancy(self, graph: ModelWorldGraph) -> List[Tuple[str, str]]:
        pairs: List[Tuple[str, str]] = []
        names = list(graph.nodes.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                ni = graph.nodes[names[i]]
                nj = graph.nodes[names[j]]
                if ni.shape == nj.shape and ni.tensor_type == nj.tensor_type:
                    pairs.append((names[i], names[j]))
        return pairs

    @staticmethod
    def _extract_model_metrics(graph: ModelWorldGraph) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {
            "layer_count": max(graph.by_layer.keys()) + 1 if graph.by_layer else 0,
            "embedding_size": 0,
            "hidden_size": 0,
            "num_heads": 0,
        }
        for name, node in graph.nodes.items():
            nl = name.lower()
            if "embed" in nl and node.shape and len(node.shape) >= 2:
                metrics["embedding_size"] = max(
                    metrics["embedding_size"], node.shape[0]
                )
            if "q_proj" in nl and node.shape:
                metrics["hidden_size"] = node.shape[0]
        return metrics

    # ═══════════════════════════════════════════════════════════════════
    #  2. UNIFIED ORACLE — method selection
    # ═══════════════════════════════════════════════════════════════════

    def select_method(
        self,
        tensor: np.ndarray,
        tensor_type: str = "weight",
        target_ratio: float = DEFAULT_TARGET_RATIO,
        max_error: float = DEFAULT_MAX_ERROR,
        name: str = "",
        max_results: int = 15,
    ) -> Tuple[List[RankedMethod], str]:
        """Select optimal compression method using ALL oracle sub-systems.

        Integration:
        1. Holographic recall (if confident → BYPASS)
        2. Quantum annealing (Ising model → method sequences)
        3. Bayesian posterior (per (method, tensor_type) performance)
        4. Genetic strategy (evolved genome → tier preferences)
        5. Knowledge graph (cross-type category affinity)
        6. NAS Pareto frontier (synergy-optimized patterns)
        7. Ensemble voting (all scores combined)
        8. Confidence-based bypass decision

        Returns
        -------
        (ranked, bypass_decision)
            ranked : list of RankedMethod
                Methods sorted by vote_score descending.
            bypass_decision : str
                One of BYPASS_HIGH_CONFIDENCE, BYPASS_MEDIUM_CONFIDENCE, TEST_FULL.
        """
        # 1. Holographic recall (fastest path)
        signature = self._compute_signature(tensor, name)
        recalled = self.holo_memory.recall(signature, min_confidence=0.80)
        if recalled is not None:
            conf = recalled["confidence"]
            mname = recalled["method_name"]
            params = recalled.get("params", {})
            inst = self._get_method_instance(mname)
            if inst is not None:
                ranked = [
                    RankedMethod(
                        name=mname,
                        instance=inst,
                        params=params,
                        expected_ratio=recalled["ratio"],
                        expected_error=recalled["error"],
                        confidence=conf,
                        vote_score=conf,
                    )
                ]
                if conf >= 0.90:
                    self._oracle_hits += 1
                    return ranked, BYPASS_HIGH_CONFIDENCE
                elif conf >= 0.80:
                    self._oracle_hits += 1
                    return ranked, BYPASS_MEDIUM_CONFIDENCE

        self._oracle_misses += 1

        # 2. Gather candidates
        candidates = self._gather_candidates(
            tensor_type=tensor_type,
            target_ratio=target_ratio,
            max_error=max_error,
        )

        if not candidates:
            candidates = self._tier_fallback()

        # 3. Ensemble voting
        ranked = self._ensemble_vote(
            candidates=candidates,
            tensor=tensor,
            tensor_type=tensor_type,
            target_ratio=target_ratio,
            max_error=max_error,
        )

        ranked.sort(key=lambda r: -r.vote_score)

        # 4. Bypass decision
        bypass = self._compute_bypass(ranked, tensor_type)

        return ranked[:max_results], bypass

    def _get_method_instance(self, name: str) -> Any:
        info = self.methods_cache.get(name)
        if info is None:
            return None
        inst = info.get("instance")
        if inst is not None:
            return inst
        try:
            cls = info.get("class")
            if cls is not None:
                inst = cls() if isinstance(cls, type) else cls
                info["instance"] = inst
                return inst
        except Exception:
            pass
        return None

    def _gather_candidates(
        self,
        tensor_type: str = "weight",
        target_ratio: float = DEFAULT_TARGET_RATIO,
        max_error: float = DEFAULT_MAX_ERROR,
    ) -> List[RankedMethod]:
        all_methods = self.methods_cache
        if not all_methods:
            return []

        candidates: List[RankedMethod] = []
        seen: Set[str] = set()

        for name, minfo in all_methods.items():
            if name in seen:
                continue
            inst = self._get_method_instance(name)
            if inst is None:
                continue
            cat = minfo.get("category", "quantization")
            try:
                from .._tier_common import get_tier as _gt, MethodTier as _MT

                tier_val = _gt(name, cat)
                tval = tier_val.value if hasattr(tier_val, "value") else int(tier_val)
            except Exception:
                tval = 5
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
        tensor: Optional[np.ndarray] = None,
        tensor_type: str = "weight",
        target_ratio: float = DEFAULT_TARGET_RATIO,
        max_error: float = DEFAULT_MAX_ERROR,
    ) -> List[RankedMethod]:
        if not candidates:
            return candidates

        method_votes: Dict[str, float] = {c.name: 0.0 for c in candidates}
        method_params: Dict[str, Dict] = {}

        def _vote(mname: str, score: float, params: Optional[Dict] = None) -> None:
            if mname in method_votes:
                method_votes[mname] = method_votes.get(mname, 0.0) + score
                if params:
                    method_params[mname] = params

        # 1. Tier baseline
        for c in candidates:
            try:
                from .._tier_common import tier_score as _ts, MethodTier as _MT

                ts = _ts(_MT(c.tier))
            except Exception:
                ts = 0.3
            _vote(c.name, ts * 0.3)

        # 2. Quantum annealing
        if tensor is not None:
            try:
                qpf_seqs = self.quantum_plasma.suggest_sequences(
                    None, target_ratio=target_ratio, n_sequences=3
                )
                for seq in qpf_seqs:
                    energy_weight = 1.0 / max(abs(seq.get("energy", 1.0)), 0.01)
                    for method_name in seq.get("methods", []):
                        _vote(method_name, energy_weight * 0.5)
            except Exception:
                pass

        # 3. Bayesian posterior
        for c in candidates:
            try:
                perf = self.bayesian_tracker.predict(c.name, tensor_type)
                _vote(c.name, perf.score * 0.25)
            except Exception:
                pass

        # 4. Knowledge graph
        try:
            best_cat = self.knowledge_graph.get_best_category(tensor_type)
            if best_cat:
                for c in candidates:
                    if c.category == best_cat:
                        _vote(c.name, 0.2)
        except Exception:
            pass

        # 5. NAS synergy
        if tensor is not None:
            try:
                from .._profiler import CompressionProfiler

                p = CompressionProfiler().profile_tensor(tensor, tensor_type)
                nas_result = self.nas_optimizer.recommend(
                    p,
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

        # Normalize
        for c in candidates:
            c.vote_score = method_votes.get(c.name, 0.0)
            max_vote = max(method_votes.values()) if method_votes else 1e-10
            c.confidence = min(1.0, c.vote_score / max(max_vote, 1e-10))
            if c.name in method_params:
                c.params = method_params[c.name]

        return candidates

    def _compute_bypass(self, ranked: List[RankedMethod], tensor_type: str) -> str:
        if not ranked:
            return TEST_FULL
        top = ranked[0]
        ensemble_conf = top.confidence
        bayesian_conf = 0.0
        try:
            perf = self.bayesian_tracker.predict(top.name, tensor_type)
            bayesian_conf = getattr(perf, "confidence", 0.0)
        except Exception:
            pass
        composite = ensemble_conf * 0.5 + bayesian_conf * 0.3 + 0.2 * top.confidence
        if composite >= 0.9:
            return BYPASS_HIGH_CONFIDENCE
        elif composite >= 0.6:
            return BYPASS_MEDIUM_CONFIDENCE
        return TEST_FULL

    def _tier_fallback(self) -> List[RankedMethod]:
        all_methods = self.methods_cache
        if not all_methods:
            return []
        candidates: List[RankedMethod] = []
        seen: Set[str] = set()
        for tval in range(1, 6):
            cnt = 0
            for name, minfo in all_methods.items():
                if name in seen:
                    continue
                cat = minfo.get("category", "quantization")
                try:
                    from .._tier_common import get_tier as _gt, MethodTier as _MT

                    tv = _gt(name, cat)
                    tv = tv.value if hasattr(tv, "value") else int(tv)
                except Exception:
                    tv = 5
                if tv != tval:
                    continue
                cnt += 1
                if cnt > 15:
                    break
                seen.add(name)
                inst = self._get_method_instance(name)
                candidates.append(
                    RankedMethod(
                        name=name,
                        instance=inst,
                        category=cat,
                        tier=tv,
                    )
                )
        return candidates

    # ═══════════════════════════════════════════════════════════════════
    #  3. CASCADE PLANNER — residual stacking
    # ═══════════════════════════════════════════════════════════════════

    def plan_cascade(
        self,
        tensor: np.ndarray,
        tensor_type: str = "weight",
        target_ratio: float = DEFAULT_TARGET_RATIO,
        max_error: float = DEFAULT_MAX_ERROR,
        name: str = "",
    ) -> CascadePlan:
        """Plan optimal cascade using ALL available strategies.

        1. Knowledge graph best pattern
        2. Tensor-type strategy (hand-tuned per type)
        3. NAS Pareto frontier
        4. Quantum annealing cascade
        5. Multiplicative stacking (Lagrangian optimization)
        6. Direct cascade patterns
        7. Fallback generic cascade
        """
        # 1. Knowledge graph
        plan = self._kg_cascade_plan(tensor_type, target_ratio)
        if plan is not None and plan.total_expected_ratio >= target_ratio:
            return plan

        # 2. Tensor-type strategy
        plan = self._tensor_type_cascade_plan(tensor_type)
        if plan is not None and plan.total_expected_ratio >= 10.0:
            return plan

        # 3. NAS Pareto
        plan = self._nas_cascade_plan(tensor, tensor_type, target_ratio, max_error)
        if plan is not None and plan.total_expected_ratio >= target_ratio:
            return plan

        # 4. Quantum annealing
        plan = self._quantum_cascade_plan(tensor, tensor_type, target_ratio)
        if plan is not None and plan.total_expected_ratio >= target_ratio:
            return plan

        # 5. Multiplicative stacking
        plan = self._stacking_cascade_plan(tensor, name, target_ratio, max_error)
        if plan is not None and plan.total_expected_ratio >= target_ratio:
            return plan

        # 6. Direct cascade patterns
        plan = self._direct_cascade_plan(tensor, tensor_type, target_ratio)
        if plan is not None and plan.n_stages > 0:
            return plan

        # 7. Fallback
        return self._fallback_cascade_plan(tensor_type, target_ratio)

    def _kg_cascade_plan(
        self, tensor_type: str, target_ratio: float
    ) -> Optional[CascadePlan]:
        try:
            from ..cascade_learner import CascadeLearner

            learner = CascadeLearner()
            best = learner.get_best_pattern(tensor_type)
            if best is not None and hasattr(best, "stages"):
                plan = CascadePlan(tensor_type=tensor_type, source="knowledge_graph")
                for stage_entry in best.stages:
                    mname = (
                        stage_entry[0]
                        if isinstance(stage_entry, (list, tuple))
                        else stage_entry
                    )
                    plan.add_stage(CascadeStage(method_name=mname))
                if plan.n_stages > 0:
                    return plan
        except Exception:
            pass
        return None

    def _tensor_type_cascade_plan(self, tensor_type: str) -> Optional[CascadePlan]:
        try:
            from .._tensor_type_strategy import _tensor_type_strategy

            strategy = _tensor_type_strategy(tensor_type)
            cascade = strategy.get("cascade", [])
            if not cascade:
                return None
            plan = CascadePlan(tensor_type=tensor_type, source="tensor_type_strategy")
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

    def _nas_cascade_plan(
        self,
        tensor: np.ndarray,
        tensor_type: str,
        target_ratio: float,
        max_error: float,
    ) -> Optional[CascadePlan]:
        try:
            from .._profiler import CompressionProfiler

            p = CompressionProfiler().profile_tensor(tensor, tensor_type)
            result = self.nas_optimizer.recommend(
                p,
                target_ratio=target_ratio,
                max_error=max_error,
                max_search_time=1.0,
            )
            stages_raw = result.get("stages", [])
            if not stages_raw:
                return None
            plan = CascadePlan(tensor_type=tensor_type, source="nas_pareto")
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

    def _quantum_cascade_plan(
        self,
        tensor: np.ndarray,
        tensor_type: str,
        target_ratio: float,
    ) -> Optional[CascadePlan]:
        try:
            from .._profiler import CompressionProfiler

            p = CompressionProfiler().profile_tensor(tensor, tensor_type)
            seqs = self.quantum_plasma.suggest_sequences(
                p, target_ratio=target_ratio, n_sequences=1
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

    def _stacking_cascade_plan(
        self,
        tensor: np.ndarray,
        name: str,
        target_ratio: float,
        max_error: float,
    ) -> Optional[CascadePlan]:
        try:
            plan_result = self.multiplicative_stacking.plan_stacking(
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

    def _direct_cascade_plan(
        self,
        tensor: np.ndarray,
        tensor_type: str,
        target_ratio: float,
    ) -> Optional[CascadePlan]:
        try:
            pattern_name = self.direct_cascade.select_pattern(
                tensor, tensor_type=tensor_type, target_ratio=target_ratio
            )
            if pattern_name == "passthrough":
                return None
            stages_config = self.direct_cascade.ALL_PATTERNS.get(pattern_name, [])
            if not stages_config:
                return None
            plan = CascadePlan(tensor_type=tensor_type, source="direct_cascade")
            for method_name, params in stages_config:
                resolved = {}
                for k, v in params.items():
                    resolved[k] = self.direct_cascade.resolve_param(k, v, tensor.shape)
                plan.add_stage(
                    CascadeStage(
                        method_name=method_name,
                        params=resolved,
                    )
                )
            return plan
        except Exception:
            return None

    @staticmethod
    def _fallback_cascade_plan(tensor_type: str, target_ratio: float) -> CascadePlan:
        plan = CascadePlan(tensor_type=tensor_type, source="fallback")
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

    # ═══════════════════════════════════════════════════════════════════
    #  4. COMPRESSION — main entry
    # ═══════════════════════════════════════════════════════════════════

    def compress(
        self,
        tensor_or_dict: Any,
        target_ratio: float = DEFAULT_TARGET_RATIO,
        max_error: float = DEFAULT_MAX_ERROR,
        name: str = "",
        mode: Optional[CompressionMode] = None,
        use_world_model: bool = True,
        use_cascade: bool = False,
        use_streaming: bool = False,
        progress_callback: Any = None,
    ) -> Any:
        """Compress a single tensor or dict of tensors.

        Automatically routes to the optimal execution path:
        1. Dict of tensors → model-level compression with world model scan
        2. Large tensor (> memory budget) → streaming pipeline
        3. High ratio (> 500) → intelligent path (oracle + cascade)
        4. Otherwise → fast path

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
        mode : CompressionMode, optional
            Force a specific mode.
        use_world_model : bool
            Force world model path.
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
            Dict of tensors: {name: (compressed, metadata, ratio, error)}.
        """
        is_dict = isinstance(tensor_or_dict, dict)

        if is_dict:
            return self._compress_model_level(
                tensor_or_dict,
                target_ratio=target_ratio,
                max_error=max_error,
                use_world_model=use_world_model,
                progress_callback=progress_callback,
            )

        tensor = tensor_or_dict

        # Determine mode
        if mode is not None:
            pass
        elif use_streaming:
            mode = CompressionMode.STREAMING
        elif use_cascade or target_ratio > 500:
            mode = CompressionMode.CASCADE
        elif use_world_model:
            mode = CompressionMode.INTELLIGENT_PATH
        elif tensor.nbytes > self._memory_budget_mb * 1024 * 1024:
            mode = CompressionMode.STREAMING
        else:
            mode = CompressionMode.FAST_PATH

        if mode == CompressionMode.STREAMING:
            return self._compress_streaming(tensor, target_ratio, max_error, name)

        if mode == CompressionMode.CASCADE:
            return self._compress_cascade(tensor, target_ratio, max_error, name)

        if mode == CompressionMode.INTELLIGENT_PATH:
            return self._compress_intelligent(tensor, target_ratio, max_error, name)

        return self._compress_fast(tensor, target_ratio, max_error, name)

    def _compress_fast(
        self,
        tensor: np.ndarray,
        target_ratio: float,
        max_error: float,
        name: str,
    ) -> Tuple[bytes, dict, float, float]:
        if hasattr(self.engine, "compress_fast"):
            return self.engine.compress_fast(
                tensor,
                name=name,
                target_ratio=target_ratio,
                max_error=max_error,
            )
        from .._helpers import compress_tensor_with_validation

        profile = self.profiler.profile_tensor(tensor, name)
        methods = []
        for mname, minfo in self.methods_cache.items():
            if mname in ("block_int8", "block_int4", "dct_spectral"):
                inst = self._get_method_instance(mname)
                if inst is not None:
                    methods.append({"instance": inst, "params": {}, "name": mname})
            if len(methods) >= 5:
                break
        error_budget = max_error / max(target_ratio, 1.0)
        return compress_tensor_with_validation(tensor, profile, methods, error_budget)

    def _compress_intelligent(
        self,
        tensor: np.ndarray,
        target_ratio: float,
        max_error: float,
        name: str,
    ) -> Tuple[bytes, dict, float, float]:
        ranked, bypass = self.select_method(
            tensor,
            tensor_type=self._classify_by_name(name),
            target_ratio=target_ratio,
            max_error=max_error,
            name=name,
            max_results=15,
        )

        # Try cascade first if ratio is high
        if target_ratio > 100:
            plan = self.plan_cascade(
                tensor,
                tensor_type=self._classify_by_name(name),
                target_ratio=target_ratio,
                max_error=max_error,
                name=name,
            )
            if plan is not None and plan.n_stages >= 2:
                try:
                    result = self._execute_cascade(
                        tensor, plan, target_ratio, max_error, name
                    )
                    if result is not None:
                        return result
                except Exception:
                    pass

        # Fallback: use ranked methods with validation
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

        from .._helpers import compress_tensor_with_validation

        profile = self.profiler.profile_tensor(tensor, name)
        error_budget = max_error / max(target_ratio, 1.0)
        return compress_tensor_with_validation(
            tensor, profile, method_list, error_budget
        )

    def _compress_cascade(
        self,
        tensor: np.ndarray,
        target_ratio: float,
        max_error: float,
        name: str,
    ) -> Tuple[bytes, dict, float, float]:
        plan = self.plan_cascade(
            tensor,
            tensor_type=self._classify_by_name(name),
            target_ratio=target_ratio,
            max_error=max_error,
            name=name,
        )
        if plan is None or plan.n_stages == 0:
            return self._compress_intelligent(tensor, target_ratio, max_error, name)

        result = self._execute_cascade(tensor, plan, target_ratio, max_error, name)
        if result is not None:
            return result

        return self._compress_intelligent(tensor, target_ratio, max_error, name)

    def _execute_cascade(
        self,
        tensor: np.ndarray,
        plan: CascadePlan,
        target_ratio: float,
        max_error: float,
        name: str,
    ) -> Optional[Tuple[bytes, dict, float, float]]:
        stacked_data: List[bytes] = []
        stacked_meta: List[dict] = []
        current = tensor.copy()
        total_ratio = 1.0
        total_error = 0.0

        for stage in plan.stages:
            inst = (
                self.engine._methods.get(stage.method_name)
                if hasattr(self.engine, "_methods")
                else None
            )
            if inst is None:
                inst = self._get_method_instance(stage.method_name)
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
            return None

        packed = bytearray()
        for sd in stacked_data:
            packed += struct.pack("<I", len(sd))
            packed += sd

        metadata = {
            "cascade": True,
            "n_stages": len(stacked_data),
            "stages": stacked_meta,
            "total_ratio": total_ratio,
            "total_error": min(total_error, 1.0),
            "original_shape": list(tensor.shape),
            "method": "cascade",
            "oracle": True,
            "source": plan.source,
        }

        loss = self.compute_loss_metrics(tensor, current, name, len(packed))
        metadata.update(
            {
                "loss_metrics": loss.to_dict(),
                "quality_grade": loss.quality_grade,
                "snr_db": loss.snr_db,
                "cosine_similarity": loss.cosine_similarity,
            }
        )

        return bytes(packed), metadata, total_ratio, min(total_error, 1.0)

    def _compress_streaming(
        self,
        tensor: np.ndarray,
        target_ratio: float,
        max_error: float,
        name: str,
    ) -> Tuple[bytes, dict, float, float]:
        if hasattr(self.engine, "_chunked_compress"):
            return self.engine._chunked_compress(tensor, target_ratio, max_error, name)
        from ..chunked_compressor import ChunkedCompressor

        compressor = ChunkedCompressor(self.engine)
        return compressor.compress_chunked(name, tensor, target_ratio, max_error)

    def _compress_model_level(
        self,
        tensors: Dict[str, np.ndarray],
        target_ratio: float = DEFAULT_TARGET_RATIO,
        max_error: float = DEFAULT_MAX_ERROR,
        use_world_model: bool = True,
        progress_callback: Any = None,
    ) -> Dict[str, Any]:
        # Phase 1: Build world model (scan ALL tensors first)
        world_profile: Optional[ModelWorldProfile] = None
        if use_world_model:
            try:
                world_profile = self.scan_model(tensors)
                logger.info(
                    "World model: %d tensors, %.2f GB, %d layers",
                    world_profile.graph.n_tensors,
                    world_profile.estimated_model_size_gb,
                    world_profile.layer_count,
                )
            except Exception as exc:
                logger.debug("World model scan failed: %s", exc)

        # Phase 2: Compress each tensor
        results: Dict[str, Any] = {}
        total_orig = 0
        total_comp = 0
        failures = 0
        method_dist: Dict[str, int] = {}

        for idx, (name, tensor) in enumerate(tensors.items()):
            if progress_callback:
                progress_callback(idx + 1, len(tensors), name)

            try:
                data, meta, ratio_val, error_val = self.compress(
                    tensor,
                    target_ratio=target_ratio,
                    max_error=max_error,
                    name=name,
                    use_world_model=use_world_model,
                )
                comp_size = len(data) if isinstance(data, (bytes, bytearray)) else 0
                total_orig += tensor.nbytes
                total_comp += comp_size

                method = meta.get("method", "unknown")
                method_dist[method] = method_dist.get(method, 0) + 1

                results[name] = {
                    "data": data,
                    "metadata": meta,
                    "ratio": ratio_val,
                    "error": error_val,
                    "method": method,
                    "original_bytes": tensor.nbytes,
                    "compressed_bytes": comp_size,
                    "tensor_type": self._classify_by_name(name),
                }
            except Exception as e:
                failures += 1
                logger.error("Compression failed for '%s': %s", name, e)
                results[name] = {
                    "data": b"",
                    "metadata": {"method": "failed", "error": str(e)},
                    "ratio": 1.0,
                    "error": 1.0,
                    "method": "failed",
                    "original_bytes": tensor.nbytes,
                    "compressed_bytes": 0,
                }

            del tensor
            gc.collect()

        overall_ratio = total_orig / max(total_comp, 1)

        # Record statistics
        self._n_compressions += len(tensors)
        self._compression_history.append(
            {
                "n_tensors": len(tensors),
                "overall_ratio": overall_ratio,
                "failures": failures,
                "method_distribution": method_dist,
                "timestamp": time.time(),
            }
        )

        results["_meta"] = {
            "total_tensors": len(tensors),
            "total_original_bytes": total_orig,
            "total_compressed_bytes": total_comp,
            "overall_ratio": overall_ratio,
            "failures": failures,
            "method_distribution": method_dist,
            "world_model": world_profile is not None,
        }

        return results

    # ═══════════════════════════════════════════════════════════════════
    #  5. STREAMING — memory-mapped for 365GB models
    # ═══════════════════════════════════════════════════════════════════

    def compress_streaming(
        self,
        model_path: str,
        output_path: str,
        target_ratio: float = DEFAULT_TARGET_RATIO,
        max_error: float = DEFAULT_MAX_ERROR,
        progress_callback: Any = None,
    ) -> Dict[str, Any]:
        """Compress a model using streaming (memory-mapped) access.

        For models that exceed available RAM (up to 365GB).
        Uses memory-mapped tensor access and progressive release.
        """
        from ..memory_mapped_engine import MemoryMappedTensorEngine
        from ..streaming_pipeline import StreamingCompressionPipeline
        from .._io import _SafetensorsIO

        # Phase 1: Scan metadata (no tensor loading)
        io = _SafetensorsIO(use_mmap=True)
        tensor_info = io.scan(model_path)
        total = len(tensor_info)
        logger.info("Streaming: %d tensors in %s", total, model_path)

        # Phase 2: Build world model from metadata
        world_profile = self.scan_model_from_metadata(tensor_info)
        logger.info(
            "World model: %d tensors, %.2f GB estimated",
            world_profile.graph.n_tensors,
            world_profile.estimated_model_size_gb,
        )

        # Phase 3: Streaming compression
        mmap_engine = MemoryMappedTensorEngine(model_path)
        pipeline = StreamingCompressionPipeline(self.engine, self._config)

        results: Dict[str, Any] = {}
        total_orig = 0
        total_comp = 0
        failures = 0

        for idx, (name, (shape, dtype_str, offset, nbytes)) in enumerate(
            tensor_info.items()
        ):
            if progress_callback:
                progress_callback(idx + 1, total, name)

            try:
                tensor = mmap_engine.read(name)
                tensor_type = self._classify_by_name(name)
                sensitivity = world_profile.sensitivity_tiers.get(name, 2)

                if sensitivity >= 3 and target_ratio > 100:
                    eff_target = target_ratio * 1.5
                    eff_error = max_error / 2.0
                else:
                    eff_target = target_ratio
                    eff_error = max_error

                data, meta, ratio_val, error_val = pipeline.compress(
                    tensor,
                    target_ratio=eff_target,
                    max_error=eff_error,
                    name=name,
                )

                comp_size = len(data) if isinstance(data, (bytes, bytearray)) else 0
                total_orig += tensor.nbytes
                total_comp += comp_size

                results[name] = {
                    "data": data,
                    "metadata": meta,
                    "ratio": ratio_val,
                    "error": error_val,
                    "method": meta.get("method", "unknown"),
                    "original_bytes": tensor.nbytes,
                    "compressed_bytes": comp_size,
                    "tensor_type": tensor_type,
                }
                del tensor
            except Exception as e:
                failures += 1
                logger.error("Streaming failed for '%s': %s", name, e)
                results[name] = {
                    "data": b"",
                    "metadata": {"method": "failed"},
                    "ratio": 1.0,
                    "error": 1.0,
                    "method": "failed",
                    "original_bytes": nbytes,
                    "compressed_bytes": 0,
                }
            gc.collect()

        overall_ratio = total_orig / max(total_comp, 1)

        results["_meta"] = {
            "streaming": True,
            "total_tensors": total,
            "total_original_bytes": total_orig,
            "total_compressed_bytes": total_comp,
            "overall_ratio": overall_ratio,
            "failures": failures,
            "world_model_size_gb": world_profile.estimated_model_size_gb,
        }

        # Write output
        if output_path:
            from .._io import _SSFIOWriter

            writer = _SSFIOWriter()
            writer.write(output_path, results)

        return results

    # ═══════════════════════════════════════════════════════════════════
    #  6. LOSS METRICS — 20+ metrics per tensor
    # ═══════════════════════════════════════════════════════════════════

    def compute_loss_metrics(
        self,
        original: np.ndarray,
        reconstructed: np.ndarray,
        name: str = "",
        compressed_size: int = 0,
    ) -> TensorLossMetrics:
        """Compute 20+ loss metrics for a compressed tensor.

        Metrics include:
        - Core errors: MSE, MAE, MaxAE, RMSE, relative L2/Linf
        - Signal-based: SNR (dB), PSNR (dB), cosine similarity
        - Statistical: KL divergence, Wasserstein, KS statistic, JS divergence
        - Distributional: mean bias, std shift, skewness shift, kurtosis shift
        - Structural: spectral norm error, effective rank error
        """
        orig = original.ravel().astype(np.float64)
        recon = reconstructed.ravel().astype(np.float64)
        n = len(orig)

        diff = orig - recon
        mse = float(np.mean(diff**2))
        mae = float(np.mean(np.abs(diff)))
        max_ae = float(np.max(np.abs(diff)))
        rmse = float(math.sqrt(mse))

        orig_norm = float(np.linalg.norm(orig))
        recon_norm = float(np.linalg.norm(recon))
        relative_error_l2 = mse / (max(orig_norm, 1e-30) ** 2)
        relative_error_linf = max_ae / max(float(np.max(np.abs(orig))), 1e-30)

        var_orig = float(np.var(orig))
        snr_db = (
            10.0 * math.log10(var_orig / max(mse, 1e-30))
            if mse > 1e-30
            else float("inf")
        )
        psnr_db = (
            10.0
            * math.log10(
                float(np.max(orig) - float(np.min(orig))) ** 2 / max(mse, 1e-30)
            )
            if mse > 1e-30
            else float("inf")
        )
        cos_sim = float(np.dot(orig, recon) / max(orig_norm * recon_norm, 1e-30))

        bins = min(256, max(10, n // 100))
        hist_orig, edges = np.histogram(orig, bins=bins, density=True)
        hist_recon, _ = np.histogram(recon, bins=edges, density=True)
        hist_orig = hist_orig + 1e-30
        hist_recon = hist_recon + 1e-30
        kl_div = float(np.sum(hist_orig * np.log(hist_orig / hist_recon)))
        js_div = float(
            0.5
            * np.sum(
                hist_orig * np.log(2 * hist_orig / (hist_orig + hist_recon) + 1e-30)
            )
            + 0.5
            * np.sum(
                hist_recon * np.log(2 * hist_recon / (hist_orig + hist_recon) + 1e-30)
            )
        )

        wasserstein = float(np.mean(np.abs(np.sort(orig) - np.sort(recon))))
        ks_stat = float(np.max(np.abs(np.cumsum(hist_orig) - np.cumsum(hist_recon))))

        mean_bias = float(np.mean(orig) - np.mean(recon))
        std_shift = float(np.std(orig) - np.std(recon))
        skew_orig = float(
            np.mean(((orig - np.mean(orig)) / max(np.std(orig), 1e-30)) ** 3)
        )
        skew_recon = float(
            np.mean(((recon - np.mean(recon)) / max(np.std(recon), 1e-30)) ** 3)
        )
        skewness_shift = skew_orig - skew_recon
        kurt_orig = (
            float(np.mean(((orig - np.mean(orig)) / max(np.std(orig), 1e-30)) ** 4))
            - 3.0
        )
        kurt_recon = (
            float(np.mean(((recon - np.mean(recon)) / max(np.std(recon), 1e-30)) ** 4))
            - 3.0
        )
        kurtosis_shift = kurt_orig - kurt_recon

        orig_3sigma = np.abs(orig - np.mean(orig)) > 3 * max(np.std(orig), 1e-30)
        if np.any(orig_3sigma):
            outlier_pres = float(
                np.mean(np.abs(diff[orig_3sigma]) < np.abs(orig[orig_3sigma]) * 0.1)
            )
        else:
            outlier_pres = 1.0

        spectral_norm_error = 0.0
        effective_rank_error = 0.0
        if original.ndim >= 2 and min(original.shape) >= 4:
            try:
                s_orig = np.linalg.svd(
                    original[
                        : min(64, original.shape[0]), : min(64, original.shape[1])
                    ],
                    compute_uv=False,
                )
                s_recon = np.linalg.svd(
                    reconstructed[
                        : min(64, reconstructed.shape[0]),
                        : min(64, reconstructed.shape[1]),
                    ],
                    compute_uv=False,
                )
                spectral_norm_error = float(
                    np.max(np.abs(s_orig - s_recon)) / max(np.max(s_orig), 1e-30)
                )
                s_orig_n = s_orig / max(np.sum(s_orig), 1e-30)
                s_recon_n = s_recon / max(np.sum(s_recon), 1e-30)
                er_orig = float(np.exp(-np.sum(s_orig_n * np.log(s_orig_n + 1e-30))))
                er_recon = float(np.exp(-np.sum(s_recon_n * np.log(s_recon_n + 1e-30))))
                effective_rank_error = abs(er_orig - er_recon) / max(er_orig, 1e-30)
            except Exception:
                pass

        # Quality grade
        q = "EXCELLENT"
        is_ok = True
        if snr_db < 10 or cos_sim < 0.8 or mse > 0.1:
            q = "UNACCEPTABLE"
            is_ok = False
        elif snr_db < 20 or cos_sim < 0.9:
            q = "POOR"
        elif snr_db < 30 or cos_sim < 0.95:
            q = "FAIR"
            is_ok = snr_db >= 20
        elif snr_db < 40 or cos_sim < 0.99:
            q = "GOOD"

        original_size = original.nbytes
        compression_ratio = original_size / max(compressed_size, 1)

        return TensorLossMetrics(
            name=name,
            original_shape=original.shape,
            compression_ratio=compression_ratio,
            mse=mse,
            mae=mae,
            max_ae=max_ae,
            rmse=rmse,
            relative_error_l2=relative_error_l2,
            relative_error_linf=relative_error_linf,
            snr_db=snr_db,
            psnr_db=psnr_db,
            cosine_similarity=cos_sim,
            kl_divergence=kl_div,
            wasserstein_distance=wasserstein,
            ks_statistic=ks_stat,
            js_divergence=js_div,
            mean_bias=mean_bias,
            std_shift=std_shift,
            skewness_shift=skewness_shift,
            kurtosis_shift=kurtosis_shift,
            outlier_preservation_rate=outlier_pres,
            spectral_norm_error=spectral_norm_error,
            effective_rank_error=effective_rank_error,
            quality_grade=q,
            is_acceptable=is_ok,
        )

    # ═══════════════════════════════════════════════════════════════════
    #  7. R&D BENCH — all-method testing and dial-in
    # ═══════════════════════════════════════════════════════════════════

    def benchmark_mode(
        self,
        tensors: Dict[str, np.ndarray],
        target_ratio: float = DEFAULT_TARGET_RATIO,
        max_error: float = DEFAULT_MAX_ERROR,
        max_methods_per_tensor: int = 30,
        use_all_methods: bool = False,
    ) -> Dict[str, Any]:
        """R&D benchmark mode: test ALL methods on representative tensors.

        1. Scans the model → world model
        2. For each tensor type, selects representative tensors
        3. Tests ALL applicable methods on each representative
        4. Builds compression profiles for every method×tensor_type
        5. Auto-discovers best cascade ordering
        6. Trains genetic strategy evolver
        7. Produces detailed report

        Parameters
        ----------
        tensors : dict of str → np.ndarray
            Model tensors to benchmark.
        target_ratio : float
            Target compression ratio for tests.
        max_error : float
            Maximum acceptable error.
        max_methods_per_tensor : int
            Max methods to test per tensor.
        use_all_methods : bool
            If True, test ALL discovered methods (slow).

        Returns
        -------
        dict
            Benchmark report with per-type best methods, cascade plans,
            and genetic strategy.
        """
        # Phase 1: Build world model
        world_profile = self.scan_model(tensors)
        logger.info("Benchmark: scanning %d tensors", world_profile.graph.n_tensors)

        # Phase 2: Select representatives per tensor type
        representatives: Dict[str, List[Tuple[str, np.ndarray]]] = {}
        for name in world_profile.graph.nodes:
            ttype = world_profile.graph.nodes[name].tensor_type
            if ttype not in representatives:
                representatives[ttype] = []
            representatives[ttype].append((name, tensors[name]))

        # Phase 3: Test methods on each representative
        report: Dict[str, Any] = {
            "world_model": {
                "n_tensors": world_profile.graph.n_tensors,
                "estimated_size_gb": world_profile.estimated_model_size_gb,
                "layer_count": world_profile.layer_count,
            },
            "per_type_results": {},
            "best_cascade_plans": {},
            "genetic_strategy": {},
            "oracle_stats": {},
        }

        all_test_results: List[Dict] = []
        candidates = self._gather_candidates()
        method_names = [c.name for c in candidates]

        for ttype, reps in representatives.items():
            type_results: List[Dict[str, Any]] = []
            type_method_scores: Dict[str, List[float]] = {}

            for rep_name, rep_tensor in reps[:3]:
                results = self._test_methods_on_tensor(
                    rep_tensor,
                    rep_name,
                    ttype,
                    max_methods=max_methods_per_tensor,
                    use_all=use_all_methods,
                )
                for r in results:
                    type_method_scores.setdefault(r["method_name"], []).append(
                        r["score"]
                    )
                    all_test_results.append(r)
                type_results.extend(results)

            # Average scores per method
            avg_scores = {
                mname: float(np.mean(scores))
                for mname, scores in type_method_scores.items()
            }
            sorted_methods = sorted(avg_scores.items(), key=lambda x: -x[1])

            # Best method for this type
            best_method = sorted_methods[0][0] if sorted_methods else "block_int8"

            report["per_type_results"][ttype] = {
                "representatives": len(reps),
                "n_tested": len(type_results),
                "best_method": best_method,
                "top_5_methods": [m for m, _ in sorted_methods[:5]],
                "avg_scores": avg_scores,
            }

            # Best cascade plan for this type
            if reps:
                cascade_plan = self.plan_cascade(
                    reps[0][1],
                    tensor_type=ttype,
                    target_ratio=target_ratio,
                    max_error=max_error,
                    name=reps[0][0],
                )
                report["best_cascade_plans"][ttype] = {
                    "source": cascade_plan.source,
                    "n_stages": cascade_plan.n_stages,
                    "stages": [
                        {"method": s.method_name, "category": s.method_category}
                        for s in cascade_plan.stages
                    ],
                }

        # Phase 4: Train genetic strategy
        if all_test_results:
            try:
                best_genome = self.genetic_evolver.evolve(all_test_results)
                report["genetic_strategy"] = {
                    "generation": self.genetic_evolver.generation,
                    "best_fitness": self.genetic_evolver.best_fitness,
                    "best_genome": best_genome,
                }
            except Exception as exc:
                logger.debug("Genetic evolution failed: %s", exc)

        # Phase 5: Oracle statistics
        report["oracle_stats"] = {
            "hits": self._oracle_hits,
            "misses": self._oracle_misses,
            "hit_rate": self._oracle_hits
            / max(self._oracle_hits + self._oracle_misses, 1),
        }

        return report

    def _test_methods_on_tensor(
        self,
        tensor: np.ndarray,
        name: str,
        tensor_type: str,
        max_methods: int = 30,
        use_all: bool = False,
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []

        if use_all:
            methods = self._gather_candidates()
        else:
            all_m = self._gather_candidates()
            methods = [m for m in all_m if m.tier <= 4][:max_methods]

        if not methods:
            methods = all_m[:max_methods]

        for rm in methods:
            if rm.instance is None:
                continue
            try:
                t0 = time.perf_counter()
                data, meta = rm.instance.compress(tensor)
                recon = rm.instance.decompress(data, meta)
                if recon.shape != tensor.shape:
                    recon = recon.reshape(tensor.shape)
                elapsed = time.perf_counter() - t0

                var = float(np.var(tensor))
                mse = float(np.mean((tensor.ravel() - recon.ravel()) ** 2))
                rel_err = mse / var if var > 1e-30 else float(mse)
                cos_sim = float(
                    np.dot(tensor.ravel(), recon.ravel())
                    / max(
                        np.linalg.norm(tensor.ravel()) * np.linalg.norm(recon.ravel()),
                        1e-30,
                    )
                )
                ratio = tensor.nbytes / max(len(data), 1)
                snr = (
                    10.0 * math.log10(var / max(mse, 1e-30))
                    if mse > 1e-30
                    else float("inf")
                )

                results.append(
                    {
                        "method_name": rm.name,
                        "category": rm.category,
                        "tier": rm.tier,
                        "ratio": ratio,
                        "error": rel_err,
                        "cosine_similarity": cos_sim,
                        "snr_db": snr,
                        "compressed_bytes": len(data),
                        "elapsed": elapsed,
                        "score": ratio * (1.0 - rel_err) * (0.5 + 0.5 * cos_sim),
                        "tensor_name": name,
                        "tensor_type": tensor_type,
                    }
                )

                # Record in memory for holographic recall
                signature = self._compute_signature(tensor, name)
                self.holo_memory.store(signature, rm.name, {}, ratio, rel_err)

                # Record in Bayesian tracker
                try:
                    self.bayesian_tracker.record(rm.name, tensor_type, ratio, rel_err)
                except Exception:
                    pass

                # Record in knowledge graph
                try:
                    self.knowledge_graph.update(
                        tensor_type, rm.category, ratio, rel_err
                    )
                except Exception:
                    pass

            except Exception as exc:
                logger.debug("Method '%s' failed on '%s': %s", rm.name, name, exc)

        results.sort(key=lambda r: -r["score"])
        return results

    # ═══════════════════════════════════════════════════════════════════
    #  8. CERTIFICATION
    # ═══════════════════════════════════════════════════════════════════

    def certify(
        self,
        original: np.ndarray,
        compressed_data: bytes,
        metadata: Dict[str, Any],
        name: str = "",
        output_dir: Optional[str] = None,
        formats: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Produce compression certificates for a tensor.

        Parameters
        ----------
        original : np.ndarray
            Original uncompressed tensor.
        compressed_data : bytes
            Compressed tensor data.
        metadata : dict
            Compression metadata.
        name : str
            Tensor name.
        output_dir : str, optional
            Directory for certificate output files.
        formats : list of str, optional
            Output formats: "json", "html", "md", "txt". Default all.

        Returns
        -------
        dict
            Certificate data.
        """
        from ..dataset_compressor import _CompressionCertificate

        cert = _CompressionCertificate(
            name=name,
            original_size=original.nbytes,
            compressed_size=len(compressed_data),
            method=metadata.get("method", "unknown"),
            ratio=metadata.get("total_ratio", 1.0),
            error=metadata.get("total_error", 0.0),
            snr_db=metadata.get("snr_db", 0.0),
            cosine_similarity=metadata.get("cosine_similarity", 0.0),
            timestamp=time.time(),
        )

        cert_data = asdict(cert)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            fmts = formats or ["json", "html", "md", "txt"]
            base = os.path.join(output_dir, f"certificate_{name.replace('.', '_')}")

            if "json" in fmts:
                with open(base + ".json", "w") as f:
                    json.dump(cert_data, f, indent=2, default=str)

            if "txt" in fmts:
                with open(base + ".txt", "w") as f:
                    f.write(self._cert_txt(cert_data))

            if "md" in fmts:
                with open(base + ".md", "w") as f:
                    f.write(self._cert_md(cert_data))

            if "html" in fmts:
                with open(base + ".html", "w") as f:
                    f.write(self._cert_html(cert_data))

        return cert_data

    def certify_model(
        self,
        results: Dict[str, Any],
        output_dir: str = "/tmp/certificates",
        formats: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Produce certificates for all tensors in a compression run."""
        fmts = formats or ["json"]
        certs = []
        meta = results.get("_meta", {})
        for name, result in results.items():
            if name == "_meta":
                continue
            data = result.get("data", b"")
            meta_dict = result.get("metadata", {})
            orig_bytes = result.get("original_bytes", 0)
            cert_data = {
                "tensor_name": name,
                "tensor_type": result.get("tensor_type", "unknown"),
                "method": result.get("method", "unknown"),
                "original_size": orig_bytes,
                "compressed_size": result.get("compressed_bytes", 0),
                "ratio": result.get("ratio", 1.0),
                "error": result.get("error", 0.0),
                "snr_db": meta_dict.get("snr_db", 0.0),
                "cosine_similarity": meta_dict.get("cosine_similarity", 0.0),
                "quality_grade": meta_dict.get("quality_grade", "UNKNOWN"),
                "timestamp": time.time(),
            }
            certs.append(cert_data)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            summary = {
                "model_certificate": {
                    "total_tensors": meta.get("total_tensors", len(certs)),
                    "overall_ratio": meta.get("overall_ratio", 0),
                    "failures": meta.get("failures", 0),
                    "method_distribution": meta.get("method_distribution", {}),
                    "timestamp": time.time(),
                },
                "tensor_certificates": certs,
            }
            if "json" in fmts:
                sp = os.path.join(output_dir, "model_certificate.json")
                with open(sp, "w") as f:
                    json.dump(summary, f, indent=2, default=str)
            if "txt" in fmts:
                sp = os.path.join(output_dir, "model_certificate.txt")
                with open(sp, "w") as f:
                    lines = ["Model Compression Certificate"]
                    lines.append("=" * 60)
                    lines.append(
                        f"Total Tensors: {meta.get('total_tensors', len(certs))}"
                    )
                    lines.append(f"Overall Ratio: {meta.get('overall_ratio', 0):.1f}x")
                    lines.append(f"Failures: {meta.get('failures', 0)}")
                    lines.append(f"\nMethod Distribution:")
                    for m, c in sorted(
                        meta.get("method_distribution", {}).items(), key=lambda x: -x[1]
                    ):
                        lines.append(f"  {m}: {c}")
                    lines.append("")
                    for c_data in certs:
                        lines.append(
                            f"{c_data['tensor_name']:50s} ratio={c_data['ratio']:>8.1f}x  error={c_data['error']:.6f}  grade={c_data['quality_grade']}"
                        )
                    f.write("\n".join(lines))

        return certs

    @staticmethod
    def _cert_txt(data: Dict[str, Any]) -> str:
        return (
            f"Compression Certificate\n"
            f"{'=' * 50}\n"
            f"Tensor: {data.get('name', 'unknown')}\n"
            f"Method: {data.get('method', 'unknown')}\n"
            f"Original Size: {data.get('original_size', 0)} bytes\n"
            f"Compressed Size: {data.get('compressed_size', 0)} bytes\n"
            f"Compression Ratio: {data.get('ratio', 1.0):.1f}x\n"
            f"Error: {data.get('error', 0.0):.6f}\n"
            f"SNR: {data.get('snr_db', 0.0):.1f} dB\n"
            f"Cosine Similarity: {data.get('cosine_similarity', 0.0):.4f}\n"
            f"Timestamp: {data.get('timestamp', 0.0):.0f}\n"
        )

    @staticmethod
    def _cert_md(data: Dict[str, Any]) -> str:
        return (
            f"# Compression Certificate\n\n"
            f"- **Tensor**: {data.get('name', 'unknown')}\n"
            f"- **Method**: {data.get('method', 'unknown')}\n"
            f"- **Original Size**: {data.get('original_size', 0):,} bytes\n"
            f"- **Compressed Size**: {data.get('compressed_size', 0):,} bytes\n"
            f"- **Compression Ratio**: {data.get('ratio', 1.0):.1f}x\n"
            f"- **Error**: {data.get('error', 0.0):.6f}\n"
            f"- **SNR**: {data.get('snr_db', 0.0):.1f} dB\n"
            f"- **Cosine Similarity**: {data.get('cosine_similarity', 0.0):.4f}\n"
        )

    @staticmethod
    def _cert_html(data: Dict[str, Any]) -> str:
        ratio = data.get("ratio", 1.0)
        snr = data.get("snr_db", 0.0)
        color = "green" if ratio > 100 else "orange"
        snr_color = "green" if snr > 30 else "red"
        return (
            f"<html><body>"
            f"<h2>Compression Certificate</h2>"
            f"<table>"
            f"<tr><td>Tensor</td><td>{data.get('name', 'unknown')}</td></tr>"
            f"<tr><td>Method</td><td>{data.get('method', 'unknown')}</td></tr>"
            f"<tr><td>Ratio</td><td style='color:{color}'>{ratio:.1f}x</td></tr>"
            f"<tr><td>Error</td><td>{data.get('error', 0.0):.6f}</td></tr>"
            f"<tr><td>SNR</td><td style='color:{snr_color}'>{snr:.1f} dB</td></tr>"
            f"<tr><td>Cosine</td><td>{data.get('cosine_similarity', 0.0):.4f}</td></tr>"
            f"</table></body></html>"
        )

    # ═══════════════════════════════════════════════════════════════════
    #  9. RECORDING — feedback for continuous learning
    # ═══════════════════════════════════════════════════════════════════

    def record_compression(
        self,
        tensor: np.ndarray,
        tensor_type: str,
        method_name: str,
        method_category: str,
        ratio: float,
        error: float,
        name: str = "",
    ) -> None:
        """Record compression outcome for continuous learning.

        Updates:
        - Holographic associative memory
        - Bayesian performance tracker
        - Compression knowledge graph
        - Genetic strategy evolver
        """
        # Holographic memory
        signature = self._compute_signature(tensor, name)
        try:
            self.holo_memory.store(signature, method_name, {}, ratio, error)
        except Exception:
            pass

        # Bayesian tracker
        try:
            self.bayesian_tracker.record(method_name, tensor_type, ratio, error)
        except Exception:
            pass

        # Knowledge graph
        try:
            self.knowledge_graph.update(tensor_type, method_category, ratio, error)
        except Exception:
            pass

        # History
        self._compression_history.append(
            {
                "tensor_name": name,
                "tensor_type": tensor_type,
                "method_name": method_name,
                "method_category": method_category,
                "ratio": ratio,
                "error": error,
                "success": error < 0.01,
                "timestamp": time.time(),
            }
        )

    # ═══════════════════════════════════════════════════════════════════
    #  10. PERSISTENCE
    # ═══════════════════════════════════════════════════════════════════

    def save_state(self, path: str) -> None:
        """Save all learned knowledge to disk."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        # Holographic memory
        mem_path = path + ".holographic_memory.npz"
        try:
            self.holo_memory.save(mem_path)
        except Exception as exc:
            logger.warning("Failed to save holographic memory: %s", exc)

        # Bayesian tracker
        try:
            self.bayesian_tracker.save(path + ".bayesian.json")
        except Exception as exc:
            logger.warning("Failed to save Bayesian tracker: %s", exc)

        # Knowledge graph
        try:
            kg = self.knowledge_graph.to_dict()
            with open(path + ".knowledge_graph.json", "w") as f:
                json.dump(kg, f, indent=2, default=str)
        except Exception as exc:
            logger.warning("Failed to save knowledge graph: %s", exc)

        # Genetic evolver
        try:
            state = {
                "generation": self.genetic_evolver.generation,
                "best_fitness": self.genetic_evolver.best_fitness,
                "best_genome": self.genetic_evolver.best_genome,
            }
            with open(path + ".genetic.json", "w") as f:
                json.dump(state, f, indent=2, default=str)
        except Exception as exc:
            logger.warning("Failed to save genetic evolver: %s", exc)

    def load_state(self, path: str) -> None:
        """Load all learned knowledge from disk."""
        # Holographic memory
        mem_path = path + ".holographic_memory.npz"
        if os.path.exists(mem_path):
            try:
                self.holo_memory.load(mem_path)
                logger.info("Loaded holographic memory from %s", mem_path)
            except Exception as exc:
                logger.warning("Failed to load holographic memory: %s", exc)

        # Bayesian tracker
        bp = path + ".bayesian.json"
        if os.path.exists(bp):
            try:
                self.bayesian_tracker.load(bp)
                logger.info("Loaded Bayesian tracker from %s", bp)
            except Exception as exc:
                logger.warning("Failed to load Bayesian tracker: %s", exc)

    # ═══════════════════════════════════════════════════════════════════
    #  STATISTICS
    # ═══════════════════════════════════════════════════════════════════

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the world model."""
        mem_stats = {}
        try:
            mem_stats = self.holo_memory.get_stats()
        except Exception:
            pass

        bayesian_stats = {}
        try:
            bayesian_stats = {
                "tracked_pairs": len(self.bayesian_tracker._performances),
            }
        except Exception:
            pass

        kg_stats = {}
        try:
            kg_stats = {
                "tensor_types": len(self.knowledge_graph._graph),
            }
        except Exception:
            pass

        return {
            "oracle": {
                "hits": self._oracle_hits,
                "misses": self._oracle_misses,
                "hit_rate": self._oracle_hits
                / max(self._oracle_hits + self._oracle_misses, 1),
            },
            "holographic_memory": mem_stats,
            "bayesian": bayesian_stats,
            "knowledge_graph": kg_stats,
            "compression_history": {
                "total_compressions": self._n_compressions,
                "total_records": len(self._compression_history),
            },
            "genetic_evolver": {
                "generation": self.genetic_evolver.generation,
                "best_fitness": self.genetic_evolver.best_fitness,
            },
        }

    # ═══════════════════════════════════════════════════════════════════
    #  STATIC HELPERS
    # ═══════════════════════════════════════════════════════════════════

    @staticmethod
    def _classify_by_name(name: str) -> str:
        nl = name.lower()
        if any(k in nl for k in ("embed", "wte", "tok_emb")):
            return "embedding"
        if any(k in nl for k in ("q_proj", "wq")):
            return "attention_q"
        if any(k in nl for k in ("k_proj", "wk")):
            return "attention_k"
        if any(k in nl for k in ("v_proj", "wv")):
            return "attention_v"
        if any(k in nl for k in ("o_proj", "wo")):
            return "attention_o"
        if any(k in nl for k in ("gate_proj", "w1")):
            return "ffn_gate"
        if any(k in nl for k in ("up_proj", "w3")):
            return "ffn_up"
        if any(k in nl for k in ("down_proj", "w2")):
            return "ffn_down"
        if any(k in nl for k in ("norm", "rms", "ln_")):
            return "norm"
        if any(k in nl for k in ("head", "lm_head")):
            return "output"
        return "weight"

    @staticmethod
    def _extract_layer_idx(name: str) -> int:
        import re

        m = re.search(r"layers\.(\d+)", name)
        return int(m.group(1)) if m else -1

    @staticmethod
    def _extract_param_type(name: str) -> str:
        nl = name.lower()
        if "weight" in nl:
            return "weight"
        if "bias" in nl:
            return "bias"
        return "other"

    @staticmethod
    def _sample_flat(tensor: np.ndarray, max_samples: int = 10000) -> np.ndarray:
        flat = tensor.ravel()
        if len(flat) <= max_samples:
            return flat.astype(np.float64)
        rng = np.random.RandomState(42)
        idx = rng.choice(len(flat), max_samples, replace=False)
        return flat[idx].astype(np.float64)

    @staticmethod
    def _lightweight_dct(x: np.ndarray) -> np.ndarray:
        n = len(x)
        x2 = np.zeros(2 * n, dtype=np.float64)
        x2[:n] = x
        x2[n:] = x[::-1]
        fft = np.fft.fft(x2)[:n]
        scale = np.sqrt(2.0 / n)
        coeffs = fft.real * scale
        coeffs[0] *= 1.0 / np.sqrt(2.0)
        return coeffs

    @staticmethod
    def _compressibility_from_profile(profile: Dict[str, Any]) -> float:
        score = 0.0
        er = profile.get("effective_rank", 0.5)
        ec = profile.get("energy_concentration", 0.0)
        if isinstance(er, (int, float)) and er < 0.3:
            score += 0.3
        if isinstance(ec, (int, float)) and ec > 0.8:
            score += 0.3
        score += 0.2 * (1.0 - min(profile.get("noise_floor", 0.0) * 10, 1.0))
        return min(score, 1.0)


def n_elements(tensor: np.ndarray) -> int:
    """Return the number of elements in a tensor."""
    return tensor.size
