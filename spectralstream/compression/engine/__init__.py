"""
Compressed Intelligence Engine — re-exports from refactored submodules.
"""

from __future__ import annotations

# ── Constants ─────────────────────────────────────────────────────────
from ._constants import (
    CHUNK_SIZE,
    DEFAULT_BLOCK_SIZE,
    DEFAULT_GROUP_SIZE,
    MAX_PROFILE_SAMPLES,
    MIN_SAMPLE_RATIO,
    QUALITY_GRADE_THRESHOLDS,
    SAFETENSORS_HEADER_LEN,
)

# ── Helpers ───────────────────────────────────────────────────────────
from ._helpers import (
    _block_diagonal_score,
    _block_sparsity_score,
    _bootstrap_error,
    _circulant_score,
    _compute_metrics,
    _compute_ratio,
    _estimate_entropy_rate,
    _estimate_noise_floor,
    _hierarchical_structure_score,
    _kolmogorov_estimate,
    _metrics_summary,
    _mutual_information_blocks,
    _nm_sparsity_score,
    _safe_bytes,
    _sample_flat,
    _structured_nm_score,
    _toeplitz_score,
    _unstructured_sparsity_score,
)

# ── Sensitivity ───────────────────────────────────────────────────────
from ._sensitivity import (
    LAYER_SENSITIVITY,
    SUBLEVEL_SENSITIVITY,
    _get_sensitivity,
)

# ── LRU Cache ─────────────────────────────────────────────────────────
from ._lru_cache import LRUCache

# ── Compression Methods ───────────────────────────────────────────────
from ._methods import (
    METHOD_REGISTRY,
    _BlockINT4,
    _BlockINT8,
    _DCTSpectral,
    _DeltaINT4,
    _FWHTCompress,
    _HadamardINT4,
    _HadamardINT8,
    _SVDCompress,
    _SparsityINT4,
    _TensorTrain,
)

# ── Compression-First Priority System ─────────────────────────────────
from .method_tiers import (
    CATEGORY_TIER_MAP,
    METHOD_TIER_MAP,
    MethodTier,
    get_tier,
    tier_score,
    log_tier_distribution,
)

# ── Float32 Preservation ──────────────────────────────────────────────
from .float32_support import Float32Preserver

# ── Method Discovery ──────────────────────────────────────────────────
from .method_discovery import MethodDiscovery

# ── Model Calibrator ──────────────────────────────────────────────────
from .model_calibrator import ModelCalibrator, calibrate_engine, calibrate_tensor

# ── Model Intelligence ───────────────────────────────────────────────
from .model_intelligence import (
    ModelIntelligence,
    ModelIntelligenceEngine,
    HighFidelityProfiler,
    MethodOutcomePredictor,
    TensorDigitalTwin,
    integrate_into_engine,
)

# ── MoE Compression ──────────────────────────────────────────────────
from .moe_compression import MoEAwareCompressor

# ── Self-Evolving Intelligence ───────────────────────────────────────
from .self_evolving_intelligence import (
    SelfEvolvingIntelligenceEngine,
    BayesianPerformanceTracker,
    GeneticStrategyEvolver,
    CompressionKnowledgeGraph,
    MethodPerformance,
    integrate_self_evolving_engine,
)

# ── Quantum Plasma Fusion ────────────────────────────────────────────
from .quantum_plasma_fusion import (
    QuantumPlasmaFusionEngine as QuantumPlasmaFusionEngine_v3,
    AnnealingResult,
    TunnelEvent,
    fuse_with_engine as fuse_v3,
)

# ── Dynamic Selector v2 ───────────────────────────────────────────────
from .dynamic_selector2 import DynamicIntelligenceSelector

# ── Data Classes ──────────────────────────────────────────────────────
from ._dataclasses import (
    CalibrationData,
    CompressedTensor,
    CompressionConfig,
    CompressionReport,
    CompressionTelemetry,
    TensorProfile,
)

# ── Profiler / Allocator / Selector ───────────────────────────────────
from ._profiler import CompressionProfiler
from ._allocator import ErrorBudgetAllocator

# ── IO ────────────────────────────────────────────────────────────────
from ._io import _CheckpointManager, _SSFIOWriter, _SafetensorsIO

# ── Orchestrator ──────────────────────────────────────────────────────
from ._orchestrator import CompressionIntelligenceEngine

# ── Cascade Learner ──────────────────────────────────────────────────
from .cascade_learner import CascadeLearner, CascadePattern

# ── Dynamic Method Tester ────────────────────────────────────────────
from .dynamic_method_tester import DynamicMethodTester, MethodTestResult

# ── Quantum Cascade ─────────────────────────────────────────────────
from .quantum_cascade import (
    CascadeResult,
    CascadeReport,
    CascadeStage,
    CascadeSuperpositionPlan,
    QuantumCascadeEngine,
    QuantumSuperpositionEngine,
    QuantumSuperpositionTest,
)

# ── Unified Intelligence ─────────────────────────────────────────────
from ._unified_intelligence import UnifiedIntelligence

# ── Streaming ─────────────────────────────────────────────────────────
from .streaming_compressor import StreamingCompressor
from .multi_shard_io import MultiShardSafetensorsIO, StreamingCompressionOrchestrator
from . import streaming as streaming_pkg

# ── World Model Compressor ────────────────────────────────────────────
from .world_model_compressor import WorldModelCompressor, compress_with_world_model

# ── Tensor Grouping Optimizer ──────────────────────────────────────────
from .grouping_optimizer import (
    TensorGroup,
    TensorGrouper,
    compress_with_grouping,
    extract_name_pattern,
    group_tensors,
    get_tensor_metadata_from_dict,
    get_tensor_metadata_from_info,
)

# ── Resonant Tensor Grouping ───────────────────────────────────────────
from .resonant_grouping import (
    ResonantGrouper,
    ResonantGroup,
    SpectralResonanceProfile,
    resonance_refine_groups,
)

# ── Parallel Compressor ──────────────────────────────────────────────
from .parallel_compressor import ParallelCompressor

# ── Holographic Resonance Oracle ──────────────────────────────────────
from .holographic_oracle import (
    HolographicOracle,
    HolographicMemoryStore,
    ResonanceSignature,
)

# ── Memory-Mapped Tensor Engine ────────────────────────────────────────
from .memory_mapped_engine import MemoryMappedTensorEngine
from .chunked_compressor import ChunkedCompressor
from .streaming_pipeline import StreamingCompressionPipeline

# ── Progressive Memory & HPC Kernel Fusion ──────────────────────────
from .progressive_release import ProgressiveMemoryManager
from .hpc_kernel_fusion import HPCKernelFusion

# ── Utils ─────────────────────────────────────────────────────────────
from ._utils import (
    compression_config_from_ss_config,
    create_engine,
    estimate_swift_ratio,
    load_compression_config,
)

# ── Target Ratio Engine & Predictor ───────────────────────────────────
from .dynamic_tuning.target_ratio_engine import (
    PredictorRegistry,
    TargetRatioEngine,
)


import logging as _logging

logger = _logging.getLogger(__name__)

# ── Pareto Streaming ──────────────────────────────────────────────────
from .dynamic_tuning.pareto_streaming import (
    ParetoFrontier,
    ParetoPoint,
    ProgressiveStreamingCompressor,
)

# ── Multiplicative Stacking ───────────────────────────────────────────
from .dynamic_tuning.multiplicative_stacking import (
    MultiplicativeStackingEngine,
    StackingPlan,
    StackingStage,
    StackingCandidate,
)

# ── Direct Cascade Engine ─────────────────────────────────────────────
from .direct_cascade import DirectCascadeEngine
from .cascade_configs import CASCADE_CONFIGS

# ── Method Stacking Engine ─────────────────────────────────────────────
from .stacking_engine import (
    MethodStackingEngine,
    try_stacking_fallback,
)

# ── Tiered Error Budgets ────────────────────────────────────────────────
from .tiered_error import (
    TIERED_BUDGETS,
    get_budget,
    get_budget_dict,
    select_cascade_pattern,
    is_within_budget,
    get_fallback_pattern,
)

# ── Compression Intelligence (archive migration) ──────────────────────
from .compression_intelligence import (
    TensorCategory,
    StrategyScore,
    TensorAnalyzer,
    CompressionStrategySelector,
    BitBudget,
    BitBudgetOptimizer,
    RateDistortionPoint,
    LagrangianRateDistortion,
    AdaptationState,
    AdaptiveMethodSelector,
    CompressionPlan,
    CompressionOrchestrator,
)

# ── Unified Intelligence Engine (archive migration) ───────────────────
from .intelligence_real import (
    CompressionStrategy,
    CompressionResult as CompressionResultV1,
    UnifiedIntelligenceEngine,
)

# ── Dynamic Tensor Intelligence (archive migration) ───────────────────
from .dynamic_tensor_intelligence import (
    TensorFeatures,
    StrategyPerformance,
    DynamicTensorIntelligence,
)

# ── Unified Quantization System (archive migration) ───────────────────
from .unified_quant_system import (
    CompressionMethod as QuantCompressionMethod,
    METHOD_NAMES,
    BLOCK_SIZE_HINTS,
    NoiseFloorDetector,
    EntropyCoder,
    HadamardPreconditioner,
    UnifiedQuantizationSystem,
    get_system,
)

# ── Compression Profiler v2 (archive migration) ──────────────────────
from .compression_profiler import (
    SensitivityResult,
    CompressionProfiler as CompressionProfilerV2,
)

# ── Compression Intelligence v2 (archive migration) ──────────────────
from .intelligence import (
    MethodScore,
    MethodEvaluator,
    CompressionIntelligence,
    CATEGORY_AFFINITY,
)

# ── Quantization Engine (archive migration) ──────────────────────────
from .quantization_engine import (
    QualityMonitor,
    StrategySelector,
    SpectralQuantizer,
    GGMLDequantizerEngine,
    UnifiedQuantizer,
    CompressionReport as QECompressionReport,
    _format_size,
)

# ── Loss Metrics ──────────────────────────────────────────────────────────
from .loss_metrics import (
    TensorLossMetrics,
    LossMetricsTracker,
    compute_tiered_error_budget,
    _grade_quality,
    QUALITY_EXCELLENT_SNR,
    QUALITY_GOOD_SNR,
    QUALITY_FAIR_SNR,
    QUALITY_POOR_SNR,
    QUALITY_EXCELLENT_MSE,
    QUALITY_GOOD_MSE,
    QUALITY_FAIR_MSE,
    QUALITY_POOR_MSE,
    QUALITY_EXCELLENT_COSINE,
    QUALITY_GOOD_COSINE,
    QUALITY_FAIR_COSINE,
    QUALITY_POOR_COSINE,
)

# ── Physics-Inspired Sub-Engines ───────────────────────────────────────
from .dynamic_tuning.time_crystal_engine import (
    TimeCrystalCompressionEngine,
    FloquetOperator,
    TimeCrystalCycle,
    CrystalMethodState,
)

from .dynamic_tuning.quantum_field_cascade import (
    QuantumFieldCascadeOptimizer,
    FeynmanVertex,
    MethodPropagator,
    ScatteringAmplitude,
)

from .dynamic_tuning.plasma_confinement import (
    PlasmaConfinementTensorShaper,
    TokamakFieldLine,
    MagneticIsland,
    SafetyFactorProfile,
)


__all__ = [
    # Parallel compressor
    "ParallelCompressor",
    # New intelligence subsystems
    "HolographicOracle",
    "HolographicMemoryStore",
    "ResonanceSignature",
    "QuantumCascadeEngine",
    "CascadeResult",
    "CascadeReport",
    "ResonantGrouper",
    "ResonantGroup",
    "SpectralResonanceProfile",
    "resonance_refine_groups",
    # Archive migration classes
    "TensorCategory",
    "StrategyScore",
    "TensorAnalyzer",
    "CompressionStrategySelector",
    "BitBudget",
    "BitBudgetOptimizer",
    "RateDistortionPoint",
    "LagrangianRateDistortion",
    "AdaptationState",
    "AdaptiveMethodSelector",
    "CompressionPlan",
    "CompressionOrchestrator",
    "CompressionStrategy",
    "CompressionResultV1",
    "UnifiedIntelligenceEngine",
    "TensorFeatures",
    "StrategyPerformance",
    "DynamicTensorIntelligence",
    "QuantCompressionMethod",
    "METHOD_NAMES",
    "BLOCK_SIZE_HINTS",
    "NoiseFloorDetector",
    "EntropyCoder",
    "HadamardPreconditioner",
    "UnifiedQuantizationSystem",
    "get_system",
    "SensitivityResult",
    "CompressionProfilerV2",
    "MethodScore",
    "MethodEvaluator",
    "CompressionIntelligence",
    "CATEGORY_AFFINITY",
    "QualityMonitor",
    "StrategySelector",
    "SpectralQuantizer",
    "GGMLDequantizerEngine",
    "UnifiedQuantizer",
    "QECompressionReport",
    "_format_size",
    "CHUNK_SIZE",
    "DEFAULT_BLOCK_SIZE",
    "DEFAULT_GROUP_SIZE",
    "MAX_PROFILE_SAMPLES",
    "MIN_SAMPLE_RATIO",
    "QUALITY_GRADE_THRESHOLDS",
    "SAFETENSORS_HEADER_LEN",
    "CascadeLearner",
    "CascadePattern",
    "UnifiedIntelligence",
    "LAYER_SENSITIVITY",
    "SUBLEVEL_SENSITIVITY",
    "LRUCache",
    "METHOD_REGISTRY",
    "CalibrationData",
    "CompressedTensor",
    "CompressionConfig",
    "CompressionIntelligenceEngine",
    "CompressionTelemetry",
    "CompressionProfiler",
    "CompressionReport",
    "WorldModelCompressor",
    "compress_with_world_model",
    "DynamicIntelligenceSelector",
    "ErrorBudgetAllocator",
    "TensorProfile",
    "MethodTier",
    "MethodDiscovery",
    "Float32Preserver",
    "METHOD_TIER_MAP",
    "CATEGORY_TIER_MAP",
    "log_tier_distribution",
    "StreamingCompressor",
    "TargetRatioEngine",
    "PredictorRegistry",
    "MultiShardSafetensorsIO",
    "StreamingCompressionOrchestrator",
    "compression_config_from_ss_config",
    "create_engine",
    "estimate_swift_ratio",
    "load_compression_config",
    "ParetoFrontier",
    "ParetoPoint",
    "ProgressiveStreamingCompressor",
    "MemoryMappedTensorEngine",
    "ChunkedCompressor",
    "StreamingCompressionPipeline",
    "MultiplicativeStackingEngine",
    "StackingPlan",
    "StackingStage",
    "StackingCandidate",
    "DirectCascadeEngine",
    "CASCADE_CONFIGS",
    "MethodStackingEngine",
    "try_stacking_fallback",
    "TimeCrystalCompressionEngine",
    "FloquetOperator",
    "TimeCrystalCycle",
    "CrystalMethodState",
    "QuantumFieldCascadeOptimizer",
    "FeynmanVertex",
    "MethodPropagator",
    "ScatteringAmplitude",
    "PlasmaConfinementTensorShaper",
    "TokamakFieldLine",
    "MagneticIsland",
    "SafetyFactorProfile",
    "_compute_metrics",
    "_get_sensitivity",
    "_BlockINT8",
    "_BlockINT4",
    "_DCTSpectral",
    "_DeltaINT4",
    "_FWHTCompress",
    "_HadamardINT8",
    "_HadamardINT4",
    "_SVDCompress",
    "_SparsityINT4",
    "_TensorTrain",
    "ModelCalibrator",
    "calibrate_engine",
    "calibrate_tensor",
    "ModelIntelligence",
    "ModelIntelligenceEngine",
    "HighFidelityProfiler",
    "MethodOutcomePredictor",
    "TensorDigitalTwin",
    "integrate_into_engine",
    "MoEAwareCompressor",
    "SelfEvolvingIntelligenceEngine",
    "BayesianPerformanceTracker",
    "GeneticStrategyEvolver",
    "CompressionKnowledgeGraph",
    "MethodPerformance",
    "integrate_self_evolving_engine",
    "QuantumPlasmaFusionEngine_v3",
    "AnnealingResult",
    "TunnelEvent",
    "fuse_v3",
    "ProgressiveMemoryManager",
    "HPCKernelFusion",
    "TensorGroup",
    "TensorGrouper",
    "compress_with_grouping",
    "extract_name_pattern",
    "group_tensors",
    "get_tensor_metadata_from_dict",
    "get_tensor_metadata_from_info",
    # Loss Metrics
    "TensorLossMetrics",
    "LossMetricsTracker",
    "compute_tiered_error_budget",
    "_grade_quality",
    "QUALITY_EXCELLENT_SNR",
    "QUALITY_GOOD_SNR",
    "QUALITY_FAIR_SNR",
    "QUALITY_POOR_SNR",
    "QUALITY_EXCELLENT_MSE",
    "QUALITY_GOOD_MSE",
    "QUALITY_FAIR_MSE",
    "QUALITY_POOR_MSE",
    "QUALITY_EXCELLENT_COSINE",
    "QUALITY_GOOD_COSINE",
    "QUALITY_FAIR_COSINE",
    "QUALITY_POOR_COSINE",
    "CascadeResult",
    "CascadeReport",
    "CascadeStage",
    "CascadeSuperpositionPlan",
    "QuantumCascadeEngine",
    "QuantumSuperpositionEngine",
    "QuantumSuperpositionTest",
    # Tiered Error Budgets
    "TIERED_BUDGETS",
    "get_budget",
    "get_budget_dict",
    "select_cascade_pattern",
    "is_within_budget",
    "get_fallback_pattern",
]
