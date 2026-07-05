import warnings as _warnings

from spectralstream.inference.config import Gemma4Config
from spectralstream.inference.loader import ModelLoader, _TensorEntry, _LRUCache
from spectralstream.inference.attention import Gemma4Attention, Gemma4RMSNorm
from spectralstream.inference.ffn import Gemma4FFN, _gelu_tanh
from spectralstream.inference.layer import TransformerLayer
from spectralstream.inference.model_config import GenericModelConfig
from spectralstream.inference.unified_loader import UnifiedModelLoader
from spectralstream.inference.intelligence_engine import (
    InferenceIntelligenceEngine,
    InferenceIntelligenceConfig,
)
from spectralstream.inference.model import CPUInferenceEngine
from spectralstream.inference.generation import (
    GenerationResult,
    PredictiveConfidenceCascade,
    StagedBlockEmission,
    ThermalNoiseInjection,
    SpeculativeDecoder,
)

_warnings.filterwarnings(
    "once",
    category=DeprecationWarning,
    module="spectralstream.inference.(model|generation|engine_legacy|moe_inference)",
)

# Unified inference system with COCONUT, Vlasov, HDC, TimeCrystal, etc.
from spectralstream.inference.coconut import (
    COCONUTEngine,
    integrate_coconut,
    coconut_action,
)
from spectralstream.inference.hrr_memory import HrrMemory, HolographicKVCache
from spectralstream.inference.vlasov import VlasovMeanFieldAttention
from spectralstream.inference.vlasov_pic import (
    VlasovPICSolverV2,
    V2PICAttentionLayer,
    AMRGrid,
    solve_screened_poisson_1d,
    tsc_deposit,
    tsc_interpolate,
    boris_push,
    monte_carlo_collisions,
    langevin_thermostat,
    FieldDiagnostics,
    identify_fast_particles,
    sub_cycle_push,
    filter_current_density,
    spectral_filter_current,
    apply_perfect_conductor_bc,
    reflect_particles,
)
from spectralstream.inference.mean_field import (
    SpectralField,
    VlasovMeanFieldAttention,
    gyrokinetic_split,
    SpectralGate,
    HDCSkipPredictor,
    SymplecticIntegrator,
    ResonanceRouter,
    HopfieldEnergy,
    MonoidalChunk,
    AdaptiveSpectralRank,
    BornMachineSampler,
    GroverAmplifier,
    CrossLayerResonance,
)
from spectralstream.inference.vlasov_attention import (
    VlasovConfig,
    VlasovAttention,
    PICStep,
    VlasovPICScheduler,
)
from spectralstream.inference.hybrid_attention import (
    AttentionMethod,
    HybridAttentionConfig,
    RouterState,
    StandardAttentionBackend,
    WaveletAttentionBackend,
    VlasovAttentionBackend,
    LinearAttentionBackend,
    AttentionRouter,
    HybridAttention,
)
from spectralstream.inference.resonance import (
    TimeCrystalResonator,
    SpectralResonanceMeter,
    AdaptivePIDController,
    ResonanceRouter,
)
from spectralstream.inference.confidence_gate import ConfidenceGate
from spectralstream.inference.attractor import (
    SpectralEntropyScorer,
    HopfieldEnergyScorer,
    AttractorScoringEnsemble,
)
from spectralstream.inference.hdc_engine import HDCBundle, NGramCascade, HDCDraftEngine
from spectralstream.inference.block_emission import BlockEmissionPipeline
from spectralstream.inference.monitor import InferenceMonitor
from spectralstream.inference.online_learning import OnlineLearningEngine
from spectralstream.inference.unified import (
    UnifiedInferenceEngine,
    UnifiedStrategyLevel,
    UNIFIED_STRATEGY_NAMES,
    create_unified_engine,
)

from spectralstream.inference.progressive_loader import (
    ProgressiveWeightLoader,
    ZigzagOrder,
    DCTWeightPacker,
    ProgressiveStage,
    STAGES_PROGRESSIVE,
    LayerPriority,
    compute_layer_priority,
    PredictivePrefetcher,
    BackgroundRefiner,
    ProgressiveQualityReport,
)
from spectralstream.inference.mmap_engine import MmapEngine, MMapTensorStore
from spectralstream.inference.persistence import StateManager
from spectralstream.inference.weight_loader import (
    WeightLoader,
    ModelConfig,
    WeightMapper,
    CompressionReport,
    load_weights,
    load_model_weights,
    validate_gguf_pipeline,
    inspect_gguf,
)

# Coherence engine (migrated from archive)
from spectralstream.inference.coherence import (
    CoherenceEngine,
    AttractorGuidedGenerator,
)

# AST inference engine (migrated from archive)
from spectralstream.inference.ast_inference import (
    ASTInferenceEngine,
    ASTGuidedGenerator,
)

# Spectral frequency-domain inference (migrated from archive)
from spectralstream.inference.spectral import (
    SpectralWeightStore,
    SpectralMatmul,
    SpectralAttention,
    SpectralFFN,
    SpectralTransformerConfig,
    SpectralForwardPass,
    FrequencyDomainOptimization,
    BenchmarkResult,
    SpectralBenchmark,
)

# 6-tier streaming engine (migrated from archive)
from spectralstream.inference.streaming_engine import (
    MemoryTier,
    TieredBlock,
    MemoryPressure,
    NUMAAllocator,
    L3DCache,
    AsyncPrefetcher,
    HDCBlockPredictorV2,
    MemoryPressureMonitor,
    BandwidthScheduler,
    L0RegisterContext,
    StreamingEngineV2,
)

__all__ = [
    "Gemma4Config",
    "GenericModelConfig",
    "UnifiedModelLoader",
    "InferenceIntelligenceEngine",
    "InferenceIntelligenceConfig",
    "ModelLoader",
    "_TensorEntry",
    "_LRUCache",
    "MmapEngine",
    "MMapTensorStore",
    "StateManager",
    "ProgressiveWeightLoader",
    "ZigzagOrder",
    "DCTWeightPacker",
    "ProgressiveStage",
    "STAGES_PROGRESSIVE",
    "LayerPriority",
    "compute_layer_priority",
    "PredictivePrefetcher",
    "BackgroundRefiner",
    "ProgressiveQualityReport",
    "Gemma4Attention",
    "Gemma4RMSNorm",
    "Gemma4FFN",
    "_gelu_tanh",
    "TransformerLayer",
    "CPUInferenceEngine",
    "GenerationResult",
    "PredictiveConfidenceCascade",
    "StagedBlockEmission",
    "ThermalNoiseInjection",
    "SpeculativeDecoder",
    "COCONUTEngine",
    "integrate_coconut",
    "coconut_action",
    "HrrMemory",
    "HolographicKVCache",
    "VlasovMeanFieldAttention",
    "VlasovPICSolverV2",
    "V2PICAttentionLayer",
    "AMRGrid",
    "solve_screened_poisson_1d",
    "tsc_deposit",
    "tsc_interpolate",
    "boris_push",
    "monte_carlo_collisions",
    "langevin_thermostat",
    "FieldDiagnostics",
    "identify_fast_particles",
    "sub_cycle_push",
    "filter_current_density",
    "spectral_filter_current",
    "apply_perfect_conductor_bc",
    "reflect_particles",
    "SpectralField",
    "gyrokinetic_split",
    "SpectralGate",
    "HDCSkipPredictor",
    "SymplecticIntegrator",
    "HopfieldEnergy",
    "MonoidalChunk",
    "AdaptiveSpectralRank",
    "BornMachineSampler",
    "GroverAmplifier",
    "CrossLayerResonance",
    "VlasovConfig",
    "VlasovAttention",
    "PICStep",
    "VlasovPICScheduler",
    "AttentionMethod",
    "HybridAttentionConfig",
    "RouterState",
    "StandardAttentionBackend",
    "WaveletAttentionBackend",
    "VlasovAttentionBackend",
    "LinearAttentionBackend",
    "AttentionRouter",
    "HybridAttention",
    "TimeCrystalResonator",
    "SpectralResonanceMeter",
    "AdaptivePIDController",
    "ResonanceRouter",
    "ConfidenceGate",
    "SpectralEntropyScorer",
    "HopfieldEnergyScorer",
    "AttractorScoringEnsemble",
    "HDCBundle",
    "NGramCascade",
    "HDCDraftEngine",
    "BlockEmissionPipeline",
    "InferenceMonitor",
    "OnlineLearningEngine",
    "UnifiedInferenceEngine",
    "UnifiedStrategyLevel",
    "UNIFIED_STRATEGY_NAMES",
    "create_unified_engine",
    "WeightLoader",
    "ModelConfig",
    "WeightMapper",
    "CompressionReport",
    "load_weights",
    "load_model_weights",
    "validate_gguf_pipeline",
    "inspect_gguf",
    # Coherence engine
    "CoherenceEngine",
    "AttractorGuidedGenerator",
    # AST inference
    "ASTInferenceEngine",
    "ASTGuidedGenerator",
    # Spectral inference
    "SpectralWeightStore",
    "SpectralMatmul",
    "SpectralAttention",
    "SpectralFFN",
    "SpectralTransformerConfig",
    "SpectralForwardPass",
    "FrequencyDomainOptimization",
    "BenchmarkResult",
    "SpectralBenchmark",
    # Streaming engine
    "MemoryTier",
    "TieredBlock",
    "MemoryPressure",
    "NUMAAllocator",
    "L3DCache",
    "AsyncPrefetcher",
    "HDCBlockPredictorV2",
    "MemoryPressureMonitor",
    "BandwidthScheduler",
    "L0RegisterContext",
    "StreamingEngineV2",
]
