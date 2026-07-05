"""Utility modules (meta-controller, multimodal prompts, sampling, tokenization, etc.)."""

from .meta_controller import (
    AutoTuner,
    BanditOptimizer,
    PerformanceModel,
    HardwareAdaptation,
    WorkloadPredictor,
    QualityController,
    ResourceController,
    OnlineLearner,
    MetaController,
)

try:
    from .multimodal_prompt import (
        SpectralVisionDCT,
        AutoChain,
        VisionTransformerEncoder,
        AudioCNNEncoder,
        AudioEncoder,
        CodeInterpreter,
        PromptOptimizer,
        PromptTemplateEngine,
        HolographicContext,
        ResonantPromptingOptimizer,
        QuantumPromptOptimizer,
        VlasovChainOfThought,
        SpectralMultiModalOrchestrator,
        MultiModalChat,
    )
except ImportError:
    pass
try:
    from .sampler_engine import (
        AdaptiveSampler,
        AttractorBasinSampler,
        BeamSearchSampler,
        BornRuleQuantumSampler,
        ContrastiveSampler,
        DiverseBeamSearch,
        EtaSampler,
        GreedySampler,
        HamiltonianTrajectorySampler,
        HolographicPatternSampler,
        LocallyTypicalSampler,
        MinPSampler,
        MirostatV1,
        MirostatV2,
        PredictorCorrectorSampler,
        QuantumCollapseSampler,
        SpectralResonanceSampler,
        SpectralSpeculativeSampler,
        TFSSampler,
        TemperatureSampler,
        TopASampler,
        TopKSampler,
        TopPSampler,
        TypicalSampler,
        VlasovFieldSampler,
        FrequencyPenalty,
        RepetitionPenalty,
        PresencePenalty,
        LogitBiasProcessor,
        TokenBanProcessor,
        GrammarConstraint,
        JSONModeProcessor,
        XMLModeProcessor,
        SamplerPipeline,
        BatchSampler,
        create_sampler,
        create_sampler_pipeline,
        build_default_pipeline,
        list_samplers,
    )
except ImportError:
    pass
try:
    from .tokenizer_engine import (
        BPETokenizer,
        SentencePieceTokenizer,
        TiktokenTokenizer,
        AutoTokenizer,
        CachedTokenizer,
        ParallelTokenizer,
        SpectralTokenizer,
    )
except ImportError:
    pass
try:
    from .predictor_ensemble import (
        PredictorEnsemble,
        HDCPredictor,
        NGramPredictor,
        SpectralHDCPredictor,
        LinearProbePredictor,
        ModelForwardPredictor,
        SlidingAccuracyTracker,
        TokenTypeClassifier,
        Oracle,
        AdaptivePruner,
    )
except ImportError:
    pass
try:
    from .dashboard import DashboardServer, ConsoleDashboard
except ImportError:
    pass
try:
    from .monitoring import EnhancedInferenceMonitor
except ImportError:
    pass
try:
    from .hardware_optimizer import (
        HardwareProbe,
        ThreadPoolOptimizer,
        MemoryPool,
        AVX2Kernels,
        CacheAwareTiling,
        VulkanGPUOffload,
    )
except ImportError:
    pass
try:
    from .math_engine import (
        MathCorrector,
        MathExpressionDetector,
        SafeEvaluator,
        MathAwarePipeline,
        HDCMathAwareness,
        SafeMathError,
    )
except ImportError:
    pass
try:
    from .simd_backend import (
        CPUFeatures,
        SIMDBackend,
        CacheAwareBlocker,
    )
except ImportError:
    pass
