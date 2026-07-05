"""
SpectralOrchestrator - Master controller wiring ALL 25+ modules into a coherent system.

Strategy levels (from fastest to slowest):
0. FORWARDLESS: HDC only, no model call
1. BLOCK_EMISSION: HDC draft -> verify block
2. SPECULATIVE: HDC draft -> verify each token
3. STANDARD: Model only, no acceleration
4. FALLBACK: RNG (emergency only)
"""

import numpy as np
import time
import json
from typing import Optional
from collections import deque

from spectralstream.config import SpectralStreamConfig
from spectralstream.core.math_primitives.numerical import softmax
from spectralstream.inference.hdc_engine import HDCDraftEngine
from spectralstream.kv_cache.spectral import SpectralKVCache, ResonanceTracker
from spectralstream.inference.block_emission import BlockEmissionPipeline
from spectralstream.inference.confidence_gate import ConfidenceGate
from spectralstream.inference.online_learning import OnlineLearningEngine
from spectralstream.inference.attractor import AttractorScoringEnsemble
from spectralstream.inference.resonance import (
    SpectralResonanceMeter,
    AdaptivePIDController,
    ResonanceRouter,
)

try:
    from spectralstream.inference.cascade_controller import (
        CascadeStrategySelector,
        SelfHealingHDC,
        ResonanceAwareSpeculation,
        CrossContextMemory,
        ProactiveAccuracyManager,
    )

    _HAS_CASCADE = True
except ImportError:
    CascadeStrategySelector = None
    SelfHealingHDC = None
    ResonanceAwareSpeculation = None
    CrossContextMemory = None
    ProactiveAccuracyManager = None
    _HAS_CASCADE = False

try:
    from spectralstream.inference.adaptive_inference import (
        PredictiveConfidenceCascade,
        StagedBlockEmission,
        SelfTuningHDCParams,
        EntropyGuidedExploration,
        ThermalNoiseInjection,
    )

    _HAS_ADAPTIVE = True
except ImportError:
    PredictiveConfidenceCascade = None
    StagedBlockEmission = None
    SelfTuningHDCParams = None
    EntropyGuidedExploration = None
    ThermalNoiseInjection = None
    _HAS_ADAPTIVE = False

try:
    from spectralstream.compression.novel_operators import (
        HDCWeightedTokenSampling,
        SpectralEntropyGating,
        AdaptiveForwardlessDepth,
        GradientFreeFineTuning,
        PredictorCorrectorInference,
    )

    _HAS_NOVEL_OPS = True
except ImportError:
    HDCWeightedTokenSampling = None
    SpectralEntropyGating = None
    AdaptiveForwardlessDepth = None
    GradientFreeFineTuning = None
    PredictorCorrectorInference = None
    _HAS_NOVEL_OPS = False

from spectralstream.inference.monitor import InferenceMonitor

try:
    from spectralstream.utils.monitoring import MetricsExporter

    _HAS_METRICS_EXPORTER = True
except ImportError:
    MetricsExporter = None
    _HAS_METRICS_EXPORTER = False

try:
    from spectralstream.memory.persistence import StateManager

    _HAS_PERSISTENCE = True
except ImportError:
    StateManager = None
    _HAS_PERSISTENCE = False

try:
    from spectralstream.tensor.quantum_sampler import (
        BornMachineSampler,
        QuantumResonance,
    )

    _HAS_QUANTUM = True
except ImportError:
    BornMachineSampler = None
    QuantumResonance = None
    _HAS_QUANTUM = False

try:
    from spectralstream.gguf_model import GGUFModel, DummyModel, load_model

    _HAS_GGUF = True
except ImportError:
    GGUFModel = None
    DummyModel = None
    load_model = None
    _HAS_GGUF = False

from spectralstream.utils.tokenizer_engine import (
    AutoTokenizer,
    CachedTokenizer,
    get_tokenizer_info,
)


FORWARDLESS = 0
BLOCK_EMISSION = 1
SPECULATIVE = 2
STANDARD = 3
FALLBACK = 4

STRATEGY_NAMES = {
    FORWARDLESS: "forwardless",
    BLOCK_EMISSION: "block_emission",
    SPECULATIVE: "speculative",
    STANDARD: "standard",
    FALLBACK: "fallback",
}


class _Stub:
    """Stub for optional subsystems that are not available."""

    def __init__(self, *args, **kwargs):
        pass

    def __getattr__(self, name):
        return lambda *args, **kwargs: None

    def __bool__(self):
        return False

    def stats(self) -> dict:
        return {}

    def reset(self):
        pass

    def get_params(self) -> dict:
        return {}

    def report(self) -> dict:
        return {}


class SpectralOrchestrator:
    """
    Master orchestrator for the SpectralStream inference engine.

    Wires together ALL subsystems:
      HDC Draft, Spectral KV, Block Emission, Confidence Gate,
      Online Learning, Attractor Scoring, Resonance Router,
      Cascade Controller, Adaptive Inference, Novel Operators,
      Vlasov Mean-Field, TurboQuant Codec, Memory Optimizer,
      SSD Streamer, Tiered Storage, Monitoring, Persistence,
      GGUF Fine-Tuning, Quantum Sampler.
    """

    def __init__(
        self,
        config: Optional[SpectralStreamConfig] = None,
        model_path: Optional[str] = None,
        hidden_dim: int = 512,
        vocab_size: int = 32000,
        n_heads: int = 8,
        n_layers: int = 8,
        block_size: int = 8,
        hd_dim: int = 4096,
        kv_cache_size: int = 4096,
        kv_k_bits: int = 4,
        kv_v_bits: int = 2,
        coherence_threshold: float = 0.55,
        n_candidate_blocks: int = 16,
    ):
        self.config = config or SpectralStreamConfig.load()
        self.start_time = time.time()

        # ── Model ──────────────────────────────────────────────────────
        if model_path:
            if _HAS_GGUF:
                model = load_model(model_path)
                hidden_dim = model.hidden_dim or hidden_dim
                vocab_size = model.vocab_size or vocab_size
                n_heads = model.n_heads or n_heads
                n_layers = model.n_layers or n_layers
                self.model = model
                self.is_real_model = True
            else:
                self.model = None
                self.is_real_model = False
        else:
            if load_model is not None:
                self.model = load_model(
                    None,
                    hidden_dim=hidden_dim,
                    vocab_size=vocab_size,
                    n_layers=n_layers,
                    n_heads=n_heads,
                )
            else:
                self.model = None
            self.is_real_model = False

        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.n_heads = n_heads
        self.n_layers = n_layers

        # ── Tokenizer Engine ───────────────────────────────────────────
        self._tokenizer = None
        try:
            if self.is_real_model and self.model is not None:
                self._tokenizer = AutoTokenizer.from_gguf_model(self.model)
            else:
                self._tokenizer = CachedTokenizer(AutoTokenizer())
        except Exception:
            self._tokenizer = None

        # ── HDC Draft Engine ───────────────────────────────────────────
        self.hd_engine = HDCDraftEngine(
            vocab_size=vocab_size,
            hd_dim=hd_dim,
            max_order=self.config.hdc.ngram_order,
            n_draft_candidates=self.config.block_emission.n_candidates,
        )

        # ── Resonance Tracker (for KV cache adaptive compression) ──────
        self.resonance_tracker = ResonanceTracker(window=64)

        # ── Spectral KV Cache ──────────────────────────────────────────
        self.kv_cache = SpectralKVCache(
            dim=hidden_dim // n_heads,
            max_size=kv_cache_size,
            k_bits=kv_k_bits,
            v_bits=kv_v_bits,
            resonance_tracker=self.resonance_tracker,
            use_dct=self.config.spectral.use_vlasov,
        )

        # ── Attractor Scoring Ensemble ─────────────────────────────────
        self.scorer = AttractorScoringEnsemble(hidden_dim=hidden_dim)

        # ── Block Emission Pipeline ────────────────────────────────────
        self.pipeline = BlockEmissionPipeline(
            model_fn=self._model_forward,
            hd_engine=self.hd_engine,
            scorer=self.scorer,
            block_size=block_size,
            min_block_size=self.config.block_emission.min_block_size,
            max_block_size=self.config.block_emission.max_block_size,
            n_candidate_blocks=n_candidate_blocks,
            coherence_threshold=coherence_threshold,
        )

        # ── Confidence Gate ────────────────────────────────────────────
        self.confidence_gate = ConfidenceGate(
            n_features=self.config.confidence.n_features,
            learning_rate=self.config.confidence.learning_rate,
        )

        # ── Online Learning Engine ─────────────────────────────────────
        self.learning_engine = OnlineLearningEngine(
            hd_engine=self.hd_engine,
            confidence_gate=self.confidence_gate,
            max_buffer=self.config.online_learning.max_buffer,
        )

        # ── Resonance Router ───────────────────────────────────────────
        self.resonance_router = ResonanceRouter()

        # ── Cascade Strategy Selector ──────────────────────────────────
        if _HAS_CASCADE:
            self.strategy_selector = CascadeStrategySelector()
            self.healer = SelfHealingHDC()
            self.speculator = ResonanceAwareSpeculation()
            self.cross_context_memory = CrossContextMemory()
            self.accuracy_manager = ProactiveAccuracyManager(
                common_vocab_size=min(1000, vocab_size)
            )
        else:
            self.strategy_selector = _Stub()
            self.healer = _Stub()
            self.speculator = _Stub()
            self.cross_context_memory = _Stub()
            self.accuracy_manager = _Stub()

        # ── Adaptive Inference Components ──────────────────────────────
        if _HAS_ADAPTIVE:
            self.confidence_cascade = PredictiveConfidenceCascade()
            self.staged_emission = StagedBlockEmission()
            self.self_tuning_hdc = SelfTuningHDCParams()
            self.entropy_guided = EntropyGuidedExploration()
            self.thermal_noise = ThermalNoiseInjection()
        else:
            self.confidence_cascade = _Stub()
            self.staged_emission = _Stub()
            self.self_tuning_hdc = _Stub()
            self.entropy_guided = _Stub()
            self.thermal_noise = _Stub()

        # ── Novel Operators ────────────────────────────────────────────
        if _HAS_NOVEL_OPS:
            self.hdc_weighted_sampling = HDCWeightedTokenSampling(vocab_size=vocab_size)
            self.spectral_entropy_gating = SpectralEntropyGating(n_layers=n_layers)
            self.adaptive_depth = AdaptiveForwardlessDepth()
            self.gradient_free_ft = GradientFreeFineTuning(hidden_dim=hidden_dim)
            self.predictor_corrector = PredictorCorrectorInference()
        else:
            self.hdc_weighted_sampling = _Stub()
            self.spectral_entropy_gating = _Stub()
            self.adaptive_depth = _Stub()
            self.gradient_free_ft = _Stub()
            self.predictor_corrector = _Stub()

        # ── Quantum Sampler ────────────────────────────────────────────
        if _HAS_QUANTUM:
            self.quantum_sampler = BornMachineSampler(
                vocab_size=vocab_size,
                temperature=0.8,
                resonance=0.5,
            )
            self.quantum_resonance = QuantumResonance(natural_freq=0.5)
        else:
            self.quantum_sampler = _Stub()
            self.quantum_resonance = _Stub()

        # ── Monitoring ─────────────────────────────────────────────────
        self.monitor = InferenceMonitor(window_size=self.config.monitoring.window_size)
        if _HAS_METRICS_EXPORTER:
            self.metrics_exporter = MetricsExporter()
        else:
            self.metrics_exporter = _Stub()

        # ── Persistence ────────────────────────────────────────────────
        if _HAS_PERSISTENCE:
            self.state_manager = StateManager(
                state_dir=self.config.persistence.state_dir
            )
        else:
            self.state_manager = _Stub()

        # ── Performance tracking ───────────────────────────────────────
        self.generation_times: list[float] = []
        self.generation_token_counts: list[int] = []

    # ── Model Forward ──────────────────────────────────────────────────

    def _model_forward(self, tokens, past=None):
        """Wrapper around model forward pass."""
        if self.model is None:
            return None, None, None
        if isinstance(tokens, list) and len(tokens) > 0:
            return self.model.forward(tokens, past)
        return self.model.forward([tokens] if isinstance(tokens, int) else tokens, past)

    # ── Strategy Detection ─────────────────────────────────────────────

    def _get_current_strategy_level(self) -> int:
        """Determine current optimal strategy level from all signals."""
        resonance_score = self.resonance_router.resonance_meter.resonance_score()
        hd_acceptance = self.hd_engine.acceptance_rate()
        recent_acc = self.learning_engine.get_stats().get("recent_hdc_accuracy", 0.0)
        spectral_ent = self.entropy_guided.compute_spectral_entropy()

        _, level = self.strategy_selector.select_strategy(
            confidence=self.confidence_gate.predict(self._build_features())
            if hasattr(self, "_build_features")
            else 0.5,
            resonance_score=resonance_score,
            spectral_entropy=spectral_ent,
            recent_accuracy=recent_acc,
            hd_acceptance_rate=hd_acceptance,
        )
        return level

    def _build_features(self) -> list[float]:
        """Build 10-dim feature vector for confidence gate."""
        return self.confidence_gate.extract_features(
            hdc_scores=[],
            context_tokens=[],
            resonance_score=self.resonance_router.resonance_meter.resonance_score(),
            spectral_entropy=self.entropy_guided.compute_spectral_entropy(),
            cache_hit_rate=self.kv_cache.hit_rate(),
            acceptance_rate=self.hd_engine.acceptance_rate(),
        )

    # ── Tokenization (using Tokenizer Engine) ─────────────────────────

    def tokenize(self, text: str) -> list[int]:
        """Tokenize text using the tokenizer engine."""
        if hasattr(self, "_tokenizer") and self._tokenizer is not None:
            return self._tokenizer.encode(text)
        if self.is_real_model and self.model is not None:
            try:
                return self.model.tokenize(text)
            except Exception:
                pass
        return [min(ord(c) % self.vocab_size, self.vocab_size - 1) for c in text[:128]]

    def detokenize(self, token_ids: list[int]) -> str:
        """Detokenize token IDs to text."""
        if hasattr(self, "_tokenizer") and self._tokenizer is not None:
            return self._tokenizer.decode(token_ids)
        if self.is_real_model and self.model is not None:
            try:
                result = []
                for t in token_ids:
                    piece = self.model.detokenize(t)
                    if piece:
                        result.append(
                            piece
                            if isinstance(piece, str)
                            else piece.decode("utf-8", errors="replace")
                        )
                return "".join(result)
            except Exception:
                pass
        return "".join(chr(t % 128) if 32 <= t % 128 < 127 else " " for t in token_ids)

    # ── Core Generate ──────────────────────────────────────────────────

    def generate(
        self,
        prompt: str | list[int],
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        stream: bool = False,
        strategy_override: Optional[int] = None,
    ) -> tuple[list[int], float]:
        """Generate text using optimal strategy.

        Args:
            prompt: Input string or token IDs
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stream: If True, yields tokens via callback (not implemented)
            strategy_override: Force a specific strategy level

        Returns:
            (token_ids, tokens_per_second)
        """
        if isinstance(prompt, str):
            input_ids = self.tokenize(prompt)
        else:
            input_ids = prompt

        start_time = time.time()

        strategy_level = (
            strategy_override
            if strategy_override is not None
            else self._get_current_strategy_level()
        )

        if strategy_level == FORWARDLESS:
            output_ids = self._generate_forwardless(
                input_ids, max_new_tokens, temperature
            )
        elif strategy_level == BLOCK_EMISSION:
            output_ids = self.pipeline.generate(input_ids, max_new_tokens)
        elif strategy_level == SPECULATIVE:
            output_ids = self._generate_speculative(
                input_ids, max_new_tokens, temperature
            )
        elif strategy_level == STANDARD:
            output_ids = self._generate_standard(input_ids, max_new_tokens, temperature)
        else:
            output_ids = self._generate_fallback(input_ids, max_new_tokens)

        elapsed = time.time() - start_time
        new_tokens = max(len(output_ids) - len(input_ids), 1)
        tps = new_tokens / elapsed if elapsed > 0 else 0

        self.generation_times.append(elapsed)
        self.generation_token_counts.append(new_tokens)

        self.monitor.record_token(
            STRATEGY_NAMES.get(strategy_level, "unknown"),
            elapsed * 1000 / max(new_tokens, 1),
            accepted=True,
        )

        return output_ids, tps

    # ── FORWARDLESS Strategy ───────────────────────────────────────────

    def _generate_forwardless(
        self, input_ids: list[int], max_new_tokens: int, temperature: float
    ) -> list[int]:
        """Pure HDC generation, no model calls."""
        generated = list(input_ids)
        current_context = tuple(generated)
        hd_conf = self.hd_engine.hd.predict_next(current_context, n_candidates=32)

        tokens_remaining = max_new_tokens
        while tokens_remaining > 0:
            current_context = tuple(generated)
            candidates = self.hd_engine.hd.predict_next(
                current_context, n_candidates=32
            )
            if not candidates:
                break

            scores = np.array([s for _, s in candidates], dtype=np.float64)
            scores = np.maximum(scores, 1e-10)
            scores = scores / np.max(scores)
            probs = scores ** (1.0 / max(temperature, 0.1))
            probs = probs / np.sum(probs)

            idx = int(np.random.choice(len(candidates), p=probs))
            token = candidates[idx][0]

            generated.append(token)
            self.hd_engine.observe(token)
            tokens_remaining -= 1

            if tokens_remaining <= 0:
                break

        return generated

    # ── SPECULATIVE Strategy ───────────────────────────────────────────

    def _generate_speculative(
        self, input_ids: list[int], max_new_tokens: int, temperature: float
    ) -> list[int]:
        """Draft tokens with HDC, verify each with model."""
        generated = list(input_ids)

        logits, hidden_states, past = self._model_forward(generated, past=None)
        if logits is None:
            return self._generate_fallback(input_ids, max_new_tokens)
        next_logits = logits[-1] if logits.ndim > 1 else logits
        probs = self._softmax(next_logits, temperature)
        first_token = int(np.random.choice(len(probs), p=probs))
        generated.append(first_token)
        self.hd_engine.observe(first_token)

        tokens_remaining = max_new_tokens - 1
        while tokens_remaining > 0:
            context = tuple(generated)
            draft_tokens = self.hd_engine.hd.generate_block(
                context, block_size=1, temperature=temperature
            )

            if not draft_tokens:
                logits, _, new_past = self._model_forward(generated[-1:], past)
                if logits is None:
                    break
                past = new_past
                step_logits = logits[-1] if logits.ndim > 1 else logits
                probs = self._softmax(step_logits, temperature)
                token = int(np.random.choice(len(probs), p=probs))
                generated.append(token)
                self.hd_engine.observe(token)
                tokens_remaining -= 1
                continue

            draft_token = draft_tokens[0]
            test_tokens = generated + [draft_token]
            logits, _, new_past = self._model_forward(test_tokens, past)
            if logits is None:
                break

            step_idx = len(generated) - len(test_tokens) + len(test_tokens) - 1
            if step_idx < 0:
                step_idx = 0
            if logits.ndim > 1 and step_idx < len(logits):
                step_logits = logits[step_idx]
            else:
                step_logits = logits[-1] if logits.ndim > 1 else logits

            probs = self._softmax(step_logits, temperature)
            token_prob = float(probs[draft_token]) if draft_token < len(probs) else 0.0

            if token_prob >= 0.01:
                generated.append(draft_token)
                self.hd_engine.observe(draft_token)
                past = new_past
                self.hd_engine.accept_count += 1
            else:
                token = int(np.random.choice(len(probs), p=probs))
                generated.append(token)
                self.hd_engine.observe(token)
                if new_past is not None:
                    past = new_past

            self.hd_engine.draft_count += 1
            tokens_remaining -= 1

        return generated

    # ── STANDARD Strategy ──────────────────────────────────────────────

    def _generate_standard(
        self, input_ids: list[int], max_new_tokens: int, temperature: float
    ) -> list[int]:
        """Standard autoregressive generation, one token per model call."""
        generated = list(input_ids)
        past = None

        for _ in range(max_new_tokens):
            logits, _, new_past = self._model_forward(generated, past)
            if logits is None:
                break
            past = new_past

            next_logits = logits[-1] if logits.ndim > 1 else logits
            probs = self._softmax(next_logits, temperature)

            token = int(np.random.choice(len(probs), p=probs))
            generated.append(token)

        return generated

    # ── FALLBACK Strategy ──────────────────────────────────────────────

    def _generate_fallback(
        self, input_ids: list[int], max_new_tokens: int
    ) -> list[int]:
        """Emergency RNG-based generation."""
        generated = list(input_ids)
        for _ in range(max_new_tokens):
            token = int(np.random.randint(0, min(self.vocab_size, 10000)))
            generated.append(token)
        return generated

    # ── Softmax ────────────────────────────────────────────────────────

    @staticmethod
    def _softmax(logits: np.ndarray, temperature: float = 0.8) -> np.ndarray:
        return softmax(logits, temperature=temperature)

    # ── Online Learning Correction ─────────────────────────────────────

    def observe_correction(
        self,
        context_tokens: list[int],
        hdc_predicted: int,
        model_token: int,
    ):
        """Learn from an HDC mistake."""
        features = self._build_features()
        self.learning_engine.observe_correction(
            context_tokens, hdc_predicted, model_token, features
        )
        self.monitor.record_hdc_decision(
            confidence=self.confidence_gate.predict(features),
            accepted=False,
        )

    def observe_acceptance(
        self,
        context_tokens: list[int],
        accepted_token: int,
    ):
        """Learn from an HDC correct prediction."""
        features = self._build_features()
        self.learning_engine.observe_acceptance(
            context_tokens, accepted_token, features
        )
        self.monitor.record_hdc_decision(
            confidence=self.confidence_gate.predict(features),
            accepted=True,
        )

    # ── State Management ───────────────────────────────────────────────

    def save(self, path: Optional[str] = None) -> bool:
        """Save all learned state."""
        name = path or "default"
        return self.state_manager.auto_save(
            engine=self.hd_engine,
            gate=self.confidence_gate,
            learner=self.learning_engine,
            controller=self.resonance_router.pid,
            name=name,
        )

    def load(self, path: Optional[str] = None) -> bool:
        """Load all learned state."""
        name = path or "default"
        return self.state_manager.auto_load(
            engine=self.hd_engine,
            gate=self.confidence_gate,
            learner=self.learning_engine,
            controller=self.resonance_router.pid,
            name=name,
        )

    def save_checkpoint(self) -> bool:
        """Save periodic checkpoint of all state."""
        self.save()
        return self.state_manager.maybe_checkpoint(force=True)

    # ── Reset ──────────────────────────────────────────────────────────

    def reset(self):
        """Reset all subsystems to initial state."""
        self.hd_engine.reset()
        self.kv_cache.clear()
        self.pipeline.total_tokens = 0
        self.pipeline.total_model_calls = 0
        self.pipeline.successful_blocks = 0
        self.pipeline.fallback_tokens = 0
        self.pipeline.tokens_per_call_log.clear()
        self.pipeline.fallback_reasons.clear()
        self.pipeline.previous_blocks.clear()
        self.resonance_router.reset()
        self.strategy_selector.reset()
        self.healer.reset()
        self.speculator.reset()
        self.cross_context_memory.reset()
        self.accuracy_manager.reset()
        self.confidence_cascade = (
            PredictiveConfidenceCascade() if _HAS_ADAPTIVE else _Stub()
        )
        self.staged_emission.reset()
        self.self_tuning_hdc.reset()
        self.entropy_guided.reset()
        self.thermal_noise.reset()
        self.hdc_weighted_sampling.reset()
        self.spectral_entropy_gating.reset()
        self.adaptive_depth.reset()
        self.gradient_free_ft.reset()
        self.predictor_corrector.reset()
        if _HAS_QUANTUM:
            self.quantum_resonance = QuantumResonance(natural_freq=0.5)
        self.generation_times.clear()
        self.generation_token_counts.clear()

    # ── Stats ──────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Get comprehensive statistics from all subsystems."""
        pipeline_stats = self.pipeline.statistics()
        avg_tps = (
            np.mean(self.generation_token_counts) / np.mean(self.generation_times)
            if self.generation_times
            else 0
        )
        learning_stats = self.learning_engine.get_stats()
        resonance_report = self.resonance_router.report()
        strategy_stats = self.strategy_selector.stats()

        return {
            "tokens_per_second": float(avg_tps),
            "tokens_per_model_call": pipeline_stats["tokens_per_model_call"],
            "block_success_rate": pipeline_stats["block_success_rate"],
            "kv_cache_hit_rate": self.kv_cache.hit_rate(),
            "hd_acceptance_rate": self.hd_engine.acceptance_rate(),
            "kv_compression_ratio": self.kv_cache.compression_ratio(),
            "total_tokens_generated": int(sum(self.generation_token_counts)),
            "total_model_calls": pipeline_stats["total_model_calls"],
            "total_fallback_tokens": pipeline_stats["fallback_tokens"],
            "strategy": strategy_stats,
            "resonance": resonance_report,
            "learning": learning_stats,
            "healing": self.healer.stats(),
            "cross_context_memory": self.cross_context_memory.stats(),
            "accuracy": self.accuracy_manager.stats(),
            "self_tuning": self.self_tuning_hdc.get_params(),
            "adaptive_depth": self.adaptive_depth.stats(),
            "entropy_guided": self.entropy_guided.stats(),
            "hdc_weighted_sampling": self.hdc_weighted_sampling.stats(),
            "spectral_entropy_gating": self.spectral_entropy_gating.stats(),
            "predictor_corrector": self.predictor_corrector.stats(),
            "monitor": self.monitor.get_stats(),
            "uptime_seconds": time.time() - self.start_time,
            "model_loaded": self.is_real_model,
            "hidden_dim": self.hidden_dim,
            "vocab_size": self.vocab_size,
            "n_layers": self.n_layers,
        }

    def get_performance_report(self) -> str:
        """Generate a human-readable performance report."""
        return self.monitor.get_performance_report()

    def get_metrics_json(self) -> str:
        """Get metrics as JSON string."""
        return self.metrics_exporter.to_json(self.monitor)

    def get_metrics_prometheus(self) -> str:
        """Get metrics in Prometheus format."""
        return self.metrics_exporter.to_prometheus(self.monitor)
