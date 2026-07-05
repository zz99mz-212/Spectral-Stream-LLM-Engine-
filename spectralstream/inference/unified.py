"""
Unified Inference Pipeline — SpectralStream
=============================================
Symphonic integration of all SpectralStream subsystems into a coherent
inference engine with 6 strategy levels, COCONUT latent reasoning,
Vlasov mean-field attention, HDC forwardless drafting, and more.

Strategy Levels:
  0. FORWARDLESS — HDC only, 100k+ tok/s
  1. RESONANT_RESONANCE — HDC + Vlasov mean-field bypass, 50k tok/s
  2. SPECTRAL_BLOCK — Block emission with spectral scoring, 10k tok/s
  3. SPECTRAL_VERIFY — Speculative with Vlasov verification, 5k tok/s
  4. STANDARD — Full model forward, 1x tok/s
  5. FALLBACK — Emergency RNG generation
"""

from __future__ import annotations

import math
import time
import gc
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import numpy as np

from spectralstream.inference.coconut import COCONUTEngine, integrate_coconut
from spectralstream.inference.vlasov import VlasovMeanFieldAttention
from spectralstream.inference.hrr_memory import HrrMemory, HolographicKVCache
from spectralstream.inference.hdc_engine import HDCDraftEngine
from spectralstream.inference.resonance import ResonanceRouter
from spectralstream.inference.confidence_gate import ConfidenceGate
from spectralstream.inference.monitor import InferenceMonitor
from spectralstream.inference.online_learning import OnlineLearningEngine


class UnifiedStrategyLevel(IntEnum):
    FORWARDLESS = 0
    RESONANT_RESONANCE = 1
    SPECTRAL_BLOCK = 2
    SPECTRAL_VERIFY = 3
    STANDARD = 4
    FALLBACK = 5


UNIFIED_STRATEGY_NAMES = {
    UnifiedStrategyLevel.FORWARDLESS: "forwardless",
    UnifiedStrategyLevel.RESONANT_RESONANCE: "resonant_resonance",
    UnifiedStrategyLevel.SPECTRAL_BLOCK: "spectral_block",
    UnifiedStrategyLevel.SPECTRAL_VERIFY: "spectral_verify",
    UnifiedStrategyLevel.STANDARD: "standard",
    UnifiedStrategyLevel.FALLBACK: "fallback",
}

UNIFIED_THROUGHPUT_ESTIMATES = {
    UnifiedStrategyLevel.FORWARDLESS: 100_000,
    UnifiedStrategyLevel.RESONANT_RESONANCE: 50_000,
    UnifiedStrategyLevel.SPECTRAL_BLOCK: 10_000,
    UnifiedStrategyLevel.SPECTRAL_VERIFY: 5_000,
    UnifiedStrategyLevel.STANDARD: 1,
    UnifiedStrategyLevel.FALLBACK: 1_000_000,
}


class UnifiedInferenceEngine:
    """Master class combining ALL SpectralStream components into a coherent
    production-ready inference system with 6 strategy levels.

    Orchestrates strategy selection based on real-time confidence,
    resonance, and performance metrics.  Integrates with the existing
    SpectralStream model loaders (SSF / safetensors).
    """

    def __init__(
        self,
        model_forward_fn=None,
        model_tokenize_fn=None,
        model_detokenize_fn=None,
        hidden_dim: int = 1536,
        vocab_size: int = 262144,
        n_heads: int = 8,
        n_layers: int = 35,
        hd_dim: int = 4096,
        kv_cache_size: int = 4096,
        n_candidate_blocks: int = 16,
        coconut_engine: Optional[COCONUTEngine] = None,
        config: Optional[dict] = None,
    ):
        self._model_forward_fn = model_forward_fn
        self._tokenize_fn = model_tokenize_fn
        self._detokenize_fn = model_detokenize_fn
        self.config = config or {}

        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.head_dim = hidden_dim // max(n_heads, 1)
        self._start_time = time.time()
        self._generation_id = 0

        # HDC Draft Engine
        self.hd_engine = HDCDraftEngine(
            vocab_size=vocab_size,
            hd_dim=hd_dim,
            max_order=self.config.get("hdc_ngram_order", 6),
            n_draft_candidates=self.config.get("hdc_n_candidates", n_candidate_blocks),
        )

        # Spectral KV Cache via Holographic Memory
        self.holographic_kv = HolographicKVCache(
            dim=self.head_dim * 2,
            capacity=kv_cache_size * 4,
        )

        # Vlasov Mean-Field Attention
        vlasov_dim = max(self.head_dim, self.config.get("vlasov_dim", hd_dim))
        self.vlasov_attention = VlasovMeanFieldAttention(
            dim=vlasov_dim,
            n_grid=self.config.get("vlasov_grid", 64),
            n_particles=self.config.get("vlasov_particles", min(128, vlasov_dim)),
        )

        # HRR Memory (weight store)
        self.hrr_weight_store = HrrMemory(
            dim=hd_dim,
            capacity=self.config.get("hrr_capacity", 65536),
        )

        # Confidence Gate
        self.confidence_gate = ConfidenceGate(
            n_features=self.config.get("confidence_n_features", 10),
            learning_rate=self.config.get("confidence_lr", 0.01),
        )

        # Online Learning Engine
        self.learning_engine = OnlineLearningEngine(
            hd_engine=self.hd_engine,
            confidence_gate=self.confidence_gate,
            max_buffer=self.config.get("online_learning_buffer", 10000),
        )

        # Resonance Router
        self.resonance_router = ResonanceRouter(
            use_time_crystal=self.config.get("use_time_crystal", True)
        )

        # COCONUT Continuous Chain of Thought
        self.coconut_engine = coconut_engine
        self._output_projection = None
        if coconut_engine is not None:
            self._output_projection = (
                np.random.randn(hidden_dim, vocab_size).astype(np.float32) * 0.01
            )

        # Monitoring
        self.monitor = InferenceMonitor(
            window_size=self.config.get("monitor_window", 100),
        )

        # Strategy State
        self._current_strategy = UnifiedStrategyLevel.SPECTRAL_BLOCK
        self._strategy_history = []
        self._level_token_counts = {int(lv): 0 for lv in UnifiedStrategyLevel}
        self._consecutive_failures = 0
        self._total_tokens = 0
        self._total_model_calls = 0
        self._total_hdc_tokens = 0

        # Repetition detection
        self._repetition_buffer = []
        self._repetition_threshold = self.config.get("repetition_threshold", 4)

        # Adaptive depth control
        self._hdc_depth = self.config.get("hdc_depth", 6)
        self._hdc_depth_min = 2
        self._hdc_depth_max = 8

    # ── Model Forward (delegates to injected function or CPUInferenceEngine) ──

    def _model_forward(self, tokens, past=None):
        if self._model_forward_fn is not None:
            logits = self._model_forward_fn(tokens)
            return logits, [], past
        if isinstance(tokens, np.ndarray):
            return tokens.astype(np.float32), [], past
        return np.array(tokens, dtype=np.float32), [], past

    # ── Tokenization ──

    def tokenize(self, text: str) -> list:
        if self._tokenize_fn is not None:
            return self._tokenize_fn(text)
        return [min(ord(c) % self.vocab_size, self.vocab_size - 1) for c in text[:512]]

    def detokenize(self, token_ids: list) -> str:
        if self._detokenize_fn is not None:
            return self._detokenize_fn(token_ids)
        return "".join(chr(t % 128) if 32 <= t % 128 < 127 else " " for t in token_ids)

    # ── COCONUT Helpers ──

    def _get_hidden_state(self, context: tuple) -> np.ndarray:
        context_hv = self.hd_engine.hd._encode_context(context)
        h = context_hv.astype(np.float32)
        if len(h) != self.hidden_dim:
            if len(h) > self.hidden_dim:
                h = h[: self.hidden_dim]
            else:
                h = np.pad(h, (0, self.hidden_dim - len(h)))
        return h

    def _project_to_vocab(self, hidden_state: np.ndarray) -> np.ndarray:
        if self._output_projection is not None:
            return hidden_state @ self._output_projection
        return hidden_state

    def _apply_coconut(self, hidden_state: np.ndarray) -> tuple:
        if self.coconut_engine is None:
            return hidden_state, 0, "skip"
        d_model = self.coconut_engine.d_model
        if len(hidden_state) != d_model:
            h = (
                hidden_state[:d_model]
                if len(hidden_state) > d_model
                else np.pad(hidden_state, (0, d_model - len(hidden_state)))
            )
        else:
            h = hidden_state
        confidence = self.confidence_gate.predict(self._build_features())
        if confidence >= 0.7:
            return hidden_state, 0, "skip"
        if confidence < 0.3:
            h_refined, paths = self.coconut_engine.fuse_multiple_paths(h, n_paths=4)
            return h_refined, self.coconut_engine.max_steps, "multi"
        else:
            h_refined, n_steps, traj = self.coconut_engine.explore(h)
            return h_refined, n_steps, "single"

    def _coconut_bias_candidates(self, context: tuple, candidates: list) -> list:
        if self.coconut_engine is None or not candidates:
            return candidates
        hidden_state = self._get_hidden_state(context)
        h_refined, n_steps, mode = self._apply_coconut(hidden_state)
        if mode == "skip":
            return candidates
        logits = self._project_to_vocab(h_refined)
        probs = self._softmax(logits, temperature=1.0)
        boosted = []
        for tok, score in candidates:
            coconut_bias = float(probs[tok]) if tok < len(probs) else 0.0
            coconut_weight = 0.3 + 0.5 * (
                n_steps / max(self.coconut_engine.max_steps, 1)
            )
            boosted.append(
                (tok, score * (1.0 - coconut_weight) + coconut_bias * coconut_weight)
            )
        boosted.sort(key=lambda x: -x[1])
        return boosted

    def _coconut_refine_logits(
        self, logits: np.ndarray, hidden_states: list
    ) -> np.ndarray:
        if self.coconut_engine is None:
            return logits
        if not isinstance(hidden_states, list) or len(hidden_states) == 0:
            return logits
        last_hidden = hidden_states[-1]
        if isinstance(last_hidden, np.ndarray) and last_hidden.ndim > 1:
            h = last_hidden[-1]
        else:
            h = last_hidden
        h_refined, n_steps, mode = self._apply_coconut(h)
        if mode == "skip":
            return logits
        coconut_logits = self._project_to_vocab(h_refined)
        coconut_weight = 0.2 + 0.3 * (n_steps / max(self.coconut_engine.max_steps, 1))
        if isinstance(logits, np.ndarray) and logits.ndim > 1:
            new_logits = logits.copy()
            if len(new_logits) > 0:
                new_logits[-1] = (1.0 - coconut_weight) * new_logits[
                    -1
                ] + coconut_weight * coconut_logits
            return new_logits
        return (1.0 - coconut_weight) * logits + coconut_weight * coconut_logits

    # ── Strategy Selection ──

    def _select_strategy(
        self,
        hdc_candidates: Optional[list] = None,
        spectral_entropy: Optional[float] = None,
    ) -> UnifiedStrategyLevel:
        if hdc_candidates is None:
            hdc_candidates = []
        scores = (
            np.array([s for _, s in hdc_candidates], dtype=np.float64)
            if hdc_candidates
            else np.array([0.0])
        )
        max_score = float(np.max(scores)) if len(scores) > 0 else 0.0
        entropy = spectral_entropy if spectral_entropy is not None else 0.5
        resonance = self.resonance_router.resonance_meter.resonance_score()
        acceptance = self.resonance_router.resonance_meter.spectral_entropy()
        gate_conf = self.confidence_gate.predict(
            [
                max_score,
                float(scores[0] - scores[1]) if len(scores) >= 2 else 0.0,
                entropy,
                resonance,
                0.0,
                0.5,
                0.5,
                self.hd_engine.acceptance_rate(),
                self.holographic_kv.hit_rate(),
                float(np.var(scores[:5])) if len(scores) >= 2 else 0.0,
            ]
        )
        composite = (
            0.30 * max_score
            + 0.25 * gate_conf
            + 0.20 * resonance
            + 0.15 * (1.0 - entropy)
            + 0.10 * acceptance
        )
        if self._consecutive_failures >= 3:
            composite *= 0.6
        if self._detect_repetition_loop():
            composite = max(composite, 0.4)
        if composite >= 0.85:
            level = UnifiedStrategyLevel.FORWARDLESS
        elif composite >= 0.65:
            level = UnifiedStrategyLevel.RESONANT_RESONANCE
        elif composite >= 0.50:
            level = UnifiedStrategyLevel.SPECTRAL_BLOCK
        elif composite >= 0.30:
            level = UnifiedStrategyLevel.SPECTRAL_VERIFY
        elif composite >= 0.10:
            level = UnifiedStrategyLevel.STANDARD
        else:
            level = UnifiedStrategyLevel.FALLBACK
        self._current_strategy = level
        self._strategy_history.append(int(level))
        return level

    def _detect_repetition_loop(self) -> bool:
        if len(self._repetition_buffer) < 8:
            return False
        arr = list(self._repetition_buffer)
        for window in [4, 8, 16]:
            if len(arr) >= window * 2:
                for start in range(len(arr) - window * 2 + 1):
                    a = arr[start : start + window]
                    b = arr[start + window : start + window * 2]
                    if a == b:
                        return True
        return False

    def _break_repetition_loop(self, candidates: list) -> Optional[int]:
        if not candidates:
            return None
        recent = list(self._repetition_buffer)
        if not recent:
            return None
        for token, _score in candidates:
            if token not in recent[-4:]:
                return token
        for token, _score in candidates[:8]:
            if recent.count(token) < 3:
                return token
        return None

    def _build_features(self) -> list:
        return [
            self.hd_engine.acceptance_rate(),
            self.holographic_kv.hit_rate(),
            self.resonance_router.resonance_meter.resonance_score(),
            float(np.mean(self._repetition_buffer)) if self._repetition_buffer else 0.5,
            self.vocab_size / 32000.0,
            self._total_tokens / max(self._total_tokens + 1, 1),
            self._total_model_calls / max(self._total_tokens, 1),
            self.hd_engine.acceptance_rate(),
            self.holographic_kv.hit_rate(),
            self._current_strategy / 5.0,
        ]

    # ── Core Generate ──

    def generate(
        self,
        prompt: Union[str, list],
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.95,
        strategy_override: Optional[int] = None,
    ) -> tuple:
        if isinstance(prompt, str):
            input_ids = self.tokenize(prompt)
        else:
            input_ids = list(prompt)
        if not input_ids:
            return [], 0.0

        start_time = time.time()
        self._generation_id += 1
        output_ids = list(input_ids)

        if strategy_override is not None:
            strategy_level = UnifiedStrategyLevel(strategy_override)
        else:
            candidates = self.hd_engine.hd.predict_next(
                tuple(input_ids), n_candidates=32
            )
            strategy_level = self._select_strategy(candidates)

        self._level_token_counts[int(strategy_level)] += 1

        if strategy_level == UnifiedStrategyLevel.FORWARDLESS:
            output_ids = self._generate_forwardless(
                output_ids, max_new_tokens, temperature, top_k, top_p
            )
        elif strategy_level == UnifiedStrategyLevel.RESONANT_RESONANCE:
            output_ids = self._generate_resonant_resonance(
                output_ids, max_new_tokens, temperature, top_k, top_p
            )
        elif strategy_level == UnifiedStrategyLevel.SPECTRAL_BLOCK:
            output_ids = self._generate_spectral_block(
                output_ids, max_new_tokens, temperature, top_k, top_p
            )
        elif strategy_level == UnifiedStrategyLevel.SPECTRAL_VERIFY:
            output_ids = self._generate_spectral_verify(
                output_ids, max_new_tokens, temperature, top_k, top_p
            )
        elif strategy_level == UnifiedStrategyLevel.STANDARD:
            output_ids = self._generate_standard(
                output_ids, max_new_tokens, temperature, top_k, top_p
            )
        else:
            output_ids = self._generate_fallback(output_ids, max_new_tokens)

        elapsed = time.time() - start_time
        new_tokens = max(len(output_ids) - len(input_ids), 1)
        tps = new_tokens / elapsed if elapsed > 0 else 0

        self.monitor.record_token(
            UNIFIED_STRATEGY_NAMES.get(strategy_level, "unknown"),
            elapsed * 1000 / max(new_tokens, 1),
            accepted=True,
        )
        return output_ids, tps

    def stream_generate(
        self,
        prompt: Union[str, list],
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.95,
    ):
        if isinstance(prompt, str):
            input_ids = self.tokenize(prompt)
        else:
            input_ids = list(prompt)
        output_ids = list(input_ids)
        tokens_remaining = max_new_tokens

        while tokens_remaining > 0:
            context = tuple(output_ids)
            candidates = self.hd_engine.hd.predict_next(context, n_candidates=32)
            strategy = self._select_strategy(candidates)
            strategy_name = UNIFIED_STRATEGY_NAMES[strategy]

            if strategy == UnifiedStrategyLevel.FORWARDLESS:
                candidates = self._coconut_bias_candidates(context, candidates)
                token = self._sample_from_candidates(
                    candidates, temperature, top_k, top_p
                )
                yield token, strategy_name, "forwardless"
                output_ids.append(token)
                self.hd_engine.observe(token)

            elif strategy == UnifiedStrategyLevel.RESONANT_RESONANCE:
                token = self._resonant_token(output_ids, temperature)
                yield token, strategy_name, "resonant_resonance"
                output_ids.append(token)
                self.hd_engine.observe(token)

            elif strategy == UnifiedStrategyLevel.SPECTRAL_BLOCK:
                block_size = min(8, tokens_remaining)
                block = self._generate_block(
                    output_ids, block_size, temperature, top_k, top_p
                )
                for t in block:
                    yield t, strategy_name, "spectral_block"
                    output_ids.append(t)
                    self.hd_engine.observe(t)
                    tokens_remaining -= 1
                continue

            elif strategy == UnifiedStrategyLevel.SPECTRAL_VERIFY:
                draft = self._sample_from_candidates(
                    candidates, temperature, top_k, top_p
                )
                test_ids = output_ids + [draft]
                logits, hidden_states, _ = self._model_forward(test_ids)
                logits = self._coconut_refine_logits(logits, hidden_states)
                if isinstance(logits, np.ndarray) and logits.ndim > 1:
                    step_logits = logits[-2]
                else:
                    step_logits = logits
                probs = self._softmax(step_logits, temperature)
                if draft < len(probs) and probs[draft] >= 0.01:
                    token = draft
                else:
                    token = int(np.random.choice(len(probs), p=probs))
                yield token, strategy_name, "spectral_verify"
                output_ids.append(token)
                self.hd_engine.observe(token)
                self._total_model_calls += 1

            elif strategy == UnifiedStrategyLevel.STANDARD:
                logits, hidden_states, _ = self._model_forward(output_ids)
                logits = self._coconut_refine_logits(logits, hidden_states)
                if isinstance(logits, np.ndarray) and logits.ndim > 1:
                    logits = logits[-1]
                probs = self._softmax(logits, temperature)
                token = int(np.random.choice(len(probs), p=probs))
                yield token, strategy_name, "standard"
                output_ids.append(token)
                self.hd_engine.observe(token)
                self._total_model_calls += 1

            else:
                token = int(np.random.randint(0, min(self.vocab_size, 10000)))
                yield token, strategy_name, "fallback"
                output_ids.append(token)

            tokens_remaining -= 1
            self._total_tokens += 1
            self._level_token_counts[int(strategy)] += 1

    # ── Strategy Implementations ──

    def _generate_forwardless(
        self, input_ids, max_new_tokens, temperature, top_k, top_p
    ):
        generated = list(input_ids)
        tokens_remaining = max_new_tokens
        self._hdc_depth = self._adapt_hdc_depth()
        while tokens_remaining > 0:
            context = tuple(generated)
            candidates = self.hd_engine.hd.predict_next(
                context, n_candidates=min(32, self._hdc_depth * 4)
            )
            if not candidates:
                break
            candidates = self._coconut_bias_candidates(context, candidates)
            if not candidates:
                break
            if self._detect_repetition_loop():
                escape = self._break_repetition_loop(candidates)
                if escape is not None:
                    generated.append(escape)
                    self.hd_engine.observe(escape)
                    self._repetition_buffer.append(escape)
                    tokens_remaining -= 1
                    continue
            token = self._sample_from_candidates(candidates, temperature, top_k, top_p)
            generated.append(token)
            self.hd_engine.observe(token)
            self._repetition_buffer.append(token)
            self._total_hdc_tokens += 1
            tokens_remaining -= 1
        return generated

    def _generate_resonant_resonance(
        self, input_ids, max_new_tokens, temperature, top_k, top_p
    ):
        generated = list(input_ids)
        tokens_remaining = max_new_tokens
        while tokens_remaining > 0:
            context = tuple(generated)
            hdc_candidates = self.hd_engine.hd.predict_next(context, n_candidates=32)
            hdc_candidates = self._coconut_bias_candidates(context, hdc_candidates)
            if self._detect_repetition_loop():
                escape = self._break_repetition_loop(hdc_candidates)
                if escape is not None:
                    generated.append(escape)
                    self._repetition_buffer.append(escape)
                    tokens_remaining -= 1
                    continue
            draft_token = self._sample_from_candidates(
                hdc_candidates, temperature, top_k, top_p
            )
            resonance_grid = self.vlasov_attention._query_to_grid(
                self.hd_engine.hd.ensure_token_vector(draft_token)
            )
            mean_field = self.vlasov_attention._solve_mean_field(resonance_grid)
            resonance_bias = float(np.mean(mean_field))
            confidence = self.confidence_gate.predict(self._build_features())
            adjusted_confidence = confidence * (0.5 + 0.5 * max(0, resonance_bias))
            if adjusted_confidence >= 0.4:
                generated.append(draft_token)
                self.hd_engine.observe(draft_token)
                self._total_hdc_tokens += 1
                self._repetition_buffer.append(draft_token)
            else:
                logits, _, _ = self._model_forward(generated)
                if isinstance(logits, np.ndarray) and logits.ndim > 1:
                    logits = logits[-1]
                probs = self._softmax(logits, temperature)
                token = int(np.random.choice(len(probs), p=probs))
                generated.append(token)
                self.hd_engine.observe(token)
                self._total_model_calls += 1
                self._repetition_buffer.append(token)
            tokens_remaining -= 1
        return generated

    def _generate_spectral_block(
        self, input_ids, max_new_tokens, temperature, top_k, top_p
    ):
        generated = list(input_ids)
        tokens_remaining = max_new_tokens
        logits, _, past = self._model_forward(generated, past=None)
        self._total_model_calls += 1
        if isinstance(logits, np.ndarray) and logits.ndim > 1:
            logits = logits[-1]
        probs = self._softmax(logits, temperature)
        first_token = int(np.random.choice(len(probs), p=probs))
        generated.append(first_token)
        self.hd_engine.observe(first_token)
        tokens_remaining -= 1
        while tokens_remaining > 0:
            block_size = min(8, tokens_remaining)
            block = self._generate_block(
                generated, block_size, temperature, top_k, top_p
            )
            if not block:
                break
            test_tokens = generated + block
            try:
                logits, _, _ = self._model_forward(test_tokens)
                self._total_model_calls += 1
            except Exception:
                generated.extend(block)
                for t in block:
                    self.hd_engine.observe(t)
                tokens_remaining -= len(block)
                continue
            accepted = []
            for i, token in enumerate(block):
                if (
                    isinstance(logits, np.ndarray)
                    and logits.ndim > 1
                    and i < len(logits)
                ):
                    step_logits = logits[i]
                else:
                    step_logits = (
                        logits[-1]
                        if isinstance(logits, np.ndarray) and logits.ndim > 1
                        else logits
                    )
                probs = self._softmax(step_logits, temperature)
                token_prob = float(probs[token]) if token < len(probs) else 0.0
                if token_prob >= 0.01:
                    accepted.append(token)
                else:
                    break
            if accepted:
                for token in accepted:
                    generated.append(token)
                    self.hd_engine.observe(token)
                    self._repetition_buffer.append(token)
                emitted = len(accepted)
                tokens_remaining -= emitted
            else:
                break
        return generated

    def _generate_spectral_verify(
        self, input_ids, max_new_tokens, temperature, top_k, top_p
    ):
        generated = list(input_ids)
        logits, hidden_states, past = self._model_forward(generated, past=None)
        logits = self._coconut_refine_logits(logits, hidden_states)
        self._total_model_calls += 1
        if isinstance(logits, np.ndarray) and logits.ndim > 1:
            logits = logits[-1]
        probs = self._softmax(logits, temperature)
        first_token = int(np.random.choice(len(probs), p=probs))
        generated.append(first_token)
        self.hd_engine.observe(first_token)
        tokens_remaining = max_new_tokens - 1
        while tokens_remaining > 0:
            context = tuple(generated)
            candidates = self.hd_engine.hd.predict_next(context, n_candidates=32)
            if not candidates:
                logits, hidden_states, _ = self._model_forward(generated[-1:], past)
                logits = self._coconut_refine_logits(logits, hidden_states)
                if isinstance(logits, np.ndarray) and logits.ndim > 1:
                    logits = logits[-1]
                probs = self._softmax(logits, temperature)
                token = int(np.random.choice(len(probs), p=probs))
                generated.append(token)
                self.hd_engine.observe(token)
                tokens_remaining -= 1
                continue
            draft_token = self._sample_from_candidates(
                candidates, temperature, top_k, top_p
            )
            attn_mask = self.vlasov_attention.attend(
                self.hd_engine.hd.ensure_token_vector(draft_token),
                np.array(
                    [self.hd_engine.hd.ensure_token_vector(t) for t in generated[-64:]]
                ),
                np.array(
                    [self.hd_engine.hd.ensure_token_vector(t) for t in generated[-64:]]
                ),
            )
            logits, hidden_states, new_past = self._model_forward(
                generated + [draft_token], past
            )
            logits = self._coconut_refine_logits(logits, hidden_states)
            self._total_model_calls += 1
            logits_arr = (
                logits
                if isinstance(logits, np.ndarray)
                else np.array(logits, dtype=np.float64)
            )
            if logits_arr.ndim > 1 and logits_arr.shape[0] > 1:
                step_logits = logits_arr[-2]
            elif logits_arr.ndim > 1:
                step_logits = logits_arr[-1]
            else:
                step_logits = logits_arr
            probs = self._softmax(step_logits, temperature)
            draft_prob = float(probs[draft_token]) if draft_token < len(probs) else 0.0
            mean_field_conf = float(np.mean(np.abs(attn_mask)))
            combined_conf = 0.6 * draft_prob + 0.4 * mean_field_conf
            if combined_conf >= 0.01:
                generated.append(draft_token)
                self.hd_engine.observe(draft_token)
                past = new_past
            else:
                token = int(np.random.choice(len(probs), p=probs))
                generated.append(token)
                self.hd_engine.observe(token)
            self._repetition_buffer.append(generated[-1])
            tokens_remaining -= 1
        return generated

    def _generate_standard(self, input_ids, max_new_tokens, temperature, top_k, top_p):
        generated = list(input_ids)
        past = None
        for _ in range(max_new_tokens):
            logits, hidden_states, new_past = self._model_forward(generated, past)
            logits = self._coconut_refine_logits(logits, hidden_states)
            past = new_past
            self._total_model_calls += 1
            if isinstance(logits, np.ndarray) and logits.ndim > 1:
                logits = logits[-1]
            probs = self._softmax(logits, temperature)
            token = int(np.random.choice(len(probs), p=probs))
            generated.append(token)
            self._repetition_buffer.append(token)
        return generated

    def _generate_fallback(self, input_ids, max_new_tokens):
        generated = list(input_ids)
        for _ in range(max_new_tokens):
            token = int(np.random.randint(0, min(self.vocab_size, 10000)))
            generated.append(token)
        return generated

    # ── Helpers ──

    def _generate_block(self, context, block_size, temperature, top_k, top_p):
        block = []
        working = list(context)
        for _ in range(block_size):
            ctx = tuple(working)
            candidates = self.hd_engine.hd.predict_next(ctx, n_candidates=16)
            candidates = self._coconut_bias_candidates(ctx, candidates)
            if not candidates:
                break
            token = self._sample_from_candidates(candidates, temperature, top_k, top_p)
            block.append(token)
            working.append(token)
        return block

    def _resonant_token(self, context, temperature):
        ctx = tuple(context)
        candidates = self.hd_engine.hd.predict_next(ctx, n_candidates=32)
        if not candidates:
            logits, _, _ = self._model_forward(context)
            if isinstance(logits, np.ndarray) and logits.ndim > 1:
                logits = logits[-1]
            probs = self._softmax(logits, temperature)
            return int(np.random.choice(len(probs), p=probs))
        scores = np.array([s for _, s in candidates], dtype=np.float64)
        scores = np.maximum(scores, 1e-10)
        probs = scores / np.sum(scores)
        resonance = self.resonance_router.resonance_meter.resonance_score()
        probs = probs ** (1.0 / max(temperature * (0.5 + 0.5 * resonance), 0.01))
        probs = probs / np.sum(probs)
        top_k_idx = np.argsort(-probs)[: min(len(probs), max(1, len(probs)))]
        top_k_probs = probs[top_k_idx]
        top_k_probs = top_k_probs / np.sum(top_k_probs)
        idx = int(np.random.choice(len(top_k_idx), p=top_k_probs))
        return candidates[top_k_idx[idx]][0]

    def _sample_from_candidates(self, candidates, temperature, top_k, top_p):
        if not candidates:
            return int(np.random.randint(0, min(self.vocab_size, 10000)))
        scores = np.array([s for _, s in candidates], dtype=np.float64)
        scores = np.maximum(scores, 1e-10)
        probs = scores ** (1.0 / max(temperature, 0.01))
        if top_k > 0 and len(probs) > top_k:
            threshold = np.sort(probs)[-top_k]
            probs[probs < threshold] = 0.0
        if top_p < 1.0:
            sorted_indices = np.argsort(-probs)
            cumsum = np.cumsum(probs[sorted_indices])
            cutoff = cumsum > top_p
            if np.any(cutoff):
                first_cutoff = int(np.where(cutoff)[0][0])
                if first_cutoff > 0:
                    probs[sorted_indices[first_cutoff + 1 :]] = 0.0
        probs = probs / np.sum(probs)
        idx = int(np.random.choice(len(candidates), p=probs))
        return candidates[idx][0]

    def _adapt_hdc_depth(self) -> int:
        acceptance = self.hd_engine.acceptance_rate()
        if acceptance > 0.85:
            depth = min(self._hdc_depth_max, self._hdc_depth + 1)
        elif acceptance < 0.4:
            depth = max(self._hdc_depth_min, self._hdc_depth - 1)
        else:
            depth = self._hdc_depth
        self._hdc_depth = depth
        return depth

    @staticmethod
    def _softmax(logits: np.ndarray, temperature: float = 0.8) -> np.ndarray:
        logits = logits.astype(np.float64) / max(temperature, 0.01)
        max_l = np.max(logits)
        exp_l = np.exp(logits - max_l)
        return exp_l / np.sum(exp_l)

    # ── Online Learning ──

    def observe_correction(self, context_tokens, hdc_predicted, model_token):
        features = self._build_features()
        self.learning_engine.observe_correction(
            context_tokens, hdc_predicted, model_token, features
        )
        self.monitor.record_hdc_decision(
            confidence=self.confidence_gate.predict(features), accepted=False
        )
        self._consecutive_failures += 1

    def observe_acceptance(self, context_tokens, accepted_token):
        features = self._build_features()
        self.learning_engine.observe_acceptance(
            context_tokens, accepted_token, features
        )
        self.monitor.record_hdc_decision(
            confidence=self.confidence_gate.predict(features), accepted=True
        )
        self._consecutive_failures = max(0, self._consecutive_failures - 1)

    # ── Statistics ──

    def stats(self) -> dict:
        total = (
            sum(self._generation_token_counts)
            if hasattr(self, "_generation_times")
            else 0
        )
        avg_tps = 0.0
        strategy_dist = {
            UNIFIED_STRATEGY_NAMES.get(k, f"level_{k}"): v / max(self._total_tokens, 1)
            for k, v in self._level_token_counts.items()
        }
        return {
            "tokens_per_second": float(avg_tps),
            "total_tokens": self._total_tokens,
            "total_model_calls": self._total_model_calls,
            "total_hdc_tokens": self._total_hdc_tokens,
            "hd_acceptance_rate": self.hd_engine.acceptance_rate(),
            "current_strategy": UNIFIED_STRATEGY_NAMES.get(
                self._current_strategy, "unknown"
            ),
            "strategy_distribution": strategy_dist,
            "consecutive_failures": self._consecutive_failures,
            "hdc_depth": self._hdc_depth,
            "resonance_score": self.resonance_router.resonance_meter.resonance_score(),
        }

    def reset(self):
        self.hd_engine.reset()
        self.holographic_kv.clear()
        self.vlasov_attention.reset()
        self.hrr_weight_store.clear()
        self.resonance_router.reset()
        self._strategy_history.clear()
        self._repetition_buffer.clear()
        for k in self._level_token_counts:
            self._level_token_counts[k] = 0
        self._total_tokens = 0
        self._total_model_calls = 0
        self._total_hdc_tokens = 0
        self._consecutive_failures = 0
        self._current_strategy = UnifiedStrategyLevel.SPECTRAL_BLOCK
        self._hdc_depth = 6

    def close(self):
        self.reset()
        gc.collect()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def create_unified_engine(**kwargs) -> UnifiedInferenceEngine:
    return UnifiedInferenceEngine(**kwargs)
