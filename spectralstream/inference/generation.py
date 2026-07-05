"""
DEPRECATED — Use InferenceIntelligenceEngine instead.

This module (generation helpers, SpeculativeDecoder, etc.) is deprecated and
will be removed in a future release.  Replace with::

    from spectralstream.inference.intelligence_engine import (
        InferenceIntelligenceEngine,
    )
"""

from __future__ import annotations

import warnings as _warnings

_warnings.warn(
    "spectralstream.inference.generation is deprecated. "
    "Use InferenceIntelligenceEngine from "
    "spectralstream.inference.intelligence_engine instead.",
    DeprecationWarning,
    stacklevel=2,
)

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple, Union

import numpy as np

from spectralstream.inference.model import CPUInferenceEngine


@dataclass
class GenerationResult:
    text: str = ""
    token_ids: List[int] = field(default_factory=list)
    tokens_per_sec: float = 0.0
    total_time_ms: float = 0.0
    strategy_used: str = "standard"
    confidence: float = 0.0
    kv_cache_ratio: float = 1.0
    metadata: dict = field(default_factory=dict)


class PredictiveConfidenceCascade:
    def __init__(self, max_depth: int = 6, min_depth: int = 2):
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.deep_threshold = 0.75
        self.shallow_threshold = 0.35

    def predict_depth_confidences(
        self, ngram_counts: dict, total_counts: dict, context: tuple
    ) -> dict[int, float]:
        confidences = {}
        order_range = range(self.min_depth, min(self.max_depth, len(context)) + 1)
        for order in order_range:
            if order > len(context):
                continue
            ctx = context[-order:]
            total = total_counts.get(order, {}).get(ctx, 0)
            if total > 0:
                max_count = max(
                    ngram_counts.get(order, {}).get(ctx, {}).values(), default=0
                )
                depth_bonus = 1.0 + 0.15 * (order - 1)
                conf = float(np.clip((max_count / total) * depth_bonus, 0.0, 1.0))
            else:
                conf = 0.0
            confidences[order] = conf
        return confidences

    def select_optimal_depth(self, confidences: dict[int, float]) -> tuple[int, str]:
        if not confidences:
            return self.min_depth, "pre_emptive"
        for d in sorted([d for d in confidences if d >= 4], reverse=True):
            if confidences[d] >= self.deep_threshold:
                return d, "deep_skip"
        best_conf = max(confidences.values())
        best_depth = max(confidences, key=confidences.get)
        if best_conf < self.shallow_threshold:
            return best_depth, "pre_emptive"
        return best_depth, "balanced"


class StagedBlockEmission:
    STAGE_CONFIGS = [
        {"name": "stage1", "tokens": 2, "threshold": 0.85},
        {"name": "stage2", "tokens": 4, "threshold": 0.70},
        {"name": "stage3", "tokens": 8, "threshold": 0.55},
    ]

    def __init__(self, max_skip_threshold: float = 0.92):
        self.max_skip_threshold = max_skip_threshold
        self._stage_idx = 0
        self._tokens_in_stage = 0
        self._total_emitted = 0

    def select_stage(self, confidence: float) -> dict:
        if confidence >= self.max_skip_threshold:
            self._stage_idx = 2
            self._tokens_in_stage = 0
            return self.STAGE_CONFIGS[2]
        for i, cfg in enumerate(self.STAGE_CONFIGS):
            if confidence >= cfg["threshold"]:
                self._stage_idx = i
                self._tokens_in_stage = 0
                return cfg
        self._stage_idx = 0
        self._tokens_in_stage = 0
        return self.STAGE_CONFIGS[0]

    def should_verify(self, tokens_generated: int) -> bool:
        cfg = self.STAGE_CONFIGS[self._stage_idx]
        self._tokens_in_stage += 1
        return self._tokens_in_stage >= cfg["tokens"]


class ThermalNoiseInjection:
    def __init__(
        self, base_amplitude: float = 0.01, decay: float = 0.995, max_history: int = 256
    ):
        self.base_amplitude = base_amplitude
        self.decay = decay
        self._history: deque = deque(maxlen=max_history)
        self._amplitude = base_amplitude

    def compute_amplitude(self, temperature: float, confidence: float) -> float:
        self._amplitude *= self.decay
        entropy_bonus = 0.0
        if len(self._history) > 10:
            recent = list(self._history)[-10:]
            if len(set(recent)) < 3:
                entropy_bonus = 0.05
        return self.base_amplitude * temperature * (1.0 - confidence) + entropy_bonus

    def inject_noise(
        self, logits: np.ndarray, temperature: float, confidence: float
    ) -> np.ndarray:
        amp = self.compute_amplitude(temperature, confidence)
        noise = np.random.randn(*logits.shape).astype(np.float32) * amp
        return logits + noise

    def detect_repetition(self, token: int) -> bool:
        self._history.append(token)
        if len(self._history) < 4:
            return False
        recent = list(self._history)[-4:]
        return len(set(recent)) <= 1


class SpeculativeDecoder:
    def __init__(
        self,
        target: CPUInferenceEngine,
        draft: CPUInferenceEngine,
        gamma: int = 4,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.95,
    ):
        self.target = target
        self.draft = draft
        self.gamma = gamma
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self._n_target_calls = 0
        self._n_draft_calls = 0
        self._n_tokens_generated = 0
        self._n_tokens_accepted = 0

    def generate(self, prompt: List[int], max_tokens: int = 100) -> List[int]:
        generated = list(prompt)
        tokens_remaining = max_tokens
        while tokens_remaining > 0:
            draft_tokens, draft_probs = self._draft_generate(
                generated, min(self.gamma, tokens_remaining)
            )
            verified, n_acc = self._verify(generated, draft_tokens, draft_probs)
            generated.extend(verified)
            tokens_remaining -= len(verified)
            self._n_tokens_generated += len(verified)
            self._n_tokens_accepted += n_acc
        return generated[len(prompt) :]

    def _draft_generate(self, context: List[int], n_tokens: int):
        draft_tokens = []
        draft_probs = []
        self.draft.reset()
        current = list(context)
        for _ in range(n_tokens):
            tokens = np.array(current, dtype=np.int32)
            logits = self.draft.forward(tokens)
            probs = np.exp(logits[-1] / max(self.temperature, 1e-10))
            probs = probs / (probs.sum() + 1e-30)
            token = int(np.random.choice(len(probs), p=probs))
            draft_tokens.append(token)
            draft_probs.append(probs)
            current.append(token)
            self._n_draft_calls += 1
        return draft_tokens, draft_probs

    def _verify(
        self, context: List[int], draft_tokens: List[int], draft_probs: List[np.ndarray]
    ):
        n_draft = len(draft_tokens)
        self.target.reset()
        verify_input = np.array(context + draft_tokens, dtype=np.int32)
        target_logits = self.target.forward(verify_input)
        self._n_target_calls += 1
        accepted = []
        n_accepted = 0
        ctx_len = len(context)
        for i in range(n_draft):
            logits_i = (
                target_logits[ctx_len + i] if target_logits.ndim > 1 else target_logits
            )
            probs = np.exp(logits_i / max(self.temperature, 1e-10))
            probs = probs / (probs.sum() + 1e-30)
            dt = draft_tokens[i]
            dp = float(draft_probs[i][dt]) if dt < len(draft_probs[i]) else 1e-10
            tp = float(probs[dt]) if dt < len(probs) else 1e-10
            if tp >= dp:
                accepted.append(dt)
                n_accepted += 1
            else:
                r = np.random.random()
                if r < tp / max(dp, 1e-10):
                    accepted.append(dt)
                    n_accepted += 1
                else:
                    resampled = int(np.random.choice(len(probs), p=probs))
                    accepted.append(resampled)
                    break
        return accepted, n_accepted

    @property
    def acceptance_rate(self) -> float:
        if self._n_tokens_generated == 0:
            return 0.0
        return self._n_tokens_accepted / self._n_tokens_generated

    @property
    def speedup_estimate(self) -> float:
        if self._n_target_calls == 0:
            return 1.0
        return self._n_tokens_generated / max(self._n_target_calls, 1)

    def reset_stats(self):
        self._n_target_calls = 0
        self._n_draft_calls = 0
        self._n_tokens_generated = 0
        self._n_tokens_accepted = 0
