from __future__ import annotations

import numpy as np
from collections import deque
from typing import Optional

from spectralstream.inference.hdc_engine import HDCDraftEngine
from spectralstream.inference.attractor import AttractorScoringEnsemble
from spectralstream.inference.resonance import AdaptivePIDController


class BlockEmissionPipeline:
    """Enhanced block emission pipeline — emit multiple tokens per forward pass.

    Core flow: draft → score → verify → emit
    """

    def __init__(
        self,
        model_fn,
        hd_engine: HDCDraftEngine,
        scorer: AttractorScoringEnsemble,
        block_size: int = 8,
        min_block_size: int = 2,
        max_block_size: int = 24,
        n_candidate_blocks: int = 16,
        coherence_threshold: float = 0.55,
    ):
        self.model_fn = model_fn
        self.hd = hd_engine
        self.scorer = scorer
        self.block_size = block_size
        self.min_block_size = min_block_size
        self.max_block_size = max_block_size
        self.n_candidate_blocks = n_candidate_blocks
        self.coherence_threshold = coherence_threshold
        self.pid = AdaptivePIDController(
            target_acceptance=0.65,
            min_block_size=min_block_size,
            max_block_size=max_block_size,
            default_block_size=block_size,
        )
        self.block_confidence = deque(maxlen=32)
        self.current_block_confidence = 0.5
        self.stage_confidences = {}
        self.hierarchical_conf = {
            "token_level": 0.0,
            "phrase_level": 0.0,
            "block_level": 0.0,
        }
        self.previous_blocks = deque(maxlen=8)
        self.total_tokens = 0
        self.total_model_calls = 0
        self.successful_blocks = 0
        self.fallback_tokens = 0
        self.tokens_per_call_log = []
        self.fallback_reasons = []

    def _determine_block_size(self, confidence: float, tokens_remaining: int) -> int:
        if confidence > 0.8:
            base = self.max_block_size
        elif confidence > 0.6:
            base = self.block_size + 4
        elif confidence > 0.4:
            base = self.block_size
        else:
            base = max(self.min_block_size, self.block_size - 2)
        jitter = int(np.random.randn() * 1.5)
        size = int(np.clip(base + jitter, self.min_block_size, self.max_block_size))
        return min(size, tokens_remaining)

    def _cascade_fallback(self, context: list, past, depth: int = 0) -> tuple:
        max_depth = 3
        if depth > max_depth:
            token = int(np.random.randint(0, min(self.hd.vocab_size, 10000)))
            self.fallback_reasons.append("rng_fallback")
            return token, past
        logits, _, new_past = self.model_fn(context[-1:], past)
        self.total_model_calls += 1
        logits = logits[-1] if logits.ndim > 1 else logits
        probs = self._softmax(logits)
        token = int(np.random.choice(len(probs), p=probs))
        self.fallback_reasons.append(f"model_single_depth{depth}")
        return token, new_past

    def _softmax(self, logits: np.ndarray, temperature: float = 0.8) -> np.ndarray:
        logits = logits.astype(np.float64) / temperature
        max_logit = np.max(logits)
        exp_logits = np.exp(logits - max_logit)
        return exp_logits / np.sum(exp_logits)

    def generate(self, input_ids: list, max_new_tokens: int = 256) -> list:
        generated = list(input_ids)
        tokens_remaining = max_new_tokens
        logits, hidden_states, past = self.model_fn(generated, past=None)
        self.total_model_calls += 1
        next_logits = logits[-1] if logits.ndim > 1 else logits
        probs = self._softmax(next_logits)
        first_token = int(np.random.choice(len(probs), p=probs))
        generated.append(first_token)
        self.hd.observe(first_token)
        tokens_remaining -= 1
        self.total_tokens += 1

        while tokens_remaining > 0:
            context_tuple = tuple(generated)
            block_confidence = float(np.mean([0.5]))
            effective_block_size = self._determine_block_size(
                block_confidence, tokens_remaining
            )
            candidate_blocks = self.hd.draft_block(block_size=effective_block_size)
            if not candidate_blocks or len(candidate_blocks[0]) == 0:
                token, new_past = self._cascade_fallback(generated, past)
                generated.append(token)
                self.hd.observe(token)
                tokens_remaining -= 1
                self.total_tokens += 1
                self.fallback_tokens += 1
                continue

            best_block = candidate_blocks[0]
            verified_block, new_past = self._verify_block(generated, best_block, past)
            if verified_block:
                for token in verified_block:
                    generated.append(token)
                    self.hd.observe(token)
                emitted = len(verified_block)
                tokens_remaining -= emitted
                self.total_tokens += emitted
                self.successful_blocks += 1
                self.tokens_per_call_log.append(emitted)
                self.previous_blocks.append(verified_block)
                acceptance_rate = emitted / max(effective_block_size, 1)
                self.pid.update(acceptance_rate)
                continue

            token, new_past = self._cascade_fallback(generated, past)
            generated.append(token)
            self.hd.observe(token)
            tokens_remaining -= 1
            self.total_tokens += 1
            self.fallback_tokens += 1

        return generated

    def _verify_block(self, context: list, candidate_block: list, past) -> tuple:
        test_tokens = context + candidate_block
        if len(test_tokens) > 2 * len(context) + 64:
            return [], None
        try:
            logits, hidden_states, new_past = self.model_fn(test_tokens, past)
            self.total_model_calls += 1
        except Exception:
            return [], None
        if logits is None:
            return [], None
        verified = []
        for i, token in enumerate(candidate_block):
            if logits.ndim > 1 and i < len(logits):
                step_logits = logits[i]
            else:
                step_logits = logits[-1] if logits.ndim > 1 else logits
            probs = self._softmax(step_logits)
            token_prob = float(probs[token]) if token < len(probs) else 0.0
            if token_prob >= 0.01:
                verified.append(token)
            else:
                break
        return verified, new_past if len(verified) == len(candidate_block) else past

    def statistics(self) -> dict:
        calls_per_token = self.total_model_calls / max(self.total_tokens, 1)
        return {
            "total_tokens": self.total_tokens,
            "total_model_calls": self.total_model_calls,
            "tokens_per_model_call": 1.0 / calls_per_token
            if calls_per_token > 0
            else 0,
            "successful_blocks": self.successful_blocks,
            "fallback_tokens": self.fallback_tokens,
            "pid_params": self.pid.params(),
            "fallback_reasons": self.fallback_reasons[-10:],
        }
