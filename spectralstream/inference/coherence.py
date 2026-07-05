from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np


class CoherenceEngine:
    def __init__(self, score_fn: Optional[callable] = None):
        self.score_fn = score_fn
        self.quality_history: deque = deque(maxlen=100)
        self.adaptation_rate = 0.1
        self.coherence_threshold = 0.4
        self.diversity_threshold = 0.3
        self.n_alternatives_generated = 0
        self.n_coherence_improvements = 0

    def score_coherence(self, tokens: list[int]) -> float:
        if len(tokens) < 4:
            return 0.5

        unique_tokens = len(set(tokens))
        total_tokens = len(tokens)
        diversity = unique_tokens / max(total_tokens, 1)

        ngram_repeat = 0
        for n in [2, 3, 4]:
            if len(tokens) >= n:
                ngrams = list(zip(*[tokens[i:] for i in range(n)]))
                unique_ngrams = len(set(ngrams))
                total_ngrams = len(ngrams)
                ngram_repeat += unique_ngrams / max(total_ngrams, 1)
        ngram_diversity = ngram_repeat / 3

        max_token = max(tokens)
        min_token = min(tokens)
        token_range = max_token - min_token
        transitions = [abs(tokens[i] - tokens[i - 1]) for i in range(1, len(tokens))]
        smoothness = 1.0 - min(1.0, float(np.mean(transitions)) / max(token_range, 1))

        coherence = (
            0.3 * diversity
            + 0.3 * ngram_diversity
            + 0.2 * smoothness
            + 0.2 * min(1.0, max(0.0, 1.0 - abs(0.5 - diversity) * 2))
        )
        return float(min(1.0, max(0.0, coherence)))

    def generate_coherent(
        self, generate_fn: callable, context: list[int], n_alternatives: int = 3
    ) -> list[int]:
        best_tokens: list[int] = []
        best_score = -1.0

        for _ in range(n_alternatives):
            tokens: list[int] = []
            ctx = list(context)
            n_gen = min(16, len(context))
            for _ in range(n_gen):
                token = self._predict_token(generate_fn, ctx)
                tokens.append(token)
                if len(ctx) > 64:
                    ctx = ctx[1:] + [token]
                else:
                    ctx = ctx + [token]

            score = self.score_coherence(tokens)
            self.n_alternatives_generated += 1

            if score > best_score:
                old_best = best_score
                best_score = score
                best_tokens = tokens
                if old_best > 0 and score > old_best * 1.1:
                    self.n_coherence_improvements += 1

        return best_tokens

    def _predict_token(self, generate_fn: callable, ctx: list[int]) -> int:
        if self.score_fn:
            return self.score_fn(ctx)
        return int(np.random.randint(0, 32000))

    def adapt_thresholds(self):
        if len(self.quality_history) < 10:
            return

        recent = list(self.quality_history)[-10:]
        avg_quality = float(np.mean(recent))

        if avg_quality > 0.7:
            self.coherence_threshold = min(
                0.6, self.coherence_threshold + self.adaptation_rate
            )
        elif avg_quality < 0.4:
            self.coherence_threshold = max(
                0.2, self.coherence_threshold - self.adaptation_rate
            )

    def get_stats(self) -> dict:
        recent_q = list(self.quality_history)
        return {
            "coherence_threshold": round(self.coherence_threshold, 3),
            "alternatives_generated": self.n_alternatives_generated,
            "coherence_improvements": self.n_coherence_improvements,
            "improvement_rate": self.n_coherence_improvements
            / max(self.n_alternatives_generated, 1),
            "recent_quality": round(float(np.mean(recent_q)), 3) if recent_q else 0.0,
        }


class AttractorGuidedGenerator:
    def __init__(self, d_state: int = 256):
        self.dim = d_state
        self.attractors: list[np.ndarray] = []
        self.current_state = np.zeros(d_state, dtype=np.float64)
        self.energy_history: deque = deque(maxlen=100)

    def update_state(self, token: int, context: list[int]):
        state_update = np.zeros(self.dim, dtype=np.float64)
        for i, t in enumerate(context[-8:]):
            pos = hash((t, i)) % self.dim
            state_update[pos] += 1.0 if hash(str(token)) % 2 == 0 else -1.0
        norm = float(np.linalg.norm(state_update))
        if norm > 0:
            state_update /= norm
        self.current_state = 0.9 * self.current_state + 0.1 * state_update
        state_norm = float(np.linalg.norm(self.current_state))
        if state_norm > 0:
            self.current_state = self.current_state / state_norm

    def energy(self) -> float:
        if not self.attractors:
            return 0.5
        energies = [float(-np.dot(self.current_state, a)) for a in self.attractors]
        return float(np.min(energies)) if energies else 0.5

    def store_attractor(self, state: np.ndarray):
        if len(self.attractors) < 100:
            self.attractors.append(state.copy())

    def is_coherent(self, threshold: float = -0.3) -> bool:
        return self.energy() < threshold
