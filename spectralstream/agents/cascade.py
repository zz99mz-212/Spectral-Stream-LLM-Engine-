from __future__ import annotations

import numpy as np
import time
import math
from collections import deque
from typing import Callable, Optional
from enum import IntEnum


class StrategyLevel(IntEnum):
    FORWARDLESS = 0
    BLOCK_EMISSION = 1
    SPECULATIVE_DECODE = 2
    SINGLE_TOKEN = 3
    FALLBACK = 4


STRATEGY_NAMES = {
    StrategyLevel.FORWARDLESS: "FORWARDLESS",
    StrategyLevel.BLOCK_EMISSION: "BLOCK_EMISSION",
    StrategyLevel.SPECULATIVE_DECODE: "SPECULATIVE_DECODE",
    StrategyLevel.SINGLE_TOKEN: "SINGLE_TOKEN",
    StrategyLevel.FALLBACK: "FALLBACK",
}

STRATEGY_EFFICIENCY = {
    StrategyLevel.FORWARDLESS: float("inf"),
    StrategyLevel.BLOCK_EMISSION: 16.0,
    StrategyLevel.SPECULATIVE_DECODE: 2.5,
    StrategyLevel.SINGLE_TOKEN: 1.0,
    StrategyLevel.FALLBACK: float("inf"),
}

STRATEGY_COST = {
    StrategyLevel.FORWARDLESS: 0.0,
    StrategyLevel.BLOCK_EMISSION: 0.0625,
    StrategyLevel.SPECULATIVE_DECODE: 0.4,
    StrategyLevel.SINGLE_TOKEN: 1.0,
    StrategyLevel.FALLBACK: 0.0,
}


class ResonanceTracker:
    def __init__(
        self,
        window: int = 64,
        kp: float = 0.4,
        ki: float = 0.08,
        kd: float = 0.15,
        target_acceptance: float = 0.75,
    ):
        self.window = window
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.target = target_acceptance

        self.acceptance_buffer: deque[float] = deque(maxlen=window)
        self.block_size_buffer: deque[int] = deque(maxlen=window)
        self.integral = 0.0
        self.prev_error = 0.0
        self.resonance_history: deque[float] = deque(maxlen=128)
        self.n_updates = 0

    def record(self, n_accepted: int, n_total: int, block_size: int = 1):
        if n_total > 0:
            rate = n_accepted / max(n_total, 1)
            self.acceptance_buffer.append(rate)
        self.block_size_buffer.append(block_size)
        self.n_updates += 1

    def resonance_score(self) -> float:
        if len(self.acceptance_buffer) < 4:
            return 0.5

        arr = np.array(list(self.acceptance_buffer), dtype=np.float64)
        mean_accept = float(np.mean(arr))

        error = self.target - mean_accept
        self.integral += error
        self.integral = np.clip(self.integral, -3.0, 3.0)
        derivative = error - self.prev_error
        self.prev_error = error

        pid_output = self.kp * error + self.ki * self.integral + self.kd * derivative

        score = 0.5 + pid_output * 2.0
        score = float(np.clip(score, 0.0, 1.0))
        self.resonance_history.append(score)

        return score

    def acceptance_rate(self) -> float:
        if not self.acceptance_buffer:
            return 0.5
        return float(np.mean(list(self.acceptance_buffer)))

    def reset(self):
        self.acceptance_buffer.clear()
        self.block_size_buffer.clear()
        self.integral = 0.0
        self.prev_error = 0.0
        self.resonance_history.clear()
        self.n_updates = 0


class PredictiveEscalation:
    def __init__(
        self,
        window: int = 16,
        trend_threshold: float = 0.012,
        entropy_threshold: float = 0.65,
        min_samples: int = 8,
    ):
        self.window = window
        self.trend_threshold = trend_threshold
        self.entropy_threshold = entropy_threshold
        self.min_samples = min_samples
        self.entropy_buffer: deque[float] = deque(maxlen=window)
        self.prediction_buffer: deque[bool] = deque(maxlen=window)
        self.early_warnings = 0
        self.false_alarms = 0
        self.correct_predictions = 0

    def observe(self, spectral_entropy: float, hdc_failed: bool = False):
        self.entropy_buffer.append(spectral_entropy)
        if hdc_failed:
            self.prediction_buffer.append(True)
        else:
            self.prediction_buffer.append(False)

    def entropy_trend(self) -> float:
        if len(self.entropy_buffer) < self.min_samples:
            return 0.0
        xs = np.arange(len(self.entropy_buffer), dtype=np.float64)
        ys = np.array(list(self.entropy_buffer), dtype=np.float64)
        slope = np.polyfit(xs, ys, 1)[0]
        return float(slope)

    def should_escalate(self) -> bool:
        if len(self.entropy_buffer) < self.min_samples:
            return False
        trend = self.entropy_trend()
        recent_entropy = float(np.mean(list(self.entropy_buffer)[-4:]))
        return trend > self.trend_threshold or recent_entropy > self.entropy_threshold

    def record_outcome(self, predicted_escalation: bool, actual_failure: bool):
        if predicted_escalation and actual_failure:
            self.correct_predictions += 1
        elif predicted_escalation and not actual_failure:
            self.false_alarms += 1
        elif not predicted_escalation and actual_failure:
            self.early_warnings += 1

    def precision(self) -> float:
        total = self.correct_predictions + self.false_alarms
        return self.correct_predictions / max(total, 1)

    def recall(self) -> float:
        total = self.correct_predictions + self.early_warnings
        return self.correct_predictions / max(total, 1)

    def reset(self):
        self.entropy_buffer.clear()
        self.prediction_buffer.clear()


class HysteresisBand:
    SWITCH_COST_MATRIX: dict[tuple[int, int], float] = {}

    def __init__(self):
        self._build_cost_matrix()
        self.switches: list[tuple[int, int, float]] = []
        self.throttled_switches = 0

    def _build_cost_matrix(self):
        levels = [0, 1, 2, 3, 4]
        for src in levels:
            for dst in levels:
                if src == dst:
                    cost = 0.0
                elif src < dst:
                    cost = 0.05 * (dst - src)
                else:
                    cost = 0.10 * (src - dst)
                if abs(dst - src) >= 2:
                    cost *= 1.5
                self.SWITCH_COST_MATRIX[(src, dst)] = cost

    def switch_cost(self, from_level: int, to_level: int) -> float:
        return self.SWITCH_COST_MATRIX.get((from_level, to_level), 1.0)

    def should_switch(
        self,
        from_level: int,
        to_level: int,
        efficiency_gain: float,
    ) -> bool:
        cost = self.switch_cost(from_level, to_level)
        benefit = efficiency_gain
        if benefit > cost:
            return True
        self.throttled_switches += 1
        return False

    def record_switch(self, from_level: int, to_level: int):
        cost = self.switch_cost(from_level, to_level)
        self.switches.append((from_level, to_level, cost))

    def stats(self) -> dict:
        return {
            "total_switches": len(self.switches),
            "throttled": self.throttled_switches,
        }

    def reset(self):
        self.switches.clear()
        self.throttled_switches = 0


class AdaptiveBatcher:
    def __init__(
        self,
        max_batch_size: int = 16,
        min_batch_size: int = 2,
        max_wait_ms: float = 2.0,
    ):
        self.max_batch_size = max_batch_size
        self.min_batch_size = min_batch_size
        self.max_wait_ms = max_wait_ms
        self.pending: list[dict] = []
        self.batches_formed = 0
        self.batched_verifications = 0
        self.avg_batch_size: deque[float] = deque(maxlen=100)

    def submit(self, verification_request: dict):
        self.pending.append(verification_request)

    def drain(self, force: bool = False) -> list[list[dict]]:
        if not self.pending:
            return []

        n = len(self.pending)
        batch_size = min(max(n, self.min_batch_size), self.max_batch_size)
        if force or n >= self.min_batch_size:
            batches = [self.pending[:batch_size]]
            self.pending = self.pending[batch_size:]
            self.batches_formed += 1
            self.batched_verifications += batch_size
            self.avg_batch_size.append(float(batch_size))
            return batches
        return []

    def pending_count(self) -> int:
        return len(self.pending)

    def mean_batch_size(self) -> float:
        if not self.avg_batch_size:
            return 0.0
        return float(np.mean(list(self.avg_batch_size)))

    def reset(self):
        self.pending.clear()
        self.batches_formed = 0
        self.batched_verifications = 0
        self.avg_batch_size.clear()


class SpectralConfidence:
    def __init__(self, window: int = 8):
        self.window = window
        self.score_history: deque[list[float]] = deque(maxlen=window)
        self.confidence_history: deque[float] = deque(maxlen=128)

    def compute_confidence(self, hdc_candidates: list[tuple[int, float]]) -> float:
        if not hdc_candidates:
            return 0.0

        scores = np.array([s for _, s in hdc_candidates], dtype=np.float64)
        scores = np.maximum(scores, 1e-10)
        scores = scores / np.max(scores)

        if len(scores) < 4:
            return float(scores[0]) if len(scores) > 0 else 0.0

        spectrum = np.abs(np.fft.fft(scores - np.mean(scores)))
        power = spectrum[: max(2, len(spectrum) // 2)]
        power = power / (np.sum(power) + 1e-10)
        spectral_entropy = -np.sum(power * np.log2(power + 1e-10))
        norm_entropy = spectral_entropy / np.log2(len(power) + 1)

        max_score = float(np.max(scores))
        score_variance = float(np.var(scores))

        confidence = (
            max_score * (1.0 - norm_entropy) * 0.7
            + (1.0 / (1.0 + score_variance * 2.0)) * 0.3
        )
        confidence = float(np.clip(confidence, 0.0, 1.0))

        self.score_history.append(scores.tolist())
        self.confidence_history.append(confidence)

        return confidence

    def spectral_entropy(self) -> float:
        if not self.score_history:
            return 0.5
        latest = np.array(self.score_history[-1], dtype=np.float64)
        if len(latest) < 4:
            return 0.5
        spectrum = np.abs(np.fft.fft(latest - np.mean(latest)))
        power = spectrum[: len(spectrum) // 2]
        power = power / (np.sum(power) + 1e-10)
        entropy = -np.sum(power * np.log2(power + 1e-10))
        return float(entropy / np.log2(len(power) + 1))

    def recent_mean_confidence(self) -> float:
        if not self.confidence_history:
            return 0.5
        return float(np.mean(list(self.confidence_history)))

    def reset(self):
        self.score_history.clear()
        self.confidence_history.clear()


class CascadeOrchestrator:
    def __init__(
        self,
        model_fn: Callable,
        hd_engine,
        confidence_gate,
        block_size: int = 16,
        min_block_size: int = 4,
        max_block_size: int = 32,
        n_candidates: int = 64,
        vocab_size: int = 262144,
        target_level_0_ratio: float = 0.70,
    ):
        self.model_fn = model_fn
        self.hd_engine = hd_engine
        self.gate = confidence_gate
        self.vocab_size = vocab_size
        self.target_level_0_ratio = target_level_0_ratio
        self.target_level_0_1_ratio = 0.90

        self.block_size = block_size
        self.min_block_size = min_block_size
        self.max_block_size = max_block_size
        self.n_candidates = n_candidates

        self.resonance_tracker = ResonanceTracker(
            window=64,
            kp=0.4,
            ki=0.08,
            kd=0.15,
            target_acceptance=0.75,
        )

        self.predictive_escalation = PredictiveEscalation(
            window=16,
            trend_threshold=0.012,
            entropy_threshold=0.65,
        )

        self.hysteresis = HysteresisBand()

        self.adaptive_batcher = AdaptiveBatcher(
            max_batch_size=16,
            min_batch_size=2,
            max_wait_ms=2.0,
        )

        self.spectral_confidence = SpectralConfidence(window=8)

        self.current_level = StrategyLevel.BLOCK_EMISSION
        self.strategy_history: deque[int] = deque(maxlen=5000)
        self.level_token_counts: dict[int, int] = {
            int(StrategyLevel.FORWARDLESS): 0,
            int(StrategyLevel.BLOCK_EMISSION): 0,
            int(StrategyLevel.SPECULATIVE_DECODE): 0,
            int(StrategyLevel.SINGLE_TOKEN): 0,
            int(StrategyLevel.FALLBACK): 0,
        }

        self.total_tokens = 0
        self.total_model_calls = 0
        self.total_hdc_tokens = 0
        self.total_model_tokens = 0
        self.total_fallback_tokens = 0

        self.consecutive_level_0 = 0
        self.consecutive_level_1 = 0
        self.consecutive_failures = 0

        self.context_window: deque[int] = deque(maxlen=128)
        self._emitted_tokens: list[int] = []
        self._emitted_strategy: int = 1
        self.start_time = time.time()

    def _select_strategy(
        self,
        hdc_candidates: list[tuple[int, float]],
    ) -> StrategyLevel:
        spectral_conf = self.spectral_confidence.compute_confidence(hdc_candidates)
        spectral_ent = self.spectral_confidence.spectral_entropy()
        resonance = self.resonance_tracker.resonance_score()
        acceptance = self.resonance_tracker.acceptance_rate()

        self.predictive_escalation.observe(spectral_ent)
        needs_escalation = self.predictive_escalation.should_escalate()

        composite = (
            0.35 * spectral_conf
            + 0.25 * resonance
            + 0.20 * acceptance
            + 0.20 * (1.0 - spectral_ent)
        )

        if composite >= 0.80:
            target_level = StrategyLevel.FORWARDLESS
        elif composite >= 0.55:
            target_level = StrategyLevel.BLOCK_EMISSION
        elif composite >= 0.35:
            target_level = StrategyLevel.SPECULATIVE_DECODE
        elif composite >= 0.15:
            target_level = StrategyLevel.SINGLE_TOKEN
        else:
            target_level = StrategyLevel.FALLBACK

        if needs_escalation and target_level < StrategyLevel.SPECULATIVE_DECODE:
            target_level = StrategyLevel(
                max(target_level, StrategyLevel.SPECULATIVE_DECODE)
            )

        if self.consecutive_failures >= 3:
            target_level = StrategyLevel(max(target_level, StrategyLevel.SINGLE_TOKEN))

        efficiency_gain = self._efficiency_gain(self.current_level, target_level)

        if not self.hysteresis.should_switch(
            int(self.current_level), int(target_level), efficiency_gain
        ):
            target_level = self.current_level

        self.current_level = target_level
        self.strategy_history.append(int(target_level))

        return target_level

    def _efficiency_gain(
        self, from_level: StrategyLevel, to_level: StrategyLevel
    ) -> float:
        from_eff = STRATEGY_EFFICIENCY.get(from_level, 1.0)
        to_eff = STRATEGY_EFFICIENCY.get(to_level, 1.0)
        if from_eff == float("inf"):
            from_eff = 100000.0
        if to_eff == float("inf"):
            to_eff = 100000.0
        return math.log2(max(to_eff, 1.0) / max(from_eff, 1.0))

    def _current_block_size(self) -> int:
        if self.current_level == StrategyLevel.BLOCK_EMISSION:
            resonance = self.resonance_tracker.resonance_score()
            size = int(
                self.min_block_size
                + (self.max_block_size - self.min_block_size) * resonance
            )
            return int(np.clip(size, self.min_block_size, self.max_block_size))
        return 1

    def _softmax(self, logits: np.ndarray, temperature: float = 0.8) -> np.ndarray:
        logits = logits.astype(np.float64) / max(temperature, 0.01)
        max_l = np.max(logits)
        exp_l = np.exp(logits - max_l)
        return exp_l / np.sum(exp_l)

    def _generate_forwardless(
        self, context: list[int], temperature: float = 0.8
    ) -> int:
        candidates = self.hd_engine.predict_next(
            tuple(context), n_candidates=self.n_candidates
        )
        if not candidates:
            return int(np.random.randint(0, self.vocab_size))
        return self._sample_from_candidates(candidates, temperature)

    def _generate_block_emission(
        self, context: list[int], temperature: float = 0.8
    ) -> tuple[list[int], int]:
        block_size = self._current_block_size()
        draft_tokens: list[int] = []
        working_context = list(context)

        for _ in range(block_size):
            candidates = self.hd_engine.predict_next(
                tuple(working_context), n_candidates=self.n_candidates
            )
            if not candidates:
                break
            token = self._sample_from_candidates(candidates, temperature)
            draft_tokens.append(token)
            working_context.append(token)

        if not draft_tokens:
            return [], 0

        model_context = list(context) + draft_tokens
        logits, _, _ = self.model_fn(model_context)
        logits = np.array(logits, dtype=np.float64)
        if logits.ndim > 1:
            logits = logits[-len(draft_tokens) :]

        accepted: list[int] = []
        model_calls = 1

        for i, draft in enumerate(draft_tokens):
            if logits.ndim > 1 and i < len(logits):
                step_logits = logits[i]
            elif logits.ndim > 1:
                step_logits = logits[-1]
            else:
                step_logits = logits

            probs = self._softmax(step_logits, temperature)
            draft_prob = float(probs[draft]) if draft < len(probs) else 0.0

            if draft_prob >= 0.01:
                accepted.append(draft)
                self.hd_engine.observe(draft)
            else:
                token = int(np.random.choice(len(probs), p=probs))
                accepted.append(token)
                self.hd_engine.observe(token)
                break

        n_accepted = len(accepted)
        n_total = len(draft_tokens)
        self.resonance_tracker.record(n_accepted, n_total, block_size)

        if n_accepted < len(draft_tokens):
            self.predictive_escalation.observe(
                self.spectral_confidence.spectral_entropy(), hdc_failed=True
            )
            self.consecutive_failures += 1
        else:
            self.consecutive_failures = max(0, self.consecutive_failures - 1)

        return accepted, model_calls

    def _generate_speculative(
        self, context: list[int], temperature: float = 0.8
    ) -> tuple[int, int]:
        candidates = self.hd_engine.predict_next(
            tuple(context), n_candidates=self.n_candidates
        )
        if not candidates:
            logits, _, _ = self.model_fn(context)
            if isinstance(logits, list):
                logits = np.array(logits, dtype=np.float64)
            if logits.ndim > 1:
                logits = logits[-1]
            probs = self._softmax(logits, temperature)
            token = int(np.random.choice(len(probs), p=probs))
            self.hd_engine.observe(token)
            return token, 1

        draft_token = self._sample_from_candidates(candidates, temperature)
        test_context = list(context) + [draft_token]
        logits, _, _ = self.model_fn(test_context)

        if isinstance(logits, list):
            logits = np.array(logits, dtype=np.float64)
        if logits.ndim > 1:
            step_logits = logits[-2]
        else:
            step_logits = logits

        probs = self._softmax(step_logits, temperature)
        draft_prob = float(probs[draft_token]) if draft_token < len(probs) else 0.0

        model_calls = 1

        if draft_prob >= 0.01:
            self.hd_engine.observe(draft_token)
            self.resonance_tracker.record(1, 1, 1)
            self.consecutive_failures = max(0, self.consecutive_failures - 1)
            return draft_token, model_calls
        else:
            token = int(np.random.choice(len(probs), p=probs))
            self.hd_engine.observe(token)
            self.resonance_tracker.record(0, 1, 1)
            self.predictive_escalation.observe(
                self.spectral_confidence.spectral_entropy(), hdc_failed=True
            )
            self.consecutive_failures += 1
            return token, model_calls

    def _generate_single(
        self, context: list[int], temperature: float = 0.8
    ) -> tuple[int, int]:
        logits, _, _ = self.model_fn(context)
        if isinstance(logits, list):
            logits = np.array(logits, dtype=np.float64)
        if logits.ndim > 1:
            logits = logits[-1]
        probs = self._softmax(logits, temperature)
        token = int(np.random.choice(len(probs), p=probs))
        self.hd_engine.observe(token)
        return token, 1

    def _generate_fallback(self, _context: list[int]) -> tuple[int, int]:
        token = int(np.random.randint(0, min(self.vocab_size, 10000)))
        return token, 0

    def _sample_from_candidates(
        self,
        candidates: list[tuple[int, float]],
        temperature: float,
    ) -> int:
        if not candidates:
            return int(np.random.randint(0, self.vocab_size))
        scores = np.array([s for _, s in candidates], dtype=np.float64)
        scores = np.maximum(scores, 1e-10)
        probs = scores ** (1.0 / max(temperature, 0.01))
        probs = probs / np.sum(probs)
        idx = int(np.random.choice(len(candidates), p=probs))
        return candidates[idx][0]

    def orchestrate(
        self, context: list[int], temperature: float = 0.8
    ) -> tuple[int, int]:
        if self._emitted_tokens:
            token = self._emitted_tokens.pop(0)
            self.total_tokens += 1
            self.total_hdc_tokens += 1
            self.level_token_counts[self._emitted_strategy] += 1
            return token, self._emitted_strategy

        candidates = self.hd_engine.predict_next(
            tuple(context), n_candidates=self.n_candidates
        )
        strategy = self._select_strategy(candidates)
        self.level_token_counts[int(strategy)] += 1
        self.total_tokens += 1

        if strategy == StrategyLevel.FORWARDLESS:
            token = self._generate_forwardless(context, temperature)
            self.total_hdc_tokens += 1
            self.consecutive_level_0 += 1
            self.consecutive_level_1 = 0
            return token, int(strategy)

        elif strategy == StrategyLevel.BLOCK_EMISSION:
            block_tokens, model_calls = self._generate_block_emission(
                context, temperature
            )
            self.total_model_calls += model_calls
            if block_tokens:
                token = block_tokens[0]
                self.total_hdc_tokens += len(block_tokens)
                self.consecutive_level_1 += 1
                self.consecutive_level_0 = 0
                self._emitted_tokens.extend(block_tokens[1:])
                self._emitted_strategy = int(strategy)
                return token, int(strategy)
            else:
                logits, _, _ = self.model_fn(context)
                if isinstance(logits, list):
                    logits = np.array(logits, dtype=np.float64)
                if logits.ndim > 1:
                    logits = logits[-1]
                probs = self._softmax(logits, temperature)
                token = int(np.random.choice(len(probs), p=probs))
                self.hd_engine.observe(token)
                self.total_model_calls += 1
                return token, int(strategy)

        elif strategy == StrategyLevel.SPECULATIVE_DECODE:
            token, model_calls = self._generate_speculative(context, temperature)
            self.total_hdc_tokens += 1
            self.total_model_calls += model_calls
            return token, int(strategy)

        elif strategy == StrategyLevel.SINGLE_TOKEN:
            token, model_calls = self._generate_single(context, temperature)
            self.total_model_tokens += 1
            self.total_model_calls += model_calls
            return token, int(strategy)

        else:
            token, _ = self._generate_fallback(context)
            self.total_fallback_tokens += 1
            return token, int(strategy)

    def generate(
        self,
        context: list[int],
        max_new_tokens: int,
        temperature: float = 0.8,
    ) -> list[int]:
        generated = list(context)
        for _ in range(max_new_tokens):
            token, _ = self.orchestrate(generated, temperature)
            generated.append(token)
        return generated

    def submit_verification(
        self, agent_id: str, draft_block: list[int], context: list[int]
    ):
        self.adaptive_batcher.submit(
            {
                "agent_id": agent_id,
                "draft": draft_block,
                "context": context,
            }
        )

    def process_batch(self) -> list[dict]:
        batches = self.adaptive_batcher.drain(
            force=(
                self.adaptive_batcher.pending_count()
                >= self.adaptive_batcher.min_batch_size
            )
        )
        results: list[dict] = []

        for batch in batches:
            contexts = [req["context"] + req["draft"] for req in batch]
            drafts = [req["draft"] for req in batch]

            max_len = max(len(c) for c in contexts)
            padded = [c + [0] * (max_len - len(c)) for c in contexts]

            batch_logits, _, _ = self.model_fn(padded)
            if isinstance(batch_logits, list):
                batch_logits = np.array(batch_logits, dtype=np.float64)

            for i, req in enumerate(batch):
                if batch_logits.ndim == 3:
                    logits = batch_logits[i]
                elif batch_logits.ndim == 2:
                    logits = batch_logits
                else:
                    logits = batch_logits

                if logits.ndim > 1:
                    logits = logits[-1]

                probs = self._softmax(logits)
                accepted = []
                for draft in drafts[i]:
                    prob = float(probs[draft]) if draft < len(probs) else 0.0
                    if prob >= 0.01:
                        accepted.append(draft)
                    else:
                        token = int(np.random.choice(len(probs), p=probs))
                        accepted.append(token)
                        break

                results.append(
                    {
                        "agent_id": req["agent_id"],
                        "tokens": accepted,
                        "n_accepted": len(accepted),
                    }
                )

        return results

    def observe_model_correction(
        self, hdc_token: int, model_token: int, features: list[float]
    ):
        self.gate.train(features, hdc_was_correct=False)
        self.predictive_escalation.observe(
            self.spectral_confidence.spectral_entropy(), hdc_failed=True
        )

    def observe_hdc_acceptance(self, features: list[float]):
        self.gate.train(features, hdc_was_correct=True)

    def level_ratio(self, level: int) -> float:
        return self.level_token_counts.get(level, 0) / max(self.total_tokens, 1)

    def level_0_1_ratio(self) -> float:
        l0 = self.level_token_counts.get(int(StrategyLevel.FORWARDLESS), 0)
        l1 = self.level_token_counts.get(int(StrategyLevel.BLOCK_EMISSION), 0)
        return (l0 + l1) / max(self.total_tokens, 1)

    def stats(self) -> dict:
        elapsed = time.time() - self.start_time
        tok_s = self.total_tokens / max(elapsed, 0.001)
        l0_ratio = self.level_ratio(int(StrategyLevel.FORWARDLESS))
        l1_ratio = self.level_ratio(int(StrategyLevel.BLOCK_EMISSION))
        l01_ratio = self.level_0_1_ratio()

        hdc_efficiency = self.total_hdc_tokens / max(self.total_model_calls, 1)
        avg_tok_per_call = self.total_tokens / max(self.total_model_calls, 1)

        return {
            "tokens_per_second": round(tok_s, 1),
            "total_tokens": self.total_tokens,
            "total_model_calls": self.total_model_calls,
            "average_tokens_per_model_call": round(avg_tok_per_call, 2),
            "hdc_efficiency_tok_per_call": round(hdc_efficiency, 2),
            "level_0_forwardless_ratio": round(l0_ratio, 4),
            "level_1_block_emission_ratio": round(l1_ratio, 4),
            "level_0_1_combined_ratio": round(l01_ratio, 4),
            "level_2_speculative_ratio": round(
                self.level_ratio(int(StrategyLevel.SPECULATIVE_DECODE)), 4
            ),
            "level_3_single_token_ratio": round(
                self.level_ratio(int(StrategyLevel.SINGLE_TOKEN)), 4
            ),
            "level_4_fallback_ratio": round(
                self.level_ratio(int(StrategyLevel.FALLBACK)), 4
            ),
            "level_0_1_target_met": l01_ratio >= 0.90,
            "current_strategy": STRATEGY_NAMES.get(self.current_level, "unknown"),
            "resonance_score": round(self.resonance_tracker.resonance_score(), 4),
            "acceptance_rate": round(self.resonance_tracker.acceptance_rate(), 4),
            "spectral_confidence": round(
                self.spectral_confidence.recent_mean_confidence(), 4
            ),
            "predictive_escalation_precision": round(
                self.predictive_escalation.precision(), 4
            ),
            "predictive_escalation_recall": round(
                self.predictive_escalation.recall(), 4
            ),
            "hysteresis_throttled": self.hysteresis.stats()["throttled"],
            "adaptive_batcher_mean_batch": round(
                self.adaptive_batcher.mean_batch_size(), 2
            ),
            "adaptive_batcher_verifications": self.adaptive_batcher.batched_verifications,
            "consecutive_failures": self.consecutive_failures,
            "hdc_tokens": self.total_hdc_tokens,
            "model_tokens": self.total_model_tokens,
            "fallback_tokens": self.total_fallback_tokens,
            "uptime_seconds": round(elapsed, 1),
        }

    def reset(self):
        self.resonance_tracker.reset()
        self.predictive_escalation.reset()
        self.hysteresis.reset()
        self.adaptive_batcher.reset()
        self.spectral_confidence.reset()
        self.strategy_history.clear()
        for k in self.level_token_counts:
            self.level_token_counts[k] = 0
        self.total_tokens = 0
        self.total_model_calls = 0
        self.total_hdc_tokens = 0
        self.total_model_tokens = 0
        self.total_fallback_tokens = 0
        self.consecutive_level_0 = 0
        self.consecutive_level_1 = 0
        self.consecutive_failures = 0
        self.context_window.clear()
        self.current_level = StrategyLevel.BLOCK_EMISSION
        self._emitted_tokens.clear()
        self.start_time = time.time()
