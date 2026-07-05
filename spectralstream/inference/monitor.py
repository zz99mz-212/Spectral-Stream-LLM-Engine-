from __future__ import annotations

import time
import numpy as np
from collections import deque, defaultdict
from typing import Optional


class InferenceMonitor:
    """Real-time inference monitoring with rolling windows."""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.token_times: deque = deque(maxlen=window_size)
        self.token_counts: deque = deque(maxlen=window_size)
        self.token_accepted: deque = deque(maxlen=window_size)
        self.model_call_latencies: deque = deque(maxlen=window_size)
        self.cache_hits = defaultdict(int)
        self.cache_misses = defaultdict(int)
        self.errors = defaultdict(int)
        self.fallbacks = defaultdict(int)
        self.strategy_calls = defaultdict(int)
        self.strategy_tokens = defaultdict(int)
        self.strategy_latencies = defaultdict(list)
        self.hdc_confidences: deque = deque(maxlen=window_size)
        self.hdc_accepted = 0
        self.hdc_total = 0
        self.confidence_gate_predictions: deque = deque(maxlen=window_size)
        self.confidence_gate_correct = 0
        self.confidence_gate_total = 0
        self.start_time = time.time()

    def record_token(self, strategy: str, latency_ms: float, accepted: bool):
        self.token_times.append(time.time())
        self.token_counts.append(1)
        self.token_accepted.append(1 if accepted else 0)
        self.strategy_calls[strategy] += 1
        self.strategy_tokens[strategy] += 1
        self.strategy_latencies[strategy].append(latency_ms)

    def record_model_call(self, latency_ms: float):
        self.model_call_latencies.append(latency_ms)

    def record_cache_hit(self, tier: str):
        self.cache_hits[tier] += 1

    def record_cache_miss(self, tier: str):
        self.cache_misses[tier] += 1

    def record_error(self, error_type: str):
        self.errors[error_type] += 1

    def record_fallback(self, reason: str):
        self.fallbacks[reason] += 1

    def record_hdc_decision(self, confidence: float, accepted: bool):
        self.hdc_confidences.append(confidence)
        self.hdc_total += 1
        if accepted:
            self.hdc_accepted += 1

    def record_confidence_gate(self, predicted: float, correct: bool):
        self.confidence_gate_predictions.append(predicted)
        self.confidence_gate_total += 1
        if correct:
            self.confidence_gate_correct += 1

    def tokens_per_second(self) -> float:
        if len(self.token_times) < 2:
            return 0.0
        elapsed = self.token_times[-1] - self.token_times[0]
        return sum(self.token_counts) / elapsed if elapsed > 0 else 0.0

    def hdc_acceptance_rate(self) -> float:
        return self.hdc_accepted / max(self.hdc_total, 1)

    def cache_hit_rate(self, tier: Optional[str] = None) -> float:
        if tier:
            hits = self.cache_hits.get(tier, 0)
            misses = self.cache_misses.get(tier, 0)
        else:
            hits = sum(self.cache_hits.values())
            misses = sum(self.cache_misses.values())
        return hits / max(hits + misses, 1)

    def strategy_breakdown(self) -> dict:
        breakdown = {}
        for strategy in set(self.strategy_calls.keys()) | set(
            self.strategy_tokens.keys()
        ):
            lats = self.strategy_latencies.get(strategy, [])
            breakdown[strategy] = {
                "calls": self.strategy_calls.get(strategy, 0),
                "tokens": self.strategy_tokens.get(strategy, 0),
                "avg_latency_ms": round(float(np.mean(lats)), 2) if lats else 0.0,
            }
        return breakdown

    def get_stats(self) -> dict:
        return {
            "timestamp": time.time(),
            "uptime_seconds": time.time() - self.start_time,
            "tokens_per_second": round(self.tokens_per_second(), 2),
            "hdc_acceptance_rate": round(self.hdc_acceptance_rate(), 4),
            "strategy_breakdown": self.strategy_breakdown(),
            "total_tokens": int(sum(self.token_counts)),
            "total_model_calls": len(self.model_call_latencies),
            "hdc_decisions": self.hdc_total,
            "hdc_accepted": self.hdc_accepted,
        }
