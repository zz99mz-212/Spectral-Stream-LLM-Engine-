"""
Inference Monitoring & Telemetry System

Tracks:
- Tokens per second (rolling window)
- Model calls per token (efficiency metric)
- HDC acceptance rate
- Confidence gate accuracy
- Cache hit rates
- Memory usage per component
- Error rates and fallback frequency
- Per-strategy performance breakdown

Exports:
- JSON stats endpoint for the server
- Real-time console dashboard
- Prometheus-style metrics
"""

import json
import time
import numpy as np
from collections import deque, defaultdict
from typing import Optional

from spectralstream.inference.monitor import InferenceMonitor


class EnhancedInferenceMonitor(InferenceMonitor):
    """Enhanced InferenceMonitor with Prometheus and console output support."""

    def __init__(self, window_size: int = 100):
        super().__init__(window_size)

    def average_latency_ms(self) -> float:
        if not self.model_call_latencies:
            return 0.0
        return float(np.mean(list(self.model_call_latencies)))

    def tokens_per_model_call(self) -> float:
        if not self.model_call_latencies:
            return 0.0
        model_calls = len(self.model_call_latencies)
        total_tokens = sum(self.token_counts)
        return total_tokens / max(model_calls, 1)

    def confidence_gate_accuracy(self) -> float:
        if self.confidence_gate_total == 0:
            return 0.0
        return self.confidence_gate_correct / self.confidence_gate_total

    def error_rate(self) -> float:
        total = sum(self.errors.values())
        if total == 0:
            return 0.0
        total_ops = sum(self.token_counts) + total
        return total / max(total_ops, 1)

    def fallback_rate(self) -> float:
        total_fb = sum(self.fallbacks.values())
        total_ops = sum(self.token_counts)
        return total_fb / max(total_ops, 1)

    def estimated_memory_bytes(self) -> dict:
        import sys as _sys

        mem = {}
        for name, obj in [
            ("token_times", self.token_times),
            ("token_counts", self.token_counts),
            ("model_call_latencies", self.model_call_latencies),
            ("hdc_confidences", self.hdc_confidences),
            ("errors", self.errors),
            ("fallbacks", self.fallbacks),
            ("strategy_latencies", self.strategy_latencies),
        ]:
            mem[name] = _sys.getsizeof(obj)
        return mem

    def get_stats(self) -> dict:
        s = super().get_stats()
        s.update(
            {
                "tokens_per_model_call": round(self.tokens_per_model_call(), 2),
                "average_latency_ms": round(self.average_latency_ms(), 2),
                "confidence_gate_accuracy": round(self.confidence_gate_accuracy(), 4),
                "cache_hit_rate": round(self.cache_hit_rate(), 4),
                "cache_hits": dict(self.cache_hits),
                "cache_misses": dict(self.cache_misses),
                "error_rate": round(self.error_rate(), 4),
                "errors": dict(self.errors),
                "fallback_rate": round(self.fallback_rate(), 4),
                "fallbacks": dict(self.fallbacks),
                "tokens_per_model_call": round(self.tokens_per_model_call(), 2),
                "average_latency_ms": round(self.average_latency_ms(), 2),
                "confidence_gate_accuracy": round(self.confidence_gate_accuracy(), 4),
                "cache_hit_rate": round(self.cache_hit_rate(), 4),
                "cache_hits": dict(self.cache_hits),
                "cache_misses": dict(self.cache_misses),
                "error_rate": round(self.error_rate(), 4),
                "errors": dict(self.errors),
                "fallback_rate": round(self.fallback_rate(), 4),
                "fallbacks": dict(self.fallbacks),
                "strategy_breakdown": self.strategy_breakdown(),
                "total_tokens": int(sum(self.token_counts)),
                "total_model_calls": len(self.model_call_latencies),
                "hdc_decisions": self.hdc_total,
                "hdc_accepted": self.hdc_accepted,
                "gate_predictions": self.confidence_gate_total,
                "gate_correct": self.confidence_gate_correct,
            }
        )
        return s

    def to_prometheus(self) -> str:
        s = self.get_stats()
        prefix = "spectralstream"
        lines = [
            f"# HELP {prefix}_uptime_seconds Uptime of the inference engine",
            f"# TYPE {prefix}_uptime_seconds gauge",
            f"{prefix}_uptime_seconds {s['uptime_seconds']:.1f}",
            "",
            f"# HELP {prefix}_tokens_per_second Current generation throughput",
            f"# TYPE {prefix}_tokens_per_second gauge",
            f"{prefix}_tokens_per_second {s['tokens_per_second']}",
            "",
            f"# HELP {prefix}_tokens_per_model_call Efficiency ratio",
            f"# TYPE {prefix}_tokens_per_model_call gauge",
            f"{prefix}_tokens_per_model_call {s['tokens_per_model_call']}",
            "",
            f"# HELP {prefix}_average_latency_ms Average model call latency",
            f"# TYPE {prefix}_average_latency_ms gauge",
            f"{prefix}_average_latency_ms {s['average_latency_ms']}",
            "",
            f"# HELP {prefix}_hdc_acceptance_rate HDC draft acceptance rate",
            f"# TYPE {prefix}_hdc_acceptance_rate gauge",
            f"{prefix}_hdc_acceptance_rate {s['hdc_acceptance_rate']}",
            "",
            f"# HELP {prefix}_confidence_gate_accuracy Confidence gate prediction accuracy",
            f"# TYPE {prefix}_confidence_gate_accuracy gauge",
            f"{prefix}_confidence_gate_accuracy {s['confidence_gate_accuracy']}",
            "",
            f"# HELP {prefix}_cache_hit_rate Overall cache hit rate",
            f"# TYPE {prefix}_cache_hit_rate gauge",
            f"{prefix}_cache_hit_rate {s['cache_hit_rate']}",
            "",
            f"# HELP {prefix}_error_rate Error rate across all operations",
            f"# TYPE {prefix}_error_rate gauge",
            f"{prefix}_error_rate {s['error_rate']}",
            "",
            f"# HELP {prefix}_fallback_rate Fallback rate",
            f"# TYPE {prefix}_fallback_rate gauge",
            f"{prefix}_fallback_rate {s['fallback_rate']}",
            "",
            f"# HELP {prefix}_total_tokens Total tokens generated",
            f"# TYPE {prefix}_total_tokens counter",
            f"{prefix}_total_tokens {s['total_tokens']}",
            "",
            f"# HELP {prefix}_total_model_calls Total model forward calls",
            f"# TYPE {prefix}_total_model_calls counter",
            f"{prefix}_total_model_calls {s['total_model_calls']}",
        ]

        for strategy, info in s.get("strategy_breakdown", {}).items():
            lines.extend(
                [
                    "",
                    f"# HELP {prefix}_strategy_calls_total Calls by strategy: {strategy}",
                    f"# TYPE {prefix}_strategy_calls_total counter",
                    f'{prefix}_strategy_calls_total{{strategy="{strategy}"}} {info["calls"]}',
                    "",
                    f"# HELP {prefix}_strategy_tokens_total Tokens by strategy: {strategy}",
                    f"# TYPE {prefix}_strategy_tokens_total counter",
                    f'{prefix}_strategy_tokens_total{{strategy="{strategy}"}} {info["tokens"]}',
                ]
            )

        for err_type, count in s.get("errors", {}).items():
            lines.extend(
                [
                    "",
                    f"# HELP {prefix}_errors_total Errors by type: {err_type}",
                    f"# TYPE {prefix}_errors_total counter",
                    f'{prefix}_errors_total{{error_type="{err_type}"}} {count}',
                ]
            )

        for fb_reason, count in s.get("fallbacks", {}).items():
            lines.extend(
                [
                    "",
                    f"# HELP {prefix}_fallbacks_total Fallbacks by reason: {fb_reason}",
                    f"# TYPE {prefix}_fallbacks_total counter",
                    f'{prefix}_fallbacks_total{{reason="{fb_reason}"}} {count}',
                ]
            )

        lines.append("")
        lines.append(f"# EOF {prefix} metrics")
        return "\n".join(lines)

    def to_console(self) -> str:
        return self.get_performance_report()

    def get_performance_report(self) -> str:
        s = self.get_stats()
        lines = [
            "=" * 56,
            "  SpectralStream Performance Report",
            "=" * 56,
            f"  Uptime:              {s['uptime_seconds']:.1f}s",
            f"  Tokens generated:    {s['total_tokens']}",
            f"  Model calls:         {s['total_model_calls']}",
            f"  Throughput:          {s['tokens_per_second']:.1f} tok/s",
            f"  Efficiency:          {s['tokens_per_model_call']:.2f} tok/call",
            f"  Avg latency:         {s['average_latency_ms']:.1f} ms",
            "",
            "  \u2500\u2500 HDC Draft Engine \u2500\u2500",
            f"  Acceptance rate:     {s['hdc_acceptance_rate']:.1%}",
            f"  Decisions:           {s['hdc_decisions']}",
            f"  Accepted:            {s['hdc_accepted']}",
            "",
            "  \u2500\u2500 Confidence Gate \u2500\u2500",
            f"  Accuracy:            {s['confidence_gate_accuracy']:.1%}",
            f"  Predictions:         {s['gate_predictions']}",
            "",
            "  \u2500\u2500 Cache \u2500\u2500",
            f"  Hit rate:            {s['cache_hit_rate']:.1%}",
        ]
        if s.get("errors"):
            lines.extend(
                [
                    "",
                    "  \u2500\u2500 Errors \u2500\u2500",
                ]
            )
            for err_type, count in s["errors"].items():
                lines.append(f"  {err_type}: {count}")
        if s.get("strategy_breakdown"):
            lines.extend(
                [
                    "",
                    "  \u2500\u2500 Strategy Breakdown \u2500\u2500",
                ]
            )
            for strategy, info in s["strategy_breakdown"].items():
                lines.append(
                    f"  {strategy:20s}  "
                    f"{info['tokens']:4d} tok  "
                    f"{info['avg_latency_ms']:6.1f} ms  "
                    f"{info['calls']:3d} calls"
                )
        lines.append("=" * 56)
        return "\n".join(lines)
