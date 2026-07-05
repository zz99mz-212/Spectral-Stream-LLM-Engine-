"""
SpectralStream Audit Trail System
=================================
Provides a chronological, append-only log of every compression and
inference operation for post-hoc analysis and compliance review.

Components:
  - CompressionAudit  — log every compression decision
  - InferenceAudit    — log every inference request
  - AuditTrail        — chronological log of all operations, export to JSON
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from spectralstream.logging_config import get_logger


# ═══════════════════════════════════════════════════════════════════════════════
# CompressionAudit
# ═══════════════════════════════════════════════════════════════════════════════


class CompressionAudit:
    """Log every compression decision with full context.

    Each entry records:
      - tensor name / shape / dtype
      - chosen method and configuration
      - quality metrics achieved (ratio, MSE, SNR, PSNR, cosine sim)
      - reason for the choice
      - fallback chain if applicable
    """

    def __init__(self, name: str = "compression_audit", max_entries: int = 100_000):
        self.logger = get_logger(name)
        self._entries: deque[dict] = deque(maxlen=max_entries)
        self._lock = threading.Lock()

    def log(
        self,
        tensor_name: str,
        method: str,
        config: Optional[dict] = None,
        ratio: float = 1.0,
        mse: float = 0.0,
        snr_db: float = 0.0,
        psnr_db: float = 0.0,
        cosine_sim: float = 1.0,
        reason: str = "",
        fallback_chain: Optional[list[str]] = None,
        shape: Optional[list[int]] = None,
        dtype: str = "float32",
        original_bytes: int = 0,
        compressed_bytes: int = 0,
        time_s: float = 0.0,
    ) -> dict:
        """Record a compression audit entry and return it."""
        entry: dict[str, Any] = {
            "type": "compression",
            "tensor_name": tensor_name,
            "shape": shape,
            "dtype": dtype,
            "method": method,
            "config": config or {},
            "ratio": round(ratio, 4),
            "mse": round(mse, 8),
            "snr_db": round(snr_db, 4),
            "psnr_db": round(psnr_db, 4),
            "cosine_sim": round(cosine_sim, 6),
            "reason": reason,
            "fallback_chain": fallback_chain or [],
            "original_bytes": original_bytes,
            "compressed_bytes": compressed_bytes,
            "time_s": round(time_s, 6),
            "timestamp": time.time(),
            "timestamp_iso": datetime.now(tz=timezone.utc).isoformat(),
        }
        self.logger.info(
            "Compress %s: method=%s ratio=%.2fx reason=%s",
            tensor_name,
            method,
            ratio,
            reason,
        )
        with self._lock:
            self._entries.append(entry)
        return entry

    def entries(self) -> list[dict]:
        with self._lock:
            return list(self._entries)

    def count(self) -> int:
        with self._lock:
            return len(self._entries)


# ═══════════════════════════════════════════════════════════════════════════════
# InferenceAudit
# ═══════════════════════════════════════════════════════════════════════════════


class InferenceAudit:
    """Log every inference request with full context.

    Each entry records:
      - request ID, prompt metadata
      - model used and strategy selected
      - latency breakdown (prefill / decode / total)
      - tokens generated, throughput
      - cache hit/miss, memory usage
      - error information if the request failed
    """

    def __init__(self, name: str = "inference_audit", max_entries: int = 100_000):
        self.logger = get_logger(name)
        self._entries: deque[dict] = deque(maxlen=max_entries)
        self._lock = threading.Lock()

    def log(
        self,
        request_id: str = "",
        model_name: str = "",
        strategy: str = "",
        prompt_tokens: int = 0,
        generated_tokens: int = 0,
        total_tokens: int = 0,
        latency_ms: float = 0.0,
        prefill_latency_ms: float = 0.0,
        decode_latency_ms: float = 0.0,
        tokens_per_second: float = 0.0,
        cache_hit_rate: float = 0.0,
        memory_peak_mb: float = 0.0,
        error: str = "",
        metadata: Optional[dict] = None,
    ) -> dict:
        """Record an inference audit entry and return it."""
        entry: dict[str, Any] = {
            "type": "inference",
            "request_id": request_id,
            "model_name": model_name,
            "strategy": strategy,
            "prompt_tokens": prompt_tokens,
            "generated_tokens": generated_tokens,
            "total_tokens": total_tokens,
            "latency_ms": round(latency_ms, 3),
            "prefill_latency_ms": round(prefill_latency_ms, 3),
            "decode_latency_ms": round(decode_latency_ms, 3),
            "tokens_per_second": round(tokens_per_second, 2),
            "cache_hit_rate": round(cache_hit_rate, 4),
            "memory_peak_mb": round(memory_peak_mb, 1),
            "error": error,
            "metadata": metadata or {},
            "timestamp": time.time(),
            "timestamp_iso": datetime.now(tz=timezone.utc).isoformat(),
        }
        if error:
            self.logger.error(
                "Inference %s failed: %s",
                request_id,
                error,
            )
        else:
            self.logger.info(
                "Inference %s: %d tokens in %.1fms (%.1f tok/s) [%s]",
                request_id,
                generated_tokens,
                latency_ms,
                tokens_per_second,
                strategy,
            )
        with self._lock:
            self._entries.append(entry)
        return entry

    def entries(self) -> list[dict]:
        with self._lock:
            return list(self._entries)

    def count(self) -> int:
        with self._lock:
            return len(self._entries)


# ═══════════════════════════════════════════════════════════════════════════════
# AuditTrail — unified chronological view
# ═══════════════════════════════════════════════════════════════════════════════


class AuditTrail:
    """Chronological log of all operations (compression + inference).

    Maintains a single sorted timeline by timestamp for cross-cutting
    analysis.  Can merge entries from both CompressionAudit and
    InferenceAudit or accept them directly via :meth:`add_entry`.
    """

    def __init__(self, max_entries: int = 200_000):
        self.logger = get_logger("audit_trail")
        self._entries: deque[dict] = deque(maxlen=max_entries)
        self._lock = threading.Lock()
        self._start_time = time.time()

    def add_entry(self, entry: dict) -> None:
        """Add a pre-built audit entry to the trail."""
        with self._lock:
            self._entries.append(entry)

    def merge(self, audit: CompressionAudit | InferenceAudit) -> int:
        """Merge entries from a CompressionAudit or InferenceAudit.

        Returns the number of entries added.
        """
        entries = audit.entries()
        count = 0
        for e in entries:
            with self._lock:
                self._entries.append(e)
            count += 1
        return count

    def entries(
        self,
        event_type: Optional[str] = None,
        since: Optional[float] = None,
        limit: Optional[int] = None,
    ) -> list[dict]:
        """Return entries, optionally filtered by type and/or time.

        Parameters
        ----------
        event_type:
            Filter to 'compression' or 'inference'.
        since:
            Unix timestamp; only return entries at or after this time.
        limit:
            Maximum number of entries to return (most recent first).
        """
        with self._lock:
            result = list(self._entries)

        if event_type:
            result = [e for e in result if e.get("type") == event_type]
        if since is not None:
            result = [e for e in result if e.get("timestamp", 0) >= since]

        result.sort(key=lambda e: e.get("timestamp", 0), reverse=True)

        if limit is not None:
            result = result[:limit]
        return result

    def count(self) -> int:
        with self._lock:
            return len(self._entries)

    def summary(self) -> dict:
        """Summary statistics of the entire audit trail."""
        with self._lock:
            entries = list(self._entries)
        if not entries:
            return {"total": 0, "compressions": 0, "inferences": 0}

        compressions = [e for e in entries if e.get("type") == "compression"]
        inferences = [e for e in entries if e.get("type") == "inference"]
        methods: dict[str, int] = {}
        for c in compressions:
            m = c.get("method", "unknown")
            methods[m] = methods.get(m, 0) + 1
        errors = sum(1 for i in inferences if i.get("error"))
        strategies: dict[str, int] = {}
        for i in inferences:
            s = i.get("strategy", "unknown")
            strategies[s] = strategies.get(s, 0) + 1

        return {
            "total": len(entries),
            "compressions": len(compressions),
            "inferences": len(inferences),
            "errors": errors,
            "methods_used": methods,
            "strategies_used": strategies,
            "uptime_s": round(time.time() - self._start_time, 1),
        }

    def export_json(self, path: str) -> None:
        """Export the full trail to a JSON file.

        Entries are sorted chronologically (oldest first).
        """
        with self._lock:
            entries = sorted(list(self._entries), key=lambda e: e.get("timestamp", 0))
        out = Path(path).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2, default=str)
        self.logger.info(
            "Exported %d audit entries to %s",
            len(entries),
            str(out),
        )

    def clear(self) -> int:
        """Clear the trail and return the number of entries removed."""
        with self._lock:
            n = len(self._entries)
            self._entries.clear()
        self.logger.info("Audit trail cleared (%d entries removed)", n)
        return n
