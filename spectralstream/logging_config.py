"""
SpectralStream Logging Infrastructure
=====================================
Provides structured logging, performance timing, and audit trail.

Components:
  - setup_logging()            — configure root logger for the entire package
  - get_logger()               — child logger with context
  - CompressionLogger          — log compression operations with metrics
  - PerformanceTimer           — context manager for timing operations
  - AuditLogger                — audit trail for all compression decisions
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


# ═══════════════════════════════════════════════════════════════════════════════
# JSON Formatter
# ═══════════════════════════════════════════════════════════════════════════════


class _JSONFormatter(logging.Formatter):
    """Emit each log record as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        if record.exc_info and record.exc_info[0] is not None:
            entry["exception"] = self.formatException(record.exc_info)
        if hasattr(record, "extra_data"):
            entry["data"] = record.extra_data
        return json.dumps(entry, default=str)


class _HumanFormatter(logging.Formatter):
    """Compact human-readable format for console output."""

    COLORS = {
        "DEBUG": "\033[36m",
        "INFO": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[35m",
    }
    RESET = "\033[0m"

    def __init__(self, use_color: bool = True):
        super().__init__()
        self.use_color = use_color and sys.stderr.isatty()

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created).strftime("%H:%M:%S.%f")[:-3]
        level = record.levelname[0:5]
        name = record.name.split(".")[-1]
        msg = record.getMessage()
        if self.use_color:
            color = self.COLORS.get(record.levelname, "")
            return f"{ts} {color}{level}{self.RESET} [{name}] {msg}"
        return f"{ts} {level} [{name}] {msg}"


# ═══════════════════════════════════════════════════════════════════════════════
# Setup
# ═══════════════════════════════════════════════════════════════════════════════

_CONFIGURED = False


def setup_logging(
    level: str = "INFO",
    json_output: bool = False,
    log_file: Optional[str] = None,
) -> None:
    """Configure root logger for the entire spectralstream package.

    Parameters
    ----------
    level:
        Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    json_output:
        If True use JSON formatter for console; otherwise human-readable.
    log_file:
        Optional path to a file that receives JSON-lines output.
    """
    global _CONFIGURED
    root = logging.getLogger("spectralstream")

    if _CONFIGURED:
        return

    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    if json_output:
        console_fmt: logging.Formatter = _JSONFormatter()
    else:
        console_fmt = _HumanFormatter()

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(console_fmt)
    root.addHandler(console_handler)

    if log_file:
        log_path = Path(log_file).expanduser()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
        file_handler.setFormatter(_JSONFormatter())
        root.addHandler(file_handler)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Get a child logger with context.

    Ensures logging is configured before returning the logger.
    """
    if not _CONFIGURED:
        setup_logging()
    return logging.getLogger(f"spectralstream.{name}")


# ═══════════════════════════════════════════════════════════════════════════════
# CompressionLogger
# ═══════════════════════════════════════════════════════════════════════════════


class CompressionLogger:
    """Log compression operations with metrics."""

    def __init__(self, name: str = "compression"):
        self.logger = get_logger(name)
        self._operations: deque = deque(maxlen=10000)
        self._lock = threading.Lock()

    def log_compression(
        self,
        tensor_name: str,
        method: str,
        ratio: float,
        error: float,
        time_s: float,
    ) -> None:
        """Log a single compression event."""
        entry = {
            "event": "compression",
            "tensor": tensor_name,
            "method": method,
            "ratio": round(ratio, 4),
            "error": round(error, 6),
            "time_s": round(time_s, 6),
            "timestamp": time.time(),
        }
        self.logger.info(
            "Compressed %s with %s: ratio=%.2fx, error=%.6f, time=%.3fs",
            tensor_name,
            method,
            ratio,
            error,
            time_s,
        )
        with self._lock:
            self._operations.append(entry)

    def log_decompression(
        self,
        tensor_name: str,
        time_s: float,
    ) -> None:
        """Log a single decompression event."""
        entry = {
            "event": "decompression",
            "tensor": tensor_name,
            "time_s": round(time_s, 6),
            "timestamp": time.time(),
        }
        self.logger.info(
            "Decompressed %s in %.3fs",
            tensor_name,
            time_s,
        )
        with self._lock:
            self._operations.append(entry)

    def log_batch(self, results: list[dict]) -> None:
        """Log a batch of compression results.

        Each dict should contain: tensor, method, ratio, error, time_s.
        """
        for r in results:
            self.log_compression(
                tensor_name=r.get("tensor", "unknown"),
                method=r.get("method", "unknown"),
                ratio=r.get("ratio", 0.0),
                error=r.get("error", 0.0),
                time_s=r.get("time_s", 0.0),
            )
        total_ratio = (
            sum(r.get("ratio", 1.0) for r in results) / len(results) if results else 1.0
        )
        self.logger.info(
            "Batch compression complete: %d tensors, avg_ratio=%.2fx",
            len(results),
            total_ratio,
        )

    def get_history(self) -> list[dict]:
        """Return recorded operations as a list."""
        with self._lock:
            return list(self._operations)

    def summary(self) -> dict:
        """Aggregate summary of all recorded operations."""
        with self._lock:
            ops = list(self._operations)
        if not ops:
            return {"total": 0}
        ratios = [o["ratio"] for o in ops if o["event"] == "compression"]
        errors = [o["error"] for o in ops if o["event"] == "compression"]
        times = [o["time_s"] for o in ops]
        methods: dict[str, int] = defaultdict(int)
        for o in ops:
            methods[o.get("method", "unknown")] += 1
        return {
            "total": len(ops),
            "compressions": len(ratios),
            "decompressions": sum(1 for o in ops if o["event"] == "decompression"),
            "avg_ratio": round(sum(ratios) / len(ratios), 4) if ratios else 0.0,
            "avg_error": round(sum(errors) / len(errors), 6) if errors else 0.0,
            "total_time_s": round(sum(times), 4),
            "methods": dict(methods),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# PerformanceTimer
# ═══════════════════════════════════════════════════════════════════════════════


class PerformanceTimer:
    """Context manager for timing code blocks.

    Usage::

        with PerformanceTimer("quantize") as timer:
            do_work()
        print(timer.elapsed)
    """

    def __init__(
        self,
        label: str = "operation",
        logger: Optional[logging.Logger] = None,
        log_on_exit: bool = True,
    ):
        self.label = label
        self.logger = logger or get_logger("timer")
        self.log_on_exit = log_on_exit
        self.start_time: float = 0.0
        self.elapsed: float = 0.0

    def __enter__(self) -> PerformanceTimer:
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.elapsed = time.perf_counter() - self.start_time
        if self.log_on_exit:
            if exc_type is not None:
                self.logger.warning(
                    "%s failed after %.4fs: %s",
                    self.label,
                    self.elapsed,
                    exc_val,
                )
            else:
                self.logger.debug(
                    "%s completed in %.4fs",
                    self.label,
                    self.elapsed,
                )


@contextmanager
def timed_operation(label: str, logger: Optional[logging.Logger] = None):
    """Shorthand context manager for timing an operation.

    Yields the elapsed time in seconds upon exit.
    """
    timer = PerformanceTimer(label, logger=logger, log_on_exit=False)
    t0 = time.perf_counter()
    try:
        yield timer
    finally:
        timer.elapsed = time.perf_counter() - t0
        (logger or get_logger("timer")).debug("%s: %.4fs", label, timer.elapsed)


# ═══════════════════════════════════════════════════════════════════════════════
# AuditLogger
# ═══════════════════════════════════════════════════════════════════════════════


class AuditLogger:
    """Audit trail for all compression decisions.

    Records every decision with tensor name, chosen method, reasoning,
    and quality metrics so that decisions can be reviewed later.
    """

    def __init__(self, name: str = "audit"):
        self.logger = get_logger(name)
        self._trail: deque = deque(maxlen=100_000)
        self._lock = threading.Lock()

    def log_decision(
        self,
        tensor: str,
        method: str,
        reason: str,
        metrics: Optional[dict] = None,
    ) -> None:
        """Record a compression decision."""
        entry = {
            "event": "decision",
            "tensor": tensor,
            "method": method,
            "reason": reason,
            "metrics": metrics or {},
            "timestamp": time.time(),
        }
        self.logger.info(
            "Decision: tensor=%s method=%s reason=%s",
            tensor,
            method,
            reason,
        )
        with self._lock:
            self._trail.append(entry)

    def log_override(
        self,
        tensor: str,
        original: str,
        override: str,
        reason: str,
    ) -> None:
        """Record an override of a compression decision."""
        entry = {
            "event": "override",
            "tensor": tensor,
            "original": original,
            "override": override,
            "reason": reason,
            "timestamp": time.time(),
        }
        self.logger.warning(
            "Override: tensor=%s %s -> %s (%s)",
            tensor,
            original,
            override,
            reason,
        )
        with self._lock:
            self._trail.append(entry)

    def export_trail(self, path: str) -> None:
        """Export the full audit trail to a JSON file."""
        with self._lock:
            trail = list(self._trail)
        out_path = Path(path).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(trail, f, indent=2, default=str)
        self.logger.info(
            "Exported %d audit entries to %s",
            len(trail),
            str(out_path),
        )

    def get_trail(self) -> list[dict]:
        """Return the full audit trail as a list."""
        with self._lock:
            return list(self._trail)

    def summary(self) -> dict:
        """Summary statistics of the audit trail."""
        with self._lock:
            trail = list(self._trail)
        if not trail:
            return {"total": 0}
        decisions = [e for e in trail if e["event"] == "decision"]
        overrides = [e for e in trail if e["event"] == "override"]
        methods: dict[str, int] = defaultdict(int)
        for e in decisions:
            methods[e["method"]] += 1
        return {
            "total": len(trail),
            "decisions": len(decisions),
            "overrides": len(overrides),
            "methods_used": dict(methods),
        }
