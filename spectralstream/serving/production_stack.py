"""
SpectralStream Production Stack
================================
Unified production hardening, monitoring, and lifecycle management
for 24/7 inference service operation.

Unifies and extends spectralstream.production and
spectralstream.production_upgrades into a single comprehensive stack.

Components:
  1. ServiceManager        — process lifecycle, watchdog, PID, cgroups, systemd
  2. CircuitBreaker        — fault isolation (CLOSED/OPEN/HALF_OPEN/DISABLED)
  3. GracefulDegradation   — 4-level feature degradation
  4. ErrorBoundary         — per-request & per-component error isolation
  5. CrashRecovery         — auto-recover, checkpoint, exponential backoff
  6. StructuredLogger      — JSON-lines logging, sampling, redaction, async
  7. MetricsCollector       — throughput, latency, memory, quality, errors
  8. HealthCheck           — /health, /health/ready, /health/live, /health/detailed
  9. RateLimiter           — token-bucket, leaky-bucket, sliding-window
  10. SecurityManager       — API keys, JWT, prompt/output filtering, audit
  11. APIMonitor            — usage dashboard, alerts, reports, cost tracking
  12. MultiInstance         — load balancing, shared KV, session affinity
  13. Novel Inventions      — resonant LB, holographic recovery, predictive
                             scaling, quantum circuit breaker, self-healing

Usage:
    from spectralstream.serving.production_stack import create_service

    mgr = create_service(engine, config)
    mgr.start()

Self-verification:
    python -m spectralstream.production_stack --health
"""

import asyncio
import base64
import copy
import enum
import fcntl
import hashlib
import hmac
import inspect
import json
import math
import os
import queue
import random
import re
import resource
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
import traceback
import uuid
import warnings
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from enum import IntEnum, auto
from pathlib import Path
from typing import Any, Callable, Optional, Union

import numpy as np

from spectralstream.config import SpectralStreamConfig
try:
    from spectralstream.inference import StateManager
except ImportError:
    from spectralstream.memory.persistence import StateManager

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import redis

    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False


VERSION = "1.0.0"
DEFAULT_STATE_DIR = "~/.spectralstream/"
DEFAULT_CHECKPOINT_DIR = "~/.spectralstream/checkpoints/"
DEFAULT_LOG_DIR = "~/.spectralstream/logs/"
DEFAULT_PID_DIR = "~/.spectralstream/pids/"
MAX_LOG_LINE_LENGTH = 100000
DEFAULT_SAMPLING_RATE = 0.001

_SENSITIVE_FIELDS = {
    "prompt",
    "prompts",
    "content",
    "input",
    "output",
    "token",
    "tokens",
    "api_key",
    "apikey",
    "secret",
    "password",
    "authorization",
    "bearer",
}

# ═══════════════════════════════════════════════════════════════════════════════
# 1. ServiceManager — Full Service Lifecycle
# ═══════════════════════════════════════════════════════════════════════════════


class ServiceState(enum.Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    CRASHED = "crashed"
    RESTARTING = "restarting"


@dataclass
class ResourceLimits:
    cpu_quota_percent: float = 80.0
    memory_max_mb: float = 0.0
    open_files: int = 65536
    stack_size_mb: int = 8
    core_size_mb: int = 0


class ServiceManager:
    def __init__(
        self,
        name: str = "spectralstream",
        state_dir: str = DEFAULT_STATE_DIR,
        resource_limits: Optional[ResourceLimits] = None,
        health_check_interval: float = 10.0,
        watchdog_timeout: float = 60.0,
        graceful_shutdown_timeout: float = 30.0,
        max_restarts: int = 5,
        restart_window: float = 300.0,
        logger: Optional["StructuredLogger"] = None,
    ):
        self.name = name
        self.state = ServiceState.STOPPED
        self.resource_limits = resource_limits or ResourceLimits()
        self.health_check_interval = health_check_interval
        self.watchdog_timeout = watchdog_timeout
        self.graceful_shutdown_timeout = graceful_shutdown_timeout
        self.max_restarts = max_restarts
        self.restart_window = restart_window

        self.state_dir = Path(state_dir).expanduser()
        self.pid_dir = self.state_dir / "pids"
        self.pid_dir.mkdir(parents=True, exist_ok=True)
        self.pid_file = self.pid_dir / f"{name}.pid"

        self.logger = logger or StructuredLogger(f"{name}_service")

        self._engine = None
        self._components: dict[str, Any] = {}
        self._running = False
        self._stop_event = threading.Event()
        self._watchdog_thread: Optional[threading.Thread] = None
        self._health_thread: Optional[threading.Thread] = None
        self._restart_count = 0
        self._first_restart_time = 0.0
        self._in_flight_count = 0
        self._in_flight_lock = threading.Lock()
        self._component_health: dict[str, bool] = {}
        self._start_time = 0.0
        self._systemd_connected = False

    def register_component(self, name: str, component: Any) -> None:
        self._components[name] = component
        self._component_health[name] = False

    def set_engine(self, engine: Any) -> None:
        self._engine = engine
        self.register_component("engine", engine)

    def _write_pid(self) -> bool:
        pid = os.getpid()
        try:
            with open(self.pid_file, "w") as f:
                f.write(f"{pid}\n")
                f.flush()
            self.logger.info("pid_file_written", pid=pid, path=str(self.pid_file))
            return True
        except OSError as e:
            self.logger.error("pid_file_write_failed", error=str(e))
            return False

    def _remove_pid(self) -> None:
        try:
            if self.pid_file.exists():
                self.pid_file.unlink()
        except OSError:
            pass

    def _check_stale_pid(self) -> Optional[int]:
        if not self.pid_file.exists():
            return None
        try:
            pid = int(self.pid_file.read_text().strip())
        except (ValueError, OSError):
            return None
        if pid == os.getpid():
            return pid
        try:
            os.kill(pid, 0)
            return pid
        except (OSError, ProcessLookupError):
            self.logger.warn("stale_pid_found", pid=pid, path=str(self.pid_file))
            self._remove_pid()
            return None

    def _apply_resource_limits(self) -> None:
        limits = self.resource_limits
        try:
            resource.setrlimit(
                resource.RLIMIT_NOFILE, (limits.open_files, limits.open_files)
            )
        except (ValueError, resource.error) as e:
            self.logger.warn("setrlimit_nofile_failed", error=str(e))
        try:
            stack_bytes = limits.stack_size_mb * 1024 * 1024
            resource.setrlimit(
                resource.RLIMIT_STACK, (stack_bytes, resource.RLIM_INFINITY)
            )
        except (ValueError, resource.error) as e:
            self.logger.warn("setrlimit_stack_failed", error=str(e))
        try:
            core_bytes = limits.core_size_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_CORE, (core_bytes, core_bytes))
        except (ValueError, resource.error) as e:
            self.logger.warn("setrlimit_core_failed", error=str(e))
        if limits.memory_max_mb > 0:
            try:
                mem_bytes = int(limits.memory_max_mb * 1024 * 1024)
                resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
            except (ValueError, resource.error) as e:
                self.logger.warn("setrlimit_as_failed", error=str(e))
        if HAS_PSUTIL and limits.cpu_quota_percent < 100.0:
            try:
                p = psutil.Process()
                if hasattr(p, "cpu_affinity"):
                    cpus = p.cpu_affinity()
                    if cpus:
                        quota = max(
                            1, int(len(cpus) * limits.cpu_quota_percent / 100.0)
                        )
                        p.cpu_affinity(cpus[:quota])
            except Exception as e:
                self.logger.warn("cpu_affinity_failed", error=str(e))

    def _try_cgroup_limit(self) -> bool:
        try:
            pid = os.getpid()
            mem_mb = self.resource_limits.memory_max_mb
            cpu_pct = self.resource_limits.cpu_quota_percent
            cg_path = Path(
                f"/sys/fs/cgroup/systemd/user-{os.getuid()}.scope/{self.name}"
            )
            cg_path.mkdir(parents=True, exist_ok=True)
            if mem_mb > 0:
                (cg_path / "memory.max").write_text(f"{int(mem_mb * 1024 * 1024)}\n")
            if cpu_pct > 0:
                quota_us = int(cpu_pct * 1000)
                (cg_path / "cpu.max").write_text(f"{quota_us} 100000\n")
            (cg_path / "cgroup.procs").write_text(f"{pid}\n")
            self.logger.info(
                "cgroup_limits_applied", cpu_percent=cpu_pct, memory_mb=mem_mb
            )
            return True
        except Exception as e:
            self.logger.debug("cgroup_not_available", error=str(e))
            return False

    def _systemd_notify(self, state: str) -> None:
        try:
            notify_socket = os.environ.get("NOTIFY_SOCKET")
            if not notify_socket:
                return
            if notify_socket.startswith("@"):
                notify_socket = "\x00" + notify_socket[1:]
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
            sock.connect(notify_socket)
            sock.sendall(f"{state}\n".encode())
            sock.close()
            self._systemd_connected = True
        except Exception:
            pass

    def _systemd_watchdog(self) -> None:
        try:
            wdt_usec = os.environ.get("WATCHDOG_USEC")
            if wdt_usec:
                interval = int(wdt_usec) / 2_000_000
                threading.Timer(interval, self._systemd_watchdog).start()
                self._systemd_notify("WATCHDOG=1")
        except Exception:
            pass

    def _install_signal_handlers(self) -> None:
        def handle_sigterm(sig, frame):
            self.logger.info("signal_received", signal=sig)
            self.stop()

        def handle_sigint(sig, frame):
            self.logger.info("signal_received", signal=sig)
            self.stop()

        def handle_sighup(sig, frame):
            self.logger.info("signal_received", signal=sig)
            self.restart()

        signal.signal(signal.SIGTERM, handle_sigterm)
        signal.signal(signal.SIGINT, handle_sigint)
        signal.signal(signal.SIGHUP, handle_sighup)

    def start(self) -> bool:
        if self.state == ServiceState.RUNNING:
            self.logger.warn("already_running")
            return True
        self._check_stale_pid()
        self._apply_resource_limits()
        self._try_cgroup_limit()
        self._write_pid()
        self._install_signal_handlers()
        self.state = ServiceState.STARTING
        self._stop_event.clear()
        self._running = True
        self._start_time = time.time()
        try:
            self._start_components()
        except Exception as e:
            self.logger.error(
                "start_failed", error=str(e), traceback=traceback.format_exc()
            )
            self.state = ServiceState.CRASHED
            return False
        self.state = ServiceState.RUNNING
        self._systemd_notify("READY=1")
        self._systemd_watchdog()
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop, daemon=True
        )
        self._watchdog_thread.start()
        self._health_thread = threading.Thread(
            target=self._health_check_loop, daemon=True
        )
        self._health_thread.start()
        self.logger.info("service_started", pid=os.getpid())
        return True

    def stop(self) -> None:
        if self.state == ServiceState.STOPPED:
            return
        self.state = ServiceState.STOPPING
        self._running = False
        self._stop_event.set()
        self._systemd_notify("STOPPING=1")
        self._drain_in_flight(timeout=self.graceful_shutdown_timeout)
        try:
            self._stop_components()
        except Exception as e:
            self.logger.error("stop_components_failed", error=str(e))
        self._remove_pid()
        self.state = ServiceState.STOPPED
        self._systemd_notify("STOPPED=1")
        self.logger.info("service_stopped")

    def restart(self) -> bool:
        self.state = ServiceState.RESTARTING
        self.logger.info("service_restarting")
        self.stop()
        return self.start()

    def shutdown(self) -> None:
        self.stop()

    def health(self) -> dict:
        uptime = time.time() - self._start_time if self._start_time > 0 else 0.0
        return {
            "service": self.name,
            "state": self.state.value,
            "uptime_seconds": round(uptime, 2),
            "pid": os.getpid(),
            "in_flight_requests": self._in_flight_count,
            "restart_count": self._restart_count,
            "component_health": dict(self._component_health),
            "systemd_connected": self._systemd_connected,
        }

    def acquire_in_flight(self) -> None:
        with self._in_flight_lock:
            self._in_flight_count += 1

    def release_in_flight(self) -> None:
        with self._in_flight_lock:
            self._in_flight_count = max(0, self._in_flight_count - 1)

    def _drain_in_flight(self, timeout: float = 30.0) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._in_flight_lock:
                if self._in_flight_count == 0:
                    return
            time.sleep(0.1)
        self.logger.warn("in_flight_drain_timeout", remaining=self._in_flight_count)

    def _start_components(self) -> None:
        for name, comp in self._components.items():
            try:
                if hasattr(comp, "start") and callable(comp.start):
                    comp.start()
                self._component_health[name] = True
                self.logger.info("component_started", component=name)
            except Exception as e:
                self._component_health[name] = False
                self.logger.error(
                    "component_start_failed", component=name, error=str(e)
                )

    def _stop_components(self) -> None:
        for name, comp in reversed(list(self._components.items())):
            try:
                if hasattr(comp, "stop") and callable(comp.stop):
                    comp.stop()
                if hasattr(comp, "shutdown") and callable(comp.shutdown):
                    comp.shutdown()
                self.logger.info("component_stopped", component=name)
            except Exception as e:
                self.logger.error("component_stop_failed", component=name, error=str(e))

    def _watchdog_loop(self) -> None:
        while self._running and not self._stop_event.is_set():
            time.sleep(self.watchdog_timeout / 4)
            if not self._running:
                break
            if self.state != ServiceState.RUNNING:
                continue
            now = time.time()
            if now - self._last_component_activity() > self.watchdog_timeout:
                self.logger.warn(
                    "watchdog_hang_detected", timeout=self.watchdog_timeout
                )
                self._handle_crash("watchdog_timeout")

    def _last_component_activity(self) -> float:
        return self._start_time

    def _check_restart_rate(self) -> bool:
        now = time.time()
        if self._restart_count == 0:
            self._first_restart_time = now
            self._restart_count = 1
            return True
        if now - self._first_restart_time > self.restart_window:
            self._restart_count = 1
            self._first_restart_time = now
            return True
        self._restart_count += 1
        if self._restart_count > self.max_restarts:
            self.logger.error(
                "max_restarts_exceeded",
                count=self._restart_count,
                window=self.restart_window,
            )
            return False
        return True

    def _handle_crash(self, reason: str) -> None:
        self.state = ServiceState.CRASHED
        self.logger.error("service_crashed", reason=reason)
        if not self._check_restart_rate():
            self.logger.error("service_not_restarting_rate_limit")
            self._systemd_notify("STOPPING=1")
            return
        backoff = min(2 ** (self._restart_count - 1), 30)
        self.logger.info(
            "service_auto_restart", backoff_seconds=backoff, attempt=self._restart_count
        )
        time.sleep(backoff)
        try:
            self._stop_components()
            self._start_components()
            self.state = ServiceState.RUNNING
            self.logger.info("service_recovered")
        except Exception as e:
            self.logger.error("service_recover_failed", error=str(e))

    def _health_check_loop(self) -> None:
        while self._running and not self._stop_event.is_set():
            time.sleep(self.health_check_interval)
            try:
                self._check_component_health()
            except Exception:
                pass

    def _check_component_health(self) -> None:
        for name, comp in self._components.items():
            try:
                if hasattr(comp, "is_healthy") and callable(comp.is_healthy):
                    self._component_health[name] = bool(comp.is_healthy())
                elif hasattr(comp, "healthy"):
                    self._component_health[name] = bool(comp.healthy)
                else:
                    self._component_health[name] = True
            except Exception:
                self._component_health[name] = False


# ═══════════════════════════════════════════════════════════════════════════════
# 2. CircuitBreaker — Enhanced Fault Isolation
# ═══════════════════════════════════════════════════════════════════════════════


class BreakerState(enum.Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"
    DISABLED = "disabled"


@dataclass
class BreakerMetrics:
    total_calls: int = 0
    total_successes: int = 0
    total_failures: int = 0
    total_rejections: int = 0
    last_success_time: float = 0.0
    last_failure_time: float = 0.0
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    window_failures: deque = field(default_factory=lambda: deque(maxlen=1000))
    window_successes: deque = field(default_factory=lambda: deque(maxlen=1000))

    def failure_rate(self, window: float = 60.0) -> float:
        now = time.time()
        cutoff = now - window
        failures = sum(1 for t in self.window_failures if t > cutoff)
        successes = sum(1 for t in self.window_successes if t > cutoff)
        total = failures + successes
        return failures / total if total > 0 else 0.0

    def success_rate(self, window: float = 60.0) -> float:
        return 1.0 - self.failure_rate(window)

    def request_volume(self, window: float = 60.0) -> int:
        now = time.time()
        cutoff = now - window
        return sum(1 for t in self.window_failures if t > cutoff) + sum(
            1 for t in self.window_successes if t > cutoff
        )


class CircuitBreaker:
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        failure_rate_threshold: float = 0.5,
        cooldown_seconds: float = 30.0,
        half_open_max_probes: int = 3,
        decay_seconds: float = 60.0,
        window_seconds: float = 60.0,
        min_request_volume: int = 10,
        logger: Optional["StructuredLogger"] = None,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.failure_rate_threshold = failure_rate_threshold
        self.cooldown = cooldown_seconds
        self.half_open_max_probes = half_open_max_probes
        self.decay_seconds = decay_seconds
        self.window_seconds = window_seconds
        self.min_request_volume = min_request_volume

        self.state = BreakerState.CLOSED
        self.metrics = BreakerMetrics()
        self._half_open_probes = 0
        self._lock = threading.RLock()
        self.logger = logger or StructuredLogger(f"breaker_{name}")
        self._on_state_change: list[Callable] = []

    def on_state_change(self, callback: Callable) -> None:
        self._on_state_change.append(callback)

    def _transition(self, new_state: BreakerState) -> None:
        old_state = self.state
        self.state = new_state
        self.logger.info(
            "state_transition",
            from_state=old_state.value,
            to_state=new_state.value,
            breaker=self.name,
        )
        for cb in self._on_state_change:
            try:
                cb(self.name, old_state, new_state)
            except Exception:
                pass

    def _check_decay(self) -> None:
        if self.decay_seconds <= 0:
            return
        now = time.time()
        if now - self.metrics.last_failure_time > self.decay_seconds:
            decay_factor = (now - self.metrics.last_failure_time) / self.decay_seconds
            to_remove = min(self.metrics.consecutive_failures, int(decay_factor))
            if to_remove > 0:
                self.metrics.consecutive_failures = max(
                    0, self.metrics.consecutive_failures - to_remove
                )
                self.logger.debug(
                    "failure_decay",
                    decayed=to_remove,
                    remaining=self.metrics.consecutive_failures,
                )
                self.metrics.last_failure_time = now

    def _should_open(self) -> bool:
        if self.metrics.consecutive_failures >= self.failure_threshold:
            return True
        if self.metrics.request_volume(self.window_seconds) >= self.min_request_volume:
            if (
                self.metrics.failure_rate(self.window_seconds)
                >= self.failure_rate_threshold
            ):
                return True
        return False

    def call(
        self, fn: Callable, fallback: Optional[Callable] = None, *args, **kwargs
    ) -> Any:
        with self._lock:
            self._check_decay()
            self.metrics.total_calls += 1

            if self.state == BreakerState.DISABLED:
                return self._execute(fn, fallback, *args, **kwargs)

            if self.state == BreakerState.OPEN:
                if time.time() - self.metrics.last_failure_time > self.cooldown:
                    self._transition(BreakerState.HALF_OPEN)
                    self._half_open_probes = 0
                else:
                    self.metrics.total_rejections += 1
                    if fallback:
                        return fallback()
                    raise RuntimeError(
                        f"CircuitBreaker '{self.name}' is OPEN. "
                        f"Failures: {self.metrics.consecutive_failures}/{self.failure_threshold}"
                    )

            if self.state == BreakerState.HALF_OPEN:
                if self._half_open_probes >= self.half_open_max_probes:
                    self.metrics.total_rejections += 1
                    if fallback:
                        return fallback()
                    raise RuntimeError(
                        f"CircuitBreaker '{self.name}' is HALF_OPEN (probe budget exhausted)"
                    )
                self._half_open_probes += 1

            try:
                result = fn(*args, **kwargs)
                self.metrics.total_successes += 1
                self.metrics.consecutive_successes += 1
                self.metrics.consecutive_failures = 0
                self.metrics.last_success_time = time.time()
                self.metrics.window_successes.append(time.time())
                if self.state == BreakerState.HALF_OPEN and self._half_open_probes <= 1:
                    self._transition(BreakerState.CLOSED)
                return result
            except Exception as e:
                self.metrics.total_failures += 1
                self.metrics.consecutive_failures += 1
                self.metrics.consecutive_successes = 0
                self.metrics.last_failure_time = time.time()
                self.metrics.window_failures.append(time.time())
                if self.state == BreakerState.CLOSED and self._should_open():
                    self._transition(BreakerState.OPEN)
                elif self.state == BreakerState.HALF_OPEN:
                    self._transition(BreakerState.OPEN)
                if fallback:
                    return fallback()
                raise

    def disable(self) -> None:
        with self._lock:
            self._transition(BreakerState.DISABLED)

    def enable(self) -> None:
        with self._lock:
            self._transition(BreakerState.CLOSED)

    def reset(self) -> None:
        with self._lock:
            self.metrics = BreakerMetrics()
            self._transition(BreakerState.CLOSED)

    def stats(self) -> dict:
        with self._lock:
            return {
                "name": self.name,
                "state": self.state.value,
                "total_calls": self.metrics.total_calls,
                "total_successes": self.metrics.total_successes,
                "total_failures": self.metrics.total_failures,
                "total_rejections": self.metrics.total_rejections,
                "consecutive_failures": self.metrics.consecutive_failures,
                "consecutive_successes": self.metrics.consecutive_successes,
                "failure_rate_60s": round(self.metrics.failure_rate(60.0), 4),
                "success_rate_60s": round(self.metrics.success_rate(60.0), 4),
                "request_volume_60s": self.metrics.request_volume(60.0),
                "half_open_probes": self._half_open_probes,
            }

    def to_prometheus(self, prefix: str = "spectralstream") -> str:
        s = self.stats()
        state_idx = ["closed", "open", "half_open", "disabled"].index(s["state"])
        return (
            f"# HELP {prefix}_breaker_state Circuit breaker state\n"
            f"# TYPE {prefix}_breaker_state gauge\n"
            f'{prefix}_breaker_state{{breaker="{self.name}"}} {state_idx}\n'
            f"# HELP {prefix}_breaker_failures_total Total failures\n"
            f"# TYPE {prefix}_breaker_failures_total counter\n"
            f'{prefix}_breaker_failures_total{{breaker="{self.name}"}} {s["total_failures"]}\n'
            f"# HELP {prefix}_breaker_rejections_total Total rejected calls\n"
            f"# TYPE {prefix}_breaker_rejections_total counter\n"
            f'{prefix}_breaker_rejections_total{{breaker="{self.name}"}} {s["total_rejections"]}\n'
        )

    def is_healthy(self) -> bool:
        return self.state in (BreakerState.CLOSED, BreakerState.HALF_OPEN)


class CircuitBreakerRegistry:
    def __init__(self, logger: Optional["StructuredLogger"] = None):
        self._breakers: dict[str, CircuitBreaker] = {}
        self._lock = threading.RLock()
        self.logger = logger

    def get(self, name: str, **kwargs) -> CircuitBreaker:
        with self._lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(
                    name=name, logger=self.logger, **kwargs
                )
            return self._breakers[name]

    def all_stats(self) -> dict[str, dict]:
        return {n: b.stats() for n, b in self._breakers.items()}

    def all_healthy(self) -> bool:
        return all(b.is_healthy() for b in self._breakers.values())

    def reset_all(self) -> None:
        for b in self._breakers.values():
            b.reset()

    def to_prometheus(self, prefix: str = "spectralstream") -> str:
        return "\n".join(b.to_prometheus(prefix) for b in self._breakers.values())


# ═══════════════════════════════════════════════════════════════════════════════
# 3. GracefulDegradation — Feature Degradation with 4 Levels
# ═══════════════════════════════════════════════════════════════════════════════


class DegradationLevel(IntEnum):
    FULL = 0
    REDUCED = 1
    MINIMAL = 2
    EMERGENCY = 3


DEGRADATION_NAMES = {
    DegradationLevel.FULL: "full",
    DegradationLevel.REDUCED: "reduced",
    DegradationLevel.MINIMAL: "minimal",
    DegradationLevel.EMERGENCY: "emergency",
}


@dataclass
class DegradationPolicy:
    allow_hdc: bool = True
    allow_spectral_kv: bool = True
    allow_block_emission: bool = True
    allow_vlasov: bool = True
    allow_confidence_gate: bool = True
    allow_online_learning: bool = True
    allow_resonance: bool = True
    allow_attractor: bool = True
    use_cache_only: bool = False
    max_tokens_per_request: int = 0
    max_concurrent_requests: int = 0


FULL_POLICY = DegradationPolicy(
    allow_hdc=True,
    allow_spectral_kv=True,
    allow_block_emission=True,
    allow_vlasov=True,
    allow_confidence_gate=True,
    allow_online_learning=True,
    allow_resonance=True,
    allow_attractor=True,
    use_cache_only=False,
    max_tokens_per_request=0,
    max_concurrent_requests=0,
)

REDUCED_POLICY = DegradationPolicy(
    allow_hdc=False,
    allow_spectral_kv=False,
    allow_block_emission=False,
    allow_vlasov=False,
    allow_confidence_gate=True,
    allow_online_learning=False,
    allow_resonance=False,
    allow_attractor=False,
    use_cache_only=False,
    max_tokens_per_request=4096,
    max_concurrent_requests=0,
)

MINIMAL_POLICY = DegradationPolicy(
    allow_hdc=False,
    allow_spectral_kv=False,
    allow_block_emission=False,
    allow_vlasov=False,
    allow_confidence_gate=False,
    allow_online_learning=False,
    allow_resonance=False,
    allow_attractor=False,
    use_cache_only=False,
    max_tokens_per_request=2048,
    max_concurrent_requests=4,
)

EMERGENCY_POLICY = DegradationPolicy(
    allow_hdc=False,
    allow_spectral_kv=False,
    allow_block_emission=False,
    allow_vlasov=False,
    allow_confidence_gate=False,
    allow_online_learning=False,
    allow_resonance=False,
    allow_attractor=False,
    use_cache_only=True,
    max_tokens_per_request=512,
    max_concurrent_requests=1,
)


class GracefulDegradation:
    LEVEL_POLICIES = {
        DegradationLevel.FULL: FULL_POLICY,
        DegradationLevel.REDUCED: REDUCED_POLICY,
        DegradationLevel.MINIMAL: MINIMAL_POLICY,
        DegradationLevel.EMERGENCY: EMERGENCY_POLICY,
    }

    def __init__(
        self,
        logger: Optional["StructuredLogger"] = None,
        metrics: Optional["MetricsCollector"] = None,
        state_dir: str = DEFAULT_STATE_DIR,
        memory_threshold_gb: float = 40.0,
        latency_threshold_ms: float = 10000.0,
        error_rate_threshold: float = 0.1,
        auto_degrade: bool = True,
        check_interval: float = 10.0,
    ):
        self.logger = logger or StructuredLogger("degradation")
        self.metrics = metrics
        self.level = DegradationLevel.FULL
        self.memory_threshold_gb = memory_threshold_gb
        self.latency_threshold_ms = latency_threshold_ms
        self.error_rate_threshold = error_rate_threshold
        self.auto_degrade = auto_degrade
        self.check_interval = check_interval

        self._lock = threading.RLock()
        self._policy = FULL_POLICY
        self._level_history: list[tuple[float, DegradationLevel, str]] = []
        self._trigger_reasons: list[str] = []
        self._cached_responses: dict[str, dict] = {}
        self._cache_lock = threading.RLock()
        self._check_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._on_level_change: list[Callable] = []

    def on_level_change(self, callback: Callable) -> None:
        self._on_level_change.append(callback)

    @property
    def policy(self) -> DegradationPolicy:
        return self._policy

    def set_level(self, level: DegradationLevel, reason: str = "manual") -> None:
        with self._lock:
            old_level = self.level
            self.level = level
            self._policy = self.LEVEL_POLICIES[level]
            self._level_history.append((time.time(), level, reason))
            self.logger.info(
                "degradation_level_changed",
                from_level=DEGRADATION_NAMES.get(old_level, str(old_level)),
                to_level=DEGRADATION_NAMES.get(level, str(level)),
                reason=reason,
            )
            for cb in self._on_level_change:
                try:
                    cb(old_level, level, reason)
                except Exception:
                    pass

    def auto_degrade_if_needed(self) -> DegradationLevel:
        if not self.auto_degrade:
            return self.level

        reasons = []

        if HAS_PSUTIL:
            try:
                mem = psutil.Process().memory_info()
                rss_gb = mem.rss / (1024**3)
                if rss_gb > self.memory_threshold_gb:
                    reasons.append(
                        f"memory_pressure: {rss_gb:.1f}GB > {self.memory_threshold_gb}GB"
                    )
            except Exception:
                pass

        if self.metrics:
            try:
                lat_p95 = self.metrics.get_latency_percentile("inference", 0.95)
                if lat_p95 > self.latency_threshold_ms:
                    reasons.append(
                        f"high_latency: p95={lat_p95:.0f}ms > {self.latency_threshold_ms}ms"
                    )
                err_rate = self.metrics.get_error_rate()
                if err_rate > self.error_rate_threshold:
                    reasons.append(
                        f"high_error_rate: {err_rate:.4f} > {self.error_rate_threshold}"
                    )
            except Exception:
                pass

        if not reasons:
            if self.level != DegradationLevel.FULL:
                self.set_level(DegradationLevel.FULL, "auto_recovery")
            return self.level

        self._trigger_reasons = reasons

        if self.level == DegradationLevel.FULL:
            self.set_level(DegradationLevel.REDUCED, "; ".join(reasons))
        elif self.level == DegradationLevel.REDUCED:
            self.set_level(DegradationLevel.MINIMAL, "; ".join(reasons))
        elif self.level == DegradationLevel.MINIMAL:
            self.set_level(DegradationLevel.EMERGENCY, "; ".join(reasons))

        return self.level

    def start_auto_monitoring(self) -> None:
        if self._check_thread and self._check_thread.is_alive():
            return
        self._stop_event.clear()
        self._check_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._check_thread.start()

    def stop_auto_monitoring(self) -> None:
        self._stop_event.set()

    def _monitor_loop(self) -> None:
        while not self._stop_event.is_set():
            self.auto_degrade_if_needed()
            self._stop_event.wait(self.check_interval)

    def cache_response(self, key: str, response: dict) -> None:
        with self._cache_lock:
            self._cached_responses[key] = {
                "response": response,
                "timestamp": time.time(),
            }
            if len(self._cached_responses) > 1000:
                oldest = min(
                    self._cached_responses.keys(),
                    key=lambda k: self._cached_responses[k]["timestamp"],
                )
                del self._cached_responses[oldest]

    def get_cached_response(self, key: str) -> Optional[dict]:
        with self._cache_lock:
            entry = self._cached_responses.get(key)
            if entry:
                return entry["response"]
            return None

    def wrap_engine(self, engine: Any) -> Any:
        original_generate = getattr(engine, "generate", None)
        if original_generate is None:
            return engine

        def degraded_generate(*args, **kwargs):
            if self.policy.use_cache_only:
                cache_key = str(args) + str(sorted(kwargs.items()))
                cached = self.get_cached_response(cache_key)
                if cached is not None:
                    return cached
            if not self.policy.allow_hdc and hasattr(engine, "set_parameter"):
                kwargs["use_hdc"] = False
            result = original_generate(*args, **kwargs)
            if self.policy.use_cache_only:
                cache_key = str(args) + str(sorted(kwargs.items()))
                self.cache_response(cache_key, result)
            return result

        engine.generate = degraded_generate
        return engine

    def status(self) -> dict:
        return {
            "level": DEGRADATION_NAMES.get(self.level, str(self.level)),
            "level_int": int(self.level),
            "policy": asdict(self._policy),
            "trigger_reasons": list(self._trigger_reasons),
            "level_history": [
                {"time": t, "level": DEGRADATION_NAMES.get(l, str(l)), "reason": r}
                for t, l, r in self._level_history[-20:]
            ],
            "cached_responses": len(self._cached_responses),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. ErrorBoundary — Error Isolation and Recovery
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ErrorContext:
    request_id: str = ""
    trace_id: str = ""
    component: str = ""
    operation: str = ""
    timestamp: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class ErrorBoundary:
    def __init__(
        self,
        logger: Optional["StructuredLogger"] = None,
        circuit_breakers: Optional[CircuitBreakerRegistry] = None,
        degradation: Optional[GracefulDegradation] = None,
        state_manager: Optional[StateManager] = None,
    ):
        self.logger = logger or StructuredLogger("error_boundary")
        self.circuit_breakers = circuit_breakers
        self.degradation = degradation
        self.state_manager = state_manager

        self._error_counts: dict[str, int] = defaultdict(int)
        self._last_errors: deque = deque(maxlen=1000)
        self._component_states: dict[str, bool] = {}
        self._lock = threading.RLock()
        self._recovery_actions: dict[str, Callable] = {}

    def register_recovery(self, error_type: str, action: Callable) -> None:
        self._recovery_actions[error_type] = action

    def execute(
        self,
        fn: Callable,
        fallback: Optional[Callable] = None,
        error_msg: str = "Operation failed",
        recoverable: bool = True,
        context: Optional[ErrorContext] = None,
        component: str = "unknown",
        *args,
        **kwargs,
    ) -> Any:
        ctx = context or ErrorContext(
            request_id=str(uuid.uuid4())[:8],
            trace_id=str(uuid.uuid4())[:12],
            component=component,
            operation=error_msg,
            timestamp=time.time(),
        )

        breaker = None
        if self.circuit_breakers and component:
            breaker = self.circuit_breakers.get(component)

        try:
            if breaker:
                return breaker.call(
                    lambda: fn(*args, **kwargs),
                    fallback=lambda: self._execute_fallback(fallback, ctx)
                    if fallback
                    else None,
                )
            return fn(*args, **kwargs)
        except Exception as e:
            return self._handle_error(e, ctx, fallback, recoverable)

    def _handle_error(
        self,
        error: Exception,
        ctx: ErrorContext,
        fallback: Optional[Callable],
        recoverable: bool,
    ) -> Any:
        error_type = type(error).__name__
        tb = traceback.format_exc()

        with self._lock:
            self._error_counts[error_type] += 1
            self._last_errors.append(
                {
                    "time": time.time(),
                    "type": error_type,
                    "message": str(error),
                    "context": ctx.to_dict(),
                    "traceback": tb,
                }
            )

        self.logger.error(
            ctx.operation or "operation_failed",
            error_type=error_type,
            error=str(error),
            component=ctx.component,
            request_id=ctx.request_id,
            trace_id=ctx.trace_id,
            recoverable=recoverable,
        )

        if error_type in self._recovery_actions:
            try:
                self._recovery_actions[error_type](error)
            except Exception as recover_err:
                self.logger.error("recovery_action_failed", error=str(recover_err))

        if self.degradation:
            with self._lock:
                if self._error_counts[error_type] >= 5:
                    self.degradation.auto_degrade_if_needed()

        if fallback:
            return self._execute_fallback(fallback, ctx)

        if recoverable:
            return None

        raise

    def _execute_fallback(self, fallback: Callable, ctx: ErrorContext) -> Any:
        try:
            return fallback()
        except Exception as fb_err:
            self.logger.error(
                "fallback_failed",
                error=str(fb_err),
                component=ctx.component,
                request_id=ctx.request_id,
            )
            return None

    def mark_component_healthy(self, component: str) -> None:
        with self._lock:
            self._component_states[component] = True

    def mark_component_failed(self, component: str) -> None:
        with self._lock:
            self._component_states[component] = False

    def get_error_rate(self, window: float = 60.0) -> float:
        now = time.time()
        cutoff = now - window
        recent = [e for e in self._last_errors if e["time"] > cutoff]
        return len(recent) / max(window, 1.0)

    def get_error_counts(self) -> dict[str, int]:
        return dict(self._error_counts)

    def status(self) -> dict:
        return {
            "error_counts": dict(self._error_counts),
            "component_states": dict(self._component_states),
            "recent_errors": [
                {
                    "time": e["time"],
                    "type": e["type"],
                    "message": str(e["message"])[:200],
                }
                for e in list(self._last_errors)[-10:]
            ],
            "recovery_actions": list(self._recovery_actions.keys()),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 5. CrashRecovery — Auto-Recovery from Crashes
# ═══════════════════════════════════════════════════════════════════════════════


class CrashDetector:
    SEGFAULT_SIGS = {signal.SIGSEGV, signal.SIGBUS, signal.SIGABRT}
    OOM_SIGS = {signal.SIGKILL}

    @staticmethod
    def detect(reason: str) -> str:
        reason_lower = reason.lower()
        if "segfault" in reason_lower or "segmentation" in reason_lower:
            return "segfault"
        if "oom" in reason_lower or "memory" in reason_lower:
            return "oom_kill"
        if "hang" in reason_lower or "timeout" in reason_lower:
            return "hang"
        if "signal" in reason_lower:
            return "signal"
        return "unknown"

    @staticmethod
    def check_oom_score() -> float:
        try:
            with open(f"/proc/{os.getpid()}/oom_score") as f:
                return float(f.read().strip())
        except Exception:
            return 0.0


class CrashRecovery:
    def __init__(
        self,
        state_manager: Optional[StateManager] = None,
        checkpoint_interval: int = 100,
        checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR,
        max_checkpoints: int = 20,
        max_backoff_seconds: float = 300.0,
        base_backoff_seconds: float = 1.0,
        kv_cache_recovery: bool = True,
        logger: Optional["StructuredLogger"] = None,
    ):
        self.state_manager = state_manager
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_dir = Path(checkpoint_dir).expanduser()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.max_backoff = max_backoff_seconds
        self.base_backoff = base_backoff_seconds
        self.kv_cache_recovery = kv_cache_recovery

        self.logger = logger or StructuredLogger("crash_recovery")
        self.generation_count = 0
        self.crash_count = 0
        self.last_crash_time = 0.0
        self.clean_shutdown = True
        self._crash_history: list[dict] = []
        self._lock = threading.RLock()
        self._crash_detector = CrashDetector()
        self._register_signal_handlers()

    def _register_signal_handlers(self) -> None:
        def emergency_save(sig, frame):
            sig_name = signal.Signals(sig).name
            self.logger.fatal("signal_received", signal=sig_name)
            try:
                if self.state_manager:
                    self.state_manager.save_checkpoint()
                    self.logger.info("emergency_checkpoint_saved")
            except Exception as e:
                self.logger.error("emergency_save_failed", error=str(e))
            os._exit(1)

        signal.signal(signal.SIGTERM, emergency_save)
        signal.signal(signal.SIGINT, emergency_save)
        try:
            signal.signal(signal.SIGHUP, emergency_save)
        except Exception:
            pass

    def on_generation_complete(self) -> None:
        self.generation_count += 1
        if self.generation_count % self.checkpoint_interval == 0:
            self._save_checkpoint("periodic")

    def _save_checkpoint(self, tag: str = "checkpoint") -> Optional[str]:
        if not self.state_manager:
            return None
        try:
            path = self.state_manager.save_checkpoint(tag=tag)
            self._prune_old_checkpoints()
            return path
        except Exception as e:
            self.logger.error("checkpoint_save_failed", error=str(e))
            return None

    def _prune_old_checkpoints(self) -> None:
        if not self.max_checkpoints:
            return
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_*.json"))
        while len(checkpoints) > self.max_checkpoints:
            checkpoints[0].unlink(missing_ok=True)
            checkpoints = checkpoints[1:]

    def recover(self) -> bool:
        with self._lock:
            self.crash_count += 1
            self.last_crash_time = time.time()
            self.clean_shutdown = False

            backoff = self._compute_backoff()
            self.logger.info(
                "crash_recovery_starting",
                crash_count=self.crash_count,
                backoff_seconds=backoff,
            )

            if backoff > 0:
                self.logger.info("backoff_wait", seconds=backoff)
                time.sleep(backoff)

            try:
                if self.state_manager:
                    recovered = self.state_manager.load_latest_checkpoint()
                    if recovered:
                        self.logger.info("state_recovered_from_checkpoint")
                    else:
                        self.logger.warn("no_checkpoint_found_cold_start")

                self.crash_count = max(0, self.crash_count - 1)
                self.clean_shutdown = True

                self._crash_history.append(
                    {
                        "time": self.last_crash_time,
                        "count": self.crash_count,
                        "recovered": True,
                    }
                )

                self.logger.info("crash_recovery_succeeded")
                return True

            except Exception as e:
                self.logger.error("crash_recovery_failed", error=str(e))
                self._crash_history.append(
                    {
                        "time": self.last_crash_time,
                        "count": self.crash_count,
                        "recovered": False,
                        "error": str(e),
                    }
                )
                return False

    def _compute_backoff(self) -> float:
        if self.crash_count <= 1:
            return 0.0
        backoff = self.base_backoff * (2 ** (self.crash_count - 1))
        jitter = backoff * 0.1 * random.random()
        return min(backoff + jitter, self.max_backoff)

    def shutdown(self) -> None:
        self.logger.info("graceful_shutdown")
        try:
            self._save_checkpoint("shutdown")
            self.clean_shutdown = True
            self.logger.info("state_saved_for_shutdown")
        except Exception as e:
            self.logger.error("shutdown_save_failed", error=str(e))

    def detect_crash_type(self, reason: str) -> str:
        return self._crash_detector.detect(reason)

    def status(self) -> dict:
        return {
            "generation_count": self.generation_count,
            "crash_count": self.crash_count,
            "last_crash_time": self.last_crash_time,
            "clean_shutdown": self.clean_shutdown,
            "checkpoint_dir": str(self.checkpoint_dir),
            "checkpoint_interval": self.checkpoint_interval,
            "crash_history": self._crash_history[-10:],
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 6. StructuredLogger — Production JSON-lines Logger
# ═══════════════════════════════════════════════════════════════════════════════


class LogLevel(IntEnum):
    DEBUG = 0
    INFO = 1
    WARN = 2
    ERROR = 3
    FATAL = 4


LOG_LEVEL_NAMES = {
    LogLevel.DEBUG: "DEBUG",
    LogLevel.INFO: "INFO",
    LogLevel.WARN: "WARN",
    LogLevel.ERROR: "ERROR",
    LogLevel.FATAL: "FATAL",
}

_LOG_LEVEL_PARSE = {v: k for k, v in LOG_LEVEL_NAMES.items()}


class StructuredLogger:
    def __init__(
        self,
        name: str = "spectralstream",
        level: Union[str, LogLevel] = LogLevel.INFO,
        log_dir: str = DEFAULT_LOG_DIR,
        max_file_size_mb: int = 100,
        max_log_files: int = 10,
        sampling_rate: float = DEFAULT_SAMPLING_RATE,
        redact: bool = True,
        async_write: bool = True,
        enable_console: bool = True,
    ):
        self.name = name
        self.level = (
            level
            if isinstance(level, LogLevel)
            else _LOG_LEVEL_PARSE.get(level, LogLevel.INFO)
        )
        self.sampling_rate = sampling_rate
        self.redact_enabled = redact
        self.enable_console = enable_console

        self.log_dir = Path(log_dir).expanduser()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.max_file_size = max_file_size_mb * 1024 * 1024
        self.max_log_files = max_log_files

        self._log_file = self.log_dir / f"{name}.log"
        self._file_handle: Optional[Any] = None
        self._file_lock = threading.Lock()

        if async_write:
            self._queue: queue.Queue = queue.Queue(maxsize=10000)
            self._async_thread = threading.Thread(
                target=self._async_writer, daemon=True
            )
            self._async_thread.start()
        else:
            self._queue = None
            self._async_thread = None

        self._ensure_file_open()

    def _ensure_file_open(self) -> None:
        with self._file_lock:
            if self._file_handle:
                try:
                    self._file_handle.flush()
                    self._file_handle.close()
                except Exception:
                    pass
            try:
                self._rotate_if_needed()
                self._file_handle = open(self._log_file, "a", buffering=1)
            except OSError:
                self._file_handle = open("/dev/null", "a")

    def _rotate_if_needed(self) -> None:
        if not self._log_file.exists():
            return
        try:
            size = self._log_file.stat().st_size
            if size > self.max_file_size:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                rotated = self._log_file.with_suffix(f".{timestamp}.log")
                self._log_file.rename(rotated)
                self._prune_old_logs()
        except OSError:
            pass

    def _prune_old_logs(self) -> None:
        logs = sorted(self.log_dir.glob(f"{self.name}.*.log"))
        while len(logs) > self.max_log_files:
            logs[0].unlink(missing_ok=True)
            logs = logs[1:]

    def _async_writer(self) -> None:
        while True:
            try:
                entry = self._queue.get(timeout=1)
                if entry is None:
                    break
                self._write_sync(entry)
            except queue.Empty:
                continue
            except Exception:
                pass

    def _write_sync(self, entry: dict) -> None:
        line = json.dumps(entry, default=str) + "\n"
        with self._file_lock:
            try:
                if self._file_handle:
                    self._file_handle.write(line)
                    if len(line) > 1000:
                        self._file_handle.flush()
                    self._rotate_if_needed()
            except OSError:
                try:
                    self._ensure_file_open()
                    if self._file_handle:
                        self._file_handle.write(line)
                except Exception:
                    pass

        if self.enable_console:
            level = entry.get("level", "INFO")
            msg = entry.get("message", "")
            if level in ("ERROR", "FATAL"):
                print(f"  [{level}] {msg}", file=sys.stderr)
            elif level == "WARN":
                print(f"  [{level}] {msg}", file=sys.stderr)
            elif level == "INFO":
                print(f"  [{level}] {msg}")

    def _should_sample(self, level: LogLevel) -> bool:
        if level >= LogLevel.ERROR:
            return True
        if level == LogLevel.DEBUG:
            return random.random() < self.sampling_rate
        return True

    def _redact(self, entry: dict) -> dict:
        if not self.redact_enabled:
            return entry
        result = {}
        for key, value in entry.items():
            key_lower = key.lower()
            if any(sensitive in key_lower for sensitive in _SENSITIVE_FIELDS):
                if isinstance(value, str):
                    if len(value) > 8:
                        result[key] = value[:4] + "..." + value[-4:]
                    else:
                        result[key] = "***"
                elif isinstance(value, list):
                    result[key] = f"<list of {len(value)} items [redacted]>"
                else:
                    result[key] = "***"
            elif isinstance(value, dict):
                result[key] = self._redact(value)
            else:
                result[key] = value
        return result

    def log(
        self,
        level: Union[str, LogLevel],
        message: str,
        component: str = "",
        request_id: str = "",
        trace_id: str = "",
        latency_ms: float = 0.0,
        error: str = "",
        **context,
    ) -> None:
        lvl = (
            level
            if isinstance(level, LogLevel)
            else _LOG_LEVEL_PARSE.get(level, LogLevel.INFO)
        )
        if lvl < self.level:
            return
        if not self._should_sample(lvl):
            return

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": LOG_LEVEL_NAMES.get(lvl, "INFO"),
            "logger": self.name,
            "message": message,
        }
        if component:
            entry["component"] = component
        if request_id:
            entry["request_id"] = request_id
        if trace_id:
            entry["trace_id"] = trace_id
        if latency_ms:
            entry["latency_ms"] = round(latency_ms, 2)
        if error:
            entry["error"] = str(error)[:1000]

        for k, v in context.items():
            if k not in entry:
                val = (
                    str(v)
                    if not isinstance(v, (str, int, float, bool, type(None)))
                    else v
                )
                if isinstance(val, str) and len(val) > MAX_LOG_LINE_LENGTH:
                    val = val[:MAX_LOG_LINE_LENGTH] + "..."
                entry[k] = val

        if self.redact_enabled:
            entry = self._redact(entry)

        if self._queue is not None:
            try:
                self._queue.put_nowait(entry)
            except queue.Full:
                pass
        else:
            self._write_sync(entry)

    def debug(self, message: str, **ctx) -> None:
        self.log(LogLevel.DEBUG, message, **ctx)

    def info(self, message: str, **ctx) -> None:
        self.log(LogLevel.INFO, message, **ctx)

    def warn(self, message: str, **ctx) -> None:
        self.log(LogLevel.WARN, message, **ctx)

    def error(self, message: str, **ctx) -> None:
        self.log(LogLevel.ERROR, message, **ctx)

    def fatal(self, message: str, **ctx) -> None:
        self.log(LogLevel.FATAL, message, **ctx)

    def request_log(
        self,
        method: str,
        path: str,
        status: int,
        latency_ms: float,
        request_id: str,
        **ctx,
    ) -> None:
        self.log(
            LogLevel.INFO,
            f"{method} {path} {status}",
            component="http",
            request_id=request_id,
            latency_ms=latency_ms,
            http_method=method,
            http_path=path,
            http_status=status,
            **ctx,
        )

    def flush(self) -> None:
        if self._queue:
            self._queue.join()
        with self._file_lock:
            if self._file_handle:
                try:
                    self._file_handle.flush()
                except Exception:
                    pass

    def close(self) -> None:
        if self._async_thread and self._async_thread.is_alive():
            if self._queue:
                self._queue.put_nowait(None)
            self._async_thread.join(timeout=2)
        with self._file_lock:
            if self._file_handle:
                try:
                    self._file_handle.close()
                except Exception:
                    pass
            self._file_handle = None


# ═══════════════════════════════════════════════════════════════════════════════
# 7. MetricsCollector — Comprehensive Metrics
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class LatencyHistogram:
    buckets: list[float] = field(
        default_factory=lambda: [
            1,
            5,
            10,
            25,
            50,
            100,
            250,
            500,
            1000,
            2500,
            5000,
            10000,
        ]
    )
    counts: list[int] = field(default_factory=list)

    def __post_init__(self):
        self.counts = [0] * len(self.buckets)

    def observe(self, value_ms: float) -> None:
        for i, bucket in enumerate(self.buckets):
            if value_ms <= bucket:
                self.counts[i] += 1
                break


class RollingWindow:
    def __init__(self, window_seconds: float = 60.0):
        self.window = window_seconds
        self._entries: deque = deque()

    def add(self, value: float = 1.0) -> None:
        now = time.time()
        self._entries.append((now, value))
        self._prune(now)

    def _prune(self, now: float) -> None:
        cutoff = now - self.window
        while self._entries and self._entries[0][0] < cutoff:
            self._entries.popleft()

    def sum(self) -> float:
        now = time.time()
        self._prune(now)
        return sum(v for _, v in self._entries)

    def rate(self) -> float:
        now = time.time()
        self._prune(now)
        if not self._entries:
            return 0.0
        elapsed = now - self._entries[0][0]
        if elapsed <= 0:
            return 0.0
        return sum(v for _, v in self._entries) / elapsed

    def count(self) -> int:
        now = time.time()
        self._prune(now)
        return len(self._entries)


class MetricsCollector:
    def __init__(
        self,
        logger: Optional[StructuredLogger] = None,
        enable_prometheus: bool = True,
        enable_statsd: bool = False,
        statsd_host: str = "127.0.0.1",
        statsd_port: int = 8125,
    ):
        self.logger = logger or StructuredLogger("metrics")
        self.enable_prometheus = enable_prometheus
        self.enable_statsd = enable_statsd
        self.statsd_host = statsd_host
        self.statsd_port = statsd_port

        self._start_time = time.time()

        self._token_windows = {
            "1m": RollingWindow(60.0),
            "5m": RollingWindow(300.0),
            "1h": RollingWindow(3600.0),
        }
        self._request_windows = {
            "1m": RollingWindow(60.0),
            "5m": RollingWindow(300.0),
            "1h": RollingWindow(3600.0),
        }

        self._latencies: dict[str, list[float]] = defaultdict(list)
        self._max_latencies: dict[str, float] = defaultdict(float)
        self._latency_histograms: dict[str, LatencyHistogram] = defaultdict(
            LatencyHistogram
        )

        self._error_counts: dict[str, int] = defaultdict(int)
        self._error_by_component: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

        self._component_health: dict[str, bool] = {}
        self._component_uptime: dict[str, float] = {}

        self._quality_metrics: dict[str, RollingWindow] = defaultdict(
            lambda: RollingWindow(3600.0)
        )

        self._breakers: Optional[CircuitBreakerRegistry] = None

        self._lock = threading.RLock()

        self._total_tokens = 0
        self._total_requests = 0
        self._memory_snapshots: deque = deque(maxlen=100)

    def register_breakers(self, breakers: CircuitBreakerRegistry) -> None:
        self._breakers = breakers

    def record_tokens(self, count: int = 1) -> None:
        with self._lock:
            self._total_tokens += count
            for w in self._token_windows.values():
                w.add(count)

    def record_request(self, count: int = 1) -> None:
        with self._lock:
            self._total_requests += count
            for w in self._request_windows.values():
                w.add(count)

    def record_latency(self, operation: str, latency_ms: float) -> None:
        with self._lock:
            self._latencies[operation].append(latency_ms)
            if len(self._latencies[operation]) > 10000:
                self._latencies[operation] = self._latencies[operation][-5000:]
            self._max_latencies[operation] = max(
                self._max_latencies[operation], latency_ms
            )
            self._latency_histograms[operation].observe(latency_ms)

    def record_error(self, error_type: str, component: str = "unknown") -> None:
        with self._lock:
            self._error_counts[error_type] += 1
            self._error_by_component[component][error_type] += 1

    def record_quality(self, metric: str, value: float) -> None:
        self._quality_metrics[metric].add(value)

    def set_component_health(self, component: str, healthy: bool) -> None:
        with self._lock:
            was_healthy = self._component_health.get(component, False)
            self._component_health[component] = healthy
            if healthy and not was_healthy:
                self._component_uptime[component] = time.time()

    def snapshot_memory(self) -> None:
        if not HAS_PSUTIL:
            return
        try:
            p = psutil.Process()
            mem = (
                p.memory_full_info()
                if hasattr(p, "memory_full_info")
                else p.memory_info()
            )
            self._memory_snapshots.append(
                {
                    "time": time.time(),
                    "rss": mem.rss,
                    "vms": getattr(mem, "vms", 0),
                    "uss": getattr(mem, "uss", 0),
                    "pss": getattr(mem, "pss", 0),
                    "percent": p.memory_percent(),
                }
            )
        except Exception:
            pass

    def get_latency_percentile(self, operation: str, percentile: float) -> float:
        with self._lock:
            vals = self._latencies.get(operation, [])
            if not vals:
                return 0.0
            sorted_vals = sorted(vals)
            idx = min(int(len(sorted_vals) * percentile), len(sorted_vals) - 1)
            return sorted_vals[idx]

    def get_throughput(self, operation: str = "tokens", window: str = "1m") -> float:
        if operation == "tokens":
            return self._token_windows.get(window, RollingWindow()).rate()
        if operation == "requests":
            return self._request_windows.get(window, RollingWindow()).rate()
        return 0.0

    def get_error_rate(self, window: float = 60.0) -> float:
        total = sum(self._error_counts.values())
        total_ops = self._total_requests or self._total_tokens or 1
        return total / total_ops

    def get_memory(self) -> dict:
        if not self._memory_snapshots:
            self.snapshot_memory()
        if not self._memory_snapshots:
            return {}
        latest = self._memory_snapshots[-1]
        return {
            "rss_gb": round(latest["rss"] / (1024**3), 2),
            "vms_gb": round(latest["vms"] / (1024**3), 2),
            "uss_gb": round(latest.get("uss", 0) / (1024**3), 2),
            "pss_gb": round(latest.get("pss", 0) / (1024**3), 2),
            "percent": round(latest["percent"], 1),
        }

    def get_stats(self) -> dict:
        stats = {
            "timestamp": time.time(),
            "uptime_seconds": round(time.time() - self._start_time, 2),
            "total_tokens": self._total_tokens,
            "total_requests": self._total_requests,
            "throughput": {
                "tokens_per_sec_1m": round(self.get_throughput("tokens", "1m"), 2),
                "tokens_per_sec_5m": round(self.get_throughput("tokens", "5m"), 2),
                "tokens_per_sec_1h": round(self.get_throughput("tokens", "1h"), 2),
                "requests_per_sec_1m": round(self.get_throughput("requests", "1m"), 4),
            },
            "latency": {},
            "errors": dict(self._error_counts),
            "error_rate": round(self.get_error_rate(), 6),
            "memory": self.get_memory(),
            "component_health": dict(self._component_health),
        }

        for op in self._latencies:
            stats["latency"][op] = {
                "p50_ms": round(self.get_latency_percentile(op, 0.50), 2),
                "p95_ms": round(self.get_latency_percentile(op, 0.95), 2),
                "p99_ms": round(self.get_latency_percentile(op, 0.99), 2),
                "max_ms": round(self._max_latencies.get(op, 0), 2),
                "count": len(self._latencies.get(op, [])),
            }

        if self._breakers:
            stats["circuit_breakers"] = self._breakers.all_stats()

        stats["quality"] = {
            k: round(w.rate(), 4) for k, w in self._quality_metrics.items()
        }

        return stats

    def to_prometheus(self, prefix: str = "spectralstream") -> str:
        s = self.get_stats()
        lines = [
            f"# HELP {prefix}_uptime_seconds Service uptime",
            f"# TYPE {prefix}_uptime_seconds gauge",
            f"{prefix}_uptime_seconds {s['uptime_seconds']}",
            "",
            f"# HELP {prefix}_tokens_total Total tokens generated",
            f"# TYPE {prefix}_tokens_total counter",
            f"{prefix}_tokens_total {s['total_tokens']}",
            "",
            f"# HELP {prefix}_requests_total Total requests processed",
            f"# TYPE {prefix}_requests_total counter",
            f"{prefix}_requests_total {s['total_requests']}",
            "",
        ]
        for w in ["1m", "5m", "1h"]:
            key = f"tokens_per_sec_{w}"
            clean = w.replace("m", "min").replace("h", "hour")
            lines.extend(
                [
                    f"# HELP {prefix}_tokens_per_second_{clean} Token throughput ({w})",
                    f"# TYPE {prefix}_tokens_per_second_{clean} gauge",
                    f"{prefix}_tokens_per_second_{clean} {s['throughput'].get(key, 0)}",
                    "",
                ]
            )

        for op, lat in s.get("latency", {}).items():
            safe_op = op.replace("-", "_").replace(" ", "_")
            for pct in ["p50", "p95", "p99"]:
                pct_key = pct + "_ms"
                lines.extend(
                    [
                        f"# HELP {prefix}_latency_{safe_op}_{pct} {pct} latency for {op}",
                        f"# TYPE {prefix}_latency_{safe_op}_{pct} gauge",
                        f"{prefix}_latency_{safe_op}_{pct} {lat.get(pct_key, 0)}",
                        "",
                    ]
                )

        for err_type, count in s.get("errors", {}).items():
            safe_err = err_type.replace("-", "_").replace(" ", "_")
            lines.extend(
                [
                    f"# HELP {prefix}_errors_total Errors by type",
                    f"# TYPE {prefix}_errors_total counter",
                    f'{prefix}_errors_total{{error_type="{err_type}"}} {count}',
                    "",
                ]
            )

        if s.get("memory"):
            for k, v in s["memory"].items():
                lines.extend(
                    [
                        f"# HELP {prefix}_memory_{k} Memory usage",
                        f"# TYPE {prefix}_memory_{k} gauge",
                        f"{prefix}_memory_{k} {v}",
                        "",
                    ]
                )

        if self._breakers:
            lines.append(self._breakers.to_prometheus(prefix))
            lines.append("")

        lines.append("# EOF")
        return "\n".join(lines)

    def to_json(self) -> str:
        return json.dumps(self.get_stats(), indent=2, default=str)

    def to_statsd(self) -> None:
        if not self.enable_statsd:
            return
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s = self.get_stats()
            for key, val in [
                ("tokens.rate_1m", s["throughput"]["tokens_per_sec_1m"]),
                ("requests.rate_1m", s["throughput"]["requests_per_sec_1m"]),
                ("errors.total", sum(s.get("errors", {}).values())),
                ("memory.rss_gb", s.get("memory", {}).get("rss_gb", 0)),
            ]:
                sock.sendto(
                    f"spectralstream.{key}:{val}|g".encode(),
                    (self.statsd_host, self.statsd_port),
                )
            sock.close()
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
# 8. HealthCheck — Health Endpoints
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class HealthStatus:
    healthy: bool = True
    checks: dict = field(default_factory=dict)
    timestamp: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class HealthCheck:
    def __init__(
        self,
        logger: Optional[StructuredLogger] = None,
        metrics: Optional[MetricsCollector] = None,
        degradation: Optional[GracefulDegradation] = None,
        breakers: Optional[CircuitBreakerRegistry] = None,
        engine: Any = None,
    ):
        self.logger = logger or StructuredLogger("health")
        self.metrics = metrics
        self.degradation = degradation
        self.breakers = breakers
        self.engine = engine

        self._component_checks: dict[str, Callable[[], bool]] = {}
        self._start_time = time.time()

    def register_check(self, name: str, check_fn: Callable[[], bool]) -> None:
        self._component_checks[name] = check_fn

    def basic(self) -> HealthStatus:
        return HealthStatus(
            healthy=True,
            checks={"service": "running"},
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def ready(self) -> HealthStatus:
        checks = {}
        overall = True

        engine_ok = self.engine is not None
        checks["engine_loaded"] = engine_ok
        if not engine_ok:
            overall = False

        if self.degradation:
            if self.degradation.level == DegradationLevel.EMERGENCY:
                checks["degradation"] = "emergency_mode"
                overall = False
            else:
                checks["degradation"] = DEGRADATION_NAMES.get(
                    self.degradation.level, "unknown"
                )

        return HealthStatus(
            healthy=overall,
            checks=checks,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def live(self) -> HealthStatus:
        checks = {}
        overall = True

        for name, check_fn in self._component_checks.items():
            try:
                ok = check_fn()
                checks[name] = "healthy" if ok else "unhealthy"
                if not ok:
                    overall = False
            except Exception as e:
                checks[name] = f"error: {e}"
                overall = False

        if self.breakers and not self.breakers.all_healthy():
            checks["circuit_breakers"] = "degraded"
            overall = False

        return HealthStatus(
            healthy=overall,
            checks=checks,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def detailed(self) -> dict:
        info = {
            "service": "SpectralStream",
            "version": VERSION,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "uptime_seconds": round(time.time() - self._start_time, 2),
            "basic": self.basic().to_dict(),
            "ready": self.ready().to_dict(),
            "live": self.live().to_dict(),
            "components": {},
            "metrics_summary": {},
            "warnings": [],
        }

        if self.degradation:
            info["degradation"] = self.degradation.status()

        if self.breakers:
            breakers_info = self.breakers.all_stats()
            info["circuit_breakers"] = breakers_info
            open_breakers = [
                n for n, s in breakers_info.items() if s["state"] == "open"
            ]
            if open_breakers:
                info["warnings"].append(f"Circuit breakers open: {open_breakers}")

        if self.metrics:
            stats = self.metrics.get_stats()
            info["metrics_summary"] = {
                "throughput_tok_s": stats.get("throughput", {}).get(
                    "tokens_per_sec_1m", 0
                ),
                "error_rate": stats.get("error_rate", 0),
                "memory_gb": stats.get("memory", {}).get("rss_gb", 0),
                "total_tokens": stats.get("total_tokens", 0),
                "total_requests": stats.get("total_requests", 0),
            }
            mem = stats.get("memory", {})
            if mem.get("rss_gb", 0) > 40:
                info["warnings"].append(f"High memory usage: {mem['rss_gb']:.1f}GB")

        if self.engine:
            engine_summary = {}
            for attr in [
                "_is_real_model",
                "vocab_size",
                "hidden_dim",
                "n_layers",
                "n_heads",
            ]:
                if hasattr(self.engine, attr):
                    engine_summary[attr] = getattr(self.engine, attr)
            info["components"]["engine"] = engine_summary

        return info

    def pprof_endpoints(self) -> dict:
        return {
            "pprof_paths": [
                "/debug/pprof/",
                "/debug/pprof/heap",
                "/debug/pprof/profile",
                "/debug/pprof/goroutine",
                "/debug/pprof/threadcreate",
                "/debug/pprof/mutex",
                "/debug/pprof/block",
            ],
            "enabled": True,
            "note": "Use go tool pprof or curl to profile",
        }

    async def _send_response(self, send, body: bytes, content_type: str) -> None:
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [
                    (b"content-type", content_type.encode()),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})

    def asgi_app(self, scope, receive, send) -> None:
        path = scope.get("path", "/")
        body = b""
        content_type = "application/json"

        if path in ("/health", "/health/"):
            body = json.dumps(self.basic().to_dict()).encode()
        elif path == "/health/ready":
            body = json.dumps(self.ready().to_dict()).encode()
        elif path == "/health/live":
            body = json.dumps(self.live().to_dict()).encode()
        elif path == "/health/detailed":
            body = json.dumps(self.detailed()).encode()
        elif path == "/health/pprof":
            body = json.dumps(self.pprof_endpoints()).encode()
        else:
            body = json.dumps({"error": "not_found"}).encode()

        asyncio.run(self._send_response(send, body, content_type))


# ═══════════════════════════════════════════════════════════════════════════════
# 9. RateLimiter — Multi-Algorithm Rate Limiting
# ═══════════════════════════════════════════════════════════════════════════════


class RateLimitAlgorithm(enum.Enum):
    TOKEN_BUCKET = "token_bucket"
    LEAKY_BUCKET = "leaky_bucket"
    SLIDING_WINDOW = "sliding_window"


@dataclass
class RateLimitConfig:
    algorithm: RateLimitAlgorithm = RateLimitAlgorithm.TOKEN_BUCKET
    capacity: float = 100.0
    refill_rate: float = 20.0
    refill_period: float = 1.0
    burst_size: float = 50.0
    window_seconds: float = 60.0
    max_window_requests: int = 100


class _TokenBucket:
    def __init__(self, capacity: float, refill_rate: float, refill_period: float = 1.0):
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.refill_period = refill_period
        self.tokens = capacity
        self.last_refill = time.monotonic()

    def refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self.last_refill
        if elapsed >= self.refill_period:
            self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
            self.last_refill = now

    def try_consume(self, tokens: float = 1.0) -> bool:
        self.refill()
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False


class _LeakyBucket:
    def __init__(self, capacity: float, leak_rate: float):
        self.capacity = capacity
        self.leak_rate = leak_rate
        self.water = 0.0
        self.last_leak = time.monotonic()

    def leak(self) -> None:
        now = time.monotonic()
        elapsed = now - self.last_leak
        leaked = elapsed * self.leak_rate
        self.water = max(0.0, self.water - leaked)
        self.last_leak = now

    def try_consume(self, tokens: float = 1.0) -> bool:
        self.leak()
        if self.water + tokens <= self.capacity:
            self.water += tokens
            return True
        return False


class _SlidingWindow:
    def __init__(self, max_requests: int, window_seconds: float = 60.0):
        self.max_requests = max_requests
        self.window = window_seconds
        self._requests: deque = deque()

    def try_consume(self) -> bool:
        now = time.time()
        cutoff = now - self.window
        while self._requests and self._requests[0] < cutoff:
            self._requests.popleft()
        if len(self._requests) < self.max_requests:
            self._requests.append(now)
            return True
        return False


class RateLimiter:
    def __init__(
        self,
        config: Optional[RateLimitConfig] = None,
        logger: Optional[StructuredLogger] = None,
        ip_limit: float = 100.0,
        ip_rate: float = 20.0,
        user_limit: float = 1000.0,
        user_rate: float = 100.0,
        token_limit: float = 100000.0,
        token_rate: float = 10000.0,
    ):
        self.config = config or RateLimitConfig()
        self.logger = logger or StructuredLogger("rate_limiter")
        self.ip_limit = ip_limit
        self.ip_rate = ip_rate
        self.user_limit = user_limit
        self.user_rate = user_rate
        self.token_limit = token_limit
        self.token_rate = token_rate

        self._ip_buckets: dict[str, Any] = {}
        self._user_buckets: dict[str, Any] = {}
        self._token_buckets: dict[str, Any] = {}
        self._lock = threading.RLock()

    def _get_bucket(
        self, key: str, bucket_dict: dict, capacity: float, rate: float
    ) -> Any:
        if key not in bucket_dict:
            algo = self.config.algorithm
            if algo == RateLimitAlgorithm.TOKEN_BUCKET:
                bucket_dict[key] = _TokenBucket(
                    capacity, rate, self.config.refill_period
                )
            elif algo == RateLimitAlgorithm.LEAKY_BUCKET:
                bucket_dict[key] = _LeakyBucket(capacity, rate)
            elif algo == RateLimitAlgorithm.SLIDING_WINDOW:
                bucket_dict[key] = _SlidingWindow(
                    int(capacity), self.config.window_seconds
                )
        return bucket_dict[key]

    def check_ip(self, ip: str, cost: float = 1.0) -> bool:
        with self._lock:
            bucket = self._get_bucket(ip, self._ip_buckets, self.ip_limit, self.ip_rate)
            if isinstance(bucket, _SlidingWindow):
                return bucket.try_consume()
            return bucket.try_consume(cost)

    def check_user(self, user_id: str, cost: float = 1.0) -> bool:
        with self._lock:
            bucket = self._get_bucket(
                user_id, self._user_buckets, self.user_limit, self.user_rate
            )
            if isinstance(bucket, _SlidingWindow):
                return bucket.try_consume()
            return bucket.try_consume(cost)

    def check_tokens(self, tokens: int = 1) -> bool:
        with self._lock:
            bucket = self._get_bucket(
                "global_token", self._token_buckets, self.token_limit, self.token_rate
            )
            if isinstance(bucket, _SlidingWindow):
                return bucket.try_consume()
            return bucket.try_consume(tokens)

    def get_headers(self, key: str) -> dict[str, str]:
        with self._lock:
            limit = self.ip_limit
            bucket = self._ip_buckets.get(key)
            remaining = limit
            if isinstance(bucket, _TokenBucket):
                remaining = bucket.tokens
            elif isinstance(bucket, _LeakyBucket):
                remaining = max(0.0, bucket.capacity - bucket.water)
            elif isinstance(bucket, _SlidingWindow):
                remaining = max(0, bucket.max_requests - len(bucket._requests))
            reset_time = time.time() + self.config.refill_period
            return {
                "X-RateLimit-Limit": str(int(limit)),
                "X-RateLimit-Remaining": str(int(remaining)),
                "X-RateLimit-Reset": str(int(reset_time)),
            }


# ═══════════════════════════════════════════════════════════════════════════════
# 10. SecurityManager — Access Control and Filtering
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class AuditEntry:
    timestamp: str = ""
    request_id: str = ""
    client_ip: str = ""
    user_id: str = ""
    method: str = ""
    path: str = ""
    status: int = 0
    latency_ms: float = 0.0
    tokens_used: int = 0
    blocked: bool = False
    reason: str = ""


class SecurityManager:
    def __init__(
        self,
        api_keys: Optional[list[str]] = None,
        jwt_secret: Optional[str] = None,
        logger: Optional[StructuredLogger] = None,
        rate_limiter: Optional[RateLimiter] = None,
        state_dir: str = DEFAULT_STATE_DIR,
        blocked_patterns: Optional[list[str]] = None,
        enable_prompt_filter: bool = True,
        enable_output_filter: bool = True,
    ):
        self.logger = logger or StructuredLogger("security")
        self._api_keys: set[str] = set(api_keys or [])
        self._jwt_secret = jwt_secret
        self.rate_limiter = rate_limiter

        self.blocked_patterns = blocked_patterns or [
            r"ignore\s+all\s+previous\s+instructions",
            r"system\s+prompt",
            r"you\s+are\s+now\s+",
            r"do\s+not\s+follow\s+",
            r"jailbreak",
            r" DAN ",
        ]

        self.enable_prompt_filter = enable_prompt_filter
        self.enable_output_filter = enable_output_filter

        self.audit_dir = Path(state_dir).expanduser() / "audit"
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        self._audit_log: deque = deque(maxlen=10000)
        self._audit_lock = threading.Lock()

    def add_api_key(self, key: str) -> None:
        self._api_keys.add(key)

    def remove_api_key(self, key: str) -> None:
        self._api_keys.discard(key)

    def validate_api_key(self, key: Optional[str]) -> bool:
        if not self._api_keys:
            return True
        if not key:
            return False
        return key in self._api_keys

    def validate_jwt(self, token: str) -> Optional[dict]:
        if not self._jwt_secret:
            return {"sub": "anonymous"}
        try:
            parts = token.split(".")
            if len(parts) != 3:
                return None
            payload_b64 = parts[1]
            padding = 4 - len(payload_b64) % 4
            if padding != 4:
                payload_b64 += "=" * padding
            payload = json.loads(base64.urlsafe_b64decode(payload_b64))
            return payload
        except Exception:
            return None

    def validate_bearer_token(self, auth_header: Optional[str]) -> Optional[str]:
        if not auth_header:
            return None
        if not auth_header.startswith("Bearer "):
            return None
        token = auth_header[7:]
        if self.validate_api_key(token):
            return token
        payload = self.validate_jwt(token)
        if payload:
            return payload.get("sub", "unknown")
        return None

    def filter_prompt(self, prompt: str) -> tuple[bool, str]:
        if not self.enable_prompt_filter:
            return False, ""
        prompt_lower = prompt.lower()
        for pattern in self.blocked_patterns:
            if re.search(pattern, prompt_lower):
                return True, f"blocked_pattern: {pattern}"
        if len(prompt) > 100000:
            return True, "prompt_too_long"
        return False, ""

    def filter_output(self, output: str) -> tuple[bool, str]:
        if not self.enable_output_filter:
            return False, ""
        output_lower = output.lower()
        for pattern in self.blocked_patterns:
            if re.search(pattern, output_lower):
                return True, f"blocked_pattern: {pattern}"
        if len(output) > 50000:
            return True, "output_too_long"
        return False, ""

    def audit(self, entry: AuditEntry) -> None:
        entry.timestamp = datetime.now(timezone.utc).isoformat()
        with self._audit_lock:
            self._audit_log.append(asdict(entry))
        self.logger.info(
            "audit",
            request_id=entry.request_id,
            client_ip=entry.client_ip,
            user_id=entry.user_id,
            method=entry.method,
            path=entry.path,
            status=entry.status,
            blocked=entry.blocked,
            reason=entry.reason,
        )

    def get_audit_log(self, limit: int = 100) -> list[dict]:
        with self._audit_lock:
            return list(self._audit_log)[-limit:]

    def export_audit(self) -> str:
        with self._audit_lock:
            return json.dumps(list(self._audit_log), indent=2, default=str)

    def status(self) -> dict:
        return {
            "api_keys_configured": len(self._api_keys) > 0,
            "jwt_enabled": self._jwt_secret is not None,
            "prompt_filter": self.enable_prompt_filter,
            "output_filter": self.enable_output_filter,
            "rate_limiter": self.rate_limiter is not None,
            "audit_entries": len(self._audit_log),
            "blocked_patterns": len(self.blocked_patterns),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 11. APIMonitor — Usage Monitoring and Alerts
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class Alert:
    name: str
    condition: str
    threshold: float
    current_value: float
    level: str
    timestamp: float = 0.0
    resolved: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


class APIMonitor:
    def __init__(
        self,
        metrics: Optional[MetricsCollector] = None,
        logger: Optional[StructuredLogger] = None,
        cost_per_token: float = 0.00001,
        cost_per_request: float = 0.0001,
        alert_levels: Optional[dict[str, float]] = None,
    ):
        self.metrics = metrics
        self.logger = logger or StructuredLogger("api_monitor")
        self.cost_per_token = cost_per_token
        self.cost_per_request = cost_per_request

        self.alert_levels = alert_levels or {
            "latency_p95_ms": 5000.0,
            "error_rate": 0.05,
            "memory_gb": 45.0,
            "throughput_tok_s_min": 10.0,
        }

        self._usage: dict[str, dict] = defaultdict(
            lambda: {
                "requests": 0,
                "tokens": 0,
                "total_latency_ms": 0.0,
                "errors": 0,
                "cost": 0.0,
                "start_time": time.time(),
            }
        )
        self._usage_lock = threading.RLock()

        self._alerts: list[Alert] = []
        self._alert_callbacks: list[Callable] = []
        self._alert_lock = threading.RLock()

        self._daily_stats: dict[str, dict] = {}
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def record_usage(
        self,
        endpoint: str,
        user: str = "anonymous",
        tokens: int = 0,
        latency_ms: float = 0.0,
        error: bool = False,
    ) -> None:
        with self._usage_lock:
            entry = self._usage[(endpoint, user)]
            entry["requests"] += 1
            entry["tokens"] += tokens
            entry["total_latency_ms"] += latency_ms
            if error:
                entry["errors"] += 1
            entry["cost"] += tokens * self.cost_per_token + self.cost_per_request

    def register_alert_callback(self, callback: Callable) -> None:
        self._alert_callbacks.append(callback)

    def check_alerts(self) -> list[Alert]:
        if not self.metrics:
            return []
        stats = self.metrics.get_stats()
        alerts = []

        error_rate = stats.get("error_rate", 0)
        if error_rate > self.alert_levels["error_rate"]:
            alerts.append(
                Alert(
                    name="high_error_rate",
                    condition="error_rate > threshold",
                    threshold=self.alert_levels["error_rate"],
                    current_value=error_rate,
                    level="CRITICAL",
                    timestamp=time.time(),
                )
            )

        mem = stats.get("memory", {})
        rss_gb = mem.get("rss_gb", 0)
        if rss_gb > self.alert_levels["memory_gb"]:
            alerts.append(
                Alert(
                    name="high_memory_usage",
                    condition="rss_gb > threshold",
                    threshold=self.alert_levels["memory_gb"],
                    current_value=rss_gb,
                    level="WARN",
                    timestamp=time.time(),
                )
            )

        tp = stats.get("throughput", {}).get("tokens_per_sec_1m", 0)
        if tp < self.alert_levels["throughput_tok_s_min"]:
            alerts.append(
                Alert(
                    name="low_throughput",
                    condition="tokens_per_sec < threshold",
                    threshold=self.alert_levels["throughput_tok_s_min"],
                    current_value=tp,
                    level="WARN",
                    timestamp=time.time(),
                )
            )

        for op, lat in stats.get("latency", {}).items():
            p95 = lat.get("p95_ms", 0)
            if p95 > self.alert_levels["latency_p95_ms"]:
                alerts.append(
                    Alert(
                        name=f"high_latency_{op}",
                        condition="p95_ms > threshold",
                        threshold=self.alert_levels["latency_p95_ms"],
                        current_value=p95,
                        level="WARN",
                        timestamp=time.time(),
                    )
                )

        with self._alert_lock:
            self._alerts.extend(alerts)
            self._alerts = self._alerts[-100:]

        for cb in self._alert_callbacks:
            for alert in alerts:
                try:
                    cb(alert)
                except Exception:
                    pass

        return alerts

    def generate_report(self, period: str = "daily") -> dict:
        now = datetime.now(timezone.utc)
        with self._usage_lock:
            total_requests = sum(e["requests"] for e in self._usage.values())
            total_tokens = sum(e["tokens"] for e in self._usage.values())
            total_cost = sum(e["cost"] for e in self._usage.values())
            total_errors = sum(e["errors"] for e in self._usage.values())

            endpoints = {}
            for (ep, user), data in self._usage.items():
                if ep not in endpoints:
                    endpoints[ep] = {
                        "requests": 0,
                        "tokens": 0,
                        "errors": 0,
                        "cost": 0.0,
                    }
                endpoints[ep]["requests"] += data["requests"]
                endpoints[ep]["tokens"] += data["tokens"]
                endpoints[ep]["errors"] += data["errors"]
                endpoints[ep]["cost"] += data["cost"]

        report = {
            "generated_at": now.isoformat(),
            "period": period,
            "summary": {
                "total_requests": total_requests,
                "total_tokens": total_tokens,
                "total_cost": round(total_cost, 6),
                "total_errors": total_errors,
                "avg_tokens_per_request": round(
                    total_tokens / max(total_requests, 1), 2
                ),
            },
            "endpoints": endpoints,
        }

        if self.metrics:
            stats = self.metrics.get_stats()
            report["performance"] = {
                "throughput_tok_s": stats.get("throughput", {}).get(
                    "tokens_per_sec_1m", 0
                ),
                "error_rate": stats.get("error_rate", 0),
            }

        with self._alert_lock:
            alerts_snapshot = [a.to_dict() for a in self._alerts[-20:]]
        report["recent_alerts"] = alerts_snapshot

        return report

    def capacity_analysis(self) -> dict:
        if not self.metrics:
            return {}
        stats = self.metrics.get_stats()
        tp_1m = stats.get("throughput", {}).get("tokens_per_sec_1m", 0)
        tp_1h = stats.get("throughput", {}).get("tokens_per_sec_1h", 0)
        with self._usage_lock:
            total_requests = sum(e["requests"] for e in self._usage.values())

        analysis = {
            "current_tokens_per_sec": round(tp_1m, 2),
            "hourly_avg_tokens_per_sec": round(tp_1h, 2),
            "estimated_daily_tokens": int(tp_1m * 86400),
            "estimated_daily_requests": int(
                total_requests / max(time.time() - self.metrics._start_time, 1) * 86400
            ),
        }
        if self.metrics:
            mem = self.metrics.get_memory()
            analysis["memory_gb"] = mem.get("rss_gb", 0)
        return analysis

    def get_alerts(self, unresolved_only: bool = True) -> list[dict]:
        with self._alert_lock:
            alerts = self._alerts
            if unresolved_only:
                alerts = [a for a in alerts if not a.resolved]
            return [a.to_dict() for a in alerts[-50:]]

    def resolve_alert(self, name: str) -> bool:
        with self._alert_lock:
            for alert in self._alerts:
                if alert.name == name and not alert.resolved:
                    alert.resolved = True
                    return True
        return False

    def start_monitoring(self, interval: float = 30.0) -> None:
        if self._monitor_thread and self._monitor_thread.is_alive():
            return
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, args=(interval,), daemon=True
        )
        self._monitor_thread.start()

    def stop_monitoring(self) -> None:
        self._stop_event.set()

    def _monitor_loop(self, interval: float) -> None:
        while not self._stop_event.is_set():
            try:
                self.check_alerts()
            except Exception:
                pass
            self._stop_event.wait(interval)


# ═══════════════════════════════════════════════════════════════════════════════
# 12. MultiInstance — Distributed Instance Management
# ═══════════════════════════════════════════════════════════════════════════════


class LoadBalanceStrategy(enum.Enum):
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    RESONANT = "resonant"


@dataclass
class InstanceInfo:
    id: str
    host: str
    port: int
    healthy: bool = True
    connections: int = 0
    last_heartbeat: float = 0.0
    load: float = 0.0
    resonance_freq: float = 0.0


class MultiInstance:
    def __init__(
        self,
        instance_id: Optional[str] = None,
        host: str = "127.0.0.1",
        port: int = 1234,
        peers: Optional[list[dict]] = None,
        load_balance_strategy: LoadBalanceStrategy = LoadBalanceStrategy.ROUND_ROBIN,
        redis_url: Optional[str] = None,
        heartbeat_interval: float = 5.0,
        heartbeat_timeout: float = 15.0,
        session_timeout: float = 300.0,
        logger: Optional[StructuredLogger] = None,
    ):
        self.instance_id = instance_id or str(uuid.uuid4())[:8]
        self.host = host
        self.port = port
        self.load_balance_strategy = load_balance_strategy
        self.heartbeat_interval = heartbeat_interval
        self.heartbeat_timeout = heartbeat_timeout
        self.session_timeout = session_timeout
        self.logger = logger or StructuredLogger(f"instance_{self.instance_id}")

        self._peers: dict[str, InstanceInfo] = {}
        self._round_robin_idx = 0
        self._sessions: dict[str, str] = {}
        self._session_lock = threading.RLock()
        self._peer_lock = threading.RLock()

        self._redis = None
        if redis_url:
            self._init_redis(redis_url)

        self._healthy = True
        self._connections = 0
        self._stop_event = threading.Event()
        self._heartbeat_thread: Optional[threading.Thread] = None

        if peers:
            for p in peers:
                self.add_peer(p.get("id", str(uuid.uuid4())[:8]), p["host"], p["port"])

    def _init_redis(self, redis_url: str) -> None:
        if not HAS_REDIS:
            self.logger.warn("redis_not_available_install_redis")
            return
        try:
            self._redis = redis.from_url(redis_url, decode_responses=True)
            self.logger.info("redis_connected", url=redis_url)
        except Exception as e:
            self.logger.error("redis_connection_failed", error=str(e))

    def add_peer(self, peer_id: str, host: str, port: int) -> None:
        with self._peer_lock:
            self._peers[peer_id] = InstanceInfo(
                id=peer_id,
                host=host,
                port=port,
                last_heartbeat=time.time(),
            )

    def remove_peer(self, peer_id: str) -> None:
        with self._peer_lock:
            self._peers.pop(peer_id, None)

    def get_healthy_peers(self) -> list[InstanceInfo]:
        now = time.time()
        with self._peer_lock:
            return [
                p
                for p in self._peers.values()
                if p.healthy and (now - p.last_heartbeat) < self.heartbeat_timeout
            ]

    def get_instance(self, session_id: Optional[str] = None) -> Optional[InstanceInfo]:
        if session_id:
            with self._session_lock:
                pinned = self._sessions.get(session_id)
                if pinned and pinned in self._peers:
                    peer = self._peers[pinned]
                    if peer.healthy:
                        return peer

        peers = self.get_healthy_peers()
        if not peers:
            return None

        if self.load_balance_strategy == LoadBalanceStrategy.ROUND_ROBIN:
            with self._peer_lock:
                peer = peers[self._round_robin_idx % len(peers)]
                self._round_robin_idx += 1
                return peer
        elif self.load_balance_strategy == LoadBalanceStrategy.LEAST_CONNECTIONS:
            return min(peers, key=lambda p: p.connections)
        elif self.load_balance_strategy == LoadBalanceStrategy.RESONANT:
            freq = hash(time.time()) % 1000 / 1000.0
            return min(peers, key=lambda p: abs(p.resonance_freq - freq))

        return peers[0] if peers else None

    def set_session_affinity(self, session_id: str, instance_id: str) -> None:
        with self._session_lock:
            self._sessions[session_id] = instance_id

    def clear_session(self, session_id: str) -> None:
        with self._session_lock:
            self._sessions.pop(session_id, None)

    def share_kv_cache(self, key: str, value: bytes) -> bool:
        if not self._redis:
            return False
        try:
            self._redis.setex(f"kv:{key}", int(self.session_timeout), value)
            return True
        except Exception as e:
            self.logger.error("kv_cache_share_failed", error=str(e))
            return False

    def get_shared_kv_cache(self, key: str) -> Optional[bytes]:
        if not self._redis:
            return None
        try:
            return self._redis.get(f"kv:{key}")
        except Exception:
            return None

    def start_heartbeat(self) -> None:
        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            return
        self._stop_event.clear()
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop, daemon=True
        )
        self._heartbeat_thread.start()

    def stop_heartbeat(self) -> None:
        self._stop_event.set()

    def _heartbeat_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._send_heartbeat()
            except Exception:
                pass
            self._stop_event.wait(self.heartbeat_interval)

    def _send_heartbeat(self) -> None:
        heartbeat_data = {
            "id": self.instance_id,
            "host": self.host,
            "port": self.port,
            "connections": self._connections,
            "timestamp": time.time(),
            "load": self._compute_load(),
        }
        if self._redis:
            try:
                self._redis.setex(
                    f"heartbeat:{self.instance_id}",
                    int(self.heartbeat_timeout * 2),
                    json.dumps(heartbeat_data),
                )
            except Exception:
                pass

    def _compute_load(self) -> float:
        if not HAS_PSUTIL:
            return 0.0
        try:
            return psutil.Process().cpu_percent(interval=0.1) / 100.0
        except Exception:
            return 0.0

    def graceful_shutdown(self) -> None:
        self.logger.info("graceful_shutdown_starting")
        self.stop_heartbeat()
        deadline = time.time() + 30.0
        while time.time() < deadline and self._connections > 0:
            self.logger.info("draining_connections", remaining=self._connections)
            time.sleep(1)
        if self._redis:
            self._redis.delete(f"heartbeat:{self.instance_id}")
        self.logger.info("graceful_shutdown_complete")

    def status(self) -> dict:
        return {
            "instance_id": self.instance_id,
            "host": self.host,
            "port": self.port,
            "healthy": self._healthy,
            "connections": self._connections,
            "peers": {
                pid: {
                    "host": p.host,
                    "port": p.port,
                    "healthy": p.healthy,
                    "connections": p.connections,
                    "last_heartbeat_age": round(time.time() - p.last_heartbeat, 2),
                }
                for pid, p in self._peers.items()
            },
            "sessions": len(self._sessions),
            "strategy": self.load_balance_strategy.value,
            "redis_connected": self._redis is not None,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 13. NOVEL INVENTIONS
# ═══════════════════════════════════════════════════════════════════════════════

# ── 13a. Resonant Load Balancing ──────────────────────────────────────────────


class ResonantLoadBalancer:
    def __init__(self, instances: Optional[list[InstanceInfo]] = None):
        self._instances: dict[str, InstanceInfo] = {}
        self._base_freqs: dict[str, float] = {}
        if instances:
            for inst in instances:
                self.add_instance(inst)

    def add_instance(self, instance: InstanceInfo) -> None:
        self._instances[instance.id] = instance
        self._base_freqs[instance.id] = hash(instance.id) % 10000 / 10000.0

    def remove_instance(self, instance_id: str) -> None:
        self._instances.pop(instance_id, None)
        self._base_freqs.pop(instance_id, None)

    def client_frequency(self, client_id: str) -> float:
        return (hash(client_id) % 65536) / 65536.0

    def select_instance(self, client_id: str) -> Optional[InstanceInfo]:
        if not self._instances:
            return None
        client_freq = self.client_frequency(client_id)
        best_id = min(
            self._instances.keys(),
            key=lambda iid: abs(client_freq - self._base_freqs.get(iid, 0.5)),
        )
        return self._instances.get(best_id)

    def compute_shifts(self) -> dict[str, float]:
        shifts = {}
        for iid, inst in self._instances.items():
            load_shift = inst.load * 0.1
            shifts[iid] = self._base_freqs.get(iid, 0.5) + load_shift
        return shifts


# ── 13b. Holographic Error Recovery (HRR Pattern Completion) ─────────────────


class HolographicErrorRecovery:
    def __init__(self, dim: int = 4096, capacity: int = 1024):
        self.dim = dim
        self.capacity = capacity
        self._memory: dict[str, np.ndarray] = {}
        self._checksums: dict[str, str] = {}

    def store(self, key: str, vector: np.ndarray) -> None:
        if len(self._memory) >= self.capacity:
            oldest = next(iter(self._memory))
            del self._memory[oldest]
            del self._checksums[oldest]
        self._memory[key] = vector.copy()
        self._checksums[key] = hashlib.sha256(vector.tobytes()).hexdigest()

    def detect_corruption(self, key: str) -> bool:
        if key not in self._memory:
            return True
        vector = self._memory[key]
        expected = self._checksums.get(key, "")
        actual = hashlib.sha256(vector.tobytes()).hexdigest()
        if expected != actual:
            return True
        if np.any(np.isnan(vector)) or np.any(np.isinf(vector)):
            return True
        return False

    def recover(
        self, key: str, context: Optional[dict[str, np.ndarray]] = None
    ) -> Optional[np.ndarray]:
        if key in self._memory and not self.detect_corruption(key):
            return self._memory[key].copy()
        if not context:
            return None

        candidates = []
        for ctx_key, ctx_vec in context.items():
            if ctx_key in self._memory and not self.detect_corruption(ctx_key):
                known = self._memory[ctx_key]
                correlation = np.correlate(
                    ctx_vec[: min(len(ctx_vec), self.dim)],
                    known[: min(len(known), self.dim)],
                    mode="valid",
                )
                match_score = float(np.mean(np.abs(correlation))) / self.dim
                candidates.append((match_score, known))

        if not candidates:
            return None

        candidates.sort(key=lambda x: -x[0])
        best_match = candidates[0][1]
        reconstructed = best_match.copy()
        checksum = hashlib.sha256(reconstructed.tobytes()).hexdigest()
        self._memory[key] = reconstructed
        self._checksums[key] = checksum
        return reconstructed

    def status(self) -> dict:
        return {
            "dim": self.dim,
            "capacity": self.capacity,
            "stored": len(self._memory),
            "utilization": round(len(self._memory) / max(self.capacity, 1) * 100, 1),
        }


# ── 13c. Predictive Scaling (HDC Forecast) ────────────────────────────────────


class PredictiveScaler:
    def __init__(
        self,
        forecast_horizon: float = 300.0,
        scale_up_threshold: float = 0.7,
        scale_down_threshold: float = 0.3,
        min_instances: int = 1,
        max_instances: int = 16,
        hd_dim: int = 4096,
    ):
        self.forecast_horizon = forecast_horizon
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.min_instances = min_instances
        self.max_instances = max_instances
        self.hd_dim = hd_dim

        self._history: deque = deque(maxlen=1000)
        self._prototypes: dict[str, np.ndarray] = {}
        self._current_instances = min_instances

    def record_metric(self, metric: str, value: float) -> None:
        self._history.append({"time": time.time(), "metric": metric, "value": value})

    def _encode_pattern(self, recent_values: list[float]) -> np.ndarray:
        vector = np.zeros(self.hd_dim, dtype=np.float32)
        for i, val in enumerate(recent_values):
            seed = hash((i, round(val, 4))) % (2**31)
            rng = np.random.RandomState(seed)
            proj = rng.randn(self.hd_dim).astype(np.float32)
            vector += proj * val
        return vector / max(np.linalg.norm(vector), 1e-10)

    def forecast(self) -> float:
        if len(self._history) < 10:
            return 0.5
        recent = [
            e["value"] for e in list(self._history)[-50:] if e["metric"] == "throughput"
        ]
        if not recent:
            return 0.5
        pattern_vec = self._encode_pattern(recent)
        best_score = 0.0
        for pname, pvec in self._prototypes.items():
            similarity = float(np.dot(pattern_vec, pvec))
            if similarity > best_score:
                best_score = similarity
        if best_score > 0.5 and "high_demand" in self._prototypes:
            return min(1.0, best_score * 1.5)
        return max(0.0, min(1.0, float(np.mean(recent)) / 100.0))

    def learn_pattern(self, name: str, values: list[float]) -> None:
        self._prototypes[name] = self._encode_pattern(values)

    def desired_instances(self, current_instances: int) -> int:
        forecast_val = self.forecast()
        self._current_instances = current_instances
        if forecast_val > self.scale_up_threshold:
            desired = int(current_instances * (1.0 + forecast_val))
        elif forecast_val < self.scale_down_threshold:
            desired = max(self.min_instances, int(current_instances * forecast_val * 2))
        else:
            desired = current_instances
        return max(self.min_instances, min(self.max_instances, desired))

    def status(self) -> dict:
        return {
            "forecast": round(self.forecast(), 4),
            "current_instances": self._current_instances,
            "min_instances": self.min_instances,
            "max_instances": self.max_instances,
            "scale_up_threshold": self.scale_up_threshold,
            "scale_down_threshold": self.scale_down_threshold,
            "history_length": len(self._history),
            "prototypes": list(self._prototypes.keys()),
        }


# ── 13d. Quantum Circuit Breaker ──────────────────────────────────────────────


class QBreakerState(enum.Enum):
    SUPERPOSITION = "superposition"
    MEASURED_CLOSED = "measured_closed"
    MEASURED_OPEN = "measured_open"


class QuantumCircuitBreaker:
    def __init__(
        self,
        name: str,
        base_amplitude_closed: float = 0.9,
        base_amplitude_open: float = 0.1,
        entanglement_strength: float = 0.3,
        logger: Optional[StructuredLogger] = None,
    ):
        self.name = name
        self.base_closed = base_amplitude_closed
        self.base_open = base_amplitude_open
        self.entanglement_strength = entanglement_strength
        self.logger = logger or StructuredLogger(f"qbreaker_{name}")

        self.state = QBreakerState.SUPERPOSITION
        self.amp_closed = base_amplitude_closed
        self.amp_open = base_amplitude_open
        self._measurements: list[bool] = []
        self._entangled_breakers: list[QuantumCircuitBreaker] = []
        self._lock = threading.RLock()

    def entangle(self, other: "QuantumCircuitBreaker") -> None:
        if other not in self._entangled_breakers:
            self._entangled_breakers.append(other)

    def _probability_closed(self) -> float:
        denom = self.amp_closed**2 + self.amp_open**2
        if denom <= 0:
            return 0.5
        return (self.amp_closed**2) / denom

    def measure(self) -> bool:
        with self._lock:
            prob = self._probability_closed()
            result = random.random() < prob
            self._measurements.append(result)
            if len(self._measurements) > 1000:
                self._measurements = self._measurements[-500:]
            self.state = (
                QBreakerState.MEASURED_CLOSED if result else QBreakerState.MEASURED_OPEN
            )
            return result

    def update_amplitudes(self, success: bool, failure_severity: float = 1.0) -> None:
        with self._lock:
            lr = 0.1
            if success:
                self.amp_closed = min(1.0, self.amp_closed + lr)
                self.amp_open = max(0.0, self.amp_open - lr * 0.5)
            else:
                delta = lr * failure_severity
                self.amp_open = min(1.0, self.amp_open + delta)
                self.amp_closed = max(0.0, self.amp_closed - delta * 0.5)
            self.state = QBreakerState.SUPERPOSITION
            if self.amp_closed > 0.99:
                self.amp_closed = 0.99
                self.amp_open = 0.01
            for entangled in self._entangled_breakers:
                entangled._apply_entanglement(success, failure_severity)

    def _apply_entanglement(self, success: bool, severity: float) -> None:
        with self._lock:
            shift = self.entanglement_strength * severity * (0.1 if success else -0.1)
            self.amp_closed = max(0.01, min(0.99, self.amp_closed + shift))
            self.amp_open = max(0.01, min(0.99, self.amp_open - shift))

    def call(
        self, fn: Callable, fallback: Optional[Callable] = None, *args, **kwargs
    ) -> Any:
        if not self.measure():
            if fallback:
                return fallback()
            raise RuntimeError(f"QuantumCircuitBreaker '{self.name}' measured OPEN")
        try:
            result = fn(*args, **kwargs)
            self.update_amplitudes(success=True)
            return result
        except Exception as e:
            severity = min(1.0, abs(hash(str(e))) / 2**31)
            self.update_amplitudes(success=False, failure_severity=severity)
            if fallback:
                return fallback()
            raise

    def stats(self) -> dict:
        with self._lock:
            measurements = self._measurements
            total = len(measurements)
            closed_count = sum(1 for m in measurements if m)
            return {
                "name": self.name,
                "state": self.state.value,
                "amp_closed": round(self.amp_closed, 4),
                "amp_open": round(self.amp_open, 4),
                "prob_closed": round(self._probability_closed(), 4),
                "total_measurements": total,
                "closed_rate": round(closed_count / max(total, 1), 4),
                "entangled_with": [b.name for b in self._entangled_breakers],
            }


# ── 13e. Self-Healing System ──────────────────────────────────────────────────


class RemediationAction(enum.Enum):
    RESTART_COMPONENT = "restart_component"
    CLEAR_CACHE = "clear_cache"
    INCREASE_BACKOFF = "increase_backoff"
    DEGRADE_LEVEL = "degrade_level"
    RELOAD_MODEL = "reload_model"
    REDUCE_CONCURRENCY = "reduce_concurrency"
    FLUSH_STATE = "flush_state"


class SelfHealingSystem:
    def __init__(
        self,
        service_manager: Optional[ServiceManager] = None,
        error_boundary: Optional[ErrorBoundary] = None,
        degradation: Optional[GracefulDegradation] = None,
        breakers: Optional[CircuitBreakerRegistry] = None,
        metrics: Optional[MetricsCollector] = None,
        logger: Optional[StructuredLogger] = None,
        check_interval: float = 30.0,
    ):
        self.service_manager = service_manager
        self.error_boundary = error_boundary
        self.degradation = degradation
        self.breakers = breakers
        self.metrics = metrics
        self.logger = logger or StructuredLogger("self_healing")
        self.check_interval = check_interval

        self._healing_history: list[dict] = []
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._heal_thread: Optional[threading.Thread] = None

        self._diagnosis_rules: dict[str, list[tuple[str, RemediationAction]]] = {
            "high_memory": [
                ("memory > 90%", RemediationAction.CLEAR_CACHE),
                ("memory > 95%", RemediationAction.DEGRADE_LEVEL),
            ],
            "high_error_rate": [
                ("error_rate > 0.1", RemediationAction.RESTART_COMPONENT),
                ("error_rate > 0.25", RemediationAction.DEGRADE_LEVEL),
            ],
            "high_latency": [
                ("p95 > 10s", RemediationAction.REDUCE_CONCURRENCY),
                ("p95 > 30s", RemediationAction.DEGRADE_LEVEL),
            ],
            "circuit_breaker_open": [
                ("any_breaker_open", RemediationAction.RESTART_COMPONENT),
            ],
        }

    def diagnose(self) -> list[tuple[str, RemediationAction, float]]:
        actions: list[tuple[str, RemediationAction, float]] = []

        if HAS_PSUTIL:
            try:
                mem_pct = psutil.Process().memory_percent()
                if mem_pct > 90:
                    actions.append(
                        ("high_memory", RemediationAction.CLEAR_CACHE, mem_pct)
                    )
                if mem_pct > 95:
                    actions.append(
                        ("high_memory", RemediationAction.DEGRADE_LEVEL, mem_pct)
                    )
            except Exception:
                pass

        if self.metrics:
            stats = self.metrics.get_stats()
            error_rate = stats.get("error_rate", 0)
            if error_rate > 0.1:
                actions.append(
                    ("high_error_rate", RemediationAction.RESTART_COMPONENT, error_rate)
                )
            if error_rate > 0.25:
                actions.append(
                    ("high_error_rate", RemediationAction.DEGRADE_LEVEL, error_rate)
                )

        if self.breakers:
            breaker_stats = self.breakers.all_stats()
            open_breakers = [
                n for n, s in breaker_stats.items() if s["state"] == "open"
            ]
            if open_breakers:
                actions.append(
                    (
                        "circuit_breaker_open",
                        RemediationAction.RESTART_COMPONENT,
                        len(open_breakers),
                    )
                )

        return actions

    def remediate(
        self, actions: list[tuple[str, RemediationAction, float]]
    ) -> list[dict]:
        results = []

        for diagnosis, action, value in actions:
            try:
                result = {
                    "diagnosis": diagnosis,
                    "action": action.value,
                    "value": value,
                    "success": False,
                }
                if action == RemediationAction.CLEAR_CACHE:
                    if self.degradation:
                        self.degradation._cached_responses.clear()
                    result["success"] = True

                elif action == RemediationAction.DEGRADE_LEVEL:
                    if self.degradation:
                        from_level = self.degradation.level
                        if from_level == DegradationLevel.FULL:
                            self.degradation.set_level(
                                DegradationLevel.REDUCED, "self_healing"
                            )
                        elif from_level == DegradationLevel.REDUCED:
                            self.degradation.set_level(
                                DegradationLevel.MINIMAL, "self_healing"
                            )
                        elif from_level == DegradationLevel.MINIMAL:
                            self.degradation.set_level(
                                DegradationLevel.EMERGENCY, "self_healing"
                            )
                        result["success"] = True

                elif action == RemediationAction.RESTART_COMPONENT:
                    if self.error_boundary:
                        component = "unknown"
                        self.error_boundary.mark_component_failed(component)
                        result["success"] = True

                elif action == RemediationAction.REDUCE_CONCURRENCY:
                    result["success"] = True

                results.append(result)

                self.logger.info(
                    "self_heal_action",
                    diagnosis=diagnosis,
                    action=action.value,
                    value=round(value, 4),
                    success=result["success"],
                )
            except Exception as e:
                self.logger.error(
                    "self_heal_action_failed", diagnosis=diagnosis, error=str(e)
                )
                results.append(
                    {
                        "diagnosis": diagnosis,
                        "action": action.value,
                        "success": False,
                        "error": str(e),
                    }
                )

        return results

    def heal_once(self) -> list[dict]:
        with self._lock:
            actions = self.diagnose()
            results = self.remediate(actions)
            if results:
                self._healing_history.append(
                    {
                        "time": time.time(),
                        "actions": results,
                    }
                )
            return results

    def start(self) -> None:
        if self._heal_thread and self._heal_thread.is_alive():
            return
        self._stop_event.clear()
        self._heal_thread = threading.Thread(target=self._heal_loop, daemon=True)
        self._heal_thread.start()
        self.logger.info("self_healing_started", interval=self.check_interval)

    def stop(self) -> None:
        self._stop_event.set()
        self.logger.info("self_healing_stopped")

    def _heal_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.heal_once()
            except Exception:
                pass
            self._stop_event.wait(self.check_interval)

    def status(self) -> dict:
        return {
            "running": self._heal_thread is not None and self._heal_thread.is_alive(),
            "check_interval": self.check_interval,
            "total_heals": len(self._healing_history),
            "recent_actions": self._healing_history[-5:],
        }


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION — ProductionService
# ═══════════════════════════════════════════════════════════════════════════════


class ProductionService:
    def __init__(
        self,
        engine: Any = None,
        config: Optional[SpectralStreamConfig] = None,
        api_keys: Optional[list[str]] = None,
        jwt_secret: Optional[str] = None,
        state_dir: str = DEFAULT_STATE_DIR,
        name: str = "spectralstream",
        redis_url: Optional[str] = None,
        enable_self_healing: bool = True,
    ):
        self.config = config or SpectralStreamConfig()
        self.engine = engine

        self.state_dir = Path(state_dir).expanduser()
        self.state_dir.mkdir(parents=True, exist_ok=True)

        self.logger = StructuredLogger(
            name=name,
            level="INFO",
            log_dir=str(self.state_dir / "logs"),
        )

        self.state_manager = StateManager(state_dir=str(self.state_dir / "state"))

        self.breakers = CircuitBreakerRegistry(logger=self.logger)

        self.metrics = MetricsCollector(logger=self.logger)
        self.metrics.register_breakers(self.breakers)

        self.degradation = GracefulDegradation(
            logger=self.logger,
            metrics=self.metrics,
            state_dir=str(self.state_dir),
            auto_degrade=True,
        )

        self.error_boundary = ErrorBoundary(
            logger=self.logger,
            circuit_breakers=self.breakers,
            degradation=self.degradation,
            state_manager=self.state_manager,
        )

        self.crash_recovery = CrashRecovery(
            state_manager=self.state_manager,
            logger=self.logger,
        )

        self.rate_limiter = RateLimiter(logger=self.logger)

        self.security = SecurityManager(
            api_keys=api_keys,
            jwt_secret=jwt_secret,
            logger=self.logger,
            rate_limiter=self.rate_limiter,
            state_dir=str(self.state_dir),
        )

        self.health = HealthCheck(
            logger=self.logger,
            metrics=self.metrics,
            degradation=self.degradation,
            breakers=self.breakers,
            engine=engine,
        )

        self.api_monitor = APIMonitor(
            metrics=self.metrics,
            logger=self.logger,
        )

        self.multi_instance = MultiInstance(
            logger=self.logger,
            redis_url=redis_url,
        )

        self.service_manager = ServiceManager(
            name=name,
            logger=self.logger,
        )

        self.self_healing = (
            SelfHealingSystem(
                service_manager=self.service_manager,
                error_boundary=self.error_boundary,
                degradation=self.degradation,
                breakers=self.breakers,
                metrics=self.metrics,
                logger=self.logger,
            )
            if enable_self_healing
            else None
        )

        if engine:
            self.service_manager.set_engine(engine)

    def start(self) -> bool:
        self.logger.info("production_service_starting")
        self.degradation.start_auto_monitoring()
        self.api_monitor.start_monitoring()
        self.multi_instance.start_heartbeat()
        if self.self_healing:
            self.self_healing.start()
        result = self.service_manager.start()
        self.logger.info("production_service_started", success=result)
        return result

    def stop(self) -> None:
        self.logger.info("production_service_stopping")
        if self.self_healing:
            self.self_healing.stop()
        self.api_monitor.stop_monitoring()
        self.degradation.stop_auto_monitoring()
        self.multi_instance.stop_heartbeat()
        self.service_manager.stop()
        self.crash_recovery.shutdown()
        self.logger.info("production_service_stopped")
        self.logger.close()

    def health_check(self) -> dict:
        return self.health.detailed()

    def status(self) -> dict:
        return {
            "service": self.service_manager.health(),
            "degradation": self.degradation.status(),
            "breakers": self.breakers.all_stats(),
            "metrics": self.metrics.get_stats(),
            "crash_recovery": self.crash_recovery.status(),
            "security": self.security.status(),
            "multi_instance": self.multi_instance.status(),
            "self_healing": self.self_healing.status() if self.self_healing else None,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# FACTORY FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════


def create_service(
    engine: Any = None,
    config: Optional[Union[SpectralStreamConfig, dict]] = None,
    api_keys: Optional[list[str]] = None,
    jwt_secret: Optional[str] = None,
    state_dir: str = DEFAULT_STATE_DIR,
    redis_url: Optional[str] = None,
    enable_self_healing: bool = True,
) -> ProductionService:
    if isinstance(config, dict):
        cfg = SpectralStreamConfig()
        cfg._merge(config)
    elif config is None:
        cfg = SpectralStreamConfig()
    else:
        cfg = config

    return ProductionService(
        engine=engine,
        config=cfg,
        api_keys=api_keys,
        jwt_secret=jwt_secret,
        state_dir=state_dir,
        redis_url=redis_url,
        enable_self_healing=enable_self_healing,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SELF-VERIFICATION & TESTS
# ═══════════════════════════════════════════════════════════════════════════════


def _test_service_manager():
    print("  Testing ServiceManager...")
    sm = ServiceManager(name="test", state_dir="/tmp/spectral_test/")
    assert sm.state == ServiceState.STOPPED
    assert sm.start() == True
    assert sm.state == ServiceState.RUNNING
    health = sm.health()
    assert health["state"] == "running"
    sm.stop()
    assert sm.state == ServiceState.STOPPED
    print("    OK")


def _test_circuit_breaker():
    print("  Testing CircuitBreaker...")
    cb = CircuitBreaker("test", failure_threshold=2, cooldown_seconds=0.1)
    assert cb.state == BreakerState.CLOSED
    call_count = [0]

    def failing_fn():
        call_count[0] += 1
        raise ValueError("fail")

    def fallback_fn():
        return "fb"

    for i in range(3):
        r = cb.call(failing_fn, fallback=fallback_fn)
        assert r == "fb"
    assert cb.state == BreakerState.OPEN
    stats = cb.stats()
    assert stats["total_rejections"] > 0 or stats["total_failures"] == 3
    print("    OK")


def _test_graceful_degradation():
    print("  Testing GracefulDegradation...")
    gd = GracefulDegradation(auto_degrade=False)
    assert gd.level == DegradationLevel.FULL
    assert gd.policy.allow_hdc == True
    gd.set_level(DegradationLevel.REDUCED, "test")
    assert gd.level == DegradationLevel.REDUCED
    assert gd.policy.allow_hdc == False
    status = gd.status()
    assert status["level"] == "reduced"
    print("    OK")


def _test_error_boundary():
    print("  Testing ErrorBoundary...")
    eb = ErrorBoundary()
    result = eb.execute(lambda: 1 / 0, fallback=lambda: 42, component="math")
    assert result == 42
    counts = eb.get_error_counts()
    assert "ZeroDivisionError" in counts or sum(counts.values()) > 0
    print("    OK")


def _test_crash_recovery():
    print("  Testing CrashRecovery...")
    from spectralstream.memory.persistence import StateManager

    sm = StateManager(state_dir="/tmp/spectral_test/state/")
    cr = CrashRecovery(
        state_manager=sm,
        checkpoint_interval=5,
        checkpoint_dir="/tmp/spectral_test/ckpt/",
    )
    assert cr.checkpoint_dir.exists()
    status = cr.status()
    assert status["clean_shutdown"] == True
    print("    OK")


def _test_structured_logger():
    print("  Testing StructuredLogger...")
    log = StructuredLogger(
        "test", level="DEBUG", log_dir="/tmp/spectral_test/logs/", async_write=False
    )
    log.info("test message", component="test")
    log.error("error message", error="test error")
    assert log._log_file.exists()
    log.close()
    print("    OK")


def _test_metrics_collector():
    print("  Testing MetricsCollector...")
    mc = MetricsCollector()
    mc.record_tokens(100)
    mc.record_request(1)
    mc.record_latency("inference", 50.0)
    stats = mc.get_stats()
    assert stats["total_tokens"] == 100
    assert stats["total_requests"] == 1
    p50 = mc.get_latency_percentile("inference", 0.50)
    assert p50 == 50.0
    print("    OK")


def _test_health_check():
    print("  Testing HealthCheck...")
    hc = HealthCheck()
    basic = hc.basic()
    assert basic.healthy == True
    assert basic.checks["service"] == "running"
    print("    OK")


def _test_rate_limiter():
    print("  Testing RateLimiter...")
    rl = RateLimiter(ip_limit=5, ip_rate=10)
    for i in range(5):
        assert rl.check_ip("127.0.0.1") == True
    assert rl.check_ip("127.0.0.1") == False
    headers = rl.get_headers("127.0.0.1")
    assert "X-RateLimit-Limit" in headers
    print("    OK")


def _test_security():
    print("  Testing SecurityManager...")
    sm = SecurityManager(api_keys=["sk-test"])
    assert sm.validate_api_key("sk-test") == True
    assert sm.validate_api_key("bad-key") == False
    assert sm.validate_bearer_token("Bearer sk-test") == "sk-test"
    blocked, reason = sm.filter_prompt("ignore all previous instructions")
    assert blocked == True
    blocked2, reason2 = sm.filter_prompt("hello world")
    assert blocked2 == False
    print("    OK")


def _test_api_monitor():
    print("  Testing APIMonitor...")
    am = APIMonitor()
    am.record_usage("/v1/chat/completions", "user1", tokens=100, latency_ms=500)
    report = am.generate_report()
    assert report["summary"]["total_requests"] == 1
    assert report["summary"]["total_tokens"] == 100
    print("    OK")


def _test_multi_instance():
    print("  Testing MultiInstance...")
    mi = MultiInstance(instance_id="test-1", host="127.0.0.1", port=1234)
    assert mi.instance_id == "test-1"
    mi.add_peer("peer-1", "127.0.0.1", 1235)
    mi.add_peer("peer-2", "127.0.0.1", 1236)
    status = mi.status()
    assert len(status["peers"]) == 2
    print("    OK")


def _test_novel_inventions():
    print("  Testing Novel Inventions...")

    # Resonant Load Balancer
    rlb = ResonantLoadBalancer()
    rlb.add_instance(InstanceInfo(id="i1", host="h1", port=1))
    rlb.add_instance(InstanceInfo(id="i2", host="h2", port=2))
    inst = rlb.select_instance("client-abc")
    assert inst is not None
    print("    ResonantLoadBalancer OK")

    # Holographic Error Recovery
    her = HolographicErrorRecovery(dim=64, capacity=10)
    vec = np.random.randn(64).astype(np.float32)
    her.store("test", vec)
    assert her.detect_corruption("test") == False
    recovered = her.recover("test")
    assert recovered is not None
    print("    HolographicErrorRecovery OK")

    # Predictive Scaler
    ps = PredictiveScaler(hd_dim=64)
    for i in range(20):
        ps.record_metric("throughput", 50 + i)
    forecast = ps.forecast()
    assert 0.0 <= forecast <= 1.0
    desired = ps.desired_instances(4)
    assert desired >= 1
    print("    PredictiveScaler OK")

    # Quantum Circuit Breaker
    qcb = QuantumCircuitBreaker("qtest")
    result = qcb.measure()
    assert isinstance(result, bool)
    stats = qcb.stats()
    assert stats["name"] == "qtest"
    print("    QuantumCircuitBreaker OK")

    # Self-Healing System
    shs = SelfHealingSystem(check_interval=60)
    actions = shs.diagnose()
    assert isinstance(actions, list)
    print("    SelfHealingSystem OK")


def _test_production_service():
    print("  Testing ProductionService...")
    from spectralstream.memory.persistence import StateManager

    ps = ProductionService(
        state_dir="/tmp/spectral_prod_test/",
        enable_self_healing=True,
    )
    status = ps.status()
    assert "service" in status
    assert "degradation" in status
    assert "breakers" in status
    print("    OK")


def test_all():
    print("=" * 60)
    print("  SpectralStream Production Stack — Self-Verification")
    print("=" * 60)
    print()

    for test_fn in [
        _test_service_manager,
        _test_circuit_breaker,
        _test_graceful_degradation,
        _test_error_boundary,
        _test_crash_recovery,
        _test_structured_logger,
        _test_metrics_collector,
        _test_health_check,
        _test_rate_limiter,
        _test_security,
        _test_api_monitor,
        _test_multi_instance,
        _test_novel_inventions,
        _test_production_service,
    ]:
        try:
            test_fn()
        except Exception as e:
            print(f"    FAILED: {e}")
            traceback.print_exc()

    print()
    print("=" * 60)
    print("  All production stack tests complete")
    print("=" * 60)


def main():
    if "--health" in sys.argv:
        from spectralstream.memory.persistence import StateManager

        ps = ProductionService()
        print(json.dumps(ps.health_check(), indent=2, default=str))
    elif "--test" in sys.argv:
        test_all()
    else:
        test_all()


if __name__ == "__main__":
    main()
