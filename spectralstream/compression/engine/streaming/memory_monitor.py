from __future__ import annotations

import gc
import time

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    psutil = None  # type: ignore[assignment]
    HAS_PSUTIL = False


class MemoryMonitor:
    """Tracks process RSS and system-wide available memory.

    Provides signals for GC collection, dynamic chunk-size adjustment,
    and peak-memory tracking. Designed for the 4–16 GB consumer-device
    target (MiMo-V2.5 = 365 GB model weights, 64 GB host RAM).

    Parameters
    ----------
    max_memory_gb : float
        Hard ceiling for process RSS in GB. Chunking will shrink
        automatically when approaching this limit.
    safety_margin : float
        Fraction of ``max_memory_gb`` reserved for OS / page cache.
        Chunking stays within ``max_memory_gb * (1 - safety_margin)``.
    gc_interval : int
        Call ``gc.collect()`` every *N* chunks / tensors.
    """

    def __init__(
        self,
        max_memory_gb: float = 16.0,
        safety_margin: float = 0.25,
        gc_interval: int = 5,
    ) -> None:
        self._max_gb: float = max_memory_gb
        self._safety_margin: float = safety_margin
        self._gc_interval: int = gc_interval
        self._peak_rss_mb: float = 0.0
        self._counter: int = 0
        self._last_signal_time: float = 0.0
        self._gc_backoff: float = 1.0

    # ── Public queries ──────────────────────────────────────────────────

    @property
    def max_memory_gb(self) -> float:
        return self._max_gb

    @max_memory_gb.setter
    def max_memory_gb(self, value: float) -> None:
        self._max_gb = value

    @property
    def safety_margin(self) -> float:
        return self._safety_margin

    @property
    def peak_rss_mb(self) -> float:
        return self._peak_rss_mb

    def current_rss_mb(self) -> float:
        """Return current process RSS in MB."""
        if not HAS_PSUTIL:
            return 0.0
        try:
            rss = psutil.Process().memory_info().rss / (1024 * 1024)
            if rss > self._peak_rss_mb:
                self._peak_rss_mb = rss
            return rss
        except (OSError, AttributeError):
            return 0.0

    def available_memory_mb(self) -> float:
        """Return system-wide available memory in MB."""
        if not HAS_PSUTIL:
            return self._max_gb * 1024
        try:
            return psutil.virtual_memory().available / (1024 * 1024)
        except (OSError, AttributeError):
            return self._max_gb * 1024

    def total_system_memory_gb(self) -> float:
        """Return total system RAM in GB (or ``max_memory_gb`` if psutil missing)."""
        if not HAS_PSUTIL:
            return self._max_gb
        try:
            return psutil.virtual_memory().total / (1024**3)
        except (OSError, AttributeError):
            return self._max_gb

    # ── Budget helpers ──────────────────────────────────────────────────

    def safe_chunk_size_bytes(self, default_chunk_mb: int = 64) -> int:
        """Return chunk size in bytes that fits within the safety margin.

        ``max_memory_gb * (1 - safety_margin)`` is the working buffer.
        We further divide by 4 so multiple chunks can coexist during a
        compress cycle without exceeding the ceiling.
        """
        budget_bytes = int(self._max_gb * (1.0 - self._safety_margin) * (1024**3) / 4)
        min_bytes = 16 * 1024 * 1024  # floor: 16 MB
        max_bytes = 512 * 1024 * 1024  # cap: 512 MB
        proposed = min(budget_bytes, default_chunk_mb * 1024 * 1024)
        return max(min_bytes, min(proposed, max_bytes))

    def tensor_ram_budget_bytes(self) -> int:
        """Return per-tensor RAM budget in bytes.

        This is the amount we allow for loading + compressing one tensor
        (or one chunk) before we must flush to disk.
        """
        usable = self._max_gb * (1.0 - self._safety_margin) * (1024**3)
        return max(16 * 1024 * 1024, int(usable / 2))

    # ── GC signalling ──────────────────────────────────────────────────

    def maybe_gc(self, force: bool = False) -> int:
        """Run ``gc.collect()`` every ``gc_interval`` calls (or on *force*).

        Returns number of collected objects.
        """
        self._counter += 1
        if not force and self._counter % self._gc_interval != 0:
            return 0
        t0 = time.perf_counter()
        collected = gc.collect()
        elapsed = time.perf_counter() - t0

        if collected > 0 and HAS_PSUTIL:
            rss = self.current_rss_mb()
            if elapsed > 0.05:
                self._gc_backoff = min(self._gc_backoff * 1.5, 10.0)

        self._last_signal_time = time.perf_counter()
        return collected

    def pressure_level(self) -> float:
        """Return a float in [0, 1] indicating memory pressure.

        0.0 = plenty of headroom, 1.0 = at or above the safety margin.
        """
        if not HAS_PSUTIL:
            return 0.0
        rss_mb = self.current_rss_mb()
        ceiling_mb = self._max_gb * 1024 * (1.0 - self._safety_margin)
        if ceiling_mb <= 0:
            return 1.0
        return min(1.0, rss_mb / ceiling_mb)

    def is_stressed(self) -> bool:
        """Return True if memory pressure exceeds 80 %."""
        return self.pressure_level() > 0.8

    # ── Reporting ──────────────────────────────────────────────────────

    def report(self) -> dict:
        """Return a snapshot dict of all monitor state."""
        return {
            "peak_rss_mb": self._peak_rss_mb,
            "current_rss_mb": self.current_rss_mb(),
            "available_mb": self.available_memory_mb(),
            "pressure": self.pressure_level(),
            "stressed": self.is_stressed(),
            "safe_chunk_bytes": self.safe_chunk_size_bytes(),
            "ram_budget_bytes": self.tensor_ram_budget_bytes(),
            "max_memory_gb": self._max_gb,
            "safety_margin": self._safety_margin,
            "gc_backoff": self._gc_backoff,
        }
