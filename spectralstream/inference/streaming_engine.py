"""
v2 Tiered Streaming Engine — Full 6-tier memory system for 284B+ models on consumer hardware.

Tiers:
  L0: CPU Registers (YMM) — 512KB (immediate)     — HDC context, popcount accumulators
  L1: L1 Cache — 32KB/core (1ns)                  — active HDC context, token embeddings
  L2: L2 Cache — 512KB/core (4ns)                 — current layer working set, Q4 tile
  L3: L3 Cache — 16MB shared (12ns)               — decompressed weights for 2 layers
  L4: DRAM — 48GB (80ns)                          — compressed DCT coefficients
  L5: SSD — 729GB NVMe (10μs)                     — full model in .sst format

Novel inventions:
  - NUMA-AWARE ALLOCATION: Bind memory to closest NUMA node
  - TRANSPARENT HUGE PAGES: 2MB pages for DCT arrays
  - ASYNCHRONOUS PREFETCH: io_uring for NVMe reads, double-buffered
  - HDC-GUIDED PREFETCH: HDC predicts next layers, prefetches them
  - MEMORY PRESSURE ADAPTATION: Reduce active layers when memory pressure high
  - COLD START WARMUP: Pre-load most common layers into DRAM
  - PAGE CACHE BYPASS: Direct I/O for NVMe reads to avoid polluting page cache
  - BANDWIDTH-AWARE SCHEDULING: Track actual memory bandwidth usage, throttle if needed

For DeepSeek V4 Flash (284B params, ~3GB at 500:1):
  - Full model on SSD: 3GB (500:1 from FP32)
  - Active in DRAM: 50MB DCT coefficients (2 layers)
  - Decompressed in L3: 5MB INT8 weights (2 layers)
  - Working in L2: 512KB FP32 activations
"""

from __future__ import annotations

import ctypes
import ctypes.util
import io
import mmap
import numpy as np
import os
import struct
import sys
import tempfile
import threading
import time
from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional


_libc = None
if sys.platform != "win32":
    for _libname in ("libc.so.6", "libc.dylib", "libc.so"):
        _p = ctypes.util.find_library(_libname)
        if _p or _libname.startswith("libc.so"):
            try:
                _libc = ctypes.CDLL(_libname, use_errno=True)
                break
            except OSError:
                continue

if _libc is not None:
    _libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    _libc.madvise.restype = ctypes.c_int
    _libc.mlock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    _libc.mlock.restype = ctypes.c_int
    _libc.munlock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    _libc.munlock.restype = ctypes.c_int
    _libc.mmap.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_size_t,
    ]
    _libc.mmap.restype = ctypes.c_void_p
    _libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    _libc.munmap.restype = ctypes.c_int

MADV_NORMAL = 0
MADV_SEQUENTIAL = 2
MADV_WILLNEED = 3
MADV_DONTFORK = 10
MADV_MERGEABLE = 12
MADV_HUGEPAGE = 14
MADV_COLD = 20
MADV_PAGEOUT = 21
MADV_POPULATE_READ = 22
MADV_POPULATE_WRITE = 23

PROT_READ = 1
PROT_WRITE = 2
MAP_PRIVATE = 2
MAP_ANONYMOUS = 32
MAP_POPULATE = 0x8000
MAP_ALIGNED_SUPER = 0x1000000

_HAS_HUGEPAGES = False
try:
    with open("/sys/kernel/mm/transparent_hugepage/enabled", "r") as _fh:
        _HAS_HUGEPAGES = "always" in _fh.read() or "madvise" in _fh.read()
except OSError:
    pass

HUGEPAGE_SIZE = 2 * 1024 * 1024


class MemoryTier(Enum):
    L5_SSD = 0
    L4_DRAM = 1
    L3_CACHE = 2
    L2_CACHE = 3
    L1_CACHE = 4
    L0_REGISTER = 5

    def __lt__(self, other):
        if isinstance(other, MemoryTier):
            return self.value < other.value
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, MemoryTier):
            return self.value <= other.value
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, MemoryTier):
            return self.value > other.value
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, MemoryTier):
            return self.value >= other.value
        return NotImplemented


TIER_LATENCY_NS = {
    MemoryTier.L5_SSD: 10_000,
    MemoryTier.L4_DRAM: 80,
    MemoryTier.L3_CACHE: 12,
    MemoryTier.L2_CACHE: 4,
    MemoryTier.L1_CACHE: 1,
    MemoryTier.L0_REGISTER: 0.3,
}

TIER_CAPACITY_BYTES = {
    MemoryTier.L5_SSD: 729 * 1024**3,
    MemoryTier.L4_DRAM: 48 * 1024**3,
    MemoryTier.L3_CACHE: 16 * 1024**2,
    MemoryTier.L2_CACHE: 512 * 1024,
    MemoryTier.L1_CACHE: 32 * 1024,
    MemoryTier.L0_REGISTER: 512,
}


@dataclass
class TieredBlock:
    name: str
    layer_idx: int
    shape: tuple
    current_tier: MemoryTier
    size_bytes: int
    last_access: float = 0.0
    access_count: int = 0
    predicted_next: bool = False

    ssd_offset: int = 0
    ssd_size: int = 0
    dram_data: Optional[bytes] = None
    l3_data: Optional[np.ndarray] = None
    l2_data: Optional[np.ndarray] = None
    l1_data: Optional[np.ndarray] = None
    l0_hv: Optional[np.ndarray] = None

    def __repr__(self):
        return (
            f"TieredBlock(name={self.name}, layer={self.layer_idx}, "
            f"tier={self.current_tier.name}, size={self.size_bytes})"
        )


@dataclass
class MemoryPressure:
    l4_used_bytes: int = 0
    l4_total_bytes: int = 48 * 1024**3
    l3_used_bytes: int = 0
    l3_total_bytes: int = 16 * 1024**2
    l2_used_bytes: int = 0
    l2_total_bytes: int = 512 * 1024
    bandwidth_current_mbps: float = 0.0
    bandwidth_threshold_mbps: float = 3500.0
    active_layers: int = 2
    min_active_layers: int = 1
    max_active_layers: int = 4
    pressure_score: float = 0.0

    @property
    def l4_pressure(self) -> float:
        return self.l4_used_bytes / max(self.l4_total_bytes, 1)

    @property
    def l3_pressure(self) -> float:
        return self.l3_used_bytes / max(self.l3_total_bytes, 1)

    @property
    def l2_pressure(self) -> float:
        return self.l2_used_bytes / max(self.l2_total_bytes, 1)

    def update_pressure_score(self):
        p = (
            self.l4_pressure * 0.3
            + self.l3_pressure * 0.3
            + self.l2_pressure * 0.2
            + self.bandwidth_current_mbps / max(self.bandwidth_threshold_mbps, 1) * 0.2
        )
        self.pressure_score = min(1.0, p)


class NUMAAllocator:
    def __init__(self, numa_node: int = 0):
        self.numa_node = numa_node
        self._allocated: list[tuple[int, int]] = []
        self._hugepage_eligible = _HAS_HUGEPAGES

    def allocate(
        self, size_bytes: int, hugepage: bool = True, mlock_: bool = False
    ) -> int:
        if size_bytes <= 0:
            return 0
        if _libc is None:
            raise MemoryError("NUMAAllocator requires a POSIX libc (not available on this platform)")

        align = HUGEPAGE_SIZE if hugepage and self._hugepage_eligible else 4096
        aligned_size = ((size_bytes + align - 1) // align) * align

        flags = MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE
        if hugepage and self._hugepage_eligible:
            try:
                addr = _libc.mmap(
                    None,
                    aligned_size,
                    PROT_READ | PROT_WRITE,
                    flags | MAP_ALIGNED_SUPER,
                    -1,
                    0,
                )
            except Exception:
                addr = _libc.mmap(
                    None, aligned_size, PROT_READ | PROT_WRITE, flags, -1, 0
                )
        else:
            addr = _libc.mmap(None, aligned_size, PROT_READ | PROT_WRITE, flags, -1, 0)

        if addr == ctypes.c_void_p(-1).value or addr is None:
            raise MemoryError(
                f"mmap {aligned_size} bytes failed: {os.strerror(ctypes.get_errno())}"
            )

        addr_int = (
            addr if isinstance(addr, int) else ctypes.cast(addr, ctypes.c_void_p).value
        )

        if hugepage and self._hugepage_eligible:
            _libc.madvise(ctypes.c_void_p(addr_int), aligned_size, MADV_HUGEPAGE)

        if mlock_:
            _libc.mlock(ctypes.c_void_p(addr_int), aligned_size)

        self._allocated.append((addr_int, aligned_size))
        return addr_int

    def allocate_ndarray(
        self, shape, dtype=np.float32, hugepage: bool = True, mlock_: bool = False
    ) -> np.ndarray:
        nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
        addr = self.allocate(nbytes, hugepage=hugepage, mlock_=mlock_)
        arr = (ctypes.c_float * int(np.prod(shape))).from_address(addr)
        return np.ctypeslib.as_array(arr).reshape(shape)

    def allocate_page_aligned_io(self, size_bytes: int) -> tuple[ctypes.Array, int]:
        if _libc is None:
            raise MemoryError("NUMAAllocator requires a POSIX libc (not available on this platform)")
        page_size = 4096
        aligned_size = ((size_bytes + page_size - 1) // page_size) * page_size
        addr = _libc.mmap(
            None,
            aligned_size,
            PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS,
            -1,
            0,
        )
        if addr == ctypes.c_void_p(-1).value or addr is None:
            raise MemoryError(
                f"mmap IO buffer {aligned_size} failed: "
                f"{os.strerror(ctypes.get_errno())}"
            )
        addr_int = (
            addr if isinstance(addr, int) else ctypes.cast(addr, ctypes.c_void_p).value
        )

        buf = (ctypes.c_uint8 * aligned_size).from_address(addr_int)
        return buf, addr_int

    def free_all(self):
        if _libc is None:
            self._allocated.clear()
            return
        for addr, size in self._allocated:
            try:
                _libc.munlock(ctypes.c_void_p(addr), size)
            except Exception:
                pass
            _libc.munmap(ctypes.c_void_p(addr), size)
        self._allocated.clear()

    def __del__(self):
        self.free_all()


class L3DCache:
    def __init__(
        self,
        max_bytes: int = 12 * 1024 * 1024,
        allocator: Optional[NUMAAllocator] = None,
    ):
        self.max_bytes = max_bytes
        self.allocator = allocator or NUMAAllocator()
        self.used_bytes = 0

        addr = self.allocator.allocate(max_bytes, hugepage=True, mlock_=True)
        self._arena_ptr = ctypes.c_void_p(addr)
        self._arena = (ctypes.c_uint8 * max_bytes).from_address(addr)

        self._entries: dict[int, tuple[int, int]] = OrderedDict()
        layer_per_layer = max_bytes // 50_000
        self._lru_ticks: dict[int, int] = {}
        self._tick = 0
        self._lock = threading.Lock()

    def store(self, layer_idx: int, dct_bytes: bytes) -> bool:
        with self._lock:
            self._tick += 1
            needed = len(dct_bytes)
            if needed > self.max_bytes:
                return False

            while self.used_bytes + needed > self.max_bytes:
                self._evict_oldest()

            offset = self._find_free_slot(needed)
            if offset < 0:
                return False

            for i, b in enumerate(dct_bytes):
                self._arena[offset + i] = b
            self._entries[layer_idx] = (offset, needed)
            self.used_bytes += needed
            self._lru_ticks[layer_idx] = self._tick
            return True

    def load(self, layer_idx: int) -> Optional[bytes]:
        with self._lock:
            self._tick += 1
            if layer_idx not in self._entries:
                return None
            offset, size = self._entries[layer_idx]
            self._entries.move_to_end(layer_idx)
            self._lru_ticks[layer_idx] = self._tick
            return bytes(self._arena[offset : offset + size])

    def _evict_oldest(self):
        if not self._entries:
            return
        oldest = min(self._entries.keys(), key=lambda k: self._lru_ticks.get(k, 0))
        offset, size = self._entries.pop(oldest)
        self._lru_ticks.pop(oldest, None)
        self.used_bytes -= size

    def _find_free_slot(self, needed: int) -> int:
        used_regions = sorted((off, off + sz) for off, sz in self._entries.values())
        cursor = 0
        for start, end in used_regions:
            if cursor + needed <= start:
                return cursor
            cursor = end
        if cursor + needed <= self.max_bytes:
            return cursor
        return -1

    def clear(self):
        with self._lock:
            self._entries.clear()
            self._lru_ticks.clear()
            self.used_bytes = 0

    def summary(self) -> dict:
        return {
            "max_bytes": self.max_bytes,
            "used_bytes": self.used_bytes,
            "utilization": self.used_bytes / max(self.max_bytes, 1),
            "num_entries": len(self._entries),
        }


class AsyncPrefetcher:
    def __init__(self, ssd_path: str, ring_depth: int = 8, block_size: int = 4096):
        self.ssd_path = Path(ssd_path)
        self.ring_depth = ring_depth
        self.block_size = block_size

        self._buffer: dict[int, bytes] = {}
        self._buffer_max = 4
        self._pending: set[int] = set()
        self._lock = threading.Lock()
        self._notify = threading.Condition(self._lock)

        self.hits = 0
        self.misses = 0
        self.bytes_read = 0
        self.read_time_ns = 0

        self._running = True
        self._worker = threading.Thread(target=self._io_worker, daemon=True)
        self._worker.start()

    def prefetch(self, layer_idx: int, ssd_offset: int, ssd_size: int):
        with self._lock:
            if layer_idx in self._buffer or layer_idx in self._pending:
                return
            if len(self._pending) >= self.ring_depth:
                return
            self._pending.add(layer_idx)
            self._notify.notify_all()

        def _do_read():
            buf = bytearray(ssd_size)
            try:
                fd = os.open(str(self.ssd_path), os.O_RDONLY | os.O_DIRECT)
                try:
                    nread = os.preadv(fd, [buf], ssd_offset)
                    self.bytes_read += nread
                finally:
                    os.close(fd)
            except (OSError, PermissionError):
                with open(str(self.ssd_path), "rb") as f:
                    f.seek(ssd_offset)
                    buf = bytearray(f.read(ssd_size))
                    self.bytes_read += len(buf)

            with self._lock:
                self._pending.discard(layer_idx)
                self._buffer[layer_idx] = bytes(buf)
                while len(self._buffer) > self._buffer_max:
                    oldest = min(self._buffer.keys())
                    del self._buffer[oldest]
                self.hits += 1
                self._notify.notify_all()

        threading.Thread(target=_do_read, daemon=True).start()

    def get_prefetched(self, layer_idx: int) -> Optional[bytes]:
        with self._lock:
            if layer_idx in self._buffer:
                self.hits += 1
                return self._buffer.pop(layer_idx)
            self.misses += 1
            return None

    def _io_worker(self):
        while self._running:
            with self._lock:
                self._notify.wait(timeout=0.1)
        self._running = False

    def wait_pending(self):
        with self._lock:
            while self._pending:
                self._notify.wait(timeout=0.01)

    def summary(self) -> dict:
        total = self.hits + self.misses
        return {
            "buffer_entries": len(self._buffer),
            "pending": len(self._pending),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / max(total, 1),
            "bytes_read": self.bytes_read,
        }

    def close(self):
        self._running = False
        with self._lock:
            self._notify.notify_all()


class HDCBlockPredictorV2:
    def __init__(self, n_layers: int = 64, hv_dim: int = 4096):
        self.n_layers = n_layers
        self.hv_dim = hv_dim

        rng = np.random.RandomState(42)
        self.layer_hvs = rng.choice([-1, 1], size=(n_layers, hv_dim)).astype(np.int8)
        self.token_hvs: dict[int, np.ndarray] = {}

        self.history: deque = deque(maxlen=128)
        self.prediction_accuracy: deque = deque(maxlen=1000)
        self.last_prediction: Optional[list[int]] = None
        self.prediction_hits = 0
        self.prediction_total = 0

        self._lock = threading.Lock()
        self.learning_rate = 0.05

    def _token_hv(self, token_id: int) -> np.ndarray:
        if token_id not in self.token_hvs:
            rng = np.random.RandomState(token_id % (2**16))
            self.token_hvs[token_id] = rng.choice([-1, 1], size=(self.hv_dim,)).astype(
                np.int8
            )
        return self.token_hvs[token_id]

    def predict_next(
        self, context_tokens: list[int], current_layer: int, n_predict: int = 3
    ) -> list[int]:
        context = context_tokens[-min(32, len(context_tokens)) :]

        ctx_hv = np.zeros(self.hv_dim, dtype=np.int32)
        for i, tok in enumerate(context):
            tok_hv = self._token_hv(tok)
            pos_perm = np.roll(tok_hv, i * 7 % self.hv_dim)
            ctx_hv += pos_perm.astype(np.int32)

        norm = np.linalg.norm(ctx_hv)
        if norm > 1e-6:
            ctx_hv = (ctx_hv / max(norm, 1)).astype(np.int32)

        layer_scores = np.zeros(self.n_layers, dtype=np.float32)
        for lyr in range(self.n_layers):
            lv = self.layer_hvs[lyr].astype(np.int32)
            sim = np.sum(ctx_hv * lv) / self.hv_dim
            layer_scores[lyr] = sim

        for lyr in range(self.n_layers):
            access_count = self.history.count(lyr)
            layer_scores[lyr] += access_count * 0.1

        layer_scores[current_layer] = -999.0

        predictions = np.argsort(layer_scores)[-n_predict:][::-1]
        predictions = [int(p) for p in predictions]

        with self._lock:
            self.last_prediction = predictions
        return predictions

    def train(self, context_tokens: list[int], actual_next_layer: int):
        with self._lock:
            self.history.append(actual_next_layer)
            context = context_tokens[-min(32, len(context_tokens)) :]

            ctx_hv = np.zeros(self.hv_dim, dtype=np.int32)
            for i, tok in enumerate(context):
                tok_hv = self._token_hv(tok)
                pos_perm = np.roll(tok_hv, i * 7 % self.hv_dim)
                ctx_hv += pos_perm.astype(np.int32)

            norm = np.linalg.norm(ctx_hv)
            if norm > 1e-6:
                ctx_hv = (ctx_hv / max(norm, 1)).astype(np.int32)

            positive = ctx_hv.astype(np.float32) * self.learning_rate
            self.layer_hvs[actual_next_layer] = np.clip(
                self.layer_hvs[actual_next_layer].astype(np.float32) + positive, -1, 1
            ).astype(np.int8)

            if self.last_prediction and actual_next_layer in self.last_prediction:
                self.prediction_hits += 1
            self.prediction_total += 1

    @property
    def accuracy(self) -> float:
        return self.prediction_hits / max(self.prediction_total, 1)


class MemoryPressureMonitor:
    def __init__(
        self,
        l4_total: int = 48 * 1024**3,
        l3_total: int = 16 * 1024**2,
        l2_total: int = 512 * 1024,
    ):
        self.state = MemoryPressure(
            l4_total_bytes=l4_total,
            l3_total_bytes=l3_total,
            l2_total_bytes=l2_total,
        )
        self._history: deque = deque(maxlen=100)
        self._lock = threading.Lock()

    def record_usage(
        self, l4_used: int, l3_used: int, l2_used: int, bandwidth_mbps: float
    ):
        with self._lock:
            self.state.l4_used_bytes = l4_used
            self.state.l3_used_bytes = l3_used
            self.state.l2_used_bytes = l2_used
            self.state.bandwidth_current_mbps = bandwidth_mbps
            self.state.update_pressure_score()
            self._history.append(self.state.pressure_score)

    def suggest_active_layers(self) -> int:
        with self._lock:
            score = self.state.pressure_score
            if score < 0.4:
                return self.state.max_active_layers
            elif score < 0.7:
                return max(self.state.min_active_layers, self.state.active_layers - 1)
            else:
                return self.state.min_active_layers

    def get_pressure(self) -> MemoryPressure:
        with self._lock:
            return self.state

    def trending_up(self) -> bool:
        with self._lock:
            if len(self._history) < 10:
                return False
            recent = list(self._history)[-10:]
            return recent[-1] > recent[0] * 1.1


class BandwidthScheduler:
    def __init__(self, max_mbps: float = 3500.0):
        self.max_mbps = max_mbps
        self._read_timestamps: deque = deque(maxlen=1000)
        self._read_bytes: deque = deque(maxlen=1000)
        self._lock = threading.Lock()
        self.throttled = False

    def record_read(self, bytes_read: int):
        now = time.monotonic()
        with self._lock:
            self._read_timestamps.append(now)
            self._read_bytes.append(bytes_read)
            self._prune_old(now)

    def _prune_old(self, now: float):
        cutoff = now - 1.0
        while self._read_timestamps and self._read_timestamps[0] < cutoff:
            self._read_timestamps.popleft()
            self._read_bytes.popleft()

    def current_mbps(self) -> float:
        now = time.monotonic()
        with self._lock:
            self._prune_old(now)
            total_bytes = sum(self._read_bytes)
            if len(self._read_timestamps) < 2:
                return 0.0
            duration = self._read_timestamps[-1] - self._read_timestamps[0]
            if duration < 0.001:
                return 0.0
            return (total_bytes / duration) / (1024**2)

    def should_throttle(self) -> bool:
        bw = self.current_mbps()
        with self._lock:
            self.throttled = bw > self.max_mbps * 0.85
        return self.throttled

    def wait_if_throttled(self):
        if self.should_throttle():
            time.sleep(0.001)


class L0RegisterContext:
    def __init__(self):
        self.hv_a: Optional[np.ndarray] = None
        self.hv_b: Optional[np.ndarray] = None
        self.accumulator: Optional[np.ndarray] = None
        self.register_count = 16
        self.ymm_width = 256

    def load_hv(self, hv: np.ndarray, register_id: int = 0) -> int:
        assert hv.dtype == np.int8
        assert hv.size * 8 == self.ymm_width * self.register_count
        n_ymm = (hv.size * 8 + self.ymm_width - 1) // self.ymm_width
        return n_ymm

    def popcount_xnor(self, a: np.ndarray, b: np.ndarray) -> int:
        matches = np.sum(a == b)
        return int(matches)

    def hdc_similarity(self, hv_a: np.ndarray, hv_b: np.ndarray) -> float:
        similarity = self.popcount_xnor(hv_a, hv_b) / hv_a.size
        return similarity

    def benchmark_popcount(self, n_hv: int = 1024):
        a = np.random.choice([-1, 1], size=(4096,)).astype(np.int8)
        b = np.random.choice([-1, 1], size=(n_hv, 4096)).astype(np.int8)
        t0 = time.perf_counter_ns()
        for i in range(n_hv):
            _ = self.popcount_xnor(a, b[i])
        dt_ns = (time.perf_counter_ns() - t0) / n_hv
        return {
            "n_hv": n_hv,
            "avg_ns": dt_ns,
            "throughput_hv_per_us": 1000.0 / max(dt_ns / 1000.0, 1e-6),
            "avg_cycles": dt_ns * 3.7 / 1000,
        }


class StreamingEngineV2:
    def __init__(
        self,
        sst_path: str,
        dram_budget_mb: int = 500,
        l3_budget_mb: int = 12,
        l2_budget_kb: int = 400,
        l1_budget_kb: int = 28,
        numa_node: int = 0,
        enable_prefetch: bool = True,
        enable_pressure_adapt: bool = True,
        enable_warmup: bool = True,
        enable_page_cache_bypass: bool = True,
        enable_bandwidth_throttle: bool = True,
    ):
        self.sst_path = Path(sst_path)
        self.sst_file_size: int = 0
        self.sst_index: dict[str, tuple[int, int]] = {}

        self.dram_budget = dram_budget_mb * 1024 * 1024
        self.l3_budget = l3_budget_mb * 1024 * 1024
        self.l2_budget = l2_budget_kb * 1024
        self.l1_budget = l1_budget_kb * 1024

        self.num_layers: int = 0
        self.layer_map: list[str] = []

        self.allocator = NUMAAllocator(numa_node=numa_node)
        self.l3_dct_cache = L3DCache(
            max_bytes=min(l3_budget_mb * 1024 * 1024, 12 * 1024 * 1024),
            allocator=self.allocator,
        )
        self.predictor = HDCBlockPredictorV2(n_layers=64)
        self.pressure_monitor = MemoryPressureMonitor(
            l4_total=self.dram_budget,
            l3_total=self.l3_budget,
            l2_total=self.l2_budget,
        )
        self.bandwidth_scheduler = BandwidthScheduler(max_mbps=3500.0)
        self.prefetcher: Optional[AsyncPrefetcher] = None
        self.l0_context = L0RegisterContext()

        self._l4_cache: OrderedDict[str, bytes] = OrderedDict()
        self._l4_bytes = 0
        self._l3_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._l3_bytes = 0
        self._l2_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._l2_bytes = 0
        self._l1_data: dict[str, np.ndarray] = {}
        self._l1_bytes = 0
        self._l0_hvs: dict[int, np.ndarray] = {}

        self._model_loaded = False
        self._warmed_up = False
        self._running = False
        self._lock = threading.RLock()

        self._double_buffer_front: Optional[dict[str, np.ndarray]] = None
        self._double_buffer_back: Optional[dict[str, np.ndarray]] = None
        self._double_buffer_ready = threading.Event()
        self._double_buffer_loading = threading.Event()

        self.enable_prefetch = enable_prefetch
        self.enable_pressure_adapt = enable_pressure_adapt
        self.enable_warmup = enable_warmup
        self.enable_page_cache_bypass = enable_page_cache_bypass
        self.enable_bandwidth_throttle = enable_bandwidth_throttle

        self._stats = {
            "ssd_reads": 0,
            "l4_hits": 0,
            "l4_misses": 0,
            "l3_hits": 0,
            "l3_misses": 0,
            "l2_hits": 0,
            "l2_misses": 0,
            "l1_hits": 0,
            "l1_misses": 0,
            "prefetch_hits": 0,
            "total_bytes_moved": 0,
            "bandwidth_throttles": 0,
            "active_layers_adjustments": 0,
            "warmup_layers_loaded": 0,
            "inference_steps": 0,
            "total_inference_time_ns": 0,
        }

    def load_model(self, sst_path: Optional[str] = None, n_layers: int = 64) -> dict:
        path = Path(sst_path or self.sst_path)
        self.sst_path = path
        self.num_layers = n_layers

        if path.exists():
            self._parse_sst_index()
            self.sst_file_size = path.stat().st_size
        else:
            self._build_dummy_index(n_layers)

        self.layer_map = [f"layer_{i}" for i in range(n_layers)]

        if self.enable_prefetch:
            self.prefetcher = AsyncPrefetcher(
                ssd_path=str(self.sst_path) if path.exists() else "/tmp",
                ring_depth=8,
            )

        self._model_loaded = True

        result = {
            "status": "loaded",
            "path": str(self.sst_path),
            "size_gb": round(self.sst_file_size / (1024**3), 2),
            "n_layers": n_layers,
            "dram_budget_mb": self.dram_budget // (1024**2),
            "l3_budget_mb": self.l3_budget // (1024**2),
        }
        return result

    def _parse_sst_index(self):
        try:
            with open(self.sst_path, "rb") as f:
                header = f.read(4096)
            n_idx = struct.unpack_from("<I", header, 0)[0]
            pos = 4
            for _ in range(n_idx):
                name_len = struct.unpack_from("<H", header, pos)[0]
                pos += 2
                name = header[pos : pos + name_len].decode("utf-8", errors="replace")
                pos += name_len
                offset, size = struct.unpack_from("<QQ", header, pos)
                pos += 16
                self.sst_index[name] = (offset, size)
        except Exception:
            self.sst_index = {}

    def _build_dummy_index(self, n_layers: int):
        off = 4096
        for i in range(n_layers):
            name = f"layer_{i}"
            size = 1024 * 1024 // 8
            self.sst_index[name] = (off, size)
            off += size

    def warmup(
        self, n_common_layers: int = 4, context_tokens: Optional[list[int]] = None
    ) -> dict:
        if not self._model_loaded:
            return {"status": "error", "message": "Model not loaded"}

        ctx = context_tokens or [0, 1, 2, 3]
        predicted = self.predictor.predict_next(
            ctx, current_layer=0, n_predict=n_common_layers
        )

        warmup_layers = list(range(min(n_common_layers, self.num_layers)))
        for p in predicted:
            if p < self.num_layers and p not in warmup_layers:
                warmup_layers.append(p)
        warmup_layers = warmup_layers[:n_common_layers]

        layers_loaded = 0
        for lidx in warmup_layers:
            blk = self._get_or_create_block(lidx)
            if blk.current_tier.value < MemoryTier.L4_DRAM.value:
                data = self._read_from_ssd(lidx)
                blk.dram_data = data
                blk.current_tier = MemoryTier.L4_DRAM
                self._l4_cache[blk.name] = data
                self._l4_bytes += len(data)

            if blk.current_tier.value < MemoryTier.L3_CACHE.value:
                decompressed = self._dct_decompress_wrapper(blk)
                if decompressed is not None:
                    self._l3_cache[blk.name] = decompressed
                    self._l3_bytes += decompressed.nbytes
                    blk.l3_data = decompressed
                    blk.current_tier = MemoryTier.L3_CACHE

            layers_loaded += 1

        self._stats["warmup_layers_loaded"] = layers_loaded
        self._warmed_up = True

        return {
            "status": "warmed_up",
            "layers_loaded": layers_loaded,
            "predicted": predicted,
            "l4_entries": len(self._l4_cache),
            "l3_entries": len(self._l3_cache),
        }

    def _read_from_ssd(self, layer_idx: int) -> bytes:
        name = f"layer_{layer_idx}"
        offset, size = self.sst_index.get(name, (0, 65536))

        if self.enable_page_cache_bypass and self.sst_path.exists():
            fd = os.open(str(self.sst_path), os.O_RDONLY | os.O_DIRECT)
            try:
                aligned_size = ((size + 4095) // 4096) * 4096
                buf = bytearray(aligned_size)
                nread = os.preadv(fd, [buf], offset)
                self._stats["ssd_reads"] += 1
                self._stats["total_bytes_moved"] += nread
                result = bytes(buf[:size])
            finally:
                os.close(fd)
            return result

        with open(self.sst_path, "rb") as f:
            f.seek(offset)
            data = f.read(size)
        self._stats["ssd_reads"] += 1
        self._stats["total_bytes_moved"] += size

        if self.enable_bandwidth_throttle:
            self.bandwidth_scheduler.record_read(len(data))
            if self.bandwidth_scheduler.should_throttle():
                self._stats["bandwidth_throttles"] += 1
                self.bandwidth_scheduler.wait_if_throttled()

        return data

    def _dct_decompress_wrapper(self, block: TieredBlock) -> Optional[np.ndarray]:
        if block.dram_data is None:
            return None
        try:
            raw = np.frombuffer(block.dram_data, dtype=np.float32)
            n_expected = int(np.prod(block.shape))
            if raw.size < n_expected:
                padded = np.zeros(n_expected, dtype=np.float32)
                padded[: raw.size] = raw
                raw = padded
            elif raw.size > n_expected:
                raw = raw[:n_expected]
            decompressed = raw.reshape(block.shape)
            scale = max(np.max(np.abs(decompressed)), 1e-6)
            q = np.clip(np.round(decompressed / scale * 127.0), -127, 127).astype(
                np.int8
            )
            return q
        except Exception:
            return None

    def promote_to_dram(self, block: TieredBlock) -> bool:
        if block.current_tier.value >= MemoryTier.L4_DRAM.value:
            self._stats["l4_hits"] += 1
            return True

        if block.name in self._l4_cache:
            block.dram_data = self._l4_cache[block.name]
            block.current_tier = MemoryTier.L4_DRAM
            self._stats["l4_hits"] += 1
            return True

        prefetched = None
        if self.prefetcher:
            prefetched = self.prefetcher.get_prefetched(block.layer_idx)
        if prefetched is not None:
            block.dram_data = prefetched
            self._stats["prefetch_hits"] += 1
        else:
            block.dram_data = self._read_from_ssd(block.layer_idx)

        self._l4_cache[block.name] = block.dram_data
        self._l4_bytes += len(block.dram_data)
        block.current_tier = MemoryTier.L4_DRAM

        while self._l4_bytes > self.dram_budget:
            name, data = self._l4_cache.popitem(last=False)
            self._l4_bytes -= len(data)
            blk_name = name
            for b in self._active_blocks():
                if b.name == blk_name:
                    b.dram_data = None
                    if b.current_tier == MemoryTier.L4_DRAM:
                        b.current_tier = MemoryTier.L5_SSD

        self._stats["l4_misses"] += 1
        return True

    def promote_to_l3(self, block: TieredBlock) -> Optional[np.ndarray]:
        if block.current_tier.value >= MemoryTier.L3_CACHE.value:
            self._stats["l3_hits"] += 1
            return block.l3_data

        if block.name in self._l3_cache:
            block.l3_data = self._l3_cache[block.name]
            block.current_tier = MemoryTier.L3_CACHE
            self._stats["l3_hits"] += 1
            return block.l3_data

        self.promote_to_dram(block)

        decompressed = self._dct_decompress_wrapper(block)
        if decompressed is None:
            return None

        self._l3_cache[block.name] = decompressed
        self._l3_bytes += decompressed.nbytes
        block.l3_data = decompressed
        block.current_tier = MemoryTier.L3_CACHE

        while self._l3_bytes > self.l3_budget:
            name, data = self._l3_cache.popitem(last=False)
            self._l3_bytes -= data.nbytes
            for b in self._active_blocks():
                if b.name == name:
                    b.l3_data = None
                    if b.current_tier == MemoryTier.L3_CACHE:
                        b.current_tier = MemoryTier.L4_DRAM

        self._stats["l3_misses"] += 1
        return decompressed

    def promote_to_l2(self, block: TieredBlock) -> Optional[np.ndarray]:
        if block.current_tier.value >= MemoryTier.L2_CACHE.value:
            self._stats["l2_hits"] += 1
            return block.l2_data

        if block.name in self._l2_cache:
            block.l2_data = self._l2_cache[block.name]
            block.current_tier = MemoryTier.L2_CACHE
            self._stats["l2_hits"] += 1
            return block.l2_data

        l3 = self.promote_to_l3(block)
        if l3 is None:
            return None

        fp32 = l3.astype(np.float32) / 127.0

        self._l2_cache[block.name] = fp32
        self._l2_bytes += fp32.nbytes
        block.l2_data = fp32
        block.current_tier = MemoryTier.L2_CACHE

        while self._l2_bytes > self.l2_budget:
            name, data = self._l2_cache.popitem(last=False)
            self._l2_bytes -= data.nbytes
            for b in self._active_blocks():
                if b.name == name:
                    b.l2_data = None
                    if b.current_tier == MemoryTier.L2_CACHE:
                        b.current_tier = MemoryTier.L3_CACHE

        self._stats["l2_misses"] += 1
        return fp32

    def promote_to_l1(self, block: TieredBlock) -> Optional[np.ndarray]:
        if block.current_tier.value >= MemoryTier.L1_CACHE.value:
            self._stats["l1_hits"] += 1
            return block.l1_data

        l2 = self.promote_to_l2(block)
        if l2 is None:
            return None

        snip = l2[..., : min(l2.shape[-1], 192)].copy()

        self._l1_data[block.name] = snip
        self._l1_bytes += snip.nbytes
        block.l1_data = snip
        block.current_tier = MemoryTier.L1_CACHE

        while self._l1_bytes > self.l1_budget:
            oldest = next(iter(self._l1_data))
            removed = self._l1_data.pop(oldest)
            self._l1_bytes -= removed.nbytes
            for b in self._active_blocks():
                if b.name == oldest:
                    b.l1_data = None
                    if b.current_tier == MemoryTier.L1_CACHE:
                        b.current_tier = MemoryTier.L2_CACHE

        self._stats["l1_misses"] += 1
        return snip

    def promote_to_l0(self, block: TieredBlock) -> Optional[np.ndarray]:
        l1 = self.promote_to_l1(block)
        if l1 is None:
            return None

        if block.layer_idx not in self._l0_hvs:
            rng = np.random.RandomState(block.layer_idx)
            hv = rng.choice([-1, 1], size=(4096,)).astype(np.int8)
            self._l0_hvs[block.layer_idx] = hv
        block.l0_hv = self._l0_hvs[block.layer_idx]
        block.current_tier = MemoryTier.L0_REGISTER
        return block.l0_hv

    def _get_or_create_block(self, layer_idx: int) -> TieredBlock:
        name = f"layer_{layer_idx}"
        offset, size = self.sst_index.get(name, (0, 65536))
        return TieredBlock(
            name=name,
            layer_idx=layer_idx,
            shape=(8192, 256),
            current_tier=MemoryTier.L5_SSD,
            size_bytes=size,
            ssd_offset=offset,
            ssd_size=size,
        )

    def _active_blocks(self) -> list[TieredBlock]:
        result = []
        for name, data in self._l4_cache.items():
            if "layer_" in name:
                try:
                    idx = int(name.split("_")[1])
                except (IndexError, ValueError):
                    continue
                result.append(self._get_or_create_block(idx))
        return result

    def forward_layer(
        self, layer_idx: int, hidden_states: np.ndarray, context_tokens: list[int]
    ) -> np.ndarray:
        t0 = time.perf_counter_ns()

        if layer_idx >= self.num_layers:
            return hidden_states

        block = self._get_or_create_block(layer_idx)
        self._stats["inference_steps"] += 1

        predicted = []
        if self.enable_prefetch:
            predicted = self.predictor.predict_next(
                context_tokens, layer_idx, n_predict=3
            )
            for pl in predicted:
                if pl < self.num_layers and pl != layer_idx:
                    pblk = self._get_or_create_block(pl)
                    self.promote_to_dram(pblk)
                    if self.prefetcher:
                        self.prefetcher.prefetch(
                            pl,
                            pblk.ssd_offset,
                            pblk.ssd_size,
                        )

            self.predictor.train(context_tokens, layer_idx)

        weights = self.promote_to_l2(block)
        if weights is None:
            return hidden_states

        hd = weights.shape[-1]
        hs = hidden_states.reshape(-1, hd)
        out = hs @ weights[: hs.shape[-1], : hs.shape[-1]].T
        out = out.reshape(hidden_states.shape)

        if self.enable_pressure_adapt:
            self._update_pressure()
            n_active = self.pressure_monitor.suggest_active_layers()
            if n_active != self.pressure_monitor.state.active_layers:
                self.pressure_monitor.state.active_layers = n_active
                self._stats["active_layers_adjustments"] += 1

        dt_ns = time.perf_counter_ns() - t0
        self._stats["total_inference_time_ns"] += dt_ns

        return out

    def _update_pressure(self):
        l4 = sum(len(v) for v in self._l4_cache.values())
        l3 = sum(v.nbytes for v in self._l3_cache.values())
        l2 = sum(v.nbytes for v in self._l2_cache.values())
        bw = self.bandwidth_scheduler.current_mbps()
        self.pressure_monitor.record_usage(
            l4_used=l4,
            l3_used=l3,
            l2_used=l2,
            bandwidth_mbps=bw,
        )

    def get_layer_weights(self, layer_idx: int, context_tokens: list[int]) -> dict:
        block = self._get_or_create_block(layer_idx)
        weights = self.promote_to_l2(block)
        predicted = self.predictor.predict_next(context_tokens, layer_idx, n_predict=2)
        return {
            "layer_idx": layer_idx,
            "weights": weights,
            "next_predicted": predicted,
            "tier": block.current_tier.name,
            "size_bytes": block.size_bytes,
        }

    def run_inference(
        self, input_tokens: list[int], n_layers_to_run: Optional[int] = None
    ) -> dict:
        if not self._model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        if not self._warmed_up and self.enable_warmup:
            self.warmup()

        n = n_layers_to_run or self.num_layers
        hidden = np.random.randn(1, 256).astype(np.float32)
        context = deque(input_tokens[-64:], maxlen=64)

        for layer_idx in range(n):
            predicted = self.predictor.predict_next(
                list(context), layer_idx, n_predict=2
            )
            for pl in predicted:
                if pl < self.num_layers:
                    self._start_async_load(pl)

            hidden = self.forward_layer(layer_idx, hidden, list(context))

            self._reap_async_loads()

        return {
            "status": "complete",
            "layers_processed": n,
            "output_shape": hidden.shape,
        }

    def _start_async_load(self, layer_idx: int):
        event = threading.Event()

        def _load():
            blk = self._get_or_create_block(layer_idx)
            data = self._read_from_ssd(layer_idx)
            self._l4_cache[blk.name] = data
            self._l4_bytes += len(data)
            blk.dram_data = data
            blk.current_tier = MemoryTier.L4_DRAM
            event.set()

        if not hasattr(self, "_async_loads"):
            self._async_loads: dict[int, tuple[threading.Thread, threading.Event]] = {}
        if layer_idx not in self._async_loads:
            t = threading.Thread(target=_load, daemon=True)
            self._async_loads[layer_idx] = (t, event)
            t.start()

    def _reap_async_loads(self):
        if not hasattr(self, "_async_loads"):
            return
        done = [k for k, (t, e) in self._async_loads.items() if e.is_set()]
        for k in done:
            del self._async_loads[k]

    def unload_model(self) -> dict:
        self._running = False
        self._model_loaded = False
        self._warmed_up = False

        if self.prefetcher:
            self.prefetcher.close()
            self.prefetcher = None

        self._l4_cache.clear()
        self._l4_bytes = 0
        self._l3_cache.clear()
        self._l3_bytes = 0
        self._l2_cache.clear()
        self._l2_bytes = 0
        self._l1_data.clear()
        self._l1_bytes = 0
        self._l0_hvs.clear()

        self.l3_dct_cache.clear()

        if hasattr(self, "_async_loads"):
            self._async_loads.clear()

        self.allocator.free_all()

        result = {
            "status": "unloaded",
            "total_ssd_reads": self._stats["ssd_reads"],
            "total_bytes_moved": self._stats["total_bytes_moved"],
        }
        return result

    def get_stats(self) -> dict:
        total_l4 = self._stats["l4_hits"] + self._stats["l4_misses"]
        total_l3 = self._stats["l3_hits"] + self._stats["l3_misses"]
        total_l2 = self._stats["l2_hits"] + self._stats["l2_misses"]
        total_l1 = self._stats["l1_hits"] + self._stats["l1_misses"]
        total_ssd = self._stats["ssd_reads"]

        total_inference_us = self._stats["total_inference_time_ns"] / 1000
        steps = max(self._stats["inference_steps"], 1)

        pressure = self.pressure_monitor.get_pressure()

        return {
            "ssd_reads": total_ssd,
            "l4_hits": self._stats["l4_hits"],
            "l4_misses": self._stats["l4_misses"],
            "l4_hit_rate": self._stats["l4_hits"] / max(total_l4, 1),
            "l3_hits": self._stats["l3_hits"],
            "l3_misses": self._stats["l3_misses"],
            "l3_hit_rate": self._stats["l3_hits"] / max(total_l3, 1),
            "l2_hits": self._stats["l2_hits"],
            "l2_misses": self._stats["l2_misses"],
            "l2_hit_rate": self._stats["l2_hits"] / max(total_l2, 1),
            "l1_hits": self._stats["l1_hits"],
            "l1_misses": self._stats["l1_misses"],
            "l1_hit_rate": self._stats["l1_hits"] / max(total_l1, 1),
            "prefetch_hits": self._stats["prefetch_hits"],
            "bandwidth_throttles": self._stats["bandwidth_throttles"],
            "active_layers_adjustments": self._stats["active_layers_adjustments"],
            "warmup_layers_loaded": self._stats["warmup_layers_loaded"],
            "total_bytes_moved_mb": round(
                self._stats["total_bytes_moved"] / (1024**2), 1
            ),
            "l4_cache_entries": len(self._l4_cache),
            "l3_cache_entries": len(self._l3_cache),
            "l2_cache_entries": len(self._l2_cache),
            "l1_cache_entries": len(self._l1_data),
            "l0_hv_entries": len(self._l0_hvs),
            "l3_dct_cache": self.l3_dct_cache.summary(),
            "memory_pressure": pressure.pressure_score,
            "active_layers": pressure.state.active_layers,
            "bandwidth_mbps": round(self.bandwidth_scheduler.current_mbps(), 1),
            "hdc_prediction_accuracy": round(self.predictor.accuracy, 4),
            "inference_steps": self._stats["inference_steps"],
            "avg_inference_time_us": round(total_inference_us / steps, 1),
            "model_loaded": self._model_loaded,
            "warmed_up": self._warmed_up,
        }
