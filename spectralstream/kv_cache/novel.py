from __future__ import annotations

import warnings as _warnings

_warnings.warn(
    "spectralstream.kv_cache.novel is deprecated. "
    "Use spectralstream.kv_cache.KVCacheManager instead.",
    DeprecationWarning,
    stacklevel=2,
)

import time
from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from spectralstream.core.math_primitives import (
    dct,
    idct,
    fwht,
    ifwht,
    LloydMaxQuantizer,
    HadamardRotator,
    DCTRotator,
    spectral_entropy,
    landau_zener_coherence,
    cosine_similarity,
    next_power_of_two,
    unit_vector,
    softmax,
)
from spectralstream.kv_cache.core import (
    EPS,
    KVCacheConfig,
    KVCacheEntry,
    QualityMetrics,
)


def _now() -> float:
    return time.monotonic()


def _entropy(probs: np.ndarray, eps: float = 1e-10) -> float:
    p = np.asarray(probs, dtype=np.float64).ravel()
    p = p / (p.sum() + eps)
    return float(-np.sum(p * np.log2(p + eps)))


def _deterministic_hash(s: str) -> int:
    h = 0x811C9DC5
    for c in s.encode():
        h = ((h * 0x01000193) ^ c) & 0xFFFFFFFF
    return h


class SpectralResonanceCache:
    def __init__(
        self, dim: int = 128, n_frequency_bins: int = 16, n_coefficients: int = 8
    ):
        self.dim = dim
        self.n_frequency_bins = n_frequency_bins
        self.n_coefficients = n_coefficients

        self._resonance_bins: dict[int, list[int]] = {
            i: [] for i in range(n_frequency_bins)
        }
        self._binned_keys: dict[int, dict] = {}
        self._key_store: dict[int, np.ndarray] = {}
        self._value_store: dict[int, np.ndarray] = {}
        self._resonant_freqs: dict[int, int] = {}
        self._step = 0
        self.hits = 0
        self.misses = 0

    def _compute_resonance(self, vec: np.ndarray) -> int:
        n = vec.shape[-1]
        x = vec.astype(np.float64)
        dct_vals = np.zeros(n, dtype=np.float64)
        for i in range(self.n_coefficients):
            dct_vals[i] = np.sum(x * np.cos(np.pi * (np.arange(n) + 0.5) * i / n))
        dct_vals *= np.sqrt(2.0 / n)
        power = dct_vals**2
        dominant = int(np.argmax(power[: self.n_coefficients]))
        bin_size = max(1, self.n_coefficients // self.n_frequency_bins)
        return min(dominant // bin_size, self.n_frequency_bins - 1)

    def store(self, key: np.ndarray, value: np.ndarray, position: int):
        freq_bin = self._compute_resonance(key)
        self._key_store[position] = key.astype(np.float32)
        self._value_store[position] = value.astype(np.float32)
        self._resonant_freqs[position] = freq_bin
        self._resonance_bins[freq_bin].append(position)
        self._binned_keys[freq_bin] = self._binned_keys.get(freq_bin, {})
        self._binned_keys[freq_bin][position] = key.astype(np.float32)
        self._step += 1

    def retrieve(self, position: int) -> Optional[tuple[np.ndarray, np.ndarray]]:
        if position in self._key_store:
            self.hits += 1
            return (self._key_store[position], self._value_store[position])
        self.misses += 1
        return None

    def query(
        self, query_vector: np.ndarray, top_k: int = 10
    ) -> list[tuple[int, float]]:
        q_freq = self._compute_resonance(query_vector)
        candidates = []
        for delta in range(self.n_frequency_bins):
            bin_idx = (q_freq + delta) % self.n_frequency_bins
            for pos in self._resonance_bins.get(bin_idx, []):
                candidates.append(pos)
            for bin_idx2 in range(
                max(0, q_freq - delta), min(self.n_frequency_bins, q_freq + delta + 1)
            ):
                pass
        if not candidates:
            for pos in self._key_store:
                candidates.append(pos)
        q = query_vector.ravel()
        q_norm = np.linalg.norm(q) + 1e-10
        results = []
        for pos in set(candidates):
            k = self._key_store.get(pos)
            if k is None:
                continue
            sim = float(np.dot(q, k.ravel())) / (
                q_norm * np.linalg.norm(k.ravel()) + 1e-10
            )
            freq_match = (
                1.0
                - abs(self._resonant_freqs.get(pos, 0) - q_freq) / self.n_frequency_bins
            )
            results.append((pos, sim * 0.7 + freq_match * 0.3))
        results.sort(key=lambda x: -x[1])
        return results[:top_k]

    def evict(self, position: int):
        self._key_store.pop(position, None)
        self._value_store.pop(position, None)
        freq_bin = self._resonant_freqs.pop(position, None)
        if freq_bin is not None and position in self._resonance_bins.get(freq_bin, []):
            self._resonance_bins[freq_bin].remove(position)
        if freq_bin is not None and freq_bin in self._binned_keys:
            self._binned_keys[freq_bin].pop(position, None)

    def clear(self):
        self._key_store.clear()
        self._value_store.clear()
        self._resonant_freqs.clear()
        self._resonance_bins.clear()
        self._binned_keys.clear()
        self.hits = 0
        self.misses = 0
        self._step = 0

    def get_stats(self) -> dict:
        return self.cache_summary()

    def cache_summary(self) -> dict:
        return {
            "type": "SpectralResonanceCache",
            "dim": self.dim,
            "n_frequency_bins": self.n_frequency_bins,
            "n_coefficients": self.n_coefficients,
            "num_positions": len(self._key_store),
            "hits": self.hits,
            "misses": self.misses,
        }


class HolographicTimeCrystal:
    def __init__(
        self,
        dim: int = 128,
        max_size: int = 4096,
        pump_period: float = 10.0,
        pump_strength: float = 0.1,
        use_fft_drive: bool = True,
    ):
        self.dim = dim
        self.max_size = max_size
        self.pump_period = pump_period
        self.pump_strength = pump_strength
        self.use_fft_drive = use_fft_drive

        self._entries: dict[int, dict] = {}
        self._positions: list[int] = []
        self._phase: dict[int, float] = {}
        self._t: float = 0.0
        self._last_pump: float = _now()

        self.hits = 0
        self.misses = 0
        self.pump_count = 0

    def store(self, key: np.ndarray, value: np.ndarray, position: int):
        phase = 2.0 * np.pi * np.random.random()
        self._entries[position] = {
            "k": key.astype(np.float32),
            "v": value.astype(np.float32),
            "coherence": 1.0,
            "stored_at": _now(),
        }
        self._phase[position] = phase
        if position not in self._positions:
            self._positions.append(position)
        if len(self._positions) > self.max_size:
            oldest = min(self._positions, key=lambda p: self._entries[p]["stored_at"])
            self._entries.pop(oldest, None)
            self._phase.pop(oldest, None)
            self._positions.remove(oldest)

    def retrieve(self, position: int) -> Optional[tuple[np.ndarray, np.ndarray]]:
        entry = self._entries.get(position)
        if entry is None:
            self.misses += 1
            return None
        self.hits += 1
        self._pump_if_needed()
        coherence = entry.get("coherence", 1.0)
        k = entry["k"] * coherence
        v = entry["v"] * coherence
        return (k, v)

    def _pump_if_needed(self):
        now = _now()
        if now - self._last_pump < self.pump_period:
            return
        if not self._entries:
            return
        self._last_pump = now
        self._t += 1.0

        n = len(self._entries)
        indices = list(self._entries.keys())
        phases = np.array([self._phase.get(p, 0.0) for p in indices])

        if self.use_fft_drive:
            drive = np.abs(np.fft.fft(np.cos(phases)))[0]
            pump = self.pump_strength * (1.0 + 0.5 * float(drive) / max(n, 1))
        else:
            pump = self.pump_strength

        for i, pos in enumerate(indices):
            entry = self._entries[pos]
            phase = self._phase.get(pos, 0.0)
            pump_factor = pump * (0.5 + 0.5 * np.cos(phase + self._t * 0.1))
            entry["coherence"] = min(1.0, entry.get("coherence", 1.0) + pump_factor)
            self._phase[pos] = phase + 0.1 + pump_factor * 0.5

        self.pump_count += 1

    def query(self, query_vec: np.ndarray, top_k: int = 10) -> list[tuple[int, float]]:
        self._pump_if_needed()
        q = query_vec.ravel()
        q_norm = np.linalg.norm(q) + 1e-10
        results = []
        for pos, entry in self._entries.items():
            k = entry["k"].ravel()
            sim = float(np.dot(q, k)) / (q_norm * np.linalg.norm(k) + 1e-10)
            coherence = entry.get("coherence", 1.0)
            results.append((pos, sim * coherence))
        results.sort(key=lambda x: -x[1])
        return results[:top_k]

    def evict(self, position: int):
        self._entries.pop(position, None)
        self._phase.pop(position, None)
        if position in self._positions:
            self._positions.remove(position)

    def clear(self):
        self._entries.clear()
        self._positions.clear()
        self._phase.clear()
        self._t = 0.0
        self._last_pump = _now()
        self.hits = 0
        self.misses = 0
        self.pump_count = 0

    def time_crystal_entropy(self) -> float:
        if not self._phase:
            return 0.0
        phases = np.array(list(self._phase.values()))
        return _entropy(np.cos(phases) + 1.0)

    def get_stats(self) -> dict:
        return self.cache_summary()

    def cache_summary(self) -> dict:
        return {
            "type": "HolographicTimeCrystal",
            "dim": self.dim,
            "num_positions": len(self._entries),
            "max_size": self.max_size,
            "pump_count": self.pump_count,
            "hits": self.hits,
            "misses": self.misses,
            "time_crystal_entropy": self.time_crystal_entropy(),
        }


class FractalCache:
    def __init__(self, dim: int = 128, n_scales: int = 3, max_positions: int = 4096):
        self.dim = dim
        self.n_scales = n_scales
        self.max_positions = max_positions

        self._coarse_store: dict[int, list[np.ndarray]] = {}
        self._detail_store: dict[int, list[np.ndarray]] = {}
        self._value_store: dict[int, np.ndarray] = {}
        self._scale_importance: dict[int, float] = {}
        self._positions: list[int] = []
        self._step = 0
        self.hits = 0
        self.misses = 0

    def _decompose(self, vec: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        n = vec.shape[-1]
        x = vec.astype(np.float64).copy()
        coarse = []
        details = []
        for scale in range(self.n_scales):
            half = max(1, n // (2 ** (scale + 1)))
            coarse_approx = np.zeros(n, dtype=np.float64)
            detail = np.zeros(n, dtype=np.float64)
            for i in range(half):
                avg = (x[2 * i] + x[2 * i + 1]) / 2.0 if 2 * i + 1 < n else x[2 * i]
                diff = (x[2 * i] - x[2 * i + 1]) / 2.0 if 2 * i + 1 < n else 0.0
                coarse_approx[2 * i : 2 * i + 2] = avg
                if 2 * i + 1 < n:
                    detail[2 * i] = diff
                    detail[2 * i + 1] = -diff
            coarse.append(coarse_approx.astype(np.float32))
            details.append(detail.astype(np.float32))
            x = coarse_approx
        return coarse, details

    def _reconstruct(
        self, coarse: list[np.ndarray], details: list[np.ndarray], up_to_scale: int
    ) -> np.ndarray:
        result = coarse[-1].copy()
        for s in range(min(up_to_scale, self.n_scales - 1), -1, -1):
            n = len(result)
            if s < len(coarse) - 1:
                result = coarse[s].copy()
            if s < len(details):
                result = result + details[s] * 0.5
        return result

    def store(self, key: np.ndarray, value: np.ndarray, position: int):
        coarse, details = self._decompose(key)
        self._coarse_store[position] = coarse
        self._detail_store[position] = details
        self._value_store[position] = value.astype(np.float32)
        self._scale_importance[position] = 1.0
        if position not in self._positions:
            self._positions.append(position)
        if len(self._positions) > self.max_positions:
            oldest = self._positions.pop(0)
            self._coarse_store.pop(oldest, None)
            self._detail_store.pop(oldest, None)
            self._value_store.pop(oldest, None)
            self._scale_importance.pop(oldest, None)
        self._step += 1

    def retrieve(
        self, position: int, scale: Optional[int] = None
    ) -> Optional[tuple[np.ndarray, np.ndarray]]:
        if position not in self._value_store:
            self.misses += 1
            return None
        self.hits += 1
        coarse = self._coarse_store.get(position)
        details = self._detail_store.get(position)
        v = self._value_store[position]
        if coarse is None or details is None:
            return (np.zeros(self.dim), v)
        s = scale if scale is not None else self.n_scales - 1
        k = self._reconstruct(coarse, details, s)
        return (k, v)

    def query(
        self, query_vector: np.ndarray, top_k: int = 10
    ) -> list[tuple[int, float]]:
        q = query_vector.ravel()
        q_norm = np.linalg.norm(q) + 1e-10
        results = []
        for pos in self._positions:
            coarse = self._coarse_store.get(pos)
            if coarse is None:
                continue
            k_coarse = coarse[-1]
            sim = float(np.dot(q, k_coarse.ravel())) / (
                q_norm * np.linalg.norm(k_coarse.ravel()) + 1e-10
            )
            results.append((pos, sim))
        results.sort(key=lambda x: -x[1])
        return results[:top_k]

    def evict(self, position: int):
        self._coarse_store.pop(position, None)
        self._detail_store.pop(position, None)
        self._value_store.pop(position, None)
        self._scale_importance.pop(position, None)
        if position in self._positions:
            self._positions.remove(position)

    def clear(self):
        self._coarse_store.clear()
        self._detail_store.clear()
        self._value_store.clear()
        self._scale_importance.clear()
        self._positions.clear()
        self.hits = 0
        self.misses = 0
        self._step = 0

    def fractal_dimension(self) -> float:
        if len(self._positions) < 2:
            return 0.0
        sizes = []
        for pos in self._positions:
            details = self._detail_store.get(pos)
            if details:
                total_energy = sum(float(np.linalg.norm(d)) for d in details)
                sizes.append(total_energy)
        if not sizes:
            return 0.0
        return float(np.log(np.mean(sizes) + 1e-10) / np.log(self.n_scales + 1))

    def get_stats(self) -> dict:
        return self.cache_summary()

    def cache_summary(self) -> dict:
        return {
            "type": "FractalCache",
            "dim": self.dim,
            "n_scales": self.n_scales,
            "num_positions": len(self._positions),
            "max_positions": self.max_positions,
            "fractal_dimension": self.fractal_dimension(),
            "hits": self.hits,
            "misses": self.misses,
        }


class AttentionWeightedEviction:
    def __init__(
        self,
        max_size: int = 4096,
        window_size: int = 64,
        heavy_hitter_ratio: float = 0.3,
        entropy_weight: float = 0.2,
        coherence_weight: float = 0.2,
        recency_weight: float = 0.3,
        attention_weight: float = 0.3,
        half_life: float = 1000.0,
    ):
        self.max_size = max_size
        self.window_size = window_size
        self.heavy_hitter_ratio = heavy_hitter_ratio
        self.entropy_weight = entropy_weight
        self.coherence_weight = coherence_weight
        self.recency_weight = recency_weight
        self.attention_weight = attention_weight
        self.half_life = half_life

        self._cumulative_attention: dict[int, float] = {}
        self._entropy_scores: dict[int, float] = {}
        self._coherence_scores: dict[int, float] = {}
        self._timestamps: dict[int, float] = {}
        self._positions: list[int] = []
        self._step = 0

    def record_attention(self, position: int, attention_score: float):
        self._cumulative_attention[position] = (
            self._cumulative_attention.get(position, 0.0) + attention_score
        )
        self._timestamps[position] = _now()

    def record_entropy(self, position: int, entropy: float):
        self._entropy_scores[position] = entropy

    def get_heavy_hitters(self, k: Optional[int] = None) -> set[int]:
        if k is None:
            k = max(1, int(self.max_size * self.heavy_hitter_ratio))
        if not self._cumulative_attention:
            return set()
        positions = list(self._cumulative_attention.keys())
        scores = [self._cumulative_attention.get(p, 0.0) for p in positions]
        if len(scores) <= k:
            return set(positions)
        idx = np.argpartition(scores, -k)[-k:]
        return {positions[i] for i in idx}

    def get_window(self) -> set[int]:
        return set(self._positions[-self.window_size :])

    def eviction_score(self, position: int) -> float:
        now = _now()
        timestamp = self._timestamps.get(position, 0.0)
        age = now - timestamp

        coherence = float(np.exp(-age / self.half_life))
        recency = float(np.exp(-age / max(self.half_life, 1))) if age > 0 else 1.0
        entropy = self._entropy_scores.get(position, 0.5)
        attn = self._cumulative_attention.get(position, 0.0)
        attn_norm = min(
            1.0, attn / (max(list(self._cumulative_attention.values()) + [1.0]))
        )

        heavy_hitter = 1.0 if position in self.get_heavy_hitters() else 0.0
        in_window = 1.0 if position in self.get_window() else 0.0

        score = (
            self.coherence_weight * coherence
            + self.recency_weight * recency
            + self.entropy_weight * (1.0 - entropy)
            + self.attention_weight * attn_norm
            + 0.5 * heavy_hitter
            + 0.5 * in_window
        )
        return score

    def should_keep(self, position: int) -> bool:
        if position in self.get_window():
            return True
        if position in self.get_heavy_hitters():
            return True
        return self.eviction_score(position) > 0.4

    def evict(self, positions: list[int], n_to_evict: int) -> list[int]:
        if n_to_evict <= 0:
            return []
        protected = self.get_window() | self.get_heavy_hitters()
        candidates = [p for p in positions if p not in protected]
        if not candidates:
            candidates = positions[-self.window_size :]
        if len(candidates) <= n_to_evict:
            return candidates
        scores = [(p, self.eviction_score(p)) for p in candidates]
        scores.sort(key=lambda x: x[1])
        return [p for p, _ in scores[:n_to_evict]]

    def update_positions(self, positions: list[int]):
        self._positions = positions
        self._step += 1

    def clear(self):
        self._cumulative_attention.clear()
        self._entropy_scores.clear()
        self._coherence_scores.clear()
        self._timestamps.clear()
        self._positions.clear()
        self._step = 0

    def get_stats(self) -> dict:
        return self.cache_summary()

    def cache_summary(self) -> dict:
        return {
            "type": "AttentionWeightedEviction",
            "max_size": self.max_size,
            "window_size": self.window_size,
            "heavy_hitter_ratio": self.heavy_hitter_ratio,
            "n_heavy_hitters": len(self.get_heavy_hitters()),
            "n_tracked": len(self._cumulative_attention),
        }


class SharedPrefixManager:
    def __init__(
        self,
        kv_cache: Any,
        max_prefixes: int = 1024,
        default_ttl: float = 300.0,
        memory_budget_mb: float = 512.0,
    ):
        self.kv_cache = kv_cache
        self.max_prefixes = max_prefixes
        self.default_ttl = default_ttl
        self.memory_budget = memory_budget_mb * 1024 * 1024

        self._prefixes: dict[str, dict] = {}
        self._tenant_prefixes: dict[str, set[str]] = defaultdict(set)
        self._lru: deque[str] = deque(maxlen=max_prefixes)
        self._hash_to_prefix: dict[int, str] = {}
        self.hits = 0
        self.misses = 0
        self.evict_count = 0
        self._current_memory = 0.0

    def _reduce_memory(self):
        while self._current_memory > self.memory_budget and self._lru:
            self._evict_lru()

    def _evict_lru(self):
        if not self._lru:
            return
        prefix_hash = self._lru.popleft()
        if prefix_hash in self._prefixes:
            info = self._prefixes[prefix_hash]
            for lp in info.get("pages", []):
                self.kv_cache.free_page(lp)
            self._current_memory -= info.get("memory_bytes", 0)
            for tenant in info.get("tenants", []):
                self._tenant_prefixes[tenant].discard(prefix_hash)
            del self._prefixes[prefix_hash]
            self.evict_count += 1

    def register_prefix(
        self,
        token_ids: list[int],
        tenant_id: str = "default",
        ttl: Optional[float] = None,
    ) -> str:
        prefix_hash = str(_deterministic_hash("_".join(str(t) for t in token_ids[:64])))
        if prefix_hash in self._prefixes:
            info = self._prefixes[prefix_hash]
            info["tenants"].add(tenant_id)
            self._tenant_prefixes[tenant_id].add(prefix_hash)
            info["access_time"] = _now()
            info["access_count"] += 1
            self.hits += 1
            return prefix_hash

        pages = self.kv_cache.alloc_pages(len(token_ids))
        mem_bytes = (
            len(pages) * self.kv_cache.tokens_per_page * self.kv_cache.dim * 4 * 2
        )
        self._current_memory += mem_bytes

        info = {
            "token_ids": list(token_ids),
            "pages": pages,
            "tenant_id": tenant_id,
            "tenants": {tenant_id},
            "created": _now(),
            "access_time": _now(),
            "access_count": 1,
            "ttl": ttl if ttl is not None else self.default_ttl,
            "memory_bytes": mem_bytes,
        }
        self._prefixes[prefix_hash] = info
        self._tenant_prefixes[tenant_id].add(prefix_hash)
        self._lru.append(prefix_hash)
        self._hash_to_prefix[_deterministic_hash(prefix_hash)] = prefix_hash
        self._reduce_memory()
        self.misses += 1
        return prefix_hash

    def lookup(self, prefix_hash: str) -> Optional[dict]:
        info = self._prefixes.get(prefix_hash)
        if info is None:
            self.misses += 1
            return None
        age = _now() - info["access_time"]
        if age > info["ttl"]:
            self._expire(prefix_hash)
            self.misses += 1
            return None
        info["access_time"] = _now()
        info["access_count"] += 1
        if prefix_hash in self._lru:
            self._lru.remove(prefix_hash)
            self._lru.append(prefix_hash)
        self.hits += 1
        return info

    def _expire(self, prefix_hash: str):
        info = self._prefixes.pop(prefix_hash, None)
        if info:
            for lp in info.get("pages", []):
                self.kv_cache.free_page(lp)
            self._current_memory -= info.get("memory_bytes", 0)
            for tenant in info.get("tenants", []):
                self._tenant_prefixes[tenant].discard(prefix_hash)
            if prefix_hash in self._lru:
                self._lru.remove(prefix_hash)

    def expire_tenant(self, tenant_id: str):
        for prefix_hash in list(self._tenant_prefixes.get(tenant_id, set())):
            self._expire(prefix_hash)

    def get_prefix_pages(self, prefix_hash: str) -> list[int]:
        info = self.lookup(prefix_hash)
        if info is None:
            return []
        pages = info.get("pages", [])
        shared = []
        for lp in pages:
            shared.append(self.kv_cache.copy_on_write(lp))
        return shared

    def cleanup_expired(self):
        now = _now()
        for prefix_hash in list(self._prefixes.keys()):
            info = self._prefixes[prefix_hash]
            if now - info["access_time"] > info["ttl"]:
                self._expire(prefix_hash)

    def get_tenant_prefixes(self, tenant_id: str) -> list[dict]:
        return [
            self._prefixes[h]
            for h in self._tenant_prefixes.get(tenant_id, set())
            if h in self._prefixes
        ]

    def clear(self):
        self._prefixes.clear()
        self._tenant_prefixes.clear()
        self._lru.clear()
        self._hash_to_prefix.clear()
        self._current_memory = 0.0
        self.hits = 0
        self.misses = 0
        self.evict_count = 0

    def get_stats(self) -> dict:
        return self.cache_summary()

    def cache_summary(self) -> dict:
        return {
            "type": "SharedPrefixManager",
            "num_prefixes": len(self._prefixes),
            "max_prefixes": self.max_prefixes,
            "num_tenants": len(self._tenant_prefixes),
            "memory_bytes_used": self._current_memory,
            "memory_budget": self.memory_budget,
            "hits": self.hits,
            "misses": self.misses,
            "evict_count": self.evict_count,
            "hit_rate": self.hits / max(self.hits + self.misses, 1),
        }
