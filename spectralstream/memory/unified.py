from __future__ import annotations

import logging
import math
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

import numpy as np

from spectralstream.core.math_primitives import (
    cascade_eviction_score,
    cosine_similarity,
    dct,
    fwht,
    generate_random_complex_vector,
    generate_random_hd_vector,
    hrr_bind,
    hrr_bundle,
    hrr_unbind,
    idct,
    ifwht,
    landau_zener_coherence,
    softmax,
    spectral_entropy,
    unit_vector,
)
from spectralstream.kv_cache.core import EPS
from spectralstream.memory.hrr import HrrMemory, ResonantMemory
from spectralstream.memory.holographic import (
    HolographicCacheHierarchy,
    HolographicKVCache,
    HolographicTier,
    HolographicWeightStore,
)

logger = logging.getLogger(__name__)


class CacheTier(IntEnum):
    L1 = 0
    L2 = 1
    L3 = 2
    L4 = 3


def _ensure_dim(x: np.ndarray, dim: int) -> np.ndarray:
    flat = x.ravel().astype(np.float64)
    if len(flat) < dim:
        flat = np.pad(flat, (0, dim - len(flat)))
    elif len(flat) > dim:
        flat = flat[:dim]
    return flat


def _importance_from_entropy(key: np.ndarray, threshold: float = 0.5) -> float:
    ent = spectral_entropy(key)
    if ent < threshold:
        return 0.5 + 0.5 * (ent / max(threshold, 1e-10))
    return 0.5 + 0.5 * min(1.0, ent)


@dataclass
class SDMLocation:
    address: np.ndarray
    data: np.ndarray
    count: int = 0


@dataclass
class PredictiveEntry:
    position: int
    quantized_error: np.ndarray
    prediction_coeffs: np.ndarray
    actual_norm: float
    error_norm: float
    timestamp: float


@dataclass
class FluxEntry:
    position: int
    key_hv: np.ndarray
    value_hv: np.ndarray
    rho: float
    bootstrap_current: float
    temperature: float
    timestamp: float
    access_count: int = 0


class UnifiedHolographicMemory:
    def __init__(
        self,
        kv_dim: int = 1024,
        weight_dim: int = 2048,
        resonant_dim: int = 1024,
        n_heads: int = 8,
        n_layers: int = 32,
        max_kv_size: int = 4096,
        l1_capacity: int = 128,
        l2_capacity: int = 4096,
    ):
        self.kv_cache = HolographicKVCache(
            dim=kv_dim,
            max_size=max_kv_size,
            n_heads=n_heads,
            n_layers=n_layers,
        )
        self.weight_store = HolographicWeightStore(
            dim=weight_dim,
            n_layers=n_layers,
        )
        self.hierarchy = HolographicCacheHierarchy(
            l1_dim=128,
            l2_dim=kv_dim,
            l3_dim=weight_dim,
            l1_capacity=l1_capacity,
            l2_capacity=l2_capacity,
        )
        self.associative_memory = HrrMemory(dim=kv_dim, decay_rate=0.001)
        self.resonant_memory = ResonantMemory(dim=resonant_dim)
        self._start_time = time.monotonic()
        self._lock = threading.Lock()

    def memory_status(self) -> dict:
        return {
            "engine": "Unified Holographic Memory v2.0",
            "uptime_seconds": time.monotonic() - self._start_time,
            "associative_memory": self.associative_memory.get_stats(),
            "kv_cache": self.kv_cache.get_stats(),
            "weight_store": self.weight_store.get_stats(),
            "hierarchy": self.hierarchy.get_stats(),
            "resonant_memory": self.resonant_memory.get_stats(),
        }

    def memory_reset(self) -> dict:
        self.kv_cache.clear()
        self.weight_store.clear()
        self.hierarchy.clear()
        self.associative_memory.clear()
        self.resonant_memory.clear()
        self._start_time = time.monotonic()
        return {"status": "ok", "message": "All holographic memory reset"}

    def store_kv(
        self,
        key: np.ndarray,
        value: np.ndarray,
        position: int,
        head_idx: int = 0,
        layer_idx: int = 0,
    ) -> None:
        self.kv_cache.store(key, value, position, head_idx, layer_idx)

    def recall_kv(
        self,
        position: int,
        head_idx: int = 0,
        layer_idx: int = 0,
    ) -> Optional[tuple[np.ndarray, np.ndarray]]:
        return self.kv_cache.retrieve(position, head_idx, layer_idx)

    def store_weight(
        self,
        weights: np.ndarray,
        layer_idx: int,
        weight_name: str = "",
        row_idx: Optional[int] = None,
    ) -> int:
        return self.weight_store.store_weight(weights, layer_idx, weight_name, row_idx)

    def recall_weight(
        self,
        layer_idx: int,
        weight_name: str = "",
        row_idx: Optional[int] = None,
        stage: Optional[int] = None,
    ) -> Optional[np.ndarray]:
        return self.weight_store.recall_weight(layer_idx, weight_name, row_idx, stage)

    def progressive_recall_weight(
        self,
        layer_idx: int,
        weight_name: str = "",
        row_idx: Optional[int] = None,
    ) -> list[np.ndarray]:
        return self.weight_store.progressive_recall(layer_idx, weight_name, row_idx)

    def associate(
        self,
        key: np.ndarray,
        value: Optional[np.ndarray] = None,
    ) -> int:
        return self.associative_memory.store(key, value)

    def recall_asso(
        self, key: np.ndarray, top_k: int = 1
    ) -> list[tuple[int, np.ndarray, float]]:
        return self.associative_memory.recall(key, top_k=top_k)

    def store_resonant(self, vector: np.ndarray) -> int:
        return self.resonant_memory.store(vector)

    def recall_resonant(
        self, query: np.ndarray, top_k: int = 5
    ) -> list[tuple[int, np.ndarray, float]]:
        return self.resonant_memory.retrieve(query, top_k=top_k)

    def store_tiered(
        self, key: str, data: np.ndarray, tier: int = CacheTier.L2
    ) -> None:
        self.hierarchy.store(key, data, tier)

    def recall_tiered(self, key: str) -> Optional[np.ndarray]:
        return self.hierarchy.retrieve(key)

    def apply_decay(self) -> None:
        with self._lock:
            self.kv_cache.apply_decay()
            self.associative_memory.apply_decay()
            self.resonant_memory.apply_decay()
            self.hierarchy.apply_decay_all()

    def clear(self) -> dict:
        return self.memory_reset()

    def get_stats(self) -> dict:
        return self.memory_status()


class KanervaSDM:
    def __init__(
        self,
        dim: int = 1024,
        n_locations: int = 4096,
        read_radius: int = 200,
        write_radius: int = 250,
        max_count: int = 100,
        seed: int = 42,
    ):
        if dim < 2:
            raise ValueError(f"dim must be >= 2, got {dim}")
        if n_locations < 1:
            raise ValueError(f"n_locations must be >= 1, got {n_locations}")

        self.dim = dim
        self.n_locations = n_locations
        self.read_radius = read_radius
        self.write_radius = write_radius
        self.max_count = max_count
        self._rng = np.random.RandomState(seed)

        self._addresses = self._rng.choice([-1, 1], size=(n_locations, dim)).astype(
            np.float64
        )
        self._data = np.zeros((n_locations, dim), dtype=np.float64)
        self._counts = np.zeros(n_locations, dtype=np.int32)
        self._lock = threading.Lock()

    def _hamming_distances(self, query: np.ndarray) -> np.ndarray:
        q_bin = np.sign(query)
        return ((self._addresses != q_bin).sum(axis=1)).astype(np.float64)

    def _hard_locate(self, query: np.ndarray) -> np.ndarray:
        dists = self._hamming_distances(query)
        return np.where(dists <= self.write_radius)[0]

    def _read_locate(self, query: np.ndarray) -> np.ndarray:
        dists = self._hamming_distances(query)
        return np.where(dists <= self.read_radius)[0]

    def store(self, key: np.ndarray, value: np.ndarray) -> int:
        k = unit_vector(key.ravel().astype(np.float64))
        v = unit_vector(value.ravel().astype(np.float64))

        with self._lock:
            indices = self._hard_locate(k)
            if len(indices) == 0:
                nearest = int(np.argmin(self._hamming_distances(k)))
                indices = np.array([nearest])

            for idx in indices:
                i = int(idx)
                if self._counts[i] < self.max_count:
                    self._data[i] += v
                    self._counts[i] += 1

            return len(indices)

    def retrieve(
        self, query: np.ndarray, top_k: int = 1
    ) -> list[tuple[int, np.ndarray, float]]:
        q = unit_vector(query.ravel().astype(np.float64))

        with self._lock:
            indices = self._read_locate(q)
            if len(indices) == 0:
                nearest = int(np.argmin(self._hamming_distances(q)))
                indices = np.array([nearest])

            results: list[tuple[int, float]] = []
            for idx in indices:
                i = int(idx)
                cnt = int(self._counts[i])
                if cnt > 0:
                    data_mean = self._data[i] / cnt
                    sim = float(np.dot(q, data_mean))
                    results.append((i, sim))

            results.sort(key=lambda x: -x[1])

            out: list[tuple[int, np.ndarray, float]] = []
            for loc_id, sim in results[:top_k]:
                i = loc_id
                cnt = int(self._counts[i])
                data_mean = self._data[i] / max(cnt, 1)
                out.append((loc_id, data_mean, sim))
            return out

    def retrieve_raw(self, query: np.ndarray) -> np.ndarray:
        q = unit_vector(query.ravel().astype(np.float64))

        with self._lock:
            indices = self._read_locate(q)
            if len(indices) == 0:
                return np.zeros(self.dim, dtype=np.float64)

            weighted_sum = np.zeros(self.dim, dtype=np.float64)
            total_weight = 0.0
            for idx in indices:
                i = int(idx)
                cnt = int(self._counts[i])
                if cnt > 0:
                    weight = float(cnt) / self.max_count
                    weighted_sum += self._data[i]
                    total_weight += weight

            if total_weight > 1e-10:
                return unit_vector(weighted_sum)
            return np.zeros(self.dim, dtype=np.float64)

    def capacity(self) -> int:
        return max(1, int(self.n_locations * self.dim / max(1, math.log2(self.dim))))

    def load_factor(self) -> float:
        return float(np.sum(self._counts > 0)) / self.n_locations

    def clear(self) -> None:
        with self._lock:
            self._data.fill(0.0)
            self._counts.fill(0)

    def get_stats(self) -> dict:
        return {
            "dim": self.dim,
            "n_locations": self.n_locations,
            "read_radius": self.read_radius,
            "write_radius": self.write_radius,
            "locations_used": int(np.sum(self._counts > 0)),
            "load_factor": self.load_factor(),
            "capacity": self.capacity(),
            "avg_count": float(np.mean(self._counts))
            if self._counts.sum() > 0
            else 0.0,
        }


class HolographicCHN:
    def __init__(
        self,
        dim: int = 1024,
        temperature: float = 1.0,
        max_iterations: int = 10,
        convergence_threshold: float = 1e-4,
    ):
        if dim < 2:
            raise ValueError(f"dim must be >= 2, got {dim}")
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")

        self.dim = dim
        self.temperature = temperature
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold

        self.memory: np.ndarray = np.zeros(dim, dtype=np.float64)
        self._patterns: dict[int, np.ndarray] = {}
        self._values: dict[int, np.ndarray] = {}
        self._next_id: int = 0
        self._lock = threading.Lock()

    def _phase_score(self, xi: np.ndarray, pattern: np.ndarray) -> float:
        xi_fft = np.fft.fft(xi.astype(np.complex128))
        p_fft = np.fft.fft(pattern.astype(np.complex128))
        cross = np.sum(xi_fft * np.conj(p_fft))
        mag = np.sqrt(np.sum(np.abs(xi_fft) ** 2) * np.sum(np.abs(p_fft) ** 2))
        if mag < 1e-10:
            return 0.0
        return float(cross.real / mag)

    def store(self, key: np.ndarray, value: np.ndarray) -> int:
        k = unit_vector(key.ravel().astype(np.float64))
        v = unit_vector(value.ravel().astype(np.float64))
        encoded = hrr_bind(k, v)

        with self._lock:
            sid = self._next_id
            self._next_id += 1
            self._patterns[sid] = k
            self._values[sid] = v
            self.memory = hrr_bundle(self.memory, encoded)
            self.memory = unit_vector(self.memory)
            return sid

    def retrieve(
        self,
        query: np.ndarray,
        top_k: int = 1,
    ) -> list[tuple[int, np.ndarray, float]]:
        q = unit_vector(query.ravel().astype(np.float64))
        xi = q.copy()

        with self._lock:
            if len(self._patterns) == 0:
                return []

            pattern_list = list(self._patterns.values())
            id_list = list(self._patterns.keys())
            n_patterns = len(pattern_list)
            pattern_matrix = np.stack(pattern_list, axis=0)

            for _iter in range(self.max_iterations):
                scores = np.zeros(n_patterns, dtype=np.float64)
                for i in range(n_patterns):
                    scores[i] = self._phase_score(xi, pattern_matrix[i])

                weights = softmax(scores * self.temperature)

                new_xi = np.zeros(self.dim, dtype=np.float64)
                for i in range(n_patterns):
                    new_xi += weights[i] * hrr_unbind(self.memory, pattern_matrix[i])

                new_xi = unit_vector(new_xi)

                diff = np.linalg.norm(new_xi - xi)
                xi = new_xi

                if diff < self.convergence_threshold:
                    break

            dot_scores = np.zeros(n_patterns, dtype=np.float64)
            for i in range(n_patterns):
                dot_scores[i] = float(np.dot(xi, pattern_matrix[i]))

            top_indices = np.argsort(-dot_scores)[:top_k]
            results: list[tuple[int, np.ndarray, float]] = []
            for idx in top_indices:
                sid = id_list[idx]
                value_est = hrr_unbind(self.memory, self._patterns[sid])
                results.append((sid, value_est, float(dot_scores[idx])))
            return results

    def clear(self) -> None:
        with self._lock:
            self.memory = np.zeros(self.dim, dtype=np.float64)
            self._patterns.clear()
            self._values.clear()
            self._next_id = 0

    def num_items(self) -> int:
        return len(self._patterns)

    def get_stats(self) -> dict:
        return {
            "dim": self.dim,
            "temperature": self.temperature,
            "max_iterations": self.max_iterations,
            "convergence_threshold": self.convergence_threshold,
            "num_items": self.num_items(),
        }


class HDCApproximateSearch:
    def __init__(
        self,
        dim: int = 1024,
        n_tables: int = 8,
        hash_bits: int = 16,
        n_probes: int = 3,
        seed: int = 42,
    ):
        if dim < 2:
            raise ValueError(f"dim must be >= 2, got {dim}")
        if n_tables < 1:
            raise ValueError(f"n_tables must be >= 1, got {n_tables}")
        if hash_bits < 1 or hash_bits > 32:
            raise ValueError(f"hash_bits must be in [1, 32], got {hash_bits}")

        self.dim = dim
        self.n_tables = n_tables
        self.hash_bits = hash_bits
        self.n_probes = n_probes
        self._n_buckets = 1 << hash_bits

        rng = np.random.RandomState(seed)
        self._projections: list[np.ndarray] = [
            rng.randn(hash_bits, dim).astype(np.float64) for _ in range(n_tables)
        ]
        self._bias: list[np.ndarray] = [
            rng.uniform(0, 1, size=hash_bits).astype(np.float64)
            for _ in range(n_tables)
        ]

        self._tables: list[dict[int, list[int]]] = [{} for _ in range(n_tables)]
        self._items: dict[int, np.ndarray] = {}
        self._next_id: int = 0
        self._lock = threading.Lock()

    def _hash(self, key: np.ndarray, table_idx: int) -> int:
        projections = self._projections[table_idx]
        bias = self._bias[table_idx]
        dots = projections @ key + bias
        bits = (dots > 0).astype(np.uint32)
        h = 0
        for b in bits[: self.hash_bits]:
            h = (h << 1) | int(b)
        return h % self._n_buckets

    def _generate_probes(self, query: np.ndarray) -> list[int]:
        primary = self._hash(query, 0)
        buckets = [primary]
        rng = np.random.RandomState(hash(query.tobytes()) & 0xFFFFFFFF)
        for _ in range(self.n_probes - 1):
            noise = rng.randn(self.dim) * 0.1
            perturbed = unit_vector(query + noise)
            alt_bucket = self._hash(perturbed, 0)
            buckets.append(alt_bucket)
        return buckets

    def store(self, key: np.ndarray) -> int:
        k = unit_vector(key.ravel().astype(np.float64))

        with self._lock:
            sid = self._next_id
            self._next_id += 1
            self._items[sid] = k

            for t in range(self.n_tables):
                bucket = self._hash(k, t)
                if bucket not in self._tables[t]:
                    self._tables[t][bucket] = []
                self._tables[t][bucket].append(sid)

            return sid

    def retrieve(
        self,
        query: np.ndarray,
        top_k: int = 5,
    ) -> list[tuple[int, float]]:
        q = unit_vector(query.ravel().astype(np.float64))

        with self._lock:
            candidate_ids: set[int] = set()
            probes = self._generate_probes(q)
            for t in range(self.n_tables):
                bucket = self._hash(q, t)
                if bucket in self._tables[t]:
                    candidate_ids.update(self._tables[t][bucket])

            if not candidate_ids:
                return []

            scores: list[tuple[int, float]] = []
            for sid in candidate_ids:
                if sid in self._items:
                    sim = cosine_similarity(q, self._items[sid])
                    scores.append((sid, sim))

            scores.sort(key=lambda x: -x[1])
            return scores[:top_k]

    def remove(self, storage_id: int) -> bool:
        with self._lock:
            if storage_id not in self._items:
                return False
            k = self._items.pop(storage_id)
            for t in range(self.n_tables):
                bucket = self._hash(k, t)
                if bucket in self._tables[t]:
                    self._tables[t][bucket] = [
                        sid for sid in self._tables[t][bucket] if sid != storage_id
                    ]
            return True

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self._tables = [{} for _ in range(self.n_tables)]
            self._next_id = 0

    def get_stats(self) -> dict:
        total_entries = sum(
            sum(len(lst) for lst in table.values()) for table in self._tables
        )
        return {
            "dim": self.dim,
            "n_tables": self.n_tables,
            "hash_bits": self.hash_bits,
            "n_probes": self.n_probes,
            "n_buckets": self._n_buckets,
            "num_items": len(self._items),
            "total_bucket_entries": total_entries,
        }


class QPhaseHolographicMemory:
    def __init__(self, dim: int = 1024, Q: int = 8):
        if dim < 2 or dim % 2 != 0:
            raise ValueError(f"dim must be even and >= 2, got {dim}")
        if Q not in (4, 8, 16):
            raise ValueError(f"Q must be in {{4, 8, 16}}, got {Q}")

        self.dim = dim
        self.Q = Q
        self._phase_step = 2.0 * math.pi / Q
        self._q_levels = np.arange(Q, dtype=np.float64) * self._phase_step

        self.memory: np.ndarray = np.zeros(dim, dtype=np.complex128)
        self._patterns: dict[int, np.ndarray] = {}
        self._next_id: int = 0
        self._lock = threading.Lock()

    def _quantize_phase(self, fv: np.ndarray) -> np.ndarray:
        phases = np.angle(fv)
        quantized = np.round(phases / self._phase_step) * self._phase_step
        mag = np.abs(fv)
        return mag * np.exp(1j * quantized)

    def _make_phv(self, vector: np.ndarray) -> np.ndarray:
        v = _ensure_dim(vector, self.dim)
        fv = np.fft.fft(v.astype(np.complex128))
        return self._quantize_phase(fv)

    def bind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a_q = self._quantize_phase(a) if np.iscomplexobj(a) else self._make_phv(a)
        b_q = self._quantize_phase(b) if np.iscomplexobj(b) else self._make_phv(b)
        return self._quantize_phase(a_q * b_q)

    def unbind(self, c: np.ndarray, a: np.ndarray) -> np.ndarray:
        a_q = self._quantize_phase(a) if np.iscomplexobj(a) else self._make_phv(a)
        return self._quantize_phase(np.conj(a_q) * c)

    def bundle(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return self._quantize_phase(a + b)

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        aq = self._quantize_phase(a) if np.iscomplexobj(a) else self._make_phv(a)
        bq = self._quantize_phase(b) if np.iscomplexobj(b) else self._make_phv(b)
        d = len(aq)
        return float((np.conj(aq) * bq).real.sum() / d)

    def store(self, key: np.ndarray, value: np.ndarray) -> int:
        k_phv = self._make_phv(key)
        v_phv = self._make_phv(value)
        encoded = self.bind(k_phv, v_phv)

        with self._lock:
            sid = self._next_id
            self._next_id += 1
            self._patterns[sid] = k_phv
            self.memory = self.bundle(self.memory, encoded)
            return sid

    def retrieve(
        self, query: np.ndarray, top_k: int = 1
    ) -> list[tuple[int, np.ndarray, float]]:
        q_phv = self._make_phv(query)

        with self._lock:
            if not self._patterns:
                return []

            scores: list[tuple[int, float]] = []
            for sid, k_phv in self._patterns.items():
                sim = self.similarity(q_phv, k_phv)
                scores.append((sid, sim))

            scores.sort(key=lambda x: -x[1])
            results: list[tuple[int, np.ndarray, float]] = []
            for sid, sim in scores[:top_k]:
                k_phv = self._patterns[sid]
                v_est = self.unbind(self.memory, k_phv)
                results.append((sid, v_est.real, sim))
            return results

    def clear(self) -> None:
        with self._lock:
            self.memory = np.zeros(self.dim, dtype=np.complex128)
            self._patterns.clear()
            self._next_id = 0

    def capacity(self) -> int:
        return max(1, int(self.Q * self.dim * math.log(self.dim) / (2.0 * math.pi)))

    def get_stats(self) -> dict:
        return {
            "dim": self.dim,
            "Q": self.Q,
            "phase_step_rad": self._phase_step,
            "num_items": len(self._patterns),
            "capacity": self.capacity(),
            "bits_per_position": int(math.log2(self.Q)),
        }


class PredictiveCodingKVCache:
    def __init__(
        self,
        dim: int = 1024,
        ar_order: int = 2,
        error_bits: int = 4,
        max_size: int = 8192,
    ):
        if dim < 2:
            raise ValueError(f"dim must be >= 2, got {dim}")
        if error_bits < 1 or error_bits > 8:
            raise ValueError(f"error_bits must be in [1, 8], got {error_bits}")

        self.dim = dim
        self.ar_order = ar_order
        self.error_bits = error_bits
        self.max_size = max_size
        self._n_levels = 1 << error_bits

        self._history: dict[int, list[np.ndarray]] = {}
        self._entries: dict[int, PredictiveEntry] = OrderedDict()
        self._next_id: int = 0
        self._quantizer_thresholds: np.ndarray = np.zeros(self._n_levels + 1)
        self._quantizer_levels: np.ndarray = np.linspace(-1.0, 1.0, self._n_levels)
        self._lock = threading.Lock()
        self._init_quantizer()

    def _init_quantizer(self) -> None:
        self._quantizer_levels = np.linspace(-1.0, 1.0, self._n_levels).astype(
            np.float64
        )
        self._quantizer_thresholds = np.linspace(
            -1.0 - 1.0 / self._n_levels,
            1.0 + 1.0 / self._n_levels,
            self._n_levels + 1,
        ).astype(np.float64)

    def _predict(self, position: int) -> Optional[np.ndarray]:
        history = self._history.get(position, [])
        if len(history) < self.ar_order:
            return None
        recent = history[-self.ar_order :]
        prediction = np.zeros(self.dim, dtype=np.float64)
        weights = np.array([0.7, 0.3][: self.ar_order], dtype=np.float64)
        for i, v in enumerate(recent):
            prediction += weights[i] * v
        return prediction

    def _quantise_error(self, error: np.ndarray) -> np.ndarray:
        max_abs = np.max(np.abs(error)) + 1e-10
        normalised = error / max_abs
        indices = np.searchsorted(self._quantizer_thresholds[1:-1], normalised)
        quantised_levels = self._quantizer_levels[
            np.clip(indices, 0, self._n_levels - 1)
        ]
        return quantised_levels.astype(np.float16)

    def _dequantise_error(self, quantised: np.ndarray, max_abs: float) -> np.ndarray:
        return quantised.astype(np.float64) * max_abs

    def store(
        self,
        key: np.ndarray,
        value: np.ndarray,
        position: int,
    ) -> None:
        k = unit_vector(key.ravel().astype(np.float64))
        v = unit_vector(value.ravel().astype(np.float64))
        kv = 0.5 * k + 0.5 * v

        with self._lock:
            if position not in self._history:
                self._history[position] = []
            self._history[position].append(kv.copy())

            if len(self._history[position]) < self.ar_order + 1:
                return

            prediction = self._predict(position)
            if prediction is None:
                return

            error = kv - prediction
            error_norm = float(np.linalg.norm(error))
            kv_norm = float(np.linalg.norm(kv))

            quantised = self._quantise_error(error)

            entry = PredictiveEntry(
                position=position,
                quantized_error=quantised,
                prediction_coeffs=np.zeros(0),
                actual_norm=kv_norm,
                error_norm=error_norm,
                timestamp=time.monotonic(),
            )
            self._entries[position] = entry
            self._evict_if_needed()

    def retrieve(self, position: int) -> Optional[tuple[np.ndarray, np.ndarray]]:
        with self._lock:
            if position not in self._entries:
                return None

            entry = self._entries[position]
            history = self._history.get(position, [])
            if len(history) < self.ar_order:
                return None

            recent = history[-self.ar_order :]
            weights = np.array([0.7, 0.3][: self.ar_order], dtype=np.float64)
            prediction = np.zeros(self.dim, dtype=np.float64)
            for i, v in enumerate(recent):
                prediction += weights[i] * v

            max_abs = entry.error_norm + 1e-10
            error = self._dequantise_error(entry.quantized_error, max_abs)
            reconstructed = prediction + error
            reconstructed = unit_vector(reconstructed)

            k_est = unit_vector(reconstructed)
            v_est = unit_vector(reconstructed)
            return (k_est, v_est)

    def _evict_if_needed(self) -> None:
        while len(self._entries) > self.max_size and len(self._entries) > 1:
            oldest_pos = next(iter(self._entries))
            del self._entries[oldest_pos]

    def compression_ratio(self) -> float:
        full_bytes = self.dim * 8
        error_bytes = self.dim * (self.error_bits / 8)
        return full_bytes / max(error_bytes, 1)

    def clear(self) -> None:
        with self._lock:
            self._history.clear()
            self._entries.clear()
            self._next_id = 0

    def get_stats(self) -> dict:
        error_norms = [e.error_norm for e in self._entries.values()]
        return {
            "dim": self.dim,
            "ar_order": self.ar_order,
            "error_bits": self.error_bits,
            "n_levels": self._n_levels,
            "num_entries": len(self._entries),
            "max_size": self.max_size,
            "compression_ratio": self.compression_ratio(),
            "avg_error_norm": float(np.mean(error_norms)) if error_norms else 0.0,
        }


class HolographicProximitySearch:
    def __init__(
        self,
        dim: int = 1024,
        n_buckets: int = 4,
        seed: int = 42,
    ):
        if dim < 2:
            raise ValueError(f"dim must be >= 2, got {dim}")
        if n_buckets < 1:
            raise ValueError(f"n_buckets must be >= 1, got {n_buckets}")

        self.dim = dim
        self.n_buckets = n_buckets
        self._rng = np.random.RandomState(seed)

        self._bucket_memories: list[np.ndarray] = [
            np.zeros(dim, dtype=np.complex128) for _ in range(n_buckets)
        ]
        self._bucket_counts: list[int] = [0] * n_buckets
        self._items: dict[int, np.ndarray] = {}
        self._item_bucket: dict[int, int] = {}
        self._next_id: int = 0
        self._lock = threading.Lock()

        self._projection = self._rng.randn(dim, dim).astype(np.float64)
        self._projection /= np.linalg.norm(self._projection, axis=0, keepdims=True)

    def _bucket_of(self, key: np.ndarray) -> int:
        h = int(np.sum(key * self._projection[:, 0])) & 0xFFFFFFFF
        return h % self.n_buckets

    def _to_freq(self, vector: np.ndarray) -> np.ndarray:
        return np.fft.fft(vector.astype(np.complex128))

    def store(self, key: np.ndarray) -> int:
        k = unit_vector(key.ravel().astype(np.float64))
        fv = self._to_freq(k)

        with self._lock:
            sid = self._next_id
            self._next_id += 1
            self._items[sid] = k
            bucket = self._bucket_of(k)
            self._item_bucket[sid] = bucket
            self._bucket_memories[bucket] += fv
            self._bucket_counts[bucket] += 1
            return sid

    def retrieve(
        self,
        query: np.ndarray,
        top_k: int = 5,
    ) -> list[tuple[int, float]]:
        q = unit_vector(query.ravel().astype(np.float64))
        q_fft = self._to_freq(q)

        with self._lock:
            all_scores: dict[int, float] = {}

            for b in range(self.n_buckets):
                if self._bucket_counts[b] == 0:
                    continue
                conv = np.fft.ifft(q_fft * np.conj(self._bucket_memories[b])).real
                n = self._bucket_counts[b]
                scores = conv / max(n, 1)

                for sid, bucket in self._item_bucket.items():
                    if bucket == b and sid in self._items:
                        all_scores[sid] = float(scores[sid % self.dim])

            if not all_scores:
                return []

            ranked = sorted(all_scores.items(), key=lambda x: -x[1])
            return ranked[:top_k]

    def remove(self, storage_id: int) -> bool:
        with self._lock:
            if storage_id not in self._items:
                return False
            k = self._items.pop(storage_id)
            bucket = self._item_bucket.pop(storage_id)
            fv = self._to_freq(k)
            self._bucket_memories[bucket] -= fv
            self._bucket_counts[bucket] = max(0, self._bucket_counts[bucket] - 1)
            return True

    def clear(self) -> None:
        with self._lock:
            for b in range(self.n_buckets):
                self._bucket_memories[b].fill(0)
                self._bucket_counts[b] = 0
            self._items.clear()
            self._item_bucket.clear()
            self._next_id = 0

    def get_stats(self) -> dict:
        return {
            "dim": self.dim,
            "n_buckets": self.n_buckets,
            "num_items": len(self._items),
            "bucket_counts": list(self._bucket_counts),
            "theoretical_speedup": f"O(D log D) vs O(N*D), D={self.dim}",
        }


class WaveletHolographicKVCache:
    def __init__(
        self,
        dim: int = 1024,
        n_levels: int = 4,
        max_size: int = 8192,
    ):
        if dim < 2:
            raise ValueError(f"dim must be >= 2, got {dim}")
        if n_levels < 1:
            raise ValueError(f"n_levels must be >= 1, got {n_levels}")

        self.dim = dim
        self.n_levels = n_levels
        self.max_size = max_size

        self._strides = [2 ** (n_levels - 1 - i) for i in range(n_levels)]
        self._level_memories: list[np.ndarray] = [
            np.zeros(dim, dtype=np.float64) for _ in range(n_levels)
        ]
        self._entries: dict[int, dict] = {}
        self._position_vectors: dict[int, np.ndarray] = {}
        self._next_id: int = 0
        self._lock = threading.Lock()

    def _get_position_vector(self, position: int) -> np.ndarray:
        if position not in self._position_vectors:
            seed = hash(f"wvpos_{position}") & 0xFFFFFFFF
            self._position_vectors[position] = generate_random_hd_vector(
                self.dim, seed=seed
            )
        return self._position_vectors[position]

    def _active_levels(self, position: int) -> list[int]:
        active = []
        for lev, stride in enumerate(self._strides):
            if position % stride == 0:
                active.append(lev)
        return active

    def store(
        self,
        key: np.ndarray,
        value: np.ndarray,
        position: int,
    ) -> int:
        k = unit_vector(key.ravel().astype(np.float64))
        v = unit_vector(value.ravel().astype(np.float64))
        pos_vec = self._get_position_vector(position)
        encoded_k = hrr_bind(pos_vec, k)
        encoded_v = hrr_bind(pos_vec, v)
        encoding = hrr_bind(encoded_k, encoded_v)

        with self._lock:
            sid = self._next_id
            self._next_id += 1

            active = self._active_levels(position)
            for lev in active:
                self._level_memories[lev] = hrr_bundle(
                    self._level_memories[lev], encoding
                )

            self._entries[sid] = {
                "position": position,
                "key": k,
                "value": v,
                "encoding": encoding,
                "active_levels": active,
                "timestamp": time.monotonic(),
            }
            self._evict_if_needed()
            return sid

    def retrieve(
        self,
        position: int,
        top_k: int = 1,
        coarse_to_fine: bool = True,
    ) -> list[tuple[int, np.ndarray, float]]:
        pos_vec = self._get_position_vector(position)
        active = self._active_levels(position)

        with self._lock:
            if not active:
                return []

            if coarse_to_fine:
                level_order = list(reversed(active))
            else:
                level_order = active

            combined_estimate = np.zeros(self.dim, dtype=np.float64)
            for lev in level_order:
                lev_mem = self._level_memories[lev]
                val_est = hrr_unbind(lev_mem, pos_vec)
                combined_estimate += val_est

            combined_estimate = unit_vector(combined_estimate)

            scores: list[tuple[int, float]] = []
            for sid, entry in self._entries.items():
                if position in self._active_levels(entry["position"]) or True:
                    sim = cosine_similarity(combined_estimate, entry["key"])
                    scores.append((sid, sim))

            scores.sort(key=lambda x: -x[1])
            results: list[tuple[int, np.ndarray, float]] = []
            for sid, sim in scores[:top_k]:
                entry = self._entries[sid]
                results.append((sid, entry["value"].copy(), sim))
            return results

    def _evict_if_needed(self) -> None:
        while len(self._entries) > self.max_size and len(self._entries) > 1:
            oldest_sid = min(
                self._entries,
                key=lambda s: self._entries[s]["timestamp"],
            )
            entry = self._entries.pop(oldest_sid)
            for lev in entry["active_levels"]:
                self._level_memories[lev] -= entry["encoding"] * 0.1

    def clear(self) -> None:
        with self._lock:
            for lev in range(self.n_levels):
                self._level_memories[lev].fill(0)
            self._entries.clear()
            self._position_vectors.clear()
            self._next_id = 0

    def get_stats(self) -> dict:
        level_counts = [0] * self.n_levels
        for entry in self._entries.values():
            for lev in entry["active_levels"]:
                level_counts[lev] += 1
        return {
            "dim": self.dim,
            "n_levels": self.n_levels,
            "strides": self._strides,
            "num_items": len(self._entries),
            "max_size": self.max_size,
            "level_counts": level_counts,
        }


class FluxSurfaceCache:
    def __init__(
        self,
        dim: int = 1024,
        max_size: int = 4096,
        n_surfaces: int = 16,
        bootstrap_gain: float = 0.01,
        confinement_strength: float = 0.1,
    ):
        if dim < 2:
            raise ValueError(f"dim must be >= 2, got {dim}")
        if n_surfaces < 2:
            raise ValueError(f"n_surfaces must be >= 2, got {n_surfaces}")

        self.dim = dim
        self.max_size = max_size
        self.n_surfaces = n_surfaces
        self.bootstrap_gain = bootstrap_gain
        self.confinement_strength = confinement_strength

        self._surfaces: list[list[FluxEntry]] = [[] for _ in range(n_surfaces)]
        self._entries: dict[int, FluxEntry] = {}
        self._next_id: int = 0
        self._global_memory = HrrMemory(dim=dim, decay_rate=0.005)
        self._position_vectors: dict[int, np.ndarray] = {}
        self._lock = threading.Lock()

    def _get_position_vector(self, position: int) -> np.ndarray:
        if position not in self._position_vectors:
            seed = hash(f"flux_{position}") & 0xFFFFFFFF
            self._position_vectors[position] = generate_random_hd_vector(
                self.dim, seed=seed
            )
        return self._position_vectors[position]

    def _compute_rho(self, key: np.ndarray, access_count: int) -> float:
        importance = _importance_from_entropy(key, threshold=0.5)
        bootstrap_effect = (
            self.confinement_strength * access_count * self.bootstrap_gain
        )
        rho = max(0.0, min(1.0, 1.0 - importance + bootstrap_effect * 0.1))
        return rho

    def _surface_index(self, rho: float) -> int:
        idx = int(rho * (self.n_surfaces - 1))
        return max(0, min(self.n_surfaces - 1, idx))

    def store(
        self,
        key: np.ndarray,
        value: np.ndarray,
        position: int,
    ) -> int:
        k = unit_vector(key.ravel().astype(np.float64))
        v = unit_vector(value.ravel().astype(np.float64))
        pos_vec = self._get_position_vector(position)
        encoding = hrr_bind(hrr_bind(pos_vec, k), v)

        with self._lock:
            rho = self._compute_rho(k, 0)
            sid = self._next_id
            self._next_id += 1

            entry = FluxEntry(
                position=position,
                key_hv=k,
                value_hv=v,
                rho=rho,
                bootstrap_current=0.0,
                temperature=1.0,
                timestamp=time.monotonic(),
            )

            surface_idx = self._surface_index(rho)
            self._surfaces[surface_idx].append(entry)
            self._entries[sid] = entry

            self._global_memory.memory = hrr_bundle(
                self._global_memory.memory, encoding
            )
            self._global_memory.memory = unit_vector(self._global_memory.memory)

            self._evict_if_needed()
            return sid

    def retrieve(
        self,
        position: int,
        top_k: int = 1,
    ) -> list[tuple[int, np.ndarray, float]]:
        pos_vec = self._get_position_vector(position)

        with self._lock:
            value_est = hrr_unbind(self._global_memory.memory, pos_vec)
            value_est = unit_vector(value_est)

            scores: list[tuple[int, float]] = []
            for sid, entry in self._entries.items():
                sim = cosine_similarity(value_est, entry.key_hv)
                scores.append((sid, sim))

            scores.sort(key=lambda x: -x[1])
            results: list[tuple[int, np.ndarray, float]] = []
            for sid, sim in scores[:top_k]:
                entry = self._entries[sid]
                entry.access_count += 1
                entry.bootstrap_current += self.bootstrap_gain
                rho_adj = max(
                    0.0,
                    entry.rho - self.confinement_strength * entry.bootstrap_current,
                )
                entry.rho = rho_adj
                results.append((sid, entry.value_hv.copy(), sim))
            return results

    def _evict_if_needed(self) -> None:
        total = sum(len(s) for s in self._surfaces)
        while total > self.max_size and total > 1:
            for surf_idx in range(self.n_surfaces - 1, -1, -1):
                if self._surfaces[surf_idx]:
                    removed = self._surfaces[surf_idx].pop()
                    total -= 1
                    for sid, entry in list(self._entries.items()):
                        if entry is removed:
                            del self._entries[sid]
                            break
                    break
            if total <= self.max_size:
                break

    def clear(self) -> None:
        with self._lock:
            self._surfaces = [[] for _ in range(self.n_surfaces)]
            self._entries.clear()
            self._global_memory.clear()
            self._position_vectors.clear()
            self._next_id = 0

    def surface_distribution(self) -> list[int]:
        return [len(s) for s in self._surfaces]

    def get_stats(self) -> dict:
        rho_values = [e.rho for e in self._entries.values()]
        bootstrap_values = [e.bootstrap_current for e in self._entries.values()]
        return {
            "dim": self.dim,
            "max_size": self.max_size,
            "n_surfaces": self.n_surfaces,
            "num_entries": len(self._entries),
            "surface_distribution": self.surface_distribution(),
            "avg_rho": float(np.mean(rho_values)) if rho_values else 0.0,
            "avg_bootstrap": float(np.mean(bootstrap_values))
            if bootstrap_values
            else 0.0,
            "bootstrap_gain": self.bootstrap_gain,
            "confinement_strength": self.confinement_strength,
        }
