from __future__ import annotations

import hashlib
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

import numpy as np

from spectralstream.core.math_primitives import (
    cascade_eviction_score,
    cosine_similarity,
    dct,
    generate_random_complex_vector,
    generate_random_hd_vector,
    hrr_bind,
    hrr_bundle,
    hrr_unbind,
    idct,
    landau_zener_coherence,
    spectral_entropy,
    unit_vector,
)
from spectralstream.kv_cache.core import EPS
from spectralstream.memory.hrr import HrrMemory


# ═══════════════════════════════════════════════════════════════════════════
# KVCacheEntry
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class KVCacheEntry:
    position: int
    head_idx: int
    layer_idx: int
    key_hv: np.ndarray
    value_hv: np.ndarray
    entropy: float
    coherence: float
    timestamp: float
    frequency: int = 1
    importance: float = 1.0
    encoding: Optional[np.ndarray] = None


# ═══════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════


def _importance_from_entropy(key: np.ndarray, threshold: float = 0.5) -> float:
    ent = spectral_entropy(key)
    if ent < threshold:
        return 0.5 + 0.5 * (ent / max(threshold, EPS))
    return 0.5 + 0.5 * min(1.0, ent)


def _eviction_score(coherence: float, frequency: float, importance: float) -> float:
    return cascade_eviction_score(
        entropy=1.0 - importance,
        coherence=coherence,
        recency=coherence,
        frequency=frequency,
    )


# ═══════════════════════════════════════════════════════════════════════════
# HolographicKVCache
# ═══════════════════════════════════════════════════════════════════════════


class HolographicKVCache:
    def __init__(
        self,
        dim: int = 1024,
        max_size: int = 4096,
        n_heads: int = 8,
        n_layers: int = 32,
        decay_rate: float = 0.01,
        coherence_half_life: float = 1000.0,
        entropy_threshold: float = 0.5,
        normalise: bool = True,
    ):
        self.dim = dim
        self.max_size = max_size
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.decay_rate = decay_rate
        self.coherence_half_life = coherence_half_life
        self.entropy_threshold = entropy_threshold
        self.normalise = normalise

        self.global_memory = HrrMemory(
            dim=dim, decay_rate=decay_rate, normalise_memory=normalise
        )
        self.per_layer_memories: dict[int, HrrMemory] = {}
        self.per_head_memories: dict[tuple[int, int], HrrMemory] = {}

        self._position_vectors: dict[int, np.ndarray] = {}
        self.entries: dict[int, KVCacheEntry] = OrderedDict()
        self._entry_positions: list[int] = []
        self._global_step: int = 0
        self._lock = threading.Lock()

        self.hits: int = 0
        self.misses: int = 0
        self.holographic_hits: int = 0

    def _get_position_vector(self, position: int) -> np.ndarray:
        if position not in self._position_vectors:
            seed = hash(f"position_{position}") & 0xFFFFFFFF
            self._position_vectors[position] = generate_random_hd_vector(
                self.dim, seed=seed
            )
        return self._position_vectors[position]

    def _get_layer_memory(self, layer_idx: int) -> HrrMemory:
        if layer_idx not in self.per_layer_memories:
            self.per_layer_memories[layer_idx] = HrrMemory(
                dim=self.dim,
                decay_rate=self.decay_rate,
                normalise_memory=self.normalise,
            )
        return self.per_layer_memories[layer_idx]

    def _get_head_memory(self, layer_idx: int, head_idx: int) -> HrrMemory:
        key = (layer_idx, head_idx)
        if key not in self.per_head_memories:
            self.per_head_memories[key] = HrrMemory(
                dim=self.dim,
                decay_rate=self.decay_rate,
                normalise_memory=self.normalise,
            )
        return self.per_head_memories[key]

    def _compute_importance(self, key: np.ndarray) -> float:
        return _importance_from_entropy(key, self.entropy_threshold)

    def store(
        self,
        key: np.ndarray,
        value: np.ndarray,
        position: int,
        head_idx: int = 0,
        layer_idx: int = 0,
    ) -> None:
        with self._lock:
            self._global_step += 1
            now = time.monotonic()
            importance = self._compute_importance(key)

            k_norm = unit_vector(key.ravel().astype(np.float64))
            v_norm = unit_vector(value.ravel().astype(np.float64))
            pos_vec = self._get_position_vector(position)

            encoded_k = self.global_memory.bind(pos_vec, k_norm)
            encoded_v = self.global_memory.bind(pos_vec, v_norm)
            encoding = self.global_memory.bind(encoded_k, encoded_v)
            encoding = encoding * importance

            self.global_memory.memory = self.global_memory.bundle(
                self.global_memory.memory, encoding
            )
            if self.normalise:
                self.global_memory.memory = unit_vector(self.global_memory.memory)

            layer_mem = self._get_layer_memory(layer_idx)
            layer_mem.memory = layer_mem.bundle(layer_mem.memory, encoding)
            if self.normalise:
                layer_mem.memory = unit_vector(layer_mem.memory)

            head_mem = self._get_head_memory(layer_idx, head_idx)
            head_mem.memory = head_mem.bundle(head_mem.memory, encoding)
            if self.normalise:
                head_mem.memory = unit_vector(head_mem.memory)

            entry = KVCacheEntry(
                position=position,
                head_idx=head_idx,
                layer_idx=layer_idx,
                key_hv=k_norm,
                value_hv=v_norm,
                entropy=spectral_entropy(key),
                coherence=1.0,
                timestamp=now,
                importance=importance,
                encoding=encoding,
            )

            self._evict_if_needed()

            self.entries[position] = entry
            if position not in self._entry_positions:
                self._entry_positions.append(position)

    def retrieve(
        self,
        position: int,
        head_idx: int = 0,
        layer_idx: int = 0,
    ) -> Optional[tuple[np.ndarray, np.ndarray]]:
        with self._lock:
            if position in self.entries:
                entry = self.entries[position]
                entry.frequency += 1
                self.hits += 1
                return (entry.key_hv.copy(), entry.value_hv.copy())

            pos_vec = self._get_position_vector(position)
            mem = self.global_memory.memory
            value_est = self.global_memory.unbind(mem, pos_vec)

            if layer_idx in self.per_layer_memories:
                layer_mem = self.per_layer_memories[layer_idx].memory
                layer_val = self.global_memory.unbind(layer_mem, pos_vec)
                value_est = 0.7 * value_est + 0.3 * layer_val

            head_key = (layer_idx, head_idx)
            if head_key in self.per_head_memories:
                head_mem = self.per_head_memories[head_key].memory
                head_val = self.global_memory.unbind(head_mem, pos_vec)
                value_est = 0.6 * value_est + 0.4 * head_val

            self.holographic_hits += 1
            self.misses += 1

            value_est = unit_vector(value_est)
            best_sim = -1.0
            best_entry: Optional[KVCacheEntry] = None
            for entry in self.entries.values():
                sim = cosine_similarity(value_est, entry.key_hv)
                if sim > best_sim:
                    best_sim = sim
                    best_entry = entry

            if best_sim > 0.7 and best_entry is not None:
                return (best_entry.key_hv.copy(), best_entry.value_hv.copy())

            return (value_est, value_est)

    def store_batch(
        self,
        keys: np.ndarray,
        values: np.ndarray,
        positions: np.ndarray,
        head_indices: Optional[np.ndarray] = None,
        layer_indices: Optional[np.ndarray] = None,
    ) -> None:
        n = len(positions)
        for i in range(n):
            hi = int(head_indices[i]) if head_indices is not None else 0
            li = int(layer_indices[i]) if layer_indices is not None else 0
            self.store(keys[i], values[i], int(positions[i]), hi, li)

    def query(self, query_vec: np.ndarray, top_k: int = 10) -> list[tuple[int, float]]:
        q = unit_vector(query_vec.ravel().astype(np.float64))
        results = []
        for pos, entry in self.entries.items():
            sim = cosine_similarity(q, entry.key_hv)
            results.append((pos, sim))
        results.sort(key=lambda x: -x[1])
        return results[:top_k]

    def _evict_if_needed(self) -> None:
        while len(self.entries) > self.max_size and len(self.entries) > 1:
            self._evict_one()

    def _evict_one(self) -> None:
        now = time.monotonic()
        max_freq = max((e.frequency for e in self.entries.values()), default=1)
        scores: dict[int, float] = {}
        for pos, entry in self.entries.items():
            age = now - entry.timestamp
            coherence = landau_zener_coherence(age, self.coherence_half_life)
            freq_norm = entry.frequency / max(max_freq, 1)
            score = _eviction_score(coherence, freq_norm, entry.importance)
            scores[pos] = score

        if not scores:
            return

        evict_pos = min(scores, key=scores.get)
        if evict_pos in self.entries:
            entry = self.entries[evict_pos]
            if entry.encoding is not None:
                self.global_memory.memory -= entry.encoding * 0.1
                if self.normalise:
                    self.global_memory.memory = unit_vector(self.global_memory.memory)
            del self.entries[evict_pos]
            if evict_pos in self._entry_positions:
                self._entry_positions.remove(evict_pos)

    def apply_decay(self) -> None:
        with self._lock:
            self.global_memory.apply_decay()
            for mem in self.per_layer_memories.values():
                mem.apply_decay()
            for mem in self.per_head_memories.values():
                mem.apply_decay()

    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / max(total, 1)

    def num_positions(self) -> int:
        return len(self.entries)

    def clear(self) -> None:
        with self._lock:
            self.global_memory.clear()
            self.per_layer_memories.clear()
            self.per_head_memories.clear()
            self._position_vectors.clear()
            self.entries.clear()
            self._entry_positions.clear()
            self._global_step = 0
            self.hits = self.misses = self.holographic_hits = 0

    def get_stats(self) -> dict:
        return {
            "dim": self.dim,
            "max_size": self.max_size,
            "num_positions": self.num_positions(),
            "hit_rate": self.hit_rate(),
            "hits": self.hits,
            "misses": self.misses,
            "holographic_hits": self.holographic_hits,
            "global_memory_norm": float(np.linalg.norm(self.global_memory.memory)),
            "n_layers_used": len(self.per_layer_memories),
            "n_heads_used": len(self.per_head_memories),
            "global_step": self._global_step,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
        }


# ═══════════════════════════════════════════════════════════════════════════
# HolographicWeightStore
# ═══════════════════════════════════════════════════════════════════════════


class HolographicWeightStore:
    def __init__(
        self,
        dim: int = 2048,
        n_layers: int = 32,
        layer_dim: int = 4096,
        compression_pct: float = 5.0,
        progressive_stages: int = 4,
        use_fhrr: bool = False,
    ):
        self.dim = dim
        self.n_layers = n_layers
        self.layer_dim = layer_dim
        self.compression_pct = compression_pct
        self.progressive_stages = progressive_stages
        self.use_fhrr = use_fhrr

        self._dct_items: dict[tuple, dict] = {}
        self._weight_metadata: dict[tuple, dict] = {}
        self._next_id: int = 0

    def _dct_encode(self, weights: np.ndarray) -> np.ndarray:
        return dct(weights.astype(np.float64)).astype(np.float64)

    def _idct_decode(self, dct_coeffs: np.ndarray) -> np.ndarray:
        return idct(dct_coeffs.astype(np.float64)).astype(np.float64)

    def store_weight(
        self,
        weights: np.ndarray,
        layer_idx: int,
        weight_name: str = "",
        row_idx: Optional[int] = None,
    ) -> int:
        w_flat = weights.ravel()
        dct_coeffs = self._dct_encode(w_flat)

        energy = dct_coeffs**2
        total_energy = float(energy.sum())
        min_coeffs = min(16, len(dct_coeffs))

        if total_energy > EPS:
            sorted_idx = np.argsort(-energy)
            cumsum = np.cumsum(energy[sorted_idx])
            target = self.compression_pct / 100.0
            n_keep = max(
                min_coeffs,
                int(np.searchsorted(cumsum / total_energy, target) + 1),
            )
            n_keep = min(n_keep, len(dct_coeffs))
            keep_idx = sorted_idx[:n_keep]
        else:
            keep_idx = np.arange(min(min_coeffs, len(dct_coeffs)))

        key = (layer_idx, weight_name, row_idx if row_idx is not None else 0)
        keep_idx_sorted = np.sort(np.asarray(keep_idx).ravel())
        n_per_stage = max(1, len(keep_idx_sorted) // self.progressive_stages)

        self._dct_items[key] = {
            "dct_full": dct_coeffs.copy(),
            "keep_idx": keep_idx_sorted,
            "shape": weights.shape,
            "n_keep": len(keep_idx_sorted),
            "n_per_stage": n_per_stage,
        }

        sid = self._next_id
        self._next_id += 1
        self._weight_metadata[key] = {
            "layer_idx": layer_idx,
            "weight_name": weight_name,
            "row_idx": row_idx,
            "shape": weights.shape,
            "n_coeffs": len(keep_idx_sorted),
            "dct_length": len(dct_coeffs),
        }
        return sid

    def recall_weight(
        self,
        layer_idx: int,
        weight_name: str = "",
        row_idx: Optional[int] = None,
        stage: Optional[int] = None,
    ) -> Optional[np.ndarray]:
        key = (layer_idx, weight_name, row_idx if row_idx is not None else 0)

        if key not in self._dct_items:
            return None

        item = self._dct_items[key]
        dct_full = item["dct_full"].copy()
        keep_idx = item["keep_idx"]
        n_per_stage = item["n_per_stage"]

        if stage is not None and stage < self.progressive_stages:
            cutoff = min(len(keep_idx), (stage + 1) * n_per_stage)
            mask = np.zeros(len(dct_full), dtype=bool)
            mask[keep_idx[:cutoff]] = True
            dct_full[~mask] = 0.0

        weights_1d = self._idct_decode(dct_full)
        return weights_1d.reshape(item["shape"])

    def progressive_recall(
        self,
        layer_idx: int,
        weight_name: str = "",
        row_idx: Optional[int] = None,
    ) -> list[np.ndarray]:
        results = []
        for stage in range(self.progressive_stages):
            w = self.recall_weight(layer_idx, weight_name, row_idx, stage=stage)
            if w is not None:
                results.append(w)
        return results

    def compress_weight_matrix(
        self,
        weight_matrix: np.ndarray,
        layer_idx: int,
        weight_name: str = "",
    ) -> list[int]:
        if weight_matrix.ndim == 1:
            return [self.store_weight(weight_matrix, layer_idx, weight_name, row_idx=0)]
        sids = []
        for row_idx in range(weight_matrix.shape[0]):
            sid = self.store_weight(
                weight_matrix[row_idx], layer_idx, weight_name, row_idx=row_idx
            )
            sids.append(sid)
        return sids

    def clear(self) -> None:
        self._dct_items.clear()
        self._weight_metadata.clear()
        self._next_id = 0

    def get_stats(self) -> dict:
        return {
            "dim": self.dim,
            "n_layers": self.n_layers,
            "num_weights": len(self._dct_items),
            "layer_dim": self.layer_dim,
            "compression_pct": self.compression_pct,
            "progressive_stages": self.progressive_stages,
            "use_fhrr": self.use_fhrr,
        }


# ═══════════════════════════════════════════════════════════════════════════
# HolographicTier / TierEntry / HolographicCacheHierarchy
# ═══════════════════════════════════════════════════════════════════════════


class HolographicTier:
    L1 = 0
    L2 = 1
    L3 = 2
    L4 = 3


@dataclass
class TierEntry:
    key: str
    data: Optional[np.ndarray]
    frequency: int = 1
    last_access: float = 0.0
    tier: int = HolographicTier.L2
    encoding: Optional[np.ndarray] = None


class HolographicCacheHierarchy:
    def __init__(
        self,
        l1_dim: int = 128,
        l2_dim: int = 1024,
        l3_dim: int = 2048,
        l1_capacity: int = 128,
        l2_capacity: int = 4096,
        l3_capacity: int = 65536,
        l4_capacity: int = 1048576,
        promotion_threshold: int = 10,
        demotion_threshold: int = 3,
        mmap_backed: bool = True,
        mmap_dir: str = "/tmp/spectralstream_holographic",
    ):
        self.l1_dim = max(128, l1_dim)
        self.l2_dim = max(128, l2_dim)
        self.l3_dim = max(128, l3_dim)
        self.l1 = HrrMemory(dim=self.l1_dim, decay_rate=0.001)
        self.l2 = HrrMemory(dim=self.l2_dim, decay_rate=0.005)
        self.l3 = HrrMemory(dim=self.l3_dim, decay_rate=0.001)

        self.l1_capacity = l1_capacity
        self.l2_capacity = l2_capacity
        self.l3_capacity = l3_capacity
        self.l4_capacity = l4_capacity
        self.promotion_threshold = promotion_threshold
        self.demotion_threshold = demotion_threshold
        self.mmap_backed = mmap_backed
        self.mmap_dir = mmap_dir

        self.l1_entries: dict[str, TierEntry] = OrderedDict()
        self.l2_entries: dict[str, TierEntry] = OrderedDict()
        self.l3_entries: dict[str, TierEntry] = OrderedDict()
        self.l4_entries: dict[str, TierEntry] = OrderedDict()
        self._all_keys: dict[str, np.ndarray] = {}

        self._lock = threading.Lock()
        self._promotion_count = 0
        self._demotion_count = 0

        if mmap_backed:
            os.makedirs(mmap_dir, exist_ok=True)

    def _mmap_path(self, key: str) -> str:
        h = hashlib.md5(key.encode()).hexdigest()
        return f"{self.mmap_dir}/{h}.npy"

    def store(self, key: str, data: np.ndarray, tier: int = HolographicTier.L2) -> None:
        with self._lock:
            if tier == HolographicTier.L1:
                self._store_l1(key, data)
            elif tier == HolographicTier.L2:
                self._store_l2(key, data)
            elif tier == HolographicTier.L3:
                self._store_l3(key, data)
            else:
                self._store_l4(key, data)

    def _store_l1(self, key: str, data: np.ndarray) -> None:
        k_vec = unit_vector(data.ravel().astype(np.float64))
        self.l1.store(k_vec)
        self._all_keys[key] = k_vec
        entry = TierEntry(key=key, data=None, tier=HolographicTier.L1, encoding=k_vec)
        self.l1_entries[key] = entry
        self._evict_l1_if_needed()

    def _store_l2(self, key: str, data: np.ndarray) -> None:
        k_vec = unit_vector(data.ravel().astype(np.float64))
        self.l2.store(k_vec)
        self._all_keys[key] = k_vec
        entry = TierEntry(key=key, data=None, tier=HolographicTier.L2, encoding=k_vec)
        self.l2_entries[key] = entry
        self._evict_l2_if_needed()

    def _store_l3(self, key: str, data: np.ndarray) -> None:
        c_vec = generate_random_complex_vector(self.l3.dim, seed=hash(key) & 0xFFFFFFFF)
        self.l3.store(c_vec)
        self._all_keys[key] = data.ravel()
        entry = TierEntry(key=key, data=None, tier=HolographicTier.L3, encoding=c_vec)
        self.l3_entries[key] = entry
        self._evict_l3_if_needed()

    def _store_l4(self, key: str, data: np.ndarray) -> None:
        if self.mmap_backed:
            path = self._mmap_path(key)
            np.save(path, data)
        entry = TierEntry(key=key, data=data, tier=HolographicTier.L4)
        self.l4_entries[key] = entry
        self._evict_l4_if_needed()

    def retrieve(self, key: str) -> Optional[np.ndarray]:
        with self._lock:
            if key in self.l1_entries:
                entry = self.l1_entries[key]
                entry.frequency += 1
                entry.last_access = time.monotonic()
                result = self.l1.recall(self._all_keys[key], top_k=1)
                if result:
                    return result[0][1]
                return entry.encoding

            if key in self.l2_entries:
                entry = self.l2_entries[key]
                entry.frequency += 1
                entry.last_access = time.monotonic()
                if entry.frequency >= self.promotion_threshold:
                    self._promote(key, HolographicTier.L2, HolographicTier.L1)
                result = self.l2.recall(self._all_keys[key], top_k=1)
                if result:
                    return result[0][1]
                return entry.encoding

            if key in self.l3_entries:
                entry = self.l3_entries[key]
                entry.frequency += 1
                entry.last_access = time.monotonic()
                if entry.frequency >= self.promotion_threshold:
                    self._promote(key, HolographicTier.L3, HolographicTier.L2)
                result = self.l3.recall(self._all_keys[key], top_k=1)
                if result:
                    return result[0][1]
                return None

            if key in self.l4_entries:
                entry = self.l4_entries[key]
                entry.frequency += 1
                entry.last_access = time.monotonic()
                if self.mmap_backed:
                    path = self._mmap_path(key)
                    try:
                        return np.load(path, mmap_mode="r")
                    except Exception:
                        pass
                return entry.data

            return None

    def _promote(self, key: str, from_tier: int, to_tier: int) -> None:
        if from_tier == HolographicTier.L3 and to_tier == HolographicTier.L2:
            if key in self.l3_entries:
                entry = self.l3_entries.pop(key)
                entry.tier = HolographicTier.L2
                k_vec = unit_vector(
                    entry.encoding.real
                    if np.iscomplexobj(entry.encoding)
                    else entry.encoding
                )
                self.l2.store(k_vec)
                self.l2_entries[key] = entry
                self._promotion_count += 1

        elif from_tier == HolographicTier.L2 and to_tier == HolographicTier.L1:
            if key in self.l2_entries:
                entry = self.l2_entries.pop(key)
                entry.tier = HolographicTier.L1
                k_vec = unit_vector(entry.encoding)
                self.l1.store(k_vec)
                self.l1_entries[key] = entry
                self._promotion_count += 1

    def _demote(self, key: str) -> None:
        if key in self.l1_entries:
            entry = self.l1_entries.pop(key)
            entry.tier = HolographicTier.L2
            self.l2.store(unit_vector(entry.encoding))
            self.l2_entries[key] = entry
            self._demotion_count += 1
        elif key in self.l2_entries:
            entry = self.l2_entries.pop(key)
            entry.tier = HolographicTier.L3
            c_vec = generate_random_complex_vector(
                self.l3.dim, seed=hash(key) & 0xFFFFFFFF
            )
            self.l3.store(c_vec)
            self.l3_entries[key] = entry
            self._demotion_count += 1

    def _evict_l1_if_needed(self) -> None:
        while len(self.l1_entries) > self.l1_capacity:
            oldest = min(self.l1_entries.values(), key=lambda e: e.last_access)
            self._demote(oldest.key)

    def _evict_l2_if_needed(self) -> None:
        while len(self.l2_entries) > self.l2_capacity:
            oldest = min(self.l2_entries.values(), key=lambda e: e.last_access)
            self._demote(oldest.key)

    def _evict_l3_if_needed(self) -> None:
        while len(self.l3_entries) > self.l3_capacity:
            oldest = min(self.l3_entries.values(), key=lambda e: e.last_access)
            self.l3_entries.pop(oldest.key, None)

    def _evict_l4_if_needed(self) -> None:
        while len(self.l4_entries) > self.l4_capacity:
            oldest = min(self.l4_entries.values(), key=lambda e: e.last_access)
            self.l4_entries.pop(oldest.key, None)

    def apply_decay_all(self) -> None:
        self.l1.apply_decay()
        self.l2.apply_decay()
        self.l3.apply_decay()

    def clear(self) -> None:
        with self._lock:
            self.l1.clear()
            self.l2.clear()
            self.l3.clear()
            self.l1_entries.clear()
            self.l2_entries.clear()
            self.l3_entries.clear()
            self.l4_entries.clear()
            self._all_keys.clear()
            self._promotion_count = 0
            self._demotion_count = 0

    def get_stats(self) -> dict:
        return {
            "l1_entries": len(self.l1_entries),
            "l2_entries": len(self.l2_entries),
            "l3_entries": len(self.l3_entries),
            "l4_entries": len(self.l4_entries),
            "l1_capacity": self.l1_capacity,
            "l2_capacity": self.l2_capacity,
            "l3_capacity": self.l3_capacity,
            "l4_capacity": self.l4_capacity,
            "promotion_count": self._promotion_count,
            "demotion_count": self._demotion_count,
            "l1_dim": self.l1_dim,
            "l2_dim": self.l2_dim,
            "l3_dim": self.l3_dim,
            "mmap_backed": self.mmap_backed,
        }


# ═══════════════════════════════════════════════════════════════════════════
# HolographicEngine
# ═══════════════════════════════════════════════════════════════════════════


class HolographicEngine:
    def __init__(
        self,
        kv_dim: int = 1024,
        weight_dim: int = 2048,
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
        self._start_time = time.monotonic()

    def memory_status(self) -> dict:
        return {
            "engine": "Holographic Memory Engine v1.0",
            "uptime_seconds": time.monotonic() - self._start_time,
            "associative_memory": self.associative_memory.get_stats(),
            "kv_cache": self.kv_cache.get_stats(),
            "weight_store": self.weight_store.get_stats(),
            "hierarchy": self.hierarchy.get_stats(),
        }

    def memory_reset(self) -> dict:
        self.kv_cache.clear()
        self.weight_store.clear()
        self.hierarchy.clear()
        self.associative_memory.clear()
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
    ) -> Optional[np.ndarray]:
        return self.weight_store.recall_weight(layer_idx, weight_name, row_idx)

    def associate(self, key: np.ndarray, value: Optional[np.ndarray] = None) -> int:
        return self.associative_memory.store(key, value)

    def recall(self, key: np.ndarray, top_k: int = 1) -> list:
        return self.associative_memory.recall(key, top_k=top_k)

    def clear(self) -> None:
        self.memory_reset()

    def get_stats(self) -> dict:
        return self.memory_status()
