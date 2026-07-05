from __future__ import annotations

import gc
import logging
import os
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

try:
    import psutil as _psutil

    def _get_process_memory_mb() -> float:
        return _psutil.Process().memory_info().rss / 1048576.0
except ImportError:

    def _get_process_memory_mb() -> float:
        return 0.0


logger = logging.getLogger(__name__)

import numpy as np

from spectralstream.core.math_primitives import (
    spectral_entropy,
    compute_mse,
    compute_snr,
)
from spectralstream.kv_cache.core import (
    EPS,
    KVCacheConfig,
    KVCacheEntry,
    QualityMetrics,
)
from spectralstream.kv_cache.compressor import CacheCompressor
from spectralstream.kv_cache.eviction import (
    EvictionPolicy,
    SpectralEviction,
    H2OEviction,
    SlidingWindowEviction,
    StreamingLLMEviction,
    ResonanceEviction,
    EntropyEviction,
    ImportanceScoring,
    PredictiveEviction,
    ClusteringEviction,
    TopologicalEviction,
    ReinforcementLearningEviction,
    HybridEviction,
    StalenessAwareEviction,
    AccessPatternEviction,
    EntropyGradient,
)


class KVCacheManager:
    def __init__(self, config: KVCacheConfig):
        self.config = config
        self._lock = threading.RLock()
        self._caches: Dict[int, OrderedDict] = {}
        self._compressed: Dict[int, Dict[int, Tuple[dict, dict]]] = {}
        self._scores: Dict[int, Dict[int, float]] = {}
        self._quality_metrics: Dict[int, Dict[int, QualityMetrics]] = {}
        self._global_step = 0
        self._total_bytes = 0
        self._hit_count = 0
        self._miss_count = 0
        self._hit_rate: float = 0.0
        self._hit_rate_history: List[float] = []
        self._cache_pressure: float = 0.0
        self._compression_level: int = 0
        self._annealing_temp: float = config.annealing_temp
        self._prev_cache: Dict[int, Dict[int, Tuple[np.ndarray, np.ndarray]]] = {}

        # Cached retrieve_all results
        self._retrieve_all_cache: Dict[int, Tuple[np.ndarray, np.ndarray, int]] = {}
        self._retrieve_all_version: int = 0

        self._store_gc_counter = 0

        if config.enable_tiering:
            os.makedirs(config.ssd_cache_path, exist_ok=True)

        policy_map: Dict[str, Any] = {
            "spectral": SpectralEviction,
            "h2o": H2OEviction,
            "sliding": SlidingWindowEviction,
            "streaming": StreamingLLMEviction,
            "resonance": ResonanceEviction,
            "entropy": EntropyEviction,
            "importance": ImportanceScoring,
            "predictive": PredictiveEviction,
            "clustering": ClusteringEviction,
            "topological": TopologicalEviction,
            "rl": ReinforcementLearningEviction,
            "hybrid": HybridEviction,
            "staleness": StalenessAwareEviction,
            "access_pattern": AccessPatternEviction,
            "entropy_gradient": EntropyGradient,
        }
        pol_cls = policy_map.get(config.eviction_policy, SpectralEviction)
        kwargs: Dict[str, Any] = {}
        if config.eviction_policy in ("sliding", "streaming"):
            kwargs["window_size"] = config.window_size
        if config.eviction_policy == "h2o":
            kwargs["heavy_hitter_frac"] = config.heavy_hitter_frac
        if config.eviction_policy == "streaming":
            kwargs["sink_tokens"] = 4
        self._eviction_policy: EvictionPolicy = (
            pol_cls(**kwargs) if kwargs else pol_cls()
        )
        self._ssd_paths: Dict[int, Dict[int, str]] = {}

    def _get_cache(self, layer_idx: int) -> OrderedDict:
        if layer_idx not in self._caches:
            self._caches[layer_idx] = OrderedDict()
            self._compressed[layer_idx] = {}
            self._scores[layer_idx] = {}
            self._quality_metrics[layer_idx] = {}
        return self._caches[layer_idx]

    def store(
        self,
        layer_idx: int,
        key: np.ndarray,
        value: np.ndarray,
        position: int,
        attention_scores: Optional[np.ndarray] = None,
    ):
        with self._lock:
            cache = self._get_cache(layer_idx)
            self._global_step += 1
            self._retrieve_all_version += 1
            method = self._select_compression_method(layer_idx)
            use_compression = method != "none"
            cache_dtype = np.float16
            entry = KVCacheEntry(
                key=key.astype(cache_dtype)
                if not use_compression
                else np.array([0], dtype=cache_dtype),
                value=value.astype(cache_dtype)
                if not use_compression
                else np.array([0], dtype=cache_dtype),
                position=position,
                layer_idx=layer_idx,
                score=0.0,
                compressed=use_compression,
            )
            if use_compression:
                k_data, v_data = CacheCompressor.compress(method, key, value)
                self._compressed[layer_idx][position] = (k_data, v_data)
                entry.compressed = True
                entry.compressed_size = self._estimate_compressed_size(k_data, v_data)
                if self.config.quality_tracking:
                    try:
                        k_dec, v_dec = CacheCompressor.decompress(
                            method, k_data, v_data
                        )
                        orig_k = key.astype(np.float32).ravel()
                        orig_v = value.astype(np.float32).ravel()
                        min_len = min(len(orig_k), len(k_dec.ravel()))
                        if min_len > 0:
                            mse_k = float(
                                compute_mse(orig_k[:min_len], k_dec.ravel()[:min_len])
                            )
                            snr_k = float(
                                compute_snr(orig_k[:min_len], k_dec.ravel()[:min_len])
                            )
                            mse_v = float(
                                compute_mse(orig_v[:min_len], v_dec.ravel()[:min_len])
                            )
                            snr_v = float(
                                compute_snr(orig_v[:min_len], v_dec.ravel()[:min_len])
                            )
                            ratio = (orig_k.nbytes + orig_v.nbytes) / (
                                entry.compressed_size + EPS
                            )
                            ent = spectral_entropy(key.ravel())
                            qm = QualityMetrics(
                                mse=(mse_k + mse_v) / 2.0,
                                snr=(snr_k + snr_v) / 2.0,
                                psnr=(snr_k + snr_v) / 2.0,
                                compression_ratio=ratio,
                                method=method,
                                bits_per_element=entry.compressed_size
                                * 8
                                / max(key.size, 1),
                                entropy=ent,
                                timestamp=time.time(),
                            )
                            entry.quality = qm
                            self._quality_metrics[layer_idx][position] = qm
                    except (ValueError, TypeError, RuntimeError):
                        pass
            else:
                entry.key = key.astype(np.float16)
                entry.value = value.astype(np.float16)
            cache[position] = entry
            self._scores[layer_idx][position] = 0.0
            self._total_bytes += entry.byte_size()
            if self.config.enable_tiering:
                self._tier_check(layer_idx, position)
            self._enforce_limit(layer_idx)
            self._auto_tune()
            if self.config.progressive_compression:
                self._progressive_compress(layer_idx, position)
            self._store_gc_counter += 1
            if self._store_gc_counter % 100 == 0:
                gc.collect()

    def _select_compression_method(self, layer_idx: int) -> str:
        method = self.config.compression_method
        if method == "none" or not self.config.adaptive_compression:
            return method
        pressure = self._get_cache_pressure()
        self._cache_pressure = pressure
        if pressure > 0.8:
            preferred = [
                "fwht_int4",
                "quantile",
                "e8_lattice",
                "adaptive_bitwidth",
                "residual_vq",
            ]
        elif pressure > 0.6:
            preferred = [
                "fwht_int8",
                "dct_sparse",
                "lloyd_max",
                "product_quantization",
                "sparse_attention",
            ]
        elif pressure > 0.4:
            preferred = ["hadamard", "spectral", "wavelet", "svd", "low_rank"]
        else:
            preferred = [method]
        return preferred[0] if method not in preferred else method

    def _get_cache_pressure(self) -> float:
        stats = self.get_cache_size()
        limit_gb = self.config.cache_size_limit_gb
        mem_gb = stats["memory_bytes"] / (1 << 30)
        return min(1.0, mem_gb / max(limit_gb * 0.01, 1e-10))

    def _estimate_compressed_size(self, k_data: Any, v_data: Any) -> int:
        if isinstance(k_data, bytes) and isinstance(v_data, bytes):
            return max(len(k_data) + len(v_data), 1)
        size = 0
        for d in (k_data, v_data):
            for val in d.values() if isinstance(d, dict) else []:
                if isinstance(val, np.ndarray):
                    size += val.nbytes
                elif (
                    isinstance(val, list)
                    and len(val) > 0
                    and isinstance(val[0], np.ndarray)
                ):
                    size += sum(v.nbytes for v in val)
                elif isinstance(val, dict):
                    size += self._estimate_compressed_size(val, {})
                else:
                    size += 8
        return max(size, 1)

    def _load_from_ssd(
        self, layer_idx: int, position: int
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        ssd_paths = self._ssd_paths.get(layer_idx, {})
        path = ssd_paths.get(position)
        if path is not None and os.path.exists(path):
            try:
                data = np.load(path)
                return data["k"], data["v"]
            except (OSError, ValueError, RuntimeError):
                pass
        return None

    def retrieve(
        self, layer_idx: int, start_pos: int, end_pos: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        with self._lock:
            cache = self._get_cache(layer_idx)
            head_dim = self.config.head_dim
            positions = np.arange(start_pos, end_pos, dtype=np.int64)
            present = np.array([pos in cache for pos in positions])

            # Check SSD for missing positions
            if not present.all():
                for i, pos in enumerate(positions):
                    if i < len(present) and not present[i]:
                        ssd_result = self._load_from_ssd(layer_idx, int(pos))
                        if ssd_result is not None:
                            k_ssd, v_ssd = ssd_result
                            entry = KVCacheEntry(
                                key=k_ssd,
                                value=v_ssd,
                                position=int(pos),
                                layer_idx=layer_idx,
                                compressed=False,
                            )
                            cache[int(pos)] = entry
                            present[i] = True

            if not present.any():
                self._miss_count += len(positions)
                self._update_hit_rate()
                return np.zeros((0, head_dim), dtype=np.float16), np.zeros(
                    (0, head_dim), dtype=np.float16
                )
            present_positions = positions[present]
            keys_list: List[np.ndarray] = []
            vals_list: List[np.ndarray] = []
            for pos in present_positions:
                entry = cache[int(pos)]
                use_compression = entry.compressed and int(pos) in self._compressed.get(
                    layer_idx, {}
                )
                if use_compression:
                    k_data, v_data = self._compressed[layer_idx][int(pos)]
                    method = self._select_compression_method(layer_idx)
                    if method == "none":
                        method = self.config.compression_method
                    k, v = CacheCompressor.decompress(method, k_data, v_data)
                    k = k.ravel()[:head_dim]
                    v = v.ravel()[:head_dim]
                    del k_data, v_data
                else:
                    k = (
                        entry.key.ravel()[:head_dim]
                        if entry.key.size >= head_dim
                        else np.pad(entry.key.ravel(), (0, head_dim - entry.key.size))[
                            :head_dim
                        ]
                    )
                    v = (
                        entry.value.ravel()[:head_dim]
                        if entry.value.size >= head_dim
                        else np.pad(
                            entry.value.ravel(), (0, head_dim - entry.value.size)
                        )[:head_dim]
                    )
                keys_list.append(k.astype(np.float16) if k.dtype != np.float16 else k)
                vals_list.append(v.astype(np.float16) if v.dtype != np.float16 else v)
                self._scores[layer_idx][int(pos)] = (
                    self._scores[layer_idx].get(int(pos), 0.0) + 1.0
                )
            self._hit_count += len(present_positions)
            self._miss_count += len(positions) - len(present_positions)
            self._update_hit_rate()
            if not keys_list:
                return np.zeros((0, head_dim), dtype=np.float16), np.zeros(
                    (0, head_dim), dtype=np.float16
                )
            result = np.stack(keys_list), np.stack(vals_list)
            if self._store_gc_counter % 50 == 0:
                gc.collect()
            return result

    def retrieve_all(self, layer_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        cache = self._get_cache(layer_idx)
        hd = self.config.head_dim
        if not cache:
            return np.zeros((0, hd), dtype=np.float16), np.zeros(
                (0, hd), dtype=np.float16
            )

        # Check cache
        cached = self._retrieve_all_cache.get(layer_idx)
        if cached is not None:
            cached_version = cached[2]
            if cached_version == self._retrieve_all_version:
                return cached[0], cached[1]

        positions = sorted(cache.keys())
        if not positions:
            return np.zeros((0, hd), dtype=np.float32), np.zeros(
                (0, hd), dtype=np.float32
            )
        result = self.retrieve(layer_idx, positions[0], positions[-1] + 1)
        self._retrieve_all_cache[layer_idx] = (
            result[0],
            result[1],
            self._retrieve_all_version,
        )
        return result

    def evict(self, layer_idx: int, n_entries: int = 1):
        with self._lock:
            cache = self._get_cache(layer_idx)
            if len(cache) == 0:
                return
            entries_list = list(cache.values())
            n = min(n_entries, len(entries_list))
            freed = 0
            for _ in range(n):
                idx = self._eviction_policy.select_eviction(entries_list)
                if idx < 0 or idx >= len(entries_list):
                    break
                evict_entry = entries_list.pop(idx)
                pos = evict_entry.position
                policy_name = self.config.eviction_policy
                freed += evict_entry.byte_size()
                logger.debug(
                    "Evicting layer=%d pos=%d policy=%s policy_type=%s",
                    layer_idx,
                    pos,
                    policy_name,
                    type(self._eviction_policy).__name__,
                )
                del cache[pos]
                self._compressed.get(layer_idx, {}).pop(pos, None)
                self._scores.get(layer_idx, {}).pop(pos, None)
                self._quality_metrics.get(layer_idx, {}).pop(pos, None)
                if (
                    self.config.enable_tiering
                    and layer_idx in self._ssd_paths
                    and pos in self._ssd_paths[layer_idx]
                ):
                    try:
                        os.remove(self._ssd_paths[layer_idx][pos])
                    except OSError:
                        pass
                    del self._ssd_paths[layer_idx][pos]
            self._total_bytes = max(0, self._total_bytes - freed)
            if isinstance(self._eviction_policy, ReinforcementLearningEviction):
                self._eviction_policy.update(hit=False)

    def clear(self):
        with self._lock:
            self._caches.clear()
            self._compressed.clear()
            self._scores.clear()
            self._quality_metrics.clear()
            self._global_step = 0
            self._total_bytes = 0
            self._ssd_paths.clear()
            self._hit_count = 0
            self._miss_count = 0
            self._hit_rate = 0.0
            self._hit_rate_history.clear()
            self._retrieve_all_cache.clear()
            self._retrieve_all_version = 0

    def get_cache_size(self) -> Dict[str, float]:
        with self._lock:
            total_entries = sum(len(c) for c in self._caches.values())
            mem_bytes = 0
            for layer_cache in self._caches.values():
                for entry in layer_cache.values():
                    mem_bytes += entry.byte_size()
            compressed_entries = sum(len(v) for v in self._compressed.values())
            return {
                "total_entries": total_entries,
                "total_layers": len(self._caches),
                "memory_bytes": mem_bytes,
                "memory_mb": mem_bytes / 1e6,
                "memory_gb": mem_bytes / (1 << 30),
                "compressed_entries": compressed_entries,
                "cache_limit_gb": self.config.cache_size_limit_gb,
                "cache_pressure": self._cache_pressure,
                "hit_rate": self._hit_rate,
                "global_step": self._global_step,
            }

    def get_cache_memory_usage(self) -> float:
        with self._lock:
            total = 0.0
            for layer_idx in self._caches:
                for entry in self._caches[layer_idx].values():
                    total += entry.byte_size()
            return total / (1024**3)

    def get_quality_report(self, layer_idx: int) -> Dict[str, Any]:
        with self._lock:
            metrics = self._quality_metrics.get(layer_idx, {})
            if not metrics:
                return {}
            mse_vals = [m.mse for m in metrics.values()]
            snr_vals = [m.snr for m in metrics.values()]
            ratio_vals = [m.compression_ratio for m in metrics.values()]
            ent_vals = [m.entropy for m in metrics.values()]
            return {
                "layer": layer_idx,
                "entries_tracked": len(metrics),
                "avg_mse": float(np.mean(mse_vals)),
                "avg_snr": float(np.mean(snr_vals)),
                "avg_compression_ratio": float(np.mean(ratio_vals)),
                "avg_entropy": float(np.mean(ent_vals)),
            }

    def prefetch_upcoming(self, n_tokens: int = 32):
        with self._lock:
            if not self.config.prefetch_enabled:
                return
            for layer_idx in self._caches:
                cache = self._caches[layer_idx]
                if not cache:
                    continue
                max_pos = max(p for p in cache.keys())
                for pos in range(max_pos + 1, max_pos + 1 + n_tokens):
                    if pos not in cache:
                        dummy = np.zeros(self.config.head_dim, dtype=np.float32)
                        self.store(layer_idx, dummy, dummy, pos)
                        self.evict(layer_idx, 1)

    def _update_hit_rate(self):
        total = self._hit_count + self._miss_count
        if total > 0:
            self._hit_rate = self._hit_count / total
            self._hit_rate_history.append(self._hit_rate)
            window = self.config.hit_rate_window
            if len(self._hit_rate_history) > window:
                self._hit_rate_history = self._hit_rate_history[-window:]

    def _auto_tune(self):
        if not self.config.auto_tune or self._global_step % 50 != 0:
            return
        if len(self._hit_rate_history) > 10:
            recent = float(np.mean(self._hit_rate_history[-10:]))
            if recent < 0.3 and self.config.quantize_bits > 4:
                self.config.quantize_bits = max(4, self.config.quantize_bits - 1)
            elif recent > 0.8 and self.config.quantize_bits < 16:
                self.config.quantize_bits = min(16, self.config.quantize_bits + 1)
            # Map quantize_bits to a real compression method
            if (
                self.config.compression_method == "none"
                and self.config.quantize_bits <= 8
            ):
                self.config.compression_method = "fwht_int8"
            elif (
                self.config.compression_method == "none"
                and self.config.quantize_bits <= 4
            ):
                self.config.compression_method = "fwht_int4"
        if self.config.simulated_annealing_eviction:
            self._annealing_temp = max(
                self.config.annealing_min_temp,
                self._annealing_temp * self.config.annealing_cooling_rate,
            )

    def _progressive_compress(self, layer_idx: int, position: int):
        if not self.config.progressive_compression:
            return
        cache = self._caches.get(layer_idx, {})
        if len(cache) < 10:
            return
        pressure = self._get_cache_pressure()
        if pressure > 0.7 and position in self._compressed.get(layer_idx, {}):
            k_data, v_data = self._compressed[layer_idx][position]
            method = self._select_compression_method(layer_idx)
            entry = cache.get(position)
            if entry is not None and entry.key.size > 1:
                k, v = entry.key, entry.value
                new_kd, new_vd = CacheCompressor.compress(method, k, v)
                old_size = self._estimate_compressed_size(k_data, v_data)
                new_size = self._estimate_compressed_size(new_kd, new_vd)
                if new_size < old_size * 0.8:
                    self._compressed[layer_idx][position] = (new_kd, new_vd)

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            total_entries = sum(len(c) for c in self._caches.values())
            total_layers = len(self._caches)
            mem_bytes = 0
            for layer_cache in self._caches.values():
                for entry in layer_cache.values():
                    mem_bytes += entry.byte_size()
            process_mem_mb = _get_process_memory_mb()
            total = self._hit_count + self._miss_count
            hit_rate = self._hit_count / max(total, 1)
            return {
                "total_entries": total_entries,
                "total_layers": total_layers,
                "cache_memory_mb": mem_bytes / 1048576.0,
                "cache_memory_gb": mem_bytes / (1 << 30),
                "cache_limit_gb": self.config.cache_size_limit_gb,
                "cache_pressure": self._get_cache_pressure(),
                "process_memory_mb": process_mem_mb,
                "hit_count": self._hit_count,
                "miss_count": self._miss_count,
                "hit_rate": round(hit_rate, 6),
                "global_step": self._global_step,
                "eviction_policy": self.config.eviction_policy,
                "compression_method": self.config.compression_method,
            }

    def __len__(self) -> int:
        return sum(len(c) for c in self._caches.values())

    def _tier_check(self, layer_idx: int, position: int):
        if not self.config.enable_tiering:
            return
        ssd_limit_mb = self.config.cache_size_limit_gb * 1024 * 0.5
        current = self.get_cache_size()
        if current["memory_mb"] > ssd_limit_mb:
            cache = self._caches.get(layer_idx, {})
            if len(cache) < 3:
                return
            candidates = sorted(
                cache.values(),
                key=lambda e: self._scores[layer_idx].get(e.position, 0.0),
            )
            evict = candidates[0]
            layer_dir = os.path.join(self.config.ssd_cache_path, f"layer_{layer_idx}")
            os.makedirs(layer_dir, exist_ok=True)
            path = os.path.join(layer_dir, f"pos_{position}.npz")
            np.savez_compressed(path, k=evict.key, v=evict.value)
            if layer_idx not in self._ssd_paths:
                self._ssd_paths[layer_idx] = {}
            self._ssd_paths[layer_idx][evict.position] = path

    def _enforce_limit(self, layer_idx: int):
        limit_bytes = self.config.cache_size_limit_gb * (1 << 30)
        if self._total_bytes <= limit_bytes:
            return
        target = self._total_bytes - int(limit_bytes * 0.8)
        while self._total_bytes > target:
            # Evict from the layer with the most entries (O(n) scan)
            max_layer = layer_idx
            max_entries = 0
            for lidx, lc in self._caches.items():
                if len(lc) > max_entries:
                    max_entries = len(lc)
                    max_layer = lidx
            if max_entries <= 4:
                break
            n = max(1, max_entries // 10)
            self.evict(max_layer, n)

    def compute_coherence(self) -> Dict[str, float]:
        if not self.config.cache_coherence_monitoring:
            return {}
        with self._lock:
            total_coherence = 0.0
            n_layers = 0
            for layer_idx, cache in self._caches.items():
                if len(cache) < 2:
                    continue
                keys = np.array([e.key.ravel()[:32] for e in cache.values()])
                sim = keys @ keys.T
                # Normalize each row by the norms of the original key vectors, not the sim matrix
                key_norms = np.linalg.norm(keys, axis=-1, keepdims=True)
                sim_norm = sim / (key_norms @ key_norms.T + EPS)
                upper = np.triu(sim_norm, k=1)
                coherence = (
                    float(upper[upper != 0].mean()) if np.any(upper != 0) else 1.0
                )
                total_coherence += coherence
                n_layers += 1
            avg_c = total_coherence / max(n_layers, 1)
            return {
                "avg_coherence": avg_c,
                "n_layers_checked": n_layers,
                "fragmentation": 1.0 - avg_c,
            }
