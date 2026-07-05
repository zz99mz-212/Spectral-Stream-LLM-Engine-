from __future__ import annotations

import numpy as np
import os
import threading
import time
import mmap
import json
import struct
import tempfile
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Callable
from dataclasses import dataclass, field


class SSDWeightStreamer:
    def __init__(
        self,
        model_path: str,
        ram_cache_layers: int = 4,
        prefetch_threads: int = 2,
        use_mmap: bool = True,
    ):
        self.model_path = Path(model_path)
        self.ram_cache_layers = ram_cache_layers
        self.use_mmap = use_mmap

        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=prefetch_threads)
        self._prefetch_futures: list[threading.Future] = []

        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._cache_access: dict[str, int] = {}

        self._index: dict[str, tuple] = {}
        self._mmap_handles: dict[str, np.memmap] = {}
        self._built_index = False

        self.hits = 0
        self.misses = 0
        self.prefetches = 0
        self.evictions = 0

    def build_index(self):
        if self._built_index:
            return
        p = self.model_path
        if p.is_dir():
            self._build_npy_index(p)
        elif p.suffix == ".gguf":
            self._build_gguf_index(p)
        else:
            self._build_npy_index(p)
        self._built_index = True

    def _build_npy_index(self, directory: Path):
        for fpath in sorted(directory.rglob("*.npy")):
            rel = fpath.relative_to(directory)
            name = str(rel.with_suffix("")).replace("/", ".")
            try:
                arr = np.load(str(fpath), mmap_mode=None)
                self._index[name] = (str(fpath), 0, arr.dtype, arr.shape)
            except Exception:
                pass

    def _build_gguf_index(self, gguf_path: Path):
        from gguf import GGUFReader

        reader = GGUFReader(str(gguf_path))
        for t in reader.tensors:
            name = t.name
            data = t.data
            if hasattr(data, "offset"):
                self._index[name] = (
                    str(gguf_path),
                    data.offset,
                    data.dtype,
                    data.shape,
                )
            else:
                self._index[name] = (str(gguf_path), 0, data.dtype, data.shape)
            self._mmap_handles[name] = self._mmap_tensor_from_reader(t, gguf_path)

    def _mmap_tensor_from_reader(self, tensor, gguf_path: Path) -> Optional[np.memmap]:
        try:
            data = tensor.data
            if hasattr(data, "offset"):
                raw = np.memmap(str(gguf_path), dtype=np.uint8, mode="r")
                offset = data.offset
                nbytes = int(np.prod(data.shape)) * np.dtype(data.dtype).itemsize
                return np.memmap(
                    str(gguf_path),
                    dtype=data.dtype,
                    mode="r",
                    offset=offset,
                    shape=data.shape,
                )
        except Exception:
            return None

    def list_tensors(self) -> list[str]:
        self.build_index()
        return list(self._index.keys())

    def get_weight(self, tensor_name: str) -> np.ndarray:
        with self._lock:
            if tensor_name in self._cache:
                self._cache.move_to_end(tensor_name)
                self._cache_access[tensor_name] = (
                    self._cache_access.get(tensor_name, 0) + 1
                )
                self.hits += 1
                return self._cache[tensor_name]
            self.misses += 1

        arr = self._load_from_storage(tensor_name)
        if arr is None:
            raise KeyError(
                f"Tensor '{tensor_name}' not found in model path {self.model_path}"
            )

        with self._lock:
            self._cache[tensor_name] = arr
            self._cache_access[tensor_name] = 1
            self._evict_if_needed()

        return arr

    def get_layer_weights(self, layer_idx: int) -> dict[str, np.ndarray]:
        self.build_index()
        prefix = f"blk.{layer_idx}."
        result = {}
        for name in list(self._index.keys()):
            if name.startswith(prefix):
                result[name] = self.get_weight(name)
        return result

    def _load_from_storage(self, tensor_name: str) -> Optional[np.ndarray]:
        self.build_index()
        info = self._index.get(tensor_name)
        if info is None:
            return None
        fpath, offset, dtype, shape = info

        if tensor_name in self._mmap_handles and self.use_mmap:
            mm = self._mmap_handles[tensor_name]
            return np.array(mm, copy=True)

        if not os.path.exists(fpath):
            return None

        if fpath.endswith(".npy"):
            return np.load(fpath)
        elif fpath.endswith(".gguf"):
            try:
                raw = np.memmap(
                    fpath, dtype=dtype, mode="r", offset=offset, shape=shape
                )
                return np.array(raw, copy=True)
            except Exception:
                raw = np.memmap(
                    fpath,
                    dtype=np.uint8,
                    mode="r",
                    offset=offset,
                    shape=(int(np.prod(shape)) * np.dtype(dtype).itemsize,),
                )
                return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()
        else:
            raw = np.memmap(fpath, dtype=dtype, mode="r", offset=offset, shape=shape)
            return np.array(raw, copy=True)

    def _evict_if_needed(self):
        while len(self._cache) > self.ram_cache_layers:
            key, val = self._cache.popitem(last=False)
            self.evictions += 1

    def prefetch(self, tensor_names: list[str]):
        def _load(name):
            try:
                arr = self._load_from_storage(name)
                if arr is not None:
                    with self._lock:
                        self._cache[name] = arr
                        self._cache_access[name] = 0
                        self._evict_if_needed()
                    self.prefetches += 1
            except Exception:
                pass

        for name in tensor_names:
            with self._lock:
                if name in self._cache:
                    continue
            if name in self._index:
                f = self._executor.submit(_load, name)
                self._prefetch_futures.append(f)

        self._prefetch_futures = [f for f in self._prefetch_futures if not f.done()]

    def prefetch_next_layer(
        self, current_layer: int, hd_prediction: Optional[list[int]] = None
    ):
        if hd_prediction:
            next_layers = hd_prediction[:3]
        else:
            next_layers = [current_layer + 1, current_layer + 2]

        tensors_to_prefetch = []
        for lidx in next_layers:
            self.build_index()
            prefix = f"blk.{lidx}."
            for name in self._index:
                if name.startswith(prefix):
                    with self._lock:
                        if name not in self._cache:
                            tensors_to_prefetch.append(name)
        if tensors_to_prefetch:
            self.prefetch(tensors_to_prefetch)

    def set_hd_engine(self, hd_engine):
        self._hd_engine = hd_engine

    def evict(self, tensor_name: str):
        with self._lock:
            self._cache.pop(tensor_name, None)
            self._cache_access.pop(tensor_name, None)

    def evict_layer(self, layer_idx: int):
        prefix = f"blk.{layer_idx}."
        with self._lock:
            keys = [k for k in self._cache if k.startswith(prefix)]
            for k in keys:
                self._cache.pop(k, None)
                self._cache_access.pop(k, None)
                self.evictions += 1

    def clear_cache(self):
        with self._lock:
            self._cache.clear()
            self._cache_access.clear()

    def get_stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / max(total, 1),
            "cache_size": len(self._cache),
            "max_cache_layers": self.ram_cache_layers,
            "evictions": self.evictions,
            "prefetches": self.prefetches,
            "tensors_indexed": len(self._index),
        }

    def close(self):
        self._executor.shutdown(wait=False)
        self._mmap_handles.clear()
        self._cache.clear()


@dataclass
class KVCacheEntry:
    key: np.ndarray
    value: np.ndarray
    position: int
    timestamp: int
    access_count: int = 0
    compressed: bool = False
    compressed_key: Optional[np.ndarray] = None
    compressed_value: Optional[np.ndarray] = None
    on_disk: bool = False
    disk_path: str = ""


class KVCacheTieredStorage:
    def __init__(
        self,
        dim: int,
        max_hot: int = 4096,
        max_warm: int = 65536,
        hot_precision: str = "f16",
        warm_precision: str = "dct_q4",
        disk_cache_dir: Optional[str] = None,
    ):
        self.dim = dim
        self.max_hot = max_hot
        self.max_warm = max_warm
        self.hot_precision = hot_precision
        self.warm_precision = warm_precision

        self._hot: list[KVCacheEntry] = []
        self._warm: dict[int, KVCacheEntry] = {}
        self._cold: dict[int, KVCacheEntry] = {}
        self._position_map: dict[int, KVCacheEntry] = {}

        self._global_step = 0
        self._disk_dir = (
            Path(disk_cache_dir)
            if disk_cache_dir
            else Path(os.path.join(tempfile.gettempdir(), "kv_cache_tiered"))
        )
        self._disk_dir.mkdir(parents=True, exist_ok=True)

        self.hot_hits = 0
        self.warm_hits = 0
        self.cold_hits = 0
        self.promotions = 0
        self.demotions = 0

        self._dct_available = True
        try:
            from numpy.fft import fft, ifft
        except ImportError:
            self._dct_available = False

    def store(self, key: np.ndarray, value: np.ndarray, position: int):
        self._global_step += 1

        entry = KVCacheEntry(
            key=np.asarray(key, dtype=np.float32),
            value=np.asarray(value, dtype=np.float32),
            position=position,
            timestamp=self._global_step,
        )

        if len(self._hot) >= self.max_hot:
            self._demote_oldest_hot()

        self._hot.append(entry)
        self._position_map[position] = entry

    def _demote_oldest_hot(self):
        if not self._hot:
            return
        oldest = min(self._hot, key=lambda e: e.timestamp)
        self._hot.remove(oldest)

        if len(self._warm) < self.max_warm:
            self._compress_entry(oldest)
            self._warm[oldest.position] = oldest
            oldest.compressed = True
        else:
            self._demote_warm_to_cold()
            self._compress_entry(oldest)
            self._warm[oldest.position] = oldest
            oldest.compressed = True

        self.demotions += 1

    def _compress_entry(self, entry: KVCacheEntry):
        if self.warm_precision == "dct_q4":
            k_comp = self._dct_compress(entry.key, bits=4)
            v_comp = self._dct_compress(entry.value, bits=4)
            entry.compressed_key = k_comp
            entry.compressed_value = v_comp
            entry.key = np.array([], dtype=np.float32)
            entry.value = np.array([], dtype=np.float32)

    def _dct_compress(self, vec: np.ndarray, bits: int = 4) -> np.ndarray:
        if not self._dct_available:
            return vec
        n = vec.shape[-1]
        x = vec.astype(np.float64)
        dct = np.zeros_like(x)
        for i in range(n):
            dct[..., i] = np.sum(
                x * np.cos(np.pi * (np.arange(n) + 0.5) * i / n), axis=-1
            )
        dct *= np.sqrt(2.0 / n)
        scale = np.max(np.abs(dct), axis=-1, keepdims=True)
        scale = np.where(scale < 1e-10, 1.0, scale)
        normalized = dct / scale
        max_q = (1 << (bits - 1)) - 1
        quantized = np.clip(np.round(normalized * max_q), -max_q, max_q).astype(np.int8)
        meta = np.array([scale[..., 0]], dtype=np.float32)
        return np.concatenate([meta.ravel(), quantized.ravel().astype(np.float32)])

    def _dct_decompress(self, compressed: np.ndarray, original_shape) -> np.ndarray:
        if not self._dct_available:
            return compressed
        n = original_shape[-1]
        scale = compressed[0]
        quantized = compressed[1:].reshape(original_shape).astype(np.float64)
        dct = quantized / scale
        idct = np.zeros_like(dct)
        for i in range(n):
            idct += dct[..., i] * np.cos(np.pi * (np.arange(n) + 0.5) * i / n)
        idct *= np.sqrt(2.0 / n)
        return idct.astype(np.float32)

    def _demote_warm_to_cold(self):
        if not self._warm:
            return
        oldest_pos = min(self._warm.keys(), key=lambda p: self._warm[p].timestamp)
        entry = self._warm.pop(oldest_pos)
        fname = self._disk_dir / f"kv_cold_{entry.position}.npz"
        np.savez_compressed(
            str(fname),
            key=entry.compressed_key if entry.compressed else entry.key,
            value=entry.compressed_value if entry.compressed else entry.value,
            compressed=entry.compressed,
        )
        entry.on_disk = True
        entry.disk_path = str(fname)
        entry.compressed_key = None
        entry.compressed_value = None
        self._cold[entry.position] = entry

    def retrieve(self, position: int) -> Optional[tuple]:
        self._global_step += 1
        entry = self._position_map.get(position)
        if entry is None:
            return None

        if not entry.on_disk and not entry.compressed:
            self.hot_hits += 1
            entry.access_count += 1
            entry.timestamp = self._global_step
            return (entry.key, entry.value)

        if entry.compressed and not entry.on_disk:
            self.warm_hits += 1
            entry.access_count += 1
            entry.timestamp = self._global_step
            k = self._dct_decompress(entry.compressed_key, (self.dim,))
            v = self._dct_decompress(entry.compressed_value, (self.dim,))
            self._promote(entry)
            return (k, v)

        if entry.on_disk:
            self.cold_hits += 1
            entry.access_count += 1
            entry.timestamp = self._global_step
            data = np.load(entry.disk_path)
            if entry.compressed:
                k = self._dct_decompress(data["key"], (self.dim,))
                v = self._dct_decompress(data["value"], (self.dim,))
            else:
                k = data["key"]
                v = data["value"]
            entry.key = k
            entry.value = v
            entry.on_disk = False
            entry.disk_path = ""
            self._promote(entry)
            return (k, v)

        return None

    def _promote(self, entry: KVCacheEntry):
        if entry.compressed:
            self._warm.pop(entry.position, None)
        if entry.on_disk:
            self._cold.pop(entry.position, None)
            entry.on_disk = False

        if entry.compressed:
            entry.key = self._dct_decompress(entry.compressed_key, (self.dim,))
            entry.value = self._dct_decompress(entry.compressed_value, (self.dim,))
            entry.compressed = False
            entry.compressed_key = None
            entry.compressed_value = None

        if len(self._hot) >= self.max_hot:
            self._demote_oldest_hot()

        self._hot.append(entry)
        self.promotions += 1

    def query(self, query_vec: np.ndarray, top_k: int = 10) -> list[tuple[int, float]]:
        results = []
        q = np.asarray(query_vec, dtype=np.float32)
        q_norm = np.linalg.norm(q) + 1e-10

        all_entries = list(self._hot) + list(self._warm.values())
        for entry in all_entries:
            if len(entry.key) > 0:
                k = entry.key
            elif entry.compressed_key is not None:
                k = self._dct_decompress(entry.compressed_key, (self.dim,))
            else:
                continue
            sim = float(np.dot(q, k)) / (q_norm * np.linalg.norm(k) + 1e-10)
            results.append((entry.position, sim))

        for entry in self._cold.values():
            if entry.on_disk and os.path.exists(entry.disk_path):
                data = np.load(entry.disk_path)
                if entry.compressed:
                    k = self._dct_decompress(data["key"], (self.dim,))
                else:
                    k = data["key"]
                sim = float(np.dot(q, k)) / (q_norm * np.linalg.norm(k) + 1e-10)
                results.append((entry.position, sim))

        results.sort(key=lambda x: -x[1])
        return results[:top_k]

    def hit_rate(self) -> dict:
        total_hot = self.hot_hits
        total_warm = self.warm_hits
        total_cold = self.cold_hits
        total = total_hot + total_warm + total_cold
        return {
            "hot_hits": total_hot,
            "warm_hits": total_warm,
            "cold_hits": total_cold,
            "total_hits": total,
            "hot_rate": total_hot / max(total, 1),
            "warm_rate": total_warm / max(total, 1),
            "cold_rate": total_cold / max(total, 1),
            "promotions": self.promotions,
            "demotions": self.demotions,
            "hot_size": len(self._hot),
            "warm_size": len(self._warm),
            "cold_size": len(self._cold),
        }

    def clear(self):
        self._hot.clear()
        self._warm.clear()
        for entry in self._cold.values():
            if entry.disk_path and os.path.exists(entry.disk_path):
                try:
                    os.remove(entry.disk_path)
                except OSError:
                    pass
        self._cold.clear()
        self._position_map.clear()
        self._global_step = 0
        self.hot_hits = 0
        self.warm_hits = 0
        self.cold_hits = 0
        self.promotions = 0
        self.demotions = 0

    def close(self):
        self.clear()


class PredictiveWeightPrefetcher:
    def __init__(self, n_layers: int, hd_engine=None):
        self.n_layers = n_layers
        self.hd_engine = hd_engine

        self._access_history: list[tuple[int, int]] = []
        self._transition_counts: dict[tuple[int, int], int] = {}
        self._layer_frequency: dict[int, int] = {}
        self._token_to_layers: dict[int, dict[int, int]] = {}

    def predict_next_layers(self, context_tokens: list[int]) -> list[int]:
        if not context_tokens:
            return list(range(min(3, self.n_layers)))

        last_token = context_tokens[-1]
        predictions = []

        if self.hd_engine is not None:
            try:
                hv = self.hd_engine.hd.ensure_token_vector(last_token)
            except AttributeError:
                hv = None
            if hv is not None:
                pred = self._predict_from_hd(last_token, hv)
                predictions.extend(pred)

        token_preds = self._token_to_layers.get(last_token, {})
        sorted_by_freq = sorted(token_preds.items(), key=lambda x: -x[1])
        for lidx, _ in sorted_by_freq[:3]:
            if lidx not in predictions:
                predictions.append(lidx)

        if len(predictions) < 3:
            most_freq = sorted(self._layer_frequency.items(), key=lambda x: -x[1])
            for lidx, _ in most_freq:
                if lidx not in predictions:
                    predictions.append(lidx)
                    if len(predictions) >= 3:
                        break

        if not predictions:
            predictions = [0, 1, 2]

        return predictions[:3]

    def _predict_from_hd(self, token_id: int, hv: np.ndarray) -> list[int]:
        if not self._access_history:
            return []
        layer_scores = np.zeros(self.n_layers, dtype=np.float32)
        for past_tok, past_layer in self._access_history[-100:]:
            try:
                past_hv = self.hd_engine.hd.ensure_token_vector(past_tok)
            except AttributeError:
                continue
            sim = float(np.count_nonzero((hv > 0) == (past_hv > 0))) / len(hv)
            if sim > 0.6:
                layer_scores[past_layer] += sim
        top = np.argsort(-layer_scores)[:3]
        return [int(t) for t in top if layer_scores[t] > 0]

    def train(self, context_tokens: list[int], accessed_layers: list[int]):
        if not context_tokens or not accessed_layers:
            return
        last_token = context_tokens[-1]
        for lidx in accessed_layers:
            self._access_history.append((last_token, lidx))
            self._layer_frequency[lidx] = self._layer_frequency.get(lidx, 0) + 1
            if last_token not in self._token_to_layers:
                self._token_to_layers[last_token] = {}
            self._token_to_layers[last_token][lidx] = (
                self._token_to_layers[last_token].get(lidx, 0) + 1
            )

    def get_stats(self) -> dict:
        return {
            "n_layers": self.n_layers,
            "access_history_len": len(self._access_history),
            "unique_tokens_tracked": len(self._token_to_layers),
            "layer_frequency": dict(
                sorted(self._layer_frequency.items(), key=lambda x: -x[1])[:10]
            ),
        }


class StreamingGGUFModel:
    def __init__(
        self,
        gguf_path: str,
        lazy: bool = True,
        ram_cache_tensors: int = 8,
        prefetch_threads: int = 2,
    ):
        self.gguf_path = gguf_path
        self.lazy = lazy
        self.ram_cache_tensors = ram_cache_tensors

        self._reader = None
        self._tensor_index: dict[str, dict] = {}
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._lock = threading.Lock()
        self._prefetch_executor = ThreadPoolExecutor(max_workers=prefetch_threads)
        self._mmaps: dict[str, np.memmap] = {}
        self._loaded_metadata = False

        self.hits = 0
        self.misses = 0

        if not lazy:
            self._load_metadata()
            self._load_all_tensors()

    def _load_metadata(self):
        if self._loaded_metadata:
            return
        from gguf import GGUFReader, GGMLQuantizationType, dequantize

        self._reader = GGUFReader(self.gguf_path)
        self._parse_metadata()
        self._index_tensors()
        self._loaded_metadata = True

    def _parse_metadata(self):
        def get_field(key: str, default=None):
            field = self._reader.fields.get(key)
            if field is None:
                return default
            data = field.parts[1] if len(field.parts) > 1 else field.parts[-1]
            if hasattr(data, "shape") and data.ndim == 0:
                return data.item()
            if isinstance(data, bytes):
                return data.decode("utf-8", errors="replace")
            if isinstance(data, np.ndarray):
                return data.item() if data.size == 1 else data.tolist()
            return data

        arch = get_field("general.architecture", "")
        self.architecture = str(arch) if arch else "unknown"

        safe = {
            "llama": "llama",
            "mistral": "llama",
            "gemma": "llama",
            "granitehybrid": "llama",
            "qwen35moe": "llama",
            "gemma4": "llama",
        }

        def arch_safe(a: str) -> str:
            return safe.get(a.lower(), a.lower())

        safe_arch = arch_safe(self.architecture)

        raw_nlayers = get_field(f"{safe_arch}.block_count")
        self.n_layers = int(raw_nlayers) if raw_nlayers else 0

        raw_hidden = get_field(f"{safe_arch}.embedding_length")
        self.hidden_dim = int(raw_hidden) if raw_hidden else 0

        raw_ff = get_field(f"{safe_arch}.feed_forward_length")
        self.ff_dim = int(raw_ff) if raw_ff else 0

        raw_nhead = get_field(f"{safe_arch}.attention.head_count")
        self.n_heads = int(raw_nhead) if raw_nhead else 0

        raw_nkv = get_field(f"{safe_arch}.attention.head_count_kv")
        self.n_kv_heads = int(raw_nkv) if raw_nkv else self.n_heads

        raw_vocab = get_field(f"{safe_arch}.vocab_size")
        self.vocab_size = int(raw_vocab) if raw_vocab else 0

        raw_ctx = get_field(f"{safe_arch}.context_length")
        self.context_length = int(raw_ctx) if raw_ctx else 2048

        raw_rope = get_field(f"{safe_arch}.rope.dimension_count")
        self.rope_dim = int(raw_rope) if raw_rope else 0

        raw_eps = get_field(f"{safe_arch}.attention.layer_norm_rms_epsilon")
        self.rms_norm_eps = float(raw_eps) if raw_eps else 1e-6

        self.head_dim = self.hidden_dim // self.n_heads if self.n_heads > 0 else 0

    def _index_tensors(self):
        for t in self._reader.tensors:
            name = t.name
            data = t.data
            dtype = data.dtype
            shape = data.shape
            if hasattr(data, "offset"):
                offset = data.offset
            else:
                offset = 0
            self._tensor_index[name] = {
                "dtype": dtype,
                "shape": shape,
                "offset": offset,
            }

    def get_tensor(self, name: str) -> Optional[np.ndarray]:
        with self._lock:
            if name in self._cache:
                self._cache.move_to_end(name)
                self.hits += 1
                return self._cache[name]
            self.misses += 1

        if not self._loaded_metadata:
            self._load_metadata()

        info = self._tensor_index.get(name)
        if info is None:
            return None

        arr = self._mmap_tensor(name, info)
        if arr is None:
            arr = self._load_tensor_direct(name, info)

        if arr is not None:
            with self._lock:
                self._cache[name] = arr
                while len(self._cache) > self.ram_cache_tensors:
                    self._cache.popitem(last=False)

        return arr

    def _mmap_tensor(self, name: str, info: dict) -> Optional[np.ndarray]:
        if name in self._mmaps:
            return np.array(self._mmaps[name], copy=True)
        try:
            offset = info["offset"]
            shape = info["shape"]
            dtype = info["dtype"]
            if offset > 0 and os.path.exists(self.gguf_path):
                mm = np.memmap(
                    self.gguf_path, dtype=dtype, mode="r", offset=offset, shape=shape
                )
                self._mmaps[name] = mm
                return np.array(mm, copy=True)
        except Exception:
            pass
        return None

    def _load_tensor_direct(self, name: str, info: dict) -> Optional[np.ndarray]:
        try:
            from gguf import GGMLQuantizationType, dequantize
        except ImportError:
            dequantize = None

        for t in self._reader.tensors:
            if t.name == name:
                data = t.data
                tensor_type = t.tensor_type if hasattr(t, "tensor_type") else None
                if (
                    data.dtype != np.float32
                    and dequantize is not None
                    and tensor_type is not None
                ):
                    try:
                        qtype = GGMLQuantizationType(tensor_type)
                        if qtype != GGMLQuantizationType.F32:
                            return dequantize(data, qtype)
                    except Exception:
                        pass
                return np.asarray(data, dtype=np.float32).copy()
        return None

    def get_layer_tensor(self, layer_idx: int, name: str) -> Optional[np.ndarray]:
        return self.get_tensor(f"blk.{layer_idx}.{name}")

    def get_layer_weights(self, layer_idx: int) -> dict[str, np.ndarray]:
        prefix = f"blk.{layer_idx}."
        result = {}
        if not self._loaded_metadata:
            self._load_metadata()
        for tname in self._tensor_index:
            if tname.startswith(prefix):
                arr = self.get_tensor(tname)
                if arr is not None:
                    result[tname] = arr
        return result

    def _load_all_tensors(self):
        if not self._loaded_metadata:
            self._load_metadata()
        for name in self._tensor_index:
            self.get_tensor(name)

    def list_tensors(self) -> list[str]:
        if not self._loaded_metadata:
            self._load_metadata()
        return list(self._tensor_index.keys())

    def summary(self) -> str:
        if not self._loaded_metadata:
            self._load_metadata()
        total = self.hits + self.misses
        return (
            f"StreamingGGUFModel: {self.gguf_path}\n"
            f"  Architecture: {self.architecture}\n"
            f"  Layers: {self.n_layers}\n"
            f"  Hidden dim: {self.hidden_dim}\n"
            f"  FF dim: {self.ff_dim}\n"
            f"  Heads: {self.n_heads} (KV: {self.n_kv_heads})\n"
            f"  Head dim: {self.head_dim}\n"
            f"  Vocab size: {self.vocab_size}\n"
            f"  Tensors: {len(self._tensor_index)}\n"
            f"  Cache hit rate: {self.hits / max(total, 1):.3f}"
        )

    def close(self):
        self._prefetch_executor.shutdown(wait=False)
        self._mmaps.clear()
        self._cache.clear()
        self._reader = None
