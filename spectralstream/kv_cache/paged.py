from __future__ import annotations

import warnings as _warnings

_warnings.warn(
    "spectralstream.kv_cache.paged is deprecated. "
    "Use spectralstream.kv_cache.KVCacheManager instead.",
    DeprecationWarning,
    stacklevel=2,
)

import os
import time
from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass, field
from enum import IntEnum
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
from spectralstream.kv_cache.spectral import (
    BAND_HIGH,
    BAND_NORMAL,
    BAND_LOW,
    BAND_COMPRESSION,
    CacheEntry,
    ResonanceTracker,
    SpectralKVCache,
)

try:
    from spectralstream.core.math_primitives import hrr_bind, hrr_unbind, hrr_bundle

    _HAS_HRR = True
except ImportError:
    _HAS_HRR = False


def _now() -> float:
    return time.monotonic()


def _deterministic_hash(s: str) -> int:
    h = 0x811C9DC5
    for c in s.encode():
        h = ((h * 0x01000193) ^ c) & 0xFFFFFFFF
    return h


@dataclass
class PageInfo:
    logical_page: int
    physical_block: int
    num_tokens: int
    ref_count: int = 1
    is_shared: bool = False
    timestamp: float = 0.0
    attention_score: float = 0.0


@dataclass
class RadixNode:
    prefix: tuple
    children: dict = field(default_factory=dict)
    parent: Optional[RadixNode] = None
    depth: int = 0
    page_refs: list[int] = field(default_factory=list)
    is_leaf: bool = True
    access_time: float = 0.0
    access_count: int = 0
    lru_key: float = 0.0


class PagedKVCache:
    def __init__(
        self,
        dim: int = 128,
        tokens_per_page: int = 16,
        num_physical_blocks: int = 1024,
        num_heads: int = 8,
    ):
        self.dim = dim
        self.tokens_per_page = tokens_per_page
        self.num_physical_blocks = num_physical_blocks
        self.num_heads = num_heads

        self.page_table: dict[int, PageInfo] = {}
        self.physical_blocks: dict[int, dict] = {}
        self._next_logical = 0
        self._free_blocks: list[int] = list(range(num_physical_blocks))
        self._logical_to_physical: dict[int, int] = {}
        self._physical_ref_counts: dict[int, int] = {}
        self._step = 0

        for pb in range(num_physical_blocks):
            self.physical_blocks[pb] = {
                "k": np.zeros((tokens_per_page, dim), dtype=np.float32),
                "v": np.zeros((tokens_per_page, dim), dtype=np.float32),
                "valid_mask": np.zeros(tokens_per_page, dtype=bool),
            }
            self._physical_ref_counts[pb] = 0

        self.hits = 0
        self.misses = 0
        self.defrag_count = 0
        self.cow_count = 0

    def _alloc_physical_block(self) -> Optional[int]:
        if not self._free_blocks:
            return None
        pb = self._free_blocks.pop()
        block = self.physical_blocks[pb]
        block["valid_mask"][:] = False
        block["k"][:] = 0.0
        block["v"][:] = 0.0
        self._physical_ref_counts[pb] = 0
        return pb

    def _free_physical_block(self, pb: int):
        if pb not in self._free_blocks:
            self._free_blocks.append(pb)
        self._physical_ref_counts[pb] = 0

    def alloc_pages(self, num_tokens: int) -> list[int]:
        num_pages = (num_tokens + self.tokens_per_page - 1) // self.tokens_per_page
        logical_pages = []
        for _ in range(num_pages):
            pb = self._alloc_physical_block()
            if pb is None:
                raise MemoryError("No free physical blocks")
            lp = self._next_logical
            self._next_logical += 1
            info = PageInfo(
                logical_page=lp,
                physical_block=pb,
                num_tokens=0,
                ref_count=1,
                timestamp=_now(),
            )
            self.page_table[lp] = info
            self._logical_to_physical[lp] = pb
            self._physical_ref_counts[pb] = 1
            logical_pages.append(lp)
        return logical_pages

    def write_page(
        self,
        logical_page: int,
        tokens_k: np.ndarray,
        tokens_v: np.ndarray,
        offset: int = 0,
    ):
        if logical_page not in self.page_table:
            return
        info = self.page_table[logical_page]
        pb = info.physical_block
        block = self.physical_blocks[pb]
        n = min(len(tokens_k), self.tokens_per_page - offset)
        block["k"][offset : offset + n] = tokens_k[:n]
        block["v"][offset : offset + n] = tokens_v[:n]
        block["valid_mask"][offset : offset + n] = True
        info.num_tokens = max(info.num_tokens, offset + n)

    def read_page(self, logical_page: int) -> Optional[tuple[np.ndarray, np.ndarray]]:
        if logical_page not in self.page_table:
            self.misses += 1
            return None
        info = self.page_table[logical_page]
        pb = info.physical_block
        block = self.physical_blocks[pb]
        valid = block["valid_mask"]
        n = int(valid.sum())
        self.hits += 1
        return (block["k"][:n].copy(), block["v"][:n].copy())

    def read_contiguous(
        self, start_logical: int, num_pages: int
    ) -> Optional[tuple[np.ndarray, np.ndarray]]:
        ks, vs = [], []
        for lp in range(start_logical, start_logical + num_pages):
            r = self.read_page(lp)
            if r is None:
                return None
            k, v = r
            ks.append(k)
            vs.append(v)
        return (np.concatenate(ks, axis=0), np.concatenate(vs, axis=0))

    def copy_on_write(self, src_logical: int) -> int:
        if src_logical not in self.page_table:
            return -1
        src_info = self.page_table[src_logical]
        pb = self._alloc_physical_block()
        if pb is None:
            raise MemoryError("No free blocks for COW")
        src_block = self.physical_blocks[src_info.physical_block]
        dst_block = self.physical_blocks[pb]
        dst_block["k"][:] = src_block["k"]
        dst_block["v"][:] = src_block["v"]
        dst_block["valid_mask"][:] = src_block["valid_mask"]
        lp = self._next_logical
        self._next_logical += 1
        info = PageInfo(
            logical_page=lp,
            physical_block=pb,
            num_tokens=src_info.num_tokens,
            ref_count=1,
            is_shared=True,
            timestamp=_now(),
        )
        self.page_table[lp] = info
        self._logical_to_physical[lp] = pb
        self._physical_ref_counts[pb] = 1
        self.cow_count += 1
        return lp

    def share_pages(self, logical_pages: list[int]) -> list[int]:
        new_pages = []
        for lp in logical_pages:
            new_pages.append(self.copy_on_write(lp))
        return new_pages

    def free_page(self, logical_page: int):
        if logical_page not in self.page_table:
            return
        info = self.page_table[logical_page]
        pb = info.physical_block
        self._physical_ref_counts[pb] -= 1
        if self._physical_ref_counts[pb] <= 0:
            self._free_physical_block(pb)
        del self.page_table[logical_page]
        self._logical_to_physical.pop(logical_page, None)

    def defragment(self):
        if len(self._free_blocks) >= self.num_physical_blocks * 0.3:
            return
        sparse = [
            (lp, info)
            for lp, info in self.page_table.items()
            if info.num_tokens < self.tokens_per_page // 2
        ]
        sparse.sort(key=lambda x: x[1].num_tokens)
        freed = 0
        for lp, info in sparse:
            if freed > len(sparse) // 2:
                break
            pb = self._alloc_physical_block()
            if pb is None:
                break
            src_block = self.physical_blocks[info.physical_block]
            dst_block = self.physical_blocks[pb]
            dst_block["k"][: info.num_tokens] = src_block["k"][: info.num_tokens]
            dst_block["v"][: info.num_tokens] = src_block["v"][: info.num_tokens]
            dst_block["valid_mask"][: info.num_tokens] = True
            self._physical_ref_counts[info.physical_block] -= 1
            if self._physical_ref_counts[info.physical_block] <= 0:
                self._free_physical_block(info.physical_block)
            info.physical_block = pb
            self._physical_ref_counts[pb] = 1
            self._logical_to_physical[lp] = pb
            freed += 1
        self.defrag_count += 1

    def num_used_blocks(self) -> int:
        return self.num_physical_blocks - len(self._free_blocks)

    def utilization(self) -> float:
        return self.num_used_blocks() / max(self.num_physical_blocks, 1)

    def clear(self):
        self.page_table.clear()
        self._logical_to_physical.clear()
        self._next_logical = 0
        self._free_blocks = list(range(self.num_physical_blocks))
        for pb in range(self.num_physical_blocks):
            self.physical_blocks[pb]["valid_mask"][:] = False
            self.physical_blocks[pb]["k"][:] = 0.0
            self.physical_blocks[pb]["v"][:] = 0.0
            self._physical_ref_counts[pb] = 0
        self.hits = 0
        self.misses = 0
        self.defrag_count = 0
        self.cow_count = 0

    def cache_summary(self) -> dict:
        return {
            "type": "PagedKVCache",
            "dim": self.dim,
            "tokens_per_page": self.tokens_per_page,
            "num_physical_blocks": self.num_physical_blocks,
            "num_logical_pages": len(self.page_table),
            "free_blocks": len(self._free_blocks),
            "utilization": self.utilization(),
            "hits": self.hits,
            "misses": self.misses,
            "defrag_count": self.defrag_count,
            "cow_count": self.cow_count,
        }


class RadixTreeCache:
    def __init__(self, kv_cache: PagedKVCache):
        self.kv_cache = kv_cache
        self.root = RadixNode(prefix=(), depth=0)
        self._node_id_counter = 0
        self._node_ids: dict[int, RadixNode] = {0: self.root}
        self._sequence_roots: dict[str, RadixNode] = {}
        self._lru_list: list[RadixNode] = []
        self.max_nodes = 4096
        self.hits = 0
        self.misses = 0

    def _new_node_id(self) -> int:
        nid = self._node_id_counter
        self._node_id_counter += 1
        return nid

    def insert_sequence(self, token_ids: list[int], seq_id: str = "") -> list[int]:
        node = self.root
        i = 0
        while i < len(token_ids):
            matched = False
            for child_token, child_node in node.children.items():
                child_prefix = child_node.prefix
                match_len = 0
                for j in range(min(len(child_prefix), len(token_ids) - i)):
                    if child_prefix[j] == token_ids[i + j]:
                        match_len += 1
                    else:
                        break
                if match_len > 0:
                    if match_len < len(child_prefix):
                        self._split_node(child_node, match_len)
                    node = child_node
                    i += match_len
                    matched = True
                    break
            if not matched:
                remaining = tuple(token_ids[i:])
                new_node = RadixNode(
                    prefix=remaining,
                    parent=node,
                    depth=node.depth + len(remaining),
                    is_leaf=True,
                    access_time=_now(),
                )
                pages = self.kv_cache.alloc_pages(len(remaining))
                new_node.page_refs = pages
                node.children[remaining[0]] = new_node
                node.is_leaf = False
                nid = self._new_node_id()
                self._node_ids[nid] = new_node
                self._lru_list.append(new_node)
                if len(self._node_ids) > self.max_nodes:
                    self._evict_lru()
                node = new_node
                break

        if seq_id:
            self._sequence_roots[seq_id] = node
        all_pages = []
        n = node
        while n and n is not self.root:
            all_pages = n.page_refs + all_pages
            n = n.parent
        return all_pages

    def _split_node(self, node: RadixNode, split_pos: int):
        common = node.prefix[:split_pos]
        suffix = node.prefix[split_pos:]
        new_node = RadixNode(
            prefix=suffix,
            parent=node,
            depth=node.depth + split_pos,
            children=node.children,
            page_refs=node.page_refs[:],
            is_leaf=node.is_leaf,
            access_time=node.access_time,
        )
        nid = self._new_node_id()
        self._node_ids[nid] = new_node
        if new_node in self._lru_list:
            self._lru_list.remove(new_node)
            self._lru_list.append(new_node)
        node.prefix = common
        node.children = {suffix[0]: new_node}
        node.page_refs = []
        node.is_leaf = False
        new_node.parent = node

    def longest_prefix_match(self, token_ids: list[int]) -> tuple[RadixNode, int]:
        node = self.root
        match_len = 0
        i = 0
        while i < len(token_ids):
            found = False
            for child_token, child_node in node.children.items():
                child_prefix = child_node.prefix
                m = 0
                for j in range(min(len(child_prefix), len(token_ids) - i)):
                    if child_prefix[j] == token_ids[i + j]:
                        m += 1
                    else:
                        break
                if m > 0:
                    node = child_node
                    i += m
                    match_len = i
                    found = True
                    break
            if not found:
                break
        node.access_time = _now()
        node.access_count += 1
        if node.is_leaf:
            self.hits += 1
        else:
            self.misses += 1
        return node, match_len

    def get_kv_pages(self, node: RadixNode) -> list[int]:
        pages = []
        while node and node is not self.root:
            pages = node.page_refs + pages
            node = node.parent
        return pages

    def get_sequence_pages(self, token_ids: list[int]) -> list[int]:
        node, match_len = self.longest_prefix_match(token_ids)
        if match_len < len(token_ids):
            remaining = token_ids[match_len:]
            pages = self.kv_cache.alloc_pages(len(remaining))
            node, _ = self._extend_node(node, remaining, pages)
        return self.get_kv_pages(node)

    def _extend_node(
        self, node: RadixNode, suffix: list[int], pages: list[int]
    ) -> tuple[RadixNode, int]:
        new_node = RadixNode(
            prefix=tuple(suffix),
            parent=node,
            depth=node.depth + len(suffix),
            page_refs=pages,
            is_leaf=True,
            access_time=_now(),
        )
        if suffix:
            node.children[suffix[0]] = new_node
        node.is_leaf = False
        nid = self._new_node_id()
        self._node_ids[nid] = new_node
        self._lru_list.append(new_node)
        if len(self._node_ids) > self.max_nodes:
            self._evict_lru()
        return new_node, len(suffix)

    def _evict_lru(self):
        if not self._lru_list:
            return
        leaf = min(
            (n for n in self._lru_list if n.is_leaf and n is not self.root),
            key=lambda n: n.access_time,
            default=None,
        )
        if leaf is None:
            leaf = self._lru_list[0]
        if leaf in self._lru_list:
            self._lru_list.remove(leaf)
        for pid in self._node_ids:
            if self._node_ids[pid] is leaf:
                del self._node_ids[pid]
                break
        for page in leaf.page_refs:
            self.kv_cache.free_page(page)
        if leaf.parent:
            for key, child in list(leaf.parent.children.items()):
                if child is leaf:
                    del leaf.parent.children[key]
                    break
            if not leaf.parent.children:
                leaf.parent.is_leaf = True

    def auto_defrag(self):
        self.kv_cache.defragment()

    def cache_summary(self) -> dict:
        return {
            "type": "RadixTreeCache",
            "num_nodes": len(self._node_ids),
            "max_nodes": self.max_nodes,
            "hits": self.hits,
            "misses": self.misses,
            "root_children": len(self.root.children),
        }


class TierLevel(IntEnum):
    HBM = 0
    DRAM = 1
    COMPRESSED = 2
    HOLOGRAPHIC = 3
    SSD_FAST = 4
    SSD_SLOW = 5


TIER_LATENCY = {
    TierLevel.HBM: 0.000_001,
    TierLevel.DRAM: 0.000_050,
    TierLevel.COMPRESSED: 0.000_100,
    TierLevel.HOLOGRAPHIC: 0.001_000,
    TierLevel.SSD_FAST: 0.005_000,
    TierLevel.SSD_SLOW: 0.020_000,
}

TIER_CAPACITY = {
    TierLevel.HBM: 1.0,
    TierLevel.DRAM: 10.0,
    TierLevel.COMPRESSED: 40.0,
    TierLevel.HOLOGRAPHIC: 100.0,
    TierLevel.SSD_FAST: 1000.0,
    TierLevel.SSD_SLOW: 10000.0,
}


@dataclass
class TierEntry:
    position: int
    head_idx: int
    layer_idx: int
    k_data: Optional[np.ndarray] = None
    v_data: Optional[np.ndarray] = None
    compressed_k: Optional[np.ndarray] = None
    compressed_v: Optional[np.ndarray] = None
    compression_meta: Optional[dict] = None
    tier: TierLevel = TierLevel.DRAM
    importance: float = 0.0
    access_count: int = 0
    timestamp: float = 0.0
    attention_score: float = 0.0
    frequency: int = 1


class TieredKVCache:
    def __init__(
        self,
        dim: int = 128,
        max_total_entries: int = 65536,
        tier_capacities: Optional[dict[TierLevel, int]] = None,
        dim_per_head: int = 128,
        num_heads: int = 8,
        num_layers: int = 32,
        use_holographic: bool = True,
        use_spectral_compression: bool = True,
        ssd_dir: str = "/tmp/spectralstream_kv_tiered",
    ):
        self.dim = dim
        self.dim_per_head = dim_per_head
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_total_entries = max_total_entries
        self.use_holographic = use_holographic
        self.use_spectral_compression = use_spectral_compression
        self.ssd_dir = ssd_dir

        if tier_capacities is None:
            self.tier_capacities = {
                TierLevel.HBM: 1024,
                TierLevel.DRAM: 8192,
                TierLevel.COMPRESSED: 16384,
                TierLevel.HOLOGRAPHIC: 32768,
                TierLevel.SSD_FAST: 65536,
                TierLevel.SSD_SLOW: 262144,
            }
        else:
            self.tier_capacities = tier_capacities

        self.tiers: dict[TierLevel, dict[int, TierEntry]] = {
            t: OrderedDict() for t in TierLevel
        }
        self._all_entries: dict[int, TierEntry] = {}
        self._next_entry_id = 0
        self._entry_id_by_pos: dict[int, int] = {}

        self._holographic_memories: dict[TierLevel, Any] = {}
        if use_holographic and _HAS_HRR and dim >= 128:
            for t in (TierLevel.HOLOGRAPHIC,):
                from spectralstream.core.math_primitives import hrr_bind as _hb

                self._holographic_memories[t] = None

        os.makedirs(ssd_dir, exist_ok=True)

        self.hits = 0
        self.misses = 0
        self.promotions = 0
        self.demotions = 0
        self.prefetches = 0
        self._step = 0

    def store(
        self,
        key: np.ndarray,
        value: np.ndarray,
        position: int,
        head_idx: int = 0,
        layer_idx: int = 0,
        initial_tier: TierLevel = TierLevel.DRAM,
        importance: float = 1.0,
    ) -> int:
        self._step += 1
        eid = self._next_entry_id
        self._next_entry_id += 1
        self._entry_id_by_pos[position] = eid

        entry = TierEntry(
            position=position,
            head_idx=head_idx,
            layer_idx=layer_idx,
            tier=initial_tier,
            importance=importance,
            timestamp=_now(),
        )

        if initial_tier == TierLevel.HBM:
            entry.k_data = key.astype(np.float32)
            entry.v_data = value.astype(np.float32)
        elif initial_tier == TierLevel.DRAM:
            entry.k_data = key.astype(np.float32)
            entry.v_data = value.astype(np.float32)
        elif initial_tier == TierLevel.COMPRESSED and self.use_spectral_compression:
            k_comp, meta_k = self._dct_compress(key)
            v_comp, meta_v = self._dct_compress(value)
            entry.compressed_k = k_comp
            entry.compressed_v = v_comp
            entry.compression_meta = {"k": meta_k, "v": meta_v}
        elif initial_tier == TierLevel.HOLOGRAPHIC and self.use_holographic:
            entry.k_data = key.astype(np.float32)
            entry.v_data = value.astype(np.float32)
        else:
            entry.k_data = key.astype(np.float32)
            entry.v_data = value.astype(np.float32)

        self._all_entries[eid] = entry
        self.tiers[initial_tier][eid] = entry
        self._evict_if_needed(initial_tier)
        return eid

    def retrieve(
        self, position: int, head_idx: int = 0, layer_idx: int = 0
    ) -> Optional[tuple[np.ndarray, np.ndarray]]:
        eid = self._entry_id_by_pos.get(position)
        if eid is None or eid not in self._all_entries:
            self.misses += 1
            return None

        entry = self._all_entries[eid]
        entry.access_count += 1
        entry.frequency += 1
        entry.timestamp = _now()

        result = self._read_entry(entry)
        if result is not None:
            self.hits += 1
        else:
            self.misses += 1
        return result

    def _read_entry(self, entry: TierEntry) -> Optional[tuple[np.ndarray, np.ndarray]]:
        if entry.tier in (TierLevel.HBM, TierLevel.DRAM) and entry.k_data is not None:
            self._maybe_promote(entry)
            return (entry.k_data, entry.v_data)

        if entry.tier == TierLevel.COMPRESSED:
            if entry.compressed_k is not None:
                k = self._dct_decompress(
                    entry.compressed_k, entry.compression_meta["k"]
                )
                v = self._dct_decompress(
                    entry.compressed_v, entry.compression_meta["v"]
                )
                entry.k_data = k
                entry.v_data = v
                entry.tier = TierLevel.DRAM
                self.promotions += 1
                return (k, v)

        if entry.tier == TierLevel.HOLOGRAPHIC and self.use_holographic:
            if entry.k_data is not None:
                return (entry.k_data, entry.v_data)
            return None

        if entry.tier in (TierLevel.SSD_FAST, TierLevel.SSD_SLOW):
            path = self._ssd_path(entry.position, entry.head_idx, entry.layer_idx)
            try:
                data = np.load(path)
                entry.k_data = data["k"]
                entry.v_data = data["v"]
                entry.tier = TierLevel.DRAM
                self.promotions += 1
                return (entry.k_data, entry.v_data)
            except Exception:
                return None

        return None

    def _maybe_promote(self, entry: TierEntry):
        if entry.tier != TierLevel.DRAM and entry.tier != TierLevel.HBM:
            return
        if entry.frequency > 5 and entry.tier == TierLevel.DRAM:
            if len(self.tiers[TierLevel.HBM]) < self.tier_capacities[TierLevel.HBM]:
                old_tier = entry.tier
                self.tiers[old_tier].pop(
                    next(e for e, v in self._all_entries.items() if v is entry), None
                )
                entry.tier = TierLevel.HBM
                self.tiers[TierLevel.HBM][id(entry)] = entry
                self.promotions += 1

    def promote(self, position: int, target: TierLevel = TierLevel.HBM):
        eid = self._entry_id_by_pos.get(position)
        if eid is None:
            return
        entry = self._all_entries[eid]
        if entry.tier == target:
            return
        result = self._read_entry(entry)
        if result is None:
            return
        k, v = result
        old_tier = entry.tier
        if old_tier in self.tiers and id(entry) in self.tiers[old_tier]:
            del self.tiers[old_tier][id(entry)]
        entry.k_data = k
        entry.v_data = v
        entry.tier = target
        self.tiers[target][id(entry)] = entry
        self.promotions += 1

    def demote(self, position: int, target: TierLevel = TierLevel.COMPRESSED):
        eid = self._entry_id_by_pos.get(position)
        if eid is None:
            return
        entry = self._all_entries[eid]
        if entry.tier >= target:
            return
        old_tier = entry.tier
        if old_tier in self.tiers:
            for k in list(self.tiers[old_tier].keys()):
                if self.tiers[old_tier][k] is entry:
                    del self.tiers[old_tier][k]
                    break
        if target == TierLevel.COMPRESSED and self.use_spectral_compression:
            k = entry.k_data if entry.k_data is not None else np.zeros(self.dim)
            v = entry.v_data if entry.v_data is not None else np.zeros(self.dim)
            k_comp, mk = self._dct_compress(k)
            v_comp, mv = self._dct_compress(v)
            entry.compressed_k = k_comp
            entry.compressed_v = v_comp
            entry.compression_meta = {"k": mk, "v": mv}
            entry.k_data = None
            entry.v_data = None
        elif target in (TierLevel.SSD_FAST, TierLevel.SSD_SLOW):
            k = entry.k_data if entry.k_data is not None else np.zeros(self.dim)
            v = entry.v_data if entry.v_data is not None else np.zeros(self.dim)
            path = self._ssd_path(entry.position, entry.head_idx, entry.layer_idx)
            np.savez(path, k=k, v=v)
            entry.k_data = None
            entry.v_data = None
        entry.tier = target
        self.tiers[target][id(entry)] = entry
        self.demotions += 1

    def prefetch(self, positions: list[int]):
        for pos in positions:
            eid = self._entry_id_by_pos.get(pos)
            if eid is None:
                continue
            entry = self._all_entries[eid]
            if entry.tier <= TierLevel.DRAM:
                continue
            self._read_entry(entry)
            self.prefetches += 1

    def query(
        self, query_vector: np.ndarray, top_k: int = 10
    ) -> list[tuple[int, float]]:
        q = query_vector.ravel()
        q_norm = np.linalg.norm(q) + 1e-10
        scores = []
        for eid, entry in self._all_entries.items():
            result = self._read_entry(entry)
            if result is None:
                continue
            k, _ = result
            k_vec = k.ravel()
            sim = float(np.dot(q, k_vec)) / (q_norm * np.linalg.norm(k_vec) + 1e-10)
            scores.append((entry.position, sim))
        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]

    def _dct_compress(self, vec: np.ndarray, bits: int = 4) -> tuple[np.ndarray, dict]:
        n = vec.shape[-1]
        x = vec.astype(np.float64)
        dct_vals = np.zeros_like(x)
        for i in range(n):
            dct_vals[..., i] = np.sum(
                x * np.cos(np.pi * (np.arange(n) + 0.5) * i / n), axis=-1
            )
        dct_vals *= np.sqrt(2.0 / n)
        scale = np.max(np.abs(dct_vals), axis=-1, keepdims=True)
        scale = np.where(scale < 1e-10, 1.0, scale)
        normalized = dct_vals / scale
        max_q = (1 << (bits - 1)) - 1
        quantized = np.clip(np.round(normalized * max_q), -max_q, max_q)
        meta = {
            "scale": float(scale.ravel()[0]) if scale.size > 0 else 1.0,
            "bits": bits,
            "shape": vec.shape,
        }
        return quantized.ravel().astype(np.int8), meta

    def _dct_decompress(self, quantized: np.ndarray, meta: dict) -> np.ndarray:
        n = meta["shape"][-1]
        scale = meta["scale"]
        q = quantized.astype(np.float64)
        dct_vals = q * scale
        idct_vals = np.zeros_like(dct_vals)
        for i in range(n):
            idct_vals += dct_vals[..., i] * np.cos(np.pi * (np.arange(n) + 0.5) * i / n)
        idct_vals *= np.sqrt(2.0 / n)
        return idct_vals.reshape(meta["shape"]).astype(np.float32)

    def _ssd_path(self, position: int, head_idx: int, layer_idx: int) -> str:
        return f"{self.ssd_dir}/kv_{layer_idx}_{head_idx}_{position}.npz"

    def _evict_if_needed(self, tier: TierLevel):
        while len(self.tiers[tier]) > self.tier_capacities[tier]:
            oldest = min(self.tiers[tier].values(), key=lambda e: e.timestamp)
            self._evict_one(oldest)

    def _evict_one(self, entry: TierEntry):
        for t in TierLevel:
            if t == entry.tier:
                continue
            if len(self.tiers[t]) < self.tier_capacities[t] * 0.9:
                self.demote(entry.position, t)
                return
        if entry.tier in self.tiers:
            for k in list(self.tiers[entry.tier].keys()):
                if self.tiers[entry.tier][k] is entry:
                    del self.tiers[entry.tier][k]
                    break
        eid = self._entry_id_by_pos.get(entry.position)
        if eid is not None:
            self._all_entries.pop(eid, None)
            self._entry_id_by_pos.pop(entry.position, None)

    def cache_summary(self) -> dict:
        return {
            "type": "TieredKVCache",
            "dim": self.dim,
            "total_entries": len(self._all_entries),
            "tier_counts": {t.name: len(self.tiers[t]) for t in TierLevel},
            "tier_capacities": {t.name: self.tier_capacities[t] for t in TierLevel},
            "hits": self.hits,
            "misses": self.misses,
            "promotions": self.promotions,
            "demotions": self.demotions,
            "prefetches": self.prefetches,
        }


class SpectralKVCache_v2(SpectralKVCache):
    def __init__(
        self,
        dim: int = 128,
        max_size: int = 4096,
        k_bits: int = 4,
        v_bits: int = 2,
        seed: int = 42,
        use_dct: bool = False,
        progressive_start_bits: int = 8,
        stochastic_refresh_interval: int = 256,
        resonance_tracker: Optional[ResonanceTracker] = None,
        n_heads: int = 4,
        use_hybrid_kv: bool = True,
        holographic_fallback_dim: int = 256,
        simd_batch_size: int = 32,
        progressive_min_bits: int = 2,
    ):
        super().__init__(
            dim=dim,
            n_heads=n_heads,
            max_size=max_size,
            k_bits=k_bits,
            v_bits=v_bits,
            seed=seed,
            use_dct=use_dct,
            progressive_start_bits=progressive_start_bits,
            stochastic_refresh_interval=stochastic_refresh_interval,
            resonance_tracker=resonance_tracker,
        )

        self.use_hybrid_kv = use_hybrid_kv
        self.holographic_fallback_dim = holographic_fallback_dim
        self.simd_batch_size = simd_batch_size
        self.progressive_min_bits = progressive_min_bits

        self._holographic_fallback = None
        self._fallback_active = False
        self.holographic_offload_count = 0
        self.batched_compress_count = 0
        self.compressed_sim_count = 0

        if use_hybrid_kv:
            self.k_rotator = HadamardRotator(self.total_dim, seed=seed)
            self.v_rotator = DCTRotator(self.total_dim)
        else:
            self.k_rotator = self.rotator
            self.v_rotator = self.rotator

    def store(self, key: np.ndarray, value: np.ndarray, position: int):
        self._global_step += 1
        fill_ratio = len(self.entries) / max(self.max_size, 1)

        if fill_ratio > 0.9 and _HAS_HRR:
            self._store_holographic_fallback(key, value, position)
            self.holographic_offload_count += 1
            self._fallback_active = True
            return

        if self.use_hybrid_kv:
            rotated_k = self.k_rotator.rotate(key.reshape(1, -1)).ravel()
            rotated_v = self.v_rotator.rotate(value.reshape(1, -1)).ravel()
        else:
            rotated_k = self.rotator.rotate(key.reshape(1, -1)).ravel()
            rotated_v = self.rotator.rotate(value.reshape(1, -1)).ravel()

        entropy = self._compute_spectral_entropy(key)
        band = self._assign_band(position, entropy)
        k_bits_band, v_bits_band = BAND_COMPRESSION[band]

        rf = self._resonance_factor()
        k_bits_band = max(2, int(k_bits_band * rf))
        v_bits_band = max(1, int(v_bits_band * rf))

        k_bits_prog = self._progressive_bits(k_bits_band, fill_ratio)
        v_bits_prog = self._progressive_bits(v_bits_band, fill_ratio)

        k_bits_final = self._adaptive_bits(position, k_bits_prog)
        v_bits_final = self._adaptive_bits(position, v_bits_prog)

        kq = self._get_quantizer("k", band, k_bits_final)
        vq = self._get_quantizer("v", band, v_bits_final)

        k_idx, _ = kq.compress(rotated_k)
        v_idx, _ = vq.compress(rotated_v)

        self._maybe_evict()

        entry = CacheEntry(
            k_indices=k_idx,
            v_indices=v_idx,
            position=position,
            band=band,
            entropy=entropy,
            coherence=1.0,
            timestamp=self._global_step,
            n_bits_k=k_bits_final,
            n_bits_v=v_bits_final,
            rotated_dim=self._rotated_dim,
        )
        self.entries.append(entry)

        if (
            self.stochastic_refresh_interval > 0
            and self._global_step % self.stochastic_refresh_interval == 0
        ):
            self._stochastic_refresh()

    def _store_holographic_fallback(
        self, key: np.ndarray, value: np.ndarray, position: int
    ):
        k_rot = self.rotator.rotate(key.reshape(1, -1)).ravel()
        v_rot = self.rotator.rotate(value.reshape(1, -1)).ravel()
        kq = self._get_quantizer("k", BAND_LOW, 2)
        vq = self._get_quantizer("v", BAND_LOW, 1)
        k_idx, _ = kq.compress(k_rot)
        v_idx, _ = vq.compress(v_rot)
        entry = CacheEntry(
            k_indices=k_idx,
            v_indices=v_idx,
            position=position,
            band=BAND_LOW,
            entropy=0.0,
            coherence=1.0,
            timestamp=self._global_step,
            n_bits_k=2,
            n_bits_v=1,
            rotated_dim=self._rotated_dim,
        )
        self.entries.append(entry)

    def query_compressed(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        use_compressed_sim: bool = True,
    ) -> list[tuple[int, float, np.ndarray]]:
        q_rotated = self.rotator.rotate(query_vector.reshape(1, -1)).ravel()
        q_norm = np.linalg.norm(q_rotated) + 1e-10
        results = []

        for entry in self.entries:
            if not hasattr(entry, "k_indices") or entry.k_indices is None:
                continue
            if not hasattr(entry, "n_bits_k"):
                continue
            if not hasattr(entry, "band"):
                continue
            band = getattr(entry, "band", BAND_NORMAL)
            n_bits = getattr(entry, "n_bits_k", 4)
            kq = self._get_quantizer("k", band, n_bits)
            try:
                k_reconstructed = kq.decompress(entry.k_indices, (self._rotated_dim,))
            except Exception:
                continue

            if use_compressed_sim:
                sim = float(np.dot(q_rotated, k_reconstructed))
                sim /= float(q_norm * np.linalg.norm(k_reconstructed) + 1e-10)
                self.compressed_sim_count += 1
            else:
                k_full = self.rotator.inverse_rotate(
                    k_reconstructed.reshape(1, -1)
                ).ravel()
                q_full = query_vector.ravel()
                sim = float(np.dot(q_full, k_full))
                sim /= float(np.linalg.norm(q_full) * np.linalg.norm(k_full) + 1e-10)

            results.append((entry.position, sim, k_reconstructed))

        results.sort(key=lambda x: -x[1])
        return results[:top_k]

    def store_batch(self, keys: np.ndarray, values: np.ndarray, positions: np.ndarray):
        self.batched_compress_count += 1
        for k, v, pos in zip(keys, values, positions):
            self.store(k, v, int(pos))

    def retrieve_holographic(
        self, position: int
    ) -> Optional[tuple[np.ndarray, np.ndarray]]:
        for entry in self.entries:
            if getattr(entry, "position", None) == position:
                band = getattr(entry, "band", BAND_LOW)
                n_bits_k = getattr(entry, "n_bits_k", 2)
                kq = self._get_quantizer("k", band, n_bits_k)
                try:
                    k = kq.decompress(entry.k_indices, (self._rotated_dim,))
                    vq = self._get_quantizer("v", band, getattr(entry, "n_bits_v", 1))
                    v = vq.decompress(entry.v_indices, (self._rotated_dim,))
                    return (k, v)
                except Exception:
                    pass
        return None

    def cache_summary_v2(self) -> dict:
        s = self.cache_summary()
        s.update(
            {
                "use_hybrid_kv": self.use_hybrid_kv,
                "fallback_active": self._fallback_active,
                "holographic_offload_count": self.holographic_offload_count,
                "batched_compress_count": self.batched_compress_count,
                "compressed_sim_count": self.compressed_sim_count,
            }
        )
        return s
