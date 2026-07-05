"""HPC Huffman coding — np.bincount freq counting, precomputed code arrays, np.packbits."""

from __future__ import annotations

import heapq
import struct
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _scale_frequencies(freqs: np.ndarray, target_sum: int) -> np.ndarray:
    """Vectorized frequency scaling."""
    if len(freqs) == 0:
        return np.array([], dtype=np.int32)
    counts = freqs[:, 1].astype(np.int32)
    total = int(counts.sum())
    if total == 0:
        return np.full(len(freqs), target_sum // max(len(freqs), 1), dtype=np.int32)
    n = len(freqs)
    scaled = np.maximum(
        (counts.astype(np.int64) * target_sum // total).astype(np.int32), 1
    )
    diff = target_sum - int(scaled.sum())
    if diff > 0:
        scaled[:diff] += 1
    elif diff < 0:
        idx = np.arange(n - 1, -1, -1, dtype=np.int32)
        mask = scaled[idx] > 1
        n_rem = np.minimum(-diff, int(mask.sum()))
        scaled[idx[mask][:n_rem]] -= 1
    return scaled


def _pack_bits(bits) -> bytes:
    if not bits:
        return b""
    return bytes(np.packbits(np.array(bits, dtype=np.uint8)))


def _unpack_bits(data, n_bits):
    if n_bits == 0:
        return []
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8))[:n_bits].tolist()


def _write_varint(val: int) -> bytes:
    u = (val << 1) ^ (val >> 31)
    buf = bytearray()
    while u > 127:
        buf.append((u & 127) | 128)
        u >>= 7
    buf.append(u & 127)
    return bytes(buf)


def _read_varint(data: bytes, offset: int) -> Tuple[int, int]:
    u = 0
    shift = 0
    while True:
        b = data[offset]
        u |= (b & 127) << shift
        shift += 7
        offset += 1
        if not (b & 128):
            break
    val = (u >> 1) ^ (-(u & 1))
    return val, offset


class HuffmanCoder:
    """Full Huffman coding with serialized tree (canonical) — precomputed code arrays."""

    def __init__(self):
        self.codes: Dict[int, str] = {}
        self.symbols: List[int] = []
        self.code_lengths: Dict[int, int] = {}
        self._code_arrays: Dict[int, np.ndarray] = {}

    def build_tree(self, freqs: Dict[int, int]):
        if not freqs:
            return

        heap = [[wt, [sym, ""]] for sym, wt in freqs.items()]
        heapq.heapify(heap)
        while len(heap) > 1:
            lo = heapq.heappop(heap)
            hi = heapq.heappop(heap)
            for pair in lo[1:]:
                pair[1] = "0" + pair[1]
            for pair in hi[1:]:
                pair[1] = "1" + pair[1]
            heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])

        huff_list = heap[0][1:] if heap else []
        self.codes = {int(sym): code for sym, code in huff_list}

        if len(self.codes) == 1:
            only_sym = list(self.codes.keys())[0]
            self.codes[only_sym] = "0"

        self.code_lengths = {sym: len(code) for sym, code in self.codes.items()}

        max_len = max(self.code_lengths.values()) if self.code_lengths else 0
        length_counts = defaultdict(int)
        for sym, length in self.code_lengths.items():
            length_counts[length] += 1

        canonical = self._build_canonical(max_len, length_counts)
        if canonical:
            self.codes = canonical

        # Precompute code arrays for vectorized encoding
        self._code_arrays = {}
        for sym, code_str in self.codes.items():
            arr = np.array([1 if c == "1" else 0 for c in code_str], dtype=np.uint8)
            self._code_arrays[sym] = arr

    def _build_canonical(
        self, max_len: int, length_counts: Dict[int, int]
    ) -> Dict[int, str]:
        if not length_counts:
            return {}

        code = 0
        length_counts_arr = [length_counts.get(i, 0) for i in range(max_len + 1)]
        next_code = [0] * (max_len + 1)
        for bits in range(1, max_len + 1):
            code = (code + length_counts_arr[bits - 1]) << 1
            next_code[bits] = code

        symbols_by_len: Dict[int, List[int]] = defaultdict(list)
        for sym, length in self.code_lengths.items():
            symbols_by_len[length].append(sym)
        for lst in symbols_by_len.values():
            lst.sort()

        canonical = {}
        for length in range(1, max_len + 1):
            for sym in symbols_by_len[length]:
                code_str = bin(next_code[length])[2:].zfill(length)
                canonical[sym] = code_str
                next_code[length] += 1

        return canonical

    def encode(self, data: np.ndarray) -> Tuple[bytes, dict]:
        flat = data.ravel().astype(np.int32)
        if len(flat) == 0:
            return b"", {"tree": {}}

        if not self.codes:
            unique, counts = np.unique(flat, return_counts=True)
            freqs = {int(unique[i]): int(counts[i]) for i in range(len(unique))}
            self.build_tree(freqs)

        # Build bits array via precomputed code arrays
        code_arrays = self._code_arrays
        total_bits = sum(
            len(code_arrays.get(int(s), np.zeros(1, dtype=np.uint8))) for s in flat
        )
        if total_bits == 0:
            return b"", {"tree": self.serialize_tree(), "n_orig": len(flat)}

        # Concatenate all code arrays
        all_bits_list = []
        for val in flat:
            s = int(val)
            arr = code_arrays.get(s)
            if arr is not None:
                all_bits_list.append(arr)
            else:
                all_bits_list.append(np.zeros(1, dtype=np.uint8))
        all_bits = (
            np.concatenate(all_bits_list)
            if all_bits_list
            else np.array([], dtype=np.uint8)
        )
        packed = bytes(np.packbits(all_bits))

        metadata = {
            "tree": self.serialize_tree(),
            "code_lengths": dict(self.code_lengths),
            "n_orig": len(flat),
        }
        return packed, metadata

    def decode(self, compressed: bytes, metadata: dict) -> np.ndarray:
        n_orig = metadata.get("n_orig", 0)
        if n_orig == 0:
            return np.array([], dtype=np.int32)

        tree_bytes = metadata.get("tree", b"")
        code_lengths = self.deserialize_tree(tree_bytes) if tree_bytes else {}

        if not code_lengths:
            return (
                np.zeros(n_orig, dtype=np.int32)
                if n_orig > 0
                else np.array([], dtype=np.int32)
            )

        max_len = max(code_lengths.values()) if code_lengths else 0
        symbols_by_len: Dict[int, List[int]] = defaultdict(list)
        for sym, length in code_lengths.items():
            symbols_by_len[length].append(sym)
        for lst in symbols_by_len.values():
            lst.sort()
        length_counts = {length: len(lst) for length, lst in symbols_by_len.items()}

        code = 0
        length_counts_arr = [length_counts.get(i, 0) for i in range(max_len + 1)]
        next_code = [0] * (max_len + 1)
        for bits_len in range(1, max_len + 1):
            code = (code + length_counts_arr[bits_len - 1]) << 1
            next_code[bits_len] = code

        # Build search table: for each code string, map to symbol
        code_to_sym: Dict[str, int] = {}
        for length in range(1, max_len + 1):
            for sym in symbols_by_len[length]:
                code_str = bin(next_code[length])[2:].zfill(length)
                code_to_sym[code_str] = sym
                next_code[length] += 1

        bits = np.unpackbits(np.frombuffer(compressed, dtype=np.uint8))
        result = np.zeros(n_orig, dtype=np.int32)
        idx = 0
        pos = 0
        max_bits = len(bits)

        while idx < n_orig and pos < max_bits:
            found = False
            length = 0
            while pos + length < max_bits and length <= max_len:
                length += 1
                chunk = bits[pos : pos + length]
                code_str = "".join(str(b) for b in chunk)
                if code_str in code_to_sym:
                    result[idx] = code_to_sym[code_str]
                    idx += 1
                    pos += length
                    found = True
                    break
            if not found:
                pos += 1

        return result

    def serialize_tree(self) -> bytes:
        if not self.code_lengths:
            return b""
        buf = bytearray()
        n_syms = len(self.code_lengths)
        buf.extend(_write_varint(n_syms))
        for sym, length in sorted(self.code_lengths.items()):
            buf.extend(_write_varint(sym))
            buf.append(length)
        return bytes(buf)

    @classmethod
    def deserialize_tree(cls, data: bytes) -> Dict[int, int]:
        if not data:
            return {}
        offset = 0
        n_syms, offset = _read_varint(data, offset)
        lengths: Dict[int, int] = {}
        for _ in range(n_syms):
            if offset >= len(data):
                break
            sym, offset = _read_varint(data, offset)
            if offset >= len(data):
                break
            length = data[offset]
            offset += 1
            lengths[sym] = length
        return lengths
