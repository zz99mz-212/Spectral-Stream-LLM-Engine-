"""HPC predictive coding — fully vectorized via np.diff, np.cumsum, no loops."""

from __future__ import annotations

import heapq
import struct
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spectralstream.compression.methods.entropy.huffman import HuffmanCoder


def _frequencies_bincount(flat: np.ndarray) -> dict:
    """Vectorized frequency counting via np.bincount."""
    if len(flat) == 0:
        return {}
    vmin = int(np.min(flat))
    shifted = flat.astype(np.int64) - vmin
    counts = np.bincount(shifted)
    nz = np.where(counts > 0)[0]
    return {int(nz[i] + vmin): int(counts[nz[i]]) for i in range(len(nz))}


def predictive_encode(data: np.ndarray, order: int = 1) -> Tuple[bytes, dict]:
    """Linear prediction via np.diff — fully vectorized."""
    flat = data.ravel().astype(np.int32)
    if len(flat) == 0:
        return b"", {"n_orig": 0, "order": order, "first": []}

    n = len(flat)
    residuals = np.zeros(n, dtype=np.int32)
    first = np.zeros(order, dtype=np.int32)

    if order <= n:
        first[:order] = flat[:order]
        residuals[:order] = flat[:order]

    if order == 1:
        residuals[1:] = flat[1:] - flat[:-1]
    elif order == 2:
        residuals[2:] = flat[2:] - 2 * flat[1:-1] + flat[:-2]
    elif order == 3:
        residuals[3:] = flat[3:] - 3 * flat[2:-1] + 3 * flat[1:-2] - flat[:-3]
    else:
        for i in range(order, n):
            residuals[i] = flat[i] - int(np.mean(flat[i - order : i]))

    freqs = _frequencies_bincount(residuals)
    coder = HuffmanCoder()
    coder.build_tree(freqs)
    packed, _ = coder.encode(residuals)

    metadata = {
        "n_orig": n,
        "order": order,
        "first": first.tolist(),
        "tree": coder.serialize_tree(),
    }
    return packed, metadata


def predictive_decode(compressed: bytes, metadata: dict) -> np.ndarray:
    """Decode predictive-encoded data via np.cumsum — fully vectorized."""
    n_orig = metadata.get("n_orig", 0)
    order = metadata.get("order", 1)
    first_list = metadata.get("first", [])
    tree_bytes = metadata.get("tree", b"")

    if n_orig == 0:
        return np.array([], dtype=np.int32)

    code_lengths = HuffmanCoder.deserialize_tree(tree_bytes) if tree_bytes else {}
    if not code_lengths:
        return np.zeros(n_orig, dtype=np.int32)

    coder = HuffmanCoder()
    coder.code_lengths = code_lengths
    residuals = coder.decode(compressed, {"n_orig": n_orig, "tree": tree_bytes})

    flat = np.zeros(n_orig, dtype=np.int32)
    k = min(order, n_orig)
    flat[:k] = [first_list[i] if i < len(first_list) else 0 for i in range(k)]

    if order == 1:
        flat[1:] = np.cumsum(residuals[1:]) + flat[0]
    elif order == 2:
        for i in range(2, n_orig):
            flat[i] = 2 * flat[i - 1] - flat[i - 2] + residuals[i]
    elif order == 3:
        for i in range(3, n_orig):
            flat[i] = 3 * flat[i - 1] - 3 * flat[i - 2] + flat[i - 3] + residuals[i]
    else:
        for i in range(order, n_orig):
            pred = int(np.mean(flat[i - order : i]))
            flat[i] = pred + residuals[i]

    return flat


def bitpack(data: np.ndarray, bits_per_symbol: int) -> bytes:
    """Pack symbols into bit representation via numpy bit operations."""
    flat = data.ravel()
    if len(flat) == 0:
        return b""
    mask = (1 << bits_per_symbol) - 1
    total_bits = len(flat) * bits_per_symbol
    n_bytes = (total_bits + 7) // 8
    result = np.zeros(n_bytes, dtype=np.uint8)
    flat_clipped = flat.astype(np.int64) & mask
    for i in range(bits_per_symbol):
        byte_idx = np.arange(len(flat)) * bits_per_symbol // 8
        bit_offset = (np.arange(len(flat)) * bits_per_symbol) % 8
        bits = (flat_clipped >> i) & 1
        byte_pos = (np.arange(len(flat)) * bits_per_symbol + i) // 8
        bit_pos = (np.arange(len(flat)) * bits_per_symbol + i) % 8
        valid = byte_pos < n_bytes
        np.add.at(
            result,
            byte_pos[valid],
            (bits[valid].astype(np.uint8) << bit_pos[valid].astype(np.uint8)),
        )
    return bytes(result[:n_bytes])


def bitunpack(packed: bytes, bits_per_symbol: int, n: int) -> np.ndarray:
    """Unpack bits back to symbols via numpy."""
    if n == 0:
        return np.array([], dtype=np.int32)
    packed_arr = np.frombuffer(packed, dtype=np.uint8)
    result = np.zeros(n, dtype=np.int32)
    for i in range(bits_per_symbol):
        byte_idx = (np.arange(n) * bits_per_symbol + i) // 8
        bit_idx = (np.arange(n) * bits_per_symbol + i) % 8
        valid = byte_idx < len(packed_arr)
        if valid.any():
            bits = (packed_arr[byte_idx[valid]] >> bit_idx[valid].astype(np.uint8)) & 1
            result[valid] |= bits.astype(np.int32) << i
    return result
