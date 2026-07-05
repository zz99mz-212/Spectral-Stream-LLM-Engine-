"""Auto-generated from rans_coding.py."""

from __future__ import annotations

import heapq
import struct
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _scale_frequencies(freqs, target_sum):
    if len(freqs) == 0:
        return np.array([], dtype=np.int32)
    counts = freqs[:, 1].astype(np.int32)
    total = int(counts.sum())
    if total == 0:
        return np.ones(len(freqs), dtype=np.int32) * (target_sum // max(len(freqs), 1))
    n = len(freqs)
    scaled = (counts.astype(np.int64) * target_sum // total).astype(np.int32)
    scaled = np.maximum(scaled, 1)
    diff = target_sum - int(scaled.sum())
    if diff > 0:
        for i in range(n):
            if diff <= 0:
                break
            scaled[i] += 1
            diff -= 1
    elif diff < 0:
        for i in range(n - 1, -1, -1):
            if diff >= 0 or scaled[i] <= 1:
                break
            scaled[i] -= 1
            diff += 1
    return scaled


def _pack_bits(bits):
    if not bits:
        return b""
    n_bytes = (len(bits) + 7) // 8
    packed = bytearray(n_bytes)
    for i, b in enumerate(bits):
        if b:
            packed[i >> 3] |= 1 << (i & 7)
    return bytes(packed)


def _unpack_bits(data, n_bits):
    bits = []
    for i in range(n_bits):
        bits.append(1 if (data[i >> 3] >> (i & 7)) & 1 else 0)
    return bits


def _write_varint(val):
    buf = bytearray()
    while val > 127:
        buf.append((val & 127) | 128)
        val >>= 7
    buf.append(val & 127)
    return bytes(buf)


def _read_varint(data, offset):
    val = 0
    shift = 0
    while True:
        b = data[offset]
        val |= (b & 127) << shift
        shift += 7
        offset += 1
        if not (b & 128):
            break
    return val, offset


def lz77_encode(data: np.ndarray, window_bits: int = 12) -> Tuple[bytes, dict]:
    """LZ77 sliding window compression.
    Returns (compressed_bytes, metadata).
    """
    flat = (
        data.ravel().astype(np.int64).clip(-(1 << 31), (1 << 31) - 1).astype(np.int32)
    )
    if len(flat) == 0:
        return b"", {"n_orig": 0, "tokens": []}

    window_size = 1 << window_bits
    lookahead = 255

    tokens: List[Tuple[int, int, int]] = []
    i = 0
    while i < len(flat):
        best_off = 0
        best_len = 0
        start = max(0, i - window_size)
        max_match = min(lookahead, len(flat) - i)
        if max_match > 0:
            for j in range(start, i):
                if flat[j] == flat[i]:
                    ml = 1
                    while (
                        ml < max_match and j + ml < i and flat[j + ml] == flat[i + ml]
                    ):
                        ml += 1
                    if ml > best_len:
                        best_off = i - j
                        best_len = ml
                        if ml == max_match:
                            break
        if best_len >= 3:
            tokens.append((best_off, best_len, 0))
            i += best_len
        else:
            tokens.append((0, 0, int(flat[i])))
            i += 1

    buf = bytearray()
    buf.extend(_write_varint(len(tokens)))
    for off, length, lit in tokens:
        if length > 0:
            buf.append(0)
            buf.extend(_write_varint(off))
            buf.extend(_write_varint(length))
        else:
            buf.append(1)
            val = lit & 0xFFFFFFFF
            buf.extend(_write_varint(val))

    metadata = {
        "n_orig": len(flat),
        "n_tokens": len(tokens),
        "dtype": str(flat.dtype),
    }
    return bytes(buf), metadata


def lz77_decode(compressed: bytes, metadata: dict) -> np.ndarray:
    """Decode LZ77 compressed data."""
    n_orig = metadata.get("n_orig", 0)
    if n_orig == 0:
        return np.array([], dtype=np.int32)

    offset = 0
    n_tokens, offset = _read_varint(compressed, offset)

    result: List[int] = []
    for _ in range(n_tokens):
        if offset >= len(compressed):
            break
        token_type = compressed[offset]
        offset += 1
        if token_type == 0:
            off, offset = _read_varint(compressed, offset)
            length, offset = _read_varint(compressed, offset)
            start = len(result) - off
            for j in range(length):
                result.append(result[start + j])
        else:
            val, offset = _read_varint(compressed, offset)
            result.append(val)

    arr = np.array(result[:n_orig], dtype=np.int64)
    arr = np.where(arr > (1 << 31) - 1, arr - (1 << 32), arr)
    return arr.astype(np.int32)


# =============================================================================
# 6. BWT + MTF + RLE
# =============================================================================

import struct


def golomb_encode(data: np.ndarray, m: int = 4) -> Tuple[bytes, dict]:
    """Golomb coding for geometrically distributed data.
    m = divisor parameter. Data values must be non-negative.
    """
    flat = data.ravel().astype(np.int32)
    if len(flat) == 0:
        return b"", {"m": m, "n_orig": 0}

    buf = bytearray()
    bits: List[int] = []

    for val in flat:
        v = int(val)
        if v < 0:
            v = 0
        q = v // m
        r = v % m

        for _ in range(q):
            bits.append(1)

        bits.append(0)

        if m & (m - 1) == 0:
            log2m = m.bit_length() - 1
            for j in range(log2m - 1, -1, -1):
                bits.append((r >> j) & 1)
        else:
            b = (m.bit_length() + 1).bit_length()
            cutoff = (1 << b) - m
            if r < cutoff:
                b -= 1
            for j in range(b - 1, -1, -1):
                bits.append((r >> j) & 1)

    packed = _pack_bits(bits)

    metadata = {
        "m": m,
        "n_orig": len(flat),
        "n_bits": len(bits),
    }
    return packed, metadata


def golomb_decode(compressed: bytes, metadata: dict, n: int) -> np.ndarray:
    """Decode Golomb-coded data."""
    if n == 0:
        return np.array([], dtype=np.int32)

    m = metadata.get("m", 4)
    bits = _unpack_bits(compressed, len(compressed) * 8)

    result = np.zeros(n, dtype=np.int32)
    idx = 0
    pos = 0
    n_bits = len(bits)

    log2m = (m & -m) == m and m > 0

    if log2m:
        log2m_val = m.bit_length() - 1
        while idx < n and pos < n_bits:
            q = 0
            while pos < n_bits and bits[pos] == 1:
                q += 1
                pos += 1
            if pos >= n_bits:
                break
            pos += 1

            r = 0
            for _ in range(log2m_val):
                if pos < n_bits:
                    r = (r << 1) | bits[pos]
                    pos += 1
            result[idx] = q * m + r
            idx += 1
    else:
        b = m.bit_length() + 1
        cutoff = (1 << b) - m
        while idx < n and pos < n_bits:
            q = 0
            while pos < n_bits and bits[pos] == 1:
                q += 1
                pos += 1
            if pos >= n_bits:
                break
            pos += 1

            if pos >= n_bits:
                break

            use_b = b - 1 if bits[pos : pos + 1] == [0] else b
            if use_b == b:
                r = 0
                for _ in range(b):
                    if pos < n_bits:
                        r = (r << 1) | bits[pos]
                        pos += 1
            else:
                r = 0
                for _ in range(b - 1):
                    if pos < n_bits:
                        r = (r << 1) | bits[pos]
                        pos += 1
            result[idx] = q * m + r
            idx += 1

    return result


# =============================================================================
# 8. Range Coder (Binary Arithmetic Coding)
# =============================================================================
