"""Auto-generated from lossless_codecs.py."""

import gzip
import struct
import zlib
from typing import Dict, Tuple

import numpy as np


def _lz4_hash(val: int) -> int:
    return (val * 2654435761) >> 20


def _lz4_compress_block(raw: bytes) -> bytes:
    n = len(raw)
    if n == 0:
        return b'\x00'
    hash_table: Dict[int, int] = {}
    result = bytearray()
    anchor = 0
    pos = 0

    def emit_last_literals():
        nonlocal anchor
        if anchor >= n:
            return
        remaining = n - anchor
        ll_part = 15 if remaining >= 15 else remaining
        result.append(ll_part << 4)
        if remaining >= 15:
            extra = remaining - 15
            while extra >= 255:
                result.append(255)
                extra -= 255
            result.append(extra)
        result.extend(raw[anchor:n])
        anchor = n

    while pos < n - 4:
        current_val = struct.unpack_from('<I', raw, pos)[0]
        h = _lz4_hash(current_val)
        ref = hash_table.get(h)
        hash_table[h] = pos
        if ref is not None and pos - ref <= 65535 and raw[ref:ref + 4] == raw[pos:pos + 4]:
            match_len = 4
            max_match = min(n - pos, 65535 + 4)
            while match_len < max_match and raw[pos + match_len] == raw[ref + match_len]:
                match_len += 1
            lit_len = pos - anchor
            ll_part = min(lit_len, 15)
            ml_part = min(match_len - 4, 15)
            token = (ll_part << 4) | ml_part
            result.append(token)
            if ll_part == 15 and lit_len > 15:
                extra = lit_len - 15
                while extra >= 255:
                    result.append(255)
                    extra -= 255
                result.append(extra)
            result.extend(raw[anchor:pos])
            result.extend(struct.pack('<H', pos - ref))
            if ml_part == 15 and match_len - 4 > 15:
                extra = match_len - 4 - 15
                while extra >= 255:
                    result.append(255)
                    extra -= 255
                result.append(extra)
            pos += match_len
            anchor = pos
        else:
            pos += 1
    emit_last_literals()
    return bytes(result)


def _zigzag_encode(x: int, bits: int) -> int:
    return (x << 1) ^ (x >> (bits - 1))


def _zigzag_decode(z: int, bits: int) -> int:
    return (z >> 1) ^ -(z & 1)


def _varint_encode(val: int) -> bytes:
    result = bytearray()
    while val >= 128:
        result.append((val & 0x7F) | 0x80)
        val >>= 7
    result.append(val & 0x7F)
    return bytes(result)


def _varint_decode(data: bytes, pos: int):
    val = 0
    shift = 0
    start = pos
    while pos < len(data):
        byte = data[pos]
        val |= (byte & 0x7F) << shift
        shift += 7
        pos += 1
        if not (byte & 0x80):
            break
    return val, pos - start


def _bitshuffle_transpose(data: np.ndarray) -> np.ndarray:
    n = len(data)
    m = n // 8
    if m == 0:
        return data
    usable = m * 8
    arr = data[:usable].reshape(m, 8)
    result = np.zeros_like(arr)
    for g in range(m):
        for bit in range(8):
            byte = 0
            for j in range(8):
                if arr[g, j] & (1 << bit):
                    byte |= (1 << j)
            result[g, bit] = byte
    return np.concatenate([result.ravel(), data[usable:]])

def lz4_compress(data: np.ndarray, level: int = 0) -> Tuple[bytes, float]:
    raw = data.tobytes()
    compressed = _lz4_compress_block(raw)
    return compressed, len(raw) / max(len(compressed), 1)



def lz4_decompress(compressed: bytes, dtype: np.dtype, shape: tuple) -> np.ndarray:
    pos = 0
    result = bytearray()
    while pos < len(compressed):
        token = compressed[pos]
        pos += 1
        lit_len = token >> 4
        if lit_len == 15:
            while True:
                extra = compressed[pos]
                pos += 1
                lit_len += extra
                if extra < 255:
                    break
        if pos + lit_len > len(compressed):
            lit_len = len(compressed) - pos
        result.extend(compressed[pos:pos + lit_len])
        pos += lit_len
        if pos >= len(compressed):
            break
        if pos + 2 > len(compressed):
            break
        offset = struct.unpack_from('<H', compressed, pos)[0]
        pos += 2
        if offset == 0:
            break
        match_len = (token & 0x0F) + 4
        if (token & 0x0F) == 15:
            while True:
                if pos >= len(compressed):
                    break
                extra = compressed[pos]
                pos += 1
                match_len += extra
                if extra < 255:
                    break
        match_start = len(result) - offset
        if match_start < 0:
            break
        for i in range(match_len):
            if match_start + i >= 0:
                result.append(result[match_start + i])
    return np.frombuffer(bytes(result), dtype=dtype).reshape(shape)


# ═══════════════════════════════════════════════════════════════════════════
# LZ77 sliding window
# ═══════════════════════════════════════════════════════════════════════════



def lz77_compress(data: np.ndarray, window_bits: int = 15) -> Tuple[bytes, float]:
    raw = data.tobytes()
    n = len(raw)
    window_size = 1 << window_bits
    result = bytearray()
    pos = 0

    while pos < n:
        best_offset = 0
        best_len = 0
        search_start = max(0, pos - window_size)
        for match_start in range(search_start, pos):
            ml = 0
            while pos + ml < n and match_start + ml < pos and raw[match_start + ml] == raw[pos + ml]:
                ml += 1
            if ml > best_len:
                best_len = ml
                best_offset = pos - match_start

        if best_len >= 3:
            result.append(1)
            result.extend(struct.pack('<HH', best_offset, best_len))
            pos += best_len
        else:
            result.append(0)
            result.extend(raw[pos:pos + 1])
            pos += 1

    return bytes(result), n / max(len(result), 1)



def lz77_decompress(compressed: bytes, dtype: np.dtype, shape: tuple) -> np.ndarray:
    pos = 0
    result = bytearray()
    while pos < len(compressed):
        flag = compressed[pos]
        pos += 1
        if flag == 0:
            result.extend(compressed[pos:pos + 1])
            pos += 1
        else:
            offset, length = struct.unpack_from('<HH', compressed, pos)
            pos += 4
            match_start = len(result) - offset
            for i in range(length):
                result.append(result[match_start + i])
    return np.frombuffer(bytes(result), dtype=dtype).reshape(shape)


# ═══════════════════════════════════════════════════════════════════════════
# Run-length encoding
# ═══════════════════════════════════════════════════════════════════════════



def rle_compress(data: np.ndarray) -> Tuple[bytes, float]:
    flat = data.ravel()
    n = len(flat)
    if n == 0:
        empty = struct.pack('<I', 0)
        return empty, 1.0
    changes = np.concatenate([[True], flat[1:] != flat[:-1], [True]])
    run_starts = np.where(changes[:-1])[0]
    run_ends = np.where(changes[1:])[0] + 1
    values = flat[run_starts]
    lengths = run_ends - run_starts
    itemsize = flat.dtype.itemsize
    result = bytearray()
    result.extend(struct.pack('<I', len(values)))
    for v, l in zip(values, lengths):
        result.extend(np.array([v]).tobytes())
        result.extend(struct.pack('<I', l))
    return bytes(result), data.nbytes / max(len(result), 1)



def rle_decompress(compressed: bytes, dtype: np.dtype, shape: tuple) -> np.ndarray:
    itemsize = dtype.itemsize
    pos = 0
    num_runs = struct.unpack_from('<I', compressed, pos)[0]
    pos += 4
    parts = []
    for _ in range(num_runs):
        val = np.frombuffer(compressed[pos:pos + itemsize], dtype=dtype, count=1)
        pos += itemsize
        count = struct.unpack_from('<I', compressed, pos)[0]
        pos += 4
        parts.append(np.repeat(val, count))
    if not parts:
        return np.empty(shape, dtype=dtype)
    return np.concatenate(parts).reshape(shape)


# ═══════════════════════════════════════════════════════════════════════════
# Delta encoding
# ═══════════════════════════════════════════════════════════════════════════



