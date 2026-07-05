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
        return b"\x00"
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
        current_val = struct.unpack_from("<I", raw, pos)[0]
        h = _lz4_hash(current_val)
        ref = hash_table.get(h)
        hash_table[h] = pos
        if (
            ref is not None
            and pos - ref <= 65535
            and raw[ref : ref + 4] == raw[pos : pos + 4]
        ):
            match_len = 4
            max_match = min(n - pos, 65535 + 4)
            while (
                match_len < max_match and raw[pos + match_len] == raw[ref + match_len]
            ):
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
            result.extend(struct.pack("<H", pos - ref))
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
                    byte |= 1 << j
            result[g, bit] = byte
    return np.concatenate([result.ravel(), data[usable:]])


def delta_compress(data: np.ndarray, order: int = 1) -> Tuple[bytes, float]:
    flat = data.ravel()
    n = len(flat)
    if order < 1:
        order = 1
    if n <= order:
        return flat.tobytes(), 1.0
    bits = flat.dtype.itemsize * 8
    if flat.dtype.kind == "f":
        int_vals = flat.view(f"uint{bits}").astype(np.uint64)
    elif flat.dtype.kind in "ui":
        int_vals = flat.astype(np.uint64)
    else:
        int_vals = flat.view(np.uint64).astype(np.uint64)
    first = np.array(int_vals[:order], dtype=np.uint64)
    diffs = np.diff(int_vals.astype(np.int64), n=order)
    header = struct.pack("<II", order, n)
    body = first.tobytes() + np.asarray(diffs, dtype=np.int64).tobytes()
    return header + body, data.nbytes / max(len(header) + len(body), 1)


def delta_decompress(
    compressed: bytes, order: int, dtype: np.dtype, shape: tuple
) -> np.ndarray:
    bits = dtype.itemsize * 8
    if len(compressed) < 8:
        return np.empty(shape, dtype=dtype)
    stored_order, n = struct.unpack_from("<II", compressed, 0)
    order = stored_order
    pos = 8
    first_n = min(order, n)
    if pos + first_n * 8 > len(compressed):
        return np.empty(shape, dtype=dtype)
    first = np.frombuffer(compressed[pos : pos + first_n * 8], dtype=np.uint64)
    pos += first_n * 8
    result = np.zeros(n, dtype=np.uint64)
    if len(first) > 0:
        result[: len(first)] = first
    if n > first_n:
        remaining = len(compressed) - pos
        num_diffs = remaining // 8 if remaining >= 8 else 0
        if num_diffs > 0 and num_diffs + first_n <= n:
            diff_vals = np.frombuffer(
                compressed[pos : pos + num_diffs * 8], dtype=np.int64
            )
            from math import comb

            for i in range(first_n, first_n + num_diffs):
                val = int(diff_vals[i - first_n])
                s = 1
                for j in range(1, order + 1):
                    val = (val + s * comb(order, j) * int(result[i - j])) & (
                        (1 << 64) - 1
                    )
                    s = -s
                result[i] = val
    if dtype.kind == "f":
        uint_dtype = np.dtype(f"uint{bits}")
        return result.astype(uint_dtype).view(dtype).reshape(shape)
    return result.astype(dtype).reshape(shape)


# ═══════════════════════════════════════════════════════════════════════════
# XOR encoding
# ═══════════════════════════════════════════════════════════════════════════


def xor_compress(data: np.ndarray) -> Tuple[bytes, float]:
    flat = data.ravel()
    n = len(flat)
    if n == 0:
        return b"", 1.0
    raw = flat.tobytes()
    itemsize = flat.dtype.itemsize
    xored = bytearray()
    xored.extend(raw[:itemsize])
    prev = int.from_bytes(raw[:itemsize], "little")
    for i in range(1, n):
        cur = int.from_bytes(raw[i * itemsize : (i + 1) * itemsize], "little")
        xor_val = prev ^ cur
        xored.extend(xor_val.to_bytes(itemsize, "little"))
        prev = cur
    return bytes(xored), data.nbytes / max(len(xored), 1)


def xor_decompress(compressed: bytes, dtype: np.dtype, shape: tuple) -> np.ndarray:
    itemsize = dtype.itemsize
    if len(compressed) < itemsize:
        return np.empty(shape, dtype=dtype)
    n_elems = (len(compressed) - itemsize) // itemsize + 1
    result = bytearray()
    result.extend(compressed[:itemsize])
    prev = int.from_bytes(compressed[:itemsize], "little")
    pos = itemsize
    for _ in range(1, n_elems):
        xor_val = int.from_bytes(compressed[pos : pos + itemsize], "little")
        cur = prev ^ xor_val
        result.extend(cur.to_bytes(itemsize, "little"))
        prev = cur
        pos += itemsize
    return np.frombuffer(bytes(result), dtype=dtype).reshape(shape)


# ═══════════════════════════════════════════════════════════════════════════
# PFOR delta (Patched Frame Of Reference)
# ═══════════════════════════════════════════════════════════════════════════


def pfor_delta_compress(data: np.ndarray, frame_bits: int = 0) -> Tuple[bytes, float]:
    flat = data.ravel()
    n = len(flat)
    if n == 0:
        return struct.pack("<qB", 0, 0) + struct.pack("<I", 0), 1.0
    bits = flat.dtype.itemsize * 8
    if flat.dtype.kind == "f":
        int_vals = flat.view(f"uint{bits}").astype(np.uint64)
    elif flat.dtype.kind in "ui":
        int_vals = flat.astype(np.uint64) if bits < 64 else flat.view(np.uint64)
    else:
        int_vals = flat.astype(np.uint64)

    int_vals_np = np.asarray(int_vals, dtype=np.uint64)
    base = int(np.min(int_vals_np))
    deltas = int_vals_np.astype(np.int64) - base
    max_delta = int(np.max(deltas))

    if frame_bits <= 0:
        frame_bits = max(1, max_delta.bit_length())
    b = min(frame_bits, max(1, max_delta.bit_length()))
    elem_bytes = max(1, (b + 7) // 8)

    packed_bytes = bytearray()
    exceptions = []
    threshold = (1 << b) if b < 63 else (1 << 63)
    for i in range(n):
        d = int(deltas[i])
        if b >= 63 or (0 <= d < threshold):
            packed_bytes.extend(
                (d & ((1 << (elem_bytes * 8)) - 1)).to_bytes(elem_bytes, "little")
            )
        else:
            packed_bytes.extend(b"\x00" * elem_bytes)
            exceptions.append((i, int(int_vals_np[i])))

    result = bytearray()
    result.extend(struct.pack("<q", base))
    result.append(b)
    result.extend(struct.pack("<I", len(exceptions)))
    result.extend(packed_bytes)
    for idx, val in exceptions:
        result.extend(struct.pack("<IQ", idx, val))

    return bytes(result), data.nbytes / max(len(result), 1)


def pfor_delta_decompress(
    compressed: bytes, dtype: np.dtype, shape: tuple
) -> np.ndarray:
    base = struct.unpack_from("<q", compressed, 0)[0]
    b = compressed[8]
    num_exc = struct.unpack_from("<I", compressed, 9)[0]
    pos = 13
    elem_bytes = max(1, (b + 7) // 8)
    exc_bytes = num_exc * 12
    n = (len(compressed) - pos - exc_bytes) // elem_bytes if elem_bytes > 0 else 0
    if n <= 0:
        return np.empty(shape, dtype=dtype)
    result = np.zeros(n, dtype=np.uint64)
    for i in range(n):
        raw_val = int.from_bytes(compressed[pos : pos + elem_bytes], "little")
        result[i] = raw_val
        pos += elem_bytes
    for _ in range(num_exc):
        idx, val = struct.unpack_from("<IQ", compressed, pos)
        pos += 12
        if idx < n:
            result[idx] = (val - base) & ((1 << 64) - 1)
    result = (result.astype(np.int64) + base).astype(np.uint64)
    bits = dtype.itemsize * 8
    if dtype.kind == "f":
        uint_dtype = np.dtype(f"uint{bits}")
        return result.astype(uint_dtype).view(dtype).reshape(shape)
    return result.astype(dtype).reshape(shape)


# ═══════════════════════════════════════════════════════════════════════════
# Dictionary compression
# ═══════════════════════════════════════════════════════════════════════════


def dictionary_compress(
    data: np.ndarray, max_dict_size: int = 256
) -> Tuple[bytes, float]:
    flat = data.ravel()
    n = len(flat)
    if n == 0:
        return struct.pack("<B", 1) + struct.pack("<II", 0, 0), 1.0
    unique = np.unique(flat)
    if len(unique) > max_dict_size:
        raw = flat.tobytes()
        return struct.pack("<B", 0) + raw, 1.0
    codebook = {v: i for i, v in enumerate(unique)}
    indices = np.array([codebook[v] for v in flat], dtype=np.uint32)
    dict_size = len(unique)
    bits_per_idx = max(1, (dict_size - 1).bit_length() if dict_size > 1 else 1)
    packed = bytearray()
    bit_pos = 0
    for idx in indices:
        if bit_pos + bits_per_idx > len(packed) * 8:
            packed.extend(b"\x00" * 4)
        word = int.from_bytes(packed[bit_pos // 8 : (bit_pos // 8) + 4], "little")
        word |= int(idx) << (bit_pos % 8)
        packed[bit_pos // 8 : (bit_pos // 8) + 4] = word.to_bytes(4, "little")
        bit_pos += bits_per_idx

    result = bytearray()
    result.extend(struct.pack("<B", 1))
    result.extend(struct.pack("<II", dict_size, n))
    for v in unique:
        result.extend(np.array([v]).tobytes())
    result.extend(struct.pack("<I", bits_per_idx))
    result.extend(packed[: int(np.ceil(bit_pos / 8))])
    return bytes(result), data.nbytes / max(len(result), 1)


def dictionary_decompress(
    compressed: bytes, dtype: np.dtype, shape: tuple
) -> np.ndarray:
    pos = 0
    mode = compressed[pos]
    pos += 1
    if mode == 0:
        return np.frombuffer(compressed[pos:], dtype=dtype).reshape(shape)
    itemsize = dtype.itemsize
    dict_size, n = struct.unpack_from("<II", compressed, pos)
    pos += 8
    if dict_size == 0:
        return np.empty(shape, dtype=dtype)
    codebook = []
    for _ in range(dict_size):
        val = np.frombuffer(compressed[pos : pos + itemsize], dtype=dtype, count=1)[0]
        codebook.append(val)
        pos += itemsize
    bits_per_idx = struct.unpack_from("<I", compressed, pos)[0]
    pos += 4
    packed = compressed[pos:]
    codebook_arr = np.array(codebook, dtype=dtype)
    max_idx = dict_size - 1
    indices = np.zeros(n, dtype=np.uint32)
    bit_pos = 0
    idx = 0
    while idx < n:
        if bit_pos + bits_per_idx > len(packed) * 8:
            break
        byte_start = bit_pos // 8
        word = int.from_bytes(packed[byte_start : byte_start + 4], "little")
        code = (word >> (bit_pos % 8)) & ((1 << bits_per_idx) - 1)
        if code <= max_idx:
            indices[idx] = code
        idx += 1
        bit_pos += bits_per_idx
    return codebook_arr[indices].reshape(shape)


# ═══════════════════════════════════════════════════════════════════════════
# Zigzag + variable-length integer encoding
# ═══════════════════════════════════════════════════════════════════════════


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


def _varint_decode(data: bytes, pos: int) -> Tuple[int, int]:
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


def zigzag_compress(data: np.ndarray) -> Tuple[bytes, float]:
    flat = data.ravel()
    n = len(flat)
    if n == 0:
        return b"", 1.0
    bits = flat.dtype.itemsize * 8
    if flat.dtype.kind == "f":
        int_view = flat.view(f"int{bits}")
    elif flat.dtype.kind == "u":
        int_view = flat.view(f"int{bits}")
    else:
        int_view = flat
    result = bytearray()
    for v in int_view:
        zz = _zigzag_encode(int(v), bits)
        result.extend(_varint_encode(zz))
    return bytes(result), data.nbytes / max(len(result), 1)


def zigzag_decompress(compressed: bytes, dtype: np.dtype, shape: tuple) -> np.ndarray:
    bits = dtype.itemsize * 8
    vals = []
    pos = 0
    while pos < len(compressed):
        zz, consumed = _varint_decode(compressed, pos)
        pos += consumed
        vals.append(_zigzag_decode(zz, bits))
    signed_dtype = np.dtype(f"int{bits}")
    arr = np.array(vals, dtype=signed_dtype)
    if dtype.kind == "f":
        return arr.view(dtype).reshape(shape)
    if dtype.kind == "u":
        return arr.view(dtype).reshape(shape)
    return arr.astype(dtype).reshape(shape)


# ═══════════════════════════════════════════════════════════════════════════
# Bitshuffle (transpose bits, then zlib)
# ═══════════════════════════════════════════════════════════════════════════


def _bitshuffle_transpose(data: np.ndarray) -> np.ndarray:
    """Bit-transpose groups of 8 bytes. Self-inverse."""
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
                    byte |= 1 << j
            result[g, bit] = byte
    return np.concatenate([result.ravel(), data[usable:]])


def bitshuffle_compress(data: np.ndarray, block_size: int = 256) -> Tuple[bytes, float]:
    raw = data.tobytes()
    n = len(raw)
    if n == 0:
        return struct.pack("<II", 0, 0), 1.0
    block_size = max(8, block_size)
    raw_np = np.frombuffer(raw, dtype=np.uint8).copy()
    shuffled = _bitshuffle_transpose(raw_np)
    header = struct.pack("<II", n, block_size)
    compressed = header + zlib.compress(bytes(shuffled), 6)
    return compressed, data.nbytes / max(len(compressed), 1)


def bitshuffle_decompress(
    compressed: bytes, dtype: np.dtype, shape: tuple
) -> np.ndarray:
    n, block_size = struct.unpack_from("<II", compressed, 0)
    if n == 0:
        return np.empty(shape, dtype=dtype)
    zlib_data = compressed[8:]
    shuffled = np.frombuffer(zlib.decompress(zlib_data), dtype=np.uint8)
    unshuffled = _bitshuffle_transpose(shuffled)
    return np.frombuffer(bytes(unshuffled[:n]), dtype=dtype).reshape(shape)
