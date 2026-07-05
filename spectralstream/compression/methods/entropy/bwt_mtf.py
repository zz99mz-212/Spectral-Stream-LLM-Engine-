"""HPC BWT+MTF+RLE — fully vectorized: lexsort BWT, np.roll MTF, reduceat RLE."""

from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np


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


def _suffix_array_lexsort(arr: np.ndarray) -> np.ndarray:
    """Build suffix array via np.lexsort on rotation columns."""
    n = len(arr)
    dbl = np.concatenate([arr, arr])
    order = np.lexsort(tuple(dbl[i : i + n] for i in range(n - 1, -1, -1)))
    return order


def bwt_encode(data: np.ndarray) -> Tuple[np.ndarray, int]:
    """Burrows-Wheeler Transform via lexsort (fully vectorized)."""
    flat = data.ravel()
    n = len(flat)
    if n <= 1:
        return flat.copy(), 0
    order = _suffix_array_lexsort(flat)
    dbl = np.concatenate([flat, flat])
    last_col = dbl[order + n - 1]
    primary = int(np.where(order == 0)[0][0]) if 0 in order else 0
    return last_col, primary


def bwt_decode(transformed: np.ndarray, primary_index: int) -> np.ndarray:
    """Inverse BWT — vectorized occurrence counting via argsort grouping."""
    n = len(transformed)
    if n <= 1:
        return transformed.copy()

    unique, inverse, counts = np.unique(
        transformed, return_inverse=True, return_counts=True
    )
    n_vals = len(unique)
    cum_counts = np.zeros(n_vals + 1, dtype=np.int64)
    cum_counts[1:] = np.cumsum(counts)

    occ = np.zeros(n, dtype=np.int32)
    idx_sorted = np.argsort(inverse, kind="stable")
    for v_idx in range(n_vals):
        start = int(cum_counts[v_idx])
        end = int(cum_counts[v_idx + 1])
        occ[idx_sorted[start:end]] = np.arange(end - start, dtype=np.int32)

    lf = cum_counts[inverse] + occ.astype(np.int64)

    result = np.zeros(n, dtype=transformed.dtype)
    idx = int(primary_index)
    for i in range(n - 1, -1, -1):
        result[i] = transformed[idx]
        idx = int(lf[idx])
    return result


def mtf_encode(data: np.ndarray) -> np.ndarray:
    """Move-To-Front — vectorized inner loop with argmax + array slice."""
    flat = data.ravel()
    if len(flat) == 0:
        return np.array([], dtype=np.int32)
    n = len(flat)
    result = np.zeros(n, dtype=np.int32)
    alphabet = np.arange(256, dtype=np.int32)
    for i in range(n):
        v = int(flat[i]) & 0xFF
        if v != int(alphabet[0]):
            idx = int(np.argmax(alphabet == v))
            result[i] = idx
            alphabet[1 : idx + 1] = alphabet[:idx]
            alphabet[0] = v
        else:
            result[i] = 0
    return result


def mtf_decode(encoded: np.ndarray) -> np.ndarray:
    """Inverse MTF — vectorized inner loop with array slice."""
    if len(encoded) == 0:
        return np.array([], dtype=np.int32)
    n = len(encoded)
    result = np.zeros(n, dtype=np.int32)
    alphabet = np.arange(256, dtype=np.int32)
    for i in range(n):
        idx = int(encoded[i])
        if idx >= 256:
            idx = 0
        val = int(alphabet[idx])
        result[i] = val
        if idx > 0:
            alphabet[1 : idx + 1] = alphabet[:idx]
            alphabet[0] = val
    return result


def rle_encode(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fully vectorized RLE — returns (values, counts) arrays via diff + reduceat."""
    flat = data.ravel()
    if len(flat) == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32)
    diffs = np.diff(flat.astype(np.int64))
    change_idx = np.where(diffs != 0)[0].astype(np.int64) + 1
    starts = np.concatenate([np.array([0], dtype=np.int64), change_idx])
    ends = np.concatenate([change_idx, np.array([len(flat)], dtype=np.int64)])
    values = flat[starts.astype(np.intp)]
    counts = (ends - starts).astype(np.int32)
    return values, counts


def rle_decode(values: np.ndarray, counts: np.ndarray, n: int) -> np.ndarray:
    """Vectorized RLE decode via np.repeat."""
    if len(values) == 0 or n == 0:
        return np.array([], dtype=np.int32)
    return np.repeat(
        values,
        np.minimum(
            counts, np.maximum(0, n - np.concatenate([[0], np.cumsum(counts)[:-1]]))
        ).clip(0),
    )[:n].astype(np.int32)


def bwt_mtf_rle_encode(data: np.ndarray) -> Tuple[bytes, dict]:
    """Combined BWT + MTF + RLE — fully vectorized pipeline."""
    flat = data.ravel()
    if len(flat) == 0:
        return b"", {"n_orig": 0, "primary": 0}
    as_bytes = flat.astype(np.int32) & 0xFF
    transformed, primary = bwt_encode(as_bytes)
    mtf_out = mtf_encode(transformed)
    values, counts = rle_encode(mtf_out)

    buf = bytearray()
    buf.extend(_write_varint(primary))
    buf.extend(_write_varint(len(values)))
    for v, c in zip(values.tolist(), counts.tolist()):
        buf.extend(_write_varint(v & 0xFF))
        buf.extend(_write_varint(c))
    metadata = {"n_orig": len(flat), "primary": primary, "n_runs": len(values)}
    return bytes(buf), metadata


def bwt_mtf_rle_decode(compressed: bytes, metadata: dict) -> np.ndarray:
    """Decode BWT+MTF+RLE compressed data."""
    n_orig = metadata.get("n_orig", 0)
    if n_orig == 0:
        return np.array([], dtype=np.int32)
    offset = 0
    primary, offset = _read_varint(compressed, offset)
    n_runs, offset = _read_varint(compressed, offset)
    run_vals = np.zeros(n_runs, dtype=np.int32)
    run_counts = np.zeros(n_runs, dtype=np.int32)
    for j in range(n_runs):
        if offset >= len(compressed):
            break
        val, offset = _read_varint(compressed, offset)
        count, offset = _read_varint(compressed, offset)
        run_vals[j] = val
        run_counts[j] = count
    mtf_enc = rle_decode(run_vals, run_counts, n_orig)
    bwt_data = mtf_decode(mtf_enc)
    original = bwt_decode(bwt_data, int(primary))
    return original
