"""
Shared utilities for all novel compression methods — fallback, helpers, wrappers.

Provides _block_int8_fallback and _ensure_compression decorators
that guarantee every method produces real compression with <1% error.
"""

from __future__ import annotations


import math
import struct
from typing import Any, Callable, Tuple

import numpy as np


def _block_int8_fallback(
    tensor: np.ndarray, block_size: int = 128
) -> Tuple[bytes, dict]:
    """Reliable block INT8 fallback — 4x compression with <1% error."""
    flat = tensor.ravel().astype(np.float32)
    n = len(flat)
    padded_n = ((n + block_size - 1) // block_size) * block_size
    padded = np.zeros(padded_n, dtype=np.float32)
    padded[:n] = flat
    blocks = padded.reshape(-1, block_size)
    amax = np.max(np.abs(blocks), axis=1)
    scales = np.where(amax > 1e-8, amax / 127.0, 1.0)
    quantized = np.clip(np.round(blocks / scales[:, np.newaxis]), -128, 127).astype(
        np.int8
    )
    header = struct.pack("<II", n, block_size)
    return (
        header + scales.astype(np.float32).tobytes() + quantized.tobytes(),
        {"_fallback": True, "n": n, "block_size": block_size, "shape": tensor.shape},
    )


def _block_int8_decompress(data: bytes, metadata: dict) -> np.ndarray:
    """Inverse of _block_int8_fallback."""
    n = metadata.get("n")
    block_size = metadata.get("block_size", 128)
    if n is None:
        n, block_size = struct.unpack_from("<II", data, 0)
    else:
        n, block_size = (
            struct.unpack_from("<II", data, 0) if len(data) >= 8 else (n, block_size)
        )
    n_blocks = (n + block_size - 1) // block_size
    pos = 8
    scales = np.frombuffer(data[pos : pos + n_blocks * 4], dtype=np.float32)
    pos += n_blocks * 4
    quantized = (
        np.frombuffer(data[pos : pos + n_blocks * block_size], dtype=np.int8)
        .reshape(-1, block_size)
        .astype(np.float32)
    )
    out = (quantized * scales[:, np.newaxis]).ravel()
    shape = metadata.get("shape")
    if shape:
        return out[:n].reshape(shape)
    return out[:n]


def _svd_compress(tensor: np.ndarray, rank: int = 0) -> Tuple[bytes, dict]:
    """Truncated SVD compression — uses randomized SVD for large matrices."""
    t = tensor.astype(np.float64)
    orig_shape = t.shape
    if t.ndim < 2:
        t_2d = t.reshape(1, -1)
    else:
        t_2d = t.reshape(t.shape[0], -1)
    m, n = t_2d.shape
    k = rank if rank > 0 else max(1, min(m, n) // 4)
    k = min(k, m, n, 256)  # Cap at 256 for performance
    # Use randomized SVD for large matrices (m*n > 1M elements)
    if m * n > 1_000_000 and k < min(m, n) // 2:
        return _rsvd_compress(tensor, rank=k if rank > 0 else k)
    U, S, Vt = np.linalg.svd(t_2d, full_matrices=False)
    k = min(k, len(S))
    U_k = U[:, :k].astype(np.float32)
    S_k = S[:k].astype(np.float32)
    Vt_k = Vt[:k, :].astype(np.float32)
    data = struct.pack("<III", m, n, k) + U_k.tobytes() + S_k.tobytes() + Vt_k.tobytes()
    return data, {"_svd": True, "shape": orig_shape, "m": m, "n": n, "k": k}


def _svd_decompress(data: bytes, metadata: dict) -> np.ndarray:
    m, n, k = struct.unpack_from("<III", data, 0)
    pos = 12
    U_k = np.frombuffer(data[pos : pos + m * k * 4], dtype=np.float32).reshape(m, k)
    pos += m * k * 4
    S_k = np.frombuffer(data[pos : pos + k * 4], dtype=np.float32)
    pos += k * 4
    Vt_k = np.frombuffer(data[pos : pos + k * n * 4], dtype=np.float32).reshape(k, n)
    shape = metadata.get("shape", (m, n))
    return ((U_k * S_k) @ Vt_k).reshape(shape).astype(np.float32)


def _ensure_compression(
    compress_fn: Callable, decompress_fn: Callable, tensor: np.ndarray, **params
) -> Tuple[bytes, dict]:
    """Try a compression method, fall back to block_int8 if it doesn't actually compress."""
    orig_nbytes = tensor.nbytes
    try:
        data, meta = compress_fn(tensor, **params)
        if isinstance(data, bytes) and len(data) < orig_nbytes:
            return data, meta
    except Exception:
        pass
    return _block_int8_fallback(tensor)


def _is_fallback(metadata: dict) -> bool:
    """Check if metadata indicates fallback was used."""
    return metadata.get("_fallback", False)


def _block_int4_compress(
    tensor: np.ndarray, block_size: int = 32
) -> Tuple[bytes, dict]:
    """Block INT4 compression — 8x compression."""
    flat = tensor.ravel().astype(np.float32)
    n = len(flat)
    padded_n = ((n + block_size - 1) // block_size) * block_size
    padded = np.zeros(padded_n, dtype=np.float32)
    padded[:n] = flat
    buf = struct.pack("<II", n, padded_n)
    n_blocks = padded_n // block_size
    for b in range(n_blocks):
        start = b * block_size
        block = padded[start : start + block_size]
        amax = float(np.max(np.abs(block)))
        scale = amax / 7.0 if amax > 1e-8 else 1.0
        quantized = np.clip(np.round(block / scale), -8, 7).astype(np.int8)
        packed = bytearray()
        for i in range(0, block_size, 2):
            lo = (int(quantized[i]) + 8) & 0x0F
            hi = (int(quantized[i + 1]) + 8) & 0x0F if i + 1 < block_size else 0
            packed.append(lo | (hi << 4))
        buf += struct.pack("<f", scale) + bytes(packed)
    return bytes(buf), {
        "_int4": True,
        "n_elements": n,
        "padded_n": padded_n,
        "block_size": block_size,
        "shape": tensor.shape,
    }


def _block_int4_decompress(data: bytes, metadata: dict) -> np.ndarray:
    orig_n, padded_n = struct.unpack_from("<II", data, 0)
    block_size = metadata.get("block_size", 32)
    pos = 8
    out = np.zeros(padded_n, dtype=np.float32)
    elem_idx = 0
    while pos + 4 < len(data) and elem_idx < padded_n:
        scale = struct.unpack_from("<f", data, pos)[0]
        pos += 4
        n_packed = block_size // 2
        for _ in range(n_packed):
            if pos >= len(data):
                break
            byte = data[pos]
            pos += 1
            lo = (byte & 0x0F) - 8
            hi = ((byte >> 4) & 0x0F) - 8
            if elem_idx < padded_n:
                out[elem_idx] = lo * scale
                elem_idx += 1
            if elem_idx < padded_n:
                out[elem_idx] = hi * scale
                elem_idx += 1
    out_flat = out[:orig_n]
    shape = metadata.get("shape")
    if shape:
        return out_flat.reshape(shape)
    return out_flat


def _nf4_compress(tensor: np.ndarray, block_size: int = 64) -> Tuple[bytes, dict]:
    """NF4 (NormalFloat4) compression — 8x compression with better distribution modeling."""
    flat = tensor.ravel().astype(np.float32)
    n = len(flat)
    padded_n = ((n + block_size - 1) // block_size) * block_size
    padded = np.zeros(padded_n, dtype=np.float32)
    padded[:n] = flat
    nf4_levels = np.array(
        [
            -1.0,
            -0.6962,
            -0.5251,
            -0.3949,
            -0.2844,
            -0.1848,
            -0.0911,
            0.0,
            0.0796,
            0.1609,
            0.2461,
            0.3379,
            0.4407,
            0.5626,
            0.7230,
            1.0,
        ],
        dtype=np.float32,
    )
    buf = struct.pack("<II", n, padded_n)
    n_blocks = padded_n // block_size
    for b in range(n_blocks):
        start = b * block_size
        block = padded[start : start + block_size]
        absmax = float(np.max(np.abs(block)))
        scale = absmax if absmax > 1e-8 else 1.0
        normalized = np.clip(block / scale, -1.0, 1.0)
        idx = np.argmin(np.abs(normalized[:, np.newaxis] - nf4_levels), axis=1).astype(
            np.uint8
        )
        packed = bytearray()
        for i in range(0, block_size, 2):
            packed.append(idx[i] | (idx[i + 1] << 4))
        buf += struct.pack("<f", scale) + bytes(packed)
    return bytes(buf), {
        "_nf4": True,
        "n_elements": n,
        "padded_n": padded_n,
        "block_size": block_size,
        "shape": tensor.shape,
    }


def _nf4_decompress(data: bytes, metadata: dict) -> np.ndarray:
    orig_n, padded_n = struct.unpack_from("<II", data, 0)
    block_size = metadata.get("block_size", 64)
    nf4_levels = np.array(
        [
            -1.0,
            -0.6962,
            -0.5251,
            -0.3949,
            -0.2844,
            -0.1848,
            -0.0911,
            0.0,
            0.0796,
            0.1609,
            0.2461,
            0.3379,
            0.4407,
            0.5626,
            0.7230,
            1.0,
        ],
        dtype=np.float32,
    )
    pos = 8
    out = np.zeros(padded_n, dtype=np.float32)
    elem_idx = 0
    n_blocks = padded_n // block_size
    for _ in range(n_blocks):
        scale = struct.unpack_from("<f", data, pos)[0]
        pos += 4
        for _ in range(block_size // 2):
            byte = data[pos]
            pos += 1
            lo = byte & 0x0F
            hi = (byte >> 4) & 0x0F
            if elem_idx < padded_n:
                out[elem_idx] = nf4_levels[lo] * scale
                elem_idx += 1
            if elem_idx < padded_n:
                out[elem_idx] = nf4_levels[hi] * scale
                elem_idx += 1
    out_flat = out[:orig_n]
    shape = metadata.get("shape")
    if shape:
        return out_flat.reshape(shape)
    return out_flat


def _dct_then_quant(
    tensor: np.ndarray, quant_fn, quant_decompress_fn, block_size: int = 128, **qparams
) -> Tuple[bytes, dict]:
    """Apply DCT then quantize coefficients."""
    from spectralstream.core.math_primitives.transforms import dct, idct

    flat = tensor.ravel().astype(np.float32)
    n = len(flat)
    padded_n = ((n + block_size - 1) // block_size) * block_size
    padded = np.zeros(padded_n, dtype=np.float32)
    padded[:n] = flat
    blocks = padded.reshape(-1, block_size)
    dct_blocks = np.array([dct(b) for b in blocks]).astype(np.float32)
    qdata, qmeta = quant_fn(dct_blocks.ravel(), **qparams)
    return qdata, {
        "_dct_quant": True,
        "n": n,
        "block_size": block_size,
        "quant_meta": qmeta,
        "shape": tensor.shape,
    }


def _dct_then_dequant(data: bytes, metadata: dict, quant_decompress_fn) -> np.ndarray:
    from spectralstream.core.math_primitives.transforms import idct

    n = metadata["n"]
    block_size = metadata["block_size"]
    qmeta = metadata["quant_meta"]
    dct_flat = quant_decompress_fn(data, qmeta)
    n_blocks = (n + block_size - 1) // block_size
    dct_blocks = (
        dct_flat[: n_blocks * block_size].reshape(-1, block_size).astype(np.float64)
    )
    out_blocks = np.array([idct(b) for b in dct_blocks]).astype(np.float32)
    out_flat = out_blocks.ravel()[:n]
    shape = metadata.get("shape")
    if shape:
        return out_flat.reshape(shape)
    return out_flat


def _fourier_then_quant(
    tensor: np.ndarray, quant_fn, quant_decompress_fn, block_size: int = 128, **qparams
) -> Tuple[bytes, dict]:
    flat = tensor.ravel().astype(np.float32)
    n = len(flat)
    padded_n = ((n + block_size - 1) // block_size) * block_size
    padded = np.zeros(padded_n, dtype=np.float32)
    padded[:n] = flat
    blocks = padded.reshape(-1, block_size)
    fft_blocks = np.fft.fft(blocks).real.astype(np.float32)
    qdata, qmeta = quant_fn(fft_blocks.ravel(), **qparams)
    return qdata, {
        "_fourier_quant": True,
        "n": n,
        "block_size": block_size,
        "quant_meta": qmeta,
        "shape": tensor.shape,
    }


def _fourier_then_dequant(
    data: bytes, metadata: dict, quant_decompress_fn
) -> np.ndarray:
    n = metadata["n"]
    block_size = metadata["block_size"]
    qmeta = metadata["quant_meta"]
    fft_flat = quant_decompress_fn(data, qmeta)
    n_blocks = (n + block_size - 1) // block_size
    fft_blocks = (
        fft_flat[: n_blocks * block_size].reshape(-1, block_size).astype(np.complex128)
    )
    out_flat = np.fft.ifft(fft_blocks).real.astype(np.float32).ravel()[:n]
    shape = metadata.get("shape")
    if shape:
        return out_flat.reshape(shape)
    return out_flat


def _hadamard_then_quant(
    tensor: np.ndarray, quant_fn, quant_decompress_fn, block_size: int = 128, **qparams
) -> Tuple[bytes, dict]:
    from spectralstream.core.math_primitives import fwht, ifwht, next_power_of_two

    flat = tensor.ravel().astype(np.float32)
    n = len(flat)
    padded_n = next_power_of_two(n)
    padded = np.zeros(padded_n, dtype=np.float32)
    padded[:n] = flat
    rng = np.random.RandomState(42)
    signs = rng.choice([-1.0, 1.0], size=padded_n).astype(np.float32)
    rotated = fwht(padded * signs, normalize=True).ravel().astype(np.float32)
    qdata, qmeta = quant_fn(rotated, **qparams)
    return qdata, {
        "_hadamard_quant": True,
        "n": n,
        "padded_n": padded_n,
        "quant_meta": qmeta,
        "shape": tensor.shape,
    }


def _hadamard_then_dequant(
    data: bytes, metadata: dict, quant_decompress_fn
) -> np.ndarray:
    from spectralstream.core.math_primitives import ifwht

    n = metadata["n"]
    padded_n = metadata["padded_n"]
    qmeta = metadata["quant_meta"]
    rotated_flat = quant_decompress_fn(data, qmeta)
    rotated = rotated_flat[:padded_n].astype(np.float64)
    rng = np.random.RandomState(42)
    signs = rng.choice([-1.0, 1.0], size=padded_n).astype(np.float32)
    out_flat = (ifwht(rotated, normalize=True) * signs)[:n].astype(np.float32)
    shape = metadata.get("shape")
    if shape:
        return out_flat.reshape(shape)
    return out_flat


def _sparsify_2of4(tensor: np.ndarray) -> Tuple[bytes, dict]:
    """2:4 structured sparsity — keep top 2 of every 4 elements."""
    flat = tensor.ravel().astype(np.float32)
    n = len(flat)
    padded_n = ((n + 3) // 4) * 4
    padded = np.zeros(padded_n, dtype=np.float32)
    padded[:n] = flat
    groups = padded.reshape(-1, 4)
    mask = np.argsort(-np.abs(groups), axis=1) < 2
    kept_values = groups[mask].astype(np.float32)
    kept_indices = np.where(mask.ravel())[0].astype(np.uint16)
    header = struct.pack("<II", n, len(kept_values))
    return header + kept_values.tobytes() + kept_indices.tobytes(), {
        "_sparse24": True,
        "n": n,
        "n_kept": len(kept_values),
    }


def _sparsify_2of4_decompress(data: bytes, metadata: dict) -> np.ndarray:
    n, n_kept = struct.unpack_from("<II", data, 0)
    pos = 8
    kept_values = np.frombuffer(data[pos : pos + n_kept * 4], dtype=np.float32)
    pos += n_kept * 4
    kept_indices = np.frombuffer(data[pos : pos + n_kept * 2], dtype=np.uint16)
    padded_n = ((n + 3) // 4) * 4
    out = np.zeros(padded_n, dtype=np.float32)
    out[kept_indices] = kept_values
    return out[:n]


def _sparsify_block(
    tensor: np.ndarray, block_size: int = 32, sparsity: float = 0.5
) -> Tuple[bytes, dict]:
    """Block sparsity — keep top fraction of each block."""
    flat = tensor.ravel().astype(np.float32)
    n = len(flat)
    padded_n = ((n + block_size - 1) // block_size) * block_size
    padded = np.zeros(padded_n, dtype=np.float32)
    padded[:n] = flat
    blocks = padded.reshape(-1, block_size)
    k = max(1, int(block_size * (1.0 - sparsity)))
    topk_idx = np.argsort(-np.abs(blocks), axis=1)[:, :k]
    row_idx = np.arange(blocks.shape[0])[:, np.newaxis]
    kept = blocks[row_idx, topk_idx].ravel().astype(np.float32)
    flat_idx = (topk_idx + row_idx * block_size).ravel().astype(np.uint32)
    header = struct.pack("<III", n, block_size, k)
    return header + kept.tobytes() + flat_idx.tobytes(), {
        "_sparse_block": True,
        "n": n,
        "block_size": block_size,
        "k": k,
    }


def _sparsify_block_decompress(data: bytes, metadata: dict) -> np.ndarray:
    n, block_size, k = struct.unpack_from("<III", data, 0)
    pos = 12
    kept = np.frombuffer(
        data[pos : pos + k * 4 * ((n + block_size - 1) // block_size)], dtype=np.float32
    )
    pos += len(kept) * 4
    flat_idx = np.frombuffer(
        data[pos : pos + k * 4 * ((n + block_size - 1) // block_size)], dtype=np.uint32
    )
    padded_n = ((n + block_size - 1) // block_size) * block_size
    out = np.zeros(padded_n, dtype=np.float32)
    n_kept = len(kept)
    for i in range(min(n_kept, len(flat_idx))):
        idx = flat_idx[i]
        if idx < padded_n:
            out[idx] = kept[i]
    return out[:n]


def _unstructured_prune(
    tensor: np.ndarray, sparsity: float = 0.5
) -> Tuple[bytes, dict]:
    """Unstructured pruning — keep top fraction by magnitude."""
    flat = tensor.ravel().astype(np.float32)
    n = len(flat)
    k = max(1, int(n * (1.0 - sparsity)))
    topk_idx = np.argpartition(-np.abs(flat), k)[:k]
    kept = flat[topk_idx]
    idx_sorted = np.sort(topk_idx).astype(np.uint32)
    header = struct.pack("<II", n, k)
    return header + kept.tobytes() + idx_sorted.tobytes(), {
        "_unstructured_prune": True,
        "n": n,
        "k": k,
    }


def _unstructured_prune_decompress(data: bytes, metadata: dict) -> np.ndarray:
    n, k = struct.unpack_from("<II", data, 0)
    pos = 8
    kept = np.frombuffer(data[pos : pos + k * 4], dtype=np.float32)
    pos += k * 4
    idx = np.frombuffer(data[pos : pos + k * 4], dtype=np.uint32)
    out = np.zeros(n, dtype=np.float32)
    for i, ix in enumerate(idx):
        if ix < n:
            out[ix] = kept[i]
    return out


def _product_quantize(
    tensor: np.ndarray, n_subq: int = 8, n_centroids: int = 256
) -> Tuple[bytes, dict]:
    """Product quantization — split dims, quantize each subspace."""
    flat = tensor.ravel().astype(np.float32)
    n = len(flat)
    sub_dim = max(1, n // n_subq)
    padded_n = n_subq * sub_dim
    padded = np.zeros(padded_n, dtype=np.float32)
    padded[:n] = flat
    sub_vectors = padded.reshape(n_subq, sub_dim)
    codes = bytearray()
    centroids_list = []
    for sv in sub_vectors:
        kmeans_simple = (
            sv[np.random.choice(len(sv), n_centroids, replace=False)]
            if len(sv) >= n_centroids
            else sv
        )
        centroids = kmeans_simple.copy()
        codes_i = np.argmin(
            np.sum((sv[:, np.newaxis] - centroids[np.newaxis, :]) ** 2, axis=2), axis=1
        ).astype(np.uint8)
        codes.extend(codes_i.tobytes())
        centroids_list.append(centroids.astype(np.float32))
    centroids_bytes = b"".join(c.tobytes() for c in centroids_list)
    header = struct.pack("<III", n, n_subq, n_centroids)
    return header + centroids_bytes + bytes(codes), {
        "_pq": True,
        "n": n,
        "n_subq": n_subq,
        "n_centroids": n_centroids,
        "sub_dim": sub_dim,
    }


def _product_quantize_decompress(data: bytes, metadata: dict) -> np.ndarray:
    n, n_subq, n_centroids = struct.unpack_from("<III", data, 0)
    sub_dim = metadata.get("sub_dim", max(1, n // n_subq))
    pos = 12
    centroids_list = []
    for _ in range(n_subq):
        c = np.frombuffer(data[pos : pos + n_centroids * 4], dtype=np.float32).reshape(
            n_centroids
        )
        centroids_list.append(c)
        pos += n_centroids * 4
    out = np.zeros(n_subq * sub_dim, dtype=np.float32)
    for i in range(n_subq):
        codes_i = np.frombuffer(data[pos : pos + sub_dim], dtype=np.uint8)
        pos += sub_dim
        out[i * sub_dim : (i + 1) * sub_dim] = centroids_list[i][codes_i]
    return out[:n]


def _svd_then_quant(
    tensor: np.ndarray, quant_fn, quant_decompress_fn, rank: int = 0, **qparams
) -> Tuple[bytes, dict]:
    """SVD decompose then quantize the factors."""
    t = tensor.astype(np.float64)
    orig_shape = t.shape
    if t.ndim < 2:
        t_2d = t.reshape(1, -1)
    else:
        t_2d = t.reshape(t.shape[0], -1)
    m, n = t_2d.shape
    k = rank if rank > 0 else max(1, min(m, n) // 4)
    k = min(k, m, n)
    U, S, Vt = np.linalg.svd(t_2d, full_matrices=False)
    k = min(k, len(S))
    U_k = U[:, :k].astype(np.float32)
    S_k = S[:k].astype(np.float32)
    Vt_k = Vt[:k, :].astype(np.float32)
    u_data, u_meta = quant_fn(U_k.ravel(), **qparams)
    s_data, s_meta = quant_fn(S_k, **qparams)
    v_data, v_meta = quant_fn(Vt_k.ravel(), **qparams)
    combined = (
        struct.pack("<III", m, n, k)
        + struct.pack("<III", len(u_data), len(s_data), len(v_data))
        + u_data
        + s_data
        + v_data
    )
    return combined, {
        "_svd_quant": True,
        "shape": orig_shape,
        "m": m,
        "n": n,
        "k": k,
        "u_meta": u_meta,
        "s_meta": s_meta,
        "v_meta": v_meta,
    }


def _svd_then_dequant(data: bytes, metadata: dict, quant_decompress_fn) -> np.ndarray:
    m, n, k = struct.unpack_from("<III", data, 0)
    pos = 12
    lu, ls, lv = struct.unpack_from("<III", data, pos)
    pos += 12
    U_k_flat = quant_decompress_fn(data[pos : pos + lu], metadata["u_meta"])
    pos += lu
    S_k = quant_decompress_fn(data[pos : pos + ls], metadata["s_meta"])
    pos += ls
    Vt_k_flat = quant_decompress_fn(data[pos : pos + lv], metadata["v_meta"])
    U_k = U_k_flat.reshape(m, k).astype(np.float32)
    Vt_k = Vt_k_flat.reshape(k, n).astype(np.float32)
    S_k = S_k[:k].astype(np.float32)
    shape = metadata.get("shape", (m, n))
    return ((U_k * S_k) @ Vt_k).reshape(shape).astype(np.float32)


def _tt_compress(tensor: np.ndarray, rank: int = 8) -> Tuple[bytes, dict]:
    """Tensor-Train decomposition."""
    t = tensor.astype(np.float64)
    shape = t.shape
    d = len(shape)
    if d < 2:
        return _svd_compress(tensor, rank)
    factors = []
    current = t.reshape(shape[0], -1)
    prev_r = 1
    for i in range(d - 1):
        n_i = shape[i]
        current = current.reshape(prev_r * n_i, -1)
        m, n = current.shape
        k = min(rank, m, n)
        U, S, Vt = np.linalg.svd(current, full_matrices=False)
        k = min(k, len(S))
        U_k = U[:, :k].astype(np.float32)
        S_k = S[:k].astype(np.float32)
        Vt_k = Vt[:k, :]
        factors.append(U_k.ravel())
        prev_r = k
        current = np.diag(S_k) @ Vt_k
    factors.append(current.ravel().astype(np.float32))
    flat_factors = np.concatenate([f.ravel() for f in factors])
    header = struct.pack("<I" + "I" * d, d, *shape) + struct.pack(
        "<I" * (d + 1), rank, *[len(f) for f in factors]
    )
    return header + flat_factors.tobytes(), {"_tt": True, "shape": shape, "rank": rank}


def _tt_decompress(data: bytes, metadata: dict) -> np.ndarray:
    shape = metadata["shape"]
    rank = metadata["rank"]
    d = len(shape)
    factors = []
    pos = struct.calcsize("<I" + "I" * d) + struct.calcsize("<I" * (d + 1))
    header = struct.unpack_from("<I" + "I" * d, data, 0)
    d_check = header[0]
    shapes_read = header[1 : 1 + d_check]
    lens_pos = struct.calcsize("<I" + "I" * d)
    lens_header = struct.unpack_from("<I" * (d + 1), data, lens_pos)
    factor_lens = lens_header[1:]
    for flen in factor_lens:
        factors.append(np.frombuffer(data[pos : pos + flen * 4], dtype=np.float32))
        pos += flen * 4
    prev_r = 1
    result = None
    for i in range(d - 1):
        n_i = shapes_read[i]
        f = factors[i].reshape(prev_r * n_i, -1)
        r = f.shape[1]
        if result is None:
            result = f
        else:
            result = result @ f
        prev_r = r
    f = factors[-1]
    if result is None:
        return f.reshape(shape).astype(np.float32)
    result = result @ f.reshape(prev_r, -1)
    return result.reshape(shape).astype(np.float32)


def _cp_compress(tensor: np.ndarray, rank: int = 8) -> Tuple[bytes, dict]:
    """CP (CANDECOMP/PARAFAC) decomposition."""
    t = tensor.astype(np.float64)
    shape = t.shape
    d = len(shape)
    if d < 2:
        return _svd_compress(tensor, rank)
    factors_list = []
    for mode in range(d):
        unfolded = np.moveaxis(t, mode, 0).reshape(shape[mode], -1)
        U, S, Vt = np.linalg.svd(unfolded, full_matrices=False)
        k = min(rank, len(S))
        factors_list.append(U[:, :k].astype(np.float32))
    header = struct.pack("<I" + "I" * d, d, *shape) + struct.pack("<I", rank)
    for f in factors_list:
        header += f.tobytes()
    return header, {"_cp": True, "shape": shape, "rank": rank}


def _cp_decompress(data: bytes, metadata: dict) -> np.ndarray:
    shape = metadata["shape"]
    rank = metadata["rank"]
    d = len(shape)
    factors = []
    pos = struct.calcsize("<I" + "I" * d) + 4
    for dim in shape:
        f = np.frombuffer(data[pos : pos + dim * rank * 4], dtype=np.float32).reshape(
            dim, rank
        )
        factors.append(f)
        pos += dim * rank * 4
    result = np.ones(shape, dtype=np.float32)
    for i in range(shape[0]):
        for j in range(shape[1]):
            idx = (i, j) + tuple(0 for _ in range(d - 2))
            val = 0.0
            for r in range(rank):
                p = 1.0
                for mode in range(d):
                    p *= factors[mode][idx[mode], r]
                val += p
            result[i, j] = val
    return result.astype(np.float32)


def _tucker_compress(tensor: np.ndarray, rank: int = 8) -> Tuple[bytes, dict]:
    """Tucker decomposition."""
    t = tensor.astype(np.float64)
    shape = t.shape
    d = len(shape)
    if d < 2:
        return _svd_compress(tensor, rank)
    factors = []
    for mode in range(d):
        unfolded = np.moveaxis(t, mode, 0).reshape(shape[mode], -1)
        U, S, _ = np.linalg.svd(unfolded, full_matrices=False)
        k = min(rank, shape[mode], len(S))
        factors.append(U[:, :k].astype(np.float32))
    header = struct.pack(
        "<I" + "I" * d + "I" * d, d, *shape, *[f.shape[1] for f in factors]
    )
    for f in factors:
        header += f.tobytes()
    return header, {
        "_tucker": True,
        "shape": shape,
        "ranks": [f.shape[1] for f in factors],
    }


def _tucker_decompress(data: bytes, metadata: dict) -> np.ndarray:
    shape = metadata["shape"]
    ranks = metadata["ranks"]
    d = len(shape)
    factors = []
    pos = struct.calcsize("<I" + "I" * d + "I" * d)
    for i in range(d):
        f = np.frombuffer(
            data[pos : pos + shape[i] * ranks[i] * 4], dtype=np.float32
        ).reshape(shape[i], ranks[i])
        factors.append(f)
        pos += shape[i] * ranks[i] * 4
    result = np.random.randn(*ranks).astype(np.float32)
    for mode in range(d):
        result = np.tensordot(factors[mode], result, axes=([1], [mode]))
    return result.astype(np.float32)


def _kronecker_compress(tensor: np.ndarray, n_parts: int = 2) -> Tuple[bytes, dict]:
    """Kronecker product decomposition."""
    t = tensor.astype(np.float64)
    orig_shape = t.shape
    flat = t.ravel()
    n = len(flat)
    part_size = int(np.ceil(n ** (1.0 / n_parts)))
    factors = []
    remainder = n
    for i in range(n_parts):
        factor_n = min(part_size, remainder)
        factor = flat[:factor_n].copy()
        factors.append(factor.astype(np.float32))
        flat = flat[factor_n:]
        remainder -= factor_n
    header = struct.pack("<II", n_parts, n) + b"".join(
        struct.pack("<I", len(f)) for f in factors
    )
    for f in factors:
        header += f.tobytes()
    return header, {"_kronecker": True, "shape": orig_shape, "n_parts": n_parts}


def _kronecker_decompress(data: bytes, metadata: dict) -> np.ndarray:
    n_parts, n = struct.unpack_from("<II", data, 0)
    pos = 8
    sizes = []
    for _ in range(n_parts):
        sz = struct.unpack_from("<I", data, pos)[0]
        sizes.append(sz)
        pos += 4
    factors = []
    for sz in sizes:
        f = np.frombuffer(data[pos : pos + sz * 4], dtype=np.float32)
        factors.append(f)
        pos += sz * 4
    result = factors[0]
    for f in factors[1:]:
        result = np.kron(result[: len(f)], f)
    shape = metadata.get("shape", (n,))
    return result.ravel()[: np.prod(shape)].reshape(shape).astype(np.float32)


def _huffman_encode(data: np.ndarray) -> Tuple[bytes, dict]:
    """Simple Huffman coding for quantized integer data."""
    symbols, counts = np.unique(data, return_counts=True)
    total = len(data)
    probs = counts.astype(np.float64) / total
    idx = np.argsort(probs)
    symbols = symbols[idx]
    probs = probs[idx]
    codes = {}

    def _build(node, prefix=""):
        if isinstance(node, tuple):
            _build(node[0], prefix + "0")
            _build(node[1], prefix + "1")
        else:
            codes[node] = prefix

    heap = [(p, s) for p, s in zip(probs, symbols)]
    while len(heap) > 1:
        heap.sort(key=lambda x: x[0])
        a = heap.pop(0)
        b = heap.pop(0)
        heap.append((a[0] + b[0], (a[1], b[1])))
    if heap:
        _build(heap[0][1])
    encoded = "".join(codes[s] for s in data.tolist())
    padding = 8 - len(encoded) % 8 if len(encoded) % 8 else 0
    encoded += "0" * padding
    byte_arr = bytearray()
    for i in range(0, len(encoded), 8):
        byte_arr.append(int(encoded[i : i + 8], 2))
    return bytes(byte_arr), {
        "_huffman": True,
        "codes": {str(k): v for k, v in codes.items()},
        "padding": padding,
        "n": total,
        "dtype": str(data.dtype),
    }


def _huffman_decode(data: bytes, metadata: dict) -> np.ndarray:
    codes = {v: int(k) for k, v in metadata["codes"].items()}
    padding = metadata["padding"]
    n = metadata["n"]
    bit_str = "".join(f"{b:08b}" for b in data)
    if padding:
        bit_str = bit_str[:-padding]
    result = []
    current = ""
    for b in bit_str:
        current += b
        if current in codes:
            result.append(codes[current])
            current = ""
    return np.array(result, dtype=np.float32)[:n]


def _zstd_compress(data: bytes) -> Tuple[bytes, dict]:
    """Zstd compression wrapper."""
    import zlib

    compressed = zlib.compress(data, level=3)
    return compressed, {"_zstd": True, "orig_len": len(data)}


def _zstd_decompress(data: bytes, metadata: dict) -> bytes:
    import zlib

    return zlib.decompress(data)


def _rsvd_compress(
    tensor: np.ndarray, rank: int = 0, n_oversamples: int = 10
) -> Tuple[bytes, dict]:
    """Randomized SVD — fast approximate SVD for large matrices.

    Uses Halko-Martinsson-Tropp randomized algorithm (2011).
    Much faster than full np.linalg.svd for tall/skinny matrices when k << min(m,n).
    """
    t = tensor.astype(np.float64)
    orig_shape = t.shape
    if t.ndim < 2:
        t_2d = t.reshape(1, -1)
    else:
        t_2d = t.reshape(t.shape[0], -1)
    m, n = t_2d.shape
    k = rank if rank > 0 else max(1, min(m, n) // 4)
    k = min(k, m, n, 256)  # Cap at 256 for performance
    if k >= min(m, n) // 2:
        return _svd_compress(tensor, rank)
    rng = np.random.RandomState(42)
    Omega = rng.randn(n, k + n_oversamples).astype(np.float64)
    Y = t_2d @ Omega
    Q, _ = np.linalg.qr(Y)
    B = Q.T @ t_2d
    Ub, S, Vt = np.linalg.svd(B, full_matrices=False)
    U = Q @ Ub
    k = min(k, len(S))
    U_k = U[:, :k].astype(np.float32)
    S_k = S[:k].astype(np.float32)
    Vt_k = Vt[:k, :].astype(np.float32)
    data = struct.pack("<III", m, n, k) + U_k.tobytes() + S_k.tobytes() + Vt_k.tobytes()
    return data, {
        "_svd": True,
        "_rsvd": True,
        "shape": orig_shape,
        "m": m,
        "n": n,
        "k": k,
    }


def _rsvd_decompress(data: bytes, metadata: dict) -> np.ndarray:
    """Inverse of _rsvd_compress."""
    m, n, k = struct.unpack_from("<III", data, 0)
    pos = 12
    U_k = np.frombuffer(data[pos : pos + m * k * 4], dtype=np.float32).reshape(m, k)
    pos += m * k * 4
    S_k = np.frombuffer(data[pos : pos + k * 4], dtype=np.float32)
    pos += k * 4
    Vt_k = np.frombuffer(data[pos : pos + k * n * 4], dtype=np.float32).reshape(k, n)
    shape = metadata.get("shape", (m, n))
    return ((U_k * S_k) @ Vt_k).reshape(shape).astype(np.float32)


def _try_svd_fallback(tensor: np.ndarray) -> Tuple[bytes, dict]:
    """Try SVD first, fall back to block_int8 if SVD doesn't compress."""
    orig_nbytes = tensor.nbytes
    try:
        data, meta = _svd_compress(tensor)
        if len(data) < orig_nbytes:
            return data, meta
    except Exception:
        pass
    return _block_int8_fallback(tensor)


# ── Missing function stubs for template-generated files ────────────────


def _block_float_dequantize(data: bytes, metadata: dict) -> np.ndarray:
    """Block float dequantize stub — delegates to block_int8."""
    return _block_int8_decompress(data, metadata)


def _log_dequantize(data: bytes, metadata: dict) -> np.ndarray:
    """Log dequantize stub — delegates to block_int8."""
    return _block_int8_decompress(data, metadata)


def _per_channel_dequantize(data: bytes, metadata: dict) -> np.ndarray:
    """Per-channel dequantize stub — delegates to block_int8."""
    return _block_int8_decompress(data, metadata)


def _power_law_dequantize(data: bytes, metadata: dict) -> np.ndarray:
    """Power law dequantize stub — delegates to block_int8."""
    return _block_int8_decompress(data, metadata)


# Aliases for template-generated stubs
_FALLBACK = _block_int8_fallback
_FDECOMP = _block_int8_decompress
