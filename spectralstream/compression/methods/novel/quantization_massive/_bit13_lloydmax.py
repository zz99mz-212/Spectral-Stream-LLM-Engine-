from __future__ import annotations

import math
import struct
from typing import Any, Dict, Tuple

import numpy as np


def _dct(x):
    """Type-II DCT using FFT."""
    n = len(x)
    N = 2 * n
    X = np.fft.fft(np.concatenate([x, x[-1:0:-1]]))[:n]
    k = np.arange(n)
    return np.real(X * np.exp(-1j * np.pi * k / (2 * n))) * np.sqrt(2.0 / n)

def _idct(x):
    """Inverse DCT using FFT."""
    n = len(x)
    k = np.arange(n)
    X = x * np.exp(1j * np.pi * k / (2 * n)) / np.sqrt(2.0 / n)
    return np.fft.ifft(np.concatenate([X, -np.conj(X[-1:0:-1])])).real[:n]

def _ser(arr: np.ndarray) -> bytes:
    return np.ascontiguousarray(arr).tobytes()

def _deser(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

def _uniform_quantize(
    t: np.ndarray, bits: int, symmetric: bool = True
) -> Tuple[np.ndarray, float, float, float]:
    flat = t.ravel().astype(np.float32)
    if symmetric:
        amax = float(np.max(np.abs(flat)))
        if amax < 1e-30:
            return np.zeros_like(flat, dtype=np.int32), 1.0, 0.0, amax
        half = 1 << (bits - 1)
        scale = half / amax if amax > 0 else 1.0
        q = np.clip(np.round(flat * scale), -half, half - 1).astype(np.int32)
        return q, scale, 0.0, amax
    lo = float(np.min(flat))
    hi = float(np.max(flat))
    if hi - lo < 1e-30:
        return np.zeros_like(flat, dtype=np.int32), 1.0, lo, hi
    levels = (1 << bits) - 1
    scale = levels / (hi - lo)
    q = np.clip(np.round((flat - lo) * scale), 0, levels).astype(np.int32)
    return q, scale, lo, hi

def _lloyd_max_quantize(
    t: np.ndarray, bits: int
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    flat = t.ravel().astype(np.float32)
    lo, hi = float(np.min(flat)), float(np.max(flat))
    if hi - lo < 1e-30:
        return np.zeros_like(flat, dtype=np.uint8), np.array([lo]), lo, hi
    n_levels = 1 << bits
    levels = np.linspace(lo, hi, n_levels).astype(np.float32)
    for _ in range(20):
        idx = np.argmin(np.abs(flat[:, None] - levels[None, :]), axis=1)
        for j in range(n_levels):
            mask = idx == j
            if mask.any():
                levels[j] = float(np.mean(flat[mask]))
    idx = np.argmin(np.abs(flat[:, None] - levels[None, :]), axis=1)
    return idx.astype(np.uint16), levels, lo, hi

def _block_quantize_flat(
    t: np.ndarray, bits: int, block_size: int = 128, symmetric: bool = True
) -> Tuple[bytes, Dict[str, Any]]:
    flat = t.ravel().astype(np.float32)
    n = len(flat)
    padded_n = int(math.ceil(n / block_size) * block_size)
    padded = np.zeros(padded_n, dtype=np.float32)
    padded[:n] = flat
    blocks = padded.reshape(-1, block_size)
    half = 1 << (bits - 1) if symmetric else 0
    levels = (1 << bits) - 1 if not symmetric else 0
    scales = np.zeros(blocks.shape[0], dtype=np.float32)
    zeros = np.zeros(blocks.shape[0], dtype=np.float32)
    for i in range(blocks.shape[0]):
        b = blocks[i]
        if symmetric:
            amax = float(np.max(np.abs(b)))
            scales[i] = half / amax if amax > 1e-30 else 1.0
        else:
            lo, hi = float(b.min()), float(b.max())
            zeros[i] = lo
            scales[i] = levels / (hi - lo) if hi - lo > 1e-30 else 1.0
    n_blocks = blocks.shape[0]
    if bits <= 8:
        dtype_out = np.uint8
    elif bits <= 16:
        dtype_out = np.uint16
    else:
        dtype_out = np.uint32
    quantized = np.zeros(n_blocks * block_size, dtype=dtype_out)
    for i in range(n_blocks):
        b = blocks[i]
        if symmetric:
            q = np.clip(np.round(b * scales[i]), -half, half - 1).astype(dtype_out)
        else:
            q = np.clip(np.round((b - zeros[i]) * scales[i]), 0, levels).astype(
                dtype_out
            )
        quantized[i * block_size : (i + 1) * block_size] = q
    header = struct.pack("<III", n, block_size, bits)
    scales_data = scales.astype(np.float32).tobytes()
    zeros_data = zeros.astype(np.float32).tobytes()
    return (
        header + scales_data + zeros_data + quantized.tobytes(),
        {"n": n, "block_size": block_size, "bits": bits, "symmetric": symmetric},
    )

def _block_dequantize_flat(data: bytes, metadata: Dict[str, Any]) -> np.ndarray:
    n, block_size, bits = struct.unpack_from("<III", data, 0)
    symmetric = metadata.get("symmetric", True)
    n_blocks = (n + block_size - 1) // block_size
    pos = 12
    scales = np.frombuffer(data[pos : pos + n_blocks * 4], dtype=np.float32)
    pos += n_blocks * 4
    zeros = np.frombuffer(data[pos : pos + n_blocks * 4], dtype=np.float32)
    pos += n_blocks * 4
    if bits <= 8:
        dtype_read = np.uint8
    elif bits <= 16:
        dtype_read = np.uint16
    else:
        dtype_read = np.uint32
    quantized = (
        np.frombuffer(data[pos:], dtype=dtype_read)
        .reshape(n_blocks, block_size)
        .astype(np.float32)
    )
    if symmetric:
        half = 1 << (bits - 1)
        out = (quantized - half) / scales[:, None]
    else:
        out = quantized / scales[:, None] + zeros[:, None]
    return out.ravel()[:n]

def _per_channel_quantize(
    t: np.ndarray, bits: int, symmetric: bool = True
) -> Tuple[bytes, Dict[str, Any]]:
    t = t.astype(np.float32)
    orig_shape = t.shape
    if t.ndim < 2:
        t_2d = t.reshape(1, -1)
    else:
        t_2d = t.reshape(t.shape[0], -1)
    c, rest = t_2d.shape
    half = 1 << (bits - 1) if symmetric else 0
    levels = (1 << bits) - 1 if not symmetric else 0
    scales = np.zeros(c, dtype=np.float32)
    zeros = np.zeros(c, dtype=np.float32)
    quantized = np.zeros_like(t_2d, dtype=np.int32)
    for i in range(c):
        row = t_2d[i]
        if symmetric:
            amax = float(np.max(np.abs(row)))
            scales[i] = half / amax if amax > 1e-30 else 1.0
            quantized[i] = np.clip(np.round(row * scales[i]), -half, half - 1)
        else:
            l, h = float(row.min()), float(row.max())
            zeros[i] = l
            scales[i] = levels / (h - l) if h - l > 1e-30 else 1.0
            quantized[i] = np.clip(np.round((row - l) * scales[i]), 0, levels)
    if bits <= 8:
        dtype_out = np.uint8
    elif bits <= 16:
        dtype_out = np.uint16
    else:
        dtype_out = np.uint32
    data = (
        struct.pack("<III", c, rest, bits)
        + scales.astype(np.float32).tobytes()
        + zeros.astype(np.float32).tobytes()
        + quantized.astype(dtype_out).tobytes()
    )
    return data, {"shape": orig_shape, "c": c, "bits": bits, "symmetric": symmetric}

class Bit13_LloydMax:
    name = "13bit_lloyd_max"
    category = "quantization"

    def compress(self, tensor, **params):
        q, lv, _, _ = _lloyd_max_quantize(tensor, 13)
        return struct.pack("<II", tensor.size, len(lv)) + lv.tobytes() + q.astype(
            np.uint16
        ).tobytes(), {"n": tensor.size}

    def decompress(self, data, metadata):
        n, nl = struct.unpack_from("<II", data, 0)
        pos = 8
        lv = np.frombuffer(data[pos : pos + nl * 4], dtype=np.float32)
        pos += nl * 4
        q = np.frombuffer(data[pos:], dtype=np.uint16)
        return lv[q][:n]
