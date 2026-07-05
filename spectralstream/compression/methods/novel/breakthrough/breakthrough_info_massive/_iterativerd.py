from __future__ import annotations

import math
import struct
from typing import Any, Dict, Tuple

import numpy as np

from spectralstream.compression.methods.novel._common import (
    _block_int8_fallback,
    _block_int8_decompress,
)


def _uniform_quantize_compress(
    tensor: np.ndarray, bits: int, block_size: int = 128
) -> Tuple[bytes, dict]:
    """Uniform quantization with block-wise scaling."""
    flat = tensor.ravel().astype(np.float32)
    n = len(flat)
    padded_n = ((n + block_size - 1) // block_size) * block_size
    padded = np.zeros(padded_n, dtype=np.float32)
    padded[:n] = flat
    blocks = padded.reshape(-1, block_size)
    half = 1 << (bits - 1)
    amax = np.max(np.abs(blocks), axis=1)
    scales = np.where(amax > 1e-8, half / amax, 1.0)
    quantized = np.clip(
        np.round(blocks * scales[:, np.newaxis]), -half, half - 1
    ).astype(np.int32 if bits > 8 else np.int8)
    header = struct.pack("<III", n, block_size, bits)
    return header + scales.astype(np.float32).tobytes() + quantized.tobytes(), {
        "n": n,
        "block_size": block_size,
        "bits": bits,
    }

def _uniform_quantize_decompress(data: bytes, metadata: dict) -> np.ndarray:
    n, block_size, bits = struct.unpack_from("<III", data, 0)
    n_blocks = (n + block_size - 1) // block_size
    pos = 12
    scales = np.frombuffer(data[pos : pos + n_blocks * 4], dtype=np.float32)
    pos += n_blocks * 4
    dtype_r = np.int32 if bits > 8 else np.int8
    half = 1 << (bits - 1)
    quantized = (
        np.frombuffer(data[pos:], dtype=dtype_r)
        .reshape(-1, block_size)
        .astype(np.float32)
    )
    out = (quantized / scales[:, np.newaxis]).ravel()
    return out[:n]

def _log_quantize_compress(tensor: np.ndarray, bits: int = 8) -> Tuple[bytes, dict]:
    """Logarithmic quantization — good for heavy-tailed data."""
    flat = tensor.ravel().astype(np.float32)
    n = len(flat)
    signs = np.sign(flat)
    abs_vals = np.abs(flat) + 1e-10
    log_vals = np.log(abs_vals)
    lo, hi = float(np.min(log_vals)), float(np.max(log_vals))
    if hi - lo < 1e-10:
        return _block_int8_fallback(tensor)
    levels = (1 << bits) - 1
    scale = levels / (hi - lo)
    quantized = np.clip(np.round((log_vals - lo) * scale), 0, levels).astype(np.uint16)
    header = struct.pack("<IIff", n, bits, lo, hi)
    return header + quantized.tobytes() + signs.astype(np.int8).tobytes(), {
        "n": n,
        "bits": bits,
        "lo": lo,
        "hi": hi,
        "log_quant": True,
    }

def _log_quantize_decompress(data: bytes, metadata: dict) -> np.ndarray:
    n, bits = struct.unpack_from("<II", data, 0)
    lo, hi = struct.unpack_from("<ff", data, 8)
    levels = (1 << bits) - 1
    quantized = np.frombuffer(data[16 : 16 + n * 2], dtype=np.uint16).astype(np.float32)
    signs = np.frombuffer(data[16 + n * 2 : 16 + n * 3], dtype=np.int8).astype(
        np.float32
    )
    log_vals = quantized / levels * (hi - lo) + lo
    return np.exp(log_vals) * np.where(signs >= 0, 1.0, -1.0)

def _dpcm_encode(tensor: np.ndarray, bits: int = 8) -> Tuple[bytes, dict]:
    """Differential pulse-code modulation."""
    flat = tensor.ravel().astype(np.float32)
    n = len(flat)
    diff = np.zeros_like(flat)
    diff[0] = flat[0]
    diff[1:] = flat[1:] - flat[:-1]
    amax = float(np.max(np.abs(diff)))
    if amax < 1e-10:
        return _block_int8_fallback(tensor)
    half = 1 << (bits - 1)
    scale = half / amax
    quantized = np.clip(np.round(diff * scale), -half, half - 1).astype(np.int16)
    header = struct.pack("<IIf", n, bits, amax)
    return header + quantized.tobytes(), {
        "n": n,
        "bits": bits,
        "amax": amax,
        "dpcm": True,
    }

def _dpcm_decode(data: bytes, metadata: dict) -> np.ndarray:
    n, bits = struct.unpack_from("<II", data, 0)
    amax = struct.unpack_from("<f", data, 8)[0]
    half = 1 << (bits - 1)
    quantized = (
        np.frombuffer(data[12:], dtype=np.int16).astype(np.float32) / half * amax
    )
    out = np.cumsum(quantized)
    return out

class IterativeRD:
    name = "itertativerd"
    category = "breakthrough_info"

    def compress(self, tensor, **params):
        return _uniform_quantize_compress(tensor, 7)

    decompress = staticmethod(_uniform_quantize_decompress)
