# --- deser.py ---
"""Module extracted from breakthrough_signal_massive.py — deser."""

from __future__ import annotations

import struct

import numpy as np


def _deser(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()


def _block_int8_decompress(data: bytes, metadata: dict) -> np.ndarray:
    n, block_size = struct.unpack_from("<II", data, 0)
    n_blocks = (n + block_size - 1) // block_size
    scales = _deser(data[8 : 8 + n_blocks * 4], np.float32)
    quantized = (
        _deser(data[8 + n_blocks * 4 :], np.int8)
        .reshape(-1, block_size)
        .astype(np.float32)
    )
    return (quantized * scales[:, np.newaxis]).ravel()[:n]


# --- ser.py ---
"""Module extracted from breakthrough_signal_massive.py — ser."""


import math
import struct


def _ser(arr: np.ndarray) -> bytes:
    return np.ascontiguousarray(arr).tobytes()


def _block_int8_compress(flat: np.ndarray, block_size: int = 128) -> Tuple[bytes, dict]:
    n = len(flat)
    padded_n = int(math.ceil(n / block_size) * block_size)
    padded = np.zeros(padded_n, dtype=np.float32)
    padded[:n] = flat
    blocks = padded.reshape(-1, block_size)
    amax = np.max(np.abs(blocks), axis=1)
    scales = np.where(amax > 1e-8, amax / 127.0, 1.0)
    quantized = np.clip(np.round(blocks / scales[:, np.newaxis]), -128, 127).astype(
        np.int8
    )
    header = struct.pack("<II", n, block_size)
    return header + _ser(scales.astype(np.float32)) + _ser(quantized), {"n": n}
