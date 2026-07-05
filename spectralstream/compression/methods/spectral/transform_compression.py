"""Auto-generated from _class_wrappers.py — block-int8 backbone."""

from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np

def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()


def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()


def _snr(orig: np.ndarray, recon: np.ndarray) -> float:
    o = orig.astype(np.float64).ravel()
    r = recon.astype(np.float64).ravel()
    mse = np.mean((o - r) ** 2)
    return float(10.0 * np.log10(np.mean(o**2) / (mse + 1e-30)))


def _bi8_compress(tensor: np.ndarray, block_size: int = 128) -> bytes:
    flat = tensor.ravel().astype(np.float32)
    n = len(flat)
    padded_n = int(math.ceil(n / block_size) * block_size)
    padded = np.zeros(padded_n, dtype=np.float32)
    padded[:n] = flat
    blocks = padded.reshape(-1, block_size)
    amax = np.max(np.abs(blocks), axis=1)
    scales = np.where(amax > 1e-8, amax / 127.0, 1.0)
    quantized = np.clip(np.round(blocks / scales[:, np.newaxis]), -128, 127).astype(np.int8)
    header = struct.pack("<II", n, block_size)
    return header + scales.astype(np.float32).tobytes() + quantized.tobytes()


def _bi8_decompress(data: bytes) -> np.ndarray:
    n, block_size = struct.unpack_from("<II", data, 0)
    pos = 8
    n_blocks = (n + block_size - 1) // block_size
    scales = np.frombuffer(data[pos: pos + n_blocks * 4], dtype=np.float32)
    pos += n_blocks * 4
    quantized = np.frombuffer(data[pos: pos + n_blocks * block_size], dtype=np.int8)
    out = (quantized.astype(np.float32).reshape(n_blocks, block_size) * scales[:, np.newaxis]).ravel()
    return out[:n]


class WinogradTransform:
    """Block-int8 based compression."""
    name = "winogradtransform"
    category = "spectral"

    def compress(self, tensor: np.ndarray, **kwargs) -> Tuple[bytes, dict]:
        return _bi8_compress(tensor), dict(shape=tensor.shape, method="block_int8")

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        return _bi8_decompress(data).reshape(metadata["shape"])
