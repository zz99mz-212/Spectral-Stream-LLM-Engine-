from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spectralstream.compression.engine._dataclasses import TensorProfile
from spectralstream.compression.engine._helpers import _compute_metrics, _compute_ratio
from spectralstream.compression.engine._methods import METHOD_REGISTRY
from spectralstream.core.math_primitives import (
    dct_2d,
    fwht,
    idct_2d,
    ifwht,
    next_power_of_two,
)


class _HadamardQuantWrapper:
    name = "hadamard_quant"
    category = "spectral"

    def compress(
        self, tensor: np.ndarray, block_size: int = 64, bits: int = 8
    ) -> Tuple[bytes, dict]:
        flat = tensor.ravel().astype(np.float32)
        n_orig = len(flat)
        padded_len = next_power_of_two(n_orig)
        padded = np.zeros(padded_len, dtype=np.float32)
        padded[:n_orig] = flat
        rng = np.random.RandomState(42)
        signs = rng.choice([-1.0, 1.0], size=padded_len).astype(np.float32)
        rotated = fwht(padded * signs, normalize=True)
        n_blocks = (padded_len + block_size - 1) // block_size
        buf = bytearray(struct.pack("<II", n_orig, padded_len))
        max_q = 127 if bits >= 8 else 7
        for b in range(n_blocks):
            start = b * block_size
            end = min(start + block_size, padded_len)
            block = rotated[start:end]
            amax = float(np.max(np.abs(block)))
            scale = amax / max_q if amax > 1e-8 else 1.0
            if bits >= 8:
                quantized = np.clip(np.round(block / scale), -128, 127).astype(np.int8)
                buf += struct.pack("<f", scale) + quantized.tobytes()
            else:
                quantized = np.clip(np.round(block / scale), -8, 7).astype(np.int8)
                packed = bytearray()
                for i in range(0, block_size, 2):
                    lo = (int(quantized[i]) + 8) & 0x0F
                    hi = (int(quantized[i + 1]) + 8) & 0x0F if i + 1 < block_size else 0
                    packed.append(lo | (hi << 4))
                buf += struct.pack("<f", scale) + bytes(packed)
        return bytes(buf), {
            "n_elements": n_orig,
            "padded_len": padded_len,
            "block_size": block_size,
            "bits": bits,
            "shape": tensor.shape,
        }

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        orig_n, padded_len = struct.unpack_from("<II", data, 0)
        block_size = metadata.get("block_size", 64)
        bits = metadata.get("bits", 8)
        pos = 8
        rotated = np.zeros(padded_len, dtype=np.float32)
        n_blocks = (padded_len + block_size - 1) // block_size
        if bits >= 8:
            for b in range(n_blocks):
                if pos + 4 > len(data):
                    break
                scale = struct.unpack_from("<f", data, pos)[0]
                pos += 4
                count = min(block_size, padded_len - b * block_size)
                raw = np.frombuffer(data[pos : pos + count], dtype=np.int8)
                pos += count
                rotated[b * block_size : b * block_size + len(raw)] = (
                    raw.astype(np.float32) * scale
                )
        else:
            elem_idx = 0
            while pos + 4 < len(data) and elem_idx < padded_len:
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
                    if elem_idx < padded_len:
                        rotated[elem_idx] = lo * scale
                        elem_idx += 1
                    if elem_idx < padded_len:
                        rotated[elem_idx] = hi * scale
                        elem_idx += 1
        rng = np.random.RandomState(42)
        signs = rng.choice([-1.0, 1.0], size=padded_len).astype(np.float32)
        result = ifwht(rotated, normalize=True) * signs
        return result[:orig_n].reshape(metadata["shape"])
