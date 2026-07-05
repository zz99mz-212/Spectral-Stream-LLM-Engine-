"""Delegates to engine _methods — single source of truth for Hadamard quantizers."""

from __future__ import annotations

import gc
from typing import Any, Tuple

import numpy as np

from spectralstream.compression.engine._methods import (
    _HadamardINT8 as _EngineHadamardINT8,
    _HadamardINT4 as _EngineHadamardINT4,
)

# Re-export engine classes (single source of truth)
_HadamardINT8 = _EngineHadamardINT8
_HadamardINT4 = _EngineHadamardINT4


class HadamardGroupWise:
    """Random Hadamard transform + group-wise affine quantization."""

    name = "hadamard_group_wise"
    category = "quantization"

    def compress(
        self, tensor: np.ndarray, block_size: int = 64, bits: int = 4
    ) -> Tuple[bytes, dict]:
        import math
        import struct

        from spectralstream.core.math_primitives import fwht, ifwht, next_power_of_two

        t = tensor.astype(np.float32)
        flat = t.ravel()
        n = len(flat)
        padded_len = next_power_of_two(n)
        padded = np.zeros(padded_len, dtype=np.float32)
        padded[:n] = flat
        rng = np.random.RandomState(42)
        signs = rng.choice([-1.0, 1.0], size=padded_len).astype(np.float32)
        rotated = fwht(padded * signs, normalize=True)
        n_blocks = (padded_len + block_size - 1) // block_size
        buf = struct.pack("<II", n, padded_len)
        n_per = 8 // bits
        for b in range(n_blocks):
            start = b * block_size
            end = min(start + block_size, padded_len)
            block = rotated[start:end]
            amax = float(np.max(np.abs(block)))
            scale = amax / max(1e-8, (1 << (bits - 1)) - 1)
            max_q = (1 << (bits - 1)) - 1
            q = np.clip(np.round(block / scale), -max_q, max_q).astype(np.int8)
            n_vals = len(q)
            n_pack = (n_vals + n_per - 1) // n_per
            q_padded = np.zeros(n_pack * n_per, dtype=np.int16)
            q_padded[:n_vals] = q.astype(np.int16)
            q_shifted = (q_padded + max_q) & ((1 << bits) - 1)
            q_2d = q_shifted.reshape(-1, n_per)
            shifts = np.array(
                [(n_per - 1 - m) * bits for m in range(n_per)], dtype=np.int16
            )
            packed = (
                (q_2d.astype(np.int16) << shifts[None, :]).sum(axis=-1).astype(np.uint8)
            )
            buf += struct.pack("<f", scale) + bytes(packed[:n_pack])
        meta = dict(shape=tensor.shape, block_size=block_size, bits=bits, n_elements=n)
        del t, flat, padded, rotated, signs
        gc.collect()
        return bytes(buf), meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        import struct

        from spectralstream.core.math_primitives import fwht, ifwht, next_power_of_two

        shape = metadata["shape"]
        block_size = metadata["block_size"]
        bits = metadata["bits"]
        n = metadata["n_elements"]
        n_orig, padded_len = struct.unpack_from("<II", data, 0)
        max_q = (1 << (bits - 1)) - 1
        n_per = 8 // bits
        pos = 8
        rotated = np.zeros(padded_len, dtype=np.float32)
        n_blocks = (padded_len + block_size - 1) // block_size
        for b in range(n_blocks):
            if pos + 4 > len(data):
                break
            scale = struct.unpack_from("<f", data, pos)[0]
            pos += 4
            count = min(block_size, padded_len - b * block_size)
            n_pack = (count + n_per - 1) // n_per
            raw = np.frombuffer(data[pos : pos + n_pack], dtype=np.uint8)
            pos += n_pack
            shifts = np.array(
                [(n_per - 1 - m) * bits for m in range(n_per)], dtype=np.uint8
            )
            vals = (
                (raw[:, None].astype(np.uint16) >> shifts[None, :]) & ((1 << bits) - 1)
            ).ravel()[:count]
            rotated[b * block_size : b * block_size + count] = (
                vals.astype(np.float32) - max_q
            ) * scale
        rng = np.random.RandomState(42)
        signs = rng.choice([-1.0, 1.0], size=padded_len).astype(np.float32)
        result = ifwht(rotated, normalize=True) * signs
        del rotated, signs
        gc.collect()
        return result[:n].reshape(shape).astype(np.float32)
