"""Delegates to engine _methods — single source of truth for block quantizers."""

from __future__ import annotations

import gc
from typing import Any, Tuple

import numpy as np

from spectralstream.compression.engine._methods import (
    _BlockINT8 as _EngineBlockINT8,
    _BlockINT4 as _EngineBlockINT4,
    _SparsityINT4 as _EngineSparsityINT4,
)

# Re-export engine classes (single source of truth)
_BlockINT8 = _EngineBlockINT8
_BlockINT4 = _EngineBlockINT4
_SparsityINT4 = _EngineSparsityINT4


class BlockFloatingPoint:
    """Shared-exponent block floating point per group."""

    name = "block_floating_point"
    category = "quantization"

    def compress(
        self, tensor: np.ndarray, block_size: int = 32, mantissa_bits: int = 8
    ) -> Tuple[bytes, dict]:
        import math
        import struct

        t = tensor.astype(np.float32)
        flat = t.ravel()
        n = len(flat)
        padded_n = int(math.ceil(n / block_size) * block_size)
        padded = np.zeros(padded_n, dtype=np.float32)
        padded[:n] = flat
        blocks = padded.reshape(-1, block_size)
        amax = np.max(np.abs(blocks), axis=1)
        exponents = np.where(amax > 1e-30, np.floor(np.log2(amax)).astype(np.int32), 0)
        scales = 2.0**exponents
        max_mantissa = (1 << (mantissa_bits - 1)) - 1
        mantissas = np.clip(
            np.round(blocks / scales[:, np.newaxis] * max_mantissa),
            -max_mantissa,
            max_mantissa,
        ).astype(np.int16)
        meta = dict(
            shape=tensor.shape,
            block_size=block_size,
            mantissa_bits=mantissa_bits,
            n_elements=n,
        )
        ex_bytes = (exponents + 128).clip(0, 255).astype(np.uint8).tobytes()
        data = struct.pack("<II", n, block_size) + ex_bytes + mantissas.tobytes()
        del t, flat, padded, blocks, amax, exponents, scales, mantissas
        gc.collect()
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        import struct

        shape = metadata["shape"]
        bs = metadata["block_size"]
        mantissa_bits = metadata["mantissa_bits"]
        n = metadata["n_elements"]
        n_blocks = (n + bs - 1) // bs
        n_s, bs_s = struct.unpack_from("<II", data, 0)
        exponents = (
            np.frombuffer(data[8 : 8 + n_blocks], dtype=np.uint8).astype(np.int32) - 128
        )
        scales = 2.0**exponents
        max_mantissa = (1 << (mantissa_bits - 1)) - 1
        mantissas = (
            np.frombuffer(data[8 + n_blocks :], dtype=np.int16)
            .reshape(n_blocks, bs)
            .astype(np.float32)
        )
        recon = (mantissas / max_mantissa) * scales[:, np.newaxis]
        del exponents, scales, mantissas
        gc.collect()
        return recon.ravel()[:n].reshape(shape).astype(np.float32)
