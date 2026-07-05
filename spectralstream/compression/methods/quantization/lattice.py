"""Auto-generated from _class_wrappers.py."""

from __future__ import annotations

import gc
import math
import struct
from typing import Any, Dict, Tuple

import numpy as np

from spectralstream.core.math_primitives import (
    dct,
    idct,
    fwht,
    ifwht,
    next_power_of_two,
    LloydMaxQuantizer,
)


class E8Lattice:
    """E8 lattice vector quantization."""

    name = "e8_lattice"
    category = "quantization"

    def compress(self, tensor: np.ndarray, block_size: int = 8) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float32).ravel()
        n_orig = len(t)
        p = (8 - n_orig % 8) % 8
        if p:
            t = np.pad(t, (0, p))
        n_blocks_8d = len(t) // 8
        blocks = t.reshape(-1, 8)
        sc = np.maximum(np.max(np.abs(blocks), axis=1, keepdims=True), 1e-10)
        norm = blocks / sc

        def _e8_nearest(x):
            v1 = np.floor(x + 0.5)
            parity = np.sum(v1, axis=1, keepdims=True) % 2
            i = np.argmax(np.abs(v1 - x), axis=1)
            row_idx = np.arange(v1.shape[0])
            adjust = np.sign(x[row_idx, i] - v1[row_idx, i])
            adjust = np.where(adjust == 0, 1.0, adjust)
            v1[row_idx, i] -= parity.ravel() * adjust
            v2 = np.floor(2.0 * x + 0.5) / 2.0
            v2_parity = np.sum(2.0 * v2, axis=1, keepdims=True) % 2
            v2[row_idx, i] -= v2_parity.ravel() * adjust * 0.5
            d1 = np.sum((v1 - x) ** 2, axis=1, keepdims=True)
            d2 = np.sum((v2 - x) ** 2, axis=1, keepdims=True)
            closer = d1 < d2
            result = np.where(closer, v1, v2)
            return result

        lp = _e8_nearest(norm)
        stored = np.round(lp * 2.0).astype(np.int8)
        meta = dict(shape=tensor.shape, p=p, n_elements=n_orig)
        data = sc.astype(np.float32).ravel().tobytes() + stored.tobytes()
        del t, blocks, sc, norm, lp, stored
        gc.collect()
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        p = metadata["p"]
        n_orig = metadata["n_elements"]
        n_blocks = (n_orig + 7) // 8
        sc = np.frombuffer(data[: n_blocks * 4], dtype=np.float32)
        stored = (
            np.frombuffer(data[n_blocks * 4 :], dtype=np.int8)
            .reshape(n_blocks, 8)
            .astype(np.float32)
            / 2.0
        )
        d = (stored * sc[:, None]).ravel()
        if p:
            d = d[:-p]
        del sc, stored
        gc.collect()
        return d.reshape(shape).astype(np.float32)


class LatticeAnchored:
    """Anchored lattice quantization with median anchor."""

    name = "lattice_anchored"
    category = "quantization"

    def compress(self, tensor: np.ndarray, block_size: int = 32) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float32)
        flat = t.ravel()
        n = len(flat)
        anchor = float(np.median(flat))
        residual = flat - anchor
        padded_n = int(math.ceil(n / block_size) * block_size)
        padded = np.zeros(padded_n, dtype=np.float32)
        padded[:n] = residual
        blocks = padded.reshape(-1, block_size)
        amax = np.max(np.abs(blocks), axis=1)
        scales = np.where(amax > 1e-8, amax / 7.0, 1.0)
        quantized = np.clip(np.round(blocks / scales[:, np.newaxis]), -7, 7).astype(
            np.int8
        )
        q_shifted = (quantized.astype(np.int16) + 8) & 0x0F
        n_pairs = blocks.shape[0] * ((block_size + 1) // 2)
        q_pairs = q_shifted.reshape(-1, 2)
        packed = (q_pairs[:, 0] | (q_pairs[:, 1] << 4)).astype(np.uint8).tobytes()
        meta = dict(
            shape=tensor.shape, block_size=block_size, anchor=anchor, n_elements=n
        )
        data = struct.pack("<f", anchor) + scales.astype(np.float32).tobytes() + packed
        del t, flat, residual, padded, blocks, quantized, q_shifted, q_pairs
        gc.collect()
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        block_size = metadata["block_size"]
        anchor = metadata["anchor"]
        n = metadata["n_elements"]
        n_blocks = (n + block_size - 1) // block_size
        anchor_v = struct.unpack_from("<f", data, 0)[0]
        scales = np.frombuffer(data[4 : 4 + n_blocks * 4], dtype=np.float32)
        packed = np.frombuffer(data[4 + n_blocks * 4 :], dtype=np.uint8)
        lo = (packed & 0x0F).astype(np.float32) - 8
        hi = ((packed >> 4) & 0x0F).astype(np.float32) - 8
        n_val = min(len(packed) * 2, n_blocks * block_size)
        out = np.empty(n_val, dtype=np.float32)
        out[0::2] = lo[: len(out[0::2])]
        out[1::2] = hi[: len(out[1::2])]
        if n_val < n_blocks * block_size:
            out = np.pad(out, (0, n_blocks * block_size - n_val))
        out = out * np.repeat(scales, block_size)[: n_blocks * block_size]
        del packed, scales, lo, hi
        gc.collect()
        return (out[:n] + anchor).reshape(shape).astype(np.float32)
