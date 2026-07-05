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


class BinaryQuant:
    """Binary quantization — mean-center, then sign bits."""

    name = "binary_quant"
    category = "quantization"

    def compress(self, tensor: np.ndarray, block_size: int = 32) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float32)
        s = t.shape
        f = t.reshape(-1, s[-1])
        nr, nc = f.shape
        mu = np.mean(f, axis=1, keepdims=True)
        f_centered = f - mu
        p = (block_size - nc % block_size) % block_size
        if p:
            f_centered = np.pad(f_centered, ((0, 0), (0, p)))
        nb = f_centered.shape[1] // block_size
        b = f_centered.reshape(nr, nb, block_size)
        sc = np.maximum(np.max(np.abs(b), axis=2, keepdims=True), 1e-10)
        bits = (b >= 0).astype(np.uint8)
        packed = np.packbits(bits.ravel())
        meta = dict(shape=tensor.shape, block_size=block_size, n_elements=tensor.size)
        data = (
            mu.astype(np.float32).tobytes()
            + sc.astype(np.float32).tobytes()
            + packed.tobytes()
        )
        del t, f, f_centered, b, sc, bits
        gc.collect()
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        block_size = metadata["block_size"]
        n = metadata["n_elements"]
        nc = shape[-1] if len(shape) >= 2 else n
        nr = (n + nc - 1) // nc
        nc_padded = int(math.ceil(nc / block_size) * block_size)
        nb = nc_padded // block_size
        n_padded = nr * nb * block_size
        mu = np.frombuffer(data[: nr * 4], dtype=np.float32).reshape(nr, 1)
        pos = nr * 4
        sc = np.frombuffer(data[pos : pos + nr * 4], dtype=np.float32).reshape(nr, 1, 1)
        pos += nr * 4
        packed = np.frombuffer(data[pos:], dtype=np.uint8)
        n_bits = min(len(packed) * 8, n_padded)
        bits_d = np.unpackbits(packed)[:n_bits].reshape(nr, nb, block_size)
        d = sc * (bits_d.astype(np.float32) * 2.0 - 1.0)
        d = d.reshape(nr, nb * block_size)[:, :nc] + mu
        del packed, bits_d, sc
        gc.collect()
        return d.ravel()[:n].reshape(shape).astype(np.float32)


class TernaryQuant:
    """Ternary quantization: mean-center, {-1, 0, +1} with scale."""

    name = "ternary_quant"
    category = "quantization"

    def compress(self, tensor: np.ndarray, block_size: int = 32) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float32)
        s = t.shape
        f = t.reshape(-1, s[-1])
        nr, nc = f.shape
        mu = np.mean(f, axis=1, keepdims=True)
        f_centered = f - mu
        p = (block_size - nc % block_size) % block_size
        if p:
            f_centered = np.pad(f_centered, ((0, 0), (0, p)))
        nb = f_centered.shape[1] // block_size
        b = f_centered.reshape(nr, nb, block_size)
        sc = np.maximum(np.max(np.abs(b), axis=2, keepdims=True), 1e-10)
        norm = np.clip(b / sc, -1.0, 1.0)
        threshold = 0.5
        tval = np.zeros_like(norm, dtype=np.int8)
        tval[norm > threshold] = 1
        tval[norm < -threshold] = -1
        uv = (tval + 1).astype(np.uint8)
        n_pad = uv.size
        n_pack_target = (n_pad + 3) // 4
        uv_padded = np.zeros(n_pack_target * 4, dtype=np.uint8)
        uv_padded[:n_pad] = uv.ravel()
        quads = uv_padded.reshape(-1, 4)
        packed = (
            (quads[:, 0].astype(np.uint8) << 6)
            | (quads[:, 1].astype(np.uint8) << 4)
            | (quads[:, 2].astype(np.uint8) << 2)
            | quads[:, 3].astype(np.uint8)
        )
        meta = dict(shape=tensor.shape, block_size=block_size, n_elements=tensor.size)
        data = (
            mu.astype(np.float32).tobytes()
            + sc.astype(np.float32).tobytes()
            + packed.tobytes()
        )
        del t, f, f_centered, b, sc, norm, tval, uv, quads
        gc.collect()
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        block_size = metadata["block_size"]
        n = metadata["n_elements"]
        nc = shape[-1] if len(shape) >= 2 else n
        nr = (n + nc - 1) // nc
        nc_padded = int(math.ceil(nc / block_size) * block_size)
        nb = nc_padded // block_size
        n_padded = nr * nb * block_size
        mu = np.frombuffer(data[: nr * 4], dtype=np.float32).reshape(nr, 1)
        pos = nr * 4
        sc = np.frombuffer(data[pos : pos + nr * 4], dtype=np.float32).reshape(nr, 1, 1)
        pos += nr * 4
        packed = np.frombuffer(data[pos:], dtype=np.uint8)
        lo = (packed >> 6) & 3
        mi1 = (packed >> 4) & 3
        mi2 = (packed >> 2) & 3
        hi = packed & 3
        uv_d = np.empty(len(packed) * 4, dtype=np.uint8)
        uv_d[0::4] = lo
        uv_d[1::4] = mi1
        uv_d[2::4] = mi2
        uv_d[3::4] = hi
        uv_d = uv_d[:n_padded].reshape(nr, nb, block_size)
        d = sc * (uv_d.astype(np.int8) - 1).astype(np.float32)
        d = d.reshape(nr, nb * block_size)[:, :nc] + mu
        del packed, uv_d, sc
        gc.collect()
        return d.ravel()[:n].reshape(shape).astype(np.float32)


class BQQBinaryQuadratic:
    """Binary Quadratic Quantization — second-order binarization."""

    name = "bqq_binary_quadratic"
    category = "quantization"

    def compress(self, tensor: np.ndarray, block_size: int = 64) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float32)
        flat = t.ravel()
        n = len(flat)
        mu = float(np.mean(flat))
        sigma = float(np.std(flat))
        padded_n = int(math.ceil(n / block_size) * block_size)
        padded = np.zeros(padded_n, dtype=np.float32)
        padded[:n] = (flat - mu) / max(sigma, 1e-10)
        blocks = padded.reshape(-1, block_size)
        amax = np.max(np.abs(blocks), axis=1)
        scales = np.where(amax > 1e-8, amax / 1.0, 1.0)
        bits = (blocks / scales[:, np.newaxis] >= 0).astype(np.uint8)
        packed = np.packbits(bits.ravel())
        meta = dict(shape=tensor.shape, n=n, block_size=block_size, mu=mu, sigma=sigma)
        data = (
            struct.pack("<ff", mu, sigma)
            + scales.astype(np.float32).tobytes()
            + packed.tobytes()
        )
        del t, flat, padded, blocks, bits
        gc.collect()
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        n = metadata["n"]
        block_size = metadata["block_size"]
        mu = metadata["mu"]
        sigma = metadata["sigma"]
        mu_s, sigma_s = struct.unpack_from("<ff", data, 0)
        n_blocks = (n + block_size - 1) // block_size
        scales = np.frombuffer(data[8 : 8 + n_blocks * 4], dtype=np.float32)
        packed = np.frombuffer(data[8 + n_blocks * 4 :], dtype=np.uint8)
        n_bits = n_blocks * block_size
        bits = np.unpackbits(packed)[:n_bits].astype(np.float32)
        bits = bits * 2.0 - 1.0
        recon = (bits * np.repeat(scales, block_size)[:n_bits]).ravel()[:n] * sigma + mu
        del packed, bits, scales
        gc.collect()
        return recon.reshape(shape).astype(np.float32)
