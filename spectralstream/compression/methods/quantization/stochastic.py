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


class StochasticRound:
    """Stochastic rounding — unbiased on average."""

    name = "stochastic_round"
    category = "quantization"

    def compress(self, tensor: np.ndarray, bits: int = 8) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float32)
        s = t.shape
        ndim = t.ndim
        f = t.reshape(-1, s[-1])
        nr, nc = f.shape
        bs = 256
        p = (bs - nc % bs) % bs
        if p:
            f = np.pad(f, ((0, 0), (0, p)))
        nb = f.shape[1] // bs
        b = f.reshape(nr, nb, bs)
        sc = np.maximum(np.max(np.abs(b), axis=2, keepdims=True), 1e-10)
        rng = np.random.RandomState(42)
        max_q = (1 << (bits - 1)) - 1
        norm = b / sc
        floor = np.floor(norm * max_q)
        frac = norm * max_q - floor
        rnd = (rng.uniform(size=frac.shape) < np.abs(frac)).astype(
            np.float32
        ) * np.sign(frac + 1e-30)
        q = (floor + rnd).clip(-max_q, max_q).astype(np.int16)
        meta = dict(
            shape=tensor.shape,
            bits=bits,
            block_size=bs,
            p=p,
            nr=nr,
            nb=nb,
            n_elements=tensor.size,
            ndim=ndim,
        )
        data = sc.astype(np.float32).tobytes() + q.tobytes()
        del t, f, b, sc, norm, floor, frac, rnd, q
        gc.collect()
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        bits = metadata["bits"]
        n = metadata["n_elements"]
        nr = metadata["nr"]
        nb = metadata["nb"]
        bs = metadata["block_size"]
        nc = shape[-1]
        max_q = (1 << (bits - 1)) - 1
        sc = np.frombuffer(data[: nr * nb * 4], dtype=np.float32).reshape(nr, nb, 1)
        q_raw = np.frombuffer(data[nr * nb * 4 :], dtype=np.int16).reshape(nr, nb, bs)
        d = sc * (q_raw.astype(np.float32) / max_q)
        flat = d.reshape(nr, nb * bs)
        recon = flat[:, :nc].reshape(shape)
        del sc, q_raw, d, flat
        gc.collect()
        return recon.astype(np.float32)


class ResidualQuant:
    """Multi-stage residual quantization (3 stages cascade)."""

    name = "residual_quant"
    category = "quantization"

    def compress(
        self, tensor: np.ndarray, n_stages: int = 3, bits: int = 8
    ) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        residual = t.copy()
        stages: list = []
        for stage in range(n_stages):
            flat = residual.ravel()
            sc = float(np.max(np.abs(flat)))
            sc = max(sc, 1e-10)
            max_q = (1 << (bits - 1)) - 1
            q = np.clip(np.round(flat / sc * max_q), -max_q, max_q).astype(np.int8)
            recon = q.astype(np.float64) * sc / max_q
            residual = residual.ravel() - recon
            stages.append((sc, q.copy()))
        data = struct.pack("<II", tensor.size, n_stages)
        for sc, q in stages:
            data += struct.pack("<f", sc) + q.tobytes()
        meta = dict(
            shape=tensor.shape, n_stages=n_stages, bits=bits, n_elements=tensor.size
        )
        del t, residual
        gc.collect()
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        n_stages = metadata["n_stages"]
        bits = metadata["bits"]
        n = metadata["n_elements"]
        max_q = (1 << (bits - 1)) - 1
        recon = np.zeros(n, dtype=np.float64)
        pos = struct.calcsize("<II")
        for _ in range(n_stages):
            sc = struct.unpack_from("<f", data, pos)[0]
            pos += 4
            q = np.frombuffer(data[pos : pos + n], dtype=np.int8).astype(np.float64)
            pos += n
            recon += q * sc / max_q
        return recon.reshape(shape).astype(np.float32)


class ErrorFeedbackQuant:
    """Block quantization with inter-block error feedback propagation."""

    name = "error_feedback_quant"
    category = "quantization"

    def compress(
        self, tensor: np.ndarray, block_size: int = 64, bits: int = 4
    ) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        flat = t.ravel()
        n = len(flat)
        padded_n = int(math.ceil(n / block_size) * block_size)
        padded = np.zeros(padded_n, dtype=np.float64)
        padded[:n] = flat
        blocks = padded.reshape(-1, block_size)
        max_q = (1 << (bits - 1)) - 1
        error = np.zeros(block_size, dtype=np.float64)
        n_blocks = blocks.shape[0]
        amax = np.max(np.abs(blocks), axis=1)
        scales = np.where(amax > 1e-8, amax / max_q, 1.0)
        q = np.clip(np.round(blocks / scales[:, np.newaxis]), -max_q, max_q).astype(
            np.int8
        )
        for b in range(1, n_blocks):
            error = blocks[b - 1] - q[b - 1].astype(np.float64) * scales[b - 1]
            blocks[b] += error
            amax_b = float(np.max(np.abs(blocks[b])))
            scales[b] = amax_b / max_q if amax_b > 1e-8 else 1.0
            q[b] = np.clip(np.round(blocks[b] / scales[b]), -max_q, max_q).astype(
                np.int8
            )
        n_per = 8 // bits
        q_padded = np.zeros(
            (n_blocks, ((block_size + n_per - 1) // n_per) * n_per), dtype=np.int16
        )
        q_padded[:, :block_size] = q.astype(np.int16)
        q_shifted = (q_padded + max_q) & ((1 << bits) - 1)
        q_2d = q_shifted.reshape(n_blocks, -1, n_per)
        shifts = np.array(
            [(n_per - 1 - m) * bits for m in range(n_per)], dtype=np.int16
        )
        packed = (
            (q_2d.astype(np.int16) << shifts[None, None, :])
            .sum(axis=-1)
            .astype(np.uint8)
        )
        buf = struct.pack("<II", n, bits)
        for b in range(n_blocks):
            n_byte = (block_size + n_per - 1) // n_per
            buf += struct.pack("<f", scales[b]) + bytes(
                packed[b * n_byte : (b + 1) * n_byte]
            )
        meta = dict(shape=tensor.shape, block_size=block_size, bits=bits, n_elements=n)
        del t, flat, padded, blocks, error, q, q_padded, q_shifted, q_2d, packed
        gc.collect()
        return bytes(buf), meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        block_size = metadata["block_size"]
        bits = metadata["bits"]
        n = metadata["n_elements"]
        max_q = (1 << (bits - 1)) - 1
        n_blocks = (n + block_size - 1) // block_size
        pos = struct.calcsize("<II")
        n_per = 8 // bits
        n_byte = (block_size + n_per - 1) // n_per
        recon = np.zeros(n_blocks * block_size, dtype=np.float64)
        scales_arr = np.zeros(n_blocks, dtype=np.float64)
        for b in range(n_blocks):
            scales_arr[b] = struct.unpack_from("<f", data, pos)[0]
            pos += 4
        raw = np.frombuffer(data[pos : pos + n_blocks * n_byte], dtype=np.uint8)
        shifts = np.array(
            [(n_per - 1 - m) * bits for m in range(n_per)], dtype=np.uint8
        )
        vals = (
            (raw[:, None].astype(np.uint16) >> shifts[None, :]) & ((1 << bits) - 1)
        ).reshape(n_blocks, -1, n_per)
        q_vals = vals.reshape(n_blocks, -1)[:, :block_size].astype(np.float64)
        for b in range(n_blocks):
            recon[b * block_size : (b + 1) * block_size] = (
                q_vals[b] - max_q
            ) * scales_arr[b]
        return recon[:n].reshape(shape).astype(np.float32)


class DCTNoiseAware:
    """DCT noise-aware — threshold coefficients below noise floor."""

    name = "dct_noise_aware"
    category = "spectral"

    def compress(
        self, tensor: np.ndarray, threshold: float = 0.01
    ) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        flat = t.ravel()
        coeffs = dct(flat)
        noise_floor = np.median(np.abs(coeffs))
        mask = np.abs(coeffs) > noise_floor * max(threshold, 0.01)
        mask_packed = np.packbits(mask)
        kept = coeffs[mask].astype(np.float16)
        meta = dict(shape=tensor.shape, n_elements=tensor.size)
        data = (
            struct.pack("<II", len(coeffs), int(np.sum(mask)))
            + mask_packed.tobytes()
            + kept.tobytes()
        )
        del t, flat, coeffs, mask, kept
        gc.collect()
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        n_total = metadata["n_elements"]
        n_coeffs, n_kept = struct.unpack_from("<II", data, 0)
        mask_bytes = (n_coeffs + 7) // 8
        mask_packed = data[8 : 8 + mask_bytes]
        mask = np.unpackbits(np.frombuffer(mask_packed, dtype=np.uint8))[
            :n_coeffs
        ].astype(bool)
        kept = np.frombuffer(data[8 + mask_bytes :], dtype=np.float16).astype(
            np.float64
        )
        coeffs = np.zeros(n_coeffs, dtype=np.float64)
        coeffs[mask] = kept[: np.sum(mask)]
        recon = idct(coeffs)
        return recon[:n_total].reshape(shape).astype(np.float32)
