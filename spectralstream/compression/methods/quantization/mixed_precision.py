"""Auto-generated from _class_wrappers.py."""

from __future__ import annotations

import math
import struct
from typing import Any, Dict, Tuple

from dataclasses import dataclass

import numpy as np

from spectralstream.core.math_primitives import (
    dct,
    idct,
    fwht,
    ifwht,
    next_power_of_two,
    LloydMaxQuantizer,
)


def _pack_bits(q: np.ndarray, bi: int, offset: int) -> np.ndarray:
    n_per = 8 // bi
    n_blocks = q.shape[0]
    bs = q.shape[1]
    q_shifted = (q.astype(np.int16) + offset) & ((1 << bi) - 1)
    q_2d = q_shifted.reshape(n_blocks, -1, n_per)
    shifts = np.array([(n_per - 1 - m) * bi for m in range(n_per)], dtype=np.int16)
    packed = (
        (q_2d.astype(np.int16) << shifts[None, None, :]).sum(axis=-1).astype(np.uint8)
    )
    return packed.ravel()


def _unpack_bits(
    data: bytes, pos: int, n_blocks: int, bs: int, bi: int, offset: int, max_q: int
) -> np.ndarray:
    n_per = 8 // bi
    n_bytes = (bs + n_per - 1) // n_per * n_blocks
    raw = np.frombuffer(data[pos : pos + n_bytes], dtype=np.uint8)
    shifts = np.array([(n_per - 1 - m) * bi for m in range(n_per)], dtype=np.uint8)
    raw_2d = raw.reshape(-1, n_per)
    vals = (
        (raw_2d[:, None, :].astype(np.int16) >> shifts[None, :, None]) & ((1 << bi) - 1)
    ).reshape(-1, n_per)
    q_vals = vals.astype(np.float32).ravel()[: n_blocks * bs]
    return (q_vals - offset) / max_q


def _unpack_pairs_4bit(q_pairs: np.ndarray) -> np.ndarray:
    lo = (q_pairs & 0x0F).astype(np.float32)
    hi = ((q_pairs >> 4) & 0x0F).astype(np.float32)
    n_elem = q_pairs.size * 2
    out = np.empty(n_elem, dtype=np.float32)
    out[0::2] = lo.ravel()
    out[1::2] = hi.ravel()
    return out


class MixedPrecision:
    """Per-block bit allocation based on block variance."""

    name = "mixed_precision"
    category = "quantization"

    def compress(self, tensor: np.ndarray, **params) -> Tuple[bytes, dict]:
        cfg = {
            "block_size": 64,
            "low_var_bits": 2,
            "med_var_bits": 4,
            "high_var_bits": 8,
        }
        cfg.update(params)
        t = tensor.astype(np.float32)
        s = t.shape
        f = t.reshape(-1, s[-1])
        nr, nc = f.shape
        bs = cfg["block_size"]
        p = (bs - nc % bs) % bs
        if p:
            f = np.pad(f, ((0, 0), (0, p)))
        nb = f.shape[1] // bs
        b = f.reshape(nr, nb, bs)
        var = np.var(b, axis=2)
        vl, vh = np.percentile(var, [33, 66])
        bits_map = np.full((nr, nb), cfg["med_var_bits"], dtype=np.uint8)
        bits_map[var < vl] = cfg["low_var_bits"]
        bits_map[var > vh] = cfg["high_var_bits"]
        buf = struct.pack("<IIII", s[-1] if s else 1, bs, nr, nb)
        buf += bits_map.tobytes()
        for i in range(nr):
            for j in range(nb):
                bi = int(bits_map[i, j])
                block = b[i, j]
                sc = float(np.max(np.abs(block)))
                sc = max(sc, 1e-10)
                if bi <= 4:
                    max_q = (1 << (bi - 1)) - 1
                    q = np.clip(np.round(block / sc * max_q), -max_q, max_q).astype(
                        np.int8
                    )
                    offset = 1 << (bi - 1)
                    n_per = 8 // bi
                    n_pack = (bs + n_per - 1) // n_per
                    q_padded = np.zeros(n_pack * n_per, dtype=np.int16)
                    q_padded[:bs] = q.astype(np.int16)
                    q_shifted = (q_padded + offset) & ((1 << bi) - 1)
                    q_2d = q_shifted.reshape(-1, n_per)
                    shifts = np.array(
                        [(n_per - 1 - m) * bi for m in range(n_per)], dtype=np.int16
                    )
                    block_packed = (
                        (q_2d.astype(np.int16) << shifts[None, :])
                        .sum(axis=-1)
                        .astype(np.uint8)
                    )
                    buf += struct.pack("<f", sc) + bytes(block_packed[:n_pack])
                else:
                    max_q = 127
                    q = np.clip(np.round(block / sc * max_q), -128, 127).astype(np.int8)
                    buf += struct.pack("<f", sc) + q.tobytes()
        meta = dict(
            n_elements=tensor.size,
            shape=tensor.shape,
            block_size=bs,
            bits_map=bits_map.tolist(),
        )
        return bytes(buf), meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        bs = metadata["block_size"]
        bits_map = np.array(metadata["bits_map"], dtype=np.uint8)
        nr, nb = bits_map.shape
        n = metadata["n_elements"]
        pos = struct.calcsize("<IIII") + nr * nb
        f = np.zeros((nr, nb, bs), dtype=np.float32)
        for i in range(nr):
            for j in range(nb):
                bi = int(bits_map[i, j])
                sc = struct.unpack_from("<f", data, pos)[0]
                pos += 4
                if bi <= 4:
                    max_q = (1 << (bi - 1)) - 1
                    n_per = 8 // bi
                    n_bytes = (bs + n_per - 1) // n_per
                    raw = np.frombuffer(data[pos : pos + n_bytes], dtype=np.uint8)
                    pos += n_bytes
                    shifts = np.array(
                        [(n_per - 1 - m) * bi for m in range(n_per)], dtype=np.uint8
                    )
                    vals = (
                        (raw[:, None].astype(np.int16) >> shifts[None, :])
                        & ((1 << bi) - 1)
                    ).ravel()[:bs]
                    f[i, j] = (vals.astype(np.float32) - (1 << (bi - 1))) / max_q * sc
                else:
                    q = np.frombuffer(data[pos : pos + bs], dtype=np.int8).astype(
                        np.float32
                    )
                    pos += bs
                    f[i, j] = q * sc / 127.0
        flat = f.reshape(nr, nb * bs)
        flat = flat[:, :n].ravel()
        return flat[:n].reshape(shape).astype(np.float32)


class DynamicBitwidth:
    """Dynamic per-group bitwidth selection."""

    name = "dynamic_bitwidth"
    category = "quantization"

    def compress(self, tensor: np.ndarray, block_size: int = 32) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float32)
        s = t.shape
        f = t.reshape(-1, s[-1])
        nr, nc = f.shape
        p = (block_size - nc % block_size) % block_size
        if p:
            f = np.pad(f, ((0, 0), (0, p)))
        nb = f.shape[1] // block_size
        b = f.reshape(nr, nb, block_size)
        var = np.var(b, axis=2)
        bits_arr = np.zeros((nr, nb), dtype=np.uint8)
        p66, p33 = np.percentile(var, [66, 33])
        bits_arr[var > p66] = 8
        bits_arr[(var > p33) & (var <= p66)] = 4
        bits_arr[var <= p33] = 2
        buf = struct.pack("<II", nr, nb)
        buf += bits_arr.tobytes()
        for i in range(nr):
            for j in range(nb):
                bi = int(bits_arr[i, j])
                block = b[i, j]
                sc = float(np.max(np.abs(block)))
                sc = max(sc, 1e-10)
                if bi <= 4:
                    max_q = (1 << (bi - 1)) - 1
                    q = np.clip(np.round(block / sc * max_q), -max_q, max_q).astype(
                        np.int8
                    )
                    offset = 1 << (bi - 1)
                    n_per = 8 // bi
                    n_pack = (block_size + n_per - 1) // n_per
                    q_padded = np.zeros(n_pack * n_per, dtype=np.int16)
                    q_padded[:block_size] = q.astype(np.int16)
                    q_shifted = (q_padded + offset) & ((1 << bi) - 1)
                    q_2d = q_shifted.reshape(-1, n_per)
                    shifts = np.array(
                        [(n_per - 1 - m) * bi for m in range(n_per)], dtype=np.int16
                    )
                    block_packed = (
                        (q_2d.astype(np.int16) << shifts[None, :])
                        .sum(axis=-1)
                        .astype(np.uint8)
                    )
                    buf += struct.pack("<f", sc) + bytes(block_packed[:n_pack])
                else:
                    max_q = 127
                    q = np.clip(np.round(block / sc * max_q), -128, 127).astype(np.int8)
                    buf += struct.pack("<f", sc) + q.tobytes()
        meta = dict(
            shape=tensor.shape,
            nr=nr,
            nb=nb,
            block_size=block_size,
            n_elements=tensor.size,
        )
        return bytes(buf), meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        nr = metadata["nr"]
        nb = metadata["nb"]
        block_size = metadata["block_size"]
        bits_arr = np.frombuffer(data[8 : 8 + nr * nb], dtype=np.uint8).reshape(nr, nb)
        pos = 8 + nr * nb
        f = np.zeros((nr, nb, block_size), dtype=np.float32)
        for i in range(nr):
            for j in range(nb):
                bi = int(bits_arr[i, j])
                sc = struct.unpack_from("<f", data, pos)[0]
                pos += 4
                if bi <= 4:
                    max_q = (1 << (bi - 1)) - 1
                    n_per = 8 // bi
                    n_bytes = (block_size + n_per - 1) // n_per
                    raw = np.frombuffer(data[pos : pos + n_bytes], dtype=np.uint8)
                    pos += n_bytes
                    shifts = np.array(
                        [(n_per - 1 - m) * bi for m in range(n_per)], dtype=np.uint8
                    )
                    vals = (
                        (raw[:, None].astype(np.int16) >> shifts[None, :])
                        & ((1 << bi) - 1)
                    ).ravel()[:block_size]
                    f[i, j] = (vals.astype(np.float32) - (1 << (bi - 1))) / max_q * sc
                else:
                    q = np.frombuffer(
                        data[pos : pos + block_size], dtype=np.int8
                    ).astype(np.float32)
                    pos += block_size
                    f[i, j] = q * sc / 127.0
        flat = f.reshape(nr, nb * block_size)
        n = metadata["n_elements"]
        return flat[:, :n].reshape(shape).astype(np.float32)


class MultiBitwidth:
    """Mixed block sizes with per-block bit assignment."""

    name = "multi_bitwidth"
    category = "quantization"

    def compress(self, tensor: np.ndarray, block_sizes=None) -> Tuple[bytes, dict]:
        if block_sizes is None:
            block_sizes = [16, 32, 64, 128]
        t = tensor.astype(np.float32)
        flat = t.ravel()
        n = len(flat)
        best_block_size = 64
        best_var = 0
        for bs in block_sizes:
            if bs > n:
                continue
            n_blocks = (n + bs - 1) // bs
            padded = np.zeros(n_blocks * bs, dtype=np.float32)
            padded[:n] = flat
            blocks = padded.reshape(n_blocks, bs)
            v = float(np.mean(np.var(blocks, axis=1)))
            if v > best_var:
                best_var = v
                best_block_size = bs
        bs = best_block_size
        padded_n = int(math.ceil(n / bs) * bs)
        padded = np.zeros(padded_n, dtype=np.float32)
        padded[:n] = flat
        blocks = padded.reshape(-1, bs)
        amax = np.max(np.abs(blocks), axis=1)
        scales = np.where(amax > 1e-8, amax / 7.0, 1.0)
        quantized = np.clip(np.round(blocks / scales[:, np.newaxis]), -7, 7).astype(
            np.int8
        )
        q_shifted = (quantized.astype(np.int16) + 8) & 0x0F
        n_pairs = blocks.shape[0] * ((bs + 1) // 2)
        q_pairs = q_shifted.reshape(-1, 2)
        packed = (q_pairs[:, 0] | (q_pairs[:, 1] << 4)).astype(np.uint8).tobytes()
        meta = dict(shape=tensor.shape, block_size=bs, n_elements=n)
        data = struct.pack("<II", n, bs) + scales.astype(np.float32).tobytes() + packed
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        bs = metadata["block_size"]
        n = metadata["n_elements"]
        n_blocks = (n + bs - 1) // bs
        n_padded = n_blocks * bs
        n_s, bs_s = struct.unpack_from("<II", data, 0)
        scales = np.frombuffer(data[8 : 8 + n_blocks * 4], dtype=np.float32)
        packed = np.frombuffer(data[8 + n_blocks * 4 :], dtype=np.uint8)
        lo = (packed & 0x0F).astype(np.float32)
        hi = ((packed >> 4) & 0x0F).astype(np.float32)
        n_val = min(len(packed) * 2, n_padded)
        out = np.empty(n_val, dtype=np.float32)
        out[0::2] = lo[: len(out[0::2])] - 8
        out[1::2] = hi[: len(out[1::2])] - 8
        if n_val < n_padded:
            out = np.pad(out, (0, n_padded - n_val))
        out = out * np.repeat(scales, bs)[:n_padded]
        return out[:n].reshape(shape).astype(np.float32)


class MixedBitwidthQuant:
    """Profile each group's sensitivity and assign bitwidth."""

    name = "mixed_bitwidth_quant"
    category = "quantization"

    def compress(self, tensor: np.ndarray, block_size: int = 32) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float32)
        s = t.shape
        f = t.reshape(-1, s[-1])
        nr, nc = f.shape
        p = (block_size - nc % block_size) % block_size
        if p:
            f = np.pad(f, ((0, 0), (0, p)))
        nb = f.shape[1] // block_size
        b = f.reshape(nr, nb, block_size)
        var = np.var(b, axis=2)
        bits_arr = np.zeros((nr, nb), dtype=np.uint8)
        p75, p50, p25 = np.percentile(var, [75, 50, 25])
        bits_arr[var > p75] = 8
        bits_arr[(var > p50) & (var <= p75)] = 4
        bits_arr[(var > p25) & (var <= p50)] = 2
        bits_arr[var <= p25] = 1
        buf = struct.pack("<II", nr, nb)
        buf += bits_arr.tobytes()
        for i in range(nr):
            for j in range(nb):
                bi = int(bits_arr[i, j])
                block = b[i, j]
                sc = float(np.max(np.abs(block)))
                sc = max(sc, 1e-10)
                if bi == 1:
                    q = (block >= 0).astype(np.uint8)
                    buf += struct.pack("<f", sc) + np.packbits(q).tobytes()
                elif bi <= 4:
                    max_q = (1 << (bi - 1)) - 1
                    q = np.clip(np.round(block / sc * max_q), -max_q, max_q).astype(
                        np.int8
                    )
                    n_per = 8 // bi
                    n_pack = (block_size + n_per - 1) // n_per
                    q_padded = np.zeros(n_pack * n_per, dtype=np.int16)
                    q_padded[:block_size] = q.astype(np.int16)
                    q_shifted = (q_padded + max_q) & ((1 << bi) - 1)
                    q_2d = q_shifted.reshape(-1, n_per)
                    shifts = np.array(
                        [(n_per - 1 - m) * bi for m in range(n_per)], dtype=np.int16
                    )
                    block_packed = (
                        (q_2d.astype(np.int16) << shifts[None, :])
                        .sum(axis=-1)
                        .astype(np.uint8)
                    )
                    buf += struct.pack("<f", sc) + bytes(block_packed[:n_pack])
                else:
                    max_q = 127
                    q = np.clip(np.round(block / sc * max_q), -128, 127).astype(np.int8)
                    buf += struct.pack("<f", sc) + q.tobytes()
        meta = dict(
            shape=tensor.shape,
            nr=nr,
            nb=nb,
            block_size=block_size,
            n_elements=tensor.size,
        )
        return bytes(buf), meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        nr = metadata["nr"]
        nb = metadata["nb"]
        block_size = metadata["block_size"]
        bits_arr = np.frombuffer(data[8 : 8 + nr * nb], dtype=np.uint8).reshape(nr, nb)
        pos = 8 + nr * nb
        f = np.zeros((nr, nb, block_size), dtype=np.float32)
        for i in range(nr):
            for j in range(nb):
                bi = int(bits_arr[i, j])
                sc = struct.unpack_from("<f", data, pos)[0]
                pos += 4
                if bi == 1:
                    n_bytes = (block_size + 7) // 8
                    bits = (
                        np.unpackbits(
                            np.frombuffer(data[pos : pos + n_bytes], dtype=np.uint8)
                        )[:block_size].astype(np.float32)
                        * 2.0
                        - 1.0
                    )
                    pos += n_bytes
                    f[i, j] = bits * sc
                elif bi <= 4:
                    max_q = (1 << (bi - 1)) - 1
                    n_per = 8 // bi
                    n_bytes = (block_size + n_per - 1) // n_per
                    raw = np.frombuffer(data[pos : pos + n_bytes], dtype=np.uint8)
                    pos += n_bytes
                    shifts = np.array(
                        [(n_per - 1 - m) * bi for m in range(n_per)], dtype=np.uint8
                    )
                    vals = (
                        (raw[:, None].astype(np.int16) >> shifts[None, :])
                        & ((1 << bi) - 1)
                    ).ravel()[:block_size]
                    f[i, j] = (vals.astype(np.float32) - max_q) / max_q * sc
                else:
                    q = np.frombuffer(
                        data[pos : pos + block_size], dtype=np.int8
                    ).astype(np.float32)
                    pos += block_size
                    f[i, j] = q * sc / 127.0
        flat = f.reshape(nr, nb * block_size)
        n = metadata["n_elements"]
        return flat[:, :n].reshape(shape).astype(np.float32)


class SensitivityAwareQuant:
    """Hessian/Fisher-aware quantization."""

    name = "sensitivity_aware_quant"
    category = "quantization"

    def compress(
        self, tensor: np.ndarray, sensitivity: float = 0.5
    ) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float32)
        s = t.shape
        f = t.reshape(-1, s[-1])
        nr, nc = f.shape
        bs = 64
        p = (bs - nc % bs) % bs
        if p:
            f = np.pad(f, ((0, 0), (0, p)))
        nb = f.shape[1] // bs
        b = f.reshape(nr, nb, bs)
        var = np.var(b, axis=2)
        vn = (var - var.min()) / (var.max() - var.min() + 1e-30)
        sens = np.clip(sensitivity, 0.0, 1.0)
        bits_arr = (
            np.round(vn * (sens * 6 + 2) + (1 - sens) * 2).clip(2, 8).astype(np.uint8)
        )
        buf = struct.pack("<II", nr, nb)
        buf += bits_arr.tobytes()
        for i in range(nr):
            for j in range(nb):
                bi = int(bits_arr[i, j])
                block = b[i, j]
                sc = float(np.max(np.abs(block)))
                sc = max(sc, 1e-10)
                max_q_val = (1 << (bi - 1)) - 1
                q = np.clip(
                    np.round(block / sc * max_q_val), -max_q_val, max_q_val
                ).astype(np.int8)
                buf += struct.pack("<f", sc) + q.tobytes()
        meta = dict(
            shape=tensor.shape, nb=nb, nr=nr, bs=bs, p=p, n_elements=tensor.size
        )
        return bytes(buf), meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        nr = metadata["nr"]
        nb = metadata["nb"]
        bs = metadata["bs"]
        bits_arr = np.frombuffer(data[8 : 8 + nr * nb], dtype=np.uint8).reshape(nr, nb)
        pos = 8 + nr * nb
        f = np.zeros((nr, nb, bs), dtype=np.float32)
        for i in range(nr):
            for j in range(nb):
                bi = int(bits_arr[i, j])
                sc = struct.unpack_from("<f", data, pos)[0]
                pos += 4
                max_q = (1 << (bi - 1)) - 1
                q = np.frombuffer(data[pos : pos + bs], dtype=np.int8).astype(
                    np.float32
                )
                pos += bs
                f[i, j] = q * sc / max_q
        flat = f.reshape(nr, nb * bs)
        n = metadata["n_elements"]
        return flat[:, :n].reshape(shape).astype(np.float32)


class GroupWiseQuant:
    """Standard group-wise affine quantization."""

    name = "group_wise_quant"
    category = "quantization"

    def compress(
        self, tensor: np.ndarray, block_size: int = 64, bits: int = 4
    ) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float32)
        flat = t.ravel()
        n = len(flat)
        padded_n = int(math.ceil(n / block_size) * block_size)
        padded = np.zeros(padded_n, dtype=np.float32)
        padded[:n] = flat
        blocks = padded.reshape(-1, block_size)
        mins = np.min(blocks, axis=1)
        maxs = np.max(blocks, axis=1)
        scales = np.where(maxs - mins > 1e-8, (maxs - mins) / ((1 << bits) - 1), 1.0)
        q = np.clip(
            np.round((blocks - mins[:, np.newaxis]) / scales[:, np.newaxis]),
            0,
            (1 << bits) - 1,
        ).astype(np.uint8)
        n_per = 8 // bits
        n_pack = (block_size + n_per - 1) // n_per
        q_padded = np.zeros((blocks.shape[0], n_pack * n_per), dtype=np.uint8)
        q_padded[:, :block_size] = q
        q_2d = q_padded.reshape(blocks.shape[0], -1, n_per)
        shifts = np.array(
            [(n_per - 1 - m) * bits for m in range(n_per)], dtype=np.uint16
        )
        packed = (
            (q_2d.astype(np.uint16) << shifts[None, None, :])
            .sum(axis=-1)
            .astype(np.uint8)
        )
        meta = dict(shape=tensor.shape, block_size=block_size, bits=bits, n_elements=n)
        data = (
            struct.pack("<II", n, block_size)
            + scales.astype(np.float32).tobytes()
            + mins.astype(np.float32).tobytes()
            + packed.tobytes()
        )
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        block_size = metadata["block_size"]
        bits = metadata["bits"]
        n = metadata["n_elements"]
        n_blocks = (n + block_size - 1) // block_size
        n_bs = struct.unpack_from("<II", data, 0)
        pos = 8
        scales = np.frombuffer(data[pos : pos + n_blocks * 4], dtype=np.float32)
        pos += n_blocks * 4
        mins = np.frombuffer(data[pos : pos + n_blocks * 4], dtype=np.float32)
        pos += n_blocks * 4
        n_per = 8 // bits
        n_pack = (block_size + n_per - 1) // n_per
        raw = np.frombuffer(data[pos : pos + n_blocks * n_pack], dtype=np.uint8)
        shifts = np.array(
            [(n_per - 1 - m) * bits for m in range(n_per)], dtype=np.uint8
        )
        vals = (
            (raw[:, None].astype(np.uint16) >> shifts[None, :]) & ((1 << bits) - 1)
        ).reshape(n_blocks, -1, n_per)
        q_vals = vals.reshape(n_blocks, -1)[:, :block_size]
        recon = q_vals.astype(np.float32) * scales[:, np.newaxis] + mins[:, np.newaxis]
        return recon.ravel()[:n].reshape(shape).astype(np.float32)


class AsymmetricQuant:
    """Asymmetric quantization with separate positive/negative handling."""

    name = "asymmetric_quant"
    category = "quantization"

    def compress(
        self, tensor: np.ndarray, block_size: int = 64, bits: int = 4
    ) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float32)
        flat = t.ravel()
        n = len(flat)
        padded_n = int(math.ceil(n / block_size) * block_size)
        padded = np.zeros(padded_n, dtype=np.float32)
        padded[:n] = flat
        blocks = padded.reshape(-1, block_size)
        pos_max = np.maximum(np.max(blocks, axis=1), 0)
        neg_min = np.minimum(np.min(blocks, axis=1), 0)
        half = 1 << (bits - 1)
        p_scale = np.where(pos_max > 1e-8, pos_max / half, 1.0)
        n_scale = np.where(neg_min < -1e-8, neg_min / -half, 1.0)
        means = np.mean(blocks, axis=1)
        use_pos = means >= 0
        n_blocks = blocks.shape[0]
        q = np.zeros((n_blocks, block_size), dtype=np.int8)
        if np.any(use_pos):
            q[use_pos] = np.clip(
                np.round(blocks[use_pos] / p_scale[use_pos, np.newaxis]),
                -half,
                half - 1,
            ).astype(np.int8)
        if np.any(~use_pos):
            q[~use_pos] = np.clip(
                np.round(blocks[~use_pos] / n_scale[~use_pos, np.newaxis]),
                -half,
                half - 1,
            ).astype(np.int8)
        n_per = 8 // bits
        n_pack = (block_size + n_per - 1) // n_per
        q_padded = np.zeros((n_blocks, n_pack * n_per), dtype=np.int16)
        q_padded[:, :block_size] = q.astype(np.int16)
        q_shifted = (q_padded + half) & ((1 << bits) - 1)
        q_2d = q_shifted.reshape(n_blocks, -1, n_per)
        shifts_arr = np.array(
            [(n_per - 1 - m) * bits for m in range(n_per)], dtype=np.int16
        )
        packed = (q_2d << shifts_arr[None, None, :]).sum(axis=-1).astype(np.uint8)
        meta = dict(shape=tensor.shape, block_size=block_size, bits=bits, n_elements=n)
        data = (
            struct.pack("<II", n, block_size)
            + p_scale.astype(np.float32).tobytes()
            + n_scale.astype(np.float32).tobytes()
            + packed.tobytes()
        )
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        block_size = metadata["block_size"]
        bits = metadata["bits"]
        n = metadata["n_elements"]
        n_blocks = (n + block_size - 1) // block_size
        pos = 8
        p_scale = np.frombuffer(data[pos : pos + n_blocks * 4], dtype=np.float32)
        pos += n_blocks * 4
        n_scale = np.frombuffer(data[pos : pos + n_blocks * 4], dtype=np.float32)
        pos += n_blocks * 4
        half = 1 << (bits - 1)
        n_per = 8 // bits
        n_pack = (block_size + n_per - 1) // n_per
        raw = np.frombuffer(data[pos : pos + n_blocks * n_pack], dtype=np.uint8)
        shifts_arr = np.array(
            [(n_per - 1 - m) * bits for m in range(n_per)], dtype=np.uint8
        )
        vals = (
            (raw[:, None].astype(np.uint16) >> shifts_arr[None, :]) & ((1 << bits) - 1)
        ).reshape(n_blocks, -1, n_per)
        q_vals = vals.reshape(n_blocks, -1)[:, :block_size].astype(np.float32) - half
        recon = np.where(
            q_vals >= 0,
            q_vals * p_scale[:, np.newaxis],
            q_vals * n_scale[:, np.newaxis],
        )
        return recon.ravel()[:n].reshape(shape).astype(np.float32)


class AdaptiveGroupQuant:
    """Adaptive group-wise quantization with per-block scaling."""

    name = "adaptive_group_quant"
    category = "quantization"

    def compress(
        self, tensor: np.ndarray, bits: int = 4, n_groups: int = 8
    ) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float32)
        s = t.shape
        f = t.reshape(-1, s[-1])
        nr, nc = f.shape
        norms = np.linalg.norm(f, axis=0)
        order = np.argsort(norms)
        group_size = nc // n_groups
        groups = []
        for g in range(n_groups):
            start = g * group_size
            end = start + group_size if g < n_groups - 1 else nc
            groups.append(order[start:end])
        max_q = (1 << (bits - 1)) - 1
        q_data = np.empty_like(f, dtype=np.int8)
        sc_data = np.zeros((nr, n_groups), dtype=np.float32)
        for g, cols in enumerate(groups):
            block = f[:, cols]
            sc = np.maximum(np.max(np.abs(block), axis=1, keepdims=True), 1e-10)
            q_data[:, cols] = np.clip(
                np.round(block / sc * max_q), -max_q, max_q
            ).astype(np.int8)
            sc_data[:, g] = sc.ravel()
        meta = dict(shape=tensor.shape, n_groups=n_groups, bits=bits)
        data = struct.pack("<II", nr, nc) + sc_data.tobytes() + q_data.tobytes()
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        n_groups = metadata["n_groups"]
        bits = metadata["bits"]
        max_q = (1 << (bits - 1)) - 1
        nr, nc = struct.unpack_from("<II", data, 0)
        f = np.zeros((nr, nc), dtype=np.float32)
        sc_data = np.frombuffer(
            data[8 : 8 + nr * n_groups * 4], dtype=np.float32
        ).reshape(nr, n_groups)
        q_data = np.frombuffer(data[8 + nr * n_groups * 4 :], dtype=np.int8).reshape(
            nr, nc
        )
        group_size = nc // n_groups
        for g in range(n_groups):
            start = g * group_size
            end = start + group_size if g < n_groups - 1 else nc
            f[:, start:end] = sc_data[:, g : g + 1] * (
                q_data[:, start:end].astype(np.float32) / max_q
            )
        return f.reshape(shape).astype(np.float32)


class OutlierAwareQuant:
    """Mixed INT4/INT8 with outlier channel detection."""

    name = "outlier_aware_quant"
    category = "quantization"

    def compress(
        self, tensor: np.ndarray, bits: int = 4, outlier_threshold: float = 3.0
    ) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float32)
        flat = t.ravel()
        mu, std = float(np.mean(flat)), float(np.std(flat))
        mask = np.abs(flat - mu) > outlier_threshold * std
        mask_packed = np.packbits(mask)
        outliers = flat[mask].astype(np.float16)
        inliers = flat[~mask]
        bs = 128
        n_in = len(inliers)
        p = (bs - n_in % bs) % bs
        if p:
            inliers = np.pad(inliers, (0, p))
        nb = len(inliers) // bs
        bi = inliers.reshape(-1, bs)
        sc_in = np.maximum(np.max(np.abs(bi), axis=1, keepdims=True), 1e-10)
        max_q = (1 << (bits - 1)) - 1
        q = np.clip(np.round(bi / sc_in * max_q), -max_q, max_q).astype(np.int8)
        meta = dict(
            shape=tensor.shape, bits=bits, p=p, n_in=n_in, n_elements=tensor.size
        )
        data = struct.pack("<II", len(flat), n_in) + (
            sc_in.astype(np.float32).tobytes()
            + q.tobytes()
            + mask_packed.tobytes()
            + outliers.tobytes()
        )
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        bits = metadata["bits"]
        n_in = metadata["n_in"]
        p = metadata["p"]
        n_total = metadata["n_elements"]
        _, n_in_stored = struct.unpack_from("<II", data, 0)
        nb = (n_in + 127) // 128
        pos = 8
        sc_in = np.frombuffer(data[pos : pos + nb * 4], dtype=np.float32).reshape(nb, 1)
        pos += nb * 4
        q = np.frombuffer(data[pos : pos + nb * 128], dtype=np.int8).reshape(nb, 128)
        pos += nb * 128
        mask_bytes = (n_total + 7) // 8
        mask_packed = data[pos : pos + mask_bytes]
        pos += mask_bytes
        outliers = np.frombuffer(data[pos:], dtype=np.float16).copy()
        max_q = (1 << (bits - 1)) - 1
        d_in = (sc_in * (q.astype(np.float32) / max_q)).ravel()[:n_in]
        full_mask = np.unpackbits(np.frombuffer(mask_packed, dtype=np.uint8))[
            :n_total
        ].astype(bool)
        d_flat = np.zeros(n_total, dtype=np.float32)
        d_flat[~full_mask] = d_in
        d_flat[full_mask] = outliers.astype(np.float32)
        return d_flat.reshape(shape).astype(np.float32)


@dataclass
class MixedPrecisionConfig:
    bit_options: Tuple[int, ...] = (2, 4, 8, 16)
    sensitivity_metric: str = "hessian"
    target_ratio: float = 0.25


class MixedPrecisionAllocation:
    METHOD_NAME = "mixed_precision_alloc"

    def __init__(self, config: Optional[MixedPrecisionConfig] = None):
        self.config = config or MixedPrecisionConfig()

    def _compute_sensitivity(self, tensor: np.ndarray) -> np.ndarray:
        mat = tensor.reshape(-1, tensor.shape[-1]).astype(np.float64)
        channel_sens = np.std(mat, axis=0)
        channel_sens = channel_sens / (np.max(channel_sens) + 1e-10)
        return channel_sens

    def _allocate_bits(
        self, sensitivity: np.ndarray, target_ratio: float
    ) -> np.ndarray:
        n_channels = len(sensitivity)
        bits = np.zeros(n_channels, dtype=np.int32)
        available = sorted(self.config.bit_options)

        total_budget = int(target_ratio * n_channels * max(available))
        sensitivity_order = np.argsort(sensitivity)[::-1]

        bits[:] = available[0]
        remaining = total_budget - int(np.sum(bits))

        for idx in sensitivity_order:
            if remaining <= 0:
                break
            current_bit = bits[idx]
            for b in reversed(available):
                if b > current_bit:
                    cost = b - current_bit
                    if cost <= remaining:
                        bits[idx] = b
                        remaining -= cost
                        break
        return bits

    def compress(
        self, tensor: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        target_ratio = kwargs.get("target_ratio", self.config.target_ratio)
        orig_shape = tensor.shape

        mat = tensor.reshape(-1, tensor.shape[-1]).astype(np.float64)
        sensitivity = self._compute_sensitivity(tensor)
        bit_alloc = self._allocate_bits(sensitivity, target_ratio)

        quantized_channels = []
        scale_tables = []
        for ch in range(mat.shape[1]):
            n_bits = bit_alloc[ch]
            ch_data = mat[:, ch]
            lo, hi = float(np.min(ch_data)), float(np.max(ch_data))
            scale = (hi - lo) / max((1 << n_bits) - 1, 1)
            offsets = ((ch_data - lo) / max(scale, 1e-10)).round().astype(np.int32)
            offsets = np.clip(offsets, 0, (1 << n_bits) - 1)
            quantized_channels.append(offsets.astype(np.uint8))
            scale_tables.append(np.array([lo, scale], dtype=np.float32))

        data_out = {
            "channels": quantized_channels,
            "scales": scale_tables,
            "bit_alloc": bit_alloc.astype(np.uint8),
        }
        meta = {"orig_shape": orig_shape, "method": "mixed_precision_alloc"}
        return data_out, meta

    def decompress(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> np.ndarray:
        channels = data["channels"]
        scales = data["scales"]
        m = len(channels[0])
        n = len(channels)
        result = np.zeros((m, n), dtype=np.float64)
        for ch in range(n):
            lo, scale = float(scales[ch][0]), float(scales[ch][1])
            result[:, ch] = channels[ch].astype(np.float64) * scale + lo
        return result.reshape(metadata["orig_shape"]).astype(np.float32)

    def estimate_ratio(self, tensor: np.ndarray, **kwargs) -> float:
        target_ratio = kwargs.get("target_ratio", self.config.target_ratio)
        return target_ratio
