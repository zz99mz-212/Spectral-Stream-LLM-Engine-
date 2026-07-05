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
        del t, f, q_data, sc_data
        gc.collect()
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
        del sc_data, q_data
        gc.collect()
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
        del t, flat, inliers, bi, sc_in, q, outliers
        gc.collect()
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
        del sc_in, q, outliers, full_mask
        gc.collect()
        return d_flat.reshape(shape).astype(np.float32)


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
        for bi_val in [2, 4, 8]:
            mask = bits_arr == bi_val
            if not np.any(mask):
                continue
            idx = np.where(mask)
            blocks = b[idx]
            sc = np.maximum(np.max(np.abs(blocks), axis=1), 1e-10)
            max_q = (1 << (bi_val - 1)) - 1
            q = np.clip(np.round(blocks / sc[:, None] * max_q), -max_q, max_q).astype(
                np.int8
            )
            for bi_idx in range(len(idx[0])):
                i, j = idx[0][bi_idx], idx[1][bi_idx]
                buf += struct.pack("<f", sc[bi_idx]) + q[bi_idx].tobytes()
        meta = dict(
            shape=tensor.shape, nb=nb, nr=nr, bs=bs, p=p, n_elements=tensor.size
        )
        del t, f, b, var, vn, bits_arr
        gc.collect()
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
        del f
        gc.collect()
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
        q_padded = np.zeros((blocks.shape[0], n_pack * n_per), dtype=np.uint16)
        q_padded[:, :block_size] = q.astype(np.uint16)
        q_2d = q_padded.reshape(blocks.shape[0], -1, n_per)
        shifts = np.array(
            [(n_per - 1 - m) * bits for m in range(n_per)], dtype=np.uint16
        )
        packed = (q_2d << shifts[None, None, :]).sum(axis=-1).astype(np.uint8)
        meta = dict(shape=tensor.shape, block_size=block_size, bits=bits, n_elements=n)
        data = (
            struct.pack("<II", n, block_size)
            + scales.astype(np.float32).tobytes()
            + mins.astype(np.float32).tobytes()
            + packed.tobytes()
        )
        del t, flat, padded, blocks, mins, maxs, scales, q, q_padded
        gc.collect()
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
        del scales, mins, raw
        gc.collect()
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
        del t, flat, padded, blocks, q, q_padded, q_shifted
        gc.collect()
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
        del p_scale, n_scale, raw
        gc.collect()
        return recon.ravel()[:n].reshape(shape).astype(np.float32)
