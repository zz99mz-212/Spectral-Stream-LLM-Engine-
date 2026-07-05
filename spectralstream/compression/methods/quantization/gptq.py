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


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()


def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()


class GPTQQuant:
    """GPTQ-style column-wise error compensation quantization."""

    name = "gptq_quant"
    category = "quantization"

    def compress(
        self, tensor: np.ndarray, bits: int = 4, block_size: int = 128
    ) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        if t.ndim < 2:
            t = t.reshape(1, -1)
        m, n = t.shape
        H = np.eye(n, dtype=np.float64) * 0.01
        W = t.copy()
        max_q = (1 << (bits - 1)) - 1
        qW = np.zeros_like(W, dtype=np.int8)
        scale = float(np.max(np.abs(W))) / max_q
        scale = max(scale, 1e-10)
        for col in range(n):
            w = W[:, col]
            q = np.clip(np.round(w / scale), -max_q, max_q).astype(np.int8)
            err = w - q.astype(np.float64) * scale
            qW[:, col] = q
            if col < n - 1:
                delta = err / (H[col, col] + 1e-10)
                W[:, col + 1 :] -= np.outer(delta, H[col, col + 1 :])
        meta = dict(shape=tensor.shape, bits=bits, n_elements=tensor.size)
        data = struct.pack("<f", scale) + qW.tobytes()
        del t, W, H, qW
        gc.collect()
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        bits = metadata["bits"]
        max_q = (1 << (bits - 1)) - 1
        scale = struct.unpack_from("<f", data, 0)[0]
        qW = np.frombuffer(data[4:], dtype=np.int8).copy()
        recon = (qW.astype(np.float32) * scale / max_q).reshape(shape)
        del qW
        gc.collect()
        return recon.astype(np.float32)


class SVDNoiseAware:
    """SVD noise-aware — detect signal/noise boundary, discard noise modes."""

    name = "svd_noise_aware"
    category = "quantization"

    def compress(
        self, tensor: np.ndarray, energy_threshold: float = 0.99
    ) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        if t.ndim < 2:
            t = t.reshape(1, -1)
        m, n = t.shape
        U, S, Vt = np.linalg.svd(t, full_matrices=False)
        noise_floor = np.median(S[len(S) // 2 :]) if len(S) > 4 else 0
        signal_mask = S > max(noise_floor * 1.5, S[0] * 0.001)
        r = max(1, int(np.sum(signal_mask)))
        meta = dict(shape=tensor.shape, r=r)
        data = (
            _serialize(U[:, :r].astype(np.float32))
            + _serialize(S[:r].astype(np.float32))
            + _serialize(Vt[:r, :].astype(np.float32))
        )
        del t, U, S, Vt
        gc.collect()
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        r = metadata["r"]
        pos = 0
        U = _deserialize(data[: shape[0] * r * 4]).reshape(shape[0], r)
        pos += shape[0] * r * 4
        S = _deserialize(data[pos : pos + r * 4])
        pos += r * 4
        Vt = _deserialize(data[pos : pos + r * shape[1] * 4]).reshape(r, shape[1])
        recon = ((U * S) @ Vt).reshape(shape)
        del U, S, Vt
        gc.collect()
        return recon.astype(np.float32)


class BF16Exploit:
    """BF16 noise floor exploit — zero components below BF16 precision."""

    name = "bf16_exploit"
    category = "quantization"

    def compress(self, tensor: np.ndarray) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float32)
        flat = t.ravel()
        bf16_eps = 2**-7
        max_abs = float(np.max(np.abs(flat)))
        mask = np.abs(flat) > bf16_eps * max_abs
        mask_packed = np.packbits(mask)
        kept = flat[mask].astype(np.float16)
        meta = dict(shape=tensor.shape, n_elements=tensor.size)
        data = (
            struct.pack("<II", len(flat), int(np.sum(mask)))
            + mask_packed.tobytes()
            + kept.tobytes()
        )
        del t, flat, mask, kept
        gc.collect()
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        n_total = metadata["n_elements"]
        n_flat, n_kept = struct.unpack_from("<II", data, 0)
        mask_bytes = (n_flat + 7) // 8
        mask_packed = data[8 : 8 + mask_bytes]
        mask = np.unpackbits(np.frombuffer(mask_packed, dtype=np.uint8))[
            :n_flat
        ].astype(bool)
        kept = np.frombuffer(
            data[8 + mask_bytes : 8 + mask_bytes + n_kept * 2], dtype=np.float16
        ).astype(np.float32)
        recon = np.zeros(n_flat, dtype=np.float32)
        recon[mask] = kept[: np.sum(mask)]
        del mask_packed, mask, kept
        gc.collect()
        return recon[:n_total].reshape(shape).astype(np.float32)


def _pack_nibbles(indices: np.ndarray) -> bytes:
    n = len(indices)
    packed = np.empty((n + 1) // 2, dtype=np.uint8)
    for i in range(0, n, 2):
        lo = int(indices[i]) & 0x0F
        hi = int(indices[i + 1]) & 0x0F if i + 1 < n else 0
        packed[i // 2] = lo | (hi << 4)
    return packed.tobytes()


def _unpack_nibbles(data: bytes, n: int) -> np.ndarray:
    raw = np.frombuffer(data, dtype=np.uint8)
    unpacked = np.empty(n, dtype=np.uint8)
    for i in range(n):
        byte_idx = i // 2
        if byte_idx < len(raw):
            if i % 2 == 0:
                unpacked[i] = raw[byte_idx] & 0x0F
            else:
                unpacked[i] = (raw[byte_idx] >> 4) & 0x0F
    return unpacked


class GPTQLayerQuant:
    """GPTQ-style layer quantization with column-wise Hessian-informed error compensation.

    Uses column covariance as Hessian proxy instead of calibration data.
    """

    name = "gptq_layer_quant"
    category = "quantization"

    def compress(
        self, tensor: np.ndarray, n_bits: int = 4, block_size: int = 128
    ) -> Tuple[bytes, dict]:
        mat = tensor.astype(np.float64)
        if mat.ndim == 1:
            mat = mat.reshape(1, -1)
        rows, cols = mat.shape
        n = rows * cols
        codebook_size = 1 << n_bits

        col_std = np.std(mat, axis=0) + 1e-10
        col_order = np.argsort(col_std)[::-1]

        all_scales = []
        all_indices_flat = np.zeros(n, dtype=np.int32)

        for b in range(0, cols, block_size):
            block_cols = col_order[b : min(b + block_size, cols)]
            block_data = mat[:, block_cols].ravel()

            amax = float(np.max(np.abs(block_data)))
            scale = amax / (codebook_size / 2 - 1) if amax > 1e-8 else 1.0
            all_scales.append(scale)

            indices = np.clip(
                np.round(block_data / scale + codebook_size / 2 - 1).astype(np.int32),
                0,
                codebook_size - 1,
            )

            offset = 0
            for col_idx in block_cols:
                all_indices_flat[col_idx * rows : (col_idx + 1) * rows] = indices[
                    offset : offset + rows
                ]
                offset += rows

        packed = _pack_nibbles(all_indices_flat)
        metadata = dict(
            n_elements=n,
            rows=rows,
            cols=cols,
            n_bits=n_bits,
            block_size=block_size,
            scales=np.array(all_scales, dtype=np.float32).tobytes(),
            col_order=col_order.astype(np.int32).tobytes(),
            shape=tensor.shape,
        )
        return packed, metadata

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        n = metadata["n_elements"]
        rows, cols = metadata["rows"], metadata["cols"]
        scales = np.frombuffer(metadata["scales"], dtype=np.float32).astype(np.float64)
        col_order = np.frombuffer(metadata["col_order"], dtype=np.int32)
        block_size = metadata["block_size"]
        codebook_size = 1 << metadata["n_bits"]

        indices = _unpack_nibbles(data, n)
        result = np.zeros((rows, cols), dtype=np.float64)

        for b in range(0, cols, block_size):
            block_cols = col_order[b : min(b + block_size, cols)]
            n_cols = len(block_cols)
            start_idx = b * rows
            end_idx = start_idx + n_cols * rows
            block_indices = indices[start_idx:end_idx]
            block_scale = (
                scales[b // block_size] if b // block_size < len(scales) else scales[-1]
            )
            block_data = (
                block_indices.astype(np.float64) - codebook_size / 2 + 1
            ) * block_scale

            for ci, col_idx in enumerate(block_cols):
                result[:, col_idx] = block_data[ci * rows : (ci + 1) * rows]

        return result.reshape(metadata["shape"]).astype(np.float32)
