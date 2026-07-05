"""Sparse quantization: keep top-K% weights at FP32, quantize rest to INT4."""

from __future__ import annotations

from typing import Tuple

import numpy as np


class SparseQuantize:
    """FP32 sparse + INT4 quantized hybrid compression.

    Keeps top P% of weights (by magnitude) at FP32 precision and quantizes
    the remaining (100-P)% to INT4. Preserves critical weights while
    achieving meaningful compression ratios.
    """

    name = "sparse_quantize"
    category = "sparsity"

    def compress(
        self,
        tensor: np.ndarray,
        sparsity: float = 0.75,
        n_bits: int = 4,
        block_size: int = 128,
    ) -> Tuple[bytes, dict]:
        flat = tensor.ravel().astype(np.float64)
        n = len(flat)
        magnitudes = np.abs(flat)
        threshold = np.percentile(magnitudes, (1 - sparsity) * 100)

        important_mask = magnitudes >= threshold
        quantize_mask = ~important_mask

        important_values = flat[important_mask]
        quantize_values = flat[quantize_mask]

        codebook_size = 1 << n_bits
        n_q = len(quantize_values)
        n_blocks = (n_q + block_size - 1) // block_size
        q_scales = []
        q_data_parts = []

        for b in range(n_blocks):
            start = b * block_size
            end = min(start + block_size, n_q)
            block = quantize_values[start:end]
            amax = float(np.max(np.abs(block)))
            scale = amax / (codebook_size / 2 - 1) if amax > 1e-8 else 1.0
            q_scales.append(scale)
            indices = np.clip(
                np.round(block / scale + codebook_size / 2 - 1).astype(np.int32),
                0,
                codebook_size - 1,
            )
            q_data_parts.append(_pack_nibbles(indices))

        metadata = dict(
            n_elements=n,
            sparsity=sparsity,
            n_bits=n_bits,
            block_size=block_size,
            important_mask=important_mask.astype(np.uint8).tobytes(),
            important_values=important_values.astype(np.float32).tobytes(),
            quantize_scales=np.array(q_scales, dtype=np.float32).tobytes(),
            quantize_data=b"".join(q_data_parts),
            n_important=int(important_mask.sum()),
            n_quantize=int(quantize_mask.sum()),
            shape=tensor.shape,
        )
        return metadata["important_values"], metadata

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        n = metadata["n_elements"]
        important_mask = np.frombuffer(
            metadata["important_mask"], dtype=np.uint8
        ).astype(bool)
        important_values = np.frombuffer(
            metadata["important_values"], dtype=np.float32
        ).astype(np.float64)
        q_scales = np.frombuffer(metadata["quantize_scales"], dtype=np.float32).astype(
            np.float64
        )
        n_quantize = metadata["n_quantize"]
        block_size = metadata["block_size"]
        codebook_size = 1 << metadata["n_bits"]

        q_values = np.zeros(n_quantize, dtype=np.float64)
        n_blocks = (n_quantize + block_size - 1) // block_size
        data_offset = 0

        for b in range(n_blocks):
            start = b * block_size
            end = min(start + block_size, n_quantize)
            block_len = end - start
            indices = _unpack_nibbles(
                metadata["quantize_data"][data_offset:], block_len
            )
            data_offset += (block_len + 1) // 2
            q_values[start:end] = (
                indices.astype(np.float64) - codebook_size / 2 + 1
            ) * q_scales[b]

        result = np.zeros(n, dtype=np.float64)
        result[important_mask] = important_values[: metadata["n_important"]]
        result[~important_mask] = q_values[:n_quantize]

        return result.reshape(metadata["shape"]).astype(np.float32)


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
