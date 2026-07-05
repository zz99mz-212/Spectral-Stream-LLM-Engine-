"""Adaptive bit-width per block: 2-bit for smooth, 4-bit for medium, 8-bit for high-variance."""

from __future__ import annotations

from typing import Tuple

import numpy as np


class BlockAdaptiveQuant:
    """Adaptive 2/4-bit per-block quantization based on variance.

    Uses per-block variance to decide bit allocation: low-variance blocks
    get 2 bits, higher variance blocks get 4 bits.
    """

    name = "block_adaptive_quant"
    category = "quantization"

    def compress(self, tensor: np.ndarray, block_size: int = 128) -> Tuple[bytes, dict]:
        flat = tensor.ravel().astype(np.float64)
        n = len(flat)
        n_blocks = (n + block_size - 1) // block_size

        block_bits = []
        block_scales = []
        block_data_list = []

        for b in range(n_blocks):
            start = b * block_size
            end = min(start + block_size, n)
            block = flat[start:end]
            block_var = float(np.var(block))
            block_max = float(np.max(np.abs(block)))

            if block_var < 1e-8:
                bits = 2
            elif block_var < 0.001:
                bits = 2
            elif block_var < 0.01:
                bits = 4
            else:
                bits = 4

            codebook_size = 1 << bits
            scale = block_max / (codebook_size / 2 - 1) if block_max > 1e-8 else 1.0
            indices = np.clip(
                np.round(block / scale + codebook_size / 2 - 1).astype(np.int32),
                0,
                codebook_size - 1,
            )

            block_bits.append(bits)
            block_scales.append(scale)
            block_data_list.append(indices.astype(np.uint8).tobytes())

        bits_map = np.array(block_bits, dtype=np.uint8)
        scales_arr = np.array(block_scales, dtype=np.float32)

        metadata = dict(
            n_elements=n,
            block_size=block_size,
            n_blocks=n_blocks,
            bits_map=bits_map.tobytes(),
            scales=scales_arr.tobytes(),
            block_data=b"".join(block_data_list),
            shape=tensor.shape,
        )
        return metadata["block_data"], metadata

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        n = metadata["n_elements"]
        block_size = metadata["block_size"]
        n_blocks = metadata["n_blocks"]
        bits_map = metadata["bits_map"]
        scales = np.frombuffer(metadata["scales"], dtype=np.float32).astype(np.float64)
        shape = metadata["shape"]

        result = np.zeros(n, dtype=np.float64)
        data_offset = 0

        for b in range(n_blocks):
            start = b * block_size
            end = min(start + block_size, n)
            block_len = end - start
            bits = bits_map[b] if b < len(bits_map) else 4
            codebook_size = 1 << bits

            raw = np.frombuffer(metadata["block_data"], dtype=np.uint8)
            indices = np.empty(block_len, dtype=np.uint8)
            for j in range(block_len):
                byte_idx = data_offset + j // 2
                if byte_idx < len(raw):
                    if j % 2 == 0:
                        indices[j] = raw[byte_idx] & 0x0F
                    else:
                        indices[j] = (raw[byte_idx] >> 4) & 0x0F
            data_offset += (block_len + 1) // 2

            result[start:end] = (
                indices.astype(np.float64) - codebook_size / 2 + 1
            ) * scales[b]

        return result.reshape(shape).astype(np.float32)
