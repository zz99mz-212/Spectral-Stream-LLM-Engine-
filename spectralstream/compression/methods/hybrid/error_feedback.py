"""Block quantization with inter-block error feedback propagation."""

from __future__ import annotations

from typing import Tuple

import numpy as np


class ErrorFeedbackQuant:
    """Block INT4 quantization with error feedback between blocks.

    After quantizing each block, the quantization error is added to the
    next block's input, preventing error accumulation across blocks.
    """

    name = "error_feedback_quant"
    category = "hybrid"

    def compress(
        self,
        tensor: np.ndarray,
        n_bits: int = 4,
        block_size: int = 128,
        feedback_gain: float = 0.8,
    ) -> Tuple[bytes, dict]:
        flat = tensor.ravel().astype(np.float64)
        n = len(flat)
        n_blocks = (n + block_size - 1) // block_size
        codebook_size = 1 << n_bits

        all_indices = []
        all_scales = []
        error_buf = np.zeros(n, dtype=np.float64)

        for b in range(n_blocks):
            start = b * block_size
            end = min(start + block_size, n)
            block = flat[start:end] + error_buf[start:end]

            amax = float(np.max(np.abs(block)))
            scale = amax / (codebook_size / 2 - 1) if amax > 1e-8 else 1.0
            all_scales.append(scale)

            indices = np.clip(
                np.round(block / scale + codebook_size / 2 - 1).astype(np.int32),
                0,
                codebook_size - 1,
            )
            all_indices.append(indices)

            reconstructed = (indices - codebook_size / 2 + 1) * scale
            error = flat[start:end] - reconstructed
            if b < n_blocks - 1:
                next_start = end
                next_end = min(next_start + block_size, n)
                error_buf[next_start:next_end] += (
                    error[: next_end - next_start] * feedback_gain
                )

        flat_indices = np.concatenate(all_indices)
        packed = _pack_nibbles(flat_indices)
        metadata = dict(
            n_elements=n,
            n_bits=n_bits,
            block_size=block_size,
            scales=np.array(all_scales, dtype=np.float32).tobytes(),
            n_blocks=n_blocks,
            shape=tensor.shape,
        )
        return packed, metadata

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        n = metadata["n_elements"]
        n_bits = metadata["n_bits"]
        block_size = metadata["block_size"]
        scales = np.frombuffer(metadata["scales"], dtype=np.float32).astype(np.float64)
        codebook_size = 1 << n_bits
        shape = metadata["shape"]

        indices = _unpack_nibbles(data, n)
        result = np.zeros(n, dtype=np.float64)
        n_blocks = (n + block_size - 1) // block_size
        for b in range(n_blocks):
            start = b * block_size
            end = min(start + block_size, n)
            result[start:end] = (
                indices[start:end].astype(np.float64) - codebook_size / 2 + 1
            ) * scales[b]

        return result.reshape(shape).astype(np.float32)


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
