"""Weight-specific non-uniform codebook via Lloyd-Max optimization."""

from __future__ import annotations

from typing import Tuple

import numpy as np


class LearnedCodebookQuant:
    """Lloyd-Max optimized non-uniform codebook per block.

    Learns optimal quantization levels for the specific weight distribution
    of each block. Better than uniform quantization for non-Gaussian distributions.
    """

    name = "learned_codebook_quant"
    category = "quantization"

    def compress(
        self,
        tensor: np.ndarray,
        n_bits: int = 4,
        block_size: int = 128,
        n_iter: int = 30,
    ) -> Tuple[bytes, dict]:
        flat = tensor.ravel().astype(np.float64)
        n = len(flat)
        codebook_size = 1 << n_bits

        n_blocks = (n + block_size - 1) // block_size
        all_indices = []
        all_centroids = []

        for b in range(n_blocks):
            start = b * block_size
            end = min(start + block_size, n)
            block = flat[start:end]

            quantiles = np.linspace(0, 100, codebook_size + 2)[1:-1]
            centroids = np.percentile(block, quantiles).astype(np.float64)

            for _ in range(n_iter):
                boundaries = (centroids[1:] + centroids[:-1]) / 2.0
                indices = np.clip(np.digitize(block, boundaries), 0, codebook_size - 1)
                new_centroids = np.array(
                    [
                        block[indices == i].mean()
                        if np.any(indices == i)
                        else centroids[i]
                        for i in range(codebook_size)
                    ]
                )
                if np.allclose(centroids, new_centroids, atol=1e-6):
                    break
                centroids = new_centroids

            boundaries = (centroids[1:] + centroids[:-1]) / 2.0
            indices = np.clip(np.digitize(block, boundaries), 0, codebook_size - 1)
            all_indices.append(indices)
            all_centroids.append(centroids)

        flat_indices = np.concatenate(all_indices)
        packed = _pack_nibbles(flat_indices)

        centroid_data = b"".join(c.astype(np.float32).tobytes() for c in all_centroids)
        metadata = dict(
            n_elements=n,
            n_bits=n_bits,
            block_size=block_size,
            centroids=centroid_data,
            n_blocks=n_blocks,
            codebook_size=codebook_size,
            shape=tensor.shape,
        )
        return packed, metadata

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        n = metadata["n_elements"]
        block_size = metadata["block_size"]
        codebook_size = metadata["codebook_size"]
        n_blocks = metadata["n_blocks"]
        shape = metadata["shape"]

        indices = _unpack_nibbles(data, n)
        centroid_data = metadata["centroids"]
        result = np.zeros(n, dtype=np.float64)

        for b in range(n_blocks):
            start = b * block_size
            end = min(start + block_size, n)
            c_offset = b * codebook_size * 4
            centroids = np.frombuffer(
                centroid_data[c_offset : c_offset + codebook_size * 4], dtype=np.float32
            ).astype(np.float64)
            result[start:end] = centroids[indices[start:end]]

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
