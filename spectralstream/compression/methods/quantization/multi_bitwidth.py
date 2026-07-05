"""Mixed block sizes with per-block INT4 quantization adaptively chosen."""

from __future__ import annotations

from typing import Tuple

import numpy as np


class MultiBitWidthArchive:
    """Mixed 4-bit quantization with adaptive block-size selection per region.

    Uses small blocks (32) for high-variance regions and large blocks (128)
    for low-variance regions, optimizing the ratio vs quality tradeoff.
    """

    name = "multi_bitwidth_archive"
    category = "quantization"

    def compress(self, tensor: np.ndarray) -> Tuple[bytes, dict]:
        flat = tensor.ravel().astype(np.float64)
        n = len(flat)
        codebook_size = 16

        regions = []
        i = 0
        while i < n:
            window = min(128, n - i)
            region_var = np.var(flat[i : i + window])

            if region_var > 0.01:
                block_size = 32
            else:
                block_size = 128

            end = min(i + block_size, n)
            block = flat[i:end]
            amax = float(np.max(np.abs(block)))
            scale = amax / (codebook_size / 2 - 1) if amax > 1e-8 else 1.0
            indices = np.clip(
                np.round(block / scale + codebook_size / 2 - 1).astype(np.int32),
                0,
                codebook_size - 1,
            )

            regions.append({"start": i, "end": end, "scale": scale, "bs": end - i})
            i = end

        n_regions = len(regions)
        all_data = bytearray()
        all_scales = []
        all_bs = []

        for r in regions:
            block = flat[r["start"] : r["end"]]
            amax = float(np.max(np.abs(block)))
            scale = amax / (codebook_size / 2 - 1) if amax > 1e-8 else 1.0
            all_scales.append(scale)
            all_bs.append(r["end"] - r["start"])
            indices = np.clip(
                np.round(block / scale + codebook_size / 2 - 1).astype(np.int32),
                0,
                codebook_size - 1,
            )
            all_data.extend(_pack_nibbles(indices))

        metadata = dict(
            n_elements=n,
            n_regions=n_regions,
            scales=np.array(all_scales, dtype=np.float32).tobytes(),
            block_sizes=np.array(all_bs, dtype=np.uint16).tobytes(),
            data=bytes(all_data),
            shape=tensor.shape,
        )
        return metadata["data"], metadata

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        n = metadata["n_elements"]
        n_regions = metadata["n_regions"]
        scales = np.frombuffer(metadata["scales"], dtype=np.float32).astype(np.float64)
        block_sizes = np.frombuffer(metadata["block_sizes"], dtype=np.uint16)
        shape = metadata["shape"]

        result = np.zeros(n, dtype=np.float64)
        data_offset = 0
        pos = 0

        for r in range(n_regions):
            bs = int(block_sizes[r])
            indices = _unpack_nibbles(metadata["data"][data_offset:], bs)
            data_offset += (bs + 1) // 2
            result[pos : pos + bs] = (indices.astype(np.float64) - 7.5) * scales[r]
            pos += bs

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
