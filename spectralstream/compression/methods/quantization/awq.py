"""AWQ-style activation-aware weight quantization using statistical channel importance."""

from __future__ import annotations

import gc
from typing import Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()


def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()


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


class AWQQuant:
    """AWQ-style activation-aware weight quantization (no-calibration variant).

    Uses column importance (standard deviation) as proxy for activation magnitude
    to weight quantization precision toward important channels.
    """

    name = "awq_quant"
    category = "quantization"

    def compress(
        self, tensor: np.ndarray, bits: int = 4, block_size: int = 128
    ) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        if t.ndim < 2:
            t = t.reshape(1, -1)
        m, n = t.shape
        max_q = (1 << (bits - 1)) - 1
        f = t.reshape(-1, n)
        nr, nc = f.shape
        channel_std = np.std(f, axis=0).astype(np.float64) + 1e-10
        max_std = float(np.max(channel_std))
        importance = (
            channel_std / max_std if max_std > 1e-10 else np.ones(n, dtype=np.float64)
        )
        s = np.where(importance > 0.01, 1.0 / (1.0 + np.log(importance)), 1.0)
        f_scaled = f * s[None, :]
        bs = min(block_size, nc)
        p = (bs - nc % bs) % bs
        if p:
            f_scaled = np.pad(f_scaled, ((0, 0), (0, p)))
            s = np.pad(s, (0, p), mode="edge")
        nb = f_scaled.shape[1] // bs
        b = f_scaled.reshape(nr, nb, bs)
        scales = np.max(np.abs(b), axis=2)
        scales = np.where(scales > 1e-10, scales / max_q, 1.0)
        q = np.clip(np.round(b / scales[:, :, None]), -max_q, max_q).astype(np.int8)
        meta = dict(shape=tensor.shape, bits=bits, n_elements=tensor.size)
        data = (
            _serialize(s[:nc].astype(np.float32))
            + _serialize(scales.astype(np.float32))
            + q.tobytes()
        )
        del t, f, f_scaled, b, channel_std, s, scales, q
        gc.collect()
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        bits = metadata["bits"]
        n = metadata["n_elements"]
        f_shape = (
            shape[0] if len(shape) >= 2 else 1,
            shape[-1] if len(shape) >= 1 else n,
        )
        nr, nc = f_shape
        bs = min(128, nc)
        p = (bs - nc % bs) % bs
        nb_padded = (nc + p + bs - 1) // bs
        s = (
            _deserialize(data[: nc * 4])
            if len(data) > nc * 4
            else np.ones(nc, dtype=np.float32)
        )
        pos = nc * 4
        scales = _deserialize(data[pos : pos + nr * nb_padded * 4]).reshape(
            nr, nb_padded, 1
        )
        pos += nr * nb_padded * 4
        q = np.frombuffer(data[pos:], dtype=np.int8).reshape(nr, nb_padded, bs)
        recon_scaled = (q.astype(np.float32) * scales).reshape(nr, nb_padded * bs)[
            :, :nc
        ]
        recon = recon_scaled / np.maximum(s[None, :nc], 1e-10)
        del s, scales, q
        gc.collect()
        return recon.reshape(shape).astype(np.float32)


class AWQActivationAwareQuant:
    """Activation-aware quantization using column importance weighting.

    Uses column L2-norm as proxy for activation magnitude to allocate wider
    quantization ranges to more important columns.
    """

    name = "awq_activation_aware"
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

        col_norms = np.linalg.norm(mat, axis=0)
        col_importance = col_norms / (col_norms.sum() + 1e-10)

        base_scale = float(np.max(np.abs(mat)))
        col_scales = base_scale * (1.0 + 0.5 * col_importance / col_importance.max())

        all_indices = np.zeros(n, dtype=np.int32)
        all_scales = np.zeros(cols, dtype=np.float64)

        for c in range(cols):
            col_data = mat[:, c]
            scale = col_scales[c]
            all_scales[c] = scale
            indices = np.clip(
                np.round(col_data / scale + codebook_size / 2 - 1).astype(np.int32),
                0,
                codebook_size - 1,
            )
            all_indices[c * rows : (c + 1) * rows] = indices

        packed = _pack_nibbles(all_indices)
        metadata = dict(
            n_elements=n,
            rows=rows,
            cols=cols,
            n_bits=n_bits,
            block_size=block_size,
            scales=all_scales.astype(np.float32).tobytes(),
            col_norms=col_norms.astype(np.float32).tobytes(),
            shape=tensor.shape,
        )
        return packed, metadata

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        n = metadata["n_elements"]
        rows, cols = metadata["rows"], metadata["cols"]
        scales = np.frombuffer(metadata["scales"], dtype=np.float32).astype(np.float64)
        codebook_size = 1 << metadata["n_bits"]

        indices = _unpack_nibbles(data, n)
        result = np.zeros((rows, cols), dtype=np.float64)
        for c in range(cols):
            col_indices = indices[c * rows : (c + 1) * rows]
            result[:, c] = (
                col_indices.astype(np.float64) - codebook_size / 2 + 1
            ) * scales[c]

        return result.reshape(metadata["shape"]).astype(np.float32)
