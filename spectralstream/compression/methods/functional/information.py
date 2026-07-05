"""Auto-generated from inr_compression.py."""

from __future__ import annotations

import math
import struct
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spectralstream.core.math_primitives import dct, idct, next_power_of_two


def _bytes(obj: Any) -> int:
    if isinstance(obj, np.ndarray):
        return obj.nbytes
    if isinstance(obj, dict):
        return sum(_bytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(_bytes(x) for x in obj)
    return 0


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()


def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class InformationBottleneck:
    """Information Bottleneck — cluster to maximize relevance, minimize complexity."""

    name = "information_bottleneck"
    category = "functional"

    def compress(self, tensor: np.ndarray, **params) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        flat = t.ravel()
        n = len(flat)
        n_clusters = min(params.get("n_clusters", 16), n)
        rng = np.random.RandomState(42)
        idx = rng.choice(n, min(n_clusters, n), replace=False)
        centroids = flat[idx].copy()
        for _ in range(30):
            dists = np.abs(flat[:, None] - centroids[None, :])
            labels = np.argmin(dists, axis=1).astype(np.int32)
            new_c = np.array(
                [
                    np.mean(flat[labels == i]) if np.any(labels == i) else centroids[i]
                    for i in range(n_clusters)
                ]
            )
            if np.allclose(centroids, new_c, atol=1e-6):
                break
            centroids = new_c
        labels = np.argmin(np.abs(flat[:, None] - centroids[None, :]), axis=1).astype(
            np.uint8
        )
        n_bits = max(1, (n_clusters - 1).bit_length())
        meta = dict(n_clusters=n_clusters, n=n, shape=t.shape, n_bits=n_bits)
        data = _serialize(centroids.astype(np.float32))
        data += labels.tobytes()
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        n_clusters = metadata["n_clusters"]
        n = metadata["n"]
        shape = metadata["shape"]
        centroids = _deserialize(data[: n_clusters * 4])
        labels = np.frombuffer(
            data[n_clusters * 4 : n_clusters * 4 + n], dtype=np.uint8
        ).copy()
        return centroids[labels].reshape(shape).astype(np.float32)



class RateDistortionOptimal:
    """Rate-Distortion optimal bit allocation by group variance."""

    name = "rate_distortion_optimal"
    category = "functional"

    def compress(self, tensor: np.ndarray, **params) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        flat = t.ravel()
        n = len(flat)
        block_size = min(params.get("block_size", 64), n)
        n_blocks = (n + block_size - 1) // block_size
        padded = np.zeros(n_blocks * block_size, dtype=np.float64)
        padded[:n] = flat
        blocks = padded.reshape(n_blocks, block_size)
        vars_b = np.var(blocks, axis=1)
        vn = (vars_b - vars_b.min()) / (vars_b.max() - vars_b.min() + 1e-30)
        bits_arr = np.round(2 + vn * 6).clip(2, 8).astype(np.uint8)
        scales = np.max(np.abs(blocks), axis=1)
        scales = np.where(scales > 1e-10, scales / 127.0, 1.0)
        quantized = np.zeros_like(blocks, dtype=np.int8)
        for i in range(n_blocks):
            max_q = (1 << (int(bits_arr[i]) - 1)) - 1
            quantized[i] = np.clip(
                np.round(blocks[i] / scales[i] * max_q), -128, 127
            ).astype(np.int8)
        meta = dict(
            block_size=block_size, n=n, shape=t.shape, bits_arr=bits_arr.tolist()
        )
        data = _serialize(scales.astype(np.float32))
        data += quantized.tobytes()
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        n = metadata["n"]
        shape = metadata["shape"]
        block_size = metadata["block_size"]
        bits_arr = np.array(metadata["bits_arr"], dtype=np.uint8)
        n_blocks = (n + block_size - 1) // block_size
        scales = _deserialize(data[: n_blocks * 4])
        quantized = (
            np.frombuffer(data[n_blocks * 4 :], dtype=np.int8)
            .copy()
            .reshape(n_blocks, block_size)
        )
        recon = np.zeros((n_blocks, block_size), dtype=np.float64)
        for i in range(n_blocks):
            max_q = (1 << (int(bits_arr[i]) - 1)) - 1
            recon[i] = quantized[i].astype(np.float64) * scales[i] / max_q
        return recon.ravel()[:n].reshape(shape).astype(np.float32)



