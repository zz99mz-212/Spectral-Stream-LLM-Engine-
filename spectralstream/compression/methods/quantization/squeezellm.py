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


def _kmeans_1d(
    data: np.ndarray, n_clusters: int, rng: np.random.RandomState, n_iter: int = 30
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(data)
    idx_c = (
        rng.choice(n, min(n_clusters, n), replace=False)
        if n > n_clusters
        else np.arange(n)
    )
    centroids = data[idx_c].copy() if len(idx_c) > 0 else np.zeros(n_clusters)
    if len(centroids) < n_clusters:
        centroids = np.pad(centroids, (0, n_clusters - len(centroids)))
    centroids = centroids[:n_clusters]
    for _ in range(n_iter):
        labels = np.argmin(np.abs(data[:, None] - centroids[None, :]), axis=1).astype(
            np.uint8
        )
        new_c = np.array(
            [
                np.mean(data[labels == i]) if np.any(labels == i) else centroids[i]
                for i in range(n_clusters)
            ]
        )
        if np.allclose(centroids, new_c, atol=1e-6):
            break
        centroids = new_c
    labels = np.argmin(np.abs(data[:, None] - centroids[None, :]), axis=1).astype(
        np.uint8
    )
    return centroids, labels


class SqueezeLLMNonuniform:
    """Non-uniform codebook with outlier separation."""

    name = "squeezellm_nonuniform"
    category = "quantization"

    def compress(self, tensor: np.ndarray, n_bits: int = 4) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float32)
        flat = t.ravel()
        n = len(flat)
        nB = 1 << n_bits
        outlier_frac = 0.005
        n_out = max(1, int(n * outlier_frac))
        order = np.argsort(np.abs(flat))[::-1]
        out_idx = order[:n_out]
        outlier_vals = flat[out_idx].astype(np.float16)
        inlier_mask = np.ones(n, dtype=bool)
        inlier_mask[out_idx] = False
        inliers = flat[inlier_mask]
        rng = np.random.RandomState(42)
        centroids, labels = _kmeans_1d(inliers, nB, rng)
        bitmask = np.packbits(inlier_mask)
        meta = dict(shape=tensor.shape, n_elements=n, n_out=n_out, n_bits=n_bits)
        data = (
            centroids.astype(np.float32).tobytes()
            + bitmask.tobytes()
            + outlier_vals.tobytes()
            + labels.tobytes()
        )
        del t, flat, inliers, centroids, labels, outlier_vals
        gc.collect()
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        n = metadata["n_elements"]
        n_out = metadata["n_out"]
        n_bits = metadata["n_bits"]
        nB = 1 << n_bits
        centroids = np.frombuffer(data[: nB * 4], dtype=np.float32)
        mask_bytes = (n + 7) // 8
        bitmask = np.frombuffer(data[nB * 4 : nB * 4 + mask_bytes], dtype=np.uint8)
        outlier_bytes = n_out * 2
        outliers = np.frombuffer(
            data[nB * 4 + mask_bytes : nB * 4 + mask_bytes + outlier_bytes],
            dtype=np.float16,
        ).astype(np.float32)
        labels = np.frombuffer(
            data[nB * 4 + mask_bytes + outlier_bytes :], dtype=np.uint8
        ).copy()
        inlier_mask = np.unpackbits(bitmask)[:n].astype(bool)
        recon = np.zeros(n, dtype=np.float32)
        recon[inlier_mask] = centroids[labels]
        n_out_actual = min(n_out, n)
        recon[~inlier_mask] = outliers[:n_out_actual]
        del centroids, bitmask, outliers, labels
        gc.collect()
        return recon.reshape(shape).astype(np.float32)


class SqueezeLLMNonUniformV2:
    """Non-uniform quantization with two-component outlier separation.

    Separates outliers (top 0.5% magnitude) and stores at FP32, quantizes
    remaining 99.5% with Lloyd-Max optimized non-uniform codebook.
    """

    name = "squeezellm_nonuniform_v2"
    category = "quantization"

    def compress(
        self,
        tensor: np.ndarray,
        n_bits: int = 4,
        outlier_pct: float = 99.5,
        n_codebook: int = 16,
    ) -> Tuple[bytes, dict]:
        flat = tensor.ravel().astype(np.float64)
        n = len(flat)

        threshold = np.percentile(np.abs(flat), outlier_pct)
        outlier_mask = np.abs(flat) >= threshold
        non_outlier_mask = ~outlier_mask

        outliers = flat[outlier_mask]
        non_outliers = flat[non_outlier_mask]

        if len(non_outliers) < n_codebook:
            outlier_mask[:] = False
            non_outliers = flat
            outliers = np.array([], dtype=np.float64)

        n_clusters = min(n_codebook, len(non_outliers))
        n_clusters = max(n_clusters, 2)
        quantiles = np.linspace(0, 100, n_clusters + 2)[1:-1]
        centroids = np.percentile(non_outliers, quantiles).astype(np.float64)

        for _ in range(20):
            boundaries = (centroids[1:] + centroids[:-1]) / 2.0
            indices = np.clip(np.digitize(non_outliers, boundaries), 0, n_clusters - 1)
            new_centroids = np.array(
                [
                    non_outliers[indices == i].mean()
                    if np.any(indices == i)
                    else centroids[i]
                    for i in range(n_clusters)
                ]
            )
            if np.allclose(centroids, new_centroids, atol=1e-6):
                break
            centroids = new_centroids

        boundaries = (centroids[1:] + centroids[:-1]) / 2.0
        non_outlier_indices = np.clip(
            np.digitize(non_outliers, boundaries), 0, n_clusters - 1
        )

        n_outlier_bytes = (n + 7) // 8
        mask_bytes = bytearray(n_outlier_bytes)
        for i in range(n):
            if outlier_mask[i]:
                mask_bytes[i // 8] |= 1 << (i % 8)

        indices_packed = _pack_nibbles(non_outlier_indices)

        metadata = dict(
            n_elements=n,
            outlier_mask=bytes(mask_bytes),
            outlier_values=outliers.astype(np.float32).tobytes(),
            centroids=centroids.astype(np.float32).tobytes(),
            indices_packed=indices_packed,
            n_outliers=int(outlier_mask.sum()),
            n_clusters=n_clusters,
            shape=tensor.shape,
        )
        return indices_packed, metadata

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        n = metadata["n_elements"]
        outlier_mask = np.zeros(n, dtype=bool)
        mask_bytes = metadata["outlier_mask"]
        for i in range(n):
            byte_idx = i // 8
            bit_idx = i % 8
            if byte_idx < len(mask_bytes) and (mask_bytes[byte_idx] >> bit_idx) & 1:
                outlier_mask[i] = True

        outlier_values = np.frombuffer(
            metadata["outlier_values"], dtype=np.float32
        ).astype(np.float64)
        centroids = np.frombuffer(metadata["centroids"], dtype=np.float32).astype(
            np.float64
        )
        n_outliers = metadata["n_outliers"]
        n_non = n - n_outliers

        non_outlier_indices = _unpack_nibbles(metadata["indices_packed"], n_non)

        result = np.zeros(n, dtype=np.float64)
        result[outlier_mask] = outlier_values[:n_outliers]
        result[~outlier_mask] = centroids[non_outlier_indices[:n_non]]

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
