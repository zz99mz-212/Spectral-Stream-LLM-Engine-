"""Auto-generated from _class_wrappers.py."""

from __future__ import annotations

import math
import struct
from typing import Any, Dict, Tuple

import numpy as np

from spectralstream.core.math_primitives import (
    dct, idct, fwht, ifwht, next_power_of_two, LloydMaxQuantizer,
)

class KMeansQuant:
    """K-means clustering quantization."""

    name = "kmeans_quant"
    category = "quantization"

    def compress(self, tensor: np.ndarray, n_clusters: int = 16) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float32)
        flat = t.ravel()
        rng = np.random.RandomState(42)
        centroids = np.zeros(n_clusters, dtype=np.float32)
        centroids[0] = flat[rng.randint(len(flat))]
        for i in range(1, n_clusters):
            dists = np.min((flat[:, None] - centroids[:i][None, :]) ** 2, axis=1)
            centroids[i] = flat[rng.choice(len(flat), p=dists / (dists.sum() + 1e-30))]
        for _ in range(20):
            labels = np.argmin(
                np.abs(flat[:, None] - centroids[None, :]), axis=1
            ).astype(np.int32)
            new_c = np.array(
                [
                    flat[labels == i].mean() if np.any(labels == i) else centroids[i]
                    for i in range(n_clusters)
                ]
            )
            if np.allclose(centroids, new_c, atol=1e-6):
                break
            centroids = new_c
        labels = np.argmin(np.abs(flat[:, None] - centroids[None, :]), axis=1).astype(
            np.uint8
        )
        meta = dict(shape=tensor.shape, n_clusters=n_clusters, n_elements=tensor.size)
        data = centroids.astype(np.float32).tobytes() + labels.tobytes()
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        n_clusters = metadata["n_clusters"]
        shape = metadata["shape"]
        centroids = np.frombuffer(data[: n_clusters * 4], dtype=np.float32)
        labels = np.frombuffer(data[n_clusters * 4 :], dtype=np.uint8).copy()
        return centroids[labels].reshape(shape).astype(np.float32)



class LloydMaxQuant:
    """Lloyd-Max MSE-optimal scalar quantization."""

    name = "lloyd_max_quant"
    category = "quantization"

    def compress(self, tensor: np.ndarray, n_bits: int = 4) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float32)
        q = LloydMaxQuantizer(n_bits=n_bits)
        q.train(t)
        indices, centroids = q.compress(t)
        meta = dict(shape=tensor.shape, n_bits=n_bits, n_elements=tensor.size)
        data = centroids.tobytes() + indices.tobytes()
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        n_bits = metadata["n_bits"]
        shape = metadata["shape"]
        n = metadata["n_elements"]
        n_levels = 1 << n_bits
        centroids = np.frombuffer(data[: n_levels * 4], dtype=np.float32)
        indices = np.frombuffer(data[n_levels * 4 :], dtype=np.uint8).copy()
        return centroids[indices].reshape(shape).astype(np.float32)



class OctopusQuant:
    """Octopus quantization — 8-group parallel quantizer banks."""

    name = "octopus_quant"
    category = "quantization"

    def compress(self, tensor: np.ndarray, n_bits: int = 4) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        flat = t.ravel()
        n = len(flat)
        N = next_power_of_two(n)
        buf = np.zeros(N, dtype=np.float64)
        buf[:n] = flat
        rng = np.random.RandomState(42)
        signs = rng.choice([-1.0, 1.0], size=N)
        buf *= signs
        h = 1
        while h < N:
            for i in range(0, N, h * 2):
                for j in range(i, i + h):
                    u, v = buf[j], buf[j + h]
                    buf[j] = u + v
                    buf[j + h] = u - v
            h <<= 1
        buf /= math.sqrt(N)
        mu, sigma = float(np.mean(buf)), float(np.std(buf))
        scale = max(abs(mu - 4 * sigma), abs(mu + 4 * sigma), 1e-8)
        normed = np.clip(buf / scale, -1.0, 1.0)
        nL = 1 << n_bits
        cents = np.linspace(-1.0, 1.0, nL)
        for _ in range(50):
            bds = (cents[1:] + cents[:-1]) / 2.0
            idx = np.digitize(normed, bds)
            nc = np.array(
                [
                    np.mean(normed[idx == i]) if np.any(idx == i) else cents[i]
                    for i in range(nL)
                ]
            )
            if np.allclose(cents, nc, atol=1e-6):
                break
            cents = nc
        idx = np.clip(np.digitize(normed, cents[1:]), 0, nL - 1).astype(np.uint8)
        meta = dict(shape=tensor.shape, n=n, N=N, n_bits=n_bits)
        data = (
            struct.pack("<ff", mu, sigma)
            + cents.astype(np.float32).tobytes()
            + idx.tobytes()
        )
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        n = metadata["n"]
        N = metadata["N"]
        mu, sigma = struct.unpack_from("<ff", data, 0)
        nL = 1 << metadata["n_bits"]
        cents = np.frombuffer(data[8 : 8 + nL * 4], dtype=np.float32)
        idx = np.frombuffer(data[8 + nL * 4 : 8 + nL * 4 + N], dtype=np.uint8).copy()
        q = cents[idx] * sigma + mu
        h = 1
        while h < N:
            for i in range(0, N, h * 2):
                for j in range(i, i + h):
                    u, v = q[j], q[j + h]
                    q[j] = u + v
                    q[j + h] = u - v
            h <<= 1
        q /= math.sqrt(N)
        rng = np.random.RandomState(42)
        signs = rng.choice([-1.0, 1.0], size=N)
        q *= signs
        return q[:n].reshape(shape).astype(np.float32)



class WeightClustering8bit:
    """K-means clustering with 256-entry 8-bit codebook."""

    name = "weight_clustering_8bit"
    category = "quantization"

    def compress(self, tensor: np.ndarray, n_clusters: int = 256) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float32)
        flat = t.ravel()
        rng = np.random.RandomState(42)
        idx_c = (
            rng.choice(len(flat), min(n_clusters, len(flat)), replace=False)
            if len(flat) > n_clusters
            else np.arange(len(flat))
        )
        centroids = flat[idx_c].copy() if len(idx_c) > 0 else np.zeros(n_clusters)
        if len(centroids) < n_clusters:
            centroids = np.pad(centroids, (0, n_clusters - len(centroids)))
        centroids = centroids[:n_clusters]
        for _ in range(30):
            labels = np.argmin(
                np.abs(flat[:, None] - centroids[None, :]), axis=1
            ).astype(np.uint8)
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
        meta = dict(shape=tensor.shape, n_clusters=n_clusters, n_elements=tensor.size)
        data = centroids.astype(np.float32).tobytes() + labels.tobytes()
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        n_clusters = metadata["n_clusters"]
        shape = metadata["shape"]
        centroids = np.frombuffer(data[: n_clusters * 4], dtype=np.float32)
        labels = np.frombuffer(data[n_clusters * 4 :], dtype=np.uint8).copy()
        return centroids[labels].reshape(shape).astype(np.float32)



