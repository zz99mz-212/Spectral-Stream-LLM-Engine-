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


def _kmeans_fit(
    vecs: np.ndarray, n_centroids: int, rng: np.random.RandomState, n_iter: int = 20
) -> Tuple[np.ndarray, np.ndarray]:
    m = vecs.shape[0]
    idx_c = rng.choice(m, min(n_centroids, m), replace=False)
    centroids = (
        vecs[idx_c].copy() if len(idx_c) > 0 else np.zeros((n_centroids, vecs.shape[1]))
    )
    if len(centroids) < n_centroids:
        centroids = np.pad(centroids, ((0, n_centroids - len(centroids)), (0, 0)))
    centroids = centroids[:n_centroids]
    for _ in range(n_iter):
        dists = np.sum((vecs[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        labels = np.argmin(dists, axis=1).astype(np.uint8)
        new_c = np.zeros_like(centroids)
        for c in range(n_centroids):
            mask = labels == c
            if np.any(mask):
                new_c[c] = np.mean(vecs[mask], axis=0)
            else:
                new_c[c] = centroids[c]
        if np.allclose(centroids, new_c, atol=1e-6):
            break
        centroids = new_c
    labels = np.argmin(
        np.sum((vecs[:, None, :] - centroids[None, :, :]) ** 2, axis=2), axis=1
    ).astype(np.uint8)
    return centroids, labels


class ProductQuantization:
    """Product quantization — split vectors into sub-quantizers."""

    name = "product_quantization"
    category = "quantization"

    def compress(
        self, tensor: np.ndarray, n_sub: int = 8, n_centroids: int = 16
    ) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        if t.ndim < 2:
            t = t.reshape(1, -1)
        m = t.shape[0]
        nc = t.shape[1]
        sub_dim = nc // n_sub
        if sub_dim < 1:
            sub_dim = 1
            n_sub = nc
        rng = np.random.RandomState(42)
        codebooks = []
        indices = np.zeros((m, n_sub), dtype=np.uint8)
        for s in range(n_sub):
            si = s * sub_dim
            ei = min(si + sub_dim, nc)
            vecs = t[:, si:ei]
            centroids, labels = _kmeans_fit(vecs, n_centroids, rng)
            codebooks.append(centroids.astype(np.float32))
            indices[:, s] = labels
        meta = dict(
            shape=tensor.shape,
            n_sub=n_sub,
            n_centroids=n_centroids,
            sub_dim=sub_dim,
            m=m,
            nc=nc,
        )
        data = b"".join(cb.tobytes() for cb in codebooks) + indices.tobytes()
        del t, codebooks, indices
        gc.collect()
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        n_sub = metadata["n_sub"]
        n_centroids = metadata["n_centroids"]
        sub_dim = metadata["sub_dim"]
        m = metadata["m"]
        nc = metadata["nc"]
        codebooks = []
        pos = 0
        for _ in range(n_sub):
            codebooks.append(
                np.frombuffer(
                    data[pos : pos + n_centroids * sub_dim * 4], dtype=np.float32
                ).reshape(n_centroids, sub_dim)
            )
            pos += n_centroids * sub_dim * 4
        indices = np.frombuffer(data[pos : pos + m * n_sub], dtype=np.uint8).reshape(
            m, n_sub
        )
        recon = np.zeros((m, nc), dtype=np.float64)
        for s in range(n_sub):
            si = s * sub_dim
            ei = min(si + sub_dim, nc)
            recon[:, si:ei] = codebooks[s][indices[:, s]]
        del codebooks, indices
        gc.collect()
        return recon.reshape(shape).astype(np.float32)


class ResidualVQ:
    """Residual vector quantization with successive codebook stages."""

    name = "residual_vq"
    category = "quantization"

    def compress(
        self, tensor: np.ndarray, n_stages: int = 3, n_centroids: int = 8
    ) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        if t.ndim < 2:
            t = t.reshape(1, -1)
        m, nc = t.shape
        rng = np.random.RandomState(42)
        residual = t.copy()
        all_indices = []
        all_codebooks = []
        for stage in range(n_stages):
            centroids, labels = _kmeans_fit(residual, n_centroids, rng)
            all_codebooks.append(centroids.astype(np.float32))
            all_indices.append(labels)
            residual = residual - centroids[labels]
        meta = dict(
            shape=tensor.shape, n_stages=n_stages, n_centroids=n_centroids, m=m, nc=nc
        )
        data = b"".join(cb.tobytes() for cb in all_codebooks)
        data += b"".join(idx.tobytes() for idx in all_indices)
        del t, residual, all_codebooks, all_indices
        gc.collect()
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        n_stages = metadata["n_stages"]
        n_centroids = metadata["n_centroids"]
        m = metadata["m"]
        nc = metadata["nc"]
        pos = 0
        recon = np.zeros((m, nc), dtype=np.float64)
        for stage in range(n_stages):
            cb = np.frombuffer(
                data[pos : pos + n_centroids * nc * 4], dtype=np.float32
            ).reshape(n_centroids, nc)
            pos += n_centroids * nc * 4
            indices = np.frombuffer(data[pos : pos + m], dtype=np.uint8)
            pos += m
            recon += cb[indices]
        gc.collect()
        return recon.reshape(shape).astype(np.float32)


class AdditiveCodebook:
    """Additive codebook quantization (AQLM-style)."""

    name = "additive_codebook"
    category = "quantization"

    def compress(
        self, tensor: np.ndarray, n_codebooks: int = 4, n_centroids: int = 8
    ) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        if t.ndim < 2:
            t = t.reshape(1, -1)
        m, nc = t.shape
        rng = np.random.RandomState(42)
        codebooks = []
        indices = np.zeros((n_codebooks, m), dtype=np.uint8)
        residual = t.copy()
        for cb_idx in range(n_codebooks):
            centroids, labels = _kmeans_fit(residual, n_centroids, rng)
            codebooks.append(centroids.astype(np.float32))
            indices[cb_idx] = labels
            residual -= centroids[labels]
        meta = dict(
            shape=tensor.shape,
            n_codebooks=n_codebooks,
            n_centroids=n_centroids,
            m=m,
            nc=nc,
        )
        data = b"".join(cb.tobytes() for cb in codebooks) + indices.tobytes()
        del t, codebooks, indices, residual
        gc.collect()
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        n_codebooks = metadata["n_codebooks"]
        n_centroids = metadata["n_centroids"]
        m = metadata["m"]
        nc = metadata["nc"]
        pos = 0
        recon = np.zeros((m, nc), dtype=np.float64)
        for cb_idx in range(n_codebooks):
            cb = np.frombuffer(
                data[pos : pos + n_centroids * nc * 4], dtype=np.float32
            ).reshape(n_centroids, nc)
            pos += n_centroids * nc * 4
        indices = np.frombuffer(data[pos:], dtype=np.uint8).reshape(n_codebooks, m)
        pos = 0
        for cb_idx in range(n_codebooks):
            cb = np.frombuffer(
                data[pos : pos + n_centroids * nc * 4], dtype=np.float32
            ).reshape(n_centroids, nc)
            pos += n_centroids * nc * 4
            recon += cb[indices[cb_idx]]
        del indices
        gc.collect()
        return recon.reshape(shape).astype(np.float32)
