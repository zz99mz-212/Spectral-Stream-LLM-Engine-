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

class CrossLayerDelta:
    """Cross-layer delta encoding — store anchor row + INT4 deltas."""

    name = "cross_layer_delta"
    category = "functional"

    def compress(self, tensor: np.ndarray, **params) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        if t.ndim < 2:
            t = t.reshape(1, -1)
        m, n = t.shape
        anchor = t[0].copy()
        max_val = max(np.abs(anchor).max(), 1e-10)
        scale = max_val / 7.0
        quant_anchor = np.clip(np.round(anchor / scale), -7, 7).astype(np.int8)
        packed_anchor = bytearray()
        for i in range(0, n, 2):
            lo = (int(quant_anchor[i]) + 8) & 0x0F
            hi = (int(quant_anchor[i + 1]) + 8) & 0x0F if i + 1 < n else 0
            packed_anchor.append(lo | (hi << 4))
        deltas = t[1:] - t[:-1]
        d_max = np.max(np.abs(deltas), axis=1)
        d_scale = np.where(d_max > 1e-10, d_max / 7.0, 1.0)
        d_quant = np.zeros((m - 1, n), dtype=np.int8)
        for i in range(m - 1):
            d_quant[i] = np.clip(np.round(deltas[i] / d_scale[i]), -7, 7).astype(
                np.int8
            )
        d_packed = bytearray()
        for i in range(m - 1):
            for j in range(0, n, 2):
                lo = (int(d_quant[i, j]) + 8) & 0x0F
                hi = (int(d_quant[i, j + 1]) + 8) & 0x0F if j + 1 < n else 0
                d_packed.append(lo | (hi << 4))
        meta = dict(n=n, m=m, shape=t.shape, anchor_scale=scale)
        data = (
            struct.pack("<ii", m, n)
            + bytes(packed_anchor)
            + _serialize(np.array([scale], dtype=np.float32))
        )
        data += _serialize(d_scale.astype(np.float32))
        data += bytes(d_packed)
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        m = metadata["m"]
        n = metadata["n"]
        shape = metadata["shape"]
        anchor_scale = metadata["anchor_scale"]
        pos = struct.calcsize("<ii")
        n_packed = (n + 1) // 2
        packed_anchor = data[pos : pos + n_packed]
        pos += n_packed
        recon = np.zeros((m, n), dtype=np.float64)
        for j in range(n):
            byte_idx = j // 2
            nibble = (
                (packed_anchor[byte_idx] >> (4 * (j % 2))) & 0x0F
                if j % 2 == 1
                else packed_anchor[byte_idx] & 0x0F
            )
            recon[0, j] = (int(nibble) - 8) * anchor_scale
        pos += 4
        d_scale = _deserialize(data[pos : pos + (m - 1) * 4])
        pos += (m - 1) * 4
        d_packed = data[pos:]
        for i in range(1, m):
            for j in range(n):
                byte_idx = ((i - 1) * n + j) // 2
                dbyte = d_packed[byte_idx]
                nibble = (
                    (dbyte >> (4 * ((i - 1) * n + j) % 2)) & 0x0F
                    if (((i - 1) * n + j) % 2 == 1)
                    else dbyte & 0x0F
                )
                delta = (int(nibble) - 8) * d_scale[i - 1]
                recon[i, j] = recon[i - 1, j] + delta
        return recon.astype(np.float32)



class HierarchicalPQ:
    """Hierarchical clustered product quantization."""

    name = "hierarchical_pq"
    category = "functional"

    def compress(self, tensor: np.ndarray, **params) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        if t.ndim < 2:
            t = t.reshape(1, -1)
        m, n = t.shape
        n_sub = min(params.get("n_subquantizers", 8), n)
        n_centroids = min(params.get("n_centroids", 16), 256)
        sub_dim = n // n_sub
        if sub_dim < 1:
            sub_dim = 1
            n_sub = n
        flat = t.ravel()
        rng = np.random.RandomState(42)
        codebooks = []
        indices = np.zeros((m, n_sub), dtype=np.uint8)
        for s in range(n_sub):
            si = s * sub_dim
            ei = min(si + sub_dim, n)
            if m == 1:
                sub_data = flat[si:ei]
            else:
                sub_data = t[:, si:ei].ravel()
            n_pad = max(1, int(np.prod(t[:, si:ei].shape) if t.ndim >= 2 else 1))
            sub_data = np.atleast_1d(sub_data)
            idx_c = (
                rng.choice(
                    len(sub_data), min(n_centroids, len(sub_data)), replace=False
                )
                if len(sub_data) > n_centroids
                else np.arange(len(sub_data))
            )
            centroids = (
                sub_data[idx_c].copy() if len(idx_c) > 0 else np.zeros(n_centroids)
            )
            if len(centroids) < n_centroids:
                centroids = np.pad(centroids, (0, n_centroids - len(centroids)))
            centroids = centroids[:n_centroids]
            labels = np.zeros(m, dtype=np.uint8)
            for _ in range(20):
                if m == 1:
                    vecs = sub_data.reshape(1, -1)
                else:
                    vecs = t[:, si:ei]
                dists = np.zeros((m, n_centroids))
                for c in range(n_centroids):
                    dists[:, c] = np.sum((vecs - centroids[c]) ** 2, axis=1)
                labels = np.argmin(dists, axis=1).astype(np.uint8)
                new_c = np.zeros(n_centroids)
                for c in range(n_centroids):
                    mask = labels == c
                    if np.any(mask):
                        new_c[c] = np.mean(vecs[mask])
                    else:
                        new_c[c] = centroids[c]
                if np.allclose(centroids, new_c, atol=1e-6):
                    break
                centroids = new_c
            codebooks.append(centroids.astype(np.float32))
            if m > 0 and n_sub > 0:
                indices[:, s] = labels
        meta = dict(
            n_sub=n_sub,
            n_centroids=n_centroids,
            sub_dim=sub_dim,
            shape=t.shape,
            m=m,
            n=n,
        )
        data = b"".join(c.tobytes() for c in codebooks)
        data += indices.tobytes()
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        n_sub = metadata["n_sub"]
        n_centroids = metadata["n_centroids"]
        sub_dim = metadata["sub_dim"]
        m = metadata["m"]
        n = metadata["n"]
        shape = metadata["shape"]
        codebooks = []
        pos = 0
        for _ in range(n_sub):
            codebooks.append(
                np.frombuffer(
                    data[pos : pos + n_centroids * 4], dtype=np.float32
                ).copy()
            )
            pos += n_centroids * 4
        indices = np.frombuffer(data[pos : pos + m * n_sub], dtype=np.uint8).reshape(
            m, n_sub
        )
        recon = np.zeros((m, n), dtype=np.float64)
        for s in range(n_sub):
            si = s * sub_dim
            ei = min(si + sub_dim, n)
            for i in range(m):
                recon[i, si:ei] = codebooks[s][indices[i, s]]
        if len(shape) == 1:
            return recon.ravel()[: shape[0]].astype(np.float32)
        return recon.reshape(shape).astype(np.float32)



class FisherInformationWeighted:
    """Fisher-information-weighted bit allocation per dimension."""

    name = "fisher_information_weighted"
    category = "functional"

    def compress(self, tensor: np.ndarray, **params) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        flat = t.ravel()
        n = len(flat)
        H = np.outer(flat, flat) + 1e-10 * np.eye(n)
        fisher = np.diag(H)
        fn = (fisher - fisher.min()) / (fisher.max() - fisher.min() + 1e-30)
        bits_arr = np.round(2 + fn * 6).clip(2, 8).astype(np.uint8)
        total_bits = int(bits_arr.sum())
        block_size = min(params.get("block_size", 64), n)
        n_blocks = (n + block_size - 1) // block_size
        padded = np.zeros(n_blocks * block_size, dtype=np.float64)
        padded[:n] = flat
        blocks = padded.reshape(n_blocks, block_size)
        scales = np.max(np.abs(blocks), axis=1)
        scales = np.where(scales > 1e-10, scales / 127.0, 1.0)
        q = np.zeros((n_blocks, block_size), dtype=np.int8)
        for i in range(n_blocks):
            max_q = (1 << int(bits_arr[i])) - 1
            q[i] = np.clip(
                np.round(blocks[i] / scales[i] * max_q / 2), -128, 127
            ).astype(np.int8)
        meta = dict(
            n=n, block_size=block_size, shape=t.shape, bits_arr=bits_arr.tolist()
        )
        data = _serialize(scales.astype(np.float32))
        data += q.tobytes()
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        n = metadata["n"]
        block_size = metadata["block_size"]
        shape = metadata["shape"]
        bits_arr = np.array(metadata["bits_arr"], dtype=np.uint8)
        n_blocks = (n + block_size - 1) // block_size
        scales = _deserialize(data[: n_blocks * 4])
        q = (
            np.frombuffer(data[n_blocks * 4 :], dtype=np.int8)
            .copy()
            .reshape(n_blocks, block_size)
        )
        recon = np.zeros((n_blocks, block_size), dtype=np.float64)
        for i in range(n_blocks):
            max_q = (1 << int(bits_arr[i])) - 1
            recon[i] = q[i].astype(np.float64) * scales[i] / max(max_q / 2, 1)
        return recon.ravel()[:n].reshape(shape).astype(np.float32)



