from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class OptimalTransportMap:
    """C9. OPTIMAL-TRANSPORT-MAP: transport map T(x) = ∇φ(x)."""

    name = "optimal_transport_map"
    category = "novel_topological"

    def compress(self, tensor: np.ndarray, n_bins: int = 32) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        m, n = t.shape
        flat = t.ravel()

        src_hist, src_edges = np.histogram(flat, bins=n_bins, density=True)
        src_centers = (src_edges[:-1] + src_edges[1:]) * 0.5

        uniform = np.ones(n_bins) / n_bins
        src_cdf = np.cumsum(src_hist) / (np.sum(src_hist) + 1e-30)
        target_cdf = np.cumsum(uniform)

        transport_map = np.interp(target_cdf, src_cdf, src_centers)

        sorted_flat = np.sort(flat)
        n_keep = max(1, int(0.02 * len(sorted_flat)))
        top_k_idx = np.argpartition(np.abs(flat), -n_keep)[-n_keep:]
        top_k_vals = flat[top_k_idx]

        meta = dict(shape=t.shape, n_bins=n_bins, n_top=len(top_k_idx))
        data = (
            _serialize(transport_map.astype(np.float32))
            + _serialize(top_k_idx.astype(np.int32))
            + top_k_vals.astype(np.float16).tobytes()
        )
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        n_bins = metadata["n_bins"]
        n_top = metadata["n_top"]
        N = int(np.prod(shape))

        transport_map = _deserialize(data[: n_bins * 4])
        pos = n_bins * 4
        top_k_idx = _deserialize(data[pos : pos + n_top * 4]).astype(int)
        pos += n_top * 4
        top_k_vals = np.frombuffer(
            data[pos : pos + n_top * 2], dtype=np.float16
        ).astype(np.float64)

        uniform_samples = np.linspace(0, 1, N)
        recon_flat = np.interp(
            uniform_samples, np.linspace(0, 1, n_bins), transport_map
        )

        for i, idx in enumerate(top_k_idx):
            if idx < N:
                recon_flat[idx] = (
                    top_k_vals[i] if i < len(top_k_vals) else recon_flat[idx]
                )

        return recon_flat.reshape(shape).astype(np.float32)
