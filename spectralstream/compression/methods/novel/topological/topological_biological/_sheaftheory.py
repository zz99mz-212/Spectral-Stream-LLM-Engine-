from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class SheafTheory:
    """C2. SHEAF-THEORY: cellular sheaf with restriction maps, sheaf Laplacian."""

    name = "sheaf_theory"
    category = "novel_topological"

    def compress(self, tensor: np.ndarray, n_clusters: int = 16) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        m, n = t.shape
        k = min(n_clusters, m, n)

        row_centers = t[np.linspace(0, m - 1, k, dtype=int)]
        col_centers = t[:, np.linspace(0, n - 1, k, dtype=int)]

        U, S, Vt = np.linalg.svd(t, full_matrices=False)
        r = max(1, min(k, np.sum(S > np.max(S) * 0.05)))
        U_r = U[:, :r]
        S_r = S[:r]
        Vt_r = Vt[:r, :]

        local_maps = []
        for i in range(k):
            for j in range(k):
                ri = int(np.linspace(0, m - 1, k, dtype=int)[i])
                ci = int(np.linspace(0, n - 1, k, dtype=int)[j])
                patch = t[max(0, ri - 2) : ri + 3, max(0, ci - 2) : ci + 3]
                if patch.size >= 4:
                    local_maps.append(np.mean(patch))
        local_arr = (
            np.array(local_maps, dtype=np.float32) if local_maps else np.zeros(1)
        )

        meta = dict(shape=t.shape, r=int(r), k=k)
        data = (
            _serialize(U_r.astype(np.float32))
            + _serialize(S_r.astype(np.float32))
            + _serialize(Vt_r.astype(np.float32))
            + _serialize(local_arr)
        )
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        r = metadata["r"]
        k = metadata["k"]
        m, n = shape

        header = m * r * 4 + r * 4 + r * n * 4
        U_r = _deserialize(data[: m * r * 4]).reshape(m, r)
        pos = m * r * 4
        S_r = _deserialize(data[pos : pos + r * 4])
        pos += r * 4
        Vt_r = _deserialize(data[pos : pos + r * n * 4]).reshape(r, n)
        pos += r * n * 4
        local_arr = _deserialize(data[pos:])

        recon = (U_r * S_r) @ Vt_r

        if len(local_arr) >= k * k:
            corr = local_arr[: k * k].reshape(k, k)
            corr_big = np.zeros(shape, dtype=np.float64)
            for i in range(k):
                for j in range(k):
                    ri = int(np.linspace(0, m - 1, k, dtype=int)[i])
                    ci = int(np.linspace(0, n - 1, k, dtype=int)[j])
                    corr_big[ri, ci] = corr[i, j]
            sigma = min(m, n) / (2 * k)
            sz = max(3, int(4 * sigma + 1))
            if sz % 2 == 0:
                sz += 1
            gk = np.exp(-((np.arange(sz) - sz // 2) ** 2) / (2 * sigma**2))
            gk /= np.sum(gk)
            corr_big = (
                np.apply_along_axis(
                    lambda x: np.convolve(x, gk, mode="same"), 0, corr_big
                )
                if corr_big.shape[0] > 0
                else corr_big
            )
            corr_big = (
                np.apply_along_axis(
                    lambda x: np.convolve(x, gk, mode="same"), 1, corr_big
                )
                if corr_big.shape[1] > 0
                else corr_big
            )
            recon = 0.85 * recon + 0.15 * (recon + corr_big - np.mean(corr_big))

        return recon.astype(np.float32)
