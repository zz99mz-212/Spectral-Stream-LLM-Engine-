from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class CohomologyCompress:
    """C5. COHOMOLOGY-COMPRESS: Čech cohomology via covering spaces."""

    name = "cohomology_compress"
    category = "novel_topological"

    def compress(self, tensor: np.ndarray, n_covers: int = 8) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        m, n = t.shape
        k = min(n_covers, m, n)

        cover_m = max(1, m // k)
        cover_n = max(1, n // k)

        cover_means = []
        cover_stds = []
        residuals = np.zeros_like(t)
        for i in range(k):
            for j in range(k):
                si, ei = i * cover_m, min((i + 1) * cover_m, m)
                sj, ej = j * cover_n, min((j + 1) * cover_n, n)
                patch = t[si:ei, sj:ej]
                cover_means.append(float(np.mean(patch)))
                cover_stds.append(float(np.std(patch)))
                residuals[si:ei, sj:ej] = patch - np.mean(patch)

        flat_res = residuals.ravel()
        thr = np.percentile(np.abs(flat_res), 85)
        mask = np.abs(residuals) > thr
        ridx = np.argwhere(mask)
        rvals = residuals[mask]

        meta = dict(shape=t.shape, k=k, cover_m=cover_m, cover_n=cover_n)
        data = (
            _serialize(np.array(cover_means, dtype=np.float32))
            + _serialize(np.array(cover_stds, dtype=np.float32))
            + _serialize(ridx.astype(np.int16))
            + rvals.astype(np.float16).tobytes()
        )
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        k = metadata["k"]
        cover_m = metadata["cover_m"]
        cover_n = metadata["cover_n"]
        m, n = shape

        n_covers = k * k
        means = _deserialize(data[: n_covers * 4])
        pos = n_covers * 4
        stds = _deserialize(data[pos : pos + n_covers * 4])
        pos += n_covers * 4

        recon = np.zeros(shape, dtype=np.float64)
        idx = 0
        for i in range(k):
            for j in range(k):
                si, ei = i * cover_m, min((i + 1) * cover_m, m)
                sj, ej = j * cover_n, min((j + 1) * cover_n, n)
                recon[si:ei, sj:ej] = (
                    np.random.RandomState(0).randn(ei - si, ej - sj) * stds[idx]
                    + means[idx]
                )
                idx += 1

        remaining = data[pos:]
        if len(remaining) >= 4:
            n_pts = len(remaining) // (4 + 2)
            ridx = np.frombuffer(remaining[: n_pts * 4], dtype=np.int16).reshape(-1, 2)
            rvals = np.frombuffer(remaining[n_pts * 4 :], dtype=np.float16).astype(
                np.float64
            )
            for ri in range(min(len(ridx), len(rvals))):
                rii, rji = int(ridx[ri, 0]) % m, int(ridx[ri, 1]) % n
                if rii < m and rji < n:
                    recon[rii, rji] += rvals[ri]

        return recon.astype(np.float32)
