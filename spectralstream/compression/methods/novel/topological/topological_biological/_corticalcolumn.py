from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class CorticalColumn:
    """D13. CORTICAL-COLUMN: N neurons → M minicolumns, centroids + deviations."""

    name = "cortical_column"
    category = "novel_biological"

    def compress(self, tensor: np.ndarray, n_columns: int = 8) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        m, n = t.shape
        k = min(n_columns, m, n)

        col_size = max(1, m // k)

        centroids = np.zeros((k, n), dtype=np.float64)
        deviations = []
        for i in range(k):
            si = i * col_size
            ei = min((i + 1) * col_size, m)
            if si >= m:
                break
            col = t[si:ei, :]
            centroids[i, :] = np.mean(col, axis=0)
            dev = col - centroids[i : i + 1, :]
            deviations.append(dev.ravel())

        all_devs = np.concatenate(deviations) if deviations else np.zeros(1)
        dev_flat = all_devs.ravel()
        thr = np.percentile(np.abs(dev_flat), 85)
        dev_mask = np.abs(dev_flat) > thr
        didx = np.argwhere(dev_mask).ravel()
        dvals = dev_flat[dev_mask]

        U_c, S_c, Vt_c = np.linalg.svd(centroids, full_matrices=False)
        r_c = max(1, min(4, len(S_c)))

        meta = dict(shape=t.shape, k=k, r_c=r_c, col_size=col_size)
        data = (
            _serialize(U_c[:, :r_c].astype(np.float32))
            + _serialize(S_c[:r_c].astype(np.float32))
            + _serialize(Vt_c[:r_c, :].astype(np.float32))
            + _serialize(didx.astype(np.int32))
            + dvals.astype(np.float16).tobytes()
        )
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        k = metadata["k"]
        r_c = metadata["r_c"]
        col_size = metadata["col_size"]
        m, n = shape

        pos = 0
        U_c = _deserialize(data[: k * r_c * 4]).reshape(k, r_c)
        pos += k * r_c * 4
        S_c = _deserialize(data[pos : pos + r_c * 4])
        pos += r_c * 4
        Vt_c = _deserialize(data[pos : pos + r_c * n * 4]).reshape(r_c, n)
        pos += r_c * n * 4

        centroids = (U_c * S_c) @ Vt_c

        recon = np.zeros((m, n), dtype=np.float64)
        for i in range(k):
            si = i * col_size
            ei = min((i + 1) * col_size, m)
            for row in range(si, ei):
                recon[row, :] = centroids[i, :]

        remaining = data[pos:]
        if len(remaining) >= 6:
            n_d = len(remaining) // (4 + 2)
            if n_d > 0:
                didx = _deserialize(remaining[: n_d * 4]).astype(int)
                dvals = np.frombuffer(remaining[n_d * 4 :], dtype=np.float16).astype(
                    np.float64
                )
                total_dev = min(len(didx), len(dvals))
                for j in range(total_dev):
                    if didx[j] < m * n:
                        ri, ci = divmod(didx[j], n)
                        if ri < m and ci < n:
                            recon[ri, ci] += dvals[j]

        return recon.astype(np.float32)
