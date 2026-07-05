from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class RetinoTopicMap:
    """D17. RETINO-TOPIC-MAP: ||w_i - w_{i+1}||² < ε, total variation smoothness."""

    name = "retino_topic_map"
    category = "novel_biological"

    def compress(self, tensor: np.ndarray, epsilon: float = 0.1) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        m, n = t.shape

        tv_row = np.sum(np.diff(t, axis=0) ** 2, axis=1)
        tv_col = np.sum(np.diff(t, axis=1) ** 2, axis=0)

        smooth_rows = tv_row < epsilon * np.max(tv_row)
        smooth_cols = tv_col < epsilon * np.max(tv_col)

        key_rows = np.where(~smooth_rows)[0]
        key_cols = np.where(~smooth_cols)[0]

        if len(key_rows) == 0:
            key_rows = np.array([0, m // 2, m - 1]) if m > 2 else np.array([0])
        if len(key_cols) == 0:
            key_cols = np.array([0, n // 2, n - 1]) if n > 2 else np.array([0])

        key_data = t[np.ix_(key_rows, key_cols)]

        meta = dict(
            shape=t.shape, key_rows=key_rows.tolist(), key_cols=key_cols.tolist()
        )
        data = _serialize(key_data.astype(np.float32))
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        key_rows = np.array(metadata["key_rows"])
        key_cols = np.array(metadata["key_cols"])
        m, n = shape

        key_data = _deserialize(data).reshape(len(key_rows), len(key_cols))

        recon = np.zeros((m, n), dtype=np.float64)
        for i in range(m):
            ri = np.searchsorted(key_rows, i)
            ri = min(ri, len(key_rows) - 1)
            r0 = key_rows[max(0, ri - 1)] if ri > 0 else key_rows[0]
            r1 = key_rows[min(ri, len(key_rows) - 1)]
            w_high = (i - r0) / (r1 - r0 + 1e-10) if r1 > r0 else 0.0
            for j in range(n):
                cj = np.searchsorted(key_cols, j)
                cj = min(cj, len(key_cols) - 1)
                c0 = key_cols[max(0, cj - 1)] if cj > 0 else key_cols[0]
                c1 = key_cols[min(cj, len(key_cols) - 1)]
                w_hc = (j - c0) / (c1 - c0 + 1e-10) if c1 > c0 else 0.0

                ri0, ri1 = max(0, ri - 1), min(len(key_rows) - 1, ri)
                cj0, cj1 = max(0, cj - 1), min(len(key_cols) - 1, cj)
                recon[i, j] = (
                    key_data[ri0, cj0] * (1 - w_high) * (1 - w_hc)
                    + key_data[min(ri0 + 1, len(key_rows) - 1), cj0]
                    * w_high
                    * (1 - w_hc)
                    + key_data[ri0, min(cj0 + 1, len(key_cols) - 1)]
                    * (1 - w_high)
                    * w_hc
                    + key_data[
                        min(ri0 + 1, len(key_rows) - 1), min(cj0 + 1, len(key_cols) - 1)
                    ]
                    * w_high
                    * w_hc
                )

        return recon.astype(np.float32)
