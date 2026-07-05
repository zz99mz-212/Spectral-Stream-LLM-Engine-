from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class EfferenceCopy:
    """D18. EFFERVENCE-COPY: W_err = W_actual - W_pred, internal forward model."""

    name = "efference_copy"
    category = "novel_biological"

    def compress(
        self, tensor: np.ndarray, prediction_rank: int = 4
    ) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        m, n = t.shape
        r = min(prediction_rank, m, n)

        U, S, Vt = np.linalg.svd(t, full_matrices=False)
        pred = (U[:, :r] * S[:r]) @ Vt[:r, :]
        error = t - pred

        e_flat = error.ravel()
        thr = np.percentile(np.abs(e_flat), 85)
        mask = np.abs(error) > thr
        eidx = np.argwhere(mask)
        evals = error[mask]

        meta = dict(shape=t.shape, r=r)
        data = (
            _serialize(U[:, :r].astype(np.float32))
            + _serialize(S[:r].astype(np.float32))
            + _serialize(Vt[:r, :].astype(np.float32))
            + _serialize(eidx.astype(np.int16))
            + evals.astype(np.float16).tobytes()
        )
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        r = metadata["r"]
        m, n = shape

        pos = 0
        U_r = _deserialize(data[: m * r * 4]).reshape(m, r)
        pos += m * r * 4
        S_r = _deserialize(data[pos : pos + r * 4])
        pos += r * 4
        Vt_r = _deserialize(data[pos : pos + r * n * 4]).reshape(r, n)
        pos += r * n * 4

        pred = (U_r * S_r) @ Vt_r

        remaining = data[pos:]
        n_e = len(remaining) // 6
        if n_e > 0:
            eidx = np.frombuffer(remaining[: n_e * 4], dtype=np.int16).reshape(-1, 2)
            evals = np.frombuffer(
                remaining[n_e * 4 : n_e * 6], dtype=np.float16
            ).astype(np.float64)
            for i in range(min(n_e, len(eidx), len(evals))):
                ri, ci = int(eidx[i, 0]), int(eidx[i, 1])
                if ri < m and ci < n:
                    pred[ri, ci] += evals[i]

        return pred.astype(np.float32)
