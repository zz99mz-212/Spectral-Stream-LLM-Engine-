from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class NeuromodulationCode:
    """D10. NEUROMODULATION-CODE: Δw = η δ e, dopamine-gated plasticity."""

    name = "neuromodulation_code"
    category = "novel_biological"

    def compress(
        self, tensor: np.ndarray, keep_fraction: float = 0.15
    ) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        m, n = t.shape

        td_error = np.random.RandomState(0).randn(m, n)
        eligibility = np.abs(t)
        delta = td_error * eligibility
        dopamine_gate = np.abs(delta) > np.percentile(
            np.abs(delta), 100 * (1.0 - keep_fraction)
        )

        didx = np.argwhere(dopamine_gate)
        dvals = t[dopamine_gate]

        U, S, Vt = np.linalg.svd(t, full_matrices=False)
        r = max(1, min(8, len(S)))

        meta = dict(shape=t.shape, r=r)
        data = (
            _serialize(U[:, :r].astype(np.float32))
            + _serialize(S[:r].astype(np.float32))
            + _serialize(Vt[:r, :].astype(np.float32))
            + _serialize(didx.astype(np.int16))
            + dvals.astype(np.float16).tobytes()
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

        recon = (U_r * S_r) @ Vt_r

        remaining = data[pos:]
        n_d = len(remaining) // 6
        if n_d > 0:
            didx = np.frombuffer(remaining[: n_d * 4], dtype=np.int16).reshape(-1, 2)
            dvals = np.frombuffer(
                remaining[n_d * 4 : n_d * 6], dtype=np.float16
            ).astype(np.float64)
            for i in range(min(n_d, len(didx), len(dvals))):
                ri, ci = int(didx[i, 0]), int(didx[i, 1])
                if ri < m and ci < n:
                    recon[ri, ci] = dvals[i]

        return recon.astype(np.float32)
