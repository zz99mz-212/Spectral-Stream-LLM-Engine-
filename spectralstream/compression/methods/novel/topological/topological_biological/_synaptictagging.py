from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class SynapticTagging:
    """D7. SYNAPTIC-TAGGING: only tagged synapses consolidated."""

    name = "synaptic_tagging"
    category = "novel_biological"

    def compress(
        self, tensor: np.ndarray, tag_fraction: float = 0.15
    ) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        m, n = t.shape

        magnitude = np.abs(t)
        threshold = np.percentile(magnitude, 100 * (1.0 - tag_fraction))
        tag_mask = magnitude > threshold

        tidx = np.argwhere(tag_mask)
        tvals = t[tag_mask]

        tag_strength = magnitude / (np.max(magnitude) + 1e-30)
        consolidated = t * tag_strength

        U, S, Vt = np.linalg.svd(consolidated, full_matrices=False)
        r = max(1, np.sum(S > np.max(S) * 0.02))

        meta = dict(shape=t.shape, r=int(r))
        data = (
            _serialize(U[:, :r].astype(np.float32))
            + _serialize(S[:r].astype(np.float32))
            + _serialize(Vt[:r, :].astype(np.float32))
            + _serialize(tidx.astype(np.int16))
            + tvals.astype(np.float16).tobytes()
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
        n_tag = len(remaining) // 6 if len(remaining) >= 6 else 0
        if n_tag > 0:
            tidx = np.frombuffer(remaining[: n_tag * 4], dtype=np.int16).reshape(-1, 2)
            tvals = np.frombuffer(
                remaining[n_tag * 4 : n_tag * 6], dtype=np.float16
            ).astype(np.float64)
            for i in range(min(n_tag, len(tidx), len(tvals))):
                ri, ci = int(tidx[i, 0]), int(tidx[i, 1])
                if ri < m and ci < n:
                    recon[ri, ci] = tvals[i]

        return recon.astype(np.float32)
