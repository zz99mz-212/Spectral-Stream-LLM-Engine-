from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class SynapseElimination:
    """D14. SYNAPSE-ELIMINATION: pruning prob ∝ 1/|w|, iterative prune+regrow."""

    name = "synapse_elimination"
    category = "novel_biological"

    def compress(
        self, tensor: np.ndarray, keep_fraction: float = 0.2, n_iter: int = 5
    ) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        m, n = t.shape

        pruned = t.copy()
        rng = np.random.RandomState(0)
        for _ in range(n_iter):
            magnitude = np.abs(pruned)
            prob = 1.0 / (magnitude + 1e-10)
            prob = prob / (np.sum(prob) + 1e-30)
            mask = rng.rand(m, n) > prob * m * n * 2
            pruned = pruned * mask

            regrowth = rng.randn(m, n) * 0.01 * (1.0 - mask)
            pruned += regrowth

        keep_mask = np.abs(pruned) > np.percentile(
            np.abs(pruned), 100 * (1.0 - keep_fraction)
        )
        kidx = np.argwhere(keep_mask)
        kvals = pruned[keep_mask]

        U, S, Vt = np.linalg.svd(pruned, full_matrices=False)
        r = max(1, min(8, len(S)))

        meta = dict(shape=t.shape, r=r)
        data = (
            _serialize(U[:, :r].astype(np.float32))
            + _serialize(S[:r].astype(np.float32))
            + _serialize(Vt[:r, :].astype(np.float32))
            + _serialize(kidx.astype(np.int16))
            + kvals.astype(np.float16).tobytes()
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
        n_k = len(remaining) // 6
        if n_k > 0:
            kidx = np.frombuffer(remaining[: n_k * 4], dtype=np.int16).reshape(-1, 2)
            kvals = np.frombuffer(
                remaining[n_k * 4 : n_k * 6], dtype=np.float16
            ).astype(np.float64)
            for i in range(min(n_k, len(kidx), len(kvals))):
                ri, ci = int(kidx[i, 0]), int(kidx[i, 1])
                if ri < m and ci < n:
                    recon[ri, ci] = kvals[i]

        return recon.astype(np.float32)
