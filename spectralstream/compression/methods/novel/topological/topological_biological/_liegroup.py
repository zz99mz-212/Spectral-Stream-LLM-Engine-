from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class LieGroup:
    """C4. LIE-GROUP: W = g_1 g_2 ... g_k with Lie algebra coordinates."""

    name = "lie_group"
    category = "novel_topological"

    def compress(self, tensor: np.ndarray, n_factors: int = 4) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        m, n = t.shape
        k = min(n_factors, m, n)

        factors = []
        residual = t.copy()
        for i in range(k):
            U, S, Vt = np.linalg.svd(residual, full_matrices=False)
            if S[0] < 1e-10:
                factors.append(np.zeros((1, 1), dtype=np.float64))
                continue
            u = U[:, 0:1]
            vt = Vt[0:1, :]
            s = S[0]
            g = np.outer(u.ravel(), vt.ravel()) * s
            factors.append(g)
            residual -= g

        rows = []
        for g in factors:
            if g.size > 0:
                rows.append(g.ravel())
        all_factors = np.concatenate(rows) if rows else np.zeros(1)

        meta = dict(shape=t.shape, n_factors=k, sizes=[g.size for g in factors])
        data = _serialize(all_factors.astype(np.float32))
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        n_factors = metadata["n_factors"]
        sizes = metadata["sizes"]
        flat = _deserialize(data)

        recon = np.zeros(shape, dtype=np.float64)
        pos = 0
        for i in range(n_factors):
            sz = sizes[i]
            if sz == 0 or pos + sz > len(flat):
                continue
            g = flat[pos : pos + sz]
            pos += sz
            if sz >= int(np.sqrt(sz)) ** 2:
                side = int(math.isqrt(sz))
                if side * side == sz:
                    gm = g.reshape(side, side)
                    if gm.shape[0] <= shape[0] and gm.shape[1] <= shape[1]:
                        recon[: gm.shape[0], : gm.shape[1]] += gm
                    else:
                        recon += np.resize(gm, shape)
                else:
                    recon += np.resize(g, shape)

        return recon.astype(np.float32)
