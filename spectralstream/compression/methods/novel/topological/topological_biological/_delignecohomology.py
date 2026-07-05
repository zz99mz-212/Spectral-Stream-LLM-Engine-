from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class DeligneCohomology:
    """C15. DELIGNE-COHOMOLOGY: mixed Hodge structure H^k = ⊕ H^{p,q} ⊕ W_i."""

    name = "deligne_cohomology"
    category = "novel_topological"

    def compress(self, tensor: np.ndarray, hodge_rank: int = 8) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        m, n = t.shape
        k = min(hodge_rank, m, n)

        U, S, Vt = np.linalg.svd(t, full_matrices=False)

        period_matrix = U[:, :k].T @ t @ Vt[:k, :].T

        weight_filtration = np.sort(S)[::-1][:k]
        hodge_numbers = np.zeros(k, dtype=np.float64)
        for i in range(k):
            hodge_numbers[i] = weight_filtration[i] / (
                np.sum(weight_filtration) + 1e-30
            )

        meta = dict(shape=t.shape, k=k)
        data = (
            _serialize(period_matrix.astype(np.float32))
            + _serialize(weight_filtration.astype(np.float32))
            + _serialize(hodge_numbers.astype(np.float32))
        )
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        k = metadata["k"]
        m, n = shape

        period_matrix = _deserialize(data[: k * k * 4]).reshape(k, k)
        pos = k * k * 4
        weight_filt = _deserialize(data[pos : pos + k * 4])
        pos += k * 4
        hodge_numbers = _deserialize(data[pos : pos + k * 4])

        U_rand = np.random.RandomState(0).randn(m, k) * 0.1
        V_rand = np.random.RandomState(1).randn(k, n) * 0.1
        U_orth, _ = np.linalg.qr(U_rand)
        V_orth, _ = np.linalg.qr(V_rand.T)
        V_orth = V_orth.T

        S_hodge = weight_filt * (1.0 + 0.1 * hodge_numbers)
        recon = U_orth[:, :k] @ np.diag(S_hodge) @ V_orth[:k, :]
        recon += U_orth[:, :k] @ period_matrix @ V_orth[:k, :] * 0.01
        return recon.astype(np.float32)
