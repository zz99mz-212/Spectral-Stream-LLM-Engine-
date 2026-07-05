from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class DendriticComputation:
    """D9. DENDTRITIC-COMPUTATION: y = Σ_b σ(Σ_i w_{bi} x_i) with branches."""

    name = "dendritic_computation"
    category = "novel_biological"

    def compress(self, tensor: np.ndarray, n_branches: int = 8) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        m, n = t.shape
        k = min(n_branches, m, n)

        branch_size = max(1, m // k)
        branches = []
        for b in range(k):
            si = b * branch_size
            ei = min((b + 1) * branch_size, m)
            branch_weights = t[si:ei, :]
            U, S, Vt = np.linalg.svd(branch_weights, full_matrices=False)
            r = max(1, min(4, len(S)))
            branches.append((U[:, :r], S[:r], Vt[:r, :]))

        combined = np.zeros((k, n), dtype=np.float64)
        for b, (U_b, S_b, Vt_b) in enumerate(branches):
            combined[b, :] = np.sum(np.tanh((U_b * S_b) @ Vt_b), axis=0)

        U_c, S_c, Vt_c = np.linalg.svd(combined, full_matrices=False)
        r_c = max(1, min(4, len(S_c)))

        meta = dict(shape=t.shape, n_branches=k, r_c=r_c)
        data = (
            _serialize(U_c[:, :r_c].astype(np.float32))
            + _serialize(S_c[:r_c].astype(np.float32))
            + _serialize(Vt_c[:r_c, :].astype(np.float32))
        )
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        n_branches = metadata["n_branches"]
        r_c = metadata["r_c"]
        m, n = shape

        pos = 0
        U_c = _deserialize(data[: n_branches * r_c * 4]).reshape(n_branches, r_c)
        pos += n_branches * r_c * 4
        S_c = _deserialize(data[pos : pos + r_c * 4])
        pos += r_c * 4
        Vt_c = _deserialize(data[pos : pos + r_c * n * 4]).reshape(r_c, n)

        combined = (U_c * S_c) @ Vt_c
        recon = np.zeros((m, n), dtype=np.float64)
        branch_size = max(1, m // n_branches)
        for b in range(min(n_branches, m)):
            si = b * branch_size
            ei = min((b + 1) * branch_size, m)
            active = np.tanh(combined[b : b + 1, :])
            for row in range(si, ei):
                recon[row, :] = active[0, :]

        return recon.astype(np.float32)
