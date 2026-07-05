from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class AstrocyteModulation:
    """D15. ASTROCYTE-MODULATION: d[Ca²⁺]/dt = J_release - J_uptake + J_influx."""

    name = "astrocyte_modulation"
    category = "novel_biological"

    def compress(self, tensor: np.ndarray, n_groups: int = 8) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        m, n = t.shape
        k = min(n_groups, m, n)

        group_size = max(1, m // k)
        group_means = np.zeros(k, dtype=np.float64)
        group_stds = np.zeros(k, dtype=np.float64)
        for i in range(k):
            si = i * group_size
            ei = min((i + 1) * group_size, m)
            group = t[si:ei, :]
            group_means[i] = float(np.mean(group))
            group_stds[i] = float(np.std(group))

        j_release = np.mean(np.maximum(t, 0))
        j_uptake = np.mean(np.minimum(t, 0))
        j_influx = np.mean(np.abs(t)) * 0.1

        U, S, Vt = np.linalg.svd(t, full_matrices=False)
        r = max(1, min(8, len(S)))

        meta = dict(shape=t.shape, r=r, k=k)
        data = (
            _serialize(U[:, :r].astype(np.float32))
            + _serialize(S[:r].astype(np.float32))
            + _serialize(Vt[:r, :].astype(np.float32))
            + _serialize(group_means.astype(np.float32))
            + _serialize(group_stds.astype(np.float32))
        )
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        r = metadata["r"]
        k = metadata["k"]
        m, n = shape

        pos = 0
        U_r = _deserialize(data[: m * r * 4]).reshape(m, r)
        pos += m * r * 4
        S_r = _deserialize(data[pos : pos + r * 4])
        pos += r * 4
        Vt_r = _deserialize(data[pos : pos + r * n * 4]).reshape(r, n)
        pos += r * n * 4

        group_means = _deserialize(data[pos : pos + k * 4])
        pos += k * 4
        group_stds = _deserialize(data[pos : pos + k * 4])

        recon = (U_r * S_r) @ Vt_r

        group_size = max(1, m // k)
        for i in range(k):
            si = i * group_size
            ei = min((i + 1) * group_size, m)
            for row in range(si, ei):
                orig_mean = float(np.mean(recon[row, :]))
                orig_std = float(np.std(recon[row, :]))
                if orig_std > 1e-10:
                    recon[row, :] = (recon[row, :] - orig_mean) / orig_std * group_stds[
                        i
                    ] + group_means[i]
                else:
                    recon[row, :] = group_means[i]

        return recon.astype(np.float32)
