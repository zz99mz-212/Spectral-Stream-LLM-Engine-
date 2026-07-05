from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class HodgeDecomposition:
    """C8. HODGE-DECOMPOSITION: W = ∇f + ∇×A + h (Helmholtz-Hodge)."""

    name = "hodge_decomposition"
    category = "novel_topological"

    def compress(
        self, tensor: np.ndarray, keep_fraction: float = 0.15
    ) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        m, n = t.shape

        grad_y, grad_x = np.gradient(t)
        f_curl_y, f_curl_x = np.gradient(grad_x, axis=0), np.gradient(-grad_y, axis=1)
        laplacian = np.gradient(grad_x, axis=0) + np.gradient(grad_y, axis=1)

        sigma = min(m, n) * 0.05
        sz = max(3, int(4 * sigma + 1))
        if sz % 2 == 0:
            sz += 1
        gk = np.exp(-((np.arange(sz) - sz // 2) ** 2) / (2 * sigma**2))
        gk /= np.sum(gk)
        harmonic = t.copy()
        for _ in range(3):
            harmonic = np.apply_along_axis(
                lambda x: np.convolve(x, gk, mode="same"), 0, harmonic
            )
            harmonic = np.apply_along_axis(
                lambda x: np.convolve(x, gk, mode="same"), 1, harmonic
            )

        curl_part = np.zeros_like(t)
        curl_part[1:-1, 1:-1] = f_curl_x[1:-1, 1:-1] + f_curl_y[1:-1, 1:-1]

        U_g, S_g, Vt_g = np.linalg.svd(grad_x, full_matrices=False)
        r_g = max(1, int(keep_fraction * len(S_g)))
        U_c, S_c, Vt_c = np.linalg.svd(curl_part, full_matrices=False)
        r_c = max(1, int(keep_fraction * len(S_c)))

        meta = dict(shape=t.shape, r_g=r_g, r_c=r_c)
        data = (
            _serialize(U_g[:, :r_g].astype(np.float32))
            + _serialize(S_g[:r_g].astype(np.float32))
            + _serialize(Vt_g[:r_g, :].astype(np.float32))
            + _serialize(U_c[:, :r_c].astype(np.float32))
            + _serialize(S_c[:r_c].astype(np.float32))
            + _serialize(Vt_c[:r_c, :].astype(np.float32))
            + harmonic.astype(np.float16).tobytes()
        )
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        r_g = metadata["r_g"]
        r_c = metadata["r_c"]
        m, n = shape

        pos = 0
        U_g = _deserialize(data[: m * r_g * 4]).reshape(m, r_g)
        pos += m * r_g * 4
        S_g = _deserialize(data[pos : pos + r_g * 4])
        pos += r_g * 4
        Vt_g = _deserialize(data[pos : pos + r_g * n * 4]).reshape(r_g, n)
        pos += r_g * n * 4

        U_c = _deserialize(data[pos : pos + m * r_c * 4]).reshape(m, r_c)
        pos += m * r_c * 4
        S_c = _deserialize(data[pos : pos + r_c * 4])
        pos += r_c * 4
        Vt_c = _deserialize(data[pos : pos + r_c * n * 4]).reshape(r_c, n)
        pos += r_c * n * 4

        n_harm = m * n
        harmonic = (
            np.frombuffer(data[pos : pos + n_harm * 2], dtype=np.float16)
            .reshape(shape)
            .astype(np.float64)
        )

        grad_recon = (U_g * S_g) @ Vt_g
        curl_recon = (U_c * S_c) @ Vt_c
        recon = grad_recon + curl_recon + harmonic
        return recon.astype(np.float32)
