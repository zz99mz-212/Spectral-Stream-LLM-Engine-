from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class GaussMap:
    """C16. GAUSS-MAP: K = det(dN), Gaussian curvature, angle defect."""

    name = "gauss_map"
    category = "novel_topological"

    def compress(self, tensor: np.ndarray, n_curvature: int = 32) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        m, n = t.shape

        grad_y, grad_x = np.gradient(t)
        grad_xx = np.gradient(grad_x, axis=0)
        grad_yy = np.gradient(grad_y, axis=1)
        grad_xy = np.gradient(grad_x, axis=1)
        grad_yx = np.gradient(grad_y, axis=0)

        K = (grad_xx * grad_yy - grad_xy * grad_yx) / (
            (1 + grad_x**2 + grad_y**2) ** 2 + 1e-30
        )
        angle_defect = 2 * np.pi - (
            np.arctan2(grad_y, grad_x + 1e-30) + np.arctan2(-grad_y, -grad_x + 1e-30)
        )

        K_flat = K.ravel()
        ad_flat = angle_defect.ravel()

        n_k = min(n_curvature, len(K_flat))
        n_ad = min(n_curvature, len(ad_flat))

        k_idx = np.argpartition(np.abs(K_flat), -n_k)[-n_k:]
        ad_idx = np.argpartition(np.abs(ad_flat), -n_ad)[-n_ad:]

        grad_mean = float(np.mean(grad_x)), float(np.mean(grad_y))
        grad_std = float(np.std(grad_x)), float(np.std(grad_y))

        meta = dict(
            shape=t.shape,
            n_k=n_k,
            n_ad=n_ad,
            gm1=grad_mean[0],
            gm2=grad_mean[1],
            gs1=grad_std[0],
            gs2=grad_std[1],
        )
        data = (
            _serialize(k_idx.astype(np.int32))
            + K_flat[k_idx].astype(np.float16).tobytes()
            + _serialize(ad_idx.astype(np.int32))
            + ad_flat[ad_idx].astype(np.float16).tobytes()
        )
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        n_k = metadata["n_k"]
        n_ad = metadata["n_ad"]
        m, n = shape

        k_idx = _deserialize(data[: n_k * 4]).astype(int)
        pos = n_k * 4
        k_vals = np.frombuffer(data[pos : pos + n_k * 2], dtype=np.float16).astype(
            np.float64
        )
        pos += n_k * 2

        ad_idx = _deserialize(data[pos : pos + n_ad * 4]).astype(int)
        pos += n_ad * 4
        ad_vals = np.frombuffer(data[pos : pos + n_ad * 2], dtype=np.float16).astype(
            np.float64
        )

        recon = np.zeros(shape, dtype=np.float64)
        for i, idx in enumerate(k_idx):
            if idx < m * n and i < len(k_vals):
                ri, ci = divmod(idx, n)
                if ri < m and ci < n:
                    recon[ri, ci] = k_vals[i]

        recon = np.cumsum(np.cumsum(recon, axis=0), axis=1)
        recon = (recon - np.mean(recon)) / (np.std(recon) + 1e-30)
        return recon.astype(np.float32)
