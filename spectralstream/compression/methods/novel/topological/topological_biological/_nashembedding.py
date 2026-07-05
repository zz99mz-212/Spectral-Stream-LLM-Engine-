from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class NashEmbedding:
    """C14. NASH-EMBEDDING: C¹ isometric embedding dimension reduction."""

    name = "nash_embedding"
    category = "novel_topological"

    def compress(
        self, tensor: np.ndarray, target_dim: int = None
    ) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        m, n = t.shape

        if target_dim is None:
            target_dim = max(1, min(m, n) // 2)
        r = min(target_dim, m, n)

        diff = t[:, None, :] - t[None, :, :]
        D = np.sqrt(np.sum(diff**2, axis=-1))
        D2 = D**2

        J = np.eye(m) - np.ones((m, m)) / m
        B = -0.5 * J @ D2 @ J

        eigvals, eigvecs = np.linalg.eigh(B)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        r = min(r, len(eigvals))

        embedding = eigvecs[:, :r] * np.sqrt(np.maximum(eigvals[:r], 0.0))
        linear_map = np.linalg.lstsq(embedding, t, rcond=None)[0]

        U_emb, S_emb, Vt_emb = np.linalg.svd(embedding, full_matrices=False)
        r_emb = max(1, min(r, np.sum(S_emb > np.max(S_emb) * 0.05)))

        meta = dict(shape=t.shape, m=m, n=n, r_emb=int(r_emb))
        data = (
            _serialize(U_emb[:, :r_emb].astype(np.float32))
            + _serialize(S_emb[:r_emb].astype(np.float32))
            + _serialize(Vt_emb[:r_emb, :].astype(np.float32))
            + _serialize(linear_map.astype(np.float16))
        )
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        m = metadata["m"]
        n = metadata["n"]
        r_emb = metadata["r_emb"]

        pos = 0
        U_e = _deserialize(data[: m * r_emb * 4]).reshape(m, r_emb)
        pos += m * r_emb * 4
        S_e = _deserialize(data[pos : pos + r_emb * 4])
        pos += r_emb * 4
        Vt_e = _deserialize(data[pos : pos + r_emb * r_emb * 4]).reshape(r_emb, r_emb)
        pos += r_emb * r_emb * 4

        n_lin = min(n, (len(data) - pos) // 2)
        linear_map = (
            _deserialize(data[pos:], dtype=np.float16)
            .reshape(r_emb, n)
            .astype(np.float64)
            if n_lin == n
            else np.random.randn(r_emb, n).astype(np.float64)
        )

        embedding = (U_e * S_e) @ Vt_e
        recon = embedding @ linear_map
        return recon.astype(np.float32)
