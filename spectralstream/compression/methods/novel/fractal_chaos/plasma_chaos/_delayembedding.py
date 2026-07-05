from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class DelayEmbedding:
    """Takens' theorem: s(t) = [w(t), w(t-τ), ..., w(t-(m-1)τ)]."""

    name = "delay_embedding"
    category = "novel_chaos"

    def compress(
        self, tensor: np.ndarray, tau: int = 2, m: int = 3
    ) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        flat = t.ravel()
        n = len(flat)

        tau_opt = max(1, min(tau, n // 4))
        m_opt = max(2, min(m, n // tau_opt))

        valid_len = n - (m_opt - 1) * tau_opt
        embedding = np.zeros((valid_len, m_opt))
        for i in range(m_opt):
            embedding[:, i] = flat[i * tau_opt : i * tau_opt + valid_len]

        U, S, Vt = np.linalg.svd(embedding, full_matrices=False)
        cum = np.cumsum(S) / np.sum(S)
        r = max(1, int(np.searchsorted(cum, 0.92)) + 1)
        r = min(r, len(S))

        meta = dict(
            shape=tensor.shape, tau=tau_opt, m=m_opt, rank=r, valid_len=valid_len
        )
        data = _serialize(U[:, :r].astype(np.float32))
        data += _serialize(S[:r].astype(np.float32))
        data += _serialize(Vt[:r, :].astype(np.float32))
        data += _serialize(np.array([flat[-(m_opt - 1) * tau_opt :]], dtype=np.float32))
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        tau = metadata["tau"]
        m = metadata["m"]
        rank = metadata["rank"]
        valid_len = metadata["valid_len"]
        n = int(np.prod(shape))

        pos = 0
        U = _deserialize(data[: valid_len * rank * 4]).reshape(valid_len, rank)
        pos += valid_len * rank * 4
        S = _deserialize(data[pos : pos + rank * 4])
        pos += rank * 4
        Vt = _deserialize(data[pos : pos + rank * m * 4]).reshape(rank, m)
        pos += rank * m * 4

        recon_embed = (U * S) @ Vt
        recon = np.zeros(n, dtype=np.float64)
        counts = np.zeros(n, dtype=np.int32)
        for i in range(m):
            start = i * tau
            end = start + valid_len
            if end <= n:
                recon[start:end] += recon_embed[:, i]
                counts[start:end] += 1
        counts = np.maximum(counts, 1)
        recon /= counts

        tail = _deserialize(data[pos:])
        if len(tail) > 0:
            recon[-len(tail) :] = tail

        return recon.reshape(shape).astype(np.float32)
