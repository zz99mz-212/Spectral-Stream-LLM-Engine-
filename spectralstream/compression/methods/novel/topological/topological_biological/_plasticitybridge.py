from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class PlasticityBridge:
    """D16. PLASTICITY-BRIDGE: PRP templates + tag locations, PRPs shared."""

    name = "plasticity_bridge"
    category = "novel_biological"

    def compress(self, tensor: np.ndarray, n_templates: int = 8) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        m, n = t.shape
        k = min(n_templates, m, n)

        t_size = max(1, m // k)
        templates = np.zeros((k, n), dtype=np.float64)
        for i in range(k):
            si = i * t_size
            ei = min((i + 1) * t_size, m)
            if si < m:
                templates[i, :] = np.mean(t[si:ei, :], axis=0)

        tag_locations = np.zeros((m, n), dtype=np.int32)
        for i in range(k):
            si = i * t_size
            ei = min((i + 1) * t_size, m)
            for row in range(si, ei):
                similarity = np.corrcoef(t[row, :], templates)[0, 1]
                tag_locations[row, :] = np.abs(similarity) > 0.3

        U_t, S_t, Vt_t = np.linalg.svd(templates, full_matrices=False)
        r_t = max(1, min(4, len(S_t)))

        meta = dict(shape=t.shape, k=k, r_t=r_t)
        data = (
            _serialize(U_t[:, :r_t].astype(np.float32))
            + _serialize(S_t[:r_t].astype(np.float32))
            + _serialize(Vt_t[:r_t, :].astype(np.float32))
            + tag_locations.astype(np.uint8).tobytes()
        )
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        k = metadata["k"]
        r_t = metadata["r_t"]
        m, n = shape

        pos = 0
        U_t = _deserialize(data[: k * r_t * 4]).reshape(k, r_t)
        pos += k * r_t * 4
        S_t = _deserialize(data[pos : pos + r_t * 4])
        pos += r_t * 4
        Vt_t = _deserialize(data[pos : pos + r_t * n * 4]).reshape(r_t, n)
        pos += r_t * n * 4

        templates = (U_t * S_t) @ Vt_t
        tag_bytes = n * m
        tag_locations = (
            np.frombuffer(data[pos : pos + tag_bytes], dtype=np.uint8)
            .reshape(m, n)
            .astype(bool)
        )

        recon = np.zeros((m, n), dtype=np.float64)
        t_size = max(1, m // k)
        for i in range(k):
            si = i * t_size
            ei = min((i + 1) * t_size, m)
            for row in range(si, ei):
                template_idx = i if i < len(templates) else 0
                recon[row, :] = templates[min(template_idx, len(templates) - 1), :]

        recon[tag_locations] *= 1.1
        return recon.astype(np.float32)
