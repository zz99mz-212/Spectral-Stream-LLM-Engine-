from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class SpectralSequence:
    """C18. SPECTRAL-SEQUENCE: Leray-Serre E^{p,q}_2 -> E_∞ page only."""

    name = "spectral_sequence"
    category = "novel_topological"

    def compress(
        self, tensor: np.ndarray, n_pages: int = 4, n_modes: int = 8
    ) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        m, n = t.shape

        U, S, Vt = np.linalg.svd(t, full_matrices=False)
        r = min(n_modes, len(S))

        pages = []
        S_page = S[:r].copy()
        for _ in range(n_pages):
            S_page = np.sqrt(np.maximum(S_page, 0.0))
            pages.append(S_page.copy())

        e_infinity = pages[-1]
        diff_maps = []
        for i in range(len(pages) - 1):
            diff_maps.append(pages[i + 1] - pages[i])

        meta = dict(shape=t.shape, r=r, n_pages=n_pages)
        data = (
            _serialize(U[:, :r].astype(np.float32))
            + _serialize(e_infinity.astype(np.float32))
            + _serialize(Vt[:r, :].astype(np.float32))
        )
        if diff_maps:
            data += _serialize(np.array(diff_maps, dtype=np.float32))
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        r = metadata["r"]
        m, n = shape

        pos = 0
        U_r = _deserialize(data[: m * r * 4]).reshape(m, r)
        pos += m * r * 4
        S_inf = _deserialize(data[pos : pos + r * 4])
        pos += r * 4
        Vt_r = _deserialize(data[pos : pos + r * n * 4]).reshape(r, n)

        S_recon = S_inf ** (2 ** metadata["n_pages"])
        return ((U_r * S_recon) @ Vt_r).astype(np.float32)
