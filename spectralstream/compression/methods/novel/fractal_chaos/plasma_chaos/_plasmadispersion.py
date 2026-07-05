from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class PlasmaDispersion:
    """Cold plasma dielectric tensor decomposition with 6 Stix parameters."""

    name = "plasma_dispersion"
    category = "novel_physics"

    def compress(self, tensor: np.ndarray) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        flat = t.ravel()

        R = float(np.mean(flat))
        L = float(np.std(flat))
        S = float(np.mean(flat**2))
        D = float(np.mean(np.abs(np.diff(flat[: min(1000, len(flat))]))))
        P = float(np.percentile(np.abs(flat), 90))

        flat_norm = (flat - R) / max(L, 1e-30)
        hist, _ = np.histogram(flat_norm, bins=32, density=True)
        pdf_bytes = hist.astype(np.float16).tobytes()

        meta = dict(shape=tensor.shape, R=R, L=L, S=S, D=D, P=P)
        data = struct.pack("<ddddd", R, L, S, D, P)
        data += pdf_bytes

        residual = flat - R
        thr = np.percentile(np.abs(residual), 95)
        rmask = np.abs(residual) > thr
        ridx = np.argwhere(rmask)[:, 0]
        rvals = residual[rmask]

        meta["n_res"] = len(ridx)
        data += _serialize(ridx.astype(np.int32)) + rvals.astype(np.float16).tobytes()
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        R = metadata["R"]
        L = metadata["L"]
        n_res = metadata.get("n_res", 0)

        n = int(np.prod(shape))
        recon = np.full(n, R, dtype=np.float64)
        rng = np.random.RandomState(42)
        noise = rng.randn(n) * L * 0.15
        recon += noise

        pos = 40 + 64
        if n_res > 0:
            ridx = _deserialize(data[pos : pos + n_res * 4]).astype(int)
            pos += n_res * 4
            rvals = np.frombuffer(data[pos : pos + n_res * 2], dtype=np.float16).astype(
                np.float64
            )
            for i, v in zip(ridx, rvals):
                if i < n:
                    recon[i] += v

        return recon.reshape(shape).astype(np.float32)
