from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class EntrainmentAnalysis:
    """Arnold tongue: φ̇ = ω - ω_ext + K sin(φ), store entrainment plateaus."""

    name = "entrainment_analysis"
    category = "novel_chaos"

    def compress(self, tensor: np.ndarray, n_freqs: int = 16) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        flat = t.ravel()
        n = len(flat)

        omega = float(np.mean(np.abs(np.fft.fft(flat)[: n // 2])))
        omega_ext_vals = np.linspace(0.5 * omega, 1.5 * omega, n_freqs)
        K = float(np.std(flat))

        entrainment = []
        for omega_ext in omega_ext_vals:
            phi = 0.0
            for _ in range(500):
                phi_dot = omega - omega_ext + K * np.sin(phi)
                phi += phi_dot * 0.01
            plateau = []
            for _ in range(100):
                phi_dot = omega - omega_ext + K * np.sin(phi)
                phi += phi_dot * 0.01
                plateau.append(phi_dot)
            entrainment.append(float(np.std(plateau)))
        entrainment = np.array(entrainment)
        plateau_regions = entrainment < np.percentile(entrainment, 30)

        n_pts = min(n, n_freqs)
        idx = np.linspace(0, n - 1, n_pts).astype(int)
        vals = flat[idx]
        residual = flat - np.interp(np.arange(n), idx, vals)
        thr = np.percentile(np.abs(residual), 92)
        rmask = np.abs(residual) > thr
        ridx = np.argwhere(rmask)[:, 0]
        rvals = residual[rmask]

        meta = dict(
            shape=tensor.shape,
            omega=omega,
            K=K,
            n_freqs=n_freqs,
            n_pts=n_pts,
            n_res=len(ridx),
        )
        data = struct.pack("<ddii", omega, K, n_freqs, n_pts)
        data += _serialize(omega_ext_vals.astype(np.float32))
        data += _serialize(entrainment.astype(np.float32))
        data += _serialize(plateau_regions.astype(np.int8))
        data += _serialize(idx.astype(np.int32)) + vals.astype(np.float16).tobytes()
        if len(ridx) > 0:
            data += (
                _serialize(ridx.astype(np.int32)) + rvals.astype(np.float16).tobytes()
            )
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        n_freqs = metadata["n_freqs"]
        n_pts = metadata["n_pts"]
        n_res = metadata.get("n_res", 0)
        n = int(np.prod(shape))

        pos = 20
        pos += n_freqs * 4
        pos += n_freqs * 4
        pos += n_freqs * 1

        idx = _deserialize(data[pos : pos + n_pts * 4]).astype(int)
        pos += n_pts * 4
        vals = np.frombuffer(data[pos : pos + n_pts * 2], dtype=np.float16).astype(
            np.float64
        )
        pos += n_pts * 2

        recon = np.interp(np.arange(n), idx, vals)

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
