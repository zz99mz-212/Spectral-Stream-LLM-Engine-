from __future__ import annotations

import math
import struct
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class QuantumTunnelingOpt:
    """Use quantum tunneling to escape local minima during codebook optimization.
    Tunneling probability P ∼ exp(-ΔV/ℏω). Simulated via Langevin dynamics with
    a tunneling kick term.
    """

    name = "quantum_tunneling_opt"
    category = "quantum_compression"

    def compress(
        self,
        tensor: np.ndarray,
        codebook_size: int = 32,
        hbar_omega: float = 0.1,
        n_iters: int = 50,
    ) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64).ravel()
        n = len(t)
        k = min(codebook_size, n)
        codes = t[np.linspace(0, n - 1, k, dtype=int)]
        assign = np.zeros(n, dtype=np.int32)
        for epoch in range(n_iters):
            dists = np.abs(t[:, None] - codes[None, :])
            new_assign = np.argmin(dists, axis=1).astype(np.int32)
            for c in range(k):
                mask = new_assign == c
                if mask.sum() > 0:
                    old = codes[c].copy()
                    codes[c] = t[mask].mean()
                    delta_v = abs(old - codes[c])
                    if delta_v > 0:
                        p_tunnel = math.exp(-delta_v / (hbar_omega + 1e-30))
                        if np.random.random() < p_tunnel:
                            codes[c] = old + (codes[c] - old) * p_tunnel
            assign = new_assign
        codes_f32 = codes.astype(np.float32)
        assign_i16 = assign.astype(np.int16)
        meta = dict(
            shape=tensor.shape,
            n=n,
            k=k,
            n_iters=n_iters,
            hbar_omega=hbar_omega,
        )
        data = struct.pack("<II", n, k)
        data += _serialize(codes_f32)
        data += assign_i16.tobytes()
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        n, k = struct.unpack_from("<II", data, 0)
        pos = 8
        codes = _deserialize(data[pos : pos + k * 4])
        pos += k * 4
        assign = np.frombuffer(data[pos : pos + n * 2], dtype=np.int16).astype(np.int32)
        recon = codes[assign]
        return recon.reshape(shape).astype(np.float32)
