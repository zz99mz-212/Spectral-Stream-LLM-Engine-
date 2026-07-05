from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class QuantumBootstrap:
    name = "quantum_bootstrap"
    category = "tensor_quantum"

    def compress(
        self, tensor: np.ndarray, n_ref: int = 4, n_ranks: int = 4
    ) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        orig_shape = t.shape
        if t.ndim == 2:
            m, n = t.shape
            u, s, vt = np.linalg.svd(t, full_matrices=False)
            r = min(n_ref, len(s))
            residual = t - (u[:, :r] * s[:r]) @ vt[:r, :]
            n_updates = min(n_ranks, m, n)
            alphas = np.zeros(n_updates, dtype=np.float64)
            psi_list = []
            for i in range(n_updates):
                uu, ss, vvt = np.linalg.svd(residual.reshape(m, n), full_matrices=False)
                if len(ss) < 1 or ss[0] < 1e-15:
                    n_updates = i
                    break
                alphas[i] = ss[0]
                psi_list.append(uu[:, 0].copy())
                psi_list.append(vvt[0, :].copy())
                residual -= ss[0] * np.outer(uu[:, 0], vvt[0, :])
            meta = dict(shape=orig_shape, n_ref=r, n_updates=n_updates)
            data = struct.pack("<ii", r, n_updates)
            data += _serialize(u[:, :r].astype(np.float32))
            data += _serialize(s[:r].astype(np.float32))
            data += _serialize(vt[:r, :].astype(np.float32))
            data += _serialize(alphas[:n_updates].astype(np.float32))
            for p in psi_list:
                data += _serialize(p.astype(np.float32))
            return data, meta
        meta = dict(shape=orig_shape, n_ref=0, n_updates=0)
        data = struct.pack("<ii", 0, 0) + _serialize(t.ravel().astype(np.float32))
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        r, n_updates = struct.unpack_from("<ii", data, 0)
        pos = 8
        if r == 0:
            flat = _deserialize(data[pos:])
            return flat[: int(np.prod(shape))].reshape(shape).astype(np.float32)
        m, n = shape
        u = _deserialize(data[pos : pos + m * r * 4]).reshape(m, r)
        pos += m * r * 4
        s = _deserialize(data[pos : pos + r * 4])
        pos += r * 4
        vt = _deserialize(data[pos : pos + r * n * 4]).reshape(r, n)
        pos += r * n * 4
        recon = (u * s) @ vt
        if n_updates > 0:
            alphas = _deserialize(data[pos : pos + n_updates * 4])
            pos += n_updates * 4
            for i in range(n_updates):
                psi = _deserialize(data[pos : pos + m * 4])
                pos += m * 4
                phi = _deserialize(data[pos : pos + n * 4])
                pos += n * 4
                recon += alphas[i] * np.outer(psi, phi)
        return recon.astype(np.float32)
