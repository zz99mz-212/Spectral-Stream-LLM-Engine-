from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class PredictiveCoding:
    """D6. PREDICTIVE-CODING: ε_i = x_i - f(W_i x_{i-1}), sparse errors."""

    name = "predictive_coding_hierarchy"
    category = "novel_biological"

    def compress(
        self, tensor: np.ndarray, n_layers: int = 3, keep_fraction: float = 0.15
    ) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        m, n = t.shape

        layers = []
        errors = []
        current = t.copy()
        for _ in range(n_layers):
            layer_size = max(1, current.shape[0] // 2)
            U, S, Vt = np.linalg.svd(current, full_matrices=False)
            r = min(layer_size, len(S))
            pred = (U[:, :r] * S[:r]) @ Vt[:r, :]
            err = current - pred
            layers.append(pred)
            errors.append(err)
            current = pred.copy()

        top = layers[-1].copy()
        all_errors = []
        for err in errors:
            e_flat = err.ravel()
            n_e = max(1, int(keep_fraction * len(e_flat)))
            idx = np.argpartition(np.abs(e_flat), -n_e)[-n_e:]
            all_errors.append((idx, e_flat[idx]))

        U_t, S_t, Vt_t = np.linalg.svd(top, full_matrices=False)
        r_t = max(1, min(8, len(S_t)))

        meta = dict(shape=t.shape, n_layers=n_layers, r_t=r_t)
        err_sizes = [len(e[0]) for e in all_errors]
        meta["err_sizes"] = err_sizes

        data = (
            _serialize(U_t[:, :r_t].astype(np.float32))
            + _serialize(S_t[:r_t].astype(np.float32))
            + _serialize(Vt_t[:r_t, :].astype(np.float32))
        )
        for idx, vals in all_errors:
            data += _serialize(idx.astype(np.int32)) + vals.astype(np.float16).tobytes()
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        n_layers = metadata["n_layers"]
        r_t = metadata["r_t"]
        err_sizes = metadata["err_sizes"]
        m, n = shape

        pos = 0
        U_t = _deserialize(data[: m * r_t * 4]).reshape(m, r_t)
        pos += m * r_t * 4
        S_t = _deserialize(data[pos : pos + r_t * 4])
        pos += r_t * 4
        Vt_t = _deserialize(data[pos : pos + r_t * n * 4]).reshape(r_t, n)
        pos += r_t * n * 4

        top = (U_t * S_t) @ Vt_t
        recon = top.copy()

        for li in range(n_layers - 1, -1, -1):
            n_e = err_sizes[li]
            idx = _deserialize(data[pos : pos + n_e * 4]).astype(int)
            pos += n_e * 4
            vals = np.frombuffer(data[pos : pos + n_e * 2], dtype=np.float16).astype(
                np.float64
            )
            pos += n_e * 2

            err = np.zeros(m * n, dtype=np.float64)
            for i in range(min(n_e, len(idx), len(vals))):
                if idx[i] < m * n:
                    err[idx[i]] = vals[i]
            recon = recon + err.reshape(m, n)

        return recon.astype(np.float32)
