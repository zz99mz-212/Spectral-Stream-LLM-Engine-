"""Auto-generated from inr_compression.py."""

from __future__ import annotations

import math
import struct
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spectralstream.core.math_primitives import dct, idct, next_power_of_two


def _bytes(obj: Any) -> int:
    if isinstance(obj, np.ndarray):
        return obj.nbytes
    if isinstance(obj, dict):
        return sum(_bytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(_bytes(x) for x in obj)
    return 0


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()


def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()


class KolmogorovComplexity:
    """MDL-based model selection — choose best among zero/mean/SVD/DCT auto-encoder."""

    name = "kolmogorov_complexity"
    category = "functional"

    def compress(self, tensor: np.ndarray, **params) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        flat = t.ravel()
        n = len(flat)
        methods = {
            "zero": (lambda: np.zeros(n), 0),
            "mean": (lambda: np.full(n, np.mean(flat)), 1),
            "svd": (lambda: self._svd_recon(t), 2),
            "dct": (lambda: self._dct_recon(t), 3),
        }
        if t.ndim < 2:
            t = t.reshape(1, -1)
        m, n_2d = t.shape
        best_mdl = float("inf")
        best_name = "mean"
        best_data = None
        for name, (fn, mid) in methods.items():
            try:
                if name == "svd":
                    U, S, Vt = np.linalg.svd(t, full_matrices=False)
                    cum = np.cumsum(S) / np.sum(S)
                    r = int(np.searchsorted(cum, 0.9)) + 1
                    params_count = r * (m + n_2d)
                    mdl = params_count * np.log2(n) + r * 32
                elif name == "dct":
                    c = dct(t)
                    k = max(1, int(n * 0.1))
                    params_count = k
                    mdl = k * np.log2(n) + k * 32
                elif name == "mean":
                    mdl = 32
                else:
                    mdl = 1
                if mdl < best_mdl:
                    best_mdl = mdl
                    best_name = name
            except (ValueError, TypeError, RuntimeError):
                pass
        if best_name == "svd":
            U, S, Vt = np.linalg.svd(t, full_matrices=False)
            cum = np.cumsum(S) / np.sum(S)
            r = int(np.searchsorted(cum, 0.9)) + 1
            data = struct.pack("<i", 2) + _serialize(U[:, :r].astype(np.float32))
            data += _serialize(S[:r].astype(np.float32))
            data += _serialize(Vt[:r, :].astype(np.float32))
        elif best_name == "dct":
            c = dct(t)
            k = max(1, int(n * 0.1))
            flat_c = c.ravel()
            idx = np.argpartition(np.abs(flat_c), -k)[-k:]
            data = struct.pack("<i", 3) + _serialize(idx.astype(np.int32))
            data += _serialize(flat_c[idx].astype(np.float32))
        elif best_name == "mean":
            data = struct.pack("<if", 1, float(np.mean(flat)))
        else:
            data = struct.pack("<i", 0)
        meta = dict(model=best_name, mdl=best_mdl, shape=tensor.shape, n=n, m=m)
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        n = metadata["n"]
        model_id = struct.unpack_from("<i", data, 0)[0]
        if model_id == 0:
            return np.zeros(shape, dtype=np.float32)
        if model_id == 1:
            mean_v = struct.unpack_from("<f", data, 4)[0]
            return np.full(shape, mean_v, dtype=np.float32)
        if model_id == 2:
            pos = 4
            r = (
                (len(data) - pos) // (4 * (shape[0] + shape[1] + 1))
                if shape[0] > 0
                else 1
            )
            U = _deserialize(data[pos : pos + shape[0] * r * 4]).reshape(shape[0], r)
            pos += shape[0] * r * 4
            S = _deserialize(data[pos : pos + r * 4])
            pos += r * 4
            Vt = _deserialize(data[pos : pos + r * shape[1] * 4]).reshape(r, shape[1])
            return ((U * S) @ Vt).reshape(shape).astype(np.float32)
        if model_id == 3:
            pos = 4
            n_elem = (len(data) - 4) // 8
            k = n_elem // 2
            idx = np.frombuffer(data[pos : pos + k * 4], dtype=np.int32).copy()
            pos += k * 4
            vals = _deserialize(data[pos : pos + k * 4])
            flat = np.zeros(n, dtype=np.float64)
            for ii, vv in zip(idx, vals):
                if ii < n:
                    flat[ii] = vv
            recon = idct(flat.reshape(shape) if len(shape) >= 2 else flat)
            return recon.reshape(shape).astype(np.float32)
        return np.zeros(shape, dtype=np.float32)

    def _svd_recon(self, t):
        U, S, Vt = np.linalg.svd(t, full_matrices=False)
        cum = np.cumsum(S) / np.sum(S)
        r = int(np.searchsorted(cum, 0.9)) + 1
        return (U[:, :r] * S[:r]) @ Vt[:r, :]

    def _dct_recon(self, t):
        c = dct(t)
        k = max(1, int(t.size * 0.1))
        flat_c = c.ravel()
        idx = np.argpartition(np.abs(flat_c), -k)[-k:]
        r = np.zeros_like(flat_c)
        r[idx] = flat_c[idx]
        return idct(r.reshape(c.shape))
