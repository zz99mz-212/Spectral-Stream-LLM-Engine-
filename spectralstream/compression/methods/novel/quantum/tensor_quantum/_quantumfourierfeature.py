from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()


def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()


class QuantumFourierFeature:
    name = "quantum_fourier_feature"
    category = "tensor_quantum"

    def compress(self, tensor: np.ndarray, n_features: int = 8) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        orig_shape = t.shape
        if t.ndim == 2:
            m, n = t.shape
            nf = min(n_features, n)
            j_arange = np.arange(n)
            phases = 2.0 * np.pi * np.arange(1, nf + 1) / nf
            cos_basis = np.cos(phases[:, None] * j_arange[None, :] / n)
            features = t @ cos_basis.T
            params = np.arctan2(features, 1.0).astype(np.float32)
            scale = float(np.linalg.norm(t))
            meta = dict(shape=orig_shape, n_features=nf, scale=scale)
            data = struct.pack("<if", nf, scale) + _serialize(params)
            return data, meta
        meta = dict(shape=orig_shape, n_features=0, scale=0.0)
        data = struct.pack("<if", 0, 0.0) + _serialize(t.ravel().astype(np.float32))
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        n_features, scale = struct.unpack_from("<if", data, 0)
        pos = struct.calcsize("<if")
        if n_features == 0:
            flat = _deserialize(data[pos:])
            return flat[: int(np.prod(shape))].reshape(shape).astype(np.float32)
        m, n = shape
        params = _deserialize(data[pos : pos + m * n_features * 4]).reshape(
            m, n_features
        )
        features = np.tan(params.astype(np.float64))
        j_arange = np.arange(n)
        phases = 2.0 * np.pi * np.arange(1, n_features + 1) / n_features
        cos_basis = np.cos(phases[:, None] * j_arange[None, :] / n)
        recon = features @ cos_basis
        scale_factor = scale / (np.linalg.norm(recon) + 1e-30)
        return (recon * scale_factor).astype(np.float32)
