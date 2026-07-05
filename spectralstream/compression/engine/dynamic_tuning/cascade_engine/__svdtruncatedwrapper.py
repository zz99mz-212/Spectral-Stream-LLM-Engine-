from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spectralstream.compression.engine._dataclasses import TensorProfile
from spectralstream.compression.engine._helpers import _compute_metrics, _compute_ratio
from spectralstream.compression.engine._methods import METHOD_REGISTRY
from spectralstream.core.math_primitives import (
    dct_2d,
    fwht,
    idct_2d,
    ifwht,
    next_power_of_two,
)
from ._stagetype import _ensure_2d


class _SVDTruncatedWrapper:
    name = "svd_truncated"
    category = "decomposition"

    def compress(self, tensor: np.ndarray, rank: int = 32) -> Tuple[bytes, dict]:
        mat = _ensure_2d(tensor).astype(np.float64)
        k = min(rank, *mat.shape)
        m, n = mat.shape
        if k <= 0 or k >= min(m, n):
            flat = tensor.astype(np.float32).ravel()
            return flat.tobytes(), {
                "shape": tensor.shape,
                "rank": 0,
                "passthrough": True,
            }
        U, s, Vt = np.linalg.svd(mat, full_matrices=False)
        U_k = U[:, :k].astype(np.float32)
        s_k = s[:k].astype(np.float32)
        Vt_k = Vt[:k, :].astype(np.float32)
        header = struct.pack("<III", m, n, k)
        data = header + U_k.tobytes() + s_k.tobytes() + Vt_k.tobytes()
        return data, {"shape": tensor.shape, "rank": k, "m": m, "n": n}

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        if metadata.get("passthrough"):
            return np.frombuffer(data, dtype=np.float32).reshape(metadata["shape"])
        if "m" in metadata and "n" in metadata and "rank" in metadata:
            m = metadata["m"]
            n = metadata["n"]
            k = metadata["rank"]
        else:
            m, n, k = struct.unpack_from("<III", data, 0)
        pos = 12
        U_k = np.frombuffer(data[pos : pos + m * k * 4], dtype=np.float32).reshape(m, k)
        pos += m * k * 4
        s_k = np.frombuffer(data[pos : pos + k * 4], dtype=np.float32)
        pos += k * 4
        Vt_k = np.frombuffer(data[pos : pos + k * n * 4], dtype=np.float32).reshape(
            k, n
        )
        recon = (U_k * s_k) @ Vt_k
        return recon.astype(np.float32).reshape(metadata["shape"])
