"""Kronecker and CUR decomposition methods."""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from .svd_decomposition import cur_decomposition
from .structured_decomposition import kronecker_decompose


class Kronecker:
    """Kronecker product decomposition: W ~ A (X) B."""

    name = "kronecker"
    category = "decomposition"

    def compress(
        self, tensor: np.ndarray, shape_a: Tuple[int, int] = None
    ) -> Tuple[bytes, dict]:
        t = np.asarray(tensor, dtype=np.float64)
        if t.ndim < 2 or min(t.shape) < 4:
            flat = t.ravel().astype(np.float32)
            return flat.astype(np.float16).tobytes(), {
                "original_shape": t.shape,
                "shape": t.shape,
                "passthrough": True,
            }
        m, n = t.shape
        if shape_a is None:
            a = max(1, int(math.isqrt(m)))
            while m % a != 0 and a > 1:
                a -= 1
            if m % a != 0 or (m // a) * (n // (m // a)) <= 0:
                flat = t.ravel().astype(np.float32)
                return flat.astype(np.float16).tobytes(), {
                    "original_shape": t.shape,
                    "shape": t.shape,
                    "passthrough": True,
                }
            shape_a = (a, m // a)
        try:
            result, ratio, snr = kronecker_decompose(t, shape_a)
        except (ValueError, np.linalg.LinAlgError):
            flat = t.ravel().astype(np.float32)
            return flat.astype(np.float16).tobytes(), {
                "original_shape": t.shape,
                "shape": t.shape,
                "passthrough": True,
            }
        data = result["A"].tobytes() + result["B"].tobytes()
        meta = dict(
            shape=result["shape"],
            A_shape=list(result["A"].shape),
            B_shape=list(result["B"].shape),
        )
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        if metadata.get("passthrough"):
            return (
                np.frombuffer(data, dtype=np.float16)
                .copy()
                .reshape(metadata["shape"])
                .astype(np.float32)
            )
        off = 0
        nA = int(np.prod(metadata["A_shape"]))
        A = np.frombuffer(data[off : off + nA * 4], dtype=np.float32).reshape(
            metadata["A_shape"]
        )
        off += nA * 4
        nB = int(np.prod(metadata["B_shape"]))
        B = np.frombuffer(data[off : off + nB * 4], dtype=np.float32).reshape(
            metadata["B_shape"]
        )
        recon = np.kron(A, B)
        return recon.reshape(metadata["shape"]).astype(np.float32)


class CURDecomposition:
    """CUR matrix decomposition: W ~ C @ U @ R."""

    name = "cur_decomposition"
    category = "decomposition"

    def compress(self, tensor: np.ndarray, rank: int = None) -> Tuple[bytes, dict]:
        t = np.asarray(tensor, dtype=np.float64)
        if t.ndim < 2 or min(t.shape) < 4:
            flat = t.ravel().astype(np.float32)
            return flat.astype(np.float16).tobytes(), {
                "original_shape": t.shape,
                "shape": t.shape,
                "passthrough": True,
            }
        if rank is None:
            rank = max(1, min(64, min(t.shape) // 4))
        try:
            result, ratio, snr = cur_decomposition(t, rank)
        except (ValueError, np.linalg.LinAlgError):
            flat = t.ravel().astype(np.float32)
            return flat.astype(np.float16).tobytes(), {
                "original_shape": t.shape,
                "shape": t.shape,
                "passthrough": True,
            }
        data = result["C"].tobytes() + result["U"].tobytes() + result["R"].tobytes()
        meta = dict(
            shape=result["shape"],
            C_shape=list(result["C"].shape),
            U_shape=list(result["U"].shape),
            R_shape=list(result["R"].shape),
        )
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        if metadata.get("passthrough"):
            return (
                np.frombuffer(data, dtype=np.float16)
                .copy()
                .reshape(metadata["shape"])
                .astype(np.float32)
            )
        off = 0
        nC = int(np.prod(metadata["C_shape"]))
        C = np.frombuffer(data[off : off + nC * 4], dtype=np.float32).reshape(
            metadata["C_shape"]
        )
        off += nC * 4
        nU = int(np.prod(metadata["U_shape"]))
        U = np.frombuffer(data[off : off + nU * 4], dtype=np.float32).reshape(
            metadata["U_shape"]
        )
        off += nU * 4
        nR = int(np.prod(metadata["R_shape"]))
        R = np.frombuffer(data[off : off + nR * 4], dtype=np.float32).reshape(
            metadata["R_shape"]
        )
        recon = C @ U @ R
        return recon.reshape(metadata["shape"]).astype(np.float32)
