"""Kronecker and CUR decomposition methods.

NOTE: Kronecker decomposition assumes W ≈ A ⊗ B structure.
Real neural network weights are NOT Kronecker products (fit error > 90% typically).
We add structure checks and SVD+Kronecker fallback for real weights.
"""

from __future__ import annotations

import logging
import math
from typing import Tuple

import numpy as np

from .svd_decomposition import cur_decomposition
from .structured_decomposition import kronecker_decompose

logger = logging.getLogger(__name__)


class Kronecker:
    """Kronecker product decomposition: W ~ A (X) B.

    Real neural network weights rarely have Kronecker structure.
    Strategy:
    1. Check Kronecker fit error: ||W - A⊗B|| / ||W||
    2. If fit error < 10%: use pure Kronecker decomposition
    3. If fit error >= 10% and < 50%: try SVD low-rank first, then
       Kronecker on the SVD factors (SVD+Kronecker)
    4. If fit error >= 50%: fall back to SVD only
    """

    name = "kronecker"
    category = "decomposition"
    GOOD_KRONECKER_FIT = 0.10  # 10%: use pure Kronecker
    MODERATE_KRONECKER_FIT = 0.50  # 50%: try SVD+Kronecker

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
        t_norm = float(np.linalg.norm(t, "fro")) + 1e-30

        # Step 1: Check Kronecker fit error
        try:
            from spectralstream.compression.adaptive_rank import (
                estimate_kronecker_fit_error,
            )

            kronecker_err = estimate_kronecker_fit_error(tensor)
        except Exception:
            kronecker_err = 1.0

        # Step 2: Pure Kronecker if fit is very good
        if kronecker_err < self.GOOD_KRONECKER_FIT:
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
                data = result["A"].tobytes() + result["B"].tobytes()
                meta = dict(
                    shape=result["shape"],
                    A_shape=list(result["A"].shape),
                    B_shape=list(result["B"].shape),
                    kronecker_method="pure",
                    kronecker_fit_error=float(kronecker_err),
                )
                return data, meta
            except (ValueError, np.linalg.LinAlgError):
                flat = t.ravel().astype(np.float32)
                return flat.astype(np.float16).tobytes(), {
                    "original_shape": t.shape,
                    "shape": t.shape,
                    "passthrough": True,
                }

        # Step 3: SVD+Kronecker for moderate fit
        if kronecker_err < self.MODERATE_KRONECKER_FIT:
            try:
                from spectralstream.compression.adaptive_rank import (
                    estimate_adaptive_rank,
                )

                svd_rank = estimate_adaptive_rank(
                    t, energy_threshold=0.999, max_rank=min(64, min(m, n) // 4)
                )
                svd_rank = max(2, min(svd_rank, min(m, n) - 1))
                U, S, Vh = np.linalg.svd(t, full_matrices=False)
                svd_rank = min(svd_rank, len(S))
                U_r = U[:, :svd_rank]
                S_r = S[:svd_rank]
                Vh_r = Vh[:svd_rank, :]

                # Try Kronecker on U_r and Vh_r separately
                u_err = estimate_kronecker_fit_error(U_r)
                v_err = estimate_kronecker_fit_error(Vh_r.T)

                if u_err < 0.3 and v_err < 0.3:
                    data_u, _ = self.compress(U_r)
                    data_v, _ = self.compress(Vh_r.T)
                    data = data_u + data_v + S_r.astype(np.float16).tobytes()
                    meta = dict(
                        shape=tensor.shape,
                        kronecker_method="svd_kronecker",
                        svd_rank=svd_rank,
                        U_shape=list(U_r.shape),
                        Vt_shape=list(Vh_r.shape),
                        kronecker_fit_error=float(kronecker_err),
                    )
                    return data, meta
            except Exception:
                pass

        # Step 4: Fall back to SVD only (Kronecker fit is terrible)
        logger.debug(
            "Kronecker fit error %.2f — using SVD fallback",
            kronecker_err,
        )
        try:
            from spectralstream.compression.adaptive_rank import (
                estimate_adaptive_rank,
            )

            svd_rank = estimate_adaptive_rank(
                t, energy_threshold=0.999, max_rank=min(128, min(m, n) // 2)
            )
            svd_rank = max(2, min(svd_rank, min(m, n) - 1))
            U, S, Vh = np.linalg.svd(t, full_matrices=False)
            svd_rank = min(svd_rank, len(S))
            data = (
                U[:, :svd_rank].astype(np.float16).tobytes()
                + S[:svd_rank].astype(np.float16).tobytes()
                + Vh[:svd_rank, :].astype(np.float16).tobytes()
            )
            meta = dict(
                shape=tensor.shape,
                kronecker_method="svd_fallback",
                svd_rank=svd_rank,
                kronecker_fit_error=float(kronecker_err),
            )
            return data, meta
        except Exception:
            flat = t.ravel().astype(np.float32)
            return flat.astype(np.float16).tobytes(), {
                "original_shape": t.shape,
                "shape": t.shape,
                "passthrough": True,
            }

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        if metadata.get("passthrough"):
            return (
                np.frombuffer(data, dtype=np.float16)
                .copy()
                .reshape(metadata["shape"])
                .astype(np.float32)
            )

        method = metadata.get("kronecker_method", "pure")

        if method == "pure":
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

        if method == "svd_fallback":
            svd_rank = metadata["svd_rank"]
            m, n = metadata["shape"]
            off = 0
            U_bytes = m * svd_rank * 2
            U_r = np.frombuffer(data[off : off + U_bytes], dtype=np.float16).reshape(
                m, svd_rank
            )
            off += U_bytes
            S_r = np.frombuffer(data[off : off + svd_rank * 2], dtype=np.float16)
            off += svd_rank * 2
            Vh_r = np.frombuffer(
                data[off : off + svd_rank * n * 2], dtype=np.float16
            ).reshape(svd_rank, n)
            return (
                (U_r.astype(np.float32) * S_r.astype(np.float32))
                @ Vh_r.astype(np.float32)
            ).reshape(metadata["shape"])

        if method == "svd_kronecker":
            svd_rank = metadata["svd_rank"]
            m, n = metadata["shape"]
            U_shape = metadata["U_shape"]
            Vt_shape = metadata["Vt_shape"]
            off = 0
            # Decompress Kronecker-compressed U_r
            nU = int(np.prod(U_shape))
            A_U = np.frombuffer(data[off : off + nU * 4], dtype=np.float32).reshape(
                U_shape[0], svd_rank
            )
            off += nU * 4
            # Decompress Kronecker-compressed Vt_r
            nV = int(np.prod(Vt_shape))
            A_V = np.frombuffer(data[off : off + nV * 4], dtype=np.float32).reshape(
                svd_rank, Vt_shape[1]
            )
            off += nV * 4
            S_r = np.frombuffer(
                data[off : off + svd_rank * 2], dtype=np.float16
            ).astype(np.float32)
            return (A_U * S_r) @ A_V

        # Fallback: try kron product
        off = 0
        nA = int(np.prod(metadata.get("A_shape", [1, 1])))
        if nA > 1:
            A = np.frombuffer(data[off : off + nA * 4], dtype=np.float32).reshape(
                metadata["A_shape"]
            )
            off += nA * 4
            nB = int(np.prod(metadata.get("B_shape", [1, 1])))
            B = np.frombuffer(data[off : off + nB * 4], dtype=np.float32).reshape(
                metadata["B_shape"]
            )
            return np.kron(A, B).reshape(metadata["shape"]).astype(np.float32)
        return np.zeros(metadata["shape"], dtype=np.float32)


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
