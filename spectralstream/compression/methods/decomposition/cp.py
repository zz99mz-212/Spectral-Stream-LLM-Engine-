"""CP / PARAFAC decomposition using alternating least squares."""

from __future__ import annotations

from typing import Tuple

import numpy as np


class CPDecomposition:
    """CP decomposition: W = sum_{r=1}^{R} a_r (X) b_r (X) c_r ..."""

    name = "cp_decomposition"
    category = "decomposition"

    def compress(
        self, tensor: np.ndarray, rank: int = None, max_iters: int = 30
    ) -> Tuple[bytes, dict]:
        t = np.asarray(tensor, dtype=np.float64)
        ndim = t.ndim
        shape = t.shape
        if ndim < 2:
            return tensor.astype(np.float16).tobytes(), {
                "original_shape": shape,
                "shape": shape,
                "passthrough": True,
            }
        if rank is None:
            rank = max(2, min(32, min(shape) // 2))
        r = max(2, min(rank, max(shape)))

        if ndim == 2:
            return self._compress_2d(t, r)

        rng = np.random.RandomState(42)
        factors = [rng.randn(s, r).astype(np.float64) for s in shape]
        self._normalize_factors(factors)

        prev_recon = np.zeros(shape, dtype=np.float64)
        for it in range(max_iters):
            for d in range(ndim):
                had = np.clip(self._compute_had(factors, d, ndim), 1e-30, 1e10)
                FtF = factors[d].T @ factors[d]
                V = FtF * had
                V_reg = (
                    V + 1e-4 * np.eye(r) * np.trace(V) / r
                    if r > 0
                    else V + 1e-4 * np.eye(r)
                )
                M = self._unfold(t, d) @ self._khatri_rao(factors, d)
                try:
                    factors[d] = np.linalg.solve(V_reg, M.T).T
                except np.linalg.LinAlgError:
                    factors[d] = np.linalg.lstsq(V_reg, M.T, rcond=None)[0].T
                factors[d] = np.clip(factors[d], -1e10, 1e10)
            self._normalize_factors(factors)
            recon = self._cp_reconstruct(factors, shape)
            diff = np.mean((recon - prev_recon) ** 2) / max(np.mean(recon**2), 1e-30)
            if np.isfinite(diff) and diff < 1e-8:
                break
            prev_recon = recon

        recon = self._cp_reconstruct(factors, shape)
        if not np.isfinite(np.mean(recon)):
            return self._compress_2d(t, r)

        data = b"".join(f.astype(np.float32).tobytes() for f in factors)
        meta = dict(
            shape=shape,
            ndim=ndim,
            rank=r,
            factor_shapes=[list(f.shape) for f in factors],
        )
        return data, meta

    def _compress_2d(self, t: np.ndarray, r: int) -> Tuple[bytes, dict]:
        U, S, Vt = np.linalg.svd(t, full_matrices=False)
        rk = min(r, len(S))
        factor0 = (U[:, :rk] * S[:rk]).astype(np.float64)
        factor1 = Vt[:rk, :].T.astype(np.float64)
        data = (
            factor0.astype(np.float32).tobytes() + factor1.astype(np.float32).tobytes()
        )
        meta = dict(
            shape=t.shape,
            ndim=2,
            rank=rk,
            factor_shapes=[list(factor0.shape), list(factor1.shape)],
        )
        return data, meta

    @staticmethod
    def _normalize_factors(factors):
        col_norms = np.sqrt(sum(np.sum(f**2, axis=0) for f in factors))
        col_norms = np.maximum(col_norms, 1e-30)
        for f in factors:
            f /= col_norms

    @staticmethod
    def _compute_had(factors, d, ndim):
        had = np.ones(factors[0].shape[1])
        for k in range(ndim):
            if k != d:
                col_norms = np.sum(factors[k] ** 2, axis=0)
                had = had * col_norms
        return had

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        if metadata.get("passthrough"):
            return (
                np.frombuffer(data, dtype=np.float16)
                .copy()
                .reshape(shape)
                .astype(np.float32)
            )
        ndim = metadata["ndim"]
        r = metadata["rank"]
        factors = []
        off = 0
        for fs in metadata["factor_shapes"]:
            n = int(np.prod(fs))
            factors.append(
                np.frombuffer(data[off : off + n * 4], dtype=np.float32).reshape(fs)
            )
            off += n * 4
        recon = self._cp_reconstruct([f.astype(np.float64) for f in factors], shape)
        return recon.astype(np.float32)

    @staticmethod
    def _cp_reconstruct(factors, shape):
        ndim = len(factors)
        if ndim == 2:
            return factors[0] @ factors[1].T
        labels = [chr(97 + i) for i in range(ndim)]
        sub = ",".join(f"{l}r" for l in labels) + "->" + "".join(labels)
        return np.einsum(sub, *factors)

    @staticmethod
    def _unfold(tensor: np.ndarray, mode: int) -> np.ndarray:
        return np.moveaxis(tensor, mode, 0).reshape(tensor.shape[mode], -1)

    @staticmethod
    def _khatri_rao(factors: list, exclude: int) -> np.ndarray:
        r = factors[0].shape[1]
        result = None
        for d in range(len(factors) - 1, -1, -1):
            if d == exclude:
                continue
            f = factors[d]
            if result is None:
                result = f
            else:
                result = np.einsum("ij,kj->ikj", result, f).reshape(
                    result.shape[0] * f.shape[0], r
                )
        return result if result is not None else np.ones((1, r))
