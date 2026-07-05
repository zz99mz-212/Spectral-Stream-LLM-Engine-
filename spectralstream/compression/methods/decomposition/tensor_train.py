"""Tensor Train and Tensor Ring decompositions via sequential SVD."""

from __future__ import annotations

import gc
import math
from typing import Tuple

import numpy as np

_SVD_LIMIT = 1024 * 1024


def _randomized_svd(X, n_components, n_oversamples=10, n_iter=3):
    m, n = X.shape
    k = min(n_components + n_oversamples, min(m, n))
    rng = np.random.RandomState(42)
    O = rng.randn(n, k).astype(X.dtype)
    Y = X @ O
    for _ in range(n_iter):
        Y = X @ (X.T @ Y)
    Q, _ = np.linalg.qr(Y)
    B = Q.T @ X
    Ub, s, Vt = np.linalg.svd(B, full_matrices=False)
    nc = min(n_components, len(s))
    U = Q @ Ub[:, :nc]
    return U[:, :nc], s[:nc], Vt[:nc, :]


def _adaptive_rank(shape: Tuple[int, ...], rank: int = None) -> int:
    if rank is not None:
        return max(1, rank)
    d = len(shape)
    if d < 2:
        return 4
    return max(2, min(64, min(shape) // 2))


class TensorTrain:
    """Tensor Train decomposition via sequential SVD."""

    name = "tensor_train"
    category = "decomposition"

    def compress(self, tensor: np.ndarray, rank: int = None) -> Tuple[bytes, dict]:
        t = np.asarray(tensor, dtype=np.float64)
        shape = t.shape
        d = len(shape)
        if d < 2 or min(shape) < 2:
            flat = t.ravel().astype(np.float32)
            return flat.astype(np.float16).tobytes(), {
                "original_shape": shape,
                "shape": shape,
                "passthrough": True,
            }
        r = _adaptive_rank(shape, rank)
        r = max(1, min(r, min(shape)))
        cores = []
        current = t.copy()
        prev_r = 1
        for k in range(d - 1):
            unfolded = current.reshape(prev_r * shape[k], -1)
            if unfolded.size > _SVD_LIMIT and min(unfolded.shape) > 512:
                U, S, Vt = _randomized_svd(unfolded, r)
            else:
                U, S, Vt = np.linalg.svd(unfolded, full_matrices=False)
            rk = min(r, max(1, U.shape[1]), max(1, Vt.shape[0]))
            core = U[:, :rk].reshape(prev_r, shape[k], rk)
            cores.append(core)
            current = (S[:rk, None] * Vt[:rk, :]).reshape(rk, -1)
            prev_r = rk
            del U, S, Vt, unfolded
            gc.collect()
        cores.append(current.reshape(prev_r, shape[-1], 1))
        data = b"".join(c.astype(np.float32).tobytes() for c in cores)
        meta = dict(shape=shape, rank=r, core_shapes=[list(c.shape) for c in cores])
        del current, cores
        gc.collect()
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        if metadata.get("passthrough"):
            return (
                np.frombuffer(data, dtype=np.float16)
                .copy()
                .reshape(shape)
                .astype(np.float32)
            )
        cores = []
        off = 0
        for cs in metadata["core_shapes"]:
            n = int(np.prod(cs))
            cores.append(
                np.frombuffer(data[off : off + n * 4], dtype=np.float32).reshape(cs)
            )
            off += n * 4
        result = cores[0]
        for core in cores[1:]:
            result = np.tensordot(result, core, axes=([-1], [0]))
        return result.reshape(shape).astype(np.float32)


class TensorRing:
    """Tensor Ring decomposition."""

    name = "tensor_ring"
    category = "decomposition"

    def compress(self, tensor: np.ndarray, rank: int = None) -> Tuple[bytes, dict]:
        t = np.asarray(tensor, dtype=np.float64)
        orig_shape = t.shape
        if t.ndim < 3 or t.size <= 8192:
            flat = t.ravel().astype(np.float32)
            return flat.astype(np.float16).tobytes(), {
                "original_shape": orig_shape,
                "shape": orig_shape,
                "passthrough": True,
            }
        shape = t.shape
        d = len(shape)
        r = _adaptive_rank(shape, rank)
        r = max(2, min(r, min(shape)))
        cores = []
        current = t.copy()
        prev_r = 1
        for k in range(d):
            unfolded = current.reshape(prev_r * shape[k], -1)
            if unfolded.size > _SVD_LIMIT and min(unfolded.shape) > 512:
                U, S, Vt = _randomized_svd(unfolded, r + 1)
            else:
                U, S, Vt = np.linalg.svd(unfolded, full_matrices=False)
            rk = min(r, max(1, U.shape[1] - 1))
            if rk < 1:
                rk = 1
            if k < d - 1:
                core = U[:, :rk].reshape(prev_r, shape[k], rk)
                current = (S[:rk, None] * Vt[:rk, :]).reshape(rk, -1)
                prev_r = rk
            else:
                core = U[:, :rk].reshape(prev_r, shape[k], rk)
                cores[0] = np.tensordot(core, cores[0], axes=([-1], [0]))
                del core
                continue
            cores.append(core)
            del U, S, Vt, unfolded
            gc.collect()
        data = b"".join(c.astype(np.float32).tobytes() for c in cores)
        meta = dict(
            shape=orig_shape, rank=r, ndim=d, core_shapes=[list(c.shape) for c in cores]
        )
        del current, cores
        gc.collect()
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata.get("shape") or metadata.get("original_shape")
        if metadata.get("passthrough"):
            return (
                np.frombuffer(data, dtype=np.float16)
                .copy()
                .reshape(shape)
                .astype(np.float32)
            )
        ndim = metadata["ndim"]
        cores = []
        off = 0
        for cs in metadata["core_shapes"]:
            n = int(np.prod(cs))
            cores.append(
                np.frombuffer(data[off : off + n * 4], dtype=np.float32).reshape(cs)
            )
            off += n * 4
        result = cores[0]
        for core in cores[1:]:
            result = np.tensordot(result, core, axes=([-1], [0]))
        if ndim >= 2:
            trace = np.trace(result, axis1=0, axis2=-1)
            return trace.reshape(shape).astype(np.float32)
        return result.ravel()[: int(np.prod(shape))].reshape(shape).astype(np.float32)


class TTOrthogonal:
    """Tensor Train with orthogonal reduction."""

    name = "tt_orthogonal"
    category = "decomposition"

    def compress(self, tensor: np.ndarray, rank: int = None) -> Tuple[bytes, dict]:
        return TensorTrain().compress(tensor, rank=rank)

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        return TensorTrain().decompress(data, metadata)


class TTSVD:
    """Tensor Train via sequential truncated SVD."""

    name = "tt_svd"
    category = "decomposition"

    def compress(self, tensor: np.ndarray, rank: int = None) -> Tuple[bytes, dict]:
        return TensorTrain().compress(tensor, rank=rank)

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        return TensorTrain().decompress(data, metadata)
