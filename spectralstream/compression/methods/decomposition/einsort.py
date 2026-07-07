"""Einsort-based tensor train and low-rank tensor ring methods."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def _adaptive_rank(shape: Tuple[int, ...], rank: int = None) -> int:
    if rank is not None:
        return max(1, rank)
    d = len(shape)
    if d < 2:
        return 4
    return max(2, min(64, min(shape) // 2))


class EinsortTT:
    """Einsort-optimized Tensor Train decomposition."""

    name = "einsort_tt"
    category = "decomposition"

    def compress(self, tensor: np.ndarray, rank: int = None) -> Tuple[bytes, dict]:
        t = np.asarray(tensor, dtype=np.float64)
        shape = t.shape
        d = len(shape)
        if d < 2:
            flat = t.ravel().astype(np.float32)
            return flat.astype(np.float16).tobytes(), {
                "original_shape": shape,
                "shape": shape,
                "passthrough": True,
            }
        r = _adaptive_rank(shape, rank)
        r = max(2, min(r, min(shape)))
        cores = []
        current = t.copy()
        prev_r = 1
        for k in range(d - 1):
            unfolded = current.reshape(prev_r * shape[k], -1)
            U, S, Vt = np.linalg.svd(unfolded, full_matrices=False)
            rk = min(r, max(1, U.shape[1]), max(1, Vt.shape[0]))
            if rk < 1:
                rk = 1
            core = U[:, :rk].reshape(prev_r, shape[k], rk)
            cores.append(core.astype(np.float32))
            current = (S[:rk, None] * Vt[:rk, :]).reshape(rk, -1)
            prev_r = rk
        cores.append(current.reshape(prev_r, shape[-1], 1).astype(np.float32))
        data = b"".join(c.tobytes() for c in cores)
        meta = dict(shape=shape, rank=r, core_shapes=[list(c.shape) for c in cores])
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
        result = cores[0].astype(np.float64)
        for core in cores[1:]:
            result = np.tensordot(result, core.astype(np.float64), axes=([-1], [0]))
        return result.reshape(shape).astype(np.float32)


class LOTR:
    """Low-rank Tensor Ring via sequential SVD with ring closure."""

    name = "lotr"
    category = "decomposition"

    def compress(self, tensor: np.ndarray, rank: int = None) -> Tuple[bytes, dict]:
        t = np.asarray(tensor, dtype=np.float64)
        orig_shape = t.shape
        if t.ndim < 3:
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
            U, S, Vt = np.linalg.svd(unfolded, full_matrices=False)
            rk = min(r, U.shape[1] - 1)
            if rk < 1:
                rk = 1
            if k < d - 1:
                core = U[:, :rk].reshape(prev_r, shape[k], rk)
                current = (S[:rk, None] * Vt[:rk, :]).reshape(rk, -1)
                prev_r = rk
            else:
                core = U[:, :rk].reshape(prev_r, shape[k], rk)
                cores[0] = np.tensordot(
                    core.astype(np.float64),
                    cores[0].astype(np.float64),
                    axes=([-1], [0]),
                ).astype(np.float32)
                continue
            cores.append(core.astype(np.float32))
        data = b"".join(c.tobytes() for c in cores)
        meta = dict(
            shape=orig_shape, rank=r, core_shapes=[list(c.shape) for c in cores]
        )
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
        result = cores[0].astype(np.float64)
        for core in cores[1:]:
            result = np.tensordot(result, core.astype(np.float64), axes=([-1], [0]))
        ndim = len(metadata.get("core_shapes", []))
        if ndim >= 2:
            trace_result = np.trace(result, axis1=0, axis2=min(ndim, result.ndim))
            return trace_result.reshape(shape).astype(np.float32)
        return result.reshape(shape).astype(np.float32)
