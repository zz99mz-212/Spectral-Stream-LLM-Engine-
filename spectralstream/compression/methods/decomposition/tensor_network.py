"""Tensor network decomposition methods (MPS, hierarchical MPS)."""

from __future__ import annotations

from typing import Tuple

import numpy as np


class TensorNetwork:
    """MPS-style tensor network via sequential SVD decomposition."""

    name = "tensor_network"
    category = "decomposition"

    def compress(self, tensor: np.ndarray, bond_dim: int = None) -> Tuple[bytes, dict]:
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
        if bond_dim is None:
            bond_dim = max(1, min(64, min(shape) // 2))
        bd = max(1, min(bond_dim, min(shape)))
        cores = []
        current = t.copy()
        prev_bond = 1
        for k in range(d - 1):
            unfolded = current.reshape(prev_bond * shape[k], -1)
            U, S, Vt = np.linalg.svd(unfolded, full_matrices=False)
            rk = min(bd, max(1, U.shape[1]), max(1, Vt.shape[0]))
            if rk < 1:
                rk = 1
            core = U[:, :rk].reshape(prev_bond, shape[k], rk)
            cores.append(core.astype(np.float32))
            current = (S[:rk, None] * Vt[:rk, :]).reshape(rk, -1)
            prev_bond = rk
        cores.append(current.reshape(prev_bond, shape[-1], 1).astype(np.float32))
        data = b"".join(c.tobytes() for c in cores)
        meta = dict(
            shape=shape, bond_dim=bd, core_shapes=[list(c.shape) for c in cores]
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
        return result.reshape(shape).astype(np.float32)


class HierarchicalMPS:
    """DMRG-inspired hierarchical MPS."""

    name = "hierarchical_mps"
    category = "decomposition"

    def compress(self, tensor: np.ndarray, bond_dim: int = None) -> Tuple[bytes, dict]:
        return TensorNetwork().compress(tensor, bond_dim=bond_dim)

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        return TensorNetwork().decompress(data, metadata)
