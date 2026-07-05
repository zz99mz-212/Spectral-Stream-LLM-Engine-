"""MERA and PEPS tensor network methods."""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np


class ADNTNMERA:
    """ADTN-MERA: Multi-scale Entanglement Renormalization Ansatz."""

    name = "adntn_mera"
    category = "decomposition"

    def compress(self, tensor: np.ndarray, bond_dim: int = None) -> Tuple[bytes, dict]:
        t = np.asarray(tensor, dtype=np.float64)
        shape = t.shape
        if t.ndim < 2 or min(shape) < 4:
            flat = t.ravel().astype(np.float32)
            return flat.astype(np.float16).tobytes(), {
                "original_shape": shape,
                "shape": shape,
                "passthrough": True,
            }
        if bond_dim is None:
            bond_dim = max(1, min(32, min(shape) // 2))
        bd = max(1, min(bond_dim, min(shape)))
        n = max(shape)

        isometries = []
        disentanglers = []
        current = t.copy()
        level = 0
        while min(current.shape) > bd * 2 and level < 4:
            m, n_c = current.shape
            k = max(bd, min(m, n_c) // 2)
            U, S, Vt = np.linalg.svd(current, full_matrices=False)
            k = min(k, len(S))
            isometries.append(U[:, :k].astype(np.float32))
            disentanglers.append(Vt[:k, :].astype(np.float32))
            current = S[:k, None] * Vt[:k, :]
            level += 1

        core = current.astype(np.float32)
        data = core.tobytes()
        for iso in isometries:
            data += iso.tobytes()
        for dis in disentanglers:
            data += dis.tobytes()

        meta = dict(
            shape=shape,
            bond_dim=bd,
            n_levels=level,
            core_shape=list(core.shape),
            iso_shapes=[list(i.shape) for i in isometries],
            dis_shapes=[list(d.shape) for d in disentanglers],
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
        n_levels = metadata["n_levels"]
        off = 0
        ncore = int(np.prod(metadata["core_shape"]))
        core = np.frombuffer(data[off : off + ncore * 4], dtype=np.float32).reshape(
            metadata["core_shape"]
        )
        off += ncore * 4

        isometries = []
        for ish in metadata["iso_shapes"]:
            ni = int(np.prod(ish))
            isometries.append(
                np.frombuffer(data[off : off + ni * 4], dtype=np.float32).reshape(ish)
            )
            off += ni * 4

        disentanglers = []
        for dsh in metadata["dis_shapes"]:
            nd = int(np.prod(dsh))
            disentanglers.append(
                np.frombuffer(data[off : off + nd * 4], dtype=np.float32).reshape(dsh)
            )
            off += nd * 4

        recon = core.astype(np.float64)
        for level in reversed(range(n_levels)):
            if level < len(isometries) and level < len(disentanglers):
                iso = isometries[level].astype(np.float64)
                dis = disentanglers[level].astype(np.float64)
                try:
                    recon = iso @ recon
                except ValueError:
                    pass

        m, n = metadata["shape"]
        if recon.ndim == 2 and (recon.shape[0] < m or recon.shape[1] < n):
            padded = np.zeros((m, n), dtype=np.float64)
            padded[: recon.shape[0], : recon.shape[1]] = recon
            recon = padded

        return (
            recon[:m, :n].astype(np.float32)
            if recon.ndim == 2
            else np.zeros((m, n), dtype=np.float32)
        )


class IPEPS2D:
    """Infinite PEPS 2D tensor network via block-wise truncated SVD."""

    name = "ipeps_2d"
    category = "decomposition"

    def compress(self, tensor: np.ndarray, bond_dim: int = None) -> Tuple[bytes, dict]:
        t = np.asarray(tensor, dtype=np.float64)
        shape = t.shape
        if t.ndim < 2 or min(shape) < 4:
            flat = t.ravel().astype(np.float32)
            return flat.astype(np.float16).tobytes(), {
                "original_shape": shape,
                "shape": shape,
                "passthrough": True,
            }
        if bond_dim is None:
            bond_dim = max(1, min(32, min(shape) // 2))
        bd = max(1, min(bond_dim, min(shape)))

        m, n = shape
        left_parts = []
        right_parts = []
        positions = []
        for i in range(0, m, 2):
            for j in range(0, n, 2):
                i_end = min(i + 2, m)
                j_end = min(j + 2, n)
                block = t[i:i_end, j:j_end]
                if block.size < 4:
                    continue
                nr, nc = block.shape
                flat_block = block.reshape(nr, -1) if nc > 1 else block.reshape(-1, 1)
                U, S, Vt = np.linalg.svd(flat_block, full_matrices=False)
                k = min(bd, len(S))
                left_parts.append((U[:, :k] * S[:k]).astype(np.float32))
                right_parts.append(Vt[:k, :].astype(np.float32))
                positions.append((i, j))

        left_data = b"".join(p.tobytes() for p in left_parts)
        right_data = b"".join(p.tobytes() for p in right_parts)
        data = left_data + right_data
        meta = dict(
            shape=shape,
            bond_dim=bd,
            n_sites=len(positions),
            left_shapes=[list(p.shape) for p in left_parts],
            right_shapes=[list(p.shape) for p in right_parts],
            positions=positions,
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
        m, n = metadata["shape"]
        recon = np.zeros((m, n), dtype=np.float64)
        n_sites = metadata["n_sites"]
        left_shapes = metadata["left_shapes"]
        right_shapes = metadata["right_shapes"]
        off = 0
        left_parts = []
        for ls in left_shapes:
            sz = int(np.prod(ls))
            left_parts.append(
                np.frombuffer(data[off : off + sz * 4], dtype=np.float32).reshape(ls)
            )
            off += sz * 4
        right_parts = []
        for rs in right_shapes:
            sz = int(np.prod(rs))
            right_parts.append(
                np.frombuffer(data[off : off + sz * 4], dtype=np.float32).reshape(rs)
            )
            off += sz * 4
        positions = metadata["positions"]
        for idx in range(n_sites):
            i, j = positions[idx]
            left = left_parts[idx].astype(np.float64)
            right = right_parts[idx].astype(np.float64)
            block = left @ right
            i_end = min(i + 2, m)
            j_end = min(j + 2, n)
            recon[i:i_end, j:j_end] += block[: i_end - i, : j_end - j]
        return recon.astype(np.float32)
