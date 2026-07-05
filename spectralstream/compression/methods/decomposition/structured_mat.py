"""Structured matrix approximation methods (BlockDiagonal, Toeplitz, Hankel)."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .structured_decomposition import (
    block_diagonal_decompose,
    toeplitz_decompose,
    hankel_decompose,
)


class BlockDiagonal:
    """Block-diagonal approximation with block-wise SVD."""

    name = "block_diagonal"
    category = "decomposition"

    def compress(self, tensor: np.ndarray, n_blocks: int = 4) -> Tuple[bytes, dict]:
        result, ratio, snr = block_diagonal_decompose(tensor, n_blocks)
        data = b"".join(u.tobytes() for u in result["U_blocks"])
        data += b"".join(s.tobytes() for s in result["s_blocks"])
        data += b"".join(v.tobytes() for v in result["Vt_blocks"])
        meta = dict(
            shape=result["shape"],
            n_blocks=result["n_blocks"],
            block_shape=list(result["block_shape"]),
            U_shapes=[list(u.shape) for u in result["U_blocks"]],
            s_shapes=[list(s.shape) for s in result["s_blocks"]],
            Vt_shapes=[list(v.shape) for v in result["Vt_blocks"]],
        )
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        n_blocks = metadata["n_blocks"]
        bm, bn = metadata["block_shape"]
        Us, Ss, Vts = [], [], []
        off = 0
        for us in metadata["U_shapes"]:
            n = int(np.prod(us))
            Us.append(
                np.frombuffer(data[off : off + n * 4], dtype=np.float32).reshape(us)
            )
            off += n * 4
        for ss in metadata["s_shapes"]:
            n = int(np.prod(ss))
            Ss.append(np.frombuffer(data[off : off + n * 4], dtype=np.float32))
            off += n * 4
        for vs in metadata["Vt_shapes"]:
            n = int(np.prod(vs))
            Vts.append(
                np.frombuffer(data[off : off + n * 4], dtype=np.float32).reshape(vs)
            )
            off += n * 4
        recon = np.zeros(metadata["shape"], dtype=np.float64)
        for k in range(n_blocks):
            i0, j0 = k * bm, k * bn
            U = Us[k].astype(np.float64)
            s = Ss[k].astype(np.float64)
            Vt = Vts[k].astype(np.float64)
            recon[i0 : i0 + bm, j0 : j0 + bn] = (U * s) @ Vt
        return recon.astype(np.float32)


class Toeplitz:
    """Toeplitz (constant diagonal) matrix approximation."""

    name = "toeplitz"
    category = "decomposition"

    def compress(self, tensor: np.ndarray) -> Tuple[bytes, dict]:
        result, ratio, snr = toeplitz_decompose(tensor)
        data = result["w"].tobytes()
        meta = dict(shape=result["shape"], w_shape=list(result["w"].shape))
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        w = np.frombuffer(data, dtype=np.float32)
        m, n = metadata["shape"]
        i = np.arange(m)[:, None]
        j = np.arange(n)[None, :]
        recon = w[j - i + m - 1]
        return recon.reshape(m, n).astype(np.float32)


class Hankel:
    """Hankel (constant anti-diagonal) matrix approximation."""

    name = "hankel"
    category = "decomposition"

    def compress(self, tensor: np.ndarray) -> Tuple[bytes, dict]:
        result, ratio, snr = hankel_decompose(tensor)
        data = result["w"].tobytes()
        meta = dict(shape=result["shape"], w_shape=list(result["w"].shape))
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        w = np.frombuffer(data, dtype=np.float32)
        m, n = metadata["shape"]
        i = np.arange(m)[:, None]
        j = np.arange(n)[None, :]
        recon = w[i + j]
        return recon.reshape(m, n).astype(np.float32)
