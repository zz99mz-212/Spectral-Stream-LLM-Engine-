"""Butterfly and Monarch matrix decomposition methods."""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from .structured_decomposition import butterfly_factorize


def _butterfly_reconstruct(blocks, N, L, m, n):
    recon = np.eye(N, dtype=np.float64)
    for level in reversed(range(L)):
        s = N // (1 << (level + 1))
        nb = N // (2 * s)
        i_base = np.arange(nb) * 2 * s
        kk = np.arange(s)
        is0 = (i_base[:, None] + kk[None, :]).ravel()
        is1 = is0 + s
        B = np.zeros((N, N), dtype=np.float64)
        blk = blocks[level]
        B[is0, is0] = blk[:, 0, 0]
        B[is0, is1] = blk[:, 0, 1]
        B[is1, is0] = blk[:, 1, 0]
        B[is1, is1] = blk[:, 1, 1]
        recon = B @ recon
    return recon[:m, :n]


class Butterfly:
    """Butterfly matrix factorization: W = B_0 @ B_1 @ ... @ B_{L-1}."""

    name = "butterfly"
    category = "decomposition"

    def compress(self, tensor: np.ndarray, n_levels: int = None) -> Tuple[bytes, dict]:
        t = np.asarray(tensor, dtype=np.float64)
        if t.ndim < 2 or min(t.shape) < 4:
            flat = t.ravel().astype(np.float32)
            return flat.astype(np.float16).tobytes(), {
                "original_shape": t.shape,
                "shape": t.shape,
                "passthrough": True,
            }
        m, n = t.shape
        N = 1 << (max(m, n) - 1).bit_length()
        if N > 4096:
            flat = t.ravel().astype(np.float32)
            return flat.astype(np.float16).tobytes(), {
                "original_shape": t.shape,
                "shape": t.shape,
                "passthrough": True,
            }
        t_energy = np.mean(t**2)
        if t_energy < 1e-30:
            flat = t.ravel().astype(np.float32)
            return flat.astype(np.float16).tobytes(), {
                "original_shape": t.shape,
                "shape": t.shape,
                "passthrough": True,
            }
        try:
            result, ratio, snr = butterfly_factorize(t, n_levels)
            free_params = sum(b.size for b in result["blocks"])
        except Exception:
            flat = t.ravel().astype(np.float32)
            return flat.astype(np.float16).tobytes(), {
                "original_shape": t.shape,
                "shape": t.shape,
                "passthrough": True,
            }
        if free_params >= m * n:
            flat = t.ravel().astype(np.float32)
            return flat.astype(np.float16).tobytes(), {
                "original_shape": t.shape,
                "shape": t.shape,
                "passthrough": True,
            }
        blocks_recon = _butterfly_reconstruct(
            result["blocks"], result["N"], result["L"], m, n
        )
        if blocks_recon is not None:
            rel_err = np.mean((t - blocks_recon) ** 2) / t_energy
            if rel_err > 2.0:
                flat = t.ravel().astype(np.float32)
                return flat.astype(np.float16).tobytes(), {
                    "original_shape": t.shape,
                    "shape": t.shape,
                    "passthrough": True,
                }
        data = b"".join(b.tobytes() for b in result["blocks"])
        meta = dict(
            shape=result["shape"],
            N=result["N"],
            L=result["L"],
            block_shapes=[list(b.shape) for b in result["blocks"]],
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
        N = metadata["N"]
        L = metadata["L"]
        blocks = []
        off = 0
        for bs in metadata["block_shapes"]:
            n = int(np.prod(bs))
            blocks.append(
                np.frombuffer(data[off : off + n * 4], dtype=np.float32).reshape(bs)
            )
            off += n * 4
        recon = np.eye(N, dtype=np.float64)
        for level in reversed(range(L)):
            s = N // (1 << (level + 1))
            nb = N // (2 * s)
            i_base = np.arange(nb) * 2 * s
            kk = np.arange(s)
            is0 = (i_base[:, None] + kk[None, :]).ravel()
            is1 = is0 + s
            B = np.zeros((N, N), dtype=np.float64)
            blk = blocks[level]
            B[is0, is0] = blk[:, 0, 0]
            B[is0, is1] = blk[:, 0, 1]
            B[is1, is0] = blk[:, 1, 0]
            B[is1, is1] = blk[:, 1, 1]
            recon = B @ recon
        m, n = metadata["shape"]
        return recon[:m, :n].astype(np.float32)


class Monarch:
    """Monarch matrix decomposition: block-diagonal + low-rank factors."""

    name = "monarch"
    category = "decomposition"

    def compress(
        self, tensor: np.ndarray, block_size: int = None
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
        if block_size is None:
            block_size = max(2, min(64, min(m, n) // 4))
        bs = max(2, min(block_size, min(m, n) // 2))
        n_col = max(1, n // bs)
        n_row = max(1, m // bs)
        bm, bn = m // n_row, n // n_col
        blocks = []
        for i in range(n_row):
            for j in range(n_col):
                bi = t[i * bm : (i + 1) * bm, j * bn : (j + 1) * bn]
                U, S, Vt = np.linalg.svd(bi, full_matrices=False)
                r = max(1, min(bs, len(S)))
                blocks.append(
                    {
                        "U": U[:, :r].astype(np.float32),
                        "S": S[:r].astype(np.float32),
                        "Vt": Vt[:r, :].astype(np.float32),
                        "i": i,
                        "j": j,
                    }
                )
        data = b"".join(
            b["U"].tobytes() + b["S"].tobytes() + b["Vt"].tobytes() for b in blocks
        )
        meta = dict(
            shape=t.shape,
            n_row=n_row,
            n_col=n_col,
            bm=bm,
            bn=bn,
            block_meta=[
                {
                    "i": b["i"],
                    "j": b["j"],
                    "U_shape": list(b["U"].shape),
                    "S_shape": list(b["S"].shape),
                    "Vt_shape": list(b["Vt"].shape),
                }
                for b in blocks
            ],
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
        n_row, n_col = metadata["n_row"], metadata["n_col"]
        bm, bn = metadata["bm"], metadata["bn"]
        recon = np.zeros((m, n), dtype=np.float64)
        off = 0
        for bm_meta in metadata["block_meta"]:
            nu = int(np.prod(bm_meta["U_shape"]))
            U = np.frombuffer(data[off : off + nu * 4], dtype=np.float32).reshape(
                bm_meta["U_shape"]
            )
            off += nu * 4
            ns = int(np.prod(bm_meta["S_shape"]))
            S = np.frombuffer(data[off : off + ns * 4], dtype=np.float32)
            off += ns * 4
            nv = int(np.prod(bm_meta["Vt_shape"]))
            Vt = np.frombuffer(data[off : off + nv * 4], dtype=np.float32).reshape(
                bm_meta["Vt_shape"]
            )
            off += nv * 4
            i, j = bm_meta["i"], bm_meta["j"]
            recon[i * bm : (i + 1) * bm, j * bn : (j + 1) * bn] = (
                U.astype(np.float64) * S.astype(np.float64)
            ) @ Vt.astype(np.float64)
        return recon.astype(np.float32)
