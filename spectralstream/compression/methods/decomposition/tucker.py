"""Tucker decomposition, Block Tucker, and Hierarchical Tucker methods."""

from __future__ import annotations

import gc
from typing import List, Tuple

import numpy as np

_SVD_LIMIT = 1024 * 1024


def _randomized_svd(X, n_components, n_oversamples=10, n_iter=3):
    """Memory-efficient randomized SVD for large unfoldings."""
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
    U = Q @ Ub[:, : min(n_components, len(s))]
    return U, s[: min(n_components, len(s))], Vt[: min(n_components, len(s)), :]


class TuckerDecomposition:
    """Tucker decomposition via HOSVD (Higher-Order SVD)."""

    name = "tucker_decomposition"
    category = "decomposition"

    def compress(
        self, tensor: np.ndarray, ranks: List[int] = None
    ) -> Tuple[bytes, dict]:
        t = np.asarray(tensor, dtype=np.float64)
        shape = t.shape
        ndim = len(shape)
        if ndim < 2:
            return tensor.astype(np.float16).tobytes(), {
                "original_shape": shape,
                "shape": shape,
                "passthrough": True,
            }
        if ranks is None:
            ranks = [max(1, min(s, 64)) for s in shape]
        ranks = [min(r, s) for r, s in zip(ranks, shape)]
        ranks = [max(1, r) for r in ranks]
        factors = []
        for d in range(ndim):
            unfolded = np.moveaxis(t, d, 0).reshape(shape[d], -1)
            target_rank = ranks[d]
            if unfolded.size > _SVD_LIMIT and min(unfolded.shape) > 512:
                U, _, _ = _randomized_svd(unfolded, target_rank)
            else:
                U, _, _ = np.linalg.svd(unfolded, full_matrices=False)
                U = U[:, :target_rank]
            factors.append(U.astype(np.float64))
        core = t.copy().astype(np.float64)
        for d in range(ndim):
            core = np.tensordot(core, factors[d].T, axes=([0], [1]))
            core = np.moveaxis(core, 0, d)
        factor_data = b"".join(f.astype(np.float32).tobytes() for f in factors)
        core_data = core.astype(np.float32).tobytes()
        data = factor_data + core_data
        meta = dict(
            shape=shape,
            ndim=ndim,
            ranks=ranks,
            factor_shapes=[list(f.shape) for f in factors],
            core_shape=list(core.shape),
        )
        del core, unfolded
        for f in factors:
            del f
        gc.collect()
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        if metadata.get("passthrough"):
            return (
                np.frombuffer(data, dtype=np.float16)
                .copy()
                .reshape(metadata["shape"])
                .astype(np.float32)
            )
        ndim = metadata["ndim"]
        factors = []
        off = 0
        for fs in metadata["factor_shapes"]:
            n = int(np.prod(fs))
            factors.append(
                np.frombuffer(data[off : off + n * 4], dtype=np.float32).reshape(fs)
            )
            off += n * 4
        ncore = int(np.prod(metadata["core_shape"]))
        core = np.frombuffer(data[off : off + ncore * 4], dtype=np.float32).reshape(
            metadata["core_shape"]
        )
        recon = core.astype(np.float64)
        for d in range(ndim):
            recon = np.tensordot(recon, factors[d].astype(np.float64), axes=([0], [1]))
            if d < ndim - 1:
                recon = np.moveaxis(recon, 0, d)
        return recon.reshape(metadata["shape"]).astype(np.float32)


class BlockTucker:
    """Block Term Tucker decomposition."""

    name = "block_tucker"
    category = "decomposition"

    def compress(
        self, tensor: np.ndarray, n_blocks: int = 4, rank_frac: float = 0.25
    ) -> Tuple[bytes, dict]:
        t = np.asarray(tensor, dtype=np.float64)
        if t.ndim < 2:
            return tensor.astype(np.float16).tobytes(), {
                "original_shape": t.shape,
                "shape": t.shape,
                "passthrough": True,
            }
        m, n = t.shape
        nb = max(2, min(n_blocks, min(m, n)))
        bm, bn = m // nb, n // nb
        blocks = []
        for i in range(nb):
            for j in range(nb):
                bi = t[i * bm : (i + 1) * bm, j * bn : (j + 1) * bn]
                r = max(1, int(min(bm, bn) * rank_frac))
                r = min(r, min(bm, bn) - 1)
                U, S, Vt = np.linalg.svd(bi, full_matrices=False)
                r = min(r, len(S))
                if r < 1:
                    r = 1
                blocks.append(
                    {
                        "U": U[:, :r].astype(np.float32),
                        "S": S[:r].astype(np.float32),
                        "Vt": Vt[:r, :].astype(np.float32),
                        "i": i,
                        "j": j,
                    }
                )
                del U, S, Vt
                gc.collect()
        data = b"".join(
            b["U"].tobytes() + b["S"].tobytes() + b["Vt"].tobytes() for b in blocks
        )
        meta = dict(
            shape=t.shape,
            n_blocks=nb,
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
        del blocks, t
        gc.collect()
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
        nb = metadata["n_blocks"]
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


class HierarchicalTucker:
    """Hierarchical Tucker via binary tree SVD."""

    name = "hierarchical_tucker"
    category = "decomposition"

    def compress(self, tensor: np.ndarray, rank: int = None) -> Tuple[bytes, dict]:
        t = np.asarray(tensor, dtype=np.float64)
        if t.ndim < 2:
            return tensor.astype(np.float16).tobytes(), {
                "original_shape": t.shape,
                "shape": t.shape,
                "passthrough": True,
            }
        m, n = t.shape
        if rank is None:
            rank = max(1, min(64, min(m, n) // 2))
        r = max(1, min(rank, min(m, n)))
        if t.size > _SVD_LIMIT and min(m, n) > 512:
            U, S, Vt = _randomized_svd(t, r)
        else:
            U, S, Vt = np.linalg.svd(t, full_matrices=False)
        r = min(r, len(S))
        U_r = U[:, :r].astype(np.float32)
        S_r = S[:r].astype(np.float32)
        Vt_r = Vt[:r, :].astype(np.float32)
        data = U_r.tobytes() + S_r.tobytes() + Vt_r.tobytes()
        meta = dict(
            shape=t.shape,
            rank=r,
            U_shape=list(U_r.shape),
            S_shape=list(S_r.shape),
            Vt_shape=list(Vt_r.shape),
        )
        del U, S, Vt, t
        gc.collect()
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
        nu = int(np.prod(metadata["U_shape"]))
        U = np.frombuffer(data[off : off + nu * 4], dtype=np.float32).reshape(
            metadata["U_shape"]
        )
        off += nu * 4
        ns = int(np.prod(metadata["S_shape"]))
        S = np.frombuffer(data[off : off + ns * 4], dtype=np.float32)
        off += ns * 4
        nv = int(np.prod(metadata["Vt_shape"]))
        Vt = np.frombuffer(data[off : off + nv * 4], dtype=np.float32).reshape(
            metadata["Vt_shape"]
        )
        recon = (U.astype(np.float64) * S.astype(np.float64)) @ Vt.astype(np.float64)
        return recon.reshape(metadata["shape"]).astype(np.float32)
