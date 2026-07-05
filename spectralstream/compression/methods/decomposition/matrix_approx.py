"""Matrix approximation methods: H-matrix, Nystrom, Random Features."""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from .svd_decomposition import nystrom_approximation


class HMatrix:
    """Hierarchical (H-) matrix approximation via recursive SVD."""

    name = "h_matrix"
    category = "decomposition"

    def compress(
        self, tensor: np.ndarray, block_size: int = 32, eps: float = 0.01
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
        bs = max(8, min(block_size, min(m, n)))
        off_diag_rank = max(1, int(min(m, n) * 0.05))
        diag_blocks = []
        off_blocks = []
        for i in range(0, m, bs):
            for j in range(0, n, bs):
                i_end = min(i + bs, m)
                j_end = min(j + bs, n)
                block = t[i:i_end, j:j_end]
                if i == j:
                    U, S, Vt = np.linalg.svd(block, full_matrices=False)
                    r = max(1, min(off_diag_rank * 2, len(S)))
                    diag_blocks.append(
                        {
                            "U": U[:, :r].astype(np.float32),
                            "S": S[:r].astype(np.float32),
                            "Vt": Vt[:r, :].astype(np.float32),
                            "i": i,
                            "j": j,
                            "ih": i_end,
                            "jh": j_end,
                        }
                    )
                else:
                    U, S, Vt = np.linalg.svd(block, full_matrices=False)
                    r = max(1, min(off_diag_rank, len(S)))
                    off_blocks.append(
                        {
                            "U": U[:, :r].astype(np.float32),
                            "S": S[:r].astype(np.float32),
                            "Vt": Vt[:r, :].astype(np.float32),
                            "i": i,
                            "j": j,
                            "ih": i_end,
                            "jh": j_end,
                        }
                    )
        all_data = []
        all_meta = []
        for b in diag_blocks + off_blocks:
            all_data.append(b["U"].tobytes() + b["S"].tobytes() + b["Vt"].tobytes())
            all_meta.append(
                {
                    "i": b["i"],
                    "j": b["j"],
                    "ih": b["ih"],
                    "jh": b["jh"],
                    "U_shape": list(b["U"].shape),
                    "S_shape": list(b["S"].shape),
                    "Vt_shape": list(b["Vt"].shape),
                }
            )
        data = b"".join(all_data)
        meta = dict(
            shape=t.shape,
            bs=bs,
            n_diag=len(diag_blocks),
            n_off=len(off_blocks),
            blocks=all_meta,
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
        off = 0
        for bm in metadata["blocks"]:
            nu = int(np.prod(bm["U_shape"]))
            U = np.frombuffer(data[off : off + nu * 4], dtype=np.float32).reshape(
                bm["U_shape"]
            )
            off += nu * 4
            ns = int(np.prod(bm["S_shape"]))
            S = np.frombuffer(data[off : off + ns * 4], dtype=np.float32)
            off += ns * 4
            nv = int(np.prod(bm["Vt_shape"]))
            Vt = np.frombuffer(data[off : off + nv * 4], dtype=np.float32).reshape(
                bm["Vt_shape"]
            )
            off += nv * 4
            i, j, ih, jh = bm["i"], bm["j"], bm["ih"], bm["jh"]
            recon[i:ih, j:jh] = (
                U.astype(np.float64) * S.astype(np.float64)
            ) @ Vt.astype(np.float64)
        return recon.astype(np.float32)


class Nystrom:
    """Nystrom approximation for symmetric PSD matrices."""

    name = "nystrom"
    category = "decomposition"

    def compress(self, tensor: np.ndarray, rank: int = None) -> Tuple[bytes, dict]:
        t = np.asarray(tensor, dtype=np.float64)
        if t.ndim < 2 or min(t.shape) < 4 or t.shape[0] != t.shape[1]:
            flat = t.ravel().astype(np.float32)
            return flat.astype(np.float16).tobytes(), {
                "original_shape": t.shape,
                "shape": t.shape,
                "passthrough": True,
            }
        if rank is None:
            rank = max(1, min(64, t.shape[0] // 4))
        sym = (t + t.T) / 2
        try:
            result, ratio, snr = nystrom_approximation(sym, rank)
        except (np.linalg.LinAlgError, ValueError):
            flat = t.ravel().astype(np.float32)
            return flat.astype(np.float16).tobytes(), {
                "original_shape": t.shape,
                "shape": t.shape,
                "passthrough": True,
            }
        recon = (
            result.get("C", np.zeros((1, 1))).astype(np.float64)
            @ result.get("W11_pinv", np.zeros((1, 1))).astype(np.float64)
            @ result.get("C", np.zeros((1, 1))).astype(np.float64).T
        )
        if recon.shape != t.shape:
            flat = t.ravel().astype(np.float32)
            return flat.astype(np.float16).tobytes(), {
                "original_shape": t.shape,
                "shape": t.shape,
                "passthrough": True,
            }
        rel_err = np.mean((t - recon) ** 2) / max(np.mean(t**2), 1e-30)
        if rel_err > 2.0:
            flat = t.ravel().astype(np.float32)
            return flat.astype(np.float16).tobytes(), {
                "original_shape": t.shape,
                "shape": t.shape,
                "passthrough": True,
            }
        data = result["C"].tobytes() + result["W11_pinv"].tobytes()
        meta = dict(
            shape=result["shape"],
            C_shape=list(result["C"].shape),
            W_shape=list(result["W11_pinv"].shape),
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
        nc = int(np.prod(metadata["C_shape"]))
        C = np.frombuffer(data[off : off + nc * 4], dtype=np.float32).reshape(
            metadata["C_shape"]
        )
        off += nc * 4
        nw = int(np.prod(metadata["W_shape"]))
        W = np.frombuffer(data[off : off + nw * 4], dtype=np.float32).reshape(
            metadata["W_shape"]
        )
        recon = C.astype(np.float64) @ W.astype(np.float64) @ C.astype(np.float64).T
        return recon.reshape(metadata["shape"]).astype(np.float32)


class RandomFeature:
    """Random Fourier feature map approximation."""

    name = "random_feature"
    category = "decomposition"

    def compress(
        self, tensor: np.ndarray, n_features: int = None
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
        if n_features is None:
            n_features = max(1, min(64, n // 4))
        nf = max(1, min(n_features, n))
        rng = np.random.RandomState(42)
        W = rng.randn(n, nf).astype(np.float64) * math.sqrt(2.0 / nf)
        b = rng.uniform(0, 2 * math.pi, size=nf).astype(np.float64)
        phi = np.cos(t @ W + b)
        try:
            coeffs, _, _, _ = np.linalg.lstsq(phi, t, rcond=None)
        except np.linalg.LinAlgError:
            flat = t.ravel().astype(np.float32)
            return flat.astype(np.float16).tobytes(), {
                "original_shape": t.shape,
                "shape": t.shape,
                "passthrough": True,
            }
        recon = phi @ coeffs
        data = phi.astype(np.float32).tobytes() + coeffs.astype(np.float32).tobytes()
        meta = dict(
            shape=t.shape,
            n_features=nf,
            m=m,
            n=n,
            phi_shape=list(phi.shape),
            coeffs_shape=list(coeffs.shape),
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
        off = 0
        nphi = int(np.prod(metadata["phi_shape"]))
        phi = np.frombuffer(data[off : off + nphi * 4], dtype=np.float32).reshape(
            metadata["phi_shape"]
        )
        off += nphi * 4
        nc = int(np.prod(metadata["coeffs_shape"]))
        coeffs = np.frombuffer(data[off : off + nc * 4], dtype=np.float32).reshape(
            metadata["coeffs_shape"]
        )
        recon = phi.astype(np.float64) @ coeffs.astype(np.float64)
        return recon.reshape(shape).astype(np.float32)
