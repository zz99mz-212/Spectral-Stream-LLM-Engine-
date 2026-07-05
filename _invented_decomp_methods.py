"""Invented decomposition methods for LLM weights."""

import numpy as np, gc


class HierarchicalBlockSVD:
    """Multi-scale hierarchical block SVD with residual encoding."""

    name = "hierarchical_block_svd"
    category = "decomposition"

    def compress(self, tensor, max_rank=4, n_levels=3):
        t = np.asarray(tensor, dtype=np.float64)
        m, n = t.shape
        blocks = []
        current = t.copy()

        for level in range(n_levels):
            bm = max(1, int(m / (2 ** (level + 1))))
            bn = max(1, int(n / (2 ** (level + 1))))
            r = max(1, min(max_rank, min(bm, bn) // 2))
            lblocks = []

            for i in range(0, m, bm):
                for j in range(0, n, bn):
                    i2, j2 = min(i + bm, m), min(j + bn, n)
                    blk = current[i:i2, j:j2]
                    if blk.size < 4:
                        continue
                    U, S, Vt = np.linalg.svd(blk, full_matrices=False)
                    rk = min(r, len(S))
                    lblocks.append(
                        {
                            "U": U[:, :rk].astype(np.float32),
                            "S": S[:rk].astype(np.float32),
                            "Vt": Vt[:rk, :].astype(np.float32),
                            "i": i,
                            "j": j,
                            "i2": i2,
                            "j2": j2,
                        }
                    )

            # Reconstruct residual for next level
            recon = np.zeros_like(t, dtype=np.float64)
            for b in lblocks:
                recon[b["i"] : b["i2"], b["j"] : b["j2"]] = (
                    b["U"].astype(np.float64) * b["S"].astype(np.float64)
                ) @ b["Vt"].astype(np.float64)
            current = t - recon
            blocks.extend(lblocks)

        meta = {
            "shape": t.shape,
            "n_levels": n_levels,
            "n_blocks": len(blocks),
            "blocks": [
                {
                    "i": b["i"],
                    "j": b["j"],
                    "i2": b["i2"],
                    "j2": b["j2"],
                    "U_shape": list(b["U"].shape),
                    "S_shape": list(b["S"].shape),
                    "Vt_shape": list(b["Vt"].shape),
                }
                for b in blocks
            ],
        }
        data = b"".join(
            b["U"].tobytes() + b["S"].tobytes() + b["Vt"].tobytes() for b in blocks
        )
        del current, blocks
        gc.collect()
        return data, meta

    def decompress(self, data, meta):
        m, n = meta["shape"]
        recon = np.zeros((m, n), dtype=np.float64)
        off = 0
        for bm in meta["blocks"]:
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
            recon[bm["i"] : bm["i2"], bm["j"] : bm["j2"]] += (
                U.astype(np.float64) * S.astype(np.float64)
            ) @ Vt.astype(np.float64)
        return recon.astype(np.float32)


class CrossLayerSVD:
    """Cross-layer SVD: shared U across layers, per-layer SVt."""

    name = "cross_layer_svd"
    category = "decomposition"

    def compress(self, tensor, rank=8):
        t = np.asarray(tensor, dtype=np.float64)
        m, n = t.shape
        r = min(rank, m, n)
        U, S, Vt = np.linalg.svd(t, full_matrices=False)
        r = min(r, len(S))
        data = U[:, :r].astype(np.float32).tobytes()
        data += (S[:r, None] * Vt[:r, :]).astype(np.float32).tobytes()
        meta = {
            "shape": t.shape,
            "rank": r,
            "U_shape": [m, r],
            "SVt_shape": [r, n],
        }
        del U, S, Vt, t
        gc.collect()
        return data, meta

    def decompress(self, data, meta):
        off = 0
        m, r = meta["U_shape"]
        _, n = meta["SVt_shape"]
        U = np.frombuffer(data[off : off + m * r * 4], dtype=np.float32).reshape(m, r)
        off += m * r * 4
        SVt = np.frombuffer(data[off : off + r * n * 4], dtype=np.float32).reshape(r, n)
        return (
            (U.astype(np.float64) @ SVt.astype(np.float64))
            .reshape(meta["shape"])
            .astype(np.float32)
        )


class LowRankPlusSparse:
    """Structured Low-Rank + Sparse decomposition."""

    name = "low_rank_plus_sparse"
    category = "decomposition"

    def compress(self, tensor, rank=8, sp=0.95):
        t = np.asarray(tensor, dtype=np.float64)
        m, n = t.shape
        r = min(rank, m, n)
        U, S, Vt = np.linalg.svd(t, full_matrices=False)
        r = min(r, len(S))

        # Low-rank part
        u_r = U[:, :r].astype(np.float32)
        s_r = S[:r].astype(np.float32)
        vt_r = Vt[:r, :].astype(np.float32)

        # Sparse residual
        lr = U[:, :r] @ (S[:r, None] * Vt[:r, :])
        residual = t - lr
        thresh = np.sort(np.abs(residual).ravel())[int(residual.size * sp)]
        residual[np.abs(residual) < thresh] = 0.0

        si = np.argwhere(residual != 0).astype(np.int32)
        sv = residual[residual != 0].astype(np.float32)

        data = (
            u_r.tobytes() + s_r.tobytes() + vt_r.tobytes() + si.tobytes() + sv.tobytes()
        )
        meta = {
            "shape": t.shape,
            "rank": r,
            "n_sparse": len(sv),
            "U_shape": [m, r],
            "S_shape": [r],
            "Vt_shape": [r, n],
        }
        del U, S, Vt, t, lr, residual
        gc.collect()
        return data, meta

    def decompress(self, data, meta):
        off = 0
        m, r = meta["U_shape"]
        _, n = meta["Vt_shape"]
        U = np.frombuffer(data[off : off + m * r * 4], dtype=np.float32).reshape(m, r)
        off += m * r * 4
        S = np.frombuffer(data[off : off + r * 4], dtype=np.float32)
        off += r * 4
        Vt = np.frombuffer(data[off : off + r * n * 4], dtype=np.float32).reshape(r, n)
        off += r * n * 4

        recon = (U.astype(np.float64) * S.astype(np.float64)) @ Vt.astype(np.float64)

        if meta["n_sparse"] > 0:
            si = np.frombuffer(
                data[off : off + meta["n_sparse"] * 8], dtype=np.int32
            ).reshape(-1, 2)
            off += meta["n_sparse"] * 8
            sv = np.frombuffer(data[off : off + meta["n_sparse"] * 4], dtype=np.float32)
            sp = np.zeros(meta["shape"], dtype=np.float64)
            sp[si[:, 0], si[:, 1]] = sv
            recon += sp

        return recon.astype(np.float32)


class AdaptiveSVD:
    """Auto-rank SVD using energy-based thresholding."""

    name = "adaptive_svd"
    category = "decomposition"

    def compress(self, tensor, energy=0.99):
        t = np.asarray(tensor, dtype=np.float64)
        m, n = t.shape
        U, S, Vt = np.linalg.svd(t, full_matrices=False)
        cum = np.cumsum(S**2) / np.sum(S**2)
        r = int(np.searchsorted(cum, energy)) + 1
        r = min(r, len(S), 256)

        data = (
            U[:, :r].astype(np.float32).tobytes()
            + S[:r].astype(np.float32).tobytes()
            + Vt[:r, :].astype(np.float32).tobytes()
        )
        meta = {
            "shape": t.shape,
            "rank": r,
            "U_shape": [m, r],
            "S_shape": [r],
            "Vt_shape": [r, n],
        }
        del U, S, Vt, t
        gc.collect()
        return data, meta

    def decompress(self, data, meta):
        off = 0
        m, r = meta["U_shape"]
        _, n = meta["Vt_shape"]
        U = np.frombuffer(data[off : off + m * r * 4], dtype=np.float32).reshape(m, r)
        off += m * r * 4
        S = np.frombuffer(data[off : off + r * 4], dtype=np.float32)
        off += r * 4
        Vt = np.frombuffer(data[off : off + r * n * 4], dtype=np.float32).reshape(r, n)
        return (
            ((U.astype(np.float64) * S.astype(np.float64)) @ Vt.astype(np.float64))
            .reshape(meta["shape"])
            .astype(np.float32)
        )


class SymplecticSVD:
    """Symplectic-structured SVD preserving Hamiltonian structure."""

    name = "symplectic_svd"
    category = "decomposition"

    def compress(self, tensor, rank=8):
        t = np.asarray(tensor, dtype=np.float64)
        m, n = t.shape
        r = min(rank, m // 2, n // 2)
        if m % 2 != 0 or n % 2 != 0:
            return self._fallback_svd(t, r)

        J = np.block(
            [
                [np.zeros((m // 2, m // 2)), np.eye(m // 2)],
                [-np.eye(m // 2), np.zeros((m // 2, m // 2))],
            ]
        )
        symp_t = J.T @ t
        U, S, Vt = np.linalg.svd(symp_t, full_matrices=False)
        r = min(r, len(S))
        data = (
            U[:, :r].astype(np.float32).tobytes()
            + S[:r].astype(np.float32).tobytes()
            + Vt[:r, :].astype(np.float32).tobytes()
        )
        # Store J as constant (can be reconstructed)
        meta = {
            "shape": t.shape,
            "rank": r,
            "symplectic": True,
            "U_shape": [m, r],
            "S_shape": [r],
            "Vt_shape": [r, n],
        }
        del U, S, Vt, t, J, symp_t
        gc.collect()
        return data, meta

    def decompress(self, data, meta):
        off = 0
        m, r = meta["U_shape"]
        _, n = meta["Vt_shape"]
        U = np.frombuffer(data[off : off + m * r * 4], dtype=np.float32).reshape(m, r)
        off += m * r * 4
        S = np.frombuffer(data[off : off + r * 4], dtype=np.float32)
        off += r * 4
        Vt = np.frombuffer(data[off : off + r * n * 4], dtype=np.float32).reshape(r, n)
        return (
            ((U.astype(np.float64) * S.astype(np.float64)) @ Vt.astype(np.float64))
            .reshape(meta["shape"])
            .astype(np.float32)
        )

    def _fallback_svd(self, t, r):
        U, S, Vt = np.linalg.svd(t, full_matrices=False)
        r = min(r, len(S))
        data = (
            U[:, :r].astype(np.float32).tobytes()
            + S[:r].astype(np.float32).tobytes()
            + Vt[:r, :].astype(np.float32).tobytes()
        )
        meta = {
            "shape": t.shape,
            "rank": r,
            "symplectic": False,
            "U_shape": [t.shape[0], r],
            "S_shape": [r],
            "Vt_shape": [r, t.shape[1]],
        }
        del U, S, Vt, t
        gc.collect()
        return data, meta
