"""Advanced Matrix Factorization Techniques.

20 advanced factorization methods that go beyond basic SVD to compress
high-rank matrices while maintaining full FP32 precision.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np


class SVDResidualDecompose:
    """SVD + Residual Compression: two-pass SVD with sparse residual."""

    METHOD_NAME = "svd_residual"
    category = "decomposition"

    def __init__(
        self,
        base_rank: int = 32,
        residual_rank: int = 16,
        residual_sparsity: float = 0.95,
    ):
        self.base_rank = base_rank
        self.residual_rank = residual_rank
        self.residual_sparsity = residual_sparsity

    def compress(
        self, tensor: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        base_rank = kwargs.get("base_rank", self.base_rank)
        residual_rank = kwargs.get("residual_rank", self.residual_rank)
        orig_shape = tensor.shape
        mat = tensor.astype(np.float64).reshape(-1, tensor.shape[-1])

        u, s, vh = np.linalg.svd(mat, full_matrices=False)
        r1 = min(base_rank, len(s))
        base = (u[:, :r1] * s[:r1]) @ vh[:r1, :]
        residual = mat - base

        ur, sr, vhr = np.linalg.svd(residual, full_matrices=False)
        r2 = min(residual_rank, len(sr))
        res_base = (ur[:, :r2] * sr[:r2]) @ vhr[:r2, :]
        res_error = residual - res_base

        threshold = (
            np.sort(np.abs(res_error.ravel()))[
                int(self.residual_sparsity * res_error.size)
            ]
            if res_error.size > 0
            else 0
        )
        sparse_mask = np.abs(res_error) > threshold

        data = {
            "U": u[:, :r1].astype(np.float32),
            "S": s[:r1].astype(np.float32),
            "Vt": vh[:r1, :].astype(np.float32),
            "Ur": ur[:, :r2].astype(np.float32),
            "Sr": sr[:r2].astype(np.float32),
            "Vhr": vhr[:r2, :].astype(np.float32),
            "res_vals": res_error[sparse_mask].astype(np.float32),
            "res_idx": np.argwhere(sparse_mask).astype(np.int32),
        }
        meta = {"orig_shape": orig_shape, "method": self.METHOD_NAME, "ranks": (r1, r2)}
        return data, meta

    def decompress(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> np.ndarray:
        base = (data["U"] * data["S"][np.newaxis, :]) @ data["Vt"]
        res = (data["Ur"] * data["Sr"][np.newaxis, :]) @ data["Vhr"]
        sparse = np.zeros_like(base)
        idx = data["res_idx"]
        sparse[idx[:, 0], idx[:, 1]] = data["res_vals"]
        return (base + res + sparse).reshape(metadata["orig_shape"])

    def estimate_ratio(self, tensor: np.ndarray, **kwargs) -> float:
        base_rank = kwargs.get("base_rank", self.base_rank)
        residual_rank = kwargs.get("residual_rank", self.residual_rank)
        orig = tensor.nbytes
        n, m = tensor.shape[0], tensor.shape[-1]
        comp = (
            base_rank * (n + m + base_rank) * 4
            + residual_rank * (n + m + residual_rank) * 4
        )
        return comp / max(orig, 1)

    def estimate_error(self, tensor: np.ndarray, **kwargs) -> dict:
        data, meta = self.compress(tensor, **kwargs)
        recon = self.decompress(data, meta)
        orig = tensor.astype(np.float64).ravel()
        rec = recon.astype(np.float64).ravel()
        mse = float(np.mean((orig - rec) ** 2))
        snr = 10 * np.log10(np.sum(orig**2) / (np.sum((orig - rec) ** 2) + 1e-30))
        cos_sim = float(
            np.dot(orig, rec) / (np.linalg.norm(orig) * np.linalg.norm(rec) + 1e-30)
        )
        return {
            "mse": mse,
            "snr_db": float(snr),
            "cosine_similarity": cos_sim,
            "rel_error": float(np.mean(np.abs(orig - rec) / (np.abs(orig) + 1e-10))),
        }


class CURFullDecompose:
    """CUR Decomposition: C x U x R with leverage score sampling."""

    METHOD_NAME = "cur_full"
    category = "decomposition"

    def __init__(self, rank: int = 32):
        self.rank = rank

    def compress(
        self, tensor: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        rank = kwargs.get("rank", self.rank)
        orig_shape = tensor.shape
        mat = tensor.astype(np.float64).reshape(-1, tensor.shape[-1])
        n, m = mat.shape

        u, s, vh = np.linalg.svd(mat, full_matrices=False)
        k = min(rank, len(s))

        col_norms = np.sum(u[:, :k] ** 2, axis=1)
        row_norms = np.sum(vh[:k, :].T ** 2, axis=1)

        c = min(k * 2, m)
        r = min(k * 2, n)

        rng = np.random.RandomState(42)
        col_idx = rng.choice(
            m, size=c, replace=False, p=col_norms / (col_norms.sum() + 1e-30)
        )
        row_idx = rng.choice(
            n, size=r, replace=False, p=row_norms / (row_norms.sum() + 1e-30)
        )

        C = mat[:, col_idx]
        R = mat[row_idx, :]
        W = mat[np.ix_(row_idx, col_idx)]

        Uw, sw, Vhw = np.linalg.svd(W, full_matrices=False)
        r_inv = min(k, len(sw), c, r)
        W_pinv = (Vhw[:r_inv, :].T * (1.0 / (sw[:r_inv] + 1e-10))) @ Uw[:, :r_inv].T

        data = {
            "C": C.astype(np.float32),
            "R": R.astype(np.float32),
            "W_pinv": W_pinv.astype(np.float32),
            "col_idx": col_idx.astype(np.int32),
            "row_idx": row_idx.astype(np.int32),
        }
        meta = {"orig_shape": orig_shape, "method": self.METHOD_NAME, "rank": r_inv}
        return data, meta

    def decompress(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> np.ndarray:
        return (data["C"] @ data["W_pinv"] @ data["R"]).reshape(metadata["orig_shape"])

    def estimate_ratio(self, tensor: np.ndarray, **kwargs) -> float:
        rank = kwargs.get("rank", self.rank)
        orig = tensor.nbytes
        c = min(rank * 2, tensor.shape[1])
        r = min(rank * 2, tensor.shape[0])
        comp = (tensor.shape[0] * c + tensor.shape[1] * r + r * c) * 4
        return comp / max(orig, 1)

    def estimate_error(self, tensor: np.ndarray, **kwargs) -> dict:
        data, meta = self.compress(tensor, **kwargs)
        recon = self.decompress(data, meta)
        orig = tensor.astype(np.float64).ravel()
        rec = recon.astype(np.float64).ravel()
        mse = float(np.mean((orig - rec) ** 2))
        snr = 10 * np.log10(np.sum(orig**2) / (np.sum((orig - rec) ** 2) + 1e-30))
        cos_sim = float(
            np.dot(orig, rec) / (np.linalg.norm(orig) * np.linalg.norm(rec) + 1e-30)
        )
        return {
            "mse": mse,
            "snr_db": float(snr),
            "cosine_similarity": cos_sim,
            "rel_error": float(np.mean(np.abs(orig - rec) / (np.abs(orig) + 1e-10))),
        }


class NystromAdvancedDecompose:
    """Advanced Nystrom approximation with leverage score sampling."""

    METHOD_NAME = "nystrom_advanced"
    category = "decomposition"

    def __init__(self, n_landmarks: int = 64, n_projections: int = 3):
        self.n_landmarks = n_landmarks
        self.n_projections = n_projections

    def compress(
        self, tensor: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        n_landmarks = kwargs.get("n_landmarks", self.n_landmarks)
        orig_shape = tensor.shape
        mat = tensor.astype(np.float64).reshape(-1, tensor.shape[-1])
        n, m = mat.shape

        k = min(n_landmarks, n, m)

        u_svd, s_svd, vh_svd = np.linalg.svd(mat, full_matrices=False)
        r_lev = min(k, len(s_svd))
        col_scores = np.sum(u_svd[:, :r_lev] ** 2, axis=1)
        row_scores = np.sum(vh_svd[:r_lev, :].T ** 2, axis=1)

        rng = np.random.RandomState(42)
        col_idx = rng.choice(
            m, size=k, replace=False, p=col_scores / (col_scores.sum() + 1e-30)
        )
        row_idx = rng.choice(
            n, size=k, replace=False, p=row_scores / (row_scores.sum() + 1e-30)
        )

        C = mat[:, col_idx]
        R = mat[row_idx, :]
        W = mat[np.ix_(row_idx, col_idx)]

        Uw, sw, Vhw = np.linalg.svd(W, full_matrices=False)
        r = min(k, len(sw), W.shape[0], W.shape[1])
        W_pinv = (Vhw[:r, :].T * (1.0 / (sw[:r] + 1e-10))) @ Uw[:, :r].T

        data = {
            "C": C.astype(np.float32),
            "R": R.astype(np.float32),
            "W_pinv": W_pinv.astype(np.float32),
            "col_idx": col_idx.astype(np.int32),
            "row_idx": row_idx.astype(np.int32),
        }
        meta = {"orig_shape": orig_shape, "method": self.METHOD_NAME}
        return data, meta

    def decompress(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> np.ndarray:
        return (data["C"] @ data["W_pinv"] @ data["R"]).reshape(metadata["orig_shape"])

    def estimate_ratio(self, tensor: np.ndarray, **kwargs) -> float:
        n_landmarks = kwargs.get("n_landmarks", self.n_landmarks)
        orig = tensor.nbytes
        k = min(n_landmarks, tensor.shape[0], tensor.shape[1])
        comp = (tensor.shape[0] * k + tensor.shape[1] * k + k * k) * 4
        return comp / max(orig, 1)

    def estimate_error(self, tensor: np.ndarray, **kwargs) -> dict:
        data, meta = self.compress(tensor, **kwargs)
        recon = self.decompress(data, meta)
        orig = tensor.astype(np.float64).ravel()
        rec = recon.astype(np.float64).ravel()
        mse = float(np.mean((orig - rec) ** 2))
        snr = 10 * np.log10(np.sum(orig**2) / (np.sum((orig - rec) ** 2) + 1e-30))
        cos_sim = float(
            np.dot(orig, rec) / (np.linalg.norm(orig) * np.linalg.norm(rec) + 1e-30)
        )
        return {
            "mse": mse,
            "snr_db": float(snr),
            "cosine_similarity": cos_sim,
            "rel_error": float(np.mean(np.abs(orig - rec) / (np.abs(orig) + 1e-10))),
        }


class RandomFeatureAdvancedDecompose:
    """Random Feature Approximation with multiple random bases."""

    METHOD_NAME = "random_feature_advanced"
    category = "decomposition"

    def __init__(self, n_features: int = 128, n_bases: int = 3, seed: int = 42):
        self.n_features = n_features
        self.n_bases = n_bases
        self.seed = seed

    def compress(
        self, tensor: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        n_features = kwargs.get("n_features", self.n_features)
        orig_shape = tensor.shape
        mat = tensor.astype(np.float64).reshape(-1, tensor.shape[-1])
        n, d = mat.shape

        rng = np.random.RandomState(self.seed)
        bases = []
        phi_list = []

        for _ in range(self.n_bases):
            W = rng.randn(d, n_features) * np.sqrt(2.0 / n_features)
            b = rng.uniform(0, 2 * np.pi, n_features)
            phi = np.cos(mat @ W + b)
            phi_list.append(phi)
            bases.append({"W": W.astype(np.float32), "b": b.astype(np.float32)})

        phi_full = np.hstack(phi_list)
        A, residuals, rank, sv = np.linalg.lstsq(phi_full, mat, rcond=None)

        data = {
            "phi": phi_full.astype(np.float32),
            "A": A.astype(np.float32),
            "bases": bases,
        }
        meta = {
            "orig_shape": orig_shape,
            "method": self.METHOD_NAME,
            "n_features": n_features * self.n_bases,
        }
        return data, meta

    def decompress(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> np.ndarray:
        return (data["phi"] @ data["A"]).reshape(metadata["orig_shape"])

    def estimate_ratio(self, tensor: np.ndarray, **kwargs) -> float:
        n_features = kwargs.get("n_features", self.n_features)
        orig = tensor.nbytes
        nf = n_features * self.n_bases
        comp = (
            tensor.shape[0] * nf * 4
            + tensor.shape[1] * nf * 4
            + tensor.shape[1] * nf * 4
        )
        return comp / max(orig, 1)

    def estimate_error(self, tensor: np.ndarray, **kwargs) -> dict:
        data, meta = self.compress(tensor, **kwargs)
        recon = self.decompress(data, meta)
        orig = tensor.astype(np.float64).ravel()
        rec = recon.astype(np.float64).ravel()
        mse = float(np.mean((orig - rec) ** 2))
        snr = 10 * np.log10(np.sum(orig**2) / (np.sum((orig - rec) ** 2) + 1e-30))
        cos_sim = float(
            np.dot(orig, rec) / (np.linalg.norm(orig) * np.linalg.norm(rec) + 1e-30)
        )
        return {
            "mse": mse,
            "snr_db": float(snr),
            "cosine_similarity": cos_sim,
            "rel_error": float(np.mean(np.abs(orig - rec) / (np.abs(orig) + 1e-10))),
        }


class BlockSVDDecompose:
    """Block SVD: independent SVD per block with adaptive rank."""

    METHOD_NAME = "block_svd"
    category = "decomposition"

    def __init__(
        self,
        block_size: int = 64,
        max_rank_per_block: int = 8,
        energy_threshold: float = 0.99,
    ):
        self.block_size = block_size
        self.max_rank_per_block = max_rank_per_block
        self.energy_threshold = energy_threshold

    def compress(
        self, tensor: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        block_size = kwargs.get("block_size", self.block_size)
        orig_shape = tensor.shape
        mat = tensor.astype(np.float64).reshape(-1, tensor.shape[-1])
        n, m = mat.shape

        bs = min(block_size, n, m)
        n_blocks_r = (n + bs - 1) // bs
        n_blocks_c = (m + bs - 1) // bs

        blocks = []
        for i in range(n_blocks_r):
            for j in range(n_blocks_c):
                r_s, c_s = i * bs, j * bs
                block = mat[r_s : r_s + bs, c_s : c_s + bs]
                if block.size == 0:
                    continue

                u, s, vh = np.linalg.svd(block, full_matrices=False)
                total_energy = np.sum(s**2)
                cumsum = np.cumsum(s**2) / (total_energy + 1e-30)
                r = int(np.searchsorted(cumsum, self.energy_threshold)) + 1
                r = max(1, min(r, self.max_rank_per_block, len(s)))

                blocks.append(
                    {
                        "u": u[:, :r].astype(np.float32),
                        "s": s[:r].astype(np.float32),
                        "vh": vh[:r, :].astype(np.float32),
                        "pos": np.array([i, j], dtype=np.int32),
                        "block_shape": np.array(block.shape, dtype=np.int32),
                    }
                )

        data = {
            "blocks": blocks,
            "grid": np.array([n_blocks_r, n_blocks_c], dtype=np.int32),
            "block_size": np.int32(bs),
            "shape": np.array([n, m], dtype=np.int32),
        }
        meta = {"orig_shape": orig_shape, "method": self.METHOD_NAME}
        return data, meta

    def decompress(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> np.ndarray:
        n, m = int(data["shape"][0]), int(data["shape"][1])
        bs = int(data["block_size"])
        result = np.zeros((n, m), dtype=np.float32)
        for b in data["blocks"]:
            i, j = int(b["pos"][0]), int(b["pos"][1])
            block = (b["u"] * b["s"][np.newaxis, :]) @ b["vh"]
            r_s, c_s = i * bs, j * bs
            bh, bw = block.shape
            result[r_s : r_s + bh, c_s : c_s + bw] = block
        return result.reshape(metadata["orig_shape"])

    def estimate_ratio(self, tensor: np.ndarray, **kwargs) -> float:
        block_size = kwargs.get("block_size", self.block_size)
        orig = tensor.nbytes
        bs = min(block_size, tensor.shape[0], tensor.shape[1])
        n_blocks = ((tensor.shape[0] + bs - 1) // bs) * (
            (tensor.shape[1] + bs - 1) // bs
        )
        comp = (
            n_blocks * self.max_rank_per_block * (bs + bs + self.max_rank_per_block) * 4
        )
        return comp / max(orig, 1)

    def estimate_error(self, tensor: np.ndarray, **kwargs) -> dict:
        data, meta = self.compress(tensor, **kwargs)
        recon = self.decompress(data, meta)
        orig = tensor.astype(np.float64).ravel()
        rec = recon.astype(np.float64).ravel()
        mse = float(np.mean((orig - rec) ** 2))
        snr = 10 * np.log10(np.sum(orig**2) / (np.sum((orig - rec) ** 2) + 1e-30))
        cos_sim = float(
            np.dot(orig, rec) / (np.linalg.norm(orig) * np.linalg.norm(rec) + 1e-30)
        )
        return {
            "mse": mse,
            "snr_db": float(snr),
            "cosine_similarity": cos_sim,
            "rel_error": float(np.mean(np.abs(orig - rec) / (np.abs(orig) + 1e-10))),
        }


class TiledLowRankDecompose:
    """Tiled Low-Rank: tiles with independent low-rank approximation."""

    METHOD_NAME = "tiled_low_rank"
    category = "decomposition"

    def __init__(self, tile_size: int = 64, rank: int = 8, overlap: int = 0):
        self.tile_size = tile_size
        self.rank = rank
        self.overlap = overlap

    def compress(
        self, tensor: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        tile_size = kwargs.get("tile_size", self.tile_size)
        rank = kwargs.get("rank", self.rank)
        orig_shape = tensor.shape
        mat = tensor.astype(np.float64).reshape(-1, tensor.shape[-1])
        n, m = mat.shape

        ts = min(tile_size, n, m)
        step = max(1, ts - self.overlap)

        tiles = []
        for i in range(0, n, step):
            for j in range(0, m, step):
                tile = mat[i : i + ts, j : j + ts]
                if tile.size == 0:
                    continue

                u, s, vh = np.linalg.svd(tile, full_matrices=False)
                r = min(rank, len(s))
                tiles.append(
                    {
                        "u": u[:, :r].astype(np.float32),
                        "s": s[:r].astype(np.float32),
                        "vh": vh[:r, :].astype(np.float32),
                        "pos": np.array([i, j], dtype=np.int32),
                        "tile_shape": np.array(tile.shape, dtype=np.int32),
                    }
                )

        data = {
            "tiles": tiles,
            "tile_size": np.int32(ts),
            "step": np.int32(step),
            "shape": np.array([n, m], dtype=np.int32),
        }
        meta = {"orig_shape": orig_shape, "method": self.METHOD_NAME}
        return data, meta

    def decompress(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> np.ndarray:
        n, m = int(data["shape"][0]), int(data["shape"][1])
        result = np.zeros((n, m), dtype=np.float32)
        count = np.zeros((n, m), dtype=np.float32)
        for t in data["tiles"]:
            i, j = int(t["pos"][0]), int(t["pos"][1])
            tile = (t["u"] * t["s"][np.newaxis, :]) @ t["vh"]
            th, tw = tile.shape
            result[i : i + th, j : j + tw] += tile
            count[i : i + th, j : j + tw] += 1.0
        count[count == 0] = 1.0
        return (result / count).reshape(metadata["orig_shape"])

    def estimate_ratio(self, tensor: np.ndarray, **kwargs) -> float:
        tile_size = kwargs.get("tile_size", self.tile_size)
        rank = kwargs.get("rank", self.rank)
        orig = tensor.nbytes
        ts = min(tile_size, tensor.shape[0], tensor.shape[1])
        step = max(1, ts - self.overlap)
        n_tiles = ((tensor.shape[0] + step - 1) // step) * (
            (tensor.shape[1] + step - 1) // step
        )
        comp = n_tiles * rank * (ts + ts + rank) * 4
        return comp / max(orig, 1)

    def estimate_error(self, tensor: np.ndarray, **kwargs) -> dict:
        data, meta = self.compress(tensor, **kwargs)
        recon = self.decompress(data, meta)
        orig = tensor.astype(np.float64).ravel()
        rec = recon.astype(np.float64).ravel()
        mse = float(np.mean((orig - rec) ** 2))
        snr = 10 * np.log10(np.sum(orig**2) / (np.sum((orig - rec) ** 2) + 1e-30))
        cos_sim = float(
            np.dot(orig, rec) / (np.linalg.norm(orig) * np.linalg.norm(rec) + 1e-30)
        )
        return {
            "mse": mse,
            "snr_db": float(snr),
            "cosine_similarity": cos_sim,
            "rel_error": float(np.mean(np.abs(orig - rec) / (np.abs(orig) + 1e-10))),
        }


class ProgressiveSVDDecompose:
    """Progressive SVD: add ranks until error threshold is met."""

    METHOD_NAME = "progressive_svd"
    category = "decomposition"

    def __init__(self, max_rank: int = 128, error_threshold: float = 0.01):
        self.max_rank = max_rank
        self.error_threshold = error_threshold

    def compress(
        self, tensor: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        max_rank = kwargs.get("max_rank", self.max_rank)
        error_threshold = kwargs.get("error_threshold", self.error_threshold)
        orig_shape = tensor.shape
        mat = tensor.astype(np.float64).reshape(-1, tensor.shape[-1])

        u, s, vh = np.linalg.svd(mat, full_matrices=False)
        total_norm = np.linalg.norm(mat)
        used_rank = 0

        for r in range(1, min(max_rank, len(s)) + 1):
            recon = (u[:, :r] * s[:r]) @ vh[:r, :]
            error = np.linalg.norm(mat - recon) / (total_norm + 1e-30)
            if error < error_threshold:
                used_rank = r
                break
        else:
            used_rank = min(max_rank, len(s))

        data = {
            "U": u[:, :used_rank].astype(np.float32),
            "S": s[:used_rank].astype(np.float32),
            "Vt": vh[:used_rank, :].astype(np.float32),
        }
        meta = {"orig_shape": orig_shape, "method": self.METHOD_NAME, "rank": used_rank}
        return data, meta

    def decompress(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> np.ndarray:
        return ((data["U"] * data["S"][np.newaxis, :]) @ data["Vt"]).reshape(
            metadata["orig_shape"]
        )

    def estimate_ratio(self, tensor: np.ndarray, **kwargs) -> float:
        max_rank = kwargs.get("max_rank", self.max_rank)
        orig = tensor.nbytes
        comp = max_rank * (tensor.shape[0] + tensor.shape[-1] + max_rank) * 4
        return comp / max(orig, 1)

    def estimate_error(self, tensor: np.ndarray, **kwargs) -> dict:
        data, meta = self.compress(tensor, **kwargs)
        recon = self.decompress(data, meta)
        orig = tensor.astype(np.float64).ravel()
        rec = recon.astype(np.float64).ravel()
        mse = float(np.mean((orig - rec) ** 2))
        snr = 10 * np.log10(np.sum(orig**2) / (np.sum((orig - rec) ** 2) + 1e-30))
        cos_sim = float(
            np.dot(orig, rec) / (np.linalg.norm(orig) * np.linalg.norm(rec) + 1e-30)
        )
        return {
            "mse": mse,
            "snr_db": float(snr),
            "cosine_similarity": cos_sim,
            "rel_error": float(np.mean(np.abs(orig - rec) / (np.abs(orig) + 1e-10))),
        }


class IncrementalSVDDecompose:
    """Incremental SVD: base SVD + delta updates."""

    METHOD_NAME = "incremental_svd"
    category = "decomposition"

    def __init__(self, base_rank: int = 32, delta_rank: int = 8):
        self.base_rank = base_rank
        self.delta_rank = delta_rank

    def compress(
        self, tensor: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        base_rank = kwargs.get("base_rank", self.base_rank)
        delta_rank = kwargs.get("delta_rank", self.delta_rank)
        orig_shape = tensor.shape
        mat = tensor.astype(np.float64).reshape(-1, tensor.shape[-1])
        n, m = mat.shape

        u, s, vh = np.linalg.svd(mat, full_matrices=False)
        r1 = min(base_rank, len(s))
        base = (u[:, :r1] * s[:r1]) @ vh[:r1, :]
        delta = mat - base

        ud, sd, vhd = np.linalg.svd(delta, full_matrices=False)
        r2 = min(delta_rank, len(sd))

        data = {
            "U": u[:, :r1].astype(np.float32),
            "S": s[:r1].astype(np.float32),
            "Vt": vh[:r1, :].astype(np.float32),
            "Ud": ud[:, :r2].astype(np.float32),
            "Sd": sd[:r2].astype(np.float32),
            "Vtd": vhd[:r2, :].astype(np.float32),
        }
        meta = {"orig_shape": orig_shape, "method": self.METHOD_NAME, "ranks": (r1, r2)}
        return data, meta

    def decompress(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> np.ndarray:
        base = (data["U"] * data["S"][np.newaxis, :]) @ data["Vt"]
        delta = (data["Ud"] * data["Sd"][np.newaxis, :]) @ data["Vtd"]
        return (base + delta).reshape(metadata["orig_shape"])

    def estimate_ratio(self, tensor: np.ndarray, **kwargs) -> float:
        base_rank = kwargs.get("base_rank", self.base_rank)
        delta_rank = kwargs.get("delta_rank", self.delta_rank)
        orig = tensor.nbytes
        n, m = tensor.shape[0], tensor.shape[-1]
        comp = (
            base_rank * (n + m + base_rank) * 4 + delta_rank * (n + m + delta_rank) * 4
        )
        return comp / max(orig, 1)

    def estimate_error(self, tensor: np.ndarray, **kwargs) -> dict:
        data, meta = self.compress(tensor, **kwargs)
        recon = self.decompress(data, meta)
        orig = tensor.astype(np.float64).ravel()
        rec = recon.astype(np.float64).ravel()
        mse = float(np.mean((orig - rec) ** 2))
        snr = 10 * np.log10(np.sum(orig**2) / (np.sum((orig - rec) ** 2) + 1e-30))
        cos_sim = float(
            np.dot(orig, rec) / (np.linalg.norm(orig) * np.linalg.norm(rec) + 1e-30)
        )
        return {
            "mse": mse,
            "snr_db": float(snr),
            "cosine_similarity": cos_sim,
            "rel_error": float(np.mean(np.abs(orig - rec) / (np.abs(orig) + 1e-10))),
        }


class SparseSVDDecompose:
    """Sparse SVD: sparse U and Vt stored as indices + values."""

    METHOD_NAME = "sparse_svd"
    category = "decomposition"

    def __init__(self, rank: int = 32, sparsity: float = 0.9):
        self.rank = rank
        self.sparsity = sparsity

    def compress(
        self, tensor: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        rank = kwargs.get("rank", self.rank)
        sparsity = kwargs.get("sparsity", self.sparsity)
        orig_shape = tensor.shape
        mat = tensor.astype(np.float64).reshape(-1, tensor.shape[-1])

        u, s, vh = np.linalg.svd(mat, full_matrices=False)
        r = min(rank, len(s))

        u_r = u[:, :r]
        vh_r = vh[:r, :]

        u_threshold = (
            np.sort(np.abs(u_r.ravel()))[int(sparsity * u_r.size)]
            if u_r.size > 0
            else 0
        )
        vh_threshold = (
            np.sort(np.abs(vh_r.ravel()))[int(sparsity * vh_r.size)]
            if vh_r.size > 0
            else 0
        )

        u_mask = np.abs(u_r) > u_threshold
        vh_mask = np.abs(vh_r) > vh_threshold

        data = {
            "U_vals": u_r[u_mask].astype(np.float32),
            "U_idx": np.argwhere(u_mask).astype(np.int32),
            "S": s[:r].astype(np.float32),
            "Vt_vals": vh_r[vh_mask].astype(np.float32),
            "Vt_idx": np.argwhere(vh_mask).astype(np.int32),
            "U_shape": np.array(u_r.shape, dtype=np.int32),
            "Vt_shape": np.array(vh_r.shape, dtype=np.int32),
        }
        meta = {"orig_shape": orig_shape, "method": self.METHOD_NAME, "rank": r}
        return data, meta

    def decompress(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> np.ndarray:
        u_shape = tuple(data["U_shape"])
        vh_shape = tuple(data["Vt_shape"])

        U = np.zeros(u_shape, dtype=np.float32)
        U_idx = data["U_idx"]
        U[U_idx[:, 0], U_idx[:, 1]] = data["U_vals"]

        Vt = np.zeros(vh_shape, dtype=np.float32)
        Vt_idx = data["Vt_idx"]
        Vt[Vt_idx[:, 0], Vt_idx[:, 1]] = data["Vt_vals"]

        return ((U * data["S"][np.newaxis, :]) @ Vt).reshape(metadata["orig_shape"])

    def estimate_ratio(self, tensor: np.ndarray, **kwargs) -> float:
        rank = kwargs.get("rank", self.rank)
        sparsity = kwargs.get("sparsity", self.sparsity)
        orig = tensor.nbytes
        n, m = tensor.shape[0], tensor.shape[-1]
        nn_u = int(n * rank * (1 - sparsity))
        nn_vt = int(m * rank * (1 - sparsity))
        comp = nn_u * 8 + nn_vt * 8 + rank * 4
        return comp / max(orig, 1)

    def estimate_error(self, tensor: np.ndarray, **kwargs) -> dict:
        data, meta = self.compress(tensor, **kwargs)
        recon = self.decompress(data, meta)
        orig = tensor.astype(np.float64).ravel()
        rec = recon.astype(np.float64).ravel()
        mse = float(np.mean((orig - rec) ** 2))
        snr = 10 * np.log10(np.sum(orig**2) / (np.sum((orig - rec) ** 2) + 1e-30))
        cos_sim = float(
            np.dot(orig, rec) / (np.linalg.norm(orig) * np.linalg.norm(rec) + 1e-30)
        )
        return {
            "mse": mse,
            "snr_db": float(snr),
            "cosine_similarity": cos_sim,
            "rel_error": float(np.mean(np.abs(orig - rec) / (np.abs(orig) + 1e-10))),
        }


class OrthogonalProcrustesDecompose:
    """Orthogonal Procrustes + SVD with random restarts."""

    METHOD_NAME = "orthogonal_procrustes"
    category = "decomposition"

    def __init__(self, rank: int = 32, n_random_restarts: int = 5):
        self.rank = rank
        self.n_random_restarts = n_random_restarts

    def compress(
        self, tensor: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        rank = kwargs.get("rank", self.rank)
        orig_shape = tensor.shape
        mat = tensor.astype(np.float64).reshape(-1, tensor.shape[-1])
        n, m = mat.shape

        best_error = np.inf
        best_Q = np.eye(n)

        u0, _, _ = np.linalg.svd(mat, full_matrices=False)
        Q_init = u0[:, : min(n, m)]

        rng = np.random.RandomState(42)
        candidates = [Q_init]
        for _ in range(self.n_random_restarts):
            random_Q, _ = np.linalg.qr(rng.randn(n, min(n, m)))
            candidates.append(random_Q)

        for Q in candidates:
            if Q.shape[1] < min(n, m):
                Q_ext = np.eye(n)
                Q_ext[:, : Q.shape[1]] = Q
                Q = Q_ext

            rotated = Q.T @ mat
            u, s, vh = np.linalg.svd(rotated, full_matrices=False)
            r = min(rank, len(s))
            recon_rotated = (u[:, :r] * s[:r]) @ vh[:r, :]
            error = np.linalg.norm(rotated - recon_rotated)
            if error < best_error:
                best_error = error
                best_Q = Q
                best_u, best_s, best_vh, best_r = u, s, vh, r

        data = {
            "Q": best_Q.astype(np.float32),
            "U": best_u[:, :best_r].astype(np.float32),
            "S": best_s[:best_r].astype(np.float32),
            "Vt": best_vh[:best_r, :].astype(np.float32),
        }
        meta = {"orig_shape": orig_shape, "method": self.METHOD_NAME, "rank": best_r}
        return data, meta

    def decompress(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> np.ndarray:
        Q = data["Q"]
        recon_rotated = (data["U"] * data["S"][np.newaxis, :]) @ data["Vt"]
        return (Q @ recon_rotated).reshape(metadata["orig_shape"])

    def estimate_ratio(self, tensor: np.ndarray, **kwargs) -> float:
        rank = kwargs.get("rank", self.rank)
        orig = tensor.nbytes
        n = tensor.shape[0]
        comp = n * n * 4 + rank * (n + tensor.shape[-1] + rank) * 4
        return comp / max(orig, 1)

    def estimate_error(self, tensor: np.ndarray, **kwargs) -> dict:
        data, meta = self.compress(tensor, **kwargs)
        recon = self.decompress(data, meta)
        orig = tensor.astype(np.float64).ravel()
        rec = recon.astype(np.float64).ravel()
        mse = float(np.mean((orig - rec) ** 2))
        snr = 10 * np.log10(np.sum(orig**2) / (np.sum((orig - rec) ** 2) + 1e-30))
        cos_sim = float(
            np.dot(orig, rec) / (np.linalg.norm(orig) * np.linalg.norm(rec) + 1e-30)
        )
        return {
            "mse": mse,
            "snr_db": float(snr),
            "cosine_similarity": cos_sim,
            "rel_error": float(np.mean(np.abs(orig - rec) / (np.abs(orig) + 1e-10))),
        }


class NonNegativeMatrixFactorize:
    """Non-Negative Matrix Factorization with multiplicative updates."""

    METHOD_NAME = "nmf"
    category = "decomposition"

    def __init__(self, rank: int = 32, max_iter: int = 100, init_scale: float = 0.1):
        self.rank = rank
        self.max_iter = max_iter
        self.init_scale = init_scale

    def compress(
        self, tensor: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        rank = kwargs.get("rank", self.rank)
        max_iter = kwargs.get("max_iter", self.max_iter)
        orig_shape = tensor.shape
        mat = tensor.astype(np.float64).reshape(-1, tensor.shape[-1])

        mat_pos = np.abs(mat)
        n, m = mat_pos.shape
        r = min(rank, n, m)

        rng = np.random.RandomState(42)
        A = rng.rand(n, r) * self.init_scale + 1e-6
        B = rng.rand(r, m) * self.init_scale + 1e-6

        eps = 1e-10
        for _ in range(max_iter):
            numerator = A.T @ mat_pos
            denominator = A.T @ A @ B + eps
            B *= numerator / denominator

            numerator = mat_pos @ B.T
            denominator = A @ B @ B.T + eps
            A *= numerator / denominator

        data = {"A": A.astype(np.float32), "B": B.astype(np.float32)}
        meta = {
            "orig_shape": orig_shape,
            "method": self.METHOD_NAME,
            "rank": r,
            "sign": 1.0 if np.all(mat >= 0) else -1.0 if np.all(mat <= 0) else 0.0,
        }
        return data, meta

    def decompress(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> np.ndarray:
        result = data["A"] @ data["B"]
        sign = metadata.get("sign", 0.0)
        if sign != 0:
            result = result * sign
        return result.reshape(metadata["orig_shape"])

    def estimate_ratio(self, tensor: np.ndarray, **kwargs) -> float:
        rank = kwargs.get("rank", self.rank)
        orig = tensor.nbytes
        comp = rank * (tensor.shape[0] + tensor.shape[-1]) * 4
        return comp / max(orig, 1)

    def estimate_error(self, tensor: np.ndarray, **kwargs) -> dict:
        data, meta = self.compress(tensor, **kwargs)
        recon = self.decompress(data, meta)
        orig = tensor.astype(np.float64).ravel()
        rec = recon.astype(np.float64).ravel()
        mse = float(np.mean((orig - rec) ** 2))
        snr = 10 * np.log10(np.sum(orig**2) / (np.sum((orig - rec) ** 2) + 1e-30))
        cos_sim = float(
            np.dot(orig, rec) / (np.linalg.norm(orig) * np.linalg.norm(rec) + 1e-30)
        )
        return {
            "mse": mse,
            "snr_db": float(snr),
            "cosine_similarity": cos_sim,
            "rel_error": float(np.mean(np.abs(orig - rec) / (np.abs(orig) + 1e-10))),
        }


class ProbabilisticMatrixFactorize:
    """Probabilistic Matrix Factorization with uncertainty estimates."""

    METHOD_NAME = "pmf"
    category = "decomposition"

    def __init__(self, rank: int = 32, max_iter: int = 50, learning_rate: float = 0.01):
        self.rank = rank
        self.max_iter = max_iter
        self.learning_rate = learning_rate

    def compress(
        self, tensor: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        rank = kwargs.get("rank", self.rank)
        max_iter = kwargs.get("max_iter", self.max_iter)
        lr = kwargs.get("learning_rate", self.learning_rate)
        orig_shape = tensor.shape
        mat = tensor.astype(np.float64).reshape(-1, tensor.shape[-1])
        n, m = mat.shape
        r = min(rank, n, m)

        mat_scale = np.max(np.abs(mat)) + 1e-10
        mat_norm = mat / mat_scale

        rng = np.random.RandomState(42)
        A = rng.randn(n, r) * 0.01
        B = rng.randn(r, m) * 0.01

        mask = np.isfinite(mat_norm)
        mat_clean = np.where(mask, mat_norm, 0)

        for i in range(max_iter):
            current_lr = lr / (1 + 0.01 * i)
            pred = A @ B
            error = np.where(mask, pred - mat_clean, 0)
            A -= current_lr * (error @ B.T + 0.001 * A)
            B -= current_lr * (A.T @ error + 0.001 * B)

            A = np.clip(A, -10, 10)
            B = np.clip(B, -10, 10)

        residual = np.where(mask, mat_clean - A @ B, 0)
        noise_var = float(np.mean(residual**2)) + 1e-10

        data = {
            "A": (A * mat_scale).astype(np.float32),
            "B": B.astype(np.float32),
            "noise_var": np.float32(noise_var * mat_scale**2),
        }
        meta = {"orig_shape": orig_shape, "method": self.METHOD_NAME, "rank": r}
        return data, meta

    def decompress(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> np.ndarray:
        return (data["A"] @ data["B"]).reshape(metadata["orig_shape"])

    def estimate_ratio(self, tensor: np.ndarray, **kwargs) -> float:
        rank = kwargs.get("rank", self.rank)
        orig = tensor.nbytes
        comp = rank * (tensor.shape[0] + tensor.shape[-1]) * 4 + 4
        return comp / max(orig, 1)

    def estimate_error(self, tensor: np.ndarray, **kwargs) -> dict:
        data, meta = self.compress(tensor, **kwargs)
        recon = self.decompress(data, meta)
        orig = tensor.astype(np.float64).ravel()
        rec = recon.astype(np.float64).ravel()
        mse = float(np.mean((orig - rec) ** 2))
        snr = 10 * np.log10(np.sum(orig**2) / (np.sum((orig - rec) ** 2) + 1e-30))
        cos_sim = float(
            np.dot(orig, rec) / (np.linalg.norm(orig) * np.linalg.norm(rec) + 1e-30)
        )
        return {
            "mse": mse,
            "snr_db": float(snr),
            "cosine_similarity": cos_sim,
            "rel_error": float(np.mean(np.abs(orig - rec) / (np.abs(orig) + 1e-10))),
        }


class BayesianMatrixFactorize:
    """Bayesian NMF with automatic rank selection via variational inference."""

    METHOD_NAME = "bayesian_nmf"
    category = "decomposition"

    def __init__(
        self,
        max_rank: int = 64,
        max_iter: int = 100,
        alpha: float = 1.0,
        beta: float = 1.0,
    ):
        self.max_rank = max_rank
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta = beta

    def compress(
        self, tensor: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        max_rank = kwargs.get("max_rank", self.max_rank)
        max_iter = kwargs.get("max_iter", self.max_iter)
        orig_shape = tensor.shape
        mat = tensor.astype(np.float64).reshape(-1, tensor.shape[-1])
        mat_pos = np.abs(mat) + 1e-10
        n, m = mat_pos.shape

        u, s, vh = np.linalg.svd(mat_pos, full_matrices=False)
        r_init = min(max_rank, len(s))

        A = np.abs(u[:, :r_init]) * np.sqrt(s[:r_init])[np.newaxis, :] + 1e-6
        B = np.abs(vh[:r_init, :]) * np.sqrt(s[:r_init])[:, np.newaxis] + 1e-6

        eps = 1e-10
        for _ in range(max_iter):
            AB = A @ B + eps
            ratio = mat_pos / AB

            A *= (ratio @ B.T) / (np.sum(B, axis=1)[np.newaxis, :] + eps)
            B *= (A.T @ ratio) / (np.sum(A, axis=0)[:, np.newaxis] + eps)

        importance = np.mean(A, axis=0) * np.mean(B, axis=1)
        active_mask = importance > 0.01 * np.max(importance)
        A_active = A[:, active_mask]
        B_active = B[active_mask, :]

        sign = 1.0 if np.all(mat >= 0) else -1.0 if np.all(mat <= 0) else 0.0

        data = {"A": A_active.astype(np.float32), "B": B_active.astype(np.float32)}
        meta = {
            "orig_shape": orig_shape,
            "method": self.METHOD_NAME,
            "rank": int(np.sum(active_mask)),
            "sign": sign,
        }
        return data, meta

    def decompress(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> np.ndarray:
        result = data["A"] @ data["B"]
        sign = metadata.get("sign", 0.0)
        if sign != 0:
            result = result * sign
        return result.reshape(metadata["orig_shape"])

    def estimate_ratio(self, tensor: np.ndarray, **kwargs) -> float:
        max_rank = kwargs.get("max_rank", self.max_rank)
        orig = tensor.nbytes
        comp = max_rank * (tensor.shape[0] + tensor.shape[-1]) * 4
        return comp / max(orig, 1)

    def estimate_error(self, tensor: np.ndarray, **kwargs) -> dict:
        data, meta = self.compress(tensor, **kwargs)
        recon = self.decompress(data, meta)
        orig = tensor.astype(np.float64).ravel()
        rec = recon.astype(np.float64).ravel()
        mse = float(np.mean((orig - rec) ** 2))
        snr = 10 * np.log10(np.sum(orig**2) / (np.sum((orig - rec) ** 2) + 1e-30))
        cos_sim = float(
            np.dot(orig, rec) / (np.linalg.norm(orig) * np.linalg.norm(rec) + 1e-30)
        )
        return {
            "mse": mse,
            "snr_db": float(snr),
            "cosine_similarity": cos_sim,
            "rel_error": float(np.mean(np.abs(orig - rec) / (np.abs(orig) + 1e-10))),
        }


class TensorTrainAdvancedDecompose:
    """Advanced Tensor-Train decomposition via TT-SVD with adaptive rank."""

    METHOD_NAME = "tt_advanced"
    category = "decomposition"

    def __init__(self, max_rank: int = 16, tol: float = 1e-4):
        self.max_rank = max_rank
        self.tol = tol

    def _tt_svd(self, tensor: np.ndarray, max_rank: int, tol: float) -> list:
        shape = tensor.shape
        n_modes = len(shape)
        cores = []
        current = tensor.copy()

        for i in range(n_modes - 1):
            m = current.shape[0]
            mat = current.reshape(m, -1)
            u, s, vh = np.linalg.svd(mat, full_matrices=False)

            if tol > 0:
                total_energy = np.sum(s**2)
                cumsum = np.cumsum(s**2) / (total_energy + 1e-30)
                r = int(np.searchsorted(cumsum, 1.0 - tol)) + 1
            else:
                r = max_rank
            r = min(r, max_rank, len(s))

            u, s, vh = u[:, :r], s[:r], vh[:r, :]
            prev_r = 1 if i == 0 else cores[-1].shape[-1]
            cores.append(u.reshape(prev_r, shape[i], r))
            current = np.diag(s) @ vh

        cores.append(current.reshape(current.shape[0], shape[-1], 1))
        return cores

    def compress(
        self, tensor: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        max_rank = kwargs.get("max_rank", self.max_rank)
        tol = kwargs.get("tol", self.tol)
        orig_shape = tensor.shape
        mat = tensor.astype(np.float64)

        if mat.ndim < 2:
            mat = mat.reshape(1, -1)

        cores = self._tt_svd(mat, max_rank, tol)

        data = {"cores": [c.astype(np.float32) for c in cores]}
        meta = {"orig_shape": orig_shape, "method": self.METHOD_NAME}
        return data, meta

    def decompress(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> np.ndarray:
        result = data["cores"][0]
        for core in data["cores"][1:]:
            if result.ndim == 3:
                result = np.einsum("ijk,klm->ijlm", result, core)
                if result.ndim == 4:
                    result = result.reshape(
                        result.shape[0],
                        result.shape[1] * result.shape[2],
                        result.shape[3],
                    )
            else:
                result = np.einsum(
                    "ijk,kl->ijl", result, core.reshape(core.shape[0], -1)
                )
        return result.reshape(metadata["orig_shape"])

    def estimate_ratio(self, tensor: np.ndarray, **kwargs) -> float:
        max_rank = kwargs.get("max_rank", self.max_rank)
        orig = tensor.nbytes
        comp = tensor.ndim * max_rank * max_rank * max(tensor.shape[0], 1) * 4
        return comp / max(orig, 1)

    def estimate_error(self, tensor: np.ndarray, **kwargs) -> dict:
        data, meta = self.compress(tensor, **kwargs)
        recon = self.decompress(data, meta)
        orig = tensor.astype(np.float64).ravel()
        rec = recon.astype(np.float64).ravel()
        mse = float(np.mean((orig - rec) ** 2))
        snr = 10 * np.log10(np.sum(orig**2) / (np.sum((orig - rec) ** 2) + 1e-30))
        cos_sim = float(
            np.dot(orig, rec) / (np.linalg.norm(orig) * np.linalg.norm(rec) + 1e-30)
        )
        return {
            "mse": mse,
            "snr_db": float(snr),
            "cosine_similarity": cos_sim,
            "rel_error": float(np.mean(np.abs(orig - rec) / (np.abs(orig) + 1e-10))),
        }


class TensorRingAdvancedDecompose:
    """Advanced Tensor-Ring decomposition with periodic boundary."""

    METHOD_NAME = "tr_advanced"
    category = "decomposition"

    def __init__(self, max_rank: int = 16):
        self.max_rank = max_rank

    def compress(
        self, tensor: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        max_rank = kwargs.get("max_rank", self.max_rank)
        orig_shape = tensor.shape
        mat = tensor.astype(np.float64).reshape(-1, tensor.shape[-1])
        n, m = mat.shape

        u, s, vh = np.linalg.svd(mat, full_matrices=False)
        r = min(max_rank, len(s))

        left = u[:, :r]
        right = np.diag(s[:r]) @ vh[:r, :]

        core1 = left.reshape(n, r)
        core2 = right.reshape(r, m)

        diag_s = np.ones(r, dtype=np.float64)

        data = {
            "core1": core1.astype(np.float32),
            "core2": core2.astype(np.float32),
            "diag_s": diag_s.astype(np.float32),
        }
        meta = {"orig_shape": orig_shape, "method": self.METHOD_NAME, "rank": r}
        return data, meta

    def decompress(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> np.ndarray:
        core1 = data["core1"]
        core2 = data["core2"]
        diag_s = data["diag_s"]
        result = core1 @ np.diag(diag_s) @ core2
        return result.reshape(metadata["orig_shape"])

    def estimate_ratio(self, tensor: np.ndarray, **kwargs) -> float:
        max_rank = kwargs.get("max_rank", self.max_rank)
        orig = tensor.nbytes
        comp = max_rank * (tensor.shape[0] + tensor.shape[-1] + 1) * 4
        return comp / max(orig, 1)

    def estimate_error(self, tensor: np.ndarray, **kwargs) -> dict:
        data, meta = self.compress(tensor, **kwargs)
        recon = self.decompress(data, meta)
        orig = tensor.astype(np.float64).ravel()
        rec = recon.astype(np.float64).ravel()
        mse = float(np.mean((orig - rec) ** 2))
        snr = 10 * np.log10(np.sum(orig**2) / (np.sum((orig - rec) ** 2) + 1e-30))
        cos_sim = float(
            np.dot(orig, rec) / (np.linalg.norm(orig) * np.linalg.norm(rec) + 1e-30)
        )
        return {
            "mse": mse,
            "snr_db": float(snr),
            "cosine_similarity": cos_sim,
            "rel_error": float(np.mean(np.abs(orig - rec) / (np.abs(orig) + 1e-10))),
        }


class CPAdvancedDecompose:
    """Advanced CP Decomposition via ALS with multiple restarts."""

    METHOD_NAME = "cp_advanced"
    category = "decomposition"

    def __init__(self, rank: int = 32, max_iter: int = 100, n_restarts: int = 3):
        self.rank = rank
        self.max_iter = max_iter
        self.n_restarts = n_restarts

    def compress(
        self, tensor: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        rank = kwargs.get("rank", self.rank)
        max_iter = kwargs.get("max_iter", self.max_iter)
        orig_shape = tensor.shape
        mat = tensor.astype(np.float64).reshape(-1, tensor.shape[-1])
        n, m = mat.shape
        r = min(rank, n, m)

        best_error = np.inf
        best_A = None
        best_B = None

        for restart in range(self.n_restarts):
            rng = np.random.RandomState(restart)
            A = rng.randn(n, r) * 0.1
            B = rng.randn(m, r) * 0.1

            for _ in range(max_iter):
                A = mat @ B @ np.linalg.inv(B.T @ B + 1e-6 * np.eye(r))
                B = mat.T @ A @ np.linalg.inv(A.T @ A + 1e-6 * np.eye(r))

            recon = A @ B.T
            error = np.linalg.norm(mat - recon)
            if error < best_error:
                best_error = error
                best_A, best_B = A.copy(), B.copy()

        data = {"A": best_A.astype(np.float32), "B": best_B.astype(np.float32)}
        meta = {"orig_shape": orig_shape, "method": self.METHOD_NAME, "rank": r}
        return data, meta

    def decompress(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> np.ndarray:
        return (data["A"] @ data["B"].T).reshape(metadata["orig_shape"])

    def estimate_ratio(self, tensor: np.ndarray, **kwargs) -> float:
        rank = kwargs.get("rank", self.rank)
        orig = tensor.nbytes
        comp = rank * (tensor.shape[0] + tensor.shape[-1]) * 4
        return comp / max(orig, 1)

    def estimate_error(self, tensor: np.ndarray, **kwargs) -> dict:
        data, meta = self.compress(tensor, **kwargs)
        recon = self.decompress(data, meta)
        orig = tensor.astype(np.float64).ravel()
        rec = recon.astype(np.float64).ravel()
        mse = float(np.mean((orig - rec) ** 2))
        snr = 10 * np.log10(np.sum(orig**2) / (np.sum((orig - rec) ** 2) + 1e-30))
        cos_sim = float(
            np.dot(orig, rec) / (np.linalg.norm(orig) * np.linalg.norm(rec) + 1e-30)
        )
        return {
            "mse": mse,
            "snr_db": float(snr),
            "cosine_similarity": cos_sim,
            "rel_error": float(np.mean(np.abs(orig - rec) / (np.abs(orig) + 1e-10))),
        }


class TuckerAdvancedDecompose:
    """Advanced Tucker Decomposition via HOSVD with core optimization."""

    METHOD_NAME = "tucker_advanced"
    category = "decomposition"

    def __init__(self, ranks: tuple = (32, 32), max_iter: int = 50):
        self.ranks = ranks
        self.max_iter = max_iter

    def compress(
        self, tensor: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        ranks = kwargs.get("ranks", self.ranks)
        orig_shape = tensor.shape
        mat = tensor.astype(np.float64).reshape(-1, tensor.shape[-1])
        n, m = mat.shape

        r0 = min(ranks[0] if len(ranks) > 0 else 32, n)
        r1 = min(ranks[1] if len(ranks) > 1 else 32, m)

        u0, s0, vh0 = np.linalg.svd(mat, full_matrices=False)
        u1, s1, vh1 = np.linalg.svd(mat.T, full_matrices=False)

        U = u0[:, :r0]
        V = u1[:, :r1]

        core = U.T @ mat @ V

        for _ in range(self.max_iter):
            recon = U @ core @ V.T
            u_new, _, _ = np.linalg.svd(recon, full_matrices=False)
            U = u_new[:, :r0]
            core = U.T @ mat @ V

            recon = U @ core @ V.T
            v_new, _, _ = np.linalg.svd(recon.T, full_matrices=False)
            V = v_new[:, :r1]
            core = U.T @ mat @ V

        data = {
            "U": U.astype(np.float32),
            "core": core.astype(np.float32),
            "V": V.astype(np.float32),
        }
        meta = {"orig_shape": orig_shape, "method": self.METHOD_NAME, "ranks": (r0, r1)}
        return data, meta

    def decompress(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> np.ndarray:
        return (data["U"] @ data["core"] @ data["V"].T).reshape(metadata["orig_shape"])

    def estimate_ratio(self, tensor: np.ndarray, **kwargs) -> float:
        ranks = kwargs.get("ranks", self.ranks)
        orig = tensor.nbytes
        r0, r1 = ranks[0], ranks[1]
        comp = (tensor.shape[0] * r0 + r0 * r1 + tensor.shape[1] * r1) * 4
        return comp / max(orig, 1)

    def estimate_error(self, tensor: np.ndarray, **kwargs) -> dict:
        data, meta = self.compress(tensor, **kwargs)
        recon = self.decompress(data, meta)
        orig = tensor.astype(np.float64).ravel()
        rec = recon.astype(np.float64).ravel()
        mse = float(np.mean((orig - rec) ** 2))
        snr = 10 * np.log10(np.sum(orig**2) / (np.sum((orig - rec) ** 2) + 1e-30))
        cos_sim = float(
            np.dot(orig, rec) / (np.linalg.norm(orig) * np.linalg.norm(rec) + 1e-30)
        )
        return {
            "mse": mse,
            "snr_db": float(snr),
            "cosine_similarity": cos_sim,
            "rel_error": float(np.mean(np.abs(orig - rec) / (np.abs(orig) + 1e-10))),
        }


class HierarchicalTuckerDecompose:
    """Hierarchical Tucker with recursive tree-structured decomposition."""

    METHOD_NAME = "hierarchical_tucker"
    category = "decomposition"

    def __init__(self, max_rank: int = 16, max_depth: int = 3):
        self.max_rank = max_rank
        self.max_depth = max_depth

    def _recursive_split(self, mat: np.ndarray, depth: int, max_rank: int) -> dict:
        n, m = mat.shape
        if depth == 0 or min(n, m) <= max_rank:
            u, s, vh = np.linalg.svd(mat, full_matrices=False)
            r = min(max_rank, len(s))
            return {"type": "leaf", "U": u[:, :r], "S": s[:r], "Vt": vh[:r, :]}

        split = n // 2
        top = mat[:split, :]
        bottom = mat[split:, :]

        return {
            "type": "node",
            "left": self._recursive_split(top, depth - 1, max_rank),
            "right": self._recursive_split(bottom, depth - 1, max_rank),
        }

    def _tree_to_factors(self, tree: dict, factors: list) -> None:
        if tree["type"] == "leaf":
            factors.append(tree)
        else:
            self._tree_to_factors(tree["left"], factors)
            self._tree_to_factors(tree["right"], factors)

    def compress(
        self, tensor: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        max_rank = kwargs.get("max_rank", self.max_rank)
        orig_shape = tensor.shape
        mat = tensor.astype(np.float64).reshape(-1, tensor.shape[-1])

        tree = self._recursive_split(mat, self.max_depth, max_rank)
        factors = []
        self._tree_to_factors(tree, factors)

        stored_factors = []
        for f in factors:
            stored_factors.append(
                {
                    "U": f["U"].astype(np.float32),
                    "S": f["S"].astype(np.float32),
                    "Vt": f["Vt"].astype(np.float32),
                }
            )

        data = {"factors": stored_factors, "n_factors": np.int32(len(stored_factors))}
        meta = {"orig_shape": orig_shape, "method": self.METHOD_NAME}
        return data, meta

    def decompress(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> np.ndarray:
        factors = data["factors"]
        blocks = []
        for f in factors:
            block = (f["U"] * f["S"][np.newaxis, :]) @ f["Vt"]
            blocks.append(block)
        return np.vstack(blocks).reshape(metadata["orig_shape"])

    def estimate_ratio(self, tensor: np.ndarray, **kwargs) -> float:
        max_rank = kwargs.get("max_rank", self.max_rank)
        orig = tensor.nbytes
        n_blocks = 2**self.max_depth
        comp = (
            n_blocks
            * max_rank
            * (tensor.shape[0] // n_blocks + tensor.shape[-1] + max_rank)
            * 4
        )
        return comp / max(orig, 1)

    def estimate_error(self, tensor: np.ndarray, **kwargs) -> dict:
        data, meta = self.compress(tensor, **kwargs)
        recon = self.decompress(data, meta)
        orig = tensor.astype(np.float64).ravel()
        rec = recon.astype(np.float64).ravel()
        mse = float(np.mean((orig - rec) ** 2))
        snr = 10 * np.log10(np.sum(orig**2) / (np.sum((orig - rec) ** 2) + 1e-30))
        cos_sim = float(
            np.dot(orig, rec) / (np.linalg.norm(orig) * np.linalg.norm(rec) + 1e-30)
        )
        return {
            "mse": mse,
            "snr_db": float(snr),
            "cosine_similarity": cos_sim,
            "rel_error": float(np.mean(np.abs(orig - rec) / (np.abs(orig) + 1e-10))),
        }


class BlockDiagonalPlusLowRankDecompose:
    """Block-Diagonal + Low-Rank: D + UVt capturing local and global structure."""

    METHOD_NAME = "block_diag_plus_lr"
    category = "decomposition"

    def __init__(self, block_size: int = 64, lr_rank: int = 16):
        self.block_size = block_size
        self.lr_rank = lr_rank

    def compress(
        self, tensor: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        block_size = kwargs.get("block_size", self.block_size)
        lr_rank = kwargs.get("lr_rank", self.lr_rank)
        orig_shape = tensor.shape
        mat = tensor.astype(np.float64).reshape(-1, tensor.shape[-1])
        n, m = mat.shape

        bs = min(block_size, n, m)
        D = np.zeros_like(mat)

        for i in range(0, n, bs):
            for j in range(0, m, bs):
                block = mat[i : i + bs, j : j + bs]
                if block.size > 0:
                    D[i : i + bs, j : j + bs] = np.diag(np.diag(block))

        residual = mat - D

        u, s, vh = np.linalg.svd(residual, full_matrices=False)
        r = min(lr_rank, len(s))

        data = {
            "D_diag": np.array([D[i, i] for i in range(min(n, m))]).astype(np.float32),
            "U": u[:, :r].astype(np.float32),
            "S": s[:r].astype(np.float32),
            "Vt": vh[:r, :].astype(np.float32),
            "shape": np.array([n, m], dtype=np.int32),
            "block_size": np.int32(bs),
        }
        meta = {"orig_shape": orig_shape, "method": self.METHOD_NAME, "rank": r}
        return data, meta

    def decompress(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> np.ndarray:
        n, m = int(data["shape"][0]), int(data["shape"][1])
        bs = int(data["block_size"])
        D_diag = data["D_diag"]

        D = np.zeros((n, m), dtype=np.float32)
        for i in range(min(n, m)):
            D[i, i] = D_diag[i]

        lr = (data["U"] * data["S"][np.newaxis, :]) @ data["Vt"]
        return (D + lr).reshape(metadata["orig_shape"])

    def estimate_ratio(self, tensor: np.ndarray, **kwargs) -> float:
        lr_rank = kwargs.get("lr_rank", self.lr_rank)
        orig = tensor.nbytes
        n = min(tensor.shape[0], tensor.shape[1])
        comp = n * 4 + lr_rank * (tensor.shape[0] + tensor.shape[-1] + lr_rank) * 4
        return comp / max(orig, 1)

    def estimate_error(self, tensor: np.ndarray, **kwargs) -> dict:
        data, meta = self.compress(tensor, **kwargs)
        recon = self.decompress(data, meta)
        orig = tensor.astype(np.float64).ravel()
        rec = recon.astype(np.float64).ravel()
        mse = float(np.mean((orig - rec) ** 2))
        snr = 10 * np.log10(np.sum(orig**2) / (np.sum((orig - rec) ** 2) + 1e-30))
        cos_sim = float(
            np.dot(orig, rec) / (np.linalg.norm(orig) * np.linalg.norm(rec) + 1e-30)
        )
        return {
            "mse": mse,
            "snr_db": float(snr),
            "cosine_similarity": cos_sim,
            "rel_error": float(np.mean(np.abs(orig - rec) / (np.abs(orig) + 1e-10))),
        }


class SkeletonDecompose:
    """Skeleton Decomposition: skeleton rows/columns + interpolation."""

    METHOD_NAME = "skeleton"
    category = "decomposition"

    def __init__(self, rank: int = 32):
        self.rank = rank

    def compress(
        self, tensor: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        rank = kwargs.get("rank", self.rank)
        orig_shape = tensor.shape
        mat = tensor.astype(np.float64).reshape(-1, tensor.shape[-1])
        n, m = mat.shape

        u, s, vh = np.linalg.svd(mat, full_matrices=False)
        r = min(rank, len(s))

        col_norms = np.sum(u[:, :r] ** 2, axis=1)
        row_norms = np.sum(vh[:r, :].T ** 2, axis=1)

        skeleton_cols = np.argsort(-col_norms)[:r]
        skeleton_rows = np.argsort(-row_norms)[:r]

        C = mat[:, skeleton_cols]
        R = mat[skeleton_rows, :]
        W = mat[np.ix_(skeleton_rows, skeleton_cols)]

        Uw, sw, Vhw = np.linalg.svd(W, full_matrices=False)
        r_inv = min(r, len(sw), W.shape[0], W.shape[1])
        W_pinv = (Vhw[:r_inv, :].T * (1.0 / (sw[:r_inv] + 1e-10))) @ Uw[:, :r_inv].T

        data = {
            "C": C.astype(np.float32),
            "R": R.astype(np.float32),
            "W_pinv": W_pinv.astype(np.float32),
            "skeleton_cols": skeleton_cols.astype(np.int32),
            "skeleton_rows": skeleton_rows.astype(np.int32),
        }
        meta = {"orig_shape": orig_shape, "method": self.METHOD_NAME, "rank": r_inv}
        return data, meta

    def decompress(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> np.ndarray:
        return (data["C"] @ data["W_pinv"] @ data["R"]).reshape(metadata["orig_shape"])

    def estimate_ratio(self, tensor: np.ndarray, **kwargs) -> float:
        rank = kwargs.get("rank", self.rank)
        orig = tensor.nbytes
        comp = rank * (tensor.shape[0] + tensor.shape[-1] + rank) * 4
        return comp / max(orig, 1)

    def estimate_error(self, tensor: np.ndarray, **kwargs) -> dict:
        data, meta = self.compress(tensor, **kwargs)
        recon = self.decompress(data, meta)
        orig = tensor.astype(np.float64).ravel()
        rec = recon.astype(np.float64).ravel()
        mse = float(np.mean((orig - rec) ** 2))
        snr = 10 * np.log10(np.sum(orig**2) / (np.sum((orig - rec) ** 2) + 1e-30))
        cos_sim = float(
            np.dot(orig, rec) / (np.linalg.norm(orig) * np.linalg.norm(rec) + 1e-30)
        )
        return {
            "mse": mse,
            "snr_db": float(snr),
            "cosine_similarity": cos_sim,
            "rel_error": float(np.mean(np.abs(orig - rec) / (np.abs(orig) + 1e-10))),
        }


ALL_ADVANCED_FACTORIZATIONS = {
    "svd_residual": SVDResidualDecompose,
    "cur_full": CURFullDecompose,
    "nystrom_advanced": NystromAdvancedDecompose,
    "random_feature_advanced": RandomFeatureAdvancedDecompose,
    "block_svd": BlockSVDDecompose,
    "tiled_low_rank": TiledLowRankDecompose,
    "progressive_svd": ProgressiveSVDDecompose,
    "incremental_svd": IncrementalSVDDecompose,
    "sparse_svd": SparseSVDDecompose,
    "orthogonal_procrustes": OrthogonalProcrustesDecompose,
    "nmf": NonNegativeMatrixFactorize,
    "pmf": ProbabilisticMatrixFactorize,
    "bayesian_nmf": BayesianMatrixFactorize,
    "tt_advanced": TensorTrainAdvancedDecompose,
    "tr_advanced": TensorRingAdvancedDecompose,
    "cp_advanced": CPAdvancedDecompose,
    "tucker_advanced": TuckerAdvancedDecompose,
    "hierarchical_tucker": HierarchicalTuckerDecompose,
    "block_diag_plus_lr": BlockDiagonalPlusLowRankDecompose,
    "skeleton": SkeletonDecompose,
}

__all__ = [
    "SVDResidualDecompose",
    "CURFullDecompose",
    "NystromAdvancedDecompose",
    "RandomFeatureAdvancedDecompose",
    "BlockSVDDecompose",
    "TiledLowRankDecompose",
    "ProgressiveSVDDecompose",
    "IncrementalSVDDecompose",
    "SparseSVDDecompose",
    "OrthogonalProcrustesDecompose",
    "NonNegativeMatrixFactorize",
    "ProbabilisticMatrixFactorize",
    "BayesianMatrixFactorize",
    "TensorTrainAdvancedDecompose",
    "TensorRingAdvancedDecompose",
    "CPAdvancedDecompose",
    "TuckerAdvancedDecompose",
    "HierarchicalTuckerDecompose",
    "BlockDiagonalPlusLowRankDecompose",
    "SkeletonDecompose",
    "ALL_ADVANCED_FACTORIZATIONS",
]
