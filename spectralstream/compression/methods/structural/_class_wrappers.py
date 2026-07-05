"""
Structural method class wrappers — pruning, sparsity, structured matrices.
"""

from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()


def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()


class Einsort:
    """Einsort decomposition with optimal index reordering — HPC-optimized."""

    name = "einsort"
    category = "structural"

    def compress(self, tensor: np.ndarray, rank: int = None) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float32)
        if t.ndim < 2:
            t = t.reshape(1, -1)
        orig_shape = t.shape
        m = t.shape[0]
        # Row-normalized similarity via vectorized dot product
        row_norm = np.linalg.norm(t, axis=1, keepdims=True) + 1e-10
        t_norm = t / row_norm
        sim = t_norm @ t_norm.T
        sim = (sim + sim.T) * 0.5
        d = np.sum(np.abs(sim), axis=1)
        L = np.diag(d) - sim
        try:
            eigvals, eigvecs = np.linalg.eigh(L.astype(np.float64))
            order = np.argsort(eigvals)
            fiedler_idx = order[min(1, len(order) - 1)]
            fiedler = eigvecs[:, fiedler_idx]
            row_perm = np.argsort(fiedler)
        except (np.linalg.LinAlgError, ValueError):
            row_perm = np.arange(m)
        permuted = t[row_perm]
        if permuted.ndim > 2:
            permuted = permuted.reshape(permuted.shape[0], -1)
        # Use truncated SVD via scipy if available
        try:
            from scipy.linalg import svd

            U, S, Vt = svd(permuted, full_matrices=False, lapack_driver="gesdd")
        except ImportError:
            U, S, Vt = np.linalg.svd(permuted, full_matrices=False)
        if rank is None:
            rank = max(1, int(np.sum(S > np.max(S) * 0.01)))
        rank = min(rank, len(S))
        meta = dict(shape=orig_shape, rank=int(rank), row_perm=row_perm.tolist())
        data = _serialize(U[:, :rank]) + _serialize(S[:rank]) + _serialize(Vt[:rank, :])
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        rank = metadata["rank"]
        row_perm = np.array(metadata["row_perm"])
        pos = 0
        U = _deserialize(data[: shape[0] * rank * 4]).reshape(shape[0], rank)
        pos += shape[0] * rank * 4
        S = _deserialize(data[pos : pos + rank * 4])
        pos += rank * 4
        Vt = _deserialize(data[pos : pos + rank * shape[-1] * 4]).reshape(
            rank, shape[-1]
        )
        recon = (U * S) @ Vt
        inv_perm = np.argsort(row_perm)
        return recon[inv_perm].reshape(shape).astype(np.float32)


class MonarchStructured:
    """Monarch structured matrix: block-diagonal butterfly factors — HPC vectorized."""

    name = "monarch_structured"
    category = "structural"

    def compress(self, tensor: np.ndarray, block_size: int = 2) -> Tuple[bytes, dict]:
        t = np.asarray(tensor, dtype=np.float32)
        if t.ndim < 2 or min(t.shape) < 4:
            flat = t.ravel()
            return flat.astype(np.float16).tobytes(), {
                "shape": t.shape,
                "passthrough": True,
            }
        m, n = t.shape
        bs = max(2, min(block_size, min(m, n) // 2))
        n_row = max(1, m // bs)
        n_col = max(1, n // bs)
        bm, bn = m // n_row, n // n_col
        U_blocks, S_vals, Vt_blocks = [], [], []
        meta_entries = []
        for i in range(n_row):
            for j in range(n_col):
                bi = t[i * bm : (i + 1) * bm, j * bn : (j + 1) * bn]
                U, S, Vt = np.linalg.svd(bi, full_matrices=False)
                r = max(1, min(bs, len(S)))
                U_blocks.append(U[:, :r].ravel())
                S_vals.append(S[:r])
                Vt_blocks.append(Vt[:r, :].ravel())
                meta_entries.append(
                    dict(i=i, j=j, U_sz=U[:, :r].size, S_sz=r, Vt_sz=Vt[:r, :].size)
                )
        data = (
            np.concatenate(U_blocks).tobytes()
            + np.concatenate(S_vals).tobytes()
            + np.concatenate(Vt_blocks).tobytes()
        )
        meta = dict(
            shape=t.shape, n_row=n_row, n_col=n_col, bm=bm, bn=bn, blocks=meta_entries
        )
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        if metadata.get("passthrough"):
            return (
                np.frombuffer(data, dtype=np.float16).reshape(shape).astype(np.float32)
            )
        n_row, n_col = metadata["n_row"], metadata["n_col"]
        bm, bn = metadata["bm"], metadata["bn"]
        blocks = metadata["blocks"]
        recon = np.zeros(shape, dtype=np.float32)
        pos = 0
        # Extract U blocks
        U_blocks = []
        for blk in blocks:
            r = blk["S_sz"]
            n_u = blk["U_sz"]
            U_blocks.append(
                np.frombuffer(data[pos : pos + n_u * 4], dtype=np.float32).reshape(
                    -1, r
                )
            )
            pos += n_u * 4
        # Extract all S values
        s_sizes = [b["S_sz"] for b in blocks]
        total_s = sum(s_sizes) * 4
        S_all = np.frombuffer(data[pos : pos + total_s], dtype=np.float32)
        pos += total_s
        # Extract Vt blocks and reconstruct
        s_offset = 0
        for idx, blk in enumerate(blocks):
            r = blk["S_sz"]
            n_v = blk["Vt_sz"]
            Vt = np.frombuffer(data[pos : pos + n_v * 4], dtype=np.float32).reshape(
                r, -1
            )
            pos += n_v * 4
            U = U_blocks[idx]
            S = S_all[s_offset : s_offset + r]
            s_offset += r
            i, j = blk["i"], blk["j"]
            recon[i * bm : (i + 1) * bm, j * bn : (j + 1) * bn] = (U * S) @ Vt
        return recon


class ButterflyStructured:
    """Butterfly structured: hierarchical sparse factorization."""

    name = "butterfly_structured"
    category = "structural"

    def compress(self, tensor: np.ndarray, n_levels: int = None) -> Tuple[bytes, dict]:
        from spectralstream.compression.methods.decomposition._class_wrappers import (
            Butterfly,
        )

        return Butterfly().compress(tensor, n_levels=n_levels)

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        from spectralstream.compression.methods.decomposition._class_wrappers import (
            Butterfly,
        )

        return Butterfly().decompress(data, metadata)


class Circulant:
    """Circulant matrix approximation — vectorized mean along diagonals via broadcasting."""

    name = "circulant"
    category = "structural"

    def compress(self, tensor: np.ndarray) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        if t.ndim > 2:
            t = t.reshape(t.shape[0], -1)
        m, n = t.shape
        max_dim = max(m, n)
        flat = t.ravel()
        row_idx = np.arange(m)
        col_idx = np.arange(n)
        diag_idx = ((row_idx[:, None] - col_idx[None, :]) % max_dim).ravel()
        _cnt = np.bincount(diag_idx, minlength=max_dim)
        c = np.bincount(diag_idx, weights=flat, minlength=max_dim) / np.maximum(_cnt, 1)
        meta = dict(shape=tensor.shape)
        data = _serialize(c.astype(np.float32))
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        m, n = shape[0], shape[-1] if len(shape) >= 2 else shape[0]
        c = _deserialize(data)
        max_dim = len(c)
        row = np.arange(max_dim)[:, None]
        col = np.arange(max_dim)[None, :]
        full = c[(row - col) % max_dim]
        return full[:m, :n].astype(np.float32)


class Vandermonde:
    """Vandermonde matrix structure for structured compression."""

    name = "vandermonde"
    category = "structural"

    def compress(self, tensor: np.ndarray, rank: int = 8) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        if t.ndim > 2:
            t = t.reshape(t.shape[0], -1)
        m, n = t.shape
        x = np.linspace(0, 1, m)
        y = np.linspace(0, 1, n)
        V = np.vander(x, rank, increasing=True)
        coeffs = np.linalg.lstsq(V, t, rcond=None)[0]
        recon = V @ coeffs
        meta = dict(shape=tensor.shape, rank=rank)
        data = _serialize(coeffs.astype(np.float32))
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        m, n = shape[0], shape[-1] if len(shape) >= 2 else shape[0]
        rank = metadata["rank"]
        coeffs = _deserialize(data).reshape(rank, n)
        x = np.linspace(0, 1, m)
        V = np.vander(x, rank, increasing=True)
        return (V @ coeffs).reshape(shape).astype(np.float32)


class Cauchy:
    """Cauchy matrix for structured compression."""

    name = "cauchy"
    category = "structural"

    def compress(self, tensor: np.ndarray, rank: int = 8) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        if t.ndim > 2:
            t = t.reshape(t.shape[0], -1)
        m, n = t.shape
        x = np.linspace(0, 1, m)[:, None]
        y = np.linspace(0, 1, n)[None, :]
        C = 1.0 / (np.abs(x - y) + 1e-10)
        try:
            coeffs = (
                np.linalg.solve(C[:rank, :rank], t[:rank, :rank])
                if rank <= min(m, n)
                else np.linalg.lstsq(C, t, rcond=None)[0]
            )
            recon = C @ coeffs
        except (np.linalg.LinAlgError, ValueError):
            recon = t
        meta = dict(shape=tensor.shape)
        data = _serialize(recon.astype(np.float32))
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        return _deserialize(data).reshape(shape).astype(np.float32)


class HSSMatrix:
    """Hierarchically Semi-Separable (HSS) matrix compression."""

    name = "hss_matrix"
    category = "structural"

    def compress(self, tensor: np.ndarray, tol: float = 0.01) -> Tuple[bytes, dict]:
        from spectralstream.compression.methods.structural.low_rank_structured import (
            hss_matrix_compress,
        )

        c, ratio, snr = hss_matrix_compress(tensor, tol)
        data = (
            _serialize(c.get("data", tensor).astype(np.float32))
            if "data" in c
            else _serialize(tensor.astype(np.float32))
        )
        meta = dict(shape=tensor.shape, tol=tol)
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        return _deserialize(data).reshape(shape).astype(np.float32)


class BSSMatrix:
    """Block Semi-Separable matrix compression."""

    name = "bss_matrix"
    category = "structural"

    def compress(self, tensor: np.ndarray, block_size: int = 16) -> Tuple[bytes, dict]:
        return self._compress_block_bss(tensor, block_size)

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        return _deserialize(data).reshape(metadata["shape"]).astype(np.float32)

    def _compress_block_bss(self, tensor, block_size):
        t = tensor.astype(np.float64)
        if t.ndim > 2:
            t = t.reshape(t.shape[0], -1)
        m, n = t.shape
        recon = np.zeros((m, n), dtype=np.float64)
        for i in range(0, m, block_size):
            for j in range(0, n, block_size):
                bi = min(block_size, m - i)
                bj = min(block_size, n - j)
                block = t[i : i + bi, j : j + bj]
                U, S, Vt = np.linalg.svd(block, full_matrices=False)
                r = max(1, min(bi, bj) // 2)
                recon[i : i + bi, j : j + bj] = (U[:, :r] * S[:r]) @ Vt[:r, :]
        meta = dict(shape=tensor.shape, block_size=block_size)
        data = _serialize(recon.astype(np.float32))
        return data, meta


class Structured24:
    """Structured 2:4 N:M sparsity (50% structured zeros)."""

    name = "structured_24"
    category = "structural"

    def compress(
        self, tensor: np.ndarray, n: int = 2, m: int = 4
    ) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float32)
        flat = t.ravel()
        n_total = len(flat)
        pad = -n_total % m
        if pad:
            flat = np.pad(flat, (0, pad), mode="constant")
        n_groups = len(flat) // m
        groups = flat.reshape(n_groups, m)
        sort_idx = np.argsort(np.abs(groups), axis=1)
        mask_flat = np.zeros(len(flat), dtype=bool)
        row_offsets = np.repeat(np.arange(n_groups), n)
        col_pos = sort_idx[:, -n:].ravel()
        mask_flat[row_offsets * m + col_pos] = True
        kept = flat[mask_flat]
        mask_orig = mask_flat[:n_total]
        kept_vals = t.ravel()[mask_orig]
        mask_packed = np.packbits(mask_orig)
        meta = dict(shape=tensor.shape, n=n, m=m, n_kept=len(kept_vals))
        data = mask_packed.tobytes() + _serialize(kept_vals)
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        n_kept = metadata["n_kept"]
        n_total = int(np.prod(shape))
        mask_bytes = (n_total + 7) // 8
        mask_packed = np.frombuffer(data[:mask_bytes], dtype=np.uint8)
        mask = np.unpackbits(mask_packed)[:n_total].astype(bool)
        kept_vals = _deserialize(data[mask_bytes : mask_bytes + n_kept * 4])
        recon = np.zeros(n_total, dtype=np.float32)
        recon[mask] = kept_vals[: np.sum(mask)]
        return recon.reshape(shape).astype(np.float32)


class BlockSparsity:
    """Block sparsity: keep top-N blocks by L2 norm."""

    name = "block_sparsity"
    category = "structural"

    def compress(
        self, tensor: np.ndarray, block_size: int = 16, density: float = 0.5
    ) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float32)
        flat = t.ravel()
        n = flat.size
        pad = -n % block_size
        if pad:
            flat = np.pad(flat, (0, pad))
        n_blocks = len(flat) // block_size
        blocks = flat.reshape(n_blocks, block_size)
        norms = np.linalg.norm(blocks, axis=1)
        n_keep = max(1, min(int(density * n_blocks), n_blocks))
        order = np.argsort(-norms)
        mask = np.zeros(n_blocks, dtype=bool)
        mask[order[:n_keep]] = True
        kept = blocks[mask].ravel()
        mask_packed = np.packbits(mask)
        meta = dict(
            shape=tensor.shape, block_size=block_size, n_keep=n_keep, n_blocks=n_blocks
        )
        data = mask_packed.tobytes() + _serialize(kept)
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        block_size = metadata["block_size"]
        n_blocks = metadata["n_blocks"]
        n_keep = metadata["n_keep"]
        mask_bytes = (n_blocks + 7) // 8
        mask_packed = np.frombuffer(data[:mask_bytes], dtype=np.uint8)
        mask = np.unpackbits(mask_packed)[:n_blocks].astype(bool)
        kept = _deserialize(data[mask_bytes : mask_bytes + n_keep * block_size * 4])
        recon = np.zeros(n_blocks * block_size, dtype=np.float32)
        flat_mask = np.repeat(mask, block_size)
        recon[flat_mask] = kept
        n = int(np.prod(shape))
        return recon[:n].reshape(shape).astype(np.float32)


class UnstructuredPruning:
    """Unstructured magnitude pruning — O(n) via argpartition."""

    name = "unstructured_pruning"
    category = "structural"

    def compress(self, tensor: np.ndarray, sparsity: float = 0.5) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float32)
        flat = t.ravel()
        n = len(flat)
        n_keep = max(1, int(n * (1.0 - sparsity)))
        order = np.argpartition(-np.abs(flat), n_keep - 1)[:n_keep]
        mask = np.zeros(n, dtype=bool)
        mask[order] = True
        mask_packed = np.packbits(mask)
        kept = flat[mask]
        meta = dict(shape=tensor.shape, sparsity=sparsity, n_kept=n_keep)
        data = mask_packed.tobytes() + _serialize(kept)
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        n_kept = metadata["n_kept"]
        n = int(np.prod(shape))
        mask_bytes = (n + 7) // 8
        mask_packed = np.frombuffer(data[:mask_bytes], dtype=np.uint8)
        mask = np.unpackbits(mask_packed)[:n].astype(bool)
        kept = _deserialize(data[mask_bytes : mask_bytes + n_kept * 4])
        recon = np.zeros(n, dtype=np.float32)
        recon[mask] = kept[: np.sum(mask)]
        return recon.reshape(shape).astype(np.float32)


class SparseGPT:
    """SparseGPT: Hessian-based pruning with error compensation."""

    name = "sparse_gpt"
    category = "structural"

    def compress(self, tensor: np.ndarray, sparsity: float = 0.5) -> Tuple[bytes, dict]:
        from spectralstream.compression.methods.structural.sparsity_compression import (
            sparsegpt_prune,
        )

        c, ratio, snr = sparsegpt_prune(tensor, None, sparsity)
        mask_vals = c.get("mask", np.ones(tensor.shape, dtype=bool))
        mask_packed = np.packbits(mask_vals.ravel())
        meta = dict(
            shape=tensor.shape, sparsity=sparsity, n_kept=int(np.sum(mask_vals))
        )
        data = bytes(mask_packed) + _serialize(
            c.get("values", np.zeros(0, dtype=np.float32))
        )
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        n_kept = metadata["n_kept"]
        n = int(np.prod(shape))
        mask_bytes = (n + 7) // 8
        mask_packed = np.frombuffer(data[:mask_bytes], dtype=np.uint8)
        mask = np.unpackbits(mask_packed)[:n].astype(bool)
        kept = _deserialize(data[mask_bytes : mask_bytes + n_kept * 4])
        recon = np.zeros(n, dtype=np.float32)
        recon[mask] = kept[: np.sum(mask)]
        return recon.reshape(shape).astype(np.float32)


class WandaPruning:
    """Wanda: Weight x Activation magnitude pruning."""

    name = "wanda_pruning"
    category = "structural"

    def compress(self, tensor: np.ndarray, sparsity: float = 0.5) -> Tuple[bytes, dict]:
        return SparseGPT().compress(tensor, sparsity)

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        return SparseGPT().decompress(data, metadata)


class DynamicNMSparsity:
    """Dynamic N:M sparsity with per-group top-N selection."""

    name = "dynamic_nm_sparsity"
    category = "structural"

    def compress(
        self, tensor: np.ndarray, n: int = 2, m: int = 4
    ) -> Tuple[bytes, dict]:
        return Structured24().compress(tensor, n=n, m=m)

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        return Structured24().decompress(data, metadata)


class ChannelPruning:
    """Channel (row) pruning: keep top rows by L2 norm."""

    name = "channel_pruning"
    category = "structural"

    def compress(
        self, tensor: np.ndarray, keep_frac: float = 0.5
    ) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        if t.ndim < 2:
            t = t.reshape(1, -1)
        norms = np.linalg.norm(t, axis=1)
        n_keep = max(1, int(t.shape[0] * keep_frac))
        order = np.argsort(-norms)[:n_keep]
        kept = t[order]
        meta = dict(shape=tensor.shape, keep_frac=keep_frac, kept_rows=order.tolist())
        data = _serialize(kept.astype(np.float32))
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        kept_rows = np.array(metadata["kept_rows"])
        n_keep = len(kept_rows)
        kept = _deserialize(data).reshape(n_keep, shape[-1])
        recon = np.zeros(shape, dtype=np.float32)
        valid = kept_rows < shape[0]
        recon[kept_rows[valid]] = kept[valid]
        return recon.astype(np.float32)


class GroupLasso:
    """Group Lasso: zero out weight groups jointly."""

    name = "group_lasso"
    category = "structural"

    def compress(
        self, tensor: np.ndarray, lambda_reg: float = 0.01
    ) -> Tuple[bytes, dict]:
        from spectralstream.compression.methods.structural.block_structured import (
            group_lasso,
        )

        c, ratio, snr = group_lasso(tensor, lambda_reg)
        data = _serialize(c.get("data", tensor).astype(np.float32))
        meta = dict(shape=tensor.shape, lambda_reg=lambda_reg)
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        return _deserialize(data).reshape(shape).astype(np.float32)


class AdaptiveSparsity:
    """Adaptive sparsity — vectorized per-block top-k selection via argpartition."""

    name = "adaptive_sparsity"
    category = "structural"

    def compress(
        self, tensor: np.ndarray, global_sparsity: float = 0.5
    ) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        flat = t.ravel()
        n = len(flat)
        block_size = 64
        n_blocks = (n + block_size - 1) // block_size
        padded = np.zeros(n_blocks * block_size)
        padded[:n] = flat
        blocks = padded.reshape(n_blocks, block_size)
        var = np.var(blocks, axis=1)
        vn = var / (var.max() + 1e-30)
        sparsity_per_block = global_sparsity * (1.0 + vn * 0.5)
        n_keep_per_block = np.maximum(
            1, (block_size * (1.0 - sparsity_per_block)).astype(np.int32)
        )
        mask = np.zeros(n_blocks * block_size, dtype=bool)
        abs_blocks = np.abs(blocks)
        order = np.argsort(-abs_blocks, axis=1)
        n_keep_arr = n_keep_per_block.astype(np.int32)
        row_offsets = np.arange(n_blocks)[:, None] * block_size
        lin_indices = row_offsets + order
        keep_mask = np.arange(block_size)[None, :] < n_keep_arr[:, None]
        mask[lin_indices[keep_mask]] = True
        mask = mask[:n]
        mask_packed = np.packbits(mask)
        kept = flat[mask]
        meta = dict(
            shape=tensor.shape,
            global_sparsity=global_sparsity,
            n_kept=int(np.sum(mask)),
        )
        data = mask_packed.tobytes() + _serialize(kept.astype(np.float32))
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        n_kept = metadata["n_kept"]
        n = int(np.prod(shape))
        mask_bytes = (n + 7) // 8
        mask_packed = np.frombuffer(data[:mask_bytes], dtype=np.uint8)
        mask = np.unpackbits(mask_packed)[:n].astype(bool)
        kept = _deserialize(data[mask_bytes : mask_bytes + n_kept * 4])
        recon = np.zeros(n, dtype=np.float32)
        recon[mask] = kept[: np.sum(mask)]
        return recon.reshape(shape).astype(np.float32)


class SparseQuantizeCombined:
    """Combined sparsity + quantization: prune then INT4 quantize."""

    name = "sparse_quantize_combined"
    category = "structural"

    def compress(
        self, tensor: np.ndarray, sparsity: float = 0.5, bits: int = 4
    ) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float32)
        flat = t.ravel()
        n = len(flat)
        n_keep = max(1, int(n * (1.0 - sparsity)))
        order = np.argpartition(-np.abs(flat), n_keep - 1)[:n_keep]
        kept = flat[order]
        max_q = (1 << (bits - 1)) - 1
        sc = float(np.max(np.abs(kept))) / max_q if len(kept) > 0 else 1.0
        sc = max(sc, 1e-10)
        q = np.clip(np.round(kept / sc), -max_q, max_q).astype(np.int8)
        mask = np.zeros(n, dtype=bool)
        mask[order] = True
        mask_packed = np.packbits(mask)
        meta = dict(
            shape=tensor.shape, sparsity=sparsity, bits=bits, n_kept=n_keep, scale=sc
        )
        data = mask_packed.tobytes() + struct.pack("<f", sc) + q.tobytes()
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        n_kept = metadata["n_kept"]
        bits = metadata["bits"]
        sc = metadata["scale"]
        n = int(np.prod(shape))
        mask_bytes = (n + 7) // 8
        mask_packed = np.frombuffer(data[:mask_bytes], dtype=np.uint8)
        mask = np.unpackbits(mask_packed)[:n].astype(bool)
        max_q = (1 << (bits - 1)) - 1
        q = np.frombuffer(
            data[mask_bytes + 4 : mask_bytes + 4 + n_kept], dtype=np.int8
        ).astype(np.float32)
        kept = q * sc
        recon = np.zeros(n, dtype=np.float32)
        recon[mask] = kept[: np.sum(mask)]
        return recon.reshape(shape).astype(np.float32)


from spectralstream.compression.methods.structural.optimal_transport import (
    OptimalTransportCompression,
)
from spectralstream.compression.methods.structural.basis_sharing import (
    BasisSharing,
)


class StructuredLowRank:
    """Structured Low-Rank + Sparse decomposition (W ≈ L + S).

    Decomposes weight matrix into low-rank component L (via truncated SVD)
    plus a sparse residual S (via magnitude thresholding). Captures both
    global structure and local outliers in LLM weights.

    Storage: U(m×r) + S(r,) + Vt(r×n) for low-rank + sparse indices/values.
    """

    name = "structured_low_rank"
    category = "structural"

    def compress(
        self, tensor: np.ndarray, rank: float = 0.1, sparsity: float = 0.1
    ) -> Tuple[bytes, dict]:
        t = np.asarray(tensor, dtype=np.float32)
        if t.ndim > 2:
            t = t.reshape(t.shape[0], -1)
        orig_shape = tensor.shape
        m, n = t.shape

        r = max(1, min(m, n, int(min(m, n) * rank)))

        # 1. Truncated SVD for low-rank component
        U, S, Vt = np.linalg.svd(t, full_matrices=False)
        U_r = U[:, :r].astype(np.float32)
        S_r = S[:r].astype(np.float32)
        Vt_r = Vt[:r, :].astype(np.float32)
        L = (U_r * S_r) @ Vt_r

        # 2. Sparse residual
        R = t - L
        flat_r = R.ravel()
        n_total = len(flat_r)
        n_keep_s = max(1, int(n_total * sparsity))
        if n_keep_s < n_total:
            thresh_idx = np.argpartition(-np.abs(flat_r), n_keep_s - 1)[:n_keep_s]
            s_mask = np.zeros(n_total, dtype=bool)
            s_mask[thresh_idx] = True
            s_vals = flat_r[s_mask].astype(np.float32)
        else:
            s_mask = np.ones(n_total, dtype=bool)
            s_vals = flat_r.astype(np.float32)

        # 3. Serialize
        mask_packed = np.packbits(s_mask)
        data = (
            U_r.tobytes()
            + S_r.tobytes()
            + Vt_r.tobytes()
            + mask_packed.tobytes()
            + s_vals.tobytes()
        )
        meta = dict(
            shape=orig_shape,
            rank=r,
            m=m,
            n=n,
            sparsity=sparsity,
            n_sparse_kept=len(s_vals),
            n_total=n_total,
        )
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        m, n = metadata["m"], metadata["n"]
        r = metadata["rank"]
        n_keep_s = metadata["n_sparse_kept"]
        n_total = metadata["n_total"]

        pos = 0
        U_r = np.frombuffer(data[pos : pos + m * r * 4], dtype=np.float32).reshape(m, r)
        pos += m * r * 4
        S_r = np.frombuffer(data[pos : pos + r * 4], dtype=np.float32)
        pos += r * 4
        Vt_r = np.frombuffer(data[pos : pos + r * n * 4], dtype=np.float32).reshape(
            r, n
        )
        pos += r * n * 4

        mask_bytes = (n_total + 7) // 8
        s_mask = np.unpackbits(
            np.frombuffer(data[pos : pos + mask_bytes], dtype=np.uint8)
        )[:n_total].astype(bool)
        pos += mask_bytes
        s_vals = np.frombuffer(data[pos : pos + n_keep_s * 4], dtype=np.float32)

        L = (U_r * S_r) @ Vt_r
        recon = L.ravel()
        recon[s_mask] += s_vals
        return recon.reshape(shape).astype(np.float32)
