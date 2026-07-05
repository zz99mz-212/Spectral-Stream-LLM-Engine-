
import math
import time
import threading
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Any, Callable, Optional, Union

import numpy as np

from spectralstream.core.math_primitives import (

    next_power_of_two as _next_power_of_two,
    softmax as _softmax,
    dct as _dct,
    idct as _idct,
    spectral_entropy as _spectral_entropy,
    fwht,
    ifwht,
)

def _csr_from_dense(dense: np.ndarray, threshold: float = 1e-10) -> tuple:
    mask = np.abs(dense) > threshold
    indices = np.where(mask)
    values = dense[indices]
    return values, indices, dense.shape

def _dense_from_csr(values: np.ndarray, indices: tuple, shape: tuple) -> np.ndarray:
    out = np.zeros(shape, dtype=np.float64)
    out[indices] = values
    return out

def _nm_mask(shape: tuple, n: int, m: int, rng_seed: Optional[int] = None) -> np.ndarray:
    rows, cols = shape
    mask = np.zeros(shape, dtype=bool)
    rng = np.random.RandomState(rng_seed)
    for i in range(rows):
        for j_block in range(0, cols, m):
            block_end = min(j_block + m, cols)
            block_size = block_end - j_block
            chosen = rng.choice(block_size, min(n, block_size), replace=False)
            for c in chosen:
                mask[i, j_block + c] = True
    return mask

def _apply_nm_pattern(weights: np.ndarray, n: int, m: int) -> np.ndarray:
    rows, cols = weights.shape
    out = weights.copy()
    for i in range(rows):
        for j_block in range(0, cols, m):
            block_end = min(j_block + m, cols)
            block = out[i, j_block:block_end]
            if len(block) > n:
                abs_vals = np.abs(block)
                threshold = np.sort(abs_vals)[-n] if n > 0 else 0
                block[abs_vals < threshold] = 0.0
    return out

def _block_mask(shape: tuple, block_h: int, block_w: int, sparsity: float,
                rng_seed: Optional[int] = None) -> np.ndarray:
    rows, cols = shape
    mask = np.ones(shape, dtype=bool)
    rng = np.random.RandomState(rng_seed)
    for i in range(0, rows, block_h):
        for j in range(0, cols, block_w):
            if rng.random() < sparsity:
                ih = min(i + block_h, rows)
                jw = min(j + block_w, cols)
                mask[i:ih, j:jw] = False
    return mask

def _energy_ratio(x: np.ndarray) -> np.ndarray:
    x_spec = _dct(x)
    power = x_spec ** 2
    total = np.sum(power)
    cum = np.cumsum(power) / (total + 1e-30)
    return cum

def _sparsity_ratio(w: np.ndarray) -> float:
    return float(np.mean(np.abs(w) < 1e-10))

def _circular_conv(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = len(a)
    A_fft = np.fft.fft(a.astype(np.complex128))
    B_fft = np.fft.fft(b.astype(np.complex128))
    return np.fft.ifft(A_fft * B_fft).real.astype(np.float64)

def _circular_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = len(a)
    A_fft = np.fft.fft(a.astype(np.complex128))
    B_fft = np.fft.fft(b.astype(np.complex128))
    return np.fft.ifft(np.conj(A_fft) * B_fft).real.astype(np.float64)

class DynamicSparseExecutor:
    def __init__(self, config: Optional[SparsityConfig] = None,
                 fallback_density: float = 0.4):
        self.config = config or SparsityConfig()
        self.fallback_density = fallback_density
        self._stats: dict[str, float] = {}

    def _should_sparse(self, density: float) -> bool:
        return density < self.fallback_density

    def _to_csr(self, mat: np.ndarray) -> tuple:
        vals, (rows, cols), shape = _csr_from_dense(mat)
        row_ptr = np.zeros(shape[0] + 1, dtype=np.int64)
        for r in range(shape[0]):
            row_ptr[r + 1] = row_ptr[r] + int(np.sum(rows == r))
        col_idx = cols.astype(np.int64)
        return vals.astype(np.float64), col_idx, row_ptr

    def _to_csc(self, mat: np.ndarray) -> tuple:
        vals, (rows, cols), shape = _csr_from_dense(mat)
        col_ptr = np.zeros(shape[1] + 1, dtype=np.int64)
        for c in range(shape[1]):
            col_ptr[c + 1] = col_ptr[c] + int(np.sum(cols == c))
        row_idx = rows.astype(np.int64)
        return vals.astype(np.float64), row_idx, col_ptr

    def sparse_gemm(self, sparse_mat: np.ndarray, dense_vec: np.ndarray,
                    format: SparseFormat = SparseFormat.DENSE) -> np.ndarray:
        density = float(np.mean(np.abs(sparse_mat) > 1e-10))
        if not self._should_sparse(density):
            return sparse_mat @ dense_vec
        if format == SparseFormat.CSR:
            vals, col_idx, row_ptr = self._to_csr(sparse_mat)
            result = np.zeros(sparse_mat.shape[0], dtype=np.float64)
            for i in range(sparse_mat.shape[0]):
                s = row_ptr[i]
                e = row_ptr[i + 1]
                for j in range(s, e):
                    result[i] += vals[j] * dense_vec[col_idx[j]]
            return result.astype(sparse_mat.dtype)
        elif format == SparseFormat.N_M:
            w_sparse = _apply_nm_pattern(sparse_mat, self.config.nm_ratio[0],
                                         self.config.nm_ratio[1])
            return w_sparse @ dense_vec
        elif format == SparseFormat.BLOCK:
            bh, bw = self.config.block_size, self.config.block_size
            rows, cols = sparse_mat.shape
            result = np.zeros(rows, dtype=np.float64)
            for i in range(0, rows, bh):
                ih = min(i + bh, rows)
                for j in range(0, cols, bw):
                    jw = min(j + bw, cols)
                    block = sparse_mat[i:ih, j:jw]
                    if np.any(np.abs(block) > 1e-10):
                        result[i:ih] += block @ dense_vec[j:jw]
            return result.astype(sparse_mat.dtype)
        else:
            return sparse_mat @ dense_vec

    def sparse_gemm_batch(self, sparse_mat: np.ndarray, dense_batch: np.ndarray,
                          format: SparseFormat = SparseFormat.DENSE) -> np.ndarray:
        density = float(np.mean(np.abs(sparse_mat) > 1e-10))
        if not self._should_sparse(density):
            return sparse_mat @ dense_batch
        result = np.zeros((sparse_mat.shape[0], dense_batch.shape[1]), dtype=np.float64)
        if format == SparseFormat.CSR:
            vals, col_idx, row_ptr = self._to_csr(sparse_mat)
            for i in range(sparse_mat.shape[0]):
                s = row_ptr[i]
                e = row_ptr[i + 1]
                for j in range(s, e):
                    result[i] += vals[j] * dense_batch[col_idx[j]]
        else:
            nonzero = np.abs(sparse_mat) > 1e-10
            rows_nz, cols_nz = np.where(nonzero)
            for idx in range(len(rows_nz)):
                result[rows_nz[idx]] += sparse_mat[rows_nz[idx], cols_nz[idx]] * dense_batch[cols_nz[idx]]
        return result.astype(sparse_mat.dtype)

    def sparse_attention(self, query: np.ndarray, keys: np.ndarray, values: np.ndarray,
                         sparsity: Optional[float] = None) -> np.ndarray:
        n_q = query.shape[0]
        n_kv = keys.shape[0]
        d = query.shape[-1]
        if sparsity is None:
            sparsity = self.config.target_sparsity
        n_keep = max(1, int(n_kv * (1.0 - sparsity)))
        q_norm = query / (np.linalg.norm(query, axis=-1, keepdims=True) + 1e-10)
        k_norm = keys / (np.linalg.norm(keys, axis=-1, keepdims=True) + 1e-10)
        if n_q <= n_keep:
            sim = q_norm @ k_norm.T
            attn = _softmax(sim)
            return attn @ values
        q_chunk_size = min(64, n_q)
        output = np.zeros_like(query)
        for i in range(0, n_q, q_chunk_size):
            chunk_end = min(i + q_chunk_size, n_q)
            q_chunk = q_norm[i:chunk_end]
            sim = q_chunk @ k_norm.T
            top_k_idx = np.argsort(-sim, axis=1)[:, :n_keep]
            batch_idx = np.arange(chunk_end - i)[:, None]
            attn_vals = np.exp(sim[batch_idx, top_k_idx])
            attn_vals = attn_vals / (attn_vals.sum(axis=1, keepdims=True) + 1e-30)
            v_selected = values[top_k_idx]
            output[i:chunk_end] = (attn_vals[:, :, None] * v_selected).sum(axis=1)
        return output

    def sparse_softmax(self, logits: np.ndarray, support_set: np.ndarray,
                       temperature: float = 1.0) -> np.ndarray:
        supported = logits[..., support_set]
        scaled = supported / max(temperature, 1e-10)
        m = scaled.max(axis=-1, keepdims=True)
        e = np.exp(scaled - m)
        probs_support = e / (e.sum(axis=-1, keepdims=True) + 1e-30)
        full = np.zeros_like(logits)
        full[..., support_set] = probs_support
        return full

    def sparse_embed(self, embed_matrix: np.ndarray, indices: np.ndarray) -> np.ndarray:
        unique_idx = np.unique(indices)
        density = len(unique_idx) / max(embed_matrix.shape[0], 1)
        if self._should_sparse(density):
            gathered = embed_matrix[indices]
        else:
            gathered = embed_matrix[indices]
        return gathered

    def nm_matmul(self, a: np.ndarray, b: np.ndarray, n: int = 2, m: int = 4) -> np.ndarray:
        a_sparse = _apply_nm_pattern(a, n, m)
        return a_sparse @ b

    def block_sparse_matmul(self, a: np.ndarray, b: np.ndarray,
                            block_h: int = 32, block_w: int = 32) -> np.ndarray:
        rows, inner = a.shape
        _, cols = b.shape
        result = np.zeros((rows, cols), dtype=np.float64)
        for i in range(0, rows, block_h):
            ih = min(i + block_h, rows)
            for j in range(0, inner, block_w):
                jw = min(j + block_w, inner)
                a_block = a[i:ih, j:jw]
                if np.any(np.abs(a_block) > 1e-10):
                    result[i:ih] += a_block @ b[j:jw]
        return result.astype(a.dtype)

    def get_stats(self) -> dict:
        return dict(self._stats)

    def reset_stats(self):
        self._stats.clear()
