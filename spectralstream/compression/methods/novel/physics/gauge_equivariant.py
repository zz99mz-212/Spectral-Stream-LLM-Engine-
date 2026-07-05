"""
Gauge-Equivariant Compression — store ONE root geometric template + N shift vectors.

For a set of related weight matrices {W_1,...,W_N} (attention heads, MoE experts):
  1. Find Fréchet mean in SVD space → shared singular vectors U_base, V_base
  2. Each W_i ≈ U_base @ diag(s_i) @ V_base^T + sparse residual E_i
  3. Store: base (compressed) + {s_i, E_i} per head/expert

Standard: N × d² = Nd²
Gauge:    d² + N×d + N×sparse(E_i) = O(d² + Nd)
For N=256, d=4096: 4B → ~20M = 200x
"""

from __future__ import annotations


import math
import struct
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _procrustes_align(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Find optimal rotation R = argmin ||A @ R - B||_F via SVD.

    A, B are both (m, k) matrices. Returns R of shape (k, k).
    """
    M = A.T @ B
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt
    return R


def _block_int8_compress(mat: np.ndarray, block_size: int = 128) -> Tuple[bytes, dict]:
    flat = mat.ravel().astype(np.float32)
    n = len(flat)
    padded_n = int(math.ceil(n / block_size) * block_size)
    padded = np.zeros(padded_n, dtype=np.float32)
    padded[:n] = flat
    blocks = padded.reshape(-1, block_size)
    amax = np.max(np.abs(blocks), axis=1, keepdims=True)
    scales = np.where(amax > 1e-8, amax / 127.0, 1.0)
    quantized = np.clip(np.round(blocks / scales), -128, 127).astype(np.int8)
    header = struct.pack("<II", n, block_size)
    meta = {"_int8": True, "n": n, "block_size": block_size}
    return header + scales.astype(np.float32).tobytes() + quantized.tobytes(), meta


def _block_int8_decompress(data: bytes, meta: dict) -> np.ndarray:
    n = meta.get("n")
    block_size = meta.get("block_size", 128)
    n_blocks = (n + block_size - 1) // block_size
    pos = 8
    scales = np.frombuffer(data[pos : pos + n_blocks * 4], dtype=np.float32)
    pos += n_blocks * 4
    quantized = np.frombuffer(data[pos : pos + n_blocks * block_size], dtype=np.int8)
    quantized = quantized.reshape(-1, block_size).astype(np.float32)
    out = (quantized * scales[:, np.newaxis]).ravel()
    return out[:n]


class GaugeEquivariant:
    """Gauge-equivariant batch compression for related tensors.

    Compresses a batch of N related matrices (attention heads, MoE experts)
    by finding a shared geometric template (Fréchet mean in SVD space)
    and storing only the gauge transformation parameters per head.

    The 'gauge' is the shared singular vector frame (U_base, V_base).
    Each head differs only by its singular values (gauge field) + sparse residual.
    """

    name = "gauge_equivariant"
    category = "revolutionary_gauge"

    def __init__(self, base_rank: int = 64, residual_sparsity: float = 0.01):
        self.base_rank = base_rank
        self.residual_sparsity = residual_sparsity

    def compress(self, tensor: np.ndarray, **params) -> Tuple[bytes, dict]:
        result, meta = self.compress_batch([tensor], **params)
        meta["_single"] = True
        return result, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        return self.decompress_batch(data, metadata)[0]

    def compress_batch(self, tensors: List[np.ndarray], **params) -> Tuple[bytes, dict]:
        """Compress a batch of related tensors via gauge symmetry.

        Algorithm:
        1. Compute truncated SVD for each tensor (top-k singular vectors)
        2. Align all singular vectors to a common reference (Procrustes)
        3. Compute Fréchet mean of aligned singular values → base_s
        4. Store: compressed base (U, S, V) + per-head (delta_s, sparse_residual)
        """
        n_tensors = len(tensors)
        if n_tensors == 0:
            return b"", {"method": "gauge_equivariant", "n_tensors": 0}

        rank = params.get("rank", self.base_rank)
        sparsity = params.get("residual_sparsity", self.residual_sparsity)

        flat_mats = [t.astype(np.float64).reshape(t.shape[0], -1) for t in tensors]
        shapes = [t.shape for t in tensors]
        m, n = flat_mats[0].shape
        k = min(rank, m, n)

        # Step 1: Truncated SVD for each
        all_u = []
        all_s = []
        all_vt = []
        for mat in flat_mats:
            u, s, vt = np.linalg.svd(mat, full_matrices=False)
            all_u.append(u[:, :k].copy())
            all_s.append(s[:k].copy())
            all_vt.append(vt[:k, :].copy())

        # Step 2: Align all to the first head via Procrustes rotation.
        # After alignment: all_u[i] @ R_u[i] ≈ base_u, all_vt[i] aligned similarly.
        base_u = all_u[0].copy()
        base_vt = all_vt[0].copy()

        all_u_aligned = [all_u[0].copy()]
        all_vt_aligned = [all_vt[0].copy()]
        residuals = []

        for i in range(1, n_tensors):
            R_u = _procrustes_align(all_u[i], base_u)
            R_v = _procrustes_align(all_vt[i].T, base_vt.T)
            u_aligned = all_u[i] @ R_u
            vt_aligned = R_v.T @ all_vt[i]
            all_u_aligned.append(u_aligned)
            all_vt_aligned.append(vt_aligned)

            recon_i = u_aligned @ np.diag(all_s[i]) @ vt_aligned
            residual = flat_mats[i] - recon_i
            residuals.append(residual)

        # Step 3: Fréchet mean of singular values (median is robust)
        aligned_s_list = [all_s[0].copy()]
        residuals = []

        for i in range(1, n_tensors):
            aligned_s_list.append(all_s[i].copy())
            recon_i = all_u_aligned[i] @ np.diag(all_s[i]) @ all_vt_aligned[i]
            residual = flat_mats[i] - recon_i
            residuals.append(residual)

        s_arr = np.array(aligned_s_list)
        base_s = np.median(s_arr, axis=0)

        # Step 4: Compress base matrix
        base_mat = base_u @ np.diag(base_s) @ base_vt
        base_data, base_meta = _block_int8_compress(base_mat)

        # Step 5: Pack per-head data
        buf = bytearray()
        buf += struct.pack("<II", n_tensors, k)
        buf += struct.pack("<I", len(base_data))
        buf += base_data

        for i in range(n_tensors):
            s_i = all_s[i]
            delta_s = s_i - base_s
            # Store delta_s as int16 (lossy but 65536 levels is sufficient)
            delta_s_quant = np.clip(np.round(delta_s * 256.0), -32768, 32767).astype(
                np.int16
            )
            buf += delta_s_quant.tobytes()

            # Sparse residual: keep top-k entries by magnitude
            if i > 0:
                res = residuals[i - 1].ravel()
                n_keep = max(1, int(res.size * sparsity))
                res_mag = np.abs(res)
                thresh = np.sort(res_mag)[-n_keep] if n_keep < res_mag.size else 0.0
                mask = res_mag >= thresh
                indices = np.where(mask)[0].astype(np.int32)
                values = res[mask].astype(np.float32)
            else:
                indices = np.array([], dtype=np.int32)
                values = np.array([], dtype=np.float32)

            buf += struct.pack("<I", len(indices))
            buf += indices.tobytes()
            buf += values.tobytes()

        metadata = {
            "method": "gauge_equivariant",
            "n_tensors": n_tensors,
            "shapes": [list(s) for s in shapes],
            "base_rank": k,
            "residual_sparsity": sparsity,
            "m": m,
            "n": n,
        }

        return bytes(buf), metadata

    def decompress_batch(self, data: bytes, metadata: dict) -> List[np.ndarray]:
        """Decompress a batch from gauge representation."""
        if len(data) < 8:
            return []
        n_tensors, k = struct.unpack_from("<II", data, 0)
        pos = 8

        if n_tensors == 0:
            return []

        shapes = metadata.get("shapes", [])
        sparsity = metadata.get("residual_sparsity", 0.01)

        base_len = struct.unpack_from("<I", data, pos)[0]
        pos += 4
        base_data = data[pos : pos + base_len]
        pos += base_len

        base_flat = _block_int8_decompress(
            base_data,
            {"_int8": True, "n": metadata["m"] * metadata["n"], "block_size": 128},
        )
        base_mat = base_flat.reshape(metadata["m"], metadata["n"])

        # Re-decompose base for singular vectors
        base_u64, base_s64, base_vt64 = np.linalg.svd(
            base_mat.astype(np.float64), full_matrices=False
        )
        base_u = base_u64[:, :k].copy()
        base_s = base_s64[:k].copy()
        base_vt = base_vt64[:k, :].copy()

        tensors = []
        for i in range(n_tensors):
            delta_s = (
                np.frombuffer(data[pos : pos + k * 2], dtype=np.int16).astype(
                    np.float64
                )
                / 256.0
            )
            pos += k * 2
            s_i = base_s + delta_s

            n_res = struct.unpack_from("<I", data, pos)[0]
            pos += 4
            if n_res > 0:
                indices = np.frombuffer(data[pos : pos + n_res * 4], dtype=np.int32)
                pos += n_res * 4
                values = np.frombuffer(data[pos : pos + n_res * 4], dtype=np.float32)
                pos += n_res * 4
            else:
                indices = np.array([], dtype=np.int32)
                values = np.array([], dtype=np.float32)

            # Reconstruct: all heads share base_u, base_vt
            w = base_u @ np.diag(s_i) @ base_vt

            if n_res > 0:
                w_flat = w.ravel()
                w_flat[indices] += values
                w = w_flat.reshape(w.shape)

            shape = shapes[i] if i < len(shapes) else (metadata["m"], metadata["n"])
            tensors.append(w.reshape(shape).astype(np.float32))

        return tensors
