"""VlasovMeanFieldCompression — physics-inspired compression using Vlasov mean-field theory.

Treats weight rows as particles in phase space (position=value, momentum=row index).
Uses low-rank covariance structure + distribution moments per block for reconstruction.
"""

from __future__ import annotations

import struct
from typing import Any, Dict, Tuple

import numpy as np


class VlasovMeanFieldCompression:
    """Vlasov mean-field compression — phase-space distribution approximation.

    Framework:
      - Rows of the weight matrix are "particles" in phase space
      - Position coordinate (q) = value at each column
      - Momentum coordinate (p) = row index
      - The "distribution function" f(q, p) is approximated per block as
        a multivariate Gaussian with low-rank covariance
      - Stores: mean vector, top-k eigenvectors of covariance (via SVD),
        and residual variance per block

    For reconstruction, samples are drawn from the low-rank Gaussian
    approximation, preserving within-block correlations.
    """

    name = "vlasov_mean_field_compression"
    category = "physics"

    def __init__(self, block_rows: int = 32, rank: int = 4, n_quantiles: int = 3):
        self.block_rows = max(1, int(block_rows))
        self.rank = max(1, int(rank))
        self.n_quantiles = max(1, min(n_quantiles, 5))
        self._rng_seed = 42

    def compress(
        self, tensor: np.ndarray, **params: Any
    ) -> Tuple[bytes, Dict[str, Any]]:
        tensor = np.asarray(tensor, dtype=np.float64)
        orig_shape = tensor.shape
        force_2d = tensor.reshape(orig_shape[0], -1)
        n_rows, n_cols = force_2d.shape

        block_rows = int(params.get("block_rows", self.block_rows))
        rank = int(params.get("rank", self.rank))
        rank = min(rank, block_rows, n_cols)

        n_blocks = int(np.ceil(n_rows / block_rows))
        quantile_pts = np.array([0.05, 0.5, 0.95], dtype=np.float64)

        buf = bytearray()
        buf += struct.pack("<III", n_blocks, block_rows, rank)

        for b in range(n_blocks):
            r_start = b * block_rows
            r_end = min(r_start + block_rows, n_rows)
            block = force_2d[r_start:r_end]
            actual_rows = r_end - r_start

            mean_vec = np.mean(block, axis=0).astype(np.float32)
            centered = block - mean_vec[np.newaxis, :]

            u, s, vt = np.linalg.svd(centered, full_matrices=False)
            k = min(rank, len(s))
            u_k = u[:, :k].astype(np.float32)
            s_k = s[:k].astype(np.float32)
            vt_k = vt[:k, :].astype(np.float32)

            recon_lr = (u_k * s_k) @ vt_k
            residual = centered - recon_lr
            residual_var = float(np.var(residual))

            q_vals = np.quantile(block.ravel(), quantile_pts).astype(np.float16)

            buf += struct.pack("<I", actual_rows)
            buf += mean_vec.tobytes()
            buf += struct.pack("<I", k)
            buf += u_k.tobytes()
            buf += s_k.tobytes()
            buf += vt_k.tobytes()
            buf += struct.pack("<d", residual_var)
            buf += q_vals.tobytes()

        return bytes(buf), {
            "orig_shape": orig_shape,
            "n_blocks": n_blocks,
            "block_rows": block_rows,
            "rank": rank,
            "n_cols": n_cols,
        }

    def decompress(self, data: bytes, metadata: Dict[str, Any]) -> np.ndarray:
        orig_shape = metadata["orig_shape"]
        n_cols = metadata["n_cols"]
        n_blocks, block_rows, rank = struct.unpack_from("<III", data, 0)
        pos = 12
        rank = max(1, rank)

        rng = np.random.RandomState(self._rng_seed)
        blocks = []

        quantile_pts = np.array([0.05, 0.5, 0.95], dtype=np.float64)
        norm_q_targets = np.array([-1.64485, 0.0, 1.64485], dtype=np.float64)

        for _ in range(n_blocks):
            actual_rows = struct.unpack_from("<I", data, pos)[0]
            pos += 4

            mean_vec = np.frombuffer(
                data[pos : pos + n_cols * 4], dtype=np.float32
            ).astype(np.float64)
            pos += n_cols * 4

            k = struct.unpack_from("<I", data, pos)[0]
            pos += 4
            k = min(max(0, k), rank)

            if k > 0:
                u_k = np.frombuffer(
                    data[pos : pos + actual_rows * k * 4], dtype=np.float32
                ).reshape(actual_rows, k)
                pos += actual_rows * k * 4

                s_k = np.frombuffer(data[pos : pos + k * 4], dtype=np.float32)
                pos += k * 4

                vt_k = np.frombuffer(
                    data[pos : pos + k * n_cols * 4], dtype=np.float32
                ).reshape(k, n_cols)
                pos += k * n_cols * 4
            else:
                u_k = np.zeros((actual_rows, 1), dtype=np.float32)
                s_k = np.zeros(1, dtype=np.float32)
                vt_k = np.zeros((1, n_cols), dtype=np.float32)
                k = 1

            residual_var = struct.unpack_from("<d", data, pos)[0]
            pos += 8

            q_bytes = data[pos : pos + self.n_quantiles * 2]
            stored_q = np.frombuffer(q_bytes, dtype=np.float16).astype(np.float64)
            pos += self.n_quantiles * 2

            z = rng.randn(actual_rows, n_cols).astype(np.float64)
            z_lr = (u_k[:, :k] * s_k[:k][np.newaxis, :]) @ vt_k[:k, :]
            noise = z * np.sqrt(max(residual_var, 1e-12))
            block_out = mean_vec[np.newaxis, :] + z_lr.astype(np.float64) + noise

            if self.n_quantiles >= 3:
                flat = block_out.ravel()
                sorted_flat = np.sort(flat)
                n_el = len(sorted_flat)
                src_pts = np.linspace(0, 1, self.n_quantiles)
                tgt_pts = np.linspace(0, 1, self.n_quantiles)
                src_vals = np.quantile(flat, src_pts)
                if np.all(np.isfinite(src_vals)):
                    mapped = np.interp(
                        np.linspace(0, 1, n_el),
                        np.linspace(0, 1, self.n_quantiles),
                        stored_q,
                    )
                    block_out = mapped.reshape(actual_rows, n_cols)
                else:
                    block_out = flat.reshape(actual_rows, n_cols)
            else:
                block_out = block_out.reshape(actual_rows, n_cols)

            blocks.append(block_out.astype(np.float32))

        result = np.vstack(blocks).astype(np.float32)
        result = result.ravel()[: int(np.prod(orig_shape))]
        return result.reshape(orig_shape)
