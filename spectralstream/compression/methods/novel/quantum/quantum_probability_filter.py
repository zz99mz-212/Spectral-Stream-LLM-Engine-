from __future__ import annotations

import math
import struct
from typing import Any, Dict, Tuple

import numpy as np

from spectralstream.compression.methods.novel._common import (
    _block_int8_fallback,
    _block_int8_decompress,
)


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()


def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()


class QuantumProbabilityFilter:
    """Quantum probability filter — SVD with probability-weighted sampling.

    Method:
    1. Compute SVD of weight matrix
    2. Treat squared singular values as quantum probability amplitudes
    3. Keep top-k by cumulative probability mass (not by magnitude threshold)
    4. Store truncated U_k, S_k, Vt_k directly (no Huffman overhead)
    """

    name = "quantum_probability_filter"
    category = "quantum_compression"

    def compress(
        self, tensor: np.ndarray, energy_threshold: float = 0.95, **params
    ) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        orig_shape = t.shape
        if t.ndim < 2:
            t_2d = t.reshape(1, -1)
        else:
            t_2d = t.reshape(t.shape[0], -1)
        m, n = t_2d.shape

        # Capped max rank for performance
        max_rank = min(m, n, 256)

        # Compute SVD
        from spectralstream.compression.methods.novel._common import (
            _rsvd_compress,
            _rsvd_decompress,
            _svd_decompress,
        )

        # Always compute SVD to get singular values for probability analysis
        # Use full SVD for small matrices, RSVD for large
        if m * n > 1_000_000 and max_rank < min(m, n) // 2:
            rsvd_data, rsvd_meta = _rsvd_compress(tensor, rank=max_rank)
            m_r, n_r, k_r = struct.unpack_from("<III", rsvd_data, 0)
            pos = 12
            U_temp = np.frombuffer(
                rsvd_data[pos : pos + m_r * k_r * 4], dtype=np.float32
            ).reshape(m_r, k_r)
            pos += m_r * k_r * 4
            S_temp = np.frombuffer(rsvd_data[pos : pos + k_r * 4], dtype=np.float32)
            pos += k_r * 4
            Vt_temp = np.frombuffer(
                rsvd_data[pos : pos + k_r * n_r * 4], dtype=np.float32
            ).reshape(k_r, n_r)
            k_full = k_r
        else:
            U_temp, S_temp, Vt_temp = np.linalg.svd(t_2d, full_matrices=False)
            k_full = len(S_temp)

        # Compute quantum probability amplitudes from squared singular values
        S_sq = S_temp**2
        total_prob = S_sq.sum()
        if total_prob < 1e-30:
            prob_amplitudes = np.ones(k_full, dtype=np.float64) / k_full
        else:
            prob_amplitudes = S_sq / total_prob

        # Find top-k by cumulative probability mass (quantum thresholding)
        cumsum = np.cumsum(prob_amplitudes)
        k = int(np.searchsorted(cumsum, energy_threshold) + 1)
        k = min(k, k_full, max_rank)

        # Truncate to k components
        U_k = U_temp[:, :k].astype(np.float32)
        S_k = S_temp[:k].astype(np.float32)
        Vt_k = Vt_temp[:k, :].astype(np.float32)

        # Store as truncated SVD
        data = (
            struct.pack("<III", m, n, k)
            + U_k.tobytes()
            + S_k.tobytes()
            + Vt_k.tobytes()
        )

        # Fallback if no compression
        if len(data) >= tensor.nbytes:
            return _block_int8_fallback(tensor)

        return data, {
            "_svd": True,
            "shape": orig_shape,
            "m": m,
            "n": n,
            "k": k,
            "energy_threshold": energy_threshold,
        }

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        from spectralstream.compression.methods.novel._common import _svd_decompress

        return _svd_decompress(data, metadata)
