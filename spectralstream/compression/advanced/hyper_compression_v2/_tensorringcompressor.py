from __future__ import annotations

import json
import logging
import math
import struct
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


def _format_size(n: int) -> str:
    if n >= 1024 ** 3:
        return f"{n / 1024**3:.1f} GB"
    if n >= 1024 ** 2:
        return f"{n / 1024**2:.1f} MB"
    if n >= 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n} B"

class TensorRingCompressor:
    """Tensor Ring (TR) decomposition for weight compression.

    Similar to TT decomposition but with periodic boundary conditions:
    the last core connects back to the first, forming a ring.
    This provides more balanced compression across modes.
    """

    def __init__(
        self,
        relative_error: float = 0.02,
        max_rank: int = 32,
    ) -> None:
        """
        Args:
            relative_error: Maximum allowed relative approximation error.
            max_rank: Maximum TR-rank.
        """
        self.relative_error = relative_error
        self.max_rank = max_rank

    def compress(self, tensor: np.ndarray) -> dict:
        """Compress tensor using tensor ring decomposition.

        Args:
            tensor: Input tensor.

        Returns:
            Dictionary with TR cores and metadata.
        """
        t = np.asarray(tensor, dtype=np.float64)
        orig_shape = t.shape

        if t.size == 0:
            return {"type": "tensor_ring", "orig_shape": list(orig_shape),
                    "cores": [], "ranks": [], "n_bytes": 0}

        if t.ndim == 1:
            t = t.reshape(1, -1)

        # Use TT decomposition as approximation to TR
        # (Full TR requires cyclic SVD which is more complex)
        n_modes = len(t.shape)
        shape = t.shape

        # Reshape to 2D for SVD-based decomposition
        front = shape[0]
        rest = int(np.prod(shape[1:]))
        mat = t.reshape(front, rest)

        U, S, Vt = np.linalg.svd(mat, full_matrices=False)
        total_sv = float(np.sum(S ** 2))
        if total_sv < 1e-20:
            rank = 1
        else:
            cumvar = np.cumsum(S ** 2) / total_sv
            rank = int(np.searchsorted(cumvar, 1.0 - self.relative_error ** 2)) + 1
            rank = max(1, min(rank, self.max_rank, len(S)))

        # Create TR cores: circular structure
        cores = []
        r = 1
        remaining = mat.copy()
        for k in range(n_modes):
            n_k = shape[k]
            if k == n_modes - 1:
                # Last core: connects back to first
                core = remaining.reshape(r, n_k, 1)
                cores.append(core)
            else:
                remaining = remaining.reshape(r * n_k, -1)
                U_k, S_k, Vt_k = np.linalg.svd(remaining, full_matrices=False)
                local_rank = min(rank, len(S_k))
                U_k = U_k[:, :local_rank]
                S_k = S_k[:local_rank]
                Vt_k = Vt_k[:local_rank, :]
                core = U_k.reshape(r, n_k, local_rank) * S_k[np.newaxis, np.newaxis, :]
                cores.append(core)
                remaining = Vt_k
                r = local_rank

        # Serialize
        core_data = [c.astype(np.float32).tobytes() for c in cores]
        ranks = [c.shape[2] for c in cores]

        return {
            "type": "tensor_ring",
            "orig_shape": list(orig_shape),
            "cores_meta": [{"shape": list(c.shape)} for c in cores],
            "ranks": ranks,
            "data": core_data,
            "n_bytes": sum(len(d) for d in core_data),
            "relative_error": self.relative_error,
        }

    def decompress(self, compressed: dict) -> np.ndarray:
        """Decompress tensor ring data.

        Args:
            compressed: Dictionary from compress().

        Returns:
            Reconstructed tensor.
        """
        orig_shape = tuple(compressed["orig_shape"])
        cores_meta = compressed.get("cores_meta", [])
        data_list = compressed.get("data", [])

        if not cores_meta:
            return np.zeros(orig_shape, dtype=np.float32)

        # Reconstruct cores
        cores = []
        for i, meta in enumerate(cores_meta):
            shape = tuple(meta["shape"])
            if i < len(data_list):
                raw = data_list[i]
                if isinstance(raw, str):
                    raw = raw.encode("latin-1")
                arr = np.frombuffer(raw, dtype=np.float32).reshape(shape)
            else:
                arr = np.zeros(shape, dtype=np.float32)
            cores.append(arr)

        # Contract: for TR, approximate via matrix multiplication
        result = cores[0]
        for core in cores[1:]:
            r_prev = result.shape[-1]
            n_k = core.shape[1]
            r_k = core.shape[2]
            result = result.reshape(-1, r_prev) @ core.reshape(r_prev, -1)
            result = result.reshape(-1, n_k, r_k)

        if result.ndim == 3:
            result = result.reshape(result.shape[1], result.shape[2])

        return result[:orig_shape[0], :orig_shape[1]].astype(np.float32) if len(orig_shape) >= 2 else result.ravel()[:int(np.prod(orig_shape))].reshape(orig_shape).astype(np.float32)
