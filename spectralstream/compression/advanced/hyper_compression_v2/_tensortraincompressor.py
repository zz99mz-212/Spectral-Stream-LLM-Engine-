from __future__ import annotations

import json
import logging
import math
import pickle
import struct
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


def _format_size(n: int) -> str:
    if n >= 1024**3:
        return f"{n / 1024**3:.1f} GB"
    if n >= 1024**2:
        return f"{n / 1024**2:.1f} MB"
    if n >= 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n} B"


class TensorTrainCompressor:
    """Tensor Train (TT) decomposition for high-dimensional weight compression.

    Decomposes a tensor W of shape (n1, n2, ..., nd) into core tensors
    G_k of shape (r_{k-1}, n_k, r_k), where r_k are TT-ranks.
    Compression is achieved when ranks are much smaller than dimensions.
    """

    def __init__(
        self,
        relative_error: float = 0.02,
        max_rank: int = 64,
    ) -> None:
        """
        Args:
            relative_error: Maximum allowed relative approximation error.
            max_rank: Maximum TT-rank.
        """
        self.relative_error = relative_error
        self.max_rank = max_rank

    def _tt_svd(self, tensor: np.ndarray) -> List[np.ndarray]:
        """Compute TT decomposition via recursive SVD.

        Args:
            tensor: Input tensor of any shape.

        Returns:
            List of TT core tensors.
        """
        shape = tensor.shape
        n_modes = len(shape)
        if n_modes == 1:
            return [tensor.reshape(1, -1, 1)]

        cores: List[np.ndarray] = []
        remaining = tensor.reshape(shape[0], -1)
        r = 1

        for k in range(n_modes - 1):
            n_k = shape[k]
            remaining = remaining.reshape(r * n_k, -1)
            U, S, Vt = np.linalg.svd(remaining, full_matrices=False)

            # Determine rank based on relative error
            total_sv = float(np.sum(S**2))
            if total_sv < 1e-20:
                rank = 1
            else:
                cumvar = np.cumsum(S**2) / total_sv
                rank = int(np.searchsorted(cumvar, 1.0 - self.relative_error**2)) + 1
                rank = max(1, min(rank, self.max_rank, len(S)))

            U = U[:, :rank]
            S = S[:rank]
            Vt = Vt[:rank, :]

            # Store core
            core = U.reshape(r, n_k, rank) * S[np.newaxis, np.newaxis, :]
            cores.append(core)

            remaining = Vt
            r = rank

        # Last core
        n_last = shape[-1]
        cores.append(remaining.reshape(r, n_last, 1))

        return cores

    def compress(self, tensor: np.ndarray) -> Tuple[bytes, Dict[str, Any]]:
        t = np.asarray(tensor, dtype=np.float64)
        orig_shape = t.shape

        if t.size == 0:
            data = pickle.dumps({"cores_meta": [], "data": [], "ranks": [1]})
            return data, {
                "orig_shape": list(orig_shape),
                "n_bytes": 0,
                "type": "tensor_train",
            }

        if t.ndim == 1:
            t = t.reshape(1, -1)

        cores = self._tt_svd(t)
        ranks = [1] + [c.shape[2] for c in cores]
        core_data = [c.astype(np.float32).tobytes() for c in cores]
        total_bytes = sum(len(d) for d in core_data)

        cores_meta = [
            {"shape": list(c.shape), "n_bytes": len(core_data[i])}
            for i, c in enumerate(cores)
        ]

        compressed_dict = {
            "cores_meta": cores_meta,
            "ranks": ranks,
            "data": core_data,
            "relative_error": self.relative_error,
        }
        data = pickle.dumps(compressed_dict)
        metadata: Dict[str, Any] = {
            "orig_shape": list(orig_shape),
            "n_bytes": total_bytes,
            "type": "tensor_train",
        }
        return data, metadata

    def decompress(self, compressed: bytes, metadata: Dict[str, Any]) -> np.ndarray:
        orig_shape = tuple(metadata["orig_shape"])
        d = pickle.loads(compressed)
        cores_meta = d.get("cores_meta", [])
        data_list = d.get("data", [])

        if not cores_meta:
            return np.zeros(orig_shape, dtype=np.float32)

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

        result = cores[0]
        for core in cores[1:]:
            r_prev = result.shape[-1]
            n_k = core.shape[1]
            r_k = core.shape[2]
            prev_matrix = result.reshape(-1, r_prev)
            core_matrix = core.reshape(r_prev, -1)
            result = prev_matrix @ core_matrix

        flat = result.ravel()[: int(np.prod(orig_shape))]
        if len(flat) < int(np.prod(orig_shape)):
            flat = np.pad(flat, (0, int(np.prod(orig_shape)) - len(flat)))
        return flat.reshape(orig_shape).astype(np.float32)
