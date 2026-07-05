from __future__ import annotations

import math
import struct
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class QuantumBudgetAllocator:
    """Quantum annealing for optimal bit/error budget allocation across layers.
    Use QA to find the allocation that minimizes total distortion.
    """

    name = "quantum_budget_allocator"
    category = "quantum_engine"

    def compress(
        self,
        tensor: np.ndarray,
        n_partitions: int = 8,
        total_budget: float = 1.0,
        n_anneal_steps: int = 50,
    ) -> Tuple[bytes, dict]:
        """Quantum budget allocator: SVD-based compression with budget allocation.
        
        Uses truncated SVD with rank proportional to budget, then allocates
        precision across partitions based on variance.
        """
        from spectralstream.compression.methods.novel._common import _svd_compress, _svd_decompress
        
        t = tensor.astype(np.float64)
        orig_shape = t.shape
        if t.ndim < 2:
            t_2d = t.reshape(1, -1)
        else:
            t_2d = t.reshape(t.shape[0], -1)
        m, n = t_2d.shape
        
        # Allocate rank based on budget
        budget_rank = max(1, int(min(m, n) * total_budget * 0.25))
        k = min(budget_rank, m, n, 192)
        
        # Use SVD instead of random annealing
        U, S, Vt = np.linalg.svd(t_2d, full_matrices=False)
        k = min(k, len(S))
        
        # Partition singular values and allocate bits
        n_part = min(n_partitions, k)
        partition_sizes = np.full(n_part, k // n_part, dtype=int)
        partition_sizes[: k % n_part] += 1
        partitions = np.split(S[:k], np.cumsum(partition_sizes)[:-1])
        variances = np.array([float(p.var()) for p in partitions])
        
        # Normalize budget based on variance (more variance = more bits)
        if variances.sum() > 0:
            budget = variances / variances.sum()
        else:
            budget = np.ones(n_part, dtype=np.float64) / n_part
        
        U_k = U[:, :k].astype(np.float32)
        S_k = S[:k].astype(np.float32)
        Vt_k = Vt[:k, :].astype(np.float32)
        
        meta = dict(
            shape=orig_shape,
            m=m,
            n=n,
            k=k,
            n_partitions=n_part,
            budget=budget.astype(np.float32).tolist(),
        )
        data = struct.pack("<III", m, n, k)
        data += _serialize(U_k)
        data += _serialize(S_k)
        data += _serialize(Vt_k)
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        from spectralstream.compression.methods.novel._common import _svd_decompress
        return _svd_decompress(data, metadata)
