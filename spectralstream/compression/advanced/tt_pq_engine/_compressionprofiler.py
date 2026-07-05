from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from spectralstream.core.math_primitives import (
    HadamardRotator,
    LloydMaxQuantizer,
    dct,
    fwht,
    idct,
    next_power_of_two,
    softmax,
    spectral_entropy,
)


class CompressionProfiler:
    """Analyze tensors and recommend compression parameters.

    Profiles tensor structure (rank, sparsity, entropy) to recommend
    optimal TT-ranks, PQ-subspace count, and expected compression ratio.
    """

    def __init__(self, target_compression: float = 0.1):
        self.target_compression = target_compression

    def _categorize_tensor(
        self, singular_values: np.ndarray, sparsity: float
    ) -> str:
        """Categorize tensor for compression strategy selection."""
        sv = singular_values / (np.sum(singular_values) + 1e-30)
        cumulative = np.cumsum(sv)
        effective_rank = int(np.searchsorted(cumulative, 0.9)) + 1

        if sparsity > 0.8:
            return "sparse"
        elif effective_rank < len(singular_values) * 0.1:
            return "low_rank"
        elif sparsity > 0.5:
            return "medium_sparse"
        else:
            return "dense"

    def analyze(self, tensor: np.ndarray) -> TensorProfile:
        """Analyze a tensor and return compression profile.

        Args:
            tensor: Input weight tensor.

        Returns:
            TensorProfile with analysis results and recommendations.
        """
        tensor = np.asarray(tensor, dtype=np.float64)
        flat = tensor.ravel()

        mean_val = float(np.mean(flat))
        std_val = float(np.std(flat))
        sparsity = float(np.mean(np.abs(flat) < 1e-10))
        ent = spectral_entropy(flat)

        # Effective condition number via SVD (on 2D reshape)
        if tensor.ndim >= 2:
            m, n = tensor.shape[0], int(np.prod(tensor.shape[1:]))
            mat = tensor.reshape(m, n)
            min_dim = min(m, n)
            if min_dim > 0:
                try:
                    sv = np.linalg.svd(mat, compute_uv=False)
                    sv_max = float(sv[0]) if len(sv) > 0 else 1.0
                    sv_min = float(sv[-1]) if len(sv) > 0 else 1e-10
                    condition_number = sv_max / max(sv_min, 1e-10)
                    effective_rank = int(np.searchsorted(
                        np.cumsum(sv ** 2) / (np.sum(sv ** 2) + 1e-30), 0.9
                    )) + 1
                except np.linalg.LinAlgError:
                    sv = np.array([std_val])
                    condition_number = 1.0
                    effective_rank = 1
            else:
                sv = np.array([std_val])
                condition_number = 1.0
                effective_rank = 1
        else:
            sv = np.array([std_val])
            condition_number = 1.0
            effective_rank = 1

        category = self._categorize_tensor(sv, sparsity)

        # Recommendations
        n_elements = tensor.size
        target_elements = int(n_elements * self.target_compression)

        if tensor.ndim >= 2:
            n_modes = min(tensor.ndim, 4)
            base_rank = max(4, min(64, int(np.sqrt(n_elements / max(target_elements, 1)))))
            tt_ranks = tuple(
                min(base_rank, int(np.prod(tensor.shape[: i + 1])),
                    int(np.prod(tensor.shape[i:])))
                for i in range(n_modes - 1)
            )
            tt_ranks = (1,) + tt_ranks + (1,)
        else:
            tt_ranks = (1, 1)

        if category == "low_rank":
            pq_subspaces = 4
        elif category == "sparse":
            pq_subspaces = 6
        else:
            pq_subspaces = 8

        estimated_cr = float(n_elements * 32) / max(target_elements * self.config_pq_bits(pq_subspaces), 1)

        return TensorProfile(
            shape=tensor.shape,
            n_elements=n_elements,
            dtype_str=str(tensor.dtype),
            mean=mean_val,
            std=std_val,
            sparsity=sparsity,
            spectral_entropy=ent,
            condition_number=condition_number,
            effective_rank=effective_rank,
            recommended_tt_ranks=tt_ranks,
            recommended_pq_subspaces=pq_subspaces,
            estimated_compression_ratio=estimated_cr,
            category=category,
        )

    def config_pq_bits(self, n_subspaces: int) -> float:
        """Estimate bits per element for given PQ config."""
        return (n_subspaces * 4) / max(n_subspaces, 1)
