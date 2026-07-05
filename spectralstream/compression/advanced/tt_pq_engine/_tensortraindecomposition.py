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


class TensorTrainDecomposition:
    """Tensor Train decomposition via TT-SVD algorithm.

    Decomposes a tensor T of shape (d1, d2, ..., dk) into cores:
        T(i1, ..., ik) = G1(i1) * G2(i2) * ... * Gk(ik)

    where Gi has shape (r_{i-1}, d_i, r_i) with r_0 = r_k = 1.

    The TT-SVD algorithm computes each core via sequential SVD,
    with adaptive rank selection based on energy retention.
    """

    def __init__(self, config: Optional[TTConfig] = None):
        self.config = config or TTConfig()
        self._cores: Optional[List[np.ndarray]] = None
        self._original_shape: Optional[Tuple[int, ...]] = None
        self._tt_ranks: Optional[Tuple[int, ...]] = None

    def _zero_pad_to_power_of_two(self, tensor: np.ndarray) -> Tuple[np.ndarray, Tuple[int, ...]]:
        """Pad tensor dimensions to next power of two for FFT-friendly ops."""
        if not self.config.zero_pad_power_of_two:
            return tensor, tensor.shape
        original_shape = tensor.shape
        new_shape = tuple(next_power_of_two(d) for d in original_shape)
        if new_shape == original_shape:
            return tensor, original_shape
        padded = np.zeros(new_shape, dtype=tensor.dtype)
        slicing = tuple(slice(0, d) for d in original_shape)
        padded[slicing] = tensor
        logger.debug("Zero-padded %s -> %s", original_shape, new_shape)
        return padded, original_shape

    def _compute_tt_rank_from_energy(
        self, s: np.ndarray, threshold: float
    ) -> int:
        """Select TT-rank based on cumulative energy retention."""
        total_energy = float(np.sum(s ** 2))
        if total_energy < 1e-30:
            return 1
        cumulative = np.cumsum(s ** 2) / total_energy
        rank = int(np.searchsorted(cumulative, threshold)) + 1
        return max(1, rank)

    def decompose(self, tensor: np.ndarray) -> List[np.ndarray]:
        """Decompose tensor into TT-cores via TT-SVD.

        Args:
            tensor: Input tensor of arbitrary order.

        Returns:
            List of TT-cores [G1, G2, ..., Gk].
        """
        tensor = np.asarray(tensor, dtype=np.float64)
        if tensor.ndim < 2:
            raise ValueError(f"Tensor must have ndim >= 2, got {tensor.ndim}")

        padded, original_shape = self._zero_pad_to_power_of_two(tensor)
        self._original_shape = original_shape

        shape = padded.shape
        n_modes = len(shape)
        max_rank = self.config.max_rank

        cores: List[np.ndarray] = []
        current = padded.reshape(shape[0], -1)
        ranks = [1]

        for mode in range(n_modes - 1):
            d_i = shape[mode]
            remaining_cols = current.shape[1]

            U, S, Vt = np.linalg.svd(current.reshape(-1, remaining_cols), full_matrices=False)

            if self.config.adaptive_rank:
                rank = min(
                    self._compute_tt_rank_from_energy(S, self.config.energy_threshold),
                    max_rank,
                    len(S),
                )
            else:
                rank = min(max_rank, len(S))

            rank = max(1, rank)
            ranks.append(rank)

            U = U[:, :rank]
            S = S[:rank]
            Vt = Vt[:rank, :]

            core = U.reshape(d_i, -1, rank).transpose(1, 0, 2)
            cores.append(core)

            current = np.diag(S) @ Vt

        # Last core
        last_rank = ranks[-1]
        d_last = shape[-1]
        cores.append(current.reshape(-1, d_last, 1).transpose(1, 0, 2))

        ranks.append(1)
        self._tt_ranks = tuple(ranks)
        self._cores = cores

        logger.info(
            "TT decomposition: shape=%s, ranks=%s, cores=%d",
            original_shape, ranks, len(cores),
        )
        return cores

    def reconstruct(self, cores: Optional[List[np.ndarray]] = None) -> np.ndarray:
        """Reconstruct tensor from TT-cores.

        Returns:
            Reconstructed tensor with original shape.
        """
        if cores is None:
            cores = self._cores
        if cores is None or self._original_shape is None:
            raise ValueError("No decomposition found; call decompose() first")

        result = cores[0]
        for i in range(1, len(cores)):
            # result: (..., r_prev), cores[i]: (d_i, r_prev, r_next) -> (..., d_i, r_next)
            result = np.einsum("...r,drp->...dp", result, cores[i])

        # Squeeze leading/trailing rank-1 dims
        result = result.reshape(result.shape[:-1])

        # Remove zero-padding
        slicing = tuple(slice(0, d) for d in self._original_shape)
        result = result[slicing]
        return result

    def compression_ratio(self) -> float:
        """Compute the compression ratio of the TT decomposition."""
        if self._cores is None or self._original_shape is None:
            return 1.0
        original_size = np.prod(self._original_shape)
        tt_size = sum(c.size for c in self._cores)
        return float(original_size / max(tt_size, 1))

    def get_ranks(self) -> Optional[Tuple[int, ...]]:
        """Return the TT-ranks."""
        return self._tt_ranks
