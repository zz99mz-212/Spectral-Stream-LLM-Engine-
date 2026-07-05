"""
Advanced Sparsity Engine for SpectralStream
=============================================
Structured sparsity with Hessian-based importance, Group Lasso,
N:M hardware-friendly patterns, block sparsity, and combined
sparse+quantize compression.

Architecture:
  1. SparseGPTPruner — Hessian-based importance, Group Lasso
  2. StructuredSparsityEngine — 2:4, 4:8, N:M, block sparsity
  3. AdaptiveSparsityAllocator — per-layer sparsity optimization
  4. SparseTensorCompressor — sparse + quantize combined
"""

from __future__ import annotations


import logging
import math
import threading
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

logger = logging.getLogger(__name__)

__all__ = [
    "SparseGPTPruner",
    "StructuredSparsityEngine",
    "AdaptiveSparsityAllocator",
    "SparseTensorCompressor",
]


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class AdvancedSparsityConfig:
    """Configuration for advanced sparsity."""

    target_sparsity: float = 0.9
    nm_pattern: Tuple[int, int] = (2, 4)
    block_size: int = 32
    group_lasso_lambda: float = 0.01
    hessian_damping: float = 0.01
    hessian_block_size: int = 128
    quantize_bits: int = 4
    adaptive_enabled: bool = True


@dataclass
class PruningResult:
    """Result of a pruning operation."""

    mask: np.ndarray
    weights: np.ndarray
    sparsity: float
    importance_scores: np.ndarray
    strategy: str
    metrics: Dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════
# 1. SparseGPTPruner — Hessian-based importance + Group Lasso
# ═══════════════════════════════════════════════════════════════════════════


class SparseGPTPruner:
    """Hessian-based structured pruning with Group Lasso.

    SparseGPT computes weight importance via the inverse Hessian
    diagonal, which captures how much each weight contributes to
    the output loss. Group Lasso adds group-level sparsity by
    penalizing entire rows/columns jointly.

    The combined importance score is:
        I_ij = w_ij^2 * [H^{-1}]_jj + lambda * ||W_group||_2
    """

    def __init__(self, config: Optional[AdvancedSparsityConfig] = None):
        self.config = config or AdvancedSparsityConfig()
        self._hessian_cache: Dict[str, np.ndarray] = {}

    def _compute_hessian_inverse(self, hessian: np.ndarray) -> np.ndarray:
        """Compute damped inverse Hessian."""
        n = hessian.shape[0]
        H = hessian + self.config.hessian_damping * np.eye(n, dtype=np.float64)
        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            H_inv = np.linalg.pinv(H)
        return H_inv

    def _group_lasso_importance(
        self, weights: np.ndarray, group_size: int = 32
    ) -> np.ndarray:
        """Compute Group Lasso importance scores.

        Groups are contiguous blocks of weights; importance is the
        L2 norm of each group.
        """
        rows, cols = weights.shape
        scores = np.zeros_like(weights, dtype=np.float64)

        for i in range(0, rows, group_size):
            end_i = min(i + group_size, rows)
            for j in range(0, cols, group_size):
                end_j = min(j + group_size, cols)
                group = weights[i:end_i, j:end_j]
                group_norm = float(np.linalg.norm(group))
                scores[i:end_i, j:end_j] = group_norm

        return scores

    def compute_importance(
        self,
        weights: np.ndarray,
        hessian: Optional[np.ndarray] = None,
        use_group_lasso: bool = True,
    ) -> np.ndarray:
        """Compute combined SparseGPT + Group Lasso importance.

        Args:
            weights: Weight matrix of shape (rows, cols).
            hessian: Optional Hessian matrix of shape (cols, cols).
            use_group_lasso: Whether to include Group Lasso term.

        Returns:
            Importance scores of same shape as weights.
        """
        weights = np.asarray(weights, dtype=np.float64)
        rows, cols = weights.shape

        # SparseGPT importance: w^2 * diag(H^{-1})
        if hessian is not None:
            H_inv = self._compute_hessian_inverse(hessian)
            hessian_diag = np.abs(np.diag(H_inv))
        else:
            hessian_diag = np.ones(cols, dtype=np.float64)

        sparsegpt_scores = weights**2 * hessian_diag[np.newaxis, :]

        if use_group_lasso:
            group_scores = self._group_lasso_importance(weights, self.config.block_size)
            # Combine
            scores = sparsegpt_scores + self.config.group_lasso_lambda * group_scores
        else:
            scores = sparsegpt_scores

        return scores

    def prune(
        self,
        weights: np.ndarray,
        target_sparsity: Optional[float] = None,
        hessian: Optional[np.ndarray] = None,
    ) -> PruningResult:
        """Prune weights using SparseGPT + Group Lasso.

        Args:
            weights: Weight matrix.
            target_sparsity: Target fraction of zeros (0-1).
            hessian: Optional Hessian for importance computation.

        Returns:
            PruningResult with mask, pruned weights, and metrics.
        """
        weights = np.asarray(weights, dtype=np.float64)
        sparsity = target_sparsity or self.config.target_sparsity

        importance = self.compute_importance(weights, hessian, use_group_lasso=True)

        # Create mask by thresholding importance
        threshold = float(np.percentile(importance, sparsity * 100))
        mask = importance >= threshold

        pruned = np.where(mask, weights, 0.0)
        actual_sparsity = 1.0 - float(np.mean(mask))

        return PruningResult(
            mask=mask,
            weights=pruned,
            sparsity=actual_sparsity,
            importance_scores=importance,
            strategy="sparsegpt_grouplasso",
            metrics={
                "threshold": threshold,
                "hessian_provided": hessian is not None,
                "group_lasso_lambda": self.config.group_lasso_lambda,
            },
        )


# ═══════════════════════════════════════════════════════════════════════════
# 2. StructuredSparsityEngine — N:M, block, channel sparsity
# ═══════════════════════════════════════════════════════════════════════════


class SparsityPattern:
    """Sparsity pattern types."""

    N_M = "n_m"
    BLOCK = "block"
    CHANNEL = "channel"
    ROW = "row"
    TILED = "tiled"


class StructuredSparsityEngine:
    """Hardware-friendly structured sparsity patterns.

    Supports:
    - N:M sparsity (e.g., 2:4 on Ampere GPUs)
    - Block sparsity (contiguous zero blocks)
    - Channel pruning (remove entire channels)
    - Row pruning (remove entire rows)
    - Tiled sparsity (tile-level patterns)
    """

    def __init__(self, config: Optional[AdvancedSparsityConfig] = None):
        self.config = config or AdvancedSparsityConfig()

    def apply_nm_sparsity(
        self,
        weights: np.ndarray,
        n: Optional[int] = None,
        m: Optional[int] = None,
    ) -> PruningResult:
        """Apply N:M structured sparsity.

        For each group of M consecutive elements, keep at most N
        with the largest magnitude.

        Args:
            weights: Weight matrix.
            n: Number of non-zeros per group (default from config).
            m: Group size (default from config).

        Returns:
            PruningResult with N:M mask.
        """
        weights = np.asarray(weights, dtype=np.float64)
        n_val = n or self.config.nm_pattern[0]
        m_val = m or self.config.nm_pattern[1]

        flat = weights.ravel()
        n_elements = len(flat)
        padded_len = int(math.ceil(n_elements / m_val) * m_val)
        padded = np.zeros(padded_len, dtype=np.float64)
        padded[:n_elements] = flat

        mask = np.zeros(padded_len, dtype=bool)
        for i in range(0, padded_len, m_val):
            group = padded[i : i + m_val]
            abs_group = np.abs(group)
            top_n_idx = np.argsort(abs_group)[-n_val:]
            mask[i : i + m_val][top_n_idx] = True

        mask = mask[:n_elements]
        pruned = np.where(mask, flat, 0.0).reshape(weights.shape)
        actual_sparsity = 1.0 - float(np.mean(mask))

        return PruningResult(
            mask=mask.reshape(weights.shape),
            weights=pruned,
            sparsity=actual_sparsity,
            importance_scores=np.abs(weights),
            strategy=f"nm_{n_val}:{m_val}",
            metrics={"n": n_val, "m": m_val},
        )

    def apply_block_sparsity(
        self,
        weights: np.ndarray,
        block_h: Optional[int] = None,
        block_w: Optional[int] = None,
        target_sparsity: Optional[float] = None,
    ) -> PruningResult:
        """Apply block-level sparsity.

        Zeros out entire blocks based on block norm.

        Args:
            weights: Weight matrix.
            block_h: Block height (default from config).
            block_w: Block width (default from config).
            target_sparsity: Target sparsity level.

        Returns:
            PruningResult with block mask.
        """
        weights = np.asarray(weights, dtype=np.float64)
        bh = block_h or self.config.block_size
        bw = block_w or self.config.block_size
        sparsity = target_sparsity or self.config.target_sparsity

        rows, cols = weights.shape
        mask = np.ones((rows, cols), dtype=bool)

        block_norms = []
        block_positions = []
        for i in range(0, rows, bh):
            for j in range(0, cols, bw):
                block = weights[i : i + bh, j : j + bw]
                block_norms.append(float(np.linalg.norm(block)))
                block_positions.append((i, j))

        if not block_norms:
            return PruningResult(
                mask=mask,
                weights=weights,
                sparsity=0.0,
                importance_scores=np.abs(weights),
                strategy="block",
            )

        n_blocks_to_prune = int(len(block_norms) * sparsity)
        sorted_idx = np.argsort(block_norms)

        for idx in sorted_idx[:n_blocks_to_prune]:
            i, j = block_positions[idx]
            mask[i : i + bh, j : j + bw] = False

        pruned = np.where(mask, weights, 0.0)
        actual_sparsity = 1.0 - float(np.mean(mask))

        return PruningResult(
            mask=mask,
            weights=pruned,
            sparsity=actual_sparsity,
            importance_scores=np.array(block_norms),
            strategy="block",
            metrics={"block_h": bh, "block_w": bw},
        )

    def apply_channel_pruning(
        self,
        weights: np.ndarray,
        keep_fraction: float = 0.7,
    ) -> PruningResult:
        """Prune entire output channels by L2 norm."""
        weights = np.asarray(weights, dtype=np.float64)
        row_norms = np.linalg.norm(weights, axis=1)

        n_keep = max(1, int(weights.shape[0] * keep_fraction))
        keep_idx = np.argsort(-row_norms)[:n_keep]

        mask = np.zeros(weights.shape, dtype=bool)
        mask[keep_idx] = True

        pruned = np.where(mask, weights, 0.0)
        actual_sparsity = 1.0 - float(np.mean(mask))

        return PruningResult(
            mask=mask,
            weights=pruned,
            sparsity=actual_sparsity,
            importance_scores=row_norms,
            strategy="channel",
            metrics={"n_keep": n_keep, "keep_fraction": keep_fraction},
        )

    def apply_row_pruning(
        self,
        weights: np.ndarray,
        target_sparsity: Optional[float] = None,
    ) -> PruningResult:
        """Prune entire rows by L2 norm."""
        weights = np.asarray(weights, dtype=np.float64)
        sparsity = target_sparsity or self.config.target_sparsity

        row_norms = np.linalg.norm(weights, axis=1)
        threshold = float(np.percentile(row_norms, sparsity * 100))

        mask = row_norms >= threshold
        mask_2d = mask[:, np.newaxis] * np.ones(weights.shape, dtype=bool)

        pruned = np.where(mask_2d, weights, 0.0)
        actual_sparsity = 1.0 - float(np.mean(mask_2d))

        return PruningResult(
            mask=mask_2d,
            weights=pruned,
            sparsity=actual_sparsity,
            importance_scores=row_norms,
            strategy="row",
        )

    def apply_tiled_sparsity(
        self,
        weights: np.ndarray,
        tile_size: int = 64,
        target_sparsity: Optional[float] = None,
        seed: int = 42,
    ) -> PruningResult:
        """Apply tile-level sparsity pattern."""
        weights = np.asarray(weights, dtype=np.float64)
        sparsity = target_sparsity or self.config.target_sparsity
        rows, cols = weights.shape

        n_tiles_h = int(math.ceil(rows / tile_size))
        n_tiles_w = int(math.ceil(cols / tile_size))

        rng = np.random.RandomState(seed)
        tile_pattern = rng.random((n_tiles_h, n_tiles_w)) > sparsity

        mask = np.zeros((rows, cols), dtype=bool)
        for ti in range(n_tiles_h):
            for tj in range(n_tiles_w):
                if tile_pattern[ti, tj]:
                    i0 = ti * tile_size
                    i1 = min(i0 + tile_size, rows)
                    j0 = tj * tile_size
                    j1 = min(j0 + tile_size, cols)
                    mask[i0:i1, j0:j1] = True

        pruned = np.where(mask, weights, 0.0)
        actual_sparsity = 1.0 - float(np.mean(mask))

        return PruningResult(
            mask=mask,
            weights=pruned,
            sparsity=actual_sparsity,
            importance_scores=np.abs(weights),
            strategy="tiled",
            metrics={"tile_size": tile_size},
        )


# ═══════════════════════════════════════════════════════════════════════════
# 3. AdaptiveSparsityAllocator — per-layer optimization
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class LayerSparsityState:
    """State tracking for a layer's sparsity."""

    layer_name: str
    base_sparsity: float
    current_sparsity: float
    sensitivity: float
    error_history: List[float] = field(default_factory=list)
    sparsity_history: List[float] = field(default_factory=list)


class AdaptiveSparsityAllocator:
    """Optimize per-layer sparsity allocation based on sensitivity.

    Sensitive layers get lower sparsity (less pruning) while
    robust layers get higher sparsity (more pruning), optimizing
    overall model quality under a global sparsity budget.
    """

    def __init__(self, config: Optional[AdvancedSparsityConfig] = None):
        self.config = config or AdvancedSparsityConfig()
        self._layers: Dict[str, LayerSparsityState] = {}
        self._lock = threading.Lock()

    def register_layer(self, name: str, base_sparsity: Optional[float] = None):
        """Register a layer for adaptive sparsity allocation."""
        with self._lock:
            self._layers[name] = LayerSparsityState(
                layer_name=name,
                base_sparsity=base_sparsity or self.config.target_sparsity,
                current_sparsity=base_sparsity or self.config.target_sparsity,
                sensitivity=1.0,
            )

    def compute_sensitivity(
        self,
        name: str,
        weights: np.ndarray,
        gradients: Optional[np.ndarray] = None,
    ) -> float:
        """Compute layer sensitivity to pruning.

        Sensitivity is based on:
        - Weight magnitude distribution
        - Gradient-weight correlation (if available)
        - Spectral entropy
        """
        weights = np.asarray(weights, dtype=np.float64)

        # Magnitude-based: high variance → more sensitive
        weight_std = float(np.std(weights))
        weight_mean = float(np.abs(np.mean(weights)))
        mag_sensitivity = weight_std / (weight_mean + 1e-10)

        # Spectral: high entropy → more sensitive
        ent = spectral_entropy(weights.ravel())
        spectral_sensitivity = ent

        # Gradient-based: high grad-weight correlation → more sensitive
        grad_sensitivity = 0.0
        if gradients is not None:
            gradients = np.asarray(gradients, dtype=np.float64)
            if gradients.shape == weights.shape:
                corr = np.corrcoef(weights.ravel(), gradients.ravel())[0, 1]
                grad_sensitivity = abs(corr) if np.isfinite(corr) else 0.0

        sensitivity = (
            0.3 * mag_sensitivity + 0.3 * spectral_sensitivity + 0.4 * grad_sensitivity
        )

        with self._lock:
            if name in self._layers:
                self._layers[name].sensitivity = sensitivity

        return sensitivity

    def allocate_sparsity(
        self,
        global_budget: float = 0.9,
    ) -> Dict[str, float]:
        """Allocate per-layer sparsity to maximize quality under budget.

        Sensitive layers get lower sparsity, robust layers get higher.

        Args:
            global_budget: Global target sparsity (0-1).

        Returns:
            Dict of {layer_name: allocated_sparsity}.
        """
        with self._lock:
            if not self._layers:
                return {}

            # Sort by sensitivity (ascending = least sensitive first)
            sorted_layers = sorted(self._layers.values(), key=lambda s: s.sensitivity)

            n = len(sorted_layers)
            allocations = {}

            for i, state in enumerate(sorted_layers):
                # Least sensitive → highest sparsity
                rank_fraction = i / max(n - 1, 1)
                allocated = global_budget * (0.8 + 0.4 * (1.0 - rank_fraction))
                allocated = np.clip(allocated, 0.1, 0.99)
                state.current_sparsity = allocated
                allocations[state.layer_name] = allocated

            return allocations

    def record_error(self, name: str, error: float):
        """Record post-pruning error for a layer."""
        with self._lock:
            if name in self._layers:
                state = self._layers[name]
                state.error_history.append(error)
                if len(state.error_history) > 100:
                    state.error_history = state.error_history[-100:]

    def get_sparsity(self, name: str) -> float:
        """Get current sparsity for a layer."""
        with self._lock:
            if name in self._layers:
                return self._layers[name].current_sparsity
            return self.config.target_sparsity

    def get_stats(self) -> Dict[str, Dict]:
        """Get statistics for all layers."""
        with self._lock:
            stats = {}
            for name, state in self._layers.items():
                stats[name] = {
                    "base_sparsity": state.base_sparsity,
                    "current_sparsity": state.current_sparsity,
                    "sensitivity": state.sensitivity,
                    "mean_error": (
                        float(np.mean(state.error_history))
                        if state.error_history
                        else 0.0
                    ),
                }
            return stats


# ═══════════════════════════════════════════════════════════════════════════
# 4. SparseTensorCompressor — sparse + quantize combined
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class SparseQuantizedResult:
    """Combined sparse + quantized representation."""

    sparse_mask: np.ndarray
    quantized_values: np.ndarray
    quantized_indices: np.ndarray
    codebook: np.ndarray
    scale: float
    original_shape: Tuple[int, ...]
    sparsity: float
    bits_per_element: float
    compression_ratio: float


class SparseTensorCompressor:
    """Combined sparse + quantize compression.

    Pipeline:
    1. Identify and zero-out unimportant weights (sparsity)
    2. Quantize remaining non-zero weights
    3. Store sparse structure + quantized values

    Achieves multiplicative compression from both sparsity and
    quantization (e.g., 90% sparse + 4-bit quant = ~80x compression).
    """

    def __init__(self, config: Optional[AdvancedSparsityConfig] = None):
        self.config = config or AdvancedSparsityConfig()
        self._quantizer = LloydMaxQuantizer(n_bits=self.config.quantize_bits)

    def compress(
        self,
        weights: np.ndarray,
        target_sparsity: Optional[float] = None,
        n_bits: Optional[int] = None,
    ) -> SparseQuantizedResult:
        """Compress weights with combined sparse + quantize.

        Args:
            weights: Input weight tensor.
            target_sparsity: Fraction of zeros (0-1).
            n_bits: Quantization bits.

        Returns:
            SparseQuantizedResult with combined representation.
        """
        weights = np.asarray(weights, dtype=np.float64)
        sparsity = target_sparsity or self.config.target_sparsity
        bits = n_bits or self.config.quantize_bits
        original_shape = weights.shape

        # Step 1: Magnitude-based pruning
        abs_weights = np.abs(weights)
        threshold = float(np.percentile(abs_weights, sparsity * 100))
        mask = abs_weights >= threshold

        non_zero_values = weights[mask]
        actual_sparsity = 1.0 - float(np.mean(mask))

        # Step 2: Quantize non-zero values
        if len(non_zero_values) == 0:
            return SparseQuantizedResult(
                sparse_mask=mask,
                quantized_values=np.array([]),
                quantized_indices=np.array([], dtype=np.int32),
                codebook=np.array([]),
                scale=1.0,
                original_shape=original_shape,
                sparsity=actual_sparsity,
                bits_per_element=0.0,
                compression_ratio=float("inf"),
            )

        scale = float(np.max(np.abs(non_zero_values)) + 1e-10)
        normalized = non_zero_values / scale

        n_levels = 1 << bits
        centroids = np.linspace(-1.0, 1.0, n_levels)
        indices = np.digitize(normalized, (centroids[:-1] + centroids[1:]) / 2)
        indices = np.clip(indices, 0, n_levels - 1).astype(np.int32)

        # Step 3: Compute metrics
        original_bits = weights.size * 32
        sparse_bits = (
            np.count_nonzero(mask) * (bits + 32) + weights.size
        )  # mask + values
        compression_ratio = original_bits / max(sparse_bits, 1)
        bits_per_element = sparse_bits / max(weights.size, 1)

        return SparseQuantizedResult(
            sparse_mask=mask,
            quantized_values=centroids[indices],
            quantized_indices=indices,
            codebook=centroids * scale,
            scale=scale,
            original_shape=original_shape,
            sparsity=actual_sparsity,
            bits_per_element=bits_per_element,
            compression_ratio=compression_ratio,
        )

    def decompress(self, result: SparseQuantizedResult) -> np.ndarray:
        """Decompress sparse + quantized representation.

        Args:
            result: SparseQuantizedResult from compress().

        Returns:
            Reconstructed weight tensor.
        """
        output = np.zeros(result.original_shape, dtype=np.float64)
        output[result.sparse_mask] = result.quantized_values
        return output

    def compress_with_strategy(
        self,
        weights: np.ndarray,
        strategy: str = "magnitude",
        target_sparsity: Optional[float] = None,
        n_bits: Optional[int] = None,
    ) -> SparseQuantizedResult:
        """Compress using a specific pruning strategy.

        Args:
            weights: Input weights.
            strategy: Pruning strategy ("magnitude", "spectral", "group_lasso").
            target_sparsity: Target sparsity.
            n_bits: Quantization bits.

        Returns:
            SparseQuantizedResult.
        """
        weights = np.asarray(weights, dtype=np.float64)

        if strategy == "magnitude":
            pass  # Default
        elif strategy == "spectral":
            # Spectral pruning: prune in DCT domain
            coeffs = dct(weights.ravel())
            power = coeffs**2
            threshold = float(
                np.percentile(
                    power, (target_sparsity or self.config.target_sparsity) * 100
                )
            )
            mask = np.zeros(weights.shape, dtype=bool)
            flat_mask = power >= threshold
            mask = flat_mask.reshape(weights.shape)
            # Override default pruning
            abs_weights = np.abs(weights)
            actual_mask = mask | (
                abs_weights
                >= float(
                    np.percentile(
                        abs_weights,
                        (target_sparsity or self.config.target_sparsity) * 100,
                    )
                )
            )
            non_zero_values = weights[actual_mask]
            sparsity = 1.0 - float(np.mean(actual_mask))

            bits = n_bits or self.config.quantize_bits
            if len(non_zero_values) == 0:
                return SparseQuantizedResult(
                    sparse_mask=actual_mask,
                    quantized_values=np.array([]),
                    quantized_indices=np.array([], dtype=np.int32),
                    codebook=np.array([]),
                    scale=1.0,
                    original_shape=weights.shape,
                    sparsity=sparsity,
                    bits_per_element=0.0,
                    compression_ratio=float("inf"),
                )

            scale = float(np.max(np.abs(non_zero_values)) + 1e-10)
            normalized = non_zero_values / scale
            n_levels = 1 << bits
            centroids = np.linspace(-1.0, 1.0, n_levels)
            indices = np.digitize(normalized, (centroids[:-1] + centroids[1:]) / 2)
            indices = np.clip(indices, 0, n_levels - 1).astype(np.int32)

            original_bits = weights.size * 32
            sparse_bits = np.count_nonzero(actual_mask) * (bits + 32) + weights.size
            return SparseQuantizedResult(
                sparse_mask=actual_mask,
                quantized_values=centroids[indices],
                quantized_indices=indices,
                codebook=centroids * scale,
                scale=scale,
                original_shape=weights.shape,
                sparsity=sparsity,
                bits_per_element=sparse_bits / max(weights.size, 1),
                compression_ratio=original_bits / max(sparse_bits, 1),
            )
        elif strategy == "group_lasso":
            # Group Lasso: prune by groups
            pruner = SparseGPTPruner(self.config)
            result = pruner.prune(weights, target_sparsity)
            mask = result.mask
            non_zero_values = weights[mask]
            sparsity = 1.0 - float(np.mean(mask))
            bits = n_bits or self.config.quantize_bits

            if len(non_zero_values) == 0:
                return SparseQuantizedResult(
                    sparse_mask=mask,
                    quantized_values=np.array([]),
                    quantized_indices=np.array([], dtype=np.int32),
                    codebook=np.array([]),
                    scale=1.0,
                    original_shape=weights.shape,
                    sparsity=sparsity,
                    bits_per_element=0.0,
                    compression_ratio=float("inf"),
                )

            scale = float(np.max(np.abs(non_zero_values)) + 1e-10)
            normalized = non_zero_values / scale
            n_levels = 1 << bits
            centroids = np.linspace(-1.0, 1.0, n_levels)
            indices = np.digitize(normalized, (centroids[:-1] + centroids[1:]) / 2)
            indices = np.clip(indices, 0, n_levels - 1).astype(np.int32)

            original_bits = weights.size * 32
            sparse_bits = np.count_nonzero(mask) * (bits + 32) + weights.size
            return SparseQuantizedResult(
                sparse_mask=mask,
                quantized_values=centroids[indices],
                quantized_indices=indices,
                codebook=centroids * scale,
                scale=scale,
                original_shape=weights.shape,
                sparsity=sparsity,
                bits_per_element=sparse_bits / max(weights.size, 1),
                compression_ratio=original_bits / max(sparse_bits, 1),
            )

        return self.compress(weights, target_sparsity, n_bits)
