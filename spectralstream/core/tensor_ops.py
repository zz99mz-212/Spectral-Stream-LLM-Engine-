"""DEPRECATED — Tensor operations module.

This module is deprecated and has been moved to _archive/core/tensor_ops.py.
Tensor folding/unfolding is now handled directly by the compression methods
that need them (e.g., CPDecomposition).

The archived copy is preserved for reference but will be removed in a future version.
"""

import warnings as _warnings

_warnings.warn(
    "core.tensor_ops is deprecated. Tensor operations are now handled "
    "directly by the compression methods that need them.",
    DeprecationWarning,
    stacklevel=2,
)

import numpy as np


def fold(tensor: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    return tensor.reshape(shape)


def unfold(tensor: np.ndarray, mode: int) -> np.ndarray:
    n_modes = tensor.ndim
    return np.moveaxis(tensor, mode, 0).reshape(tensor.shape[mode], -1)


def mode_n_product(tensor: np.ndarray, matrix: np.ndarray, mode: int) -> np.ndarray:
    n_modes = tensor.ndim
    perm = list(range(n_modes))
    perm[0], perm[mode] = perm[mode], perm[0]
    tensor_t = np.transpose(tensor, perm)
    shape_t = tensor_t.shape
    unfolded = tensor_t.reshape(shape_t[0], -1)
    result = matrix @ unfolded
    new_shape = (matrix.shape[0],) + shape_t[1:]
    result = result.reshape(new_shape)
    return np.transpose(result, perm)


__all__ = ["fold", "unfold", "mode_n_product"]
