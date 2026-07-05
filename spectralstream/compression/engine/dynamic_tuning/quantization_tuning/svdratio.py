"""Module extracted from quantization_tuning.py — svdratio."""

from __future__ import annotations

import math
from typing import Optional, Tuple

from . import FP32_BYTES, TunedParams


def _svd_ratio(rank: int, shape: Tuple[int, ...]) -> Tuple[float, float]:
    """Estimate ratio and MSE for truncated SVD of a 2D matrix.
    Returns (ratio, mse_factor).
    Storage: U (m x r) + S (r) + Vt (r x n) in float32.
    """
    if len(shape) < 2:
        return 1.0, 1.0
    m, n = int(shape[0]), int(shape[1])
    stored = rank * (m + n + 1) * 4
    total = m * n * 4
    ratio = total / stored if stored > 0 else float("inf")
    mse_factor = 1.0 - (rank / min(m, n))
    return ratio, max(mse_factor, 0.001)


def tune_density_matrix(
    target_ratio: float, n_elements: int, sigma_sq: float = 1.0
) -> TunedParams:
    """Tune DensityMatrix (rank). SVD-based density matrix approximation."""
    best: Optional[TunedParams] = None
    best_err = float("inf")
    shape = (int(math.isqrt(n_elements)), n_elements // int(math.isqrt(n_elements)))
    for rank in (1, 2, 4, 8, 16, 32, 64, 128):
        if rank > min(shape):
            continue
        ratio, mse_factor = _svd_ratio(rank, shape)
        mse = mse_factor * sigma_sq
        err = abs(ratio - target_ratio)
        if err < best_err:
            best_err = err
            best = TunedParams(
                method_name="density_matrix",
                params={"rank": rank},
                estimated_ratio=ratio,
                estimated_mse=mse,
            )
    assert best is not None
    return best


def tune_quantum_entanglement(
    target_ratio: float, n_elements: int, sigma_sq: float = 1.0
) -> TunedParams:
    """Tune QuantumEntanglement (rank). Schmidt decomposition => DensityMatrix."""
    result = tune_density_matrix(target_ratio, n_elements, sigma_sq)
    result.method_name = "quantum_entanglement"
    return result


def tune_gyrokinetic(
    target_ratio: float, n_elements: int, sigma_sq: float = 1.0
) -> TunedParams:
    """Tune Gyrokinetic (n_gyro_angles). Reduced plasma model."""
    best: Optional[TunedParams] = None
    best_err = float("inf")
    for n_gyro in (2, 4, 8, 16, 32):
        m = int(math.isqrt(n_elements))
        n = n_elements // m
        r = min(n_gyro, min(m, n) // 4)
        if r < 1:
            r = 1
        ratio, mse_factor = _svd_ratio(r, (m, n))
        mse = mse_factor * sigma_sq
        err = abs(ratio - target_ratio)
        if err < best_err:
            best_err = err
            best = TunedParams(
                method_name="gyrokinetic",
                params={"n_gyro_angles": n_gyro},
                estimated_ratio=ratio,
                estimated_mse=mse,
            )
    assert best is not None
    return best


def tune_resonance_modes(
    target_ratio: float, n_elements: int, sigma_sq: float = 1.0
) -> TunedParams:
    """Tune ResonanceModes (energy_threshold). SVD energy thresholding."""
    shape = (int(math.isqrt(n_elements)), n_elements // int(math.isqrt(n_elements)))
    best: Optional[TunedParams] = None
    best_err = float("inf")
    for energy_keep in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999):
        r = max(1, int(min(shape) * (1 - energy_keep)))
        ratio, mse_factor = _svd_ratio(r, shape)
        mse = (1 - energy_keep) * sigma_sq
        err = abs(ratio - target_ratio)
        if err < best_err:
            best_err = err
            best = TunedParams(
                method_name="resonance_modes",
                params={"energy_threshold": energy_keep},
                estimated_ratio=ratio,
                estimated_mse=mse,
            )
    assert best is not None
    return best


def tune_resonance_compression(
    target_ratio: float, n_elements: int, sigma_sq: float = 1.0
) -> TunedParams:
    """Tune ResonanceCompression (energy_threshold). Same as ResonanceModes."""
    result = tune_resonance_modes(target_ratio, n_elements, sigma_sq)
    result.method_name = "resonance_compression"
    return result


def tune_topological_functional(
    target_ratio: float, n_elements: int, sigma_sq: float = 1.0
) -> TunedParams:
    """Tune TopologicalFunctional (rank). SVD-based geometric codebook."""
    result = tune_resonance_modes(target_ratio, n_elements, sigma_sq)
    result.method_name = "topological_functional"
    return result


def tune_manifold_learning(
    target_ratio: float, n_elements: int, sigma_sq: float = 1.0
) -> TunedParams:
    """Tune ManifoldLearning (n_components). PCA-style via SVD."""
    result = tune_density_matrix(target_ratio, n_elements, sigma_sq)
    result.method_name = "manifold_learning"
    return result


def tune_hamiltonian_dynamical(
    target_ratio: float, n_elements: int, sigma_sq: float = 1.0
) -> TunedParams:
    """Tune HamiltonianDynamical (n_modes). SVD mode truncation."""
    result = tune_density_matrix(target_ratio, n_elements, sigma_sq)
    result.method_name = "hamiltonian_dynamical"
    return result
