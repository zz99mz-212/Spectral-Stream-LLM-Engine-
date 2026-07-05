"""Module extracted from quantization_tuning.py — tunequantumstate."""

from __future__ import annotations

import math
from typing import Optional

from . import FP32_BYTES, TunedParams


def tune_quantum_state(
    target_ratio: float, n_elements: int, sigma_sq: float = 1.0
) -> TunedParams:
    """Tune QuantumState (bond_dim). MPS bond dimension."""
    best: Optional[TunedParams] = None
    best_err = float("inf")
    for bond_dim in (2, 4, 6, 8, 12, 16, 24, 32):
        n_sites = int(math.log2(n_elements)) if n_elements > 1 else 1
        stored = n_sites * bond_dim * 2 * bond_dim * 4
        total = n_elements * 4
        ratio = total / stored if stored > 0 else float("inf")
        mse = sigma_sq / (bond_dim**2)
        err = abs(ratio - target_ratio)
        if err < best_err:
            best_err = err
            best = TunedParams(
                method_name="quantum_state",
                params={"bond_dim": bond_dim},
                estimated_ratio=ratio,
                estimated_mse=mse,
            )
    assert best is not None
    return best


def tune_quantum_tensor_network(
    target_ratio: float, n_elements: int, sigma_sq: float = 1.0
) -> TunedParams:
    """Tune QuantumTensorNetwork (bond_dim). MPS bond compression."""
    result = tune_quantum_state(target_ratio, n_elements, sigma_sq)
    result.method_name = "quantum_tensor_network"
    return result
