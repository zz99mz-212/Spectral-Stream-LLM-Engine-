"""Module extracted from quantization_tuning.py — tuneplasmafield."""

from __future__ import annotations

from typing import Optional

from . import FP32_BYTES, TunedParams


def tune_plasma_field(
    target_ratio: float, n_elements: int, sigma_sq: float = 1.0
) -> TunedParams:
    """Tune PlasmaField (keep_fraction). DFT with dominant mode retention."""
    best: Optional[TunedParams] = None
    best_err = float("inf")
    for keep_frac in (0.01, 0.02, 0.05, 0.1, 0.2, 0.5):
        bpe = keep_frac * 8.0
        ratio = FP32_BYTES / bpe if bpe > 0 else float("inf")
        mse = (1 - keep_frac) * sigma_sq * 0.3
        err = abs(ratio - target_ratio)
        if err < best_err:
            best_err = err
            best = TunedParams(
                method_name="plasma_field",
                params={"keep_fraction": keep_frac},
                estimated_ratio=ratio,
                estimated_mse=mse,
            )
    assert best is not None
    return best


def tune_harmonic_oscillator(
    target_ratio: float, n_elements: int, sigma_sq: float = 1.0
) -> TunedParams:
    """Tune HarmonicOscillator (keep_fraction). 2D DFT mode retention."""
    result = tune_plasma_field(target_ratio, n_elements, sigma_sq)
    result.method_name = "harmonic_oscillator"
    return result


def tune_fourier_neural_op(
    target_ratio: float, n_elements: int, sigma_sq: float = 1.0
) -> TunedParams:
    """Tune FourierNeuralOp (keep_fraction). FFT + spectral filter."""
    result = tune_plasma_field(target_ratio, n_elements, sigma_sq)
    result.method_name = "fourier_neural_op"
    return result
