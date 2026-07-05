# --- mse.py ---
"""Module extracted from fractal_entropy_cross.py — mse."""

from __future__ import annotations


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))
def _snr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    signal = np.var(original)
    noise = _mse(original, reconstructed)
    return float(10 * np.log10(signal / noise)) if noise > 1e-30 else float("inf")