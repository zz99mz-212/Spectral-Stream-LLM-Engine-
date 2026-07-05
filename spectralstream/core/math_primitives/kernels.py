"""Yukawa / spectral kernel functions and band constants."""

import numpy as np

BAND_HIGH: int = 0
BAND_NORMAL: int = 1
BAND_LOW: int = 2
BAND_COMPRESSION: str = "compression"


def yukawa_kernel_1d(
    x: np.ndarray, mass: float = 1.0, amplitude: float = 1.0
) -> np.ndarray:
    r = np.abs(x)
    return amplitude * np.exp(-mass * r) / (r + 1e-30)


def apply_spectral_kernel(
    spectrum: np.ndarray, kernel: np.ndarray, axis: int = -1
) -> np.ndarray:
    spectrum = np.asarray(spectrum, dtype=np.complex128)
    kernel = np.asarray(kernel, dtype=np.float64)
    if kernel.ndim > 1:
        kernel = kernel.ravel()
    k_len = len(kernel)
    s = [slice(None)] * spectrum.ndim
    s[axis] = slice(0, min(k_len, spectrum.shape[axis]))
    result = spectrum.copy()
    result[tuple(s)] *= kernel[: spectrum.shape[axis]]
    return result
