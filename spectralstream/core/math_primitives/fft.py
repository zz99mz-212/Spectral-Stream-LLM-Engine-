"""FFT helpers (numpy-based, no scipy dependency)."""

import numpy as np


def fft(x: np.ndarray, axis: int = -1) -> np.ndarray:
    return np.fft.fft(x, axis=axis)


def ifft(x: np.ndarray, axis: int = -1) -> np.ndarray:
    return np.fft.ifft(x, axis=axis)


def rfft(x: np.ndarray, axis: int = -1) -> np.ndarray:
    return np.fft.rfft(x, axis=axis)


def irfft(x: np.ndarray, n: int, axis: int = -1) -> np.ndarray:
    return np.fft.irfft(x, n=n, axis=axis)


def fftfreq(n: int, d: float = 1.0) -> np.ndarray:
    return np.fft.fftfreq(n, d=d)
