"""Spectral analysis: entropy, power density, band limit, energy concentration."""

import numpy as np


def spectral_entropy(x: np.ndarray, n_bins: int = 64) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    if len(x) < 4:
        return 0.0
    try:
        power = np.abs(np.fft.fft(x)) ** 2
        psd = power[: len(power) // 2] + 1e-30
        psd /= np.sum(psd)
        entropy = -np.sum(psd * np.log2(psd))
        max_entropy = np.log2(len(psd))
        return float(entropy / max_entropy) if max_entropy > 0 else 0.0
    except Exception:
        return 0.0


def spectral_power_density(x: np.ndarray, n_bins: int = 64) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    if len(x) < 2:
        return np.array([0.0])
    power = np.abs(np.fft.fft(x)) ** 2
    half = len(power) // 2
    return power[:half]


def band_limit(x: np.ndarray, cutoff_ratio: float = 0.1) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    orig_shape = x.shape
    flat = x.ravel()
    n = len(flat)
    spectrum = np.fft.fft(flat)
    cutoff = max(1, int(n * cutoff_ratio))
    spectrum[cutoff:-cutoff] = 0.0
    return np.fft.ifft(spectrum).real.reshape(orig_shape)


def energy_concentration(x: np.ndarray, top_fraction: float = 0.1) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    if len(x) < 2:
        return 1.0
    try:
        power = np.abs(np.fft.fft(x)) ** 2
        total = float(np.sum(power))
        if total < 1e-30:
            return 1.0
        sorted_power = np.sort(power)[::-1]
        n_top = max(1, int(len(sorted_power) * top_fraction))
        return float(np.sum(sorted_power[:n_top]) / total)
    except Exception:
        return 0.0


def auto_keep_fraction(coeffs: np.ndarray, target_energy: float = 0.99) -> float:
    """Determine fraction of largest coefficients needed to retain target_energy.

    Works on transform-domain coefficients (DCT, FWHT, wavelet, Fourier).
    For noisy weight tensors with spread energy, returns ~0.8-0.99 (low compression).
    For structured weights with concentrated energy, returns ~0.1-0.5 (high compression).

    Parameters
    ----------
    coeffs : np.ndarray
        Transform coefficients (any shape, will be flattened).
        Supports real and complex-valued coefficients.
    target_energy : float
        Target fraction of total energy to preserve (default 0.99).

    Returns
    -------
    float
        Fraction of largest coefficients needed to reach target_energy.
    """
    coeffs = np.asarray(coeffs)
    n = coeffs.size
    if n < 2:
        return 1.0
    power = np.abs(coeffs.ravel()) ** 2
    total = float(np.sum(power))
    if total < 1e-30:
        return 1.0
    sorted_power = np.sort(power)[::-1]
    cumulative = np.cumsum(sorted_power)
    n_keep = int(np.searchsorted(cumulative, target_energy * total) + 1)
    n_keep = min(n_keep, n)
    return max(n_keep / n, 1.0 / n)
