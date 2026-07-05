"""DCT/IDCT (1D, 2D), zigzag, FWHT/IFWHT transforms.

DCT via FFT (O(N log N)) for forward; IDCT via IFFT (O(N log N)) for inverse.
All operations are vectorized NumPy. No N×N matrix allocation.
"""

import gc
import math
from typing import Dict, Optional

import numpy as np

from spectralstream.core.math_primitives.prng import next_power_of_two


# ---------------------------------------------------------------------------
# DCT-II via FFT (O(N log N))
# ---------------------------------------------------------------------------


def _dct_matrix(n: int) -> np.ndarray:
    """DCT-II matrix of size n×n."""
    x = np.arange(n, dtype=np.float64)[:, None]
    y = np.arange(n, dtype=np.float64)[None, :]
    return np.cos(np.pi * (2 * y + 1) * x / (2 * n)) * np.sqrt(2.0 / n)


def _dct_via_fft_1d(x: np.ndarray) -> np.ndarray:
    """DCT-II via 2N-point FFT. ND array: operates along LAST axis."""
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[-1]
    shape = list(x.shape)
    z = np.zeros(shape[:-1] + [2 * n], dtype=np.float64)
    z[..., :n] = x
    z[..., n:] = x[..., ::-1]
    Z = np.fft.fft(z)[..., :n]
    k = np.arange(n, dtype=np.float64)
    for _ in range(x.ndim - 1):
        k = np.expand_dims(k, axis=0)
    result = np.real(Z * np.exp(-1j * np.pi * k / (2.0 * n)))
    result *= np.sqrt(0.5 / n)
    result[..., 0] /= np.sqrt(2.0)
    return result


def _idct_via_fft_1d(x: np.ndarray) -> np.ndarray:
    """DCT-III (IDCT) via 2N-point IFFT. O(N log N), no N×N matrix allocation.

    DCT-III: x[n] = Σₖ αₖ·C[k]·cos(π·k·(n+0.5)/N), where
    α₀ = 1/√N, αₖ = √(2/N) for k>0.

    Constructs A[k] = αₖ·C[k]·exp(j·π·k/(2N)) for k=0..N-1, then
    IDCT[n] = Re(2N·IFFT(A)[n]).
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[-1]
    shape = list(x.shape)

    # 2N complex array — only 4× input size in bytes, vs N² for DCT matrix
    z = np.zeros(shape[:-1] + [2 * n], dtype=np.complex128)
    k = np.arange(n, dtype=np.float64)
    for _ in range(x.ndim - 1):
        k = np.expand_dims(k, axis=0)

    z[..., 0] = x[..., 0] / np.sqrt(n)
    if n > 1:
        z[..., 1:n] = (
            x[..., 1:] * np.sqrt(2.0 / n) * np.exp(1j * np.pi * k[..., 1:] / (2.0 * n))
        )

    Z = np.fft.ifft(z)[..., :n]
    result = np.real(Z) * (2.0 * n)

    return result


# ---------------------------------------------------------------------------
# DCT matrix (cached, DEPRECATED — kept for DCTRotator compatibility)
# Raises OOM protection error for n > 4096; use FFT-based IDCT instead.
# ---------------------------------------------------------------------------

_DCT_MATRIX_CACHE: Dict[int, np.ndarray] = {}
_DCT_MATRIX_MAX_N = 4096


def _dct_matrix(n: int) -> np.ndarray:
    if n > _DCT_MATRIX_MAX_N:
        raise MemoryError(
            f"_dct_matrix({n}) would allocate {n * n * 8 / 1024**3:.1f} GiB — "
            f"use FFT-based IDCT instead (size limit: {_DCT_MATRIX_MAX_N})"
        )
    if n in _DCT_MATRIX_CACHE:
        return _DCT_MATRIX_CACHE[n]
    k = np.arange(n, dtype=np.float64)
    v = k + 0.5
    M = np.cos(np.outer(k, v * np.pi / n))
    M[0] *= 1.0 / np.sqrt(n)
    M[1:] *= np.sqrt(2.0 / n)
    _DCT_MATRIX_CACHE[n] = M
    return M


# ---------------------------------------------------------------------------
# Public 1D DCT/IDCT
# ---------------------------------------------------------------------------


def dct(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        raise ValueError("Empty input to dct")
    if x.ndim == 1:
        return _dct_via_fft_1d(x)
    return np.apply_along_axis(_dct_via_fft_1d, axis, x)


def idct(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        raise ValueError("Empty input to idct")
    if x.ndim == 1:
        return _idct_via_fft_1d(x)
    return np.apply_along_axis(_idct_via_fft_1d, axis, x)


# ---------------------------------------------------------------------------
# 2D DCT/IDCT
# ---------------------------------------------------------------------------


def dct_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"dct_2d requires 2D input, got shape {x.shape}")
    return _dct_via_fft_1d(_dct_via_fft_1d(x).T).T


def idct_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"idct_2d requires 2D input, got shape {x.shape}")
    return _idct_via_fft_1d(_idct_via_fft_1d(x).T).T


# ---------------------------------------------------------------------------
# Zigzag indices (precomputed per size)
# ---------------------------------------------------------------------------

_ZIGZAG_CACHE: Dict[int, np.ndarray] = {}


def zigzag_indices(n: int) -> np.ndarray:
    if n in _ZIGZAG_CACHE:
        return _ZIGZAG_CACHE[n].copy()
    zz = np.zeros((n, n), dtype=np.int32)
    idx = 0
    for s in range(2 * n - 1):
        if s % 2 == 0:
            i = min(s, n - 1)
            j = s - i
            while i >= 0 and j < n:
                zz[i, j] = idx
                idx += 1
                i -= 1
                j += 1
        else:
            j = min(s, n - 1)
            i = s - j
            while j >= 0 and i < n:
                zz[i, j] = idx
                idx += 1
                i += 1
                j -= 1
    _ZIGZAG_CACHE[n] = zz.copy()
    return zz


# ---------------------------------------------------------------------------
# FWHT (vectorized, in-place)
# ---------------------------------------------------------------------------


def _fwht_vectorized(x: np.ndarray, normalize: bool = False) -> np.ndarray:
    n = x.shape[-1]
    h = 1
    while h < n:
        shape = x.shape[:-1] + (-1, 2 * h)
        x_2d = x.reshape(shape)
        even = x_2d[..., :h]
        odd = x_2d[..., h:]
        even_copy = even.copy()
        even += odd
        np.subtract(even_copy, odd, out=odd)
        h <<= 1
    if normalize:
        x /= np.sqrt(n)
    return x


def fwht(x: np.ndarray, normalize: bool = False) -> np.ndarray:
    x = np.asarray(x).copy()
    n = x.shape[-1]
    n2 = next_power_of_two(n)
    if n2 != n:
        pad_width = n2 - n
        x = np.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, pad_width)], mode="constant")
    if x.ndim == 1:
        result = _fwht_vectorized(x.reshape(1, -1), normalize).reshape(-1)
    else:
        result = _fwht_vectorized(x, normalize)
    if n2 != n:
        result = result[..., :n]
        if normalize:
            result *= np.sqrt(n2 / n)
    return result


def ifwht(x: np.ndarray, normalize: bool = False) -> np.ndarray:
    x = np.asarray(x).copy()
    if normalize:
        return fwht(x, normalize=True)
    n = x.shape[-1]
    n2 = next_power_of_two(n)
    if n2 != n:
        pad_width = n2 - n
        x = np.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, pad_width)], mode="constant")
    if x.ndim == 1:
        result = _fwht_vectorized(x.reshape(1, -1)).reshape(-1)
    else:
        result = _fwht_vectorized(x)
    result = result / n2
    if n2 != n:
        result = result[..., :n] * n2 / n
    return result


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def effective_rank(matrix: np.ndarray, max_samples: Optional[int] = None) -> float:
    matrix = np.asarray(matrix, dtype=np.float64)
    m, n = matrix.shape
    k = min(m, n, max_samples or 999999)
    if k < 2:
        return float(k)
    sv = np.linalg.svd(matrix[: min(m, k), : min(n, k)], compute_uv=False)
    sv = sv / (sv[0] + 1e-30)
    entropy = -np.sum(sv * np.log(sv + 1e-30))
    return float(np.exp(entropy / math.log(k)))


# ── Backward-compat alias ────────────────────────────────────────────────────
_idct_via_matrix_1d = _idct_via_fft_1d
