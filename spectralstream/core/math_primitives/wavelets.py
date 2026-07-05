"""Wavelet transforms (Haar, Daubechies-4)."""

import math
from typing import Tuple

import numpy as np


class WaveletTransform:
    """Haar and Daubechies-4 wavelet via lifting scheme. O(N)."""

    @staticmethod
    def haar_forward_1d(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n = len(x)
        x = np.asarray(x, dtype=np.float64)
        if n % 2 == 1:
            x = np.append(x, 0.0)
        even = x[0::2]
        odd = x[1::2]
        approx = (even + odd) * 0.5
        detail = (even - odd) * 0.5
        return approx, detail

    @staticmethod
    def haar_inverse_1d(approx: np.ndarray, detail: np.ndarray) -> np.ndarray:
        approx = np.asarray(approx, dtype=np.float64)
        detail = np.asarray(detail, dtype=np.float64)
        n = len(approx)
        out = np.empty(2 * n, dtype=np.float64)
        out[0::2] = approx + detail
        out[1::2] = approx - detail
        return out

    @staticmethod
    def _db4_constants() -> Tuple[float, float, float, float]:
        s = np.sqrt(3.0)
        return (
            s / 4.0,
            (np.sqrt(3.0) - 2.0) / 4.0,
            -np.sqrt(3.0) / 4.0,
            (2.0 * np.sqrt(3.0) - 3.0) / 12.0,
        )

    @staticmethod
    def daubechies4_forward_1d(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x = np.asarray(x, dtype=np.float64)
        n = len(x)
        if n % 2 == 1:
            x = np.append(x, 0.0)
        alpha, beta, gamma, delta = WaveletTransform._db4_constants()
        even = x[0::2].copy()
        odd = x[1::2].copy()
        odd -= alpha * (np.roll(even, -1) + even)
        even -= beta * (np.roll(odd, -1) + odd)
        odd -= gamma * (np.roll(even, -1) + even)
        even -= delta * (np.roll(odd, -1) + odd)
        even *= math.sqrt(2.0)
        odd *= math.sqrt(2.0)
        return even, odd

    @staticmethod
    def daubechies4_inverse_1d(approx: np.ndarray, detail: np.ndarray) -> np.ndarray:
        approx = np.asarray(approx, dtype=np.float64)
        detail = np.asarray(detail, dtype=np.float64)
        even = approx / math.sqrt(2.0)
        odd = detail / math.sqrt(2.0)
        alpha, beta, gamma, delta = WaveletTransform._db4_constants()
        even += delta * (np.roll(odd, -1) + odd)
        odd += gamma * (np.roll(even, -1) + even)
        even += beta * (np.roll(odd, -1) + odd)
        odd += alpha * (np.roll(even, -1) + even)
        out = np.empty(len(even) * 2, dtype=np.float64)
        out[0::2] = even
        out[1::2] = odd
        return out

    @staticmethod
    def multi_level_decompose(
        x: np.ndarray, wavelet: str = "haar", max_level: int = 6
    ) -> list:
        x = np.asarray(x, dtype=np.float64)
        forward_fn = (
            WaveletTransform.haar_forward_1d
            if wavelet == "haar"
            else WaveletTransform.daubechies4_forward_1d
        )
        levels = []
        current = x
        level = 0
        while level < max_level and len(current) > 2:
            approx, detail = forward_fn(current)
            levels.append((level, approx, detail))
            current = approx
            level += 1
        levels.append((level, current, np.array([], dtype=np.float64)))
        return levels

    @staticmethod
    def multi_level_reconstruct(levels: list, wavelet: str = "haar") -> np.ndarray:
        inverse_fn = (
            WaveletTransform.haar_inverse_1d
            if wavelet == "haar"
            else WaveletTransform.daubechies4_inverse_1d
        )
        current = levels[-1][1]
        for _, _, detail in reversed(levels[:-1]):
            if len(detail) == 0:
                current = inverse_fn(current, np.zeros_like(current))
            else:
                current = inverse_fn(current, detail)
        return current
