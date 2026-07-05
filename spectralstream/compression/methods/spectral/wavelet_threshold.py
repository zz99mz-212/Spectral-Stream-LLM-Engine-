from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

METHOD_NAME = "wavelet_threshold"

__all__ = ["WaveletConfig", "WaveletThresholdCompression", "METHOD_NAME"]


@dataclass
class WaveletConfig:
    wavelet: str = "haar"
    n_levels: int = 3
    threshold_mode: str = "soft"
    threshold_sigma: float = 1.0


class WaveletThresholdCompression:
    METHOD_NAME = METHOD_NAME

    def __init__(self, config: Optional[WaveletConfig] = None):
        self.config = config or WaveletConfig()

    def _haar_forward(
        self, data: np.ndarray, axis: int = -1
    ) -> Tuple[np.ndarray, np.ndarray]:
        n = data.shape[axis]
        if n < 2:
            return data, np.zeros_like(data)

        even = data.take(np.arange(0, n, 2), axis=axis)
        odd = data.take(np.arange(1, n, 2), axis=axis)
        approx = (even + odd) / 2.0
        detail = (even - odd) / 2.0
        return approx, detail

    def _haar_inverse(
        self, approx: np.ndarray, detail: np.ndarray, axis: int = -1
    ) -> np.ndarray:
        n = 2 * approx.shape[axis]
        result = (
            np.empty(list(approx.shape[:-1]) + [n], dtype=np.float64)
            if axis == -1
            else None
        )
        if result is None:
            result = np.empty_like(np.concatenate([approx, detail], axis=axis))
        even = approx + detail
        odd = approx - detail
        result = np.empty(
            list(even.shape[:-1]) + [even.shape[-1] + odd.shape[-1]], dtype=np.float64
        )
        result[..., 0::2] = even
        result[..., 1::2] = odd
        return result

    def _dwt2d(self, matrix: np.ndarray, n_levels: int) -> list:
        coeffs = []
        current = matrix.astype(np.float64)
        for _ in range(n_levels):
            if current.shape[0] < 2 or current.shape[1] < 2:
                break
            approx_h, detail_h = self._haar_forward(current, axis=1)
            approx_v, detail_v = self._haar_forward(approx_h, axis=0)
            _, detail_d = self._haar_forward(detail_h, axis=0)
            coeffs.append((approx_v, detail_v, detail_d, detail_h))
            current = approx_v
        coeffs.append((current,))
        return coeffs

    def _idwt2d(self, coeffs: list) -> np.ndarray:
        current = coeffs[-1][0]
        for layer in reversed(coeffs[:-1]):
            approx_v, detail_v, detail_d, detail_h = layer
            current = self._haar_inverse(current, detail_v, axis=0)
            full_h = self._haar_inverse(current, detail_d, axis=0)
            current = self._haar_inverse(approx_v, full_h, axis=1)
        return current

    def _threshold(self, data: np.ndarray, sigma: float, mode: str) -> np.ndarray:
        threshold_val = sigma * np.std(data) * np.sqrt(2 * np.log(max(len(data), 2)))
        if mode == "soft":
            return np.sign(data) * np.maximum(np.abs(data) - threshold_val, 0)
        elif mode == "hard":
            return data * (np.abs(data) >= threshold_val)
        return data

    def compress(
        self, tensor: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        n_levels = kwargs.get("n_levels", self.config.n_levels)
        sigma = kwargs.get("threshold_sigma", self.config.threshold_sigma)
        mode = kwargs.get("threshold_mode", self.config.threshold_mode)
        orig_shape = tensor.shape

        mat = tensor.reshape(-1, tensor.shape[-1]).astype(np.float64)
        coeffs = self._dwt2d(mat, n_levels)

        for i in range(len(coeffs) - 1):
            approx_v, detail_v, detail_d, detail_h = coeffs[i]
            coeffs[i] = (
                approx_v,
                self._threshold(detail_v, sigma, mode),
                self._threshold(detail_d, sigma, mode),
                self._threshold(detail_h, sigma, mode),
            )

        data_out = {"coeffs": coeffs, "n_levels": n_levels}
        meta = {"orig_shape": orig_shape, "method": METHOD_NAME}
        return data_out, meta

    def decompress(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> np.ndarray:
        coeffs = data["coeffs"]
        recon = self._idwt2d(coeffs)
        return recon.reshape(metadata["orig_shape"]).astype(np.float32)

    def estimate_ratio(self, tensor: np.ndarray, **kwargs) -> float:
        sigma = kwargs.get("threshold_sigma", self.config.threshold_sigma)
        orig = tensor.nbytes
        nnz_fraction = min(1.0, sigma * 0.3)
        comp = orig * nnz_fraction
        return comp / max(orig, 1)
