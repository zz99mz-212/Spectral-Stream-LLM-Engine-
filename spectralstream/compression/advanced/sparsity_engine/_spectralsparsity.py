
import math
import time
import threading
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Any, Callable, Optional, Union

import numpy as np

from spectralstream.core.math_primitives import (

    next_power_of_two as _next_power_of_two,
    softmax as _softmax,
    dct as _dct,
    idct as _idct,
    spectral_entropy as _spectral_entropy,
    fwht,
    ifwht,
)

def _csr_from_dense(dense: np.ndarray, threshold: float = 1e-10) -> tuple:
    mask = np.abs(dense) > threshold
    indices = np.where(mask)
    values = dense[indices]
    return values, indices, dense.shape

def _dense_from_csr(values: np.ndarray, indices: tuple, shape: tuple) -> np.ndarray:
    out = np.zeros(shape, dtype=np.float64)
    out[indices] = values
    return out

def _nm_mask(shape: tuple, n: int, m: int, rng_seed: Optional[int] = None) -> np.ndarray:
    rows, cols = shape
    mask = np.zeros(shape, dtype=bool)
    rng = np.random.RandomState(rng_seed)
    for i in range(rows):
        for j_block in range(0, cols, m):
            block_end = min(j_block + m, cols)
            block_size = block_end - j_block
            chosen = rng.choice(block_size, min(n, block_size), replace=False)
            for c in chosen:
                mask[i, j_block + c] = True
    return mask

def _apply_nm_pattern(weights: np.ndarray, n: int, m: int) -> np.ndarray:
    rows, cols = weights.shape
    out = weights.copy()
    for i in range(rows):
        for j_block in range(0, cols, m):
            block_end = min(j_block + m, cols)
            block = out[i, j_block:block_end]
            if len(block) > n:
                abs_vals = np.abs(block)
                threshold = np.sort(abs_vals)[-n] if n > 0 else 0
                block[abs_vals < threshold] = 0.0
    return out

def _block_mask(shape: tuple, block_h: int, block_w: int, sparsity: float,
                rng_seed: Optional[int] = None) -> np.ndarray:
    rows, cols = shape
    mask = np.ones(shape, dtype=bool)
    rng = np.random.RandomState(rng_seed)
    for i in range(0, rows, block_h):
        for j in range(0, cols, block_w):
            if rng.random() < sparsity:
                ih = min(i + block_h, rows)
                jw = min(j + block_w, cols)
                mask[i:ih, j:jw] = False
    return mask

def _energy_ratio(x: np.ndarray) -> np.ndarray:
    x_spec = _dct(x)
    power = x_spec ** 2
    total = np.sum(power)
    cum = np.cumsum(power) / (total + 1e-30)
    return cum

def _sparsity_ratio(w: np.ndarray) -> float:
    return float(np.mean(np.abs(w) < 1e-10))

def _circular_conv(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = len(a)
    A_fft = np.fft.fft(a.astype(np.complex128))
    B_fft = np.fft.fft(b.astype(np.complex128))
    return np.fft.ifft(A_fft * B_fft).real.astype(np.float64)

def _circular_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = len(a)
    A_fft = np.fft.fft(a.astype(np.complex128))
    B_fft = np.fft.fft(b.astype(np.complex128))
    return np.fft.ifft(np.conj(A_fft) * B_fft).real.astype(np.float64)

class SpectralSparsity:
    def __init__(self, config: Optional[SparsityConfig] = None):
        self.config = config or SparsityConfig()
        self.band_config = SpectralBandConfig()
        if self.band_config.keep_ratios is None:
            self.band_config.keep_ratios = [
                1.0, 0.8, 0.6, 0.4, 0.3, 0.2, 0.1, 0.05
            ]
        self._coeff_cache: dict[str, np.ndarray] = {}

    def _get_band_boundaries(self, n_coeffs: int) -> list[tuple[int, int]]:
        n_bands = self.band_config.n_bands
        bands = []
        for b in range(n_bands):
            start = int(b * n_coeffs / n_bands)
            end = int((b + 1) * n_coeffs / n_bands)
            if b == n_bands - 1:
                end = n_coeffs
            bands.append((start, end))
        return bands

    def dct_sparsify(self, x: np.ndarray, keep_fraction: Optional[float] = None,
                     return_coeffs: bool = False) -> Union[np.ndarray, tuple]:
        if keep_fraction is None:
            keep_fraction = self.config.spectral_keep_fraction
        x_spec = _dct(x)
        n = x_spec.shape[-1]
        n_keep = max(1, int(n * keep_fraction))
        x_spec[..., n_keep:] = 0.0
        x_recon = _idct(x_spec)
        if return_coeffs:
            return x_recon, x_spec
        return x_recon

    def compress_by_energy(self, x: np.ndarray, energy_fraction: float = 0.99) -> np.ndarray:
        x_spec = _dct(x)
        power = x_spec ** 2
        total = np.sum(power, axis=-1, keepdims=True)
        cum = np.cumsum(power, axis=-1) / (total + 1e-30)
        n_keep = int(np.argmax(cum >= energy_fraction, axis=-1).max()) + 1
        x_spec[..., n_keep:] = 0.0
        return _idct(x_spec)

    def adaptive_band_sparsify(self, x: np.ndarray,
                                keep_ratios: Optional[list[float]] = None) -> np.ndarray:
        if keep_ratios is None:
            keep_ratios = self.band_config.keep_ratios
        x_spec = _dct(x)
        bands = self._get_band_boundaries(x_spec.shape[-1])
        for b_idx, (start, end) in enumerate(bands):
            ratio = keep_ratios[min(b_idx, len(keep_ratios) - 1)]
            if ratio >= 1.0:
                continue
            band_len = end - start
            n_keep = max(1, int(band_len * ratio))
            band = x_spec[..., start:end]
            power = band ** 2
            idx = np.argsort(-power, axis=-1)
            keep_mask = np.zeros_like(band, dtype=bool)
            batch_idx = np.arange(band.shape[0])[:, None] if band.ndim > 1 else np.arange(1)[:, None]
            if band.ndim > 1:
                keep_mask[batch_idx, idx[..., :n_keep]] = True
            else:
                keep_mask[idx[:n_keep]] = True
            x_spec[..., start:end][~keep_mask] = 0.0
        return _idct(x_spec)

    def progressive_sparsify(self, x: np.ndarray, n_stages: int = 4) -> list[np.ndarray]:
        x_spec = _dct(x)
        n = x_spec.shape[-1]
        stages = []
        for stage in range(1, n_stages + 1):
            n_keep = int(n * stage / n_stages)
            spec_copy = x_spec.copy()
            spec_copy[..., n_keep:] = 0.0
            stages.append(_idct(spec_copy))
        return stages

    def spectral_matmul(self, a_spec: np.ndarray, b_spec: np.ndarray) -> np.ndarray:
        nonzero_a = np.abs(a_spec) > 1e-10
        nonzero_b = np.abs(b_spec) > 1e-10
        joint = nonzero_a[..., :, None] & nonzero_b[..., None, :]
        if float(np.mean(joint)) < 0.5:
            result = np.zeros((a_spec.shape[0], b_spec.shape[-1]), dtype=np.float64)
            rows_nz, cols_a_nz = np.where(nonzero_a)
            cols_b_nz, out_nz = np.where(nonzero_b.T)
            for idx in range(len(rows_nz)):
                r = rows_nz[idx]
                c_a = cols_a_nz[idx]
                result[r] += a_spec[r, c_a] * b_spec[c_a]
            return result.astype(a_spec.dtype)
        return a_spec @ b_spec

    def spectral_activation_sparsify(self, activations: np.ndarray,
                                      keep_fraction: Optional[float] = None) -> np.ndarray:
        if keep_fraction is None:
            keep_fraction = self.config.spectral_keep_fraction
        act_spec = _dct(activations)
        power = act_spec ** 2
        total = np.sum(power, axis=-1, keepdims=True)
        cum = np.cumsum(power / (total + 1e-30), axis=-1)
        n_keep = max(1, int(act_spec.shape[-1] * keep_fraction))
        act_spec[..., n_keep:] = 0.0
        return _idct(act_spec)

    def get_coeffs(self, x: np.ndarray, key: str = "") -> np.ndarray:
        coeffs = _dct(x)
        if key:
            self._coeff_cache[key] = coeffs
        return coeffs

    def from_coeffs(self, coeffs: np.ndarray, key: str = "") -> np.ndarray:
        result = _idct(coeffs)
        return result
