"""HPC wavelet methods — batched vectorized lifting scheme, no row-wise Python loops, np.sqrt everywhere."""

from __future__ import annotations

import gc
import struct
from typing import Any, Tuple

import numpy as np

from spectralstream.core.math_primitives import (
    auto_keep_fraction,
)
from spectralstream.compression._dtype_utils import (
    detect_storage_dtype,
    convert_to_storage,
    convert_from_storage,
    encode_dtype_code,
    decode_dtype_code,
)


def _batch_haar_forward_2d(x: np.ndarray, max_level: int) -> list:
    """Haar forward transform on all rows of a 2D array simultaneously."""
    levels = []
    current = x.copy()
    for level in range(max_level):
        if current.shape[1] <= 2:
            break
        even = current[:, 0::2]
        odd = current[:, 1::2]
        approx = (even + odd) * 0.5
        detail = (even - odd) * 0.5
        levels.append((level, approx, detail))
        current = approx
    levels.append((level, current, np.array([], dtype=np.float64)))
    return levels


def _batch_haar_inverse_2d(levels: list, n_out: int) -> np.ndarray:
    """Haar inverse transform on all rows simultaneously."""
    current = levels[-1][1]
    for level_idx, _, detail in reversed(levels[:-1]):
        n = current.shape[1]
        out = np.empty((current.shape[0], n * 2), dtype=np.float64)
        if detail.size > 0:
            out[:, 0::2] = current + detail
            out[:, 1::2] = current - detail
        else:
            out[:, 0::2] = current
            out[:, 1::2] = current
        current = out
    return current[:, :n_out]


def _batch_db4_forward_2d(x: np.ndarray, max_level: int) -> list:
    """Daubechies-4 forward transform on all rows simultaneously."""
    sqrt3 = np.sqrt(3.0)
    alpha = sqrt3 / 4.0
    beta = (sqrt3 - 2.0) / 4.0
    gamma = -sqrt3 / 4.0
    delta = (2.0 * sqrt3 - 3.0) / 12.0
    sqrt2 = np.sqrt(2.0)

    levels = []
    current = x.copy()
    for level in range(max_level):
        if current.shape[1] <= 2:
            break
        even = current[:, 0::2].copy()
        odd = current[:, 1::2].copy()
        odd -= alpha * (np.roll(even, -1, axis=1) + even)
        even -= beta * (np.roll(odd, -1, axis=1) + odd)
        odd -= gamma * (np.roll(even, -1, axis=1) + even)
        even -= delta * (np.roll(odd, -1, axis=1) + odd)
        even *= sqrt2
        odd *= sqrt2
        levels.append((level, even, odd))
        current = even
    levels.append((level, current, np.array([], dtype=np.float64)))
    return levels


def _batch_db4_inverse_2d(levels: list, n_out: int) -> np.ndarray:
    """Daubechies-4 inverse transform on all rows simultaneously."""
    sqrt3 = np.sqrt(3.0)
    alpha = sqrt3 / 4.0
    beta = (sqrt3 - 2.0) / 4.0
    gamma = -sqrt3 / 4.0
    delta = (2.0 * sqrt3 - 3.0) / 12.0
    sqrt2 = np.sqrt(2.0)

    current = levels[-1][1]
    for level_idx, _, detail in reversed(levels[:-1]):
        n = current.shape[1]
        even = current / sqrt2
        odd = detail / sqrt2 if detail.size > 0 else current / sqrt2
        even += delta * (np.roll(odd, -1, axis=1) + odd)
        odd += gamma * (np.roll(even, -1, axis=1) + even)
        even += beta * (np.roll(odd, -1, axis=1) + odd)
        odd += alpha * (np.roll(even, -1, axis=1) + even)
        out = np.empty((current.shape[0], n * 2), dtype=np.float64)
        out[:, 0::2] = even
        out[:, 1::2] = odd
        current = out
    return current[:, :n_out]


class WaveletHaar:
    """Haar wavelet thresholding — batched across rows."""

    name = "wavelet_haar"
    category = "spectral"

    def compress(
        self,
        tensor: np.ndarray,
        level: int = 3,
        keep_fraction: float | None = None,
        target_energy: float = 0.99,
    ) -> Tuple[bytes, dict]:
        storage_dtype = detect_storage_dtype(tensor)
        sd_code = int(encode_dtype_code(storage_dtype))
        orig = tensor.astype(np.float64)
        if orig.ndim == 1:
            orig = orig.reshape(1, -1)
        m, n = orig.shape

        levels = _batch_haar_forward_2d(orig, max_level=level)

        coeffs = np.zeros((m, n), dtype=np.float64)
        pos = 0
        for lvl, approx, detail in levels[:-1]:
            coeffs[:, pos : pos + detail.shape[1]] = detail
            pos += detail.shape[1]
        coeffs[:, pos : pos + levels[-1][1].shape[1]] = levels[-1][1]

        flat = coeffs.ravel()
        if keep_fraction is None:
            kf = auto_keep_fraction(flat, target_energy)
        else:
            kf = keep_fraction
        k = max(1, int(kf * flat.size))
        idx = np.argpartition(np.abs(flat), -k)[-k:]
        meta = dict(
            shape=orig.shape,
            level=level,
            keep_fraction=kf,
            target_energy=target_energy,
            n_kept=k,
            _storage_dtype=sd_code,
        )
        data = (
            struct.pack("<ii", m, n)
            + idx.astype(np.int32).tobytes()
            + convert_to_storage(flat[idx], storage_dtype).tobytes()
        )
        del coeffs, flat, levels
        gc.collect()
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        level = metadata.get("level", 3)
        ndim = metadata.get("ndim", 2)
        sd = decode_dtype_code(metadata.get("_storage_dtype", 0))
        es = int(sd.itemsize)
        m, n = shape
        k = metadata["n_kept"]
        pos = struct.calcsize("<ii")
        idx = np.frombuffer(data[pos : pos + k * 4], dtype=np.int32).copy().astype(int)
        pos += k * 4
        vals = convert_from_storage(
            np.frombuffer(data[pos : pos + k * es], dtype=sd), sd
        ).astype(np.float64)
        thresh = np.zeros(m * n, dtype=np.float64)
        thresh[idx] = vals
        thresh = thresh.reshape(m, n)

        detail_sizes = [n // (2 ** (l + 1)) for l in range(level)]
        approx_size = n // (2**level)
        recon_levels = []
        offset = 0
        for l in range(level):
            d = thresh[:, offset : offset + detail_sizes[l]].copy()
            recon_levels.append((l, None, d))
            offset += detail_sizes[l]
        approx = thresh[:, offset : offset + approx_size].copy()
        recon_levels.append((level, approx, np.array([], dtype=np.float64)))

        result = _batch_haar_inverse_2d(recon_levels, n)
        out = result.astype(np.float32)
        if ndim == 1:
            out = out.ravel()
        return out


class _WaveletGeneric:
    """Shared batched Daubechies/Symlet logic."""

    @staticmethod
    def compress_generic(
        tensor: np.ndarray,
        wavelet: str,
        level: int,
        keep_fraction: float | None,
        target_energy: float,
    ) -> Tuple[bytes, dict]:
        storage_dtype = detect_storage_dtype(tensor)
        sd_code = int(encode_dtype_code(storage_dtype))
        orig = tensor.astype(np.float64)
        ndim = orig.ndim
        if ndim == 1:
            orig = orig.reshape(1, -1)
        m, n = orig.shape

        levels = _batch_db4_forward_2d(orig, max_level=level)

        coeffs = np.zeros((m, n), dtype=np.float64)
        pos = 0
        for lvl, approx, detail in levels[:-1]:
            coeffs[:, pos : pos + detail.shape[1]] = detail
            pos += detail.shape[1]
        coeffs[:, pos : pos + levels[-1][1].shape[1]] = levels[-1][1]

        flat = coeffs.ravel()
        if keep_fraction is None:
            kf = auto_keep_fraction(flat, target_energy)
        else:
            kf = keep_fraction
        k = max(1, int(kf * flat.size))
        idx = np.argpartition(np.abs(flat), -k)[-k:]
        meta = dict(
            shape=orig.shape,
            level=level,
            keep_fraction=kf,
            target_energy=target_energy,
            n_kept=k,
            ndim=ndim,
            _storage_dtype=sd_code,
        )
        data = (
            struct.pack("<ii", m, n)
            + idx.astype(np.int32).tobytes()
            + convert_to_storage(flat[idx], storage_dtype).tobytes()
        )
        del coeffs, flat, levels
        gc.collect()
        return data, meta

    @staticmethod
    def decompress_generic(
        data: bytes,
        metadata: dict,
        wavelet: str,
    ) -> np.ndarray:
        shape = metadata["shape"]
        level = metadata.get("level", 3)
        ndim = metadata.get("ndim", 2)
        sd = decode_dtype_code(metadata.get("_storage_dtype", 0))
        es = int(sd.itemsize)
        m, n = shape
        k = metadata["n_kept"]
        pos = struct.calcsize("<ii")
        idx = np.frombuffer(data[pos : pos + k * 4], dtype=np.int32).copy().astype(int)
        pos += k * 4
        vals = convert_from_storage(
            np.frombuffer(data[pos : pos + k * es], dtype=sd), sd
        ).astype(np.float64)
        thresh = np.zeros(m * n, dtype=np.float64)
        thresh[idx] = vals
        thresh = thresh.reshape(m, n)

        detail_sizes = [n // (2 ** (l + 1)) for l in range(level)]
        approx_size = n // (2**level)
        recon_levels = []
        offset = 0
        for l in range(level):
            d = thresh[:, offset : offset + detail_sizes[l]].copy()
            recon_levels.append((l, None, d))
            offset += detail_sizes[l]
        approx = thresh[:, offset : offset + approx_size].copy()
        recon_levels.append((level, approx, np.array([], dtype=np.float64)))

        result = _batch_db4_inverse_2d(recon_levels, n)
        out = result.astype(np.float32)
        if ndim == 1:
            out = out.ravel()
        return out


class WaveletDaubechies:
    """Daubechies wavelet — batched across rows."""

    name = "wavelet_daubechies"
    category = "spectral"

    def compress(
        self,
        tensor: np.ndarray,
        level: int = 3,
        keep_fraction: float | None = None,
        target_energy: float = 0.99,
    ) -> Tuple[bytes, dict]:
        return _WaveletGeneric.compress_generic(
            tensor, "db4", level, keep_fraction, target_energy
        )

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        return _WaveletGeneric.decompress_generic(data, metadata, "db4")


class WaveletSymlet:
    """Symlet wavelet compression — batched across rows."""

    name = "wavelet_symlet"
    category = "spectral"

    def compress(
        self,
        tensor: np.ndarray,
        level: int = 3,
        keep_fraction: float | None = None,
        target_energy: float = 0.99,
    ) -> Tuple[bytes, dict]:
        return _WaveletGeneric.compress_generic(
            tensor, "sym4", level, keep_fraction, target_energy
        )

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        return _WaveletGeneric.decompress_generic(data, metadata, "sym4")


class WaveletScattering:
    """Wavelet scattering transform — batched across rows."""

    name = "wavelet_scattering"
    category = "spectral"

    def compress(
        self,
        tensor: np.ndarray,
        level: int = 2,
        keep_fraction: float | None = None,
        target_energy: float = 0.99,
    ) -> Tuple[bytes, dict]:
        storage_dtype = detect_storage_dtype(tensor)
        sd_code = int(encode_dtype_code(storage_dtype))
        orig = tensor.astype(np.float64)
        m, n = orig.shape
        levels = _batch_haar_forward_2d(orig, max_level=level)
        s2 = np.zeros((m, n), dtype=np.float64)
        for lvl, approx, detail in levels[:-1]:
            ds = detail.shape[1]
            s2[:, :ds] = np.maximum(s2[:, :ds], np.abs(detail))
        flat = s2.ravel()
        if keep_fraction is None:
            kf = auto_keep_fraction(flat, target_energy)
        else:
            kf = keep_fraction
        k = max(1, int(kf * flat.size))
        idx = np.argpartition(np.abs(flat), -k)[-k:]
        meta = dict(
            shape=orig.shape,
            level=level,
            keep_fraction=kf,
            target_energy=target_energy,
            n_kept=k,
            _storage_dtype=sd_code,
        )
        data = (
            struct.pack("<ii", m, n)
            + idx.astype(np.int32).tobytes()
            + convert_to_storage(flat[idx], storage_dtype).tobytes()
        )
        del s2, levels
        gc.collect()
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        level = metadata.get("level", 2)
        sd = decode_dtype_code(metadata.get("_storage_dtype", 0))
        es = int(sd.itemsize)
        m, n = shape
        k = metadata["n_kept"]
        pos = struct.calcsize("<ii")
        idx = np.frombuffer(data[pos : pos + k * 4], dtype=np.int32).copy().astype(int)
        pos += k * 4
        vals = convert_from_storage(
            np.frombuffer(data[pos : pos + k * es], dtype=sd), sd
        ).astype(np.float64)
        coeffs = np.zeros(m * n, dtype=np.float64)
        coeffs[idx] = vals
        coeffs = coeffs.reshape(m, n)
        detail_sizes = [n // (2 ** (l + 1)) for l in range(level)]
        approx_size = n // (2**level)
        recon_levels = []
        offset = 0
        for l in range(level):
            d = coeffs[:, offset : offset + detail_sizes[l]].copy()
            recon_levels.append((l, None, d))
            offset += detail_sizes[l]
        approx = np.zeros((m, approx_size), dtype=np.float64)
        recon_levels.append((level, approx, np.array([], dtype=np.float64)))
        result = _batch_haar_inverse_2d(recon_levels, n)
        return result.astype(np.float32)
