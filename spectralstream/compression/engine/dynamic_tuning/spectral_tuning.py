"""Dynamic tuning for ALL spectral/transform compression methods.

Each tuner wraps a spectral transform (DCT, FFT, Hadamard, Wavelet, etc.)
and auto-tunes the coefficient threshold to hit ANY target compression ratio.

Key formulas (from research):
    DCT compression: keep top k coefficients → ratio = n / (k + overhead)
    Error = 1 - cumulative_energy(k) / total_energy
    Energy decay model: power law E(k) ∝ k^{-α}
"""

from __future__ import annotations


import math
import struct
from typing import Any, Dict, Optional, Tuple

import numpy as np

from spectralstream.core.math_primitives import (
    dct,
    dct_2d,
    fft,
    fwht,
    idct,
    idct_2d,
    ifft,
    ifwht,
    next_power_of_two,
)
from spectralstream.core.math_primitives.wavelets import WaveletTransform


# ── Core tuning primitives ──────────────────────────────────────────────


def estimate_energy_decay(
    coefficients: np.ndarray, min_coeffs: int = 4
) -> Tuple[float, float, np.ndarray]:
    """Fit a power-law decay model to sorted squared coefficients.

    Returns (alpha, total_energy, cumulative_energy) where:
        alpha         — exponent in E(k) ∝ k^{-α}
        total_energy  — sum of squared coefficients
        cumulative_energy — normalised cumulative sum (descending order)

    The power-law is fitted via log-log linear regression on the
    largest ``min_coeffs`` coefficients.
    """
    coeffs = np.asarray(coefficients, dtype=np.float64).ravel()
    energy = coeffs * coeffs
    total = float(np.sum(energy))
    if total < 1e-30 or len(energy) < min_coeffs:
        return 0.0, total, np.ones_like(energy) / max(len(energy), 1)

    sorted_idx = np.argsort(energy)[::-1]
    sorted_energy = energy[sorted_idx]
    cum = np.cumsum(sorted_energy) / total

    n_fit = min(min_coeffs, len(sorted_energy))
    xs = np.log(np.arange(1, n_fit + 1, dtype=np.float64))
    ys = np.log(sorted_energy[:n_fit] + 1e-30)
    A = np.vstack([xs, np.ones(n_fit)]).T
    try:
        coeffs_fit, *_ = np.linalg.lstsq(A, ys, rcond=None)
        alpha = float(max(0.0, -coeffs_fit[0]))
    except np.linalg.LinAlgError:
        alpha = 0.0

    return alpha, total, cum


def k_for_ratio(
    n_elements: int,
    target_ratio: float,
    overhead_per_coeff: float = 1.5,
    fixed_overhead_bytes: int = 16,
    element_bytes: int = 4,
    max_k: Optional[int] = None,
) -> int:
    """Compute the number of coefficients to keep for a target compression ratio.

    Solves:
        target_ratio = n * element_bytes / (fixed_overhead + k * coeff_cost)
    where coeff_cost = overhead_per_coeff * element_bytes.

    Returns k clipped to [1, n_elements].
    """
    if target_ratio <= 1.0:
        return n_elements
    coeff_cost = overhead_per_coeff * element_bytes
    # target_ratio = n * element_bytes / (fixed_overhead + k * coeff_cost)
    # => k = (n * element_bytes / target_ratio - fixed_overhead) / coeff_cost
    k = max(
        1,
        int(
            (n_elements * element_bytes / target_ratio - fixed_overhead_bytes)
            / coeff_cost
        ),
    )
    if max_k is not None:
        k = min(k, max_k)
    return min(k, n_elements)


def threshold_for_k(coefficients: np.ndarray, k: int) -> Tuple[float, np.ndarray]:
    """Find the magnitude threshold to keep the top-k coefficients.

    Returns (threshold, kept_mask) where kept_mask[i] is True for
    coefficients whose magnitude >= threshold (with tie-breaking via
    argpartition).
    """
    coeffs = np.asarray(coefficients, dtype=np.float64).ravel()
    n = len(coeffs)
    k_actual = min(max(k, 1), n)

    if k_actual >= n:
        return 0.0, np.ones(n, dtype=bool)

    abs_coeffs = np.abs(coeffs)
    threshold = float(np.partition(abs_coeffs, n - k_actual)[n - k_actual])
    mask = abs_coeffs >= threshold

    n_kept = int(np.sum(mask))
    if n_kept > k_actual:
        tie_idx = np.where(abs_coeffs == threshold)[0]
        n_remove = n_kept - k_actual
        remove = tie_idx[np.argsort(np.abs(coeffs[tie_idx]))[:n_remove]]
        mask[remove] = False
    elif n_kept < k_actual:
        missing = k_actual - n_kept
        unused = np.where(~mask)[0]
        add = unused[np.argsort(abs_coeffs[unused])[-missing:]]
        mask[add] = True

    return threshold, mask


def tune_coefficient_threshold(
    coefficients: np.ndarray,
    target_ratio: float,
    n_elements: int,
    overhead_per_coeff: float = 1.5,
    fixed_overhead_bytes: int = 16,
    element_bytes: int = 4,
) -> Tuple[int, float, np.ndarray, Dict[str, float]]:
    """Auto-tune coefficient threshold to hit a target compression ratio.

    Args:
        coefficients: Transformed coefficients (flat).
        target_ratio: Desired compression ratio.
        n_elements: Total number of original elements.
        overhead_per_coeff: Storage bytes per kept coefficient per input
            element byte.  Default 1.5 = 6 bytes (4 index + 2 value) / 4.
        fixed_overhead_bytes: Fixed header + trailer bytes.
        element_bytes: Bytes per input element.

    Returns:
        (k, threshold, mask, info) where:
            k         — number of coefficients kept
            threshold — magnitude threshold applied
            mask      — boolean mask of kept coefficients
            info      — dict with alpha, energy_retained, predicted_error,
                        effective_ratio
    """
    alpha, total_energy, cum_energy = estimate_energy_decay(coefficients)

    k = k_for_ratio(
        n_elements,
        target_ratio,
        overhead_per_coeff,
        fixed_overhead_bytes,
        element_bytes,
    )
    threshold, mask = threshold_for_k(coefficients, k)

    n_kept = int(np.sum(mask))
    coeff_cost = overhead_per_coeff * element_bytes
    compressed_bytes = fixed_overhead_bytes + n_kept * coeff_cost
    effective_ratio = (n_elements * element_bytes) / max(compressed_bytes, 1)

    energy_retained = float(np.sum(coefficients[mask] ** 2) / max(total_energy, 1e-30))
    predicted_error = 1.0 - energy_retained

    info = {
        "alpha": alpha,
        "total_energy": total_energy,
        "energy_retained": energy_retained,
        "predicted_error": predicted_error,
        "effective_ratio": effective_ratio,
        "target_ratio": target_ratio,
        "k": k,
        "n_kept": n_kept,
    }

    return n_kept, threshold, mask, info


# ── Tuner base ──────────────────────────────────────────────────────────


class SpectralTunerBase:
    """Base class for spectral method tuners.

    Subclass must provide:
        name                  — str identifier
        transform             — callable(flat) -> coefficients
        inverse               — callable(coefficients) -> reconstructed flat
        _overhead_per_coeff   — float (bytes per kept coeff / bytes per element)

    Default storage:  int32 index (4B) + float16 value (2B) per coefficient.
    For float32 input  (4B/element) → overhead = 6/4 = 1.5.
    For complex input  (8B/element) → overhead = 6/8 = 0.75.

    Override  ``_serialize_coeffs`` / ``_deserialize_coeffs`` for custom
    coefficient storage (e.g. complex → real+imag pairs).
    """

    name: str = "base"

    def transform(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def inverse(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _overhead_per_coeff(self) -> float:
        """Return per-coefficient byte overhead in units of input elements.

        Default: 6 bytes per kept coeff (4B index + 2B float16 value),
        divided by 4 bytes per float32 input element => 1.5.
        """
        return 1.5

    # ── Serialisation hooks (override for complex / custom types) ───

    def _serialize_coeffs(self, coeffs: np.ndarray, mask: np.ndarray) -> bytes:
        """Serialize kept coefficients to bytes (default: float16)."""
        return coeffs[mask].astype(np.float16).tobytes()

    def _deserialize_coeffs(self, data: bytes, n_kept: int) -> np.ndarray:
        """Deserialize kept coefficients (default: float16 → float64)."""
        return np.frombuffer(data, dtype=np.float16, count=n_kept).astype(np.float64)

    def _coeff_size_bytes(self) -> int:
        """Bytes per stored coefficient (default: 2 for float16)."""
        return 2

    # ── Public API ─────────────────────────────────────────────────

    def compress_at_ratio(
        self,
        tensor: np.ndarray,
        target_ratio: float,
        return_info: bool = False,
    ) -> Tuple[bytes, dict]:
        """Compress ``tensor`` targeting ``target_ratio``.

        Auto-tunes the coefficient threshold.  Returns (data, metadata)
        compatible with the engine's compress/decompress protocol.
        """
        flat = tensor.astype(np.float64).ravel()
        n = len(flat)

        coeffs = self.transform(flat.copy())
        k, threshold, mask, info = tune_coefficient_threshold(
            coeffs, target_ratio, n, self._overhead_per_coeff()
        )

        kept_idx = np.where(mask)[0].astype(np.int32)
        kept_bytes = self._serialize_coeffs(coeffs, mask)

        header = struct.pack("<II", n, len(kept_idx))
        data = header + kept_idx.tobytes() + kept_bytes + struct.pack("<d", threshold)

        meta: dict = {
            "shape": tensor.shape,
            "method": self.name,
            "n_kept": len(kept_idx),
            "n_elements": n,
            "target_ratio": target_ratio,
            "threshold": threshold,
        }
        if return_info:
            meta["tuning_info"] = info
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        """Decompress data produced by ``compress_at_ratio``."""
        n_elements = metadata["n_elements"]
        n_kept = metadata["n_kept"]
        pos = 8
        kept_idx = np.frombuffer(data[pos : pos + n_kept * 4], dtype=np.int32).copy()
        pos += n_kept * 4
        kept_vals = self._deserialize_coeffs(data[pos:], n_kept)
        pos += n_kept * self._coeff_size_bytes()

        coeffs = np.zeros(n_elements, dtype=np.float64)
        coeffs[kept_idx] = kept_vals

        flat = self.inverse(coeffs)
        return flat.reshape(metadata["shape"]).astype(np.float32)

    def tune(
        self,
        tensor: np.ndarray,
        target_ratio: float,
    ) -> Dict[str, Any]:
        """Return tuning diagnostics without compressing."""
        flat = tensor.astype(np.float64).ravel()
        coeffs = self.transform(flat.copy())
        _, _, _, info = tune_coefficient_threshold(
            coeffs, target_ratio, len(flat), self._overhead_per_coeff()
        )
        return info


# ── DCT Tuner ──────────────────────────────────────────────────────────


class DCTSpectralTuner(SpectralTunerBase):
    """Dynamic tuning for 1D DCT coefficient thresholding.

    ratio = n * 4 / (16 + k * 6) ≈ n / (k * 1.5)
    Energy decay: DCT coefficients of natural signals follow ~k^{-1} to ~k^{-2}.
    """

    name = "dct_tuned"

    def transform(self, x: np.ndarray) -> np.ndarray:
        return dct(x)

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return idct(x)


class DCT2DSpectralTuner(SpectralTunerBase):
    """Dynamic tuning for 2D DCT coefficient thresholding.

    Applies full-frame 2D DCT then thresholds globally.  Best for
    structured weight matrices with strong 2D frequency locality.
    """

    name = "dct_2d_tuned"

    def transform(self, x: np.ndarray) -> np.ndarray:
        n = len(x)
        side = int(math.isqrt(n))
        if side * side != n:
            return dct(x)
        mat = x.reshape(side, side)
        return dct_2d(mat).ravel()

    def inverse(self, x: np.ndarray) -> np.ndarray:
        n = len(x)
        side = int(math.isqrt(n))
        if side * side != n:
            return idct(x)
        mat = x.reshape(side, side)
        return idct_2d(mat).ravel()


# ── FFT Tuner ──────────────────────────────────────────────────────────


class _ComplexSpectralTunerMixin(SpectralTunerBase):
    """Mixin for tuners whose coefficients are complex.

    Stores each coefficient as 2 × float16 (real + imag).
    Thresholding uses magnitude = sqrt(real² + imag²).
    """

    def _overhead_per_coeff(self) -> float:
        index_bytes = 4  # int32
        value_bytes = 4  # 2 × float16
        element_bytes = 4  # float32 input
        return (index_bytes + value_bytes) / element_bytes

    def _serialize_coeffs(self, coeffs: np.ndarray, mask: np.ndarray) -> bytes:
        kept = coeffs[mask]
        interleaved = np.empty(len(kept) * 2, dtype=np.float16)
        interleaved[0::2] = np.real(kept).astype(np.float16)
        interleaved[1::2] = np.imag(kept).astype(np.float16)
        return interleaved.tobytes()

    def _deserialize_coeffs(self, data: bytes, n_kept: int) -> np.ndarray:
        raw = np.frombuffer(data, dtype=np.float16, count=n_kept * 2)
        real = raw[0::2].astype(np.float64)
        imag = raw[1::2].astype(np.float64)
        return real + 1j * imag

    def _coeff_size_bytes(self) -> int:
        return 4  # 2 × float16 per complex


class FFTSpectralTuner(_ComplexSpectralTunerMixin):
    """Dynamic tuning for FFT coefficient thresholding.

    Coefficients are complex → stored as 2 × float16 (real + imag) per kept bin.
    Threshold selection uses magnitude = sqrt(real² + imag²).
    """

    name = "fft_tuned"

    def transform(self, x: np.ndarray) -> np.ndarray:
        return fft(x)

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return np.real(ifft(x))


class RFFTSpectralTuner(_ComplexSpectralTunerMixin):
    """Dynamic tuning for real-input FFT (rfft).

    rfft returns only the non-redundant positive frequencies (n//2+1 complex).
    Coefficients are complex → stored as 2 × float16 per bin.
    """

    name = "rfft_tuned"

    def transform(self, x: np.ndarray) -> np.ndarray:
        from spectralstream.core.math_primitives import rfft

        return rfft(x)

    def inverse(self, x: np.ndarray) -> np.ndarray:
        from spectralstream.core.math_primitives import irfft

        n = 2 * (len(x) - 1)
        return irfft(x, n=n)

    def compress_at_ratio(
        self,
        tensor: np.ndarray,
        target_ratio: float,
        return_info: bool = False,
    ) -> Tuple[bytes, dict]:
        flat = tensor.astype(np.float64).ravel()
        n = len(flat)
        coeffs = self.transform(flat.copy())
        n_coeffs = len(coeffs)
        k, threshold, mask, info = tune_coefficient_threshold(
            coeffs, target_ratio, n, self._overhead_per_coeff()
        )
        kept_idx = np.where(mask)[0].astype(np.int32)
        kept_bytes = self._serialize_coeffs(coeffs, mask)
        header = struct.pack("<II", n, len(kept_idx))
        data = header + kept_idx.tobytes() + kept_bytes + struct.pack("<d", threshold)
        meta: dict = {
            "shape": tensor.shape,
            "method": self.name,
            "n_kept": len(kept_idx),
            "n_elements": n,
            "n_coeffs": n_coeffs,
            "target_ratio": target_ratio,
            "threshold": threshold,
        }
        if return_info:
            meta["tuning_info"] = info
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        n_elements = metadata["n_elements"]
        n_kept = metadata["n_kept"]
        n_coeffs = metadata.get("n_coeffs", n_elements // 2 + 1)
        pos = 8
        kept_idx = np.frombuffer(data[pos : pos + n_kept * 4], dtype=np.int32).copy()
        pos += n_kept * 4
        kept_vals = self._deserialize_coeffs(data[pos:], n_kept)
        coeffs = np.zeros(n_coeffs, dtype=np.complex128)
        coeffs[kept_idx] = kept_vals
        flat = self.inverse(coeffs)
        return flat.reshape(metadata["shape"]).astype(np.float32)


# ── Hadamard / FWHT Tuner ──────────────────────────────────────────────


class HadamardSpectralTuner(SpectralTunerBase):
    """Dynamic tuning for Fast Walsh-Hadamard Transform thresholding.

    FWHT coefficients of natural signals tend to be more uniform than DCT,
    but energy is still concentrated in a subset.  Same overhead as DCT (1.5).

    Note: FWHT requires input padded to power-of-two length.  The transform
    output length equals the padded length, not the original input length.
    """

    name = "hadamard_tuned"

    def __init__(self) -> None:
        super().__init__()
        self._padded_len: int = 0

    def transform(self, x: np.ndarray) -> np.ndarray:
        self._padded_len = next_power_of_two(len(x))
        buf = np.zeros(self._padded_len, dtype=np.float64)
        buf[: len(x)] = x
        return fwht(buf, normalize=True)

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return ifwht(x, normalize=True)[: len(x)]

    def compress_at_ratio(
        self,
        tensor: np.ndarray,
        target_ratio: float,
        return_info: bool = False,
    ) -> Tuple[bytes, dict]:
        flat = tensor.astype(np.float64).ravel()
        n = len(flat)
        coeffs = self.transform(flat.copy())
        n_coeffs = len(coeffs)
        k, threshold, mask, info = tune_coefficient_threshold(
            coeffs, target_ratio, n, self._overhead_per_coeff()
        )
        kept_idx = np.where(mask)[0].astype(np.int32)
        kept_bytes = self._serialize_coeffs(coeffs, mask)
        header = struct.pack("<II", n, len(kept_idx))
        data = header + kept_idx.tobytes() + kept_bytes + struct.pack("<d", threshold)
        meta: dict = {
            "shape": tensor.shape,
            "method": self.name,
            "n_kept": len(kept_idx),
            "n_elements": n,
            "n_coeffs": n_coeffs,
            "padded_len": self._padded_len,
            "target_ratio": target_ratio,
            "threshold": threshold,
        }
        if return_info:
            meta["tuning_info"] = info
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        n_elements = metadata["n_elements"]
        n_kept = metadata["n_kept"]
        n_coeffs = metadata.get("n_coeffs", n_elements)
        pos = 8
        kept_idx = np.frombuffer(data[pos : pos + n_kept * 4], dtype=np.int32).copy()
        pos += n_kept * 4
        kept_vals = self._deserialize_coeffs(data[pos:], n_kept)
        coeffs = np.zeros(n_coeffs, dtype=np.float64)
        coeffs[kept_idx] = kept_vals
        flat = self.inverse(coeffs)
        return flat.reshape(metadata["shape"]).astype(np.float32)


# ── Wavelet Tuner ──────────────────────────────────────────────────────


class WaveletSpectralTuner(SpectralTunerBase):
    """Dynamic tuning for multi-level wavelet coefficient thresholding.

    Uses Haar wavelet by default (db4 available via ``wavelet`` param).
    Multi-level decomposition up to ``max_level`` levels.
    """

    name = "wavelet_tuned"

    def __init__(
        self,
        wavelet: str = "haar",
        max_level: int = 4,
    ) -> None:
        self.wavelet = wavelet
        self.max_level = max_level
        self._wt = WaveletTransform()

    def transform(self, x: np.ndarray) -> np.ndarray:
        n = len(x)
        min_len = max(2, n // (2**self.max_level))
        levels = self._wt.multi_level_decompose(
            x, wavelet=self.wavelet, max_level=self.max_level
        )
        coeffs = []
        for level, approx, detail in levels:
            coeffs.append(approx if len(detail) == 0 else detail)
        result = np.concatenate(coeffs) if coeffs else np.array([], dtype=np.float64)
        if len(result) != n:
            result = np.resize(result, n)
        return result

    def inverse(self, x: np.ndarray) -> np.ndarray:
        levels = self._wt.multi_level_decompose(
            x, wavelet=self.wavelet, max_level=self.max_level
        )
        return self._wt.multi_level_reconstruct(levels, wavelet=self.wavelet)


# ── Unified engine (placeholder for future wiring) ─────────────────────
