"""
Compression Profiler — tensor analysis and sensitivity profiling
================================================================
Statistical profiling, spectral profiling, sensitivity analysis,
and method recommendation per tensor.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spectralstream.core.math_primitives import (
    dct,
    idct,
    spectral_entropy,
    cosine_similarity,
)
from spectralstream.compression.engine._dataclasses import TensorProfile

logger = logging.getLogger(__name__)


@dataclass
class SensitivityResult:
    method: str
    bit_widths: List[int]
    mse_per_bit: List[float]
    compression_ratios: List[float]
    optimal_bit_width: int
    cliff_point: Optional[int] = None


class CompressionProfiler:
    def __init__(
        self,
        n_sensitivity_bits: Optional[List[int]] = None,
        kurtosis_threshold: float = 3.0,
        sparsity_threshold: float = 0.80,
        concentration_threshold: float = 0.30,
    ) -> None:
        self.n_sensitivity_bits = n_sensitivity_bits or [1, 2, 3, 4, 5, 6, 8]
        self.kurtosis_threshold = kurtosis_threshold
        self.sparsity_threshold = sparsity_threshold
        self.concentration_threshold = concentration_threshold
        self._profiles: List[TensorProfile] = []

    def profile(self, tensor: np.ndarray, name: Optional[str] = None) -> TensorProfile:
        tensor = np.asarray(tensor)
        t = TensorProfile(
            name=name or "",
            shape=tensor.shape,
            dtype=str(tensor.dtype),
            n_elements=tensor.size,
            nbytes=tensor.nbytes,
        )
        flat = tensor.ravel().astype(np.float64)
        self._compute_statistics(t, flat)
        self._compute_spectral_properties(t, flat)
        self._compute_compressibility(t)
        self._recommend_method(t)
        self._profiles.append(t)
        return t

    def sensitivity_analysis(
        self,
        tensor: np.ndarray,
        method: str = "dct",
        bit_widths: Optional[List[int]] = None,
    ) -> SensitivityResult:
        tensor = np.asarray(tensor, dtype=np.float64)
        if bit_widths is None:
            bit_widths = self.n_sensitivity_bits
        flat = tensor.ravel()
        mse_list: List[float] = []
        ratio_list: List[float] = []
        cliff_point: Optional[int] = None
        prev_mse = 0.0
        for bits in bit_widths:
            if method == "dct":
                mse, ratio = self._test_dct_quantize(tensor, bits)
            elif method == "uniform":
                mse, ratio = self._test_uniform_quantize(tensor, bits)
            elif method == "hadamard":
                mse, ratio = self._test_hadamard_quantize(tensor, bits)
            else:
                mse, ratio = 0.0, 1.0
            mse_list.append(mse)
            ratio_list.append(ratio)
            if prev_mse > 1e-10 and mse > prev_mse * 10:
                if cliff_point is None:
                    cliff_point = bits
            prev_mse = mse
        optimal_bits = bit_widths[-1]
        for i, (mse_v, b) in enumerate(zip(mse_list, bit_widths)):
            if mse_v < 0.01:
                optimal_bits = b
                break
        return SensitivityResult(
            method=method,
            bit_widths=list(bit_widths),
            mse_per_bit=mse_list,
            compression_ratios=ratio_list,
            optimal_bit_width=optimal_bits,
            cliff_point=cliff_point,
        )

    def batch_profile(self, tensors: Dict[str, np.ndarray]) -> Dict[str, TensorProfile]:
        return {
            name: self.profile(tensor, name=name) for name, tensor in tensors.items()
        }

    def get_recommendation_summary(
        self, profiles: Optional[List[TensorProfile]] = None
    ) -> Dict[str, Any]:
        if profiles is None:
            profiles = self._profiles
        if not profiles:
            return {"count": 0}
        method_counts: Dict[str, int] = {}
        total_bytes = 0
        for p in profiles:
            method_counts[p.recommended_method] = (
                method_counts.get(p.recommended_method, 0) + 1
            )
            total_bytes += p.nbytes
        return {
            "count": len(profiles),
            "total_bytes": total_bytes,
            "method_distribution": method_counts,
            "avg_sparsity": float(np.mean([p.sparsity for p in profiles])),
            "avg_entropy": float(np.mean([p.spectral_entropy for p in profiles])),
        }

    def _compute_statistics(self, t: TensorProfile, flat: np.ndarray) -> None:
        if flat.size == 0:
            return
        t.mean = float(np.mean(flat))
        t.std = float(np.std(flat))
        t.min_val = float(np.min(flat))
        t.max_val = float(np.max(flat))
        t.sparsity = float(np.mean(flat == 0.0))
        if t.std > 1e-10:
            centered = (flat - t.mean) / t.std
            t.skewness = float(np.mean(centered**3))
            t.kurtosis = float(np.mean(centered**4) - 3.0)
        else:
            t.skewness = 0.0
            t.kurtosis = 0.0

    def _compute_spectral_properties(self, t: TensorProfile, flat: np.ndarray) -> None:
        n = flat.size
        if n < 4:
            t.spectral_entropy = 0.0
            t.spectral_concentration = 1.0
            t.energy_concentration = 1.0
            t.effective_rank = 1.0
            return
        sample = flat[: min(n, 4096)]
        t.spectral_entropy = spectral_entropy(sample)
        try:
            coeffs = dct(sample)
            energy = coeffs**2
            total_energy = float(np.sum(energy))
            if total_energy > 1e-10:
                sorted_energy = np.sort(energy.ravel())[::-1]
                cumulative = np.cumsum(sorted_energy) / total_energy
                k_90 = int(np.searchsorted(cumulative, 0.90)) + 1
                t.spectral_concentration = k_90 / max(n, 1)
                t.energy_concentration = t.spectral_concentration
            else:
                t.spectral_concentration = 1.0
                t.energy_concentration = 1.0
        except (ValueError, np.linalg.LinAlgError):
            t.spectral_concentration = 1.0
            t.energy_concentration = 1.0
        if t.n_elements > 1 and len(t.shape) == 2:
            try:
                s = np.linalg.svd(
                    flat.reshape(t.shape)[
                        : min(t.shape[0], 256), : min(t.shape[1], 256)
                    ],
                    compute_uv=False,
                )
                s_norm = s / (np.sum(s) + 1e-10)
                nonzero = s_norm[s_norm > 1e-10]
                t.effective_rank = float(np.exp(-np.sum(nonzero * np.log(nonzero))))
            except np.linalg.LinAlgError:
                t.effective_rank = min(t.shape) if len(t.shape) >= 2 else 1.0
        else:
            t.effective_rank = 1.0

    def _compute_compressibility(self, t: TensorProfile) -> None:
        score = 0.0
        weights = 0.0
        score += 0.3 * t.sparsity
        weights += 0.3
        score += 0.3 * (1.0 - t.spectral_concentration)
        weights += 0.3
        if t.kurtosis > 0:
            kurt_score = min(t.kurtosis / 10.0, 1.0)
        else:
            kurt_score = 0.0
        score += 0.2 * kurt_score
        weights += 0.2
        if len(t.shape) >= 2:
            max_rank = min(t.shape)
            rank_ratio = t.effective_rank / max(max_rank, 1)
            score += 0.2 * (1.0 - rank_ratio)
        weights += 0.2
        t.compressibility_score = score / max(weights, 1e-10)

    def _recommend_method(self, t: TensorProfile) -> None:
        if t.sparsity > self.sparsity_threshold:
            t.recommended_method = "sparsify"
            t.recommended_bits = 0
        elif t.spectral_concentration < self.concentration_threshold:
            t.recommended_method = "spectral_dct"
            t.recommended_bits = 4
        elif len(t.shape) == 2 and t.effective_rank < min(t.shape) * 0.3:
            t.recommended_method = "low_rank"
            t.recommended_bits = 4
        elif t.compressibility_score > 0.6:
            t.recommended_method = "hadamard_quantize"
            t.recommended_bits = 4
        else:
            t.recommended_method = "spectral_dct"
            t.recommended_bits = 4
        if t.kurtosis > self.kurtosis_threshold:
            t.recommended_bits = min(t.recommended_bits + 1, 8)

    def _test_dct_quantize(self, tensor: np.ndarray, bits: int) -> Tuple[float, float]:
        flat = tensor.ravel().astype(np.float64)
        n = flat.size
        n_padded = 1 << (n - 1).bit_length() if n > 1 else 1
        padded = np.pad(flat, (0, n_padded - n)) if n_padded > n else flat
        coeffs = dct(padded)
        n_levels = 1 << bits
        scale = max(abs(float(np.max(coeffs))), abs(float(np.min(coeffs))), 1e-8)
        half_range = max(n_levels / 2 - 1, 1)
        quantized = np.clip(
            np.round((coeffs / scale) * half_range), -n_levels / 2, half_range
        )
        dequantized = (quantized / half_range) * scale
        reconstructed = idct(dequantized)
        mse = float(np.mean((padded - reconstructed[:n_padded]) ** 2))
        compressed_bytes = n_padded * bits // 8 + 8
        ratio = tensor.nbytes / max(compressed_bytes, 1)
        return mse, ratio

    def _test_uniform_quantize(
        self, tensor: np.ndarray, bits: int
    ) -> Tuple[float, float]:
        flat = tensor.ravel().astype(np.float64)
        n_levels = 1 << bits
        lo, hi = float(np.min(flat)), float(np.max(flat))
        scale = (hi - lo) / max(n_levels - 1, 1)
        if scale < 1e-10:
            scale = 1.0
        quantized = np.clip(np.round((flat - lo) / scale), 0, n_levels - 1)
        dequantized = quantized * scale + lo
        mse = float(np.mean((flat - dequantized) ** 2))
        compressed_bytes = flat.size * bits // 8 + 16
        ratio = tensor.nbytes / max(compressed_bytes, 1)
        return mse, ratio

    def _test_hadamard_quantize(
        self, tensor: np.ndarray, bits: int
    ) -> Tuple[float, float]:
        from spectralstream.core.math_primitives import HadamardRotator

        flat = tensor.ravel().astype(np.float32)
        n = flat.size
        n_rotated = 1 << (n - 1).bit_length() if n > 1 else 1
        rotator = HadamardRotator(dim=n)
        padded = np.zeros(n_rotated, dtype=np.float32)
        padded[:n] = flat
        rotated = rotator.rotate(padded.reshape(1, -1)).ravel()
        n_levels = 1 << bits
        lo, hi = float(np.min(rotated)), float(np.max(rotated))
        scale = (hi - lo) / max(n_levels - 1, 1)
        quantized = np.clip(np.round((rotated - lo) / scale), 0, n_levels - 1)
        dequantized = quantized * scale + lo
        inv_rotated = rotator.inverse_rotate(dequantized.reshape(1, -1)).ravel()
        mse = float(np.mean((padded - inv_rotated) ** 2))
        compressed_bytes = n_rotated * bits // 8 + 16
        ratio = tensor.nbytes / max(compressed_bytes, 1)
        return mse, ratio
