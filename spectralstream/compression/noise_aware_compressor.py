"""
Noise-Aware Compressor for SpectralStream
==========================================
Compression that exploits the noise floor in tensor data.
Detects signal/noise boundary in singular value spectrum,
discards the noise subspace, and exploits BF16 quantization
noise floor for higher compression ratios.

Key insight: Neural network weight matrices contain both signal
and noise. By identifying where signal ends and noise begins
in the singular value spectrum, we can discard noise subspace
components without meaningful accuracy loss, achieving compression
beyond what signal-only methods allow.
"""

from __future__ import annotations


import logging
import math
import pickle
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from spectralstream.core.math_primitives import (
    LloydMaxQuantizer,
    dct,
    idct,
    spectral_entropy,
    cosine_similarity,
    BAND_COMPRESSION,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Noise Floor Detection Methods
# ═══════════════════════════════════════════════════════════════════════════


class NoiseFloorDetector:
    """Detects signal/noise boundary in a singular value spectrum.

    Uses multiple heuristics:
        1. Marchenko-Pastur distribution fitting
        2. eigenvalue ratio test (Ginibre ensemble)
        3. Scree plot elbow detection
        4. Bayesian information criterion
    """

    @staticmethod
    def marchenko_pastur_bound(
        singular_values: np.ndarray,
        n_rows: int,
        n_cols: int,
    ) -> int:
        """Estimate signal rank via Marchenko-Pastur distribution.

        The MP distribution describes eigenvalues of random matrices.
        Signal components lie above the MP upper bound.

        Args:
            singular_values: Sorted singular values (descending).
            n_rows: Number of rows in original matrix.
            n_cols: Number of columns in original matrix.

        Returns:
            Estimated number of signal components.
        """
        n = max(n_rows, n_cols)
        m = min(n_rows, n_cols)
        if m == 0:
            return 0

        gamma = m / n
        sigma_sq = float(np.var(singular_values)) if len(singular_values) > 1 else 1.0
        sigma_sq = max(sigma_sq, 1e-10)

        # MP upper bound
        mp_upper = sigma_sq * (1 + math.sqrt(gamma)) ** 2

        # Count singular values above MP bound
        signal_count = int(np.sum(singular_values**2 > mp_upper))
        return max(1, signal_count)

    @staticmethod
    def eigenvalue_ratio_test(
        singular_values: np.ndarray,
        threshold: float = 2.0,
    ) -> int:
        """Detect signal subspace via eigenvalue ratio test.

        Compares consecutive eigenvalue ratios. Signal eigenvalues
        form a cluster at the top; noise eigenvalues form a flat tail.

        Args:
            singular_values: Sorted singular values (descending).
            threshold: Minimum ratio to consider as signal boundary.

        Returns:
            Estimated number of signal components.
        """
        if len(singular_values) < 2:
            return len(singular_values)

        ratios = singular_values[:-1] / (singular_values[1:] + 1e-10)

        # Find first index where ratio drops below threshold
        for i in range(len(ratios)):
            if ratios[i] < threshold:
                return max(1, i)

        return max(1, len(singular_values) - 1)

    @staticmethod
    def scree_elbow_detect(
        singular_values: np.ndarray,
    ) -> int:
        """Detect elbow in scree plot (singular value curve).

        Uses the Kneedle algorithm variant: find the point of
        maximum curvature in the cumulative energy curve.

        Args:
            singular_values: Sorted singular values (descending).

        Returns:
            Estimated number of signal components.
        """
        n = len(singular_values)
        if n < 3:
            return n

        total_energy = float(np.sum(singular_values**2))
        if total_energy < 1e-10:
            return 1

        cumulative = np.cumsum(singular_values**2) / total_energy

        # Normalize x and y to [0, 1]
        x = np.linspace(0, 1, n)
        y = cumulative

        # Line from first to last point
        line_start = np.array([x[0], y[0]])
        line_end = np.array([x[-1], y[-1]])
        line_vec = line_end - line_start
        line_len = np.linalg.norm(line_vec)
        if line_len < 1e-10:
            return 1
        line_unit = line_vec / line_len

        # Distance from each point to the line
        max_dist = 0.0
        elbow_idx = 1
        for i in range(1, n - 1):
            pt = np.array([x[i], y[i]])
            vec = pt - line_start
            proj = np.dot(vec, line_unit)
            closest = line_start + proj * line_unit
            dist = np.linalg.norm(pt - closest)
            if dist > max_dist:
                max_dist = dist
                elbow_idx = i

        return max(1, elbow_idx)

    @staticmethod
    def bayesian_threshold(
        singular_values: np.ndarray,
        n_rows: int,
        n_cols: int,
    ) -> int:
        """Bayesian information criterion for signal rank estimation.

        Balances model complexity (number of components) against
        reconstruction quality.

        Args:
            singular_values: Sorted singular values (descending).
            n_rows: Number of rows in original matrix.
            n_cols: Number of columns in original matrix.

        Returns:
            Estimated optimal number of signal components.
        """
        n = len(singular_values)
        if n == 0:
            return 0

        total_energy = float(np.sum(singular_values**2))
        if total_energy < 1e-10:
            return 1

        total_samples = n_rows * n_cols
        best_bic = float("inf")
        best_rank = 1

        for rank in range(1, n + 1):
            # Reconstruction error from truncated SVD
            signal_energy = float(np.sum(singular_values[:rank] ** 2))
            noise_energy = total_energy - signal_energy
            noise_var = max(
                noise_energy / max(total_samples - rank * (n_rows + n_cols - rank), 1),
                1e-10,
            )

            # Log-likelihood (assuming Gaussian noise)
            log_lik = (
                -0.5 * total_samples * math.log(2 * math.pi * noise_var)
                - 0.5 * noise_energy / noise_var
            )

            # BIC penalty
            n_params = rank * (n_rows + n_cols - rank)
            bic = -2 * log_lik + n_params * math.log(max(total_samples, 1))

            if bic < best_bic:
                best_bic = bic
                best_rank = rank

        return max(1, best_rank)


# ═══════════════════════════════════════════════════════════════════════════
# Noise-Aware Compression Result
# ═══════════════════════════════════════════════════════════════════════════


class NoiseAwareResult:
    """Container for noise-aware compression output."""

    __slots__ = (
        "compressed_data",
        "original_shape",
        "method",
        "compression_ratio",
        "reconstruction_error",
        "signal_rank",
        "noise_floor_estimate",
        "metadata",
    )

    def __init__(
        self,
        compressed_data: Any,
        original_shape: Tuple[int, ...],
        method: str,
        compression_ratio: float,
        reconstruction_error: float,
        signal_rank: int,
        noise_floor_estimate: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.compressed_data = compressed_data
        self.original_shape = original_shape
        self.method = method
        self.compression_ratio = compression_ratio
        self.reconstruction_error = reconstruction_error
        self.signal_rank = signal_rank
        self.noise_floor_estimate = noise_floor_estimate
        self.metadata = metadata or {}

    def __repr__(self) -> str:
        return (
            f"NoiseAwareResult(method={self.method!r}, "
            f"ratio={self.compression_ratio:.2f}x, "
            f"signal_rank={self.signal_rank}, "
            f"noise_floor={self.noise_floor_estimate:.6f})"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Noise-Aware Compressor
# ═══════════════════════════════════════════════════════════════════════════


class NoiseAwareCompressor:
    """Compression that exploits the noise floor in neural network weights.

    Key techniques:
        1. SVD-based noise subspace detection and removal
        2. DCT-domain noise floor exploitation
        3. BF16 quantization-aware compression (exploit the
           ~0.78% relative error floor of BF16)
        4. Spectral thresholding in transform domain

    The noise floor of BF16 format is approximately:
        eps_bf16 ≈ 2^(-7) ≈ 0.0078 (relative)

    Any weight component below this floor is indistinguishable from
    quantization noise, so we can discard it for free.

    Usage:
        compressor = NoiseAwareCompressor()
        result = compressor.compress(weight_matrix)
        decompressed = compressor.decompress(result)
    """

    def __init__(
        self,
        noise_floor_method: str = "auto",
        bf16_noise_floor: float = 0.0078,
        energy_threshold: float = 0.95,
        min_signal_rank: int = 1,
        max_signal_ratio: float = 0.8,
        enable_spectral_thresholding: bool = True,
    ) -> None:
        self.noise_floor_method = noise_floor_method
        self.bf16_noise_floor = bf16_noise_floor
        self.energy_threshold = energy_threshold
        self.min_signal_rank = min_signal_rank
        self.max_signal_ratio = max_signal_ratio
        self.enable_spectral_thresholding = enable_spectral_thresholding

        self._detector = NoiseFloorDetector()
        self._quantizers: Dict[int, LloydMaxQuantizer] = {}

        logger.info(
            "NoiseAwareCompressor initialized: method=%s, bf16_floor=%.4f, "
            "energy_threshold=%.2f",
            noise_floor_method,
            bf16_noise_floor,
            energy_threshold,
        )

    # ── Public API ────────────────────────────────────────────────────────

    def compress(
        self,
        tensor: np.ndarray,
        method: Optional[str] = None,
        **kwargs: Any,
    ) -> Tuple[bytes, Dict[str, Any]]:
        """Compress a tensor by exploiting noise floor.

        Args:
            tensor: Input weight tensor.
            method: Compression method ('svd_noise', 'dct_noise',
                    'bf16_exploit', 'auto').
            **kwargs: Additional parameters.

        Returns:
            Tuple of (compressed_bytes, metadata_dict).
        """
        tensor = np.asarray(tensor, dtype=np.float64)
        start = time.perf_counter()

        if method is None or method == "auto":
            method = self._auto_select_method(tensor)

        if method == "svd_noise":
            result = self._compress_svd_noise(tensor, **kwargs)
        elif method == "dct_noise":
            result = self._compress_dct_noise(tensor, **kwargs)
        elif method == "bf16_exploit":
            result = self._compress_bf16_exploit(tensor, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method!r}")

        elapsed = (time.perf_counter() - start) * 1000.0
        result.metadata["compress_time_ms"] = elapsed

        logger.info(
            "Noise-aware compress: method=%s, shape=%s, ratio=%.2fx, "
            "signal_rank=%d, noise_floor=%.6f, time=%.2fms",
            method,
            tensor.shape,
            result.compression_ratio,
            result.signal_rank,
            result.noise_floor_estimate,
            elapsed,
        )

        data_bytes = pickle.dumps(result.compressed_data)
        metadata: Dict[str, Any] = {
            "orig_shape": list(result.original_shape),
            "method": result.method,
            "signal_rank": result.signal_rank,
            "noise_floor_estimate": result.noise_floor_estimate,
            "compression_ratio": result.compression_ratio,
            "reconstruction_error": result.reconstruction_error,
            "compressed_data_size": len(data_bytes),
        }
        metadata.update(result.metadata)
        return data_bytes, metadata

    def decompress(self, data: bytes, metadata: Dict[str, Any]) -> np.ndarray:
        """Decompress from noise-aware representation.

        Args:
            data: Compressed bytes from compress().
            metadata: Metadata dict from compress().

        Returns:
            Reconstructed numpy array.
        """
        method = metadata["method"]
        compressed_data = pickle.loads(data)
        orig_shape = tuple(metadata["orig_shape"])

        if method == "svd_noise":
            return self._decompress_svd_noise(compressed_data, orig_shape)
        elif method == "dct_noise":
            return self._decompress_dct_noise(compressed_data, orig_shape)
        elif method == "bf16_exploit":
            return self._decompress_bf16_exploit(compressed_data, orig_shape)
        else:
            raise ValueError(f"Unknown method: {method!r}")

    def detect_noise_floor(
        self,
        tensor: np.ndarray,
    ) -> Tuple[int, float]:
        """Detect signal/noise boundary in a tensor.

        Args:
            tensor: Input tensor (2D for SVD, 1D for spectral).

        Returns:
            Tuple of (signal_rank, estimated_noise_floor).
        """
        tensor = np.asarray(tensor, dtype=np.float64)

        if tensor.ndim == 2:
            return self._detect_svd_noise_floor(tensor)
        else:
            return self._detect_spectral_noise_floor(tensor)

    def get_compression_potential(
        self,
        tensor: np.ndarray,
    ) -> Dict[str, float]:
        """Estimate how compressible a tensor is via noise exploitation.

        Args:
            tensor: Input tensor.

        Returns:
            Dictionary with compression potential metrics.
        """
        tensor = np.asarray(tensor, dtype=np.float64)

        if tensor.ndim == 2:
            signal_rank, noise_floor = self._detect_svd_noise_floor(tensor)
            U, s, Vt = np.linalg.svd(tensor, full_matrices=False)
            total_energy = float(np.sum(s**2))
            signal_energy = float(np.sum(s[:signal_rank] ** 2))
            noise_energy = total_energy - signal_energy
        else:
            flat = tensor.ravel()
            coeffs = np.abs(dct(flat))
            total_energy = float(np.sum(coeffs**2))
            threshold = self.bf16_noise_floor * np.max(np.abs(flat))
            signal_mask = coeffs > threshold
            signal_energy = float(np.sum(coeffs[signal_mask] ** 2))
            noise_energy = total_energy - signal_energy
            signal_rank = int(np.sum(signal_mask))

        total_elements = tensor.size
        return {
            "signal_rank": float(signal_rank),
            "signal_fraction": signal_rank / max(total_elements, 1),
            "noise_fraction": 1.0 - signal_rank / max(total_elements, 1),
            "signal_energy_fraction": signal_energy / max(total_energy, 1e-10),
            "noise_energy_fraction": noise_energy / max(total_energy, 1e-10),
            "max_compression_from_noise": total_elements / max(signal_rank, 1),
        }

    # ── SVD Noise-Aware Compression ───────────────────────────────────────

    def _compress_svd_noise(
        self,
        tensor: np.ndarray,
        **kwargs: Any,
    ) -> NoiseAwareResult:
        """Compress via SVD with noise subspace removal.

        Steps:
            1. Compute full SVD.
            2. Detect signal/noise boundary.
            3. Discard noise subspace.
            4. Quantize signal subspace components.
        """
        if tensor.ndim < 2:
            # Fall back to 1D spectral noise removal
            return self._compress_dct_noise(tensor, **kwargs)

        n_rows, n_cols = tensor.shape
        U, s, Vt = np.linalg.svd(tensor, full_matrices=False)

        # Detect signal rank
        method = kwargs.get("detection_method", self.noise_floor_method)
        if method == "auto" or method is None:
            method = self._choose_detection_method(s, n_rows, n_cols)

        if method == "marchenko_pastur":
            signal_rank = self._detector.marchenko_pastur_bound(s, n_rows, n_cols)
        elif method == "eigenvalue_ratio":
            signal_rank = self._detector.eigenvalue_ratio_test(s)
        elif method == "scree":
            signal_rank = self._detector.scree_elbow_detect(s)
        elif method == "bayesian":
            signal_rank = self._detector.bayesian_threshold(s, n_rows, n_cols)
        else:
            signal_rank = self._detector.scree_elbow_detect(s)

        # Enforce constraints
        max_rank = int(min(n_rows, n_cols) * self.max_signal_ratio)
        signal_rank = max(self.min_signal_rank, min(signal_rank, max_rank))

        # Energy-based refinement
        total_energy = float(np.sum(s**2))
        if total_energy > 1e-10:
            energy_cumsum = np.cumsum(s[:signal_rank] ** 2) / total_energy
            # Trim to energy threshold
            for i in range(signal_rank - 1, 0, -1):
                if energy_cumsum[i] >= self.energy_threshold:
                    signal_rank = i + 1
                    break

        # Extract signal subspace
        U_s = U[:, :signal_rank]
        s_s = s[:signal_rank]
        Vt_s = Vt[:signal_rank, :]

        # Estimate noise floor from discarded singular values
        noise_sv = s[signal_rank:]
        noise_floor = float(np.mean(noise_sv**2)) if len(noise_sv) > 0 else 0.0

        # Quantize signal components
        quant_bits = kwargs.get("quantize_bits", 4)
        quantizer = self._get_quantizer(quant_bits)

        # Pack signal components into a flat array for quantization
        signal_flat = np.concatenate([s_s, U_s.ravel(), Vt_s.ravel()])
        if not quantizer.trained:
            quantizer.train(signal_flat)
        indices, centroids = quantizer.compress(signal_flat)

        # Reconstruction
        dequant = centroids[indices]
        s_recon = dequant[:signal_rank]
        U_recon = dequant[signal_rank : signal_rank + n_rows * signal_rank].reshape(
            n_rows, signal_rank
        )
        Vt_recon = dequant[signal_rank + n_rows * signal_rank :].reshape(
            signal_rank, n_cols
        )

        reconstructed = U_recon @ np.diag(s_recon) @ Vt_recon
        mse = float(np.mean((tensor - reconstructed) ** 2))

        original_bytes = tensor.nbytes
        compressed_bytes = indices.nbytes + centroids.nbytes
        ratio = original_bytes / max(compressed_bytes, 1)

        return NoiseAwareResult(
            compressed_data={
                "indices": indices,
                "centroids": centroids,
                "signal_rank": signal_rank,
                "n_rows": n_rows,
                "n_cols": n_cols,
                "quant_bits": quant_bits,
            },
            original_shape=tensor.shape,
            method="svd_noise",
            compression_ratio=ratio,
            reconstruction_error=mse,
            signal_rank=signal_rank,
            noise_floor_estimate=noise_floor,
            metadata={
                "detection_method": method,
                "noise_sv_count": len(noise_sv),
                "energy_retained": float(np.sum(s_s**2)) / max(total_energy, 1e-10),
            },
        )

    def _decompress_svd_noise(
        self,
        data: Dict[str, Any],
        shape: Tuple[int, ...],
    ) -> np.ndarray:
        """Reconstruct from SVD noise-aware representation."""
        indices = data["indices"]
        centroids = data["centroids"]
        signal_rank = data["signal_rank"]
        n_rows = data["n_rows"]
        n_cols = data["n_cols"]
        quant_bits = data["quant_bits"]

        quantizer = self._get_quantizer(quant_bits)
        quantizer.centroids = centroids
        quantizer.trained = True

        dequant = quantizer.centroids[indices]
        s_recon = dequant[:signal_rank]
        U_recon = dequant[signal_rank : signal_rank + n_rows * signal_rank].reshape(
            n_rows, signal_rank
        )
        Vt_recon = dequant[signal_rank + n_rows * signal_rank :].reshape(
            signal_rank, n_cols
        )

        return (U_recon @ np.diag(s_recon) @ Vt_recon).reshape(shape)

    # ── DCT Noise-Aware Compression ───────────────────────────────────────

    def _compress_dct_noise(
        self,
        tensor: np.ndarray,
        **kwargs: Any,
    ) -> NoiseAwareResult:
        """Compress via DCT with noise floor thresholding.

        Discards DCT coefficients whose magnitude falls below
        the estimated noise floor.
        """
        flat = tensor.ravel().astype(np.float64)
        n = flat.size

        # Compute noise floor estimate
        noise_floor = self.bf16_noise_floor * np.max(np.abs(flat))
        if noise_floor < 1e-10:
            noise_floor = float(np.std(flat) * 0.1)

        # DCT
        coeffs = dct(flat)

        # Signal thresholding
        signal_mask = np.abs(coeffs) > noise_floor
        signal_indices = np.where(signal_mask)[0]
        signal_values = coeffs[signal_mask]

        signal_rank = len(signal_indices)
        total_energy = float(np.sum(coeffs**2))
        signal_energy = float(np.sum(signal_values**2))

        # Reconstruct
        sparse_coeffs = np.zeros(n, dtype=np.float64)
        sparse_coeffs[signal_indices] = signal_values
        reconstructed = idct(sparse_coeffs)

        mse = float(np.mean((flat - reconstructed) ** 2))
        original_bytes = tensor.nbytes
        compressed_bytes = signal_values.nbytes + signal_indices.nbytes + 16
        ratio = original_bytes / max(compressed_bytes, 1)

        return NoiseAwareResult(
            compressed_data={
                "indices": signal_indices,
                "values": signal_values,
                "noise_floor": noise_floor,
                "n_original": n,
            },
            original_shape=tensor.shape,
            method="dct_noise",
            compression_ratio=ratio,
            reconstruction_error=mse,
            signal_rank=signal_rank,
            noise_floor_estimate=noise_floor,
            metadata={
                "signal_fraction": signal_rank / max(n, 1),
                "energy_retained": signal_energy / max(total_energy, 1e-10),
            },
        )

    def _decompress_dct_noise(
        self,
        data: Dict[str, Any],
        shape: Tuple[int, ...],
    ) -> np.ndarray:
        """Reconstruct from DCT noise-aware representation."""
        indices = data["indices"]
        values = data["values"]
        n = data["n_original"]

        coeffs = np.zeros(n, dtype=np.float64)
        coeffs[indices] = values
        return idct(coeffs).reshape(shape)

    # ── BF16 Noise Floor Exploitation ─────────────────────────────────────

    def _compress_bf16_exploit(
        self,
        tensor: np.ndarray,
        **kwargs: Any,
    ) -> NoiseAwareResult:
        """Exploit BF16 quantization noise floor for compression.

        BF16 has ~8 exponent bits + 7 mantissa bits, giving
        relative precision of ~2^(-7) ≈ 0.78%. Any weight
        component smaller than this relative to the local scale
        is effectively zero in BF16.

        Strategy:
            1. Identify components below BF16 noise floor.
            2. Zero them out (they're lost anyway in BF16).
            3. Encode remaining components at reduced precision.
        """
        flat = tensor.ravel().astype(np.float64)
        n = flat.size

        # BF16 noise floor analysis
        local_scale = np.maximum(np.abs(flat), 1e-10)
        relative_error = self.bf16_noise_floor

        # Components below BF16 floor
        below_floor = np.abs(flat) < relative_error * local_scale
        above_floor = ~below_floor

        # Also apply spectral thresholding for additional compression
        if self.enable_spectral_thresholding:
            coeffs = np.abs(dct(flat))
            spectral_threshold = float(np.median(coeffs) * 0.1)
            spectral_below = coeffs < spectral_threshold
            # Combine: remove both BF16-invisible and spectrally insignificant
            combined_below = below_floor | spectral_below
            above_mask = ~combined_below
        else:
            above_mask = above_floor

        signal_values = flat[above_mask]
        signal_indices = np.where(above_mask)[0]
        signal_rank = len(signal_indices)

        # Quantize signal components at reduced precision (6 bits is enough
        # since BF16 noise floor already limits precision)
        quant_bits = kwargs.get("quantize_bits", 6)
        quantizer = self._get_quantizer(quant_bits)
        if not quantizer.trained:
            quantizer.train(signal_values)
        indices, centroids = quantizer.compress(signal_values)

        # Reconstruction
        dequant = centroids[indices]
        reconstructed_flat = np.zeros(n, dtype=np.float64)
        reconstructed_flat[signal_indices] = dequant

        mse = float(np.mean((flat - reconstructed_flat) ** 2))
        original_bytes = tensor.nbytes
        compressed_bytes = signal_indices.nbytes + indices.nbytes + centroids.nbytes
        ratio = original_bytes / max(compressed_bytes, 1)

        # Noise floor estimate
        noise_components = flat[below_floor]
        noise_floor = (
            float(np.mean(noise_components**2)) if len(noise_components) > 0 else 0.0
        )

        return NoiseAwareResult(
            compressed_data={
                "signal_indices": signal_indices,
                "quantized_indices": indices,
                "centroids": centroids,
                "n_original": n,
                "quant_bits": quant_bits,
            },
            original_shape=tensor.shape,
            method="bf16_exploit",
            compression_ratio=ratio,
            reconstruction_error=mse,
            signal_rank=signal_rank,
            noise_floor_estimate=noise_floor,
            metadata={
                "bf16_noise_floor": relative_error,
                "below_floor_fraction": float(np.mean(below_floor)),
                "above_floor_fraction": float(np.mean(above_floor)),
            },
        )

    def _decompress_bf16_exploit(
        self,
        data: Dict[str, Any],
        shape: Tuple[int, ...],
    ) -> np.ndarray:
        """Reconstruct from BF16-exploited representation."""
        signal_indices = data["signal_indices"]
        quantized_indices = data["quantized_indices"]
        centroids = data["centroids"]
        n = data["n_original"]
        quant_bits = data["quant_bits"]

        quantizer = self._get_quantizer(quant_bits)
        quantizer.centroids = centroids
        quantizer.trained = True

        dequant = quantizer.centroids[quantized_indices]
        flat = np.zeros(n, dtype=np.float64)
        flat[signal_indices] = dequant
        return flat.reshape(shape)

    # ── Noise Floor Detection Helpers ─────────────────────────────────────

    def _detect_svd_noise_floor(
        self,
        matrix: np.ndarray,
    ) -> Tuple[int, float]:
        """Detect signal/noise boundary via SVD.

        Returns:
            Tuple of (signal_rank, noise_floor_estimate).
        """
        U, s, Vt = np.linalg.svd(matrix, full_matrices=False)
        n_rows, n_cols = matrix.shape

        signal_rank = self._detector.scree_elbow_detect(s)
        noise_sv = s[signal_rank:]
        noise_floor = float(np.mean(noise_sv**2)) if len(noise_sv) > 0 else 0.0

        return signal_rank, noise_floor

    def _detect_spectral_noise_floor(
        self,
        signal: np.ndarray,
    ) -> Tuple[int, float]:
        """Detect signal/noise boundary in 1-D via DCT.

        Returns:
            Tuple of (signal_coefficient_count, noise_floor_estimate).
        """
        flat = signal.ravel().astype(np.float64)
        coeffs = np.abs(dct(flat))

        total_energy = float(np.sum(coeffs**2))
        if total_energy < 1e-10:
            return 1, 0.0

        # Estimate noise floor from high-frequency coefficients
        n = len(coeffs)
        noise_band = coeffs[n * 3 // 4 :]
        noise_floor = float(np.mean(noise_band**2))

        signal_mask = coeffs > noise_floor * 2
        signal_count = int(np.sum(signal_mask))

        return max(1, signal_count), noise_floor

    def _auto_select_method(self, tensor: np.ndarray) -> str:
        """Auto-select best noise-aware method."""
        if tensor.ndim == 2:
            n_rows, n_cols = tensor.shape
            # For 2D matrices, SVD noise removal is most effective
            if min(n_rows, n_cols) <= 512:
                return "svd_noise"
            else:
                return "dct_noise"
        else:
            return "dct_noise"

    def _choose_detection_method(
        self,
        singular_values: np.ndarray,
        n_rows: int,
        n_cols: int,
    ) -> str:
        """Choose best noise floor detection method."""
        n = len(singular_values)
        # For small matrices, use eigenvalue ratio
        if n < 20:
            return "eigenvalue_ratio"
        # For large well-conditioned, use Marchenko-Pastur
        gamma = min(n_rows, n_cols) / max(n_rows, n_cols)
        if 0.3 < gamma < 0.9:
            return "marchenko_pastur"
        # Default to scree (robust)
        return "scree"

    def _get_quantizer(self, n_bits: int) -> LloydMaxQuantizer:
        if n_bits not in self._quantizers:
            self._quantizers[n_bits] = LloydMaxQuantizer(n_bits=n_bits)
        return self._quantizers[n_bits]
