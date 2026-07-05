"""
Hadamard Preconditioner for SpectralStream
============================================
Block-diagonal Fast Walsh-Hadamard Transform (FWHT) with random
sign flips, incoherence transforms via QR rotation and butterfly
matrices, and spectral shaping for quantization-aware preconditioning.

Architecture:
  1. HadamardPreconditioner — block-diagonal FWHT + random sign flips
  2. IncoherenceTransform — QR random rotation, butterfly matrix
  3. SpectralShaping — spectral equalization, error diffusion
"""

from __future__ import annotations


import logging
import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from spectralstream.core.math_primitives import (
    HadamardRotator,
    dct,
    fwht,
    idct,
    ifwht,
    next_power_of_two,
    softmax,
    spectral_entropy,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class HadamardPreconditionerConfig:
    """Configuration for Hadamard preconditioning."""
    block_size: int = 64
    n_sign_flips: int = 128
    normalize: bool = True
    seed: int = 42


@dataclass
class IncoherenceConfig:
    """Configuration for incoherence transforms."""
    rotation_dim: int = 256
    butterfly_stages: int = 4
    seed: int = 42


@dataclass
class SpectralShapingConfig:
    """Configuration for spectral shaping."""
    n_bands: int = 8
    equalization_strength: float = 0.5
    error_diffusion_weight: float = 0.1


# ═══════════════════════════════════════════════════════════════════════════
# 1. HadamardPreconditioner — block-diagonal FWHT
# ═══════════════════════════════════════════════════════════════════════════

class HadamardPreconditioner:
    """Block-diagonal FWHT preconditioner with random sign flips.

    Applies a randomized Hadamard transform to decorrelate weight
    elements before quantization, improving quantization quality.

    The transform is:
        y = H_block @ (signs * x)

    where H_block is a block-diagonal matrix of normalized Hadamard
    matrices and signs are random +/-1 patterns.

    This is an orthogonal transform that preserves L2 norm and
    improves the conditioning of the weight matrix for quantization.
    """

    def __init__(self, config: Optional[HadamardPreconditionerConfig] = None):
        self.config = config or HadamardPreconditionerConfig()
        self._rotators: dict[int, HadamardRotator] = {}
        self._signs_cache: dict[int, np.ndarray] = {}
        self._rng = np.random.RandomState(self.config.seed)

    def _get_rotator(self, dim: int) -> HadamardRotator:
        """Get or create a HadamardRotator for the given dimension."""
        if dim not in self._rotators:
            self._rotators[dim] = HadamardRotator(dim, seed=self.config.seed + dim)
        return self._rotators[dim]

    def _get_signs(self, n: int) -> np.ndarray:
        """Generate random sign pattern for the given length."""
        if n not in self._signs_cache:
            self._signs_cache[n] = self._rng.choice(
                [-1.0, 1.0], size=n
            ).astype(np.float32)
        return self._signs_cache[n]

    def precondition(self, x: np.ndarray) -> Tuple[np.ndarray, dict]:
        """Apply block-diagonal Hadamard preconditioning.

        Args:
            x: Input array of shape (n_rows, n_cols) or (n,).

        Returns:
            (preconditioned, metadata) where metadata contains
            the signs and block info for inversion.
        """
        x = np.asarray(x, dtype=np.float32)
        original_shape = x.shape

        if x.ndim == 1:
            # 1D: apply single block
            padded_len = next_power_of_two(len(x))
            padded = np.zeros(padded_len, dtype=np.float32)
            padded[: len(x)] = x
            signs = self._get_signs(padded_len)
            rotated = padded * signs
            result = fwht(rotated, normalize=self.config.normalize)
            return result[:len(x)].reshape(original_shape), {
                "signs": signs, "padded_len": padded_len,
                "original_len": len(x),
            }

        # 2D: apply block-diagonal along columns
        n_rows, n_cols = x.shape
        block_size = min(self.config.block_size, next_power_of_two(n_cols))

        # Pad columns to block_size multiple
        n_blocks = max(1, n_cols // block_size)
        padded_cols = n_blocks * block_size
        if padded_cols < n_cols:
            padded_cols = next_power_of_two(n_cols)
            n_blocks = padded_cols // block_size

        padded = np.zeros((n_rows, padded_cols), dtype=np.float32)
        padded[:, : n_cols] = x

        signs = self._get_signs(padded_cols)
        block_signs = signs.reshape(n_blocks, block_size)

        result = np.zeros_like(padded)
        for b in range(n_blocks):
            lo = b * block_size
            hi = lo + block_size
            block = padded[:, lo:hi] * block_signs[b]
            for row_idx in range(n_rows):
                result[row_idx, lo:hi] = fwht(
                    block[row_idx], normalize=self.config.normalize
                )

        return result[:, : n_cols].reshape(original_shape), {
            "signs": block_signs,
            "block_size": block_size,
            "n_blocks": n_blocks,
            "padded_cols": padded_cols,
        }

    def inverse_precondition(
        self, y: np.ndarray, metadata: dict
    ) -> np.ndarray:
        """Invert the Hadamard preconditioning.

        Args:
            y: Preconditioned array.
            metadata: Metadata from precondition().

        Returns:
            Original array.
        """
        y = np.asarray(y, dtype=np.float32)
        original_shape = y.shape

        if y.ndim == 1:
            padded_len = metadata["padded_len"]
            original_len = metadata["original_len"]
            signs = metadata["signs"]
            padded = np.zeros(padded_len, dtype=np.float32)
            padded[:len(y)] = y
            result = ifwht(padded, normalize=self.config.normalize)
            result = result * signs
            return result[:original_len].reshape(original_shape)

        block_signs = metadata["signs"]
        block_size = metadata["block_size"]
        n_blocks = metadata["n_blocks"]
        padded_cols = metadata["padded_cols"]
        n_cols = y.shape[1]

        padded = np.zeros((y.shape[0], padded_cols), dtype=np.float32)
        padded[:, :n_cols] = y

        result = np.zeros_like(padded)
        for b in range(n_blocks):
            lo = b * block_size
            hi = lo + block_size
            for row_idx in range(y.shape[0]):
                result[row_idx, lo:hi] = ifwht(
                    padded[row_idx, lo:hi], normalize=self.config.normalize
                )
            result[:, lo:hi] = result[:, lo:hi] * block_signs[b]

        return result[:, :n_cols].reshape(original_shape)


# ═══════════════════════════════════════════════════════════════════════════
# 2. IncoherenceTransform — QR rotation, butterfly matrix
# ═══════════════════════════════════════════════════════════════════════════

class IncoherenceTransform:
    """Incoherence transform to spread weight energy uniformly.

    Makes quantization more effective by ensuring that no single
    element dominates, using:
    1. Random QR rotation: O = Q where Q^T Q = I from QR of random matrix
    2. Butterfly matrix: O(n log n) structured rotation
    """

    def __init__(self, config: Optional[IncoherenceConfig] = None):
        self.config = config or IncoherenceConfig()
        self._rotation_cache: dict[int, np.ndarray] = {}
        self._rng = np.random.RandomState(self.config.seed)

    def _get_random_rotation(self, dim: int) -> np.ndarray:
        """Generate a random orthogonal rotation matrix via QR."""
        if dim not in self._rotation_cache:
            random_mat = self._rng.randn(dim, dim).astype(np.float64)
            Q, _ = np.linalg.qr(random_mat)
            self._rotation_cache[dim] = Q.astype(np.float32)
        return self._rotation_cache[dim]

    def _butterfly_matrix(self, n: int) -> np.ndarray:
        """Construct a butterfly matrix for O(n log n) rotation.

        Butterfly matrix is a product of sparse factors:
            B = B_1 @ B_2 @ ... @ B_k
        where each B_i is a block-diagonal matrix of 2x2 rotations.
        """
        n = next_power_of_two(n)
        result = np.eye(n, dtype=np.float64)

        stages = max(1, int(math.log2(n)))
        for stage in range(min(stages, self.config.butterfly_stages)):
            block_size = 1 << (stage + 1)
            half = block_size // 2
            theta = 2.0 * np.pi / block_size

            for i in range(0, n, block_size):
                for j in range(half):
                    idx1 = i + j
                    idx2 = i + j + half
                    if idx1 < n and idx2 < n:
                        angle = theta * j * (stage + 1)
                        c, s = math.cos(angle), math.sin(angle)
                        # Apply 2x2 rotation
                        r1 = result[idx1].copy()
                        r2 = result[idx2].copy()
                        result[idx1] = c * r1 + s * r2
                        result[idx2] = -s * r1 + c * r2

        return result.astype(np.float32)

    def qr_rotate(self, x: np.ndarray) -> Tuple[np.ndarray, dict]:
        """Apply random QR rotation for incoherence.

        Args:
            x: Input array.

        Returns:
            (rotated, metadata)
        """
        x = np.asarray(x, dtype=np.float64)
        original_shape = x.shape

        if x.ndim == 1:
            dim = len(x)
            Q = self._get_random_rotation(dim).astype(np.float64)
            rotated = Q @ x
            return rotated.astype(x.dtype), {"matrix": Q, "method": "qr"}

        # 2D: rotate along last axis
        dim = x.shape[-1]
        Q = self._get_random_rotation(dim).astype(np.float64)
        rotated = x @ Q.T
        return rotated.astype(x.dtype), {"matrix": Q, "method": "qr"}

    def inverse_qr_rotate(
        self, y: np.ndarray, metadata: dict
    ) -> np.ndarray:
        """Invert QR rotation."""
        y = np.asarray(y, dtype=np.float64)
        Q = metadata["matrix"].astype(np.float64)

        if y.ndim == 1:
            return (Q.T @ y).astype(y.dtype)
        return (y @ Q).astype(y.dtype)

    def butterfly_rotate(self, x: np.ndarray) -> Tuple[np.ndarray, dict]:
        """Apply butterfly matrix rotation.

        O(n log n) structured rotation that is faster than full QR
        for large dimensions.
        """
        x = np.asarray(x, dtype=np.float64)
        original_shape = x.shape

        if x.ndim == 1:
            n = len(x)
            B = self._butterfly_matrix(n)
            rotated = B @ x
            return rotated.astype(x.dtype), {"matrix": B, "method": "butterfly"}

        dim = x.shape[-1]
        B = self._butterfly_matrix(dim)
        rotated = x @ B.T
        return rotated.astype(x.dtype), {"matrix": B, "method": "butterfly"}

    def inverse_butterfly_rotate(
        self, y: np.ndarray, metadata: dict
    ) -> np.ndarray:
        """Invert butterfly rotation."""
        y = np.asarray(y, dtype=np.float64)
        B = metadata["matrix"].astype(np.float64)

        if y.ndim == 1:
            return (B.T @ y).astype(y.dtype)
        return (y @ B).astype(y.dtype)


# ═══════════════════════════════════════════════════════════════════════════
# 3. SpectralShaping — equalization, error diffusion
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SpectralProfile:
    """Spectral profile of a signal."""
    band_energies: np.ndarray
    spectral_centroid: float
    spectral_rolloff: float
    dominant_frequency: float


class SpectralShaping:
    """Spectral equalization and error diffusion for quantization.

    Pre-shapes the frequency content of weights to improve
    quantization quality:
    1. Spectral equalization: flattens the power spectrum
    2. Error diffusion: spreads quantization error to high frequencies
    """

    def __init__(self, config: Optional[SpectralShapingConfig] = None):
        self.config = config or SpectralShapingConfig()

    def compute_spectral_profile(self, x: np.ndarray) -> SpectralProfile:
        """Compute the spectral profile of a signal."""
        x = np.asarray(x, dtype=np.float64).ravel()
        n = len(x)

        coeffs = dct(x)
        power = coeffs ** 2
        total_power = np.sum(power) + 1e-30

        # Band energies
        band_size = max(1, n // self.config.n_bands)
        band_energies = np.zeros(self.config.n_bands)
        for b in range(self.config.n_bands):
            lo = b * band_size
            hi = min(lo + band_size, n)
            band_energies[b] = np.sum(power[lo:hi])
        band_energies /= total_power

        # Spectral centroid
        freqs = np.arange(n, dtype=np.float64)
        spectral_centroid = float(np.sum(freqs * power) / total_power)

        # Spectral rolloff
        cumulative = np.cumsum(power) / total_power
        rolloff_idx = int(np.searchsorted(cumulative, 0.85))
        spectral_rolloff = float(rolloff_idx)

        # Dominant frequency
        dominant_frequency = float(np.argmax(power))

        return SpectralProfile(
            band_energies=band_energies,
            spectral_centroid=spectral_centroid,
            spectral_rolloff=spectral_rolloff,
            dominant_frequency=dominant_frequency,
        )

    def equalize_spectrum(
        self, x: np.ndarray, strength: Optional[float] = None
    ) -> Tuple[np.ndarray, dict]:
        """Spectral equalization to flatten the power spectrum.

        Boosts attenuated frequencies and attenuates dominant ones,
        making the signal more uniform for quantization.

        Args:
            x: Input signal.
            strength: Equalization strength (0-1). Default from config.

        Returns:
            (equalized, metadata)
        """
        x = np.asarray(x, dtype=np.float64).ravel()
        strength = strength if strength is not None else self.config.equalization_strength

        coeffs = dct(x)
        power = np.abs(coeffs) + 1e-10

        # Target: flat spectrum at median power
        target_power = np.median(power)
        equalization = target_power / power
        equalization = np.clip(equalization, 0.1, 10.0)

        # Apply with strength control
        equalization = 1.0 + strength * (equalization - 1.0)
        equalized_coeffs = coeffs * equalization
        equalized = idct(equalized_coeffs)

        metadata = {
            "strength": strength,
            "power_range": float(np.max(power) / (np.min(power) + 1e-10)),
            "equalization_range": float(np.max(equalization) / np.min(equalization)),
        }

        return equalized.astype(x.dtype), metadata

    def error_diffuse(
        self, x: np.ndarray, quantized: np.ndarray
    ) -> np.ndarray:
        """Apply Floyd-Steinberg-style error diffusion in spectral domain.

        Spreads quantization error from low to high frequencies,
        preserving low-frequency content that matters most for quality.

        Args:
            x: Original signal.
            quantized: Quantized signal.

        Returns:
            Error-diffused signal.
        """
        x = np.asarray(x, dtype=np.float64).ravel()
        quantized = np.asarray(quantized, dtype=np.float64).ravel()

        error = x - quantized

        # Transfer error to DCT domain
        error_coeffs = dct(error)
        power = np.abs(error_coeffs) + 1e-10

        # Boost high-frequency error, suppress low-frequency error
        n = len(error_coeffs)
        freq_weight = np.linspace(0.1, 1.0, n)
        diffused_coeffs = error_coeffs * freq_weight * self.config.error_diffusion_weight

        # Reconstruct
        diffused_error = idct(diffused_coeffs)
        result = quantized + diffused_error

        return result.astype(x.dtype)

    def preprocess_for_quantization(
        self, x: np.ndarray
    ) -> Tuple[np.ndarray, dict]:
        """Full spectral preprocessing pipeline for quantization.

        1. Compute spectral profile
        2. Equalize spectrum
        3. Return shaped signal and metadata for post-processing

        Args:
            x: Input weights.

        Returns:
            (shaped, metadata) for use before quantization.
        """
        profile = self.compute_spectral_profile(x)
        equalized, eq_meta = self.equalize_spectrum(x)

        metadata = {
            "spectral_profile": profile,
            "equalization": eq_meta,
        }

        return equalized, metadata

    def postprocess_after_quantization(
        self,
        original: np.ndarray,
        quantized: np.ndarray,
        metadata: dict,
    ) -> np.ndarray:
        """Post-process after quantization using error diffusion.

        Args:
            original: Original weights.
            quantized: Quantized weights.
            metadata: Metadata from preprocess_for_quantization.

        Returns:
            Error-diffused result.
        """
        return self.error_diffuse(original, quantized)
