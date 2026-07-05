"""
SIMD-Optimized Backend for SpectralStream
==========================================
CPU-optimized operations with vectorized DCT, FWHT, quantization,
cache-aware block processing, and CPU feature detection.

Uses numpy vectorization for portable SIMD-like performance,
with optional numba JIT acceleration when available.
"""

from __future__ import annotations

import logging
import os
import struct
import sys
import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from spectralstream.core.math_primitives.transforms import (
    dct,
    idct,
    fwht,
    ifwht,
    zigzag_indices,
)
from spectralstream.core.math_primitives.quantization import LloydMaxQuantizer

# BAND_COMPRESSION is already exported from spectralstream.core.math_primitives.kernels
from spectralstream.core.math_primitives import BAND_COMPRESSION

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# CPU Feature Detection
# ═══════════════════════════════════════════════════════════════════════════


class CPUFeatures:
    """Detect available CPU SIMD features on x86/ARM."""

    def __init__(self) -> None:
        self.sse4: bool = False
        self.avx2: bool = False
        self.avx512: bool = False
        self.neon: bool = False
        self.amx: bool = False
        self.n_cores: int = os.cpu_count() or 1
        self.l1_cache_kb: int = 32
        self.l2_cache_kb: int = 256
        self.l3_cache_kb: int = 8192
        self._detect()

    def _detect(self) -> None:
        """Detect CPU features via /proc/cpuinfo on Linux, fallback to platform."""
        try:
            if sys.platform == "linux" and os.path.exists("/proc/cpuinfo"):
                self._detect_linux()
            elif sys.platform == "darwin":
                self._detect_macos()
            else:
                self._detect_fallback()
        except Exception as e:
            logger.warning("CPU feature detection failed: %s", e)
            self._detect_fallback()

        # Try to read cache sizes from sysfs
        self._detect_cache_sizes()

        logger.info(
            "CPU features: SSE4=%s, AVX2=%s, AVX-512=%s, NEON=%s, cores=%d",
            self.sse4,
            self.avx2,
            self.avx512,
            self.neon,
            self.n_cores,
        )

    def _detect_linux(self) -> None:
        """Detect via /proc/cpuinfo flags."""
        try:
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if line.strip().startswith("flags"):
                        flags = line.split(":")[1].strip().split()
                        self.sse4 = "sse4_1" in flags or "sse4_2" in flags
                        self.avx2 = "avx2" in flags
                        self.avx512f = "avx512f" in flags
                        self.avx512 = any(f.startswith("avx512") for f in flags)
                        break
        except (OSError, IndexError):
            pass

    def _detect_macos(self) -> None:
        """Detect on macOS via sysctl."""
        try:
            import subprocess

            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.features"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            features = result.stdout.upper()
            self.sse4 = "SSE4" in features
            self.avx2 = "AVX2" in features
            self.avx512 = "AVX512" in features
        except Exception:
            self._detect_fallback()

    def _detect_fallback(self) -> None:
        """Assume basic capabilities if detection fails."""
        self.sse4 = True
        self.avx2 = True

    def _detect_cache_sizes(self) -> None:
        """Read cache sizes from sysfs on Linux."""
        if sys.platform != "linux":
            return
        cache_paths = [
            ("/sys/devices/system/cpu/cpu0/cache/index0/size", "l1_cache_kb"),
            ("/sys/devices/system/cpu/cpu0/cache/index2/size", "l2_cache_kb"),
            ("/sys/devices/system/cpu/cpu0/cache/index3/size", "l3_cache_kb"),
        ]
        for path, attr in cache_paths:
            try:
                with open(path, "r") as f:
                    val = f.read().strip()
                    kb = int(val.replace("K", "").replace("k", ""))
                    setattr(self, attr, kb)
            except (OSError, ValueError):
                pass

    @property
    def has_avx(self) -> bool:
        """Whether AVX2 or better is available."""
        return self.avx2 or self.avx512

    @property
    def simd_width(self) -> int:
        """Effective SIMD width in elements (float32)."""
        if self.avx512:
            return 16
        elif self.avx2:
            return 8
        elif self.sse4:
            return 4
        return 1

    def __repr__(self) -> str:
        return (
            f"CPUFeatures(cores={self.n_cores}, "
            f"sse4={self.sse4}, avx2={self.avx2}, avx512={self.avx512})"
        )


# Global instance
_cpu_features: Optional[CPUFeatures] = None


def get_cpu_features() -> CPUFeatures:
    """Get or create the global CPU feature detector."""
    global _cpu_features
    if _cpu_features is None:
        _cpu_features = CPUFeatures()
    return _cpu_features


# ═══════════════════════════════════════════════════════════════════════════
# Cache-Aware Block Operations
# ═══════════════════════════════════════════════════════════════════════════


class CacheAwareBlocker:
    """Compute optimal block sizes for cache-resident operations.

    Determines tile sizes that fit in L1/L2 cache to minimize
    cache misses during matrix operations.
    """

    def __init__(
        self,
        element_size: int = 4,
        l1_cache_bytes: int = 32 * 1024,
        l2_cache_bytes: int = 256 * 1024,
    ) -> None:
        self.element_size = element_size
        self.l1_cache_bytes = l1_cache_bytes
        self.l2_cache_bytes = l2_cache_bytes

    def l1_block_size(self, ndim: int = 2) -> int:
        """Compute block size that fits in L1 cache."""
        if ndim == 1:
            return self.l1_cache_bytes // self.element_size
        # For 2D: sqrt(L1 / element_size) per side
        per_side = int(np.sqrt(self.l1_cache_bytes / self.element_size))
        return max(1, per_side)

    def l2_block_size(self, ndim: int = 2) -> int:
        """Compute block size that fits in L2 cache."""
        if ndim == 1:
            return self.l2_cache_bytes // self.element_size
        per_side = int(np.sqrt(self.l2_cache_bytes / self.element_size))
        return max(1, per_side)

    def compute_tiling(
        self,
        rows: int,
        cols: int,
        element_size: int = 4,
    ) -> Tuple[int, int]:
        """Compute optimal tile dimensions for a matrix operation.

        Returns (tile_rows, tile_cols) that maximize cache utilization.
        """
        block = self.l1_block_size(2)
        tile_rows = min(block, rows)
        tile_cols = min(block, cols)
        # Ensure tile fits in L1
        while tile_rows * tile_cols * element_size > self.l1_cache_bytes:
            if tile_rows > tile_cols:
                tile_rows = max(1, tile_rows // 2)
            else:
                tile_cols = max(1, tile_cols // 2)
        return tile_rows, tile_cols


# ═══════════════════════════════════════════════════════════════════════════
# SIMD Backend
# ═══════════════════════════════════════════════════════════════════════════


class SIMDBackend:
    """CPU-optimized backend with vectorized spectral operations.

    Provides SIMD-accelerated (via numpy) implementations of:
        - DCT/IDCT (Type-II)
        - FWHT/IFWHT
        - Lloyd-Max quantization
        - Block-tiled matrix operations
        - Cache-aware memory access patterns

    All operations fall back to pure-numpy when hardware SIMD is unavailable.
    """

    def __init__(
        self,
        use_cache_blocking: bool = True,
        default_quantize_bits: int = 4,
    ) -> None:
        self.features = get_cpu_features()
        self.use_cache_blocking = use_cache_blocking
        self.default_quantize_bits = default_quantize_bits

        self._blocker = CacheAwareBlocker(
            element_size=4,
            l1_cache_bytes=self.features.l1_cache_kb * 1024,
            l2_cache_bytes=self.features.l2_cache_kb * 1024,
        )

        self._quantizers: Dict[int, LloydMaxQuantizer] = {}

        # Performance counters
        self._op_counts: Dict[str, int] = {}
        self._op_times: Dict[str, float] = {}

        logger.info(
            "SIMDBackend initialized: SIMD_width=%d, cache_blocking=%s",
            self.features.simd_width,
            use_cache_blocking,
        )

    # ── DCT Operations ────────────────────────────────────────────────────

    def vectorized_dct(
        self,
        signal: np.ndarray,
        axis: int = -1,
    ) -> np.ndarray:
        """Vectorized DCT-II using numpy FFT-based approach.

        Processes multiple signals in parallel when input is batched.

        Args:
            signal: Input signal(s). 1D or 2D (batch x length).
            axis: Axis along which to apply DCT.

        Returns:
            DCT coefficients, same shape as input.
        """
        self._count_op("dct")
        signal = np.asarray(signal, dtype=np.float64)

        if signal.ndim == 1:
            return dct(signal).astype(np.float32)

        # Batched: apply DCT along specified axis
        result = np.empty_like(signal, dtype=np.float32)
        for i in range(signal.shape[0]):
            result[i] = dct(signal[i]).astype(np.float32)
        return result

    def vectorized_idct(
        self,
        coeffs: np.ndarray,
        axis: int = -1,
    ) -> np.ndarray:
        """Vectorized inverse DCT-II."""
        self._count_op("idct")
        coeffs = np.asarray(coeffs, dtype=np.float64)

        if coeffs.ndim == 1:
            return idct(coeffs).astype(np.float32)

        result = np.empty_like(coeffs, dtype=np.float32)
        for i in range(coeffs.shape[0]):
            result[i] = idct(coeffs[i]).astype(np.float32)
        return result

    # ── FWHT Operations ───────────────────────────────────────────────────

    def vectorized_fwht(
        self,
        signal: np.ndarray,
        normalize: bool = False,
    ) -> np.ndarray:
        """Vectorized Fast Walsh-Hadamard Transform.

        Processes batched inputs with numpy vectorization.

        Args:
            signal: Input signal(s). Length must be power of 2.
            normalize: Whether to apply 1/sqrt(n) normalization.

        Returns:
            Walsh-Hadamard coefficients.
        """
        self._count_op("fwht")
        signal = np.asarray(signal, dtype=np.float32)

        if signal.ndim == 1:
            return fwht(signal, normalize=normalize)

        result = np.empty_like(signal, dtype=np.float32)
        for i in range(signal.shape[0]):
            result[i] = fwht(signal[i], normalize=normalize)
        return result

    def vectorized_ifwht(
        self,
        coeffs: np.ndarray,
        normalize: bool = True,
    ) -> np.ndarray:
        """Vectorized inverse FWHT (self-inverse)."""
        self._count_op("ifwht")
        return self.vectorized_fwht(coeffs, normalize=normalize)

    # ── Quantization Operations ───────────────────────────────────────────

    def vectorized_quantize(
        self,
        data: np.ndarray,
        n_bits: Optional[int] = None,
        train_on_first: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Vectorized Lloyd-Max scalar quantization.

        Args:
            data: Input data to quantize.
            n_bits: Bits per element (default: self.default_quantize_bits).
            train_on_first: Train quantizer on first call if not trained.

        Returns:
            Tuple of (quantized_indices, centroids).
        """
        self._count_op("quantize")
        n_bits = n_bits or self.default_quantize_bits
        quantizer = self._get_quantizer(n_bits)

        data = np.asarray(data)
        flat = data.ravel().astype(np.float64)

        if not quantizer.trained and train_on_first:
            quantizer.train(flat)

        return quantizer.compress(flat)

    def vectorized_dequantize(
        self,
        indices: np.ndarray,
        centroids: np.ndarray,
        shape: Tuple[int, ...],
    ) -> np.ndarray:
        """Vectorized dequantization from indices and centroids."""
        self._count_op("dequantize")
        flat = centroids[indices.ravel()]
        return flat.reshape(shape)

    # ── Block-Tiled Matrix Operations ─────────────────────────────────────

    def tiled_matmul(
        self,
        a: np.ndarray,
        b: np.ndarray,
        tile_size: Optional[int] = None,
    ) -> np.ndarray:
        """Cache-aware tiled matrix multiplication.

        Divides the multiplication into tiles that fit in L1 cache
        to minimize cache misses.

        Args:
            a: Left matrix (m x k).
            b: Right matrix (k x n).
            tile_size: Tile size (auto-computed if None).

        Returns:
            Result matrix (m x n).
        """
        self._count_op("tiled_matmul")
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)

        if tile_size is None:
            tile_size = self._blocker.l1_block_size(2)

        m, k = a.shape
        k2, n = b.shape
        if k != k2:
            raise ValueError(f"Shape mismatch: a={a.shape}, b={b.shape}")

        result = np.zeros((m, n), dtype=np.float64)

        for i in range(0, m, tile_size):
            for j in range(0, n, tile_size):
                for l in range(0, k, tile_size):
                    i_end = min(i + tile_size, m)
                    j_end = min(j + tile_size, n)
                    l_end = min(l + tile_size, k)
                    result[i:i_end, j:j_end] += (
                        a[i:i_end, l:l_end] @ b[l:l_end, j:j_end]
                    )
        return result

    def tiled_dct_2d(
        self,
        matrix: np.ndarray,
        tile_size: Optional[int] = None,
    ) -> np.ndarray:
        """Cache-aware 2D DCT via block processing.

        Applies DCT to each tile independently, reducing peak memory.

        Args:
            matrix: 2D input matrix.
            tile_size: Tile size (auto-computed if None).

        Returns:
            2D DCT coefficients.
        """
        self._count_op("tiled_dct_2d")
        matrix = np.asarray(matrix, dtype=np.float64)
        rows, cols = matrix.shape

        if tile_size is None:
            tile_size = self._blocker.l1_block_size(2)

        result = np.zeros_like(matrix, dtype=np.float32)

        for i in range(0, rows, tile_size):
            for j in range(0, cols, tile_size):
                i_end = min(i + tile_size, rows)
                j_end = min(j + tile_size, cols)
                tile = matrix[i:i_end, j:j_end]
                result[i:i_end, j:j_end] = dct(dct(tile, axis=0), axis=1).astype(
                    np.float32
                )

        return result

    def tiled_idct_2d(
        self,
        coeffs: np.ndarray,
        tile_size: Optional[int] = None,
    ) -> np.ndarray:
        """Cache-aware 2D inverse DCT via block processing."""
        self._count_op("tiled_idct_2d")
        coeffs = np.asarray(coeffs, dtype=np.float64)
        rows, cols = coeffs.shape

        if tile_size is None:
            tile_size = self._blocker.l1_block_size(2)

        result = np.zeros_like(coeffs, dtype=np.float32)

        for i in range(0, rows, tile_size):
            for j in range(0, cols, tile_size):
                i_end = min(i + tile_size, rows)
                j_end = min(j + tile_size, cols)
                tile = coeffs[i:i_end, j:j_end]
                result[i:i_end, j:j_end] = idct(idct(tile, axis=0), axis=1).astype(
                    np.float32
                )

        return result

    # ── Zigzag Scan ───────────────────────────────────────────────────────

    def zigzag_scan(self, matrix: np.ndarray) -> np.ndarray:
        """Apply JPEG-style zigzag scan to 2D DCT coefficients.

        Args:
            matrix: 2D coefficient matrix (must be square).

        Returns:
            1D array in zigzag order.
        """
        n = matrix.shape[0]
        order = zigzag_indices(n)
        return matrix.ravel()[order.ravel()]

    def zigzag_unscan(self, flat: np.ndarray, n: int) -> np.ndarray:
        """Reverse zigzag scan to reconstruct 2D matrix.

        Args:
            flat: 1D zigzag-ordered array.
            n: Output matrix size (n x n).

        Returns:
            2D matrix.
        """
        order = zigzag_indices(n)
        matrix = np.zeros(n * n, dtype=flat.dtype)
        matrix[order.ravel()] = flat
        return matrix.reshape(n, n)

    # ── SIMD-Accelerated Kernels ──────────────────────────────────────────

    def simd_softmax(
        self,
        x: np.ndarray,
        axis: int = -1,
        temperature: float = 1.0,
    ) -> np.ndarray:
        """Numerically stable softmax with numpy vectorization.

        Uses the log-sum-exp trick for numerical stability.
        """
        self._count_op("softmax")
        x = np.asarray(x, dtype=np.float64)
        scaled = x / max(temperature, 1e-10)
        m = scaled.max(axis=axis, keepdims=True)
        e = np.exp(scaled - m)
        return (e / (e.sum(axis=axis, keepdims=True) + 1e-30)).astype(np.float32)

    def simd_layer_norm(
        self,
        x: np.ndarray,
        weight: Optional[np.ndarray] = None,
        bias: Optional[np.ndarray] = None,
        eps: float = 1e-5,
    ) -> np.ndarray:
        """Vectorized layer normalization."""
        self._count_op("layer_norm")
        x = np.asarray(x, dtype=np.float64)
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        normalized = (x - mean) / np.sqrt(var + eps)

        if weight is not None:
            normalized = normalized * weight
        if bias is not None:
            normalized = normalized + bias

        return normalized.astype(np.float32)

    def simd_rms_norm(
        self,
        x: np.ndarray,
        weight: Optional[np.ndarray] = None,
        eps: float = 1e-6,
    ) -> np.ndarray:
        """Vectorized RMS normalization."""
        self._count_op("rms_norm")
        x = np.asarray(x, dtype=np.float64)
        rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps)
        normalized = (x / rms) * np.sqrt(x.shape[-1])

        if weight is not None:
            normalized = normalized * weight

        return normalized.astype(np.float32)

    def simd_gelu(self, x: np.ndarray) -> np.ndarray:
        """Vectorized GELU activation (tanh approximation)."""
        self._count_op("gelu")
        x = np.asarray(x, dtype=np.float64)
        return (
            0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))
        ).astype(np.float32)

    def simd_silu(self, x: np.ndarray) -> np.ndarray:
        """Vectorized SiLU (Swish) activation."""
        self._count_op("silu")
        x = np.asarray(x, dtype=np.float64)
        return (x / (1.0 + np.exp(-x))).astype(np.float32)

    def simd_matmul(
        self,
        a: np.ndarray,
        b: np.ndarray,
    ) -> np.ndarray:
        """Vectorized matrix multiplication (delegates to numpy BLAS)."""
        self._count_op("matmul")
        return (
            np.asarray(a, dtype=np.float32) @ np.asarray(b, dtype=np.float32)
        ).astype(np.float32)

    # ── Performance Tracking ──────────────────────────────────────────────

    def _count_op(self, op_name: str) -> None:
        """Increment operation counter."""
        self._op_counts[op_name] = self._op_counts.get(op_name, 0) + 1

    def get_op_stats(self) -> Dict[str, int]:
        """Get operation counts since last reset."""
        return dict(self._op_counts)

    def reset_stats(self) -> None:
        """Reset operation counters."""
        self._op_counts.clear()
        self._op_times.clear()

    # ── Internal Helpers ──────────────────────────────────────────────────

    def _get_quantizer(self, n_bits: int) -> LloydMaxQuantizer:
        """Get or create quantizer for given bit width."""
        if n_bits not in self._quantizers:
            self._quantizers[n_bits] = LloydMaxQuantizer(n_bits=n_bits)
        return self._quantizers[n_bits]
