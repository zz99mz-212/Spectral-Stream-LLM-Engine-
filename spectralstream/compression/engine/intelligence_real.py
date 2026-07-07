"""
Unified Intelligence Engine — Master Coordinator
=================================================
Coordinates all compression and inference subsystems with per-tensor
strategy selection, cross-layer prediction, and structured logging.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spectralstream.core.math_primitives import (
    LloydMaxQuantizer,
    HadamardRotator,
    dct,
    idct,
    fwht,
    ifwht,
    spectral_entropy,
    cosine_similarity,
    softmax,
    BAND_HIGH,
    BAND_NORMAL,
    BAND_LOW,
    BAND_COMPRESSION,
)

logger = logging.getLogger(__name__)


class CompressionStrategy:
    SPECTRAL_DCT = "spectral_dct"
    HADAMARD_QUANTIZE = "hadamard_quantize"
    LOW_RANK = "low_rank"
    PRODUCT_QUANTIZE = "product_quantize"
    SPARSIFY = "sparsify"
    WAVELET = "wavelet"
    IDENTITY = "identity"


class CompressionResult:
    __slots__ = (
        "compressed_data",
        "original_shape",
        "strategy",
        "compression_ratio",
        "reconstruction_error",
        "metadata",
    )

    def __init__(
        self,
        compressed_data: Any,
        original_shape: Tuple[int, ...],
        strategy: str,
        compression_ratio: float,
        reconstruction_error: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.compressed_data = compressed_data
        self.original_shape = original_shape
        self.strategy = strategy
        self.compression_ratio = compression_ratio
        self.reconstruction_error = reconstruction_error
        self.metadata = metadata or {}

    def __repr__(self) -> str:
        return (
            f"CompressionResult(strategy={self.strategy!r}, "
            f"ratio={self.compression_ratio:.2f}x, "
            f"error={self.reconstruction_error:.6f})"
        )


class UnifiedIntelligenceEngine:
    def __init__(
        self,
        default_strategy: str = CompressionStrategy.SPECTRAL_DCT,
        target_ratio: float = 4.0,
        error_tolerance: float = 0.01,
        max_bits_per_element: int = 8,
        enable_cross_layer: bool = True,
        n_quantize_bits: int = 4,
    ) -> None:
        self.default_strategy = default_strategy
        self.target_ratio = target_ratio
        self.error_tolerance = error_tolerance
        self.max_bits_per_element = max_bits_per_element
        self.enable_cross_layer = enable_cross_layer
        self.n_quantize_bits = n_quantize_bits
        self._prev_decompressed: Dict[int, np.ndarray] = {}
        self._layer_stats: Dict[int, Dict[str, float]] = {}
        self._compression_history: List[CompressionResult] = []
        self._strategy_errors: Dict[str, List[float]] = {}
        self._strategy_ratios: Dict[str, List[float]] = {}
        self._quantizers: Dict[int, LloydMaxQuantizer] = {}
        self._hadamard_rotators: Dict[int, HadamardRotator] = {}

    def compress_tensor(
        self,
        tensor: np.ndarray,
        layer_id: int = 0,
        strategy: Optional[str] = None,
        **kwargs: Any,
    ) -> CompressionResult:
        tensor = np.asarray(tensor)
        start = time.perf_counter()
        if strategy is None:
            strategy = self._select_strategy(tensor, layer_id)
        if strategy == CompressionStrategy.SPECTRAL_DCT:
            result = self._compress_spectral_dct(tensor, layer_id, **kwargs)
        elif strategy == CompressionStrategy.HADAMARD_QUANTIZE:
            result = self._compress_hadamard_quantize(tensor, layer_id, **kwargs)
        elif strategy == CompressionStrategy.LOW_RANK:
            result = self._compress_low_rank(tensor, layer_id, **kwargs)
        elif strategy == CompressionStrategy.PRODUCT_QUANTIZE:
            result = self._compress_product_quantize(tensor, layer_id, **kwargs)
        elif strategy == CompressionStrategy.SPARSIFY:
            result = self._compress_sparsify(tensor, layer_id, **kwargs)
        elif strategy == CompressionStrategy.WAVELET:
            result = self._compress_wavelet(tensor, layer_id, **kwargs)
        else:
            result = CompressionResult(
                tensor.copy(), tensor.shape, CompressionStrategy.IDENTITY, 1.0, 0.0
            )
        elapsed = time.perf_counter() - start
        result.metadata["compress_time_ms"] = elapsed * 1000.0
        if self.enable_cross_layer:
            decompressed = self.decompress_tensor(result)
            self._prev_decompressed[layer_id] = decompressed
        self._strategy_errors.setdefault(strategy, []).append(
            result.reconstruction_error
        )
        self._strategy_ratios.setdefault(strategy, []).append(result.compression_ratio)
        self._compression_history.append(result)
        return result

    def decompress_tensor(self, result: CompressionResult) -> np.ndarray:
        strategy = result.strategy
        data = result.compressed_data
        if strategy == CompressionStrategy.IDENTITY:
            return np.asarray(data).copy()
        elif strategy == CompressionStrategy.SPECTRAL_DCT:
            return self._decompress_spectral_dct(data, result.original_shape)
        elif strategy == CompressionStrategy.HADAMARD_QUANTIZE:
            return self._decompress_hadamard_quantize(data, result.original_shape)
        elif strategy == CompressionStrategy.LOW_RANK:
            return self._decompress_low_rank(data, result.original_shape)
        elif strategy == CompressionStrategy.PRODUCT_QUANTIZE:
            return self._decompress_product_quantize(data, result.original_shape)
        elif strategy == CompressionStrategy.SPARSIFY:
            return self._decompress_sparsify(data, result.original_shape)
        elif strategy == CompressionStrategy.WAVELET:
            return self._decompress_wavelet(data, result.original_shape)
        else:
            return np.zeros(result.original_shape, dtype=np.float64)

    def get_previous_decompressed(self, layer_id: int) -> Optional[np.ndarray]:
        return self._prev_decompressed.get(layer_id)

    def predict_cross_layer_residual(
        self, tensor: np.ndarray, ref_layer_id: int
    ) -> np.ndarray:
        ref = self._prev_decompressed.get(ref_layer_id)
        if ref is None:
            return np.zeros_like(tensor)
        ref_flat = ref.ravel()[: tensor.size]
        target_flat = tensor.ravel()
        if len(ref_flat) != len(target_flat):
            if len(ref_flat) > len(target_flat):
                ref_flat = ref_flat[: len(target_flat)]
            else:
                ref_flat = np.pad(ref_flat, (0, len(target_flat) - len(ref_flat)))
        ref_norm = ref_flat / (np.linalg.norm(ref_flat) + 1e-10)
        alpha = float(np.dot(target_flat, ref_norm))
        beta = float(np.mean(target_flat)) - alpha * float(np.mean(ref_norm))
        predicted = alpha * ref_flat + beta
        return predicted.reshape(tensor.shape)

    def get_strategy_stats(self) -> Dict[str, Dict[str, float]]:
        stats: Dict[str, Dict[str, float]] = {}
        for s in self._strategy_errors:
            errors = self._strategy_errors[s]
            ratios = self._strategy_ratios.get(s, [])
            stats[s] = {
                "count": float(len(errors)),
                "mean_error": float(np.mean(errors)) if errors else 0.0,
                "max_error": float(np.max(errors)) if errors else 0.0,
                "mean_ratio": float(np.mean(ratios)) if ratios else 0.0,
            }
        return stats

    def _select_strategy(self, tensor: np.ndarray, layer_id: int) -> str:
        flat = tensor.ravel().astype(np.float64)
        n_elements = flat.size
        sparsity = float(np.mean(np.abs(flat) < 1e-6))
        if sparsity > 0.80:
            return CompressionStrategy.SPARSIFY
        concentration = 1.0
        if n_elements >= 4:
            spec = np.abs(dct(flat[: min(n_elements, 4096)]))
            total_energy = float(np.sum(spec**2))
            if total_energy > 1e-10:
                sorted_energy = np.sort(spec.ravel() ** 2)[::-1]
                cumulative = np.cumsum(sorted_energy) / total_energy
                k_90 = int(np.searchsorted(cumulative, 0.90)) + 1
                concentration = k_90 / max(n_elements, 1)
        if tensor.ndim == 2 and tensor.shape[0] > 1 and tensor.shape[1] > 1:
            try:
                s = np.linalg.svd(tensor, compute_uv=False)
                total_sv = float(np.sum(s**2))
                if total_sv > 1e-10:
                    cum_sv = np.cumsum(s**2) / total_sv
                    rank_90 = int(np.searchsorted(cum_sv, 0.90)) + 1
                    rank_ratio = rank_90 / min(tensor.shape)
                    if rank_ratio < 0.3:
                        return CompressionStrategy.LOW_RANK
            except np.linalg.LinAlgError:
                pass
        if concentration < 0.3:
            return CompressionStrategy.SPECTRAL_DCT
        return CompressionStrategy.HADAMARD_QUANTIZE

    def _compress_spectral_dct(
        self, tensor: np.ndarray, layer_id: int, **kwargs: Any
    ) -> CompressionResult:
        flat = tensor.ravel().astype(np.float64)
        n = flat.size
        n_padded = 1 << (n - 1).bit_length() if n > 1 else 1
        flat_padded = np.pad(flat, (0, n_padded - n)) if n_padded != n else flat
        coeffs = dct(flat_padded)
        sorted_sq = np.sort(np.abs(coeffs) ** 2)[::-1]
        total_energy = float(np.sum(sorted_sq))
        if total_energy > 1e-10:
            cumulative = np.cumsum(sorted_sq) / total_energy
            k_keep = max(1, min(int(np.searchsorted(cumulative, 0.95)) + 1, n_padded))
        else:
            k_keep = n_padded
        quantizer = self._get_quantizer(self.n_quantize_bits)
        kept_coeffs = coeffs[:k_keep]
        if not quantizer.trained:
            quantizer.train(kept_coeffs)
        quantized_indices, centroids = quantizer.compress(kept_coeffs)
        reconstructed = quantizer.centroids[quantized_indices]
        error_coeffs = np.zeros(n_padded, dtype=np.float64)
        error_coeffs[:k_keep] = reconstructed
        reconstructed_signal = idct(error_coeffs)
        mse = float(np.mean((flat_padded - reconstructed_signal) ** 2))
        ratio = tensor.nbytes / max(len(quantized_indices) + len(centroids) * 8, 1)
        return CompressionResult(
            compressed_data={
                "indices": quantized_indices,
                "centroids": centroids,
                "k_keep": k_keep,
                "n_padded": n_padded,
                "n_original": n,
                "quantize_bits": self.n_quantize_bits,
            },
            original_shape=tensor.shape,
            strategy=CompressionStrategy.SPECTRAL_DCT,
            compression_ratio=ratio,
            reconstruction_error=mse,
            metadata={"k_keep": k_keep, "n_padded": n_padded},
        )

    def _decompress_spectral_dct(
        self, data: Dict[str, Any], shape: Tuple[int, ...]
    ) -> np.ndarray:
        quantizer = self._get_quantizer(data["quantize_bits"])
        quantizer.centroids = data["centroids"]
        quantizer.trained = True
        coeffs = np.zeros(data["n_padded"], dtype=np.float64)
        coeffs[: data["k_keep"]] = data["centroids"][data["indices"]]
        return idct(coeffs)[: data["n_original"]].reshape(shape)

    def _compress_hadamard_quantize(
        self, tensor: np.ndarray, layer_id: int, **kwargs: Any
    ) -> CompressionResult:
        flat = tensor.ravel().astype(np.float32)
        n = flat.size
        n_rotated = 1 << (n - 1).bit_length() if n > 1 else 1
        rotator = self._get_hadamard_rotator(n)
        padded = np.zeros(n_rotated, dtype=np.float32)
        padded[:n] = flat
        rotated = rotator.rotate(padded.reshape(1, -1)).ravel()
        quantizer = self._get_quantizer(self.n_quantize_bits)
        if not quantizer.trained:
            quantizer.train(rotated)
        indices, centroids = quantizer.compress(rotated)
        dequantized = centroids[indices]
        inv_rotated = rotator.inverse_rotate(dequantized.reshape(1, -1)).ravel()
        mse = float(np.mean((padded - inv_rotated) ** 2))
        ratio = tensor.nbytes / max(len(indices) + len(centroids) * 8, 1)
        return CompressionResult(
            compressed_data={
                "indices": indices,
                "centroids": centroids,
                "n_original": n,
                "n_rotated": n_rotated,
                "quantize_bits": self.n_quantize_bits,
            },
            original_shape=tensor.shape,
            strategy=CompressionStrategy.HADAMARD_QUANTIZE,
            compression_ratio=ratio,
            reconstruction_error=mse,
        )

    def _decompress_hadamard_quantize(
        self, data: Dict[str, Any], shape: Tuple[int, ...]
    ) -> np.ndarray:
        rotator = self._get_hadamard_rotator(data["n_original"])
        dequantized = data["centroids"][data["indices"]]
        inv_rotated = rotator.inverse_rotate(dequantized.reshape(1, -1)).ravel()
        return inv_rotated[: data["n_original"]].reshape(shape)

    def _compress_low_rank(
        self,
        tensor: np.ndarray,
        layer_id: int,
        keep_energy: float = 0.90,
        **kwargs: Any,
    ) -> CompressionResult:
        if tensor.ndim < 2:
            return self._compress_spectral_dct(tensor, layer_id, **kwargs)
        U, s, Vt = np.linalg.svd(tensor, full_matrices=False)
        total_energy = float(np.sum(s**2))
        if total_energy > 1e-10:
            cum = np.cumsum(s**2) / total_energy
            rank = max(1, min(int(np.searchsorted(cum, keep_energy)) + 1, len(s)))
        else:
            rank = 1
        U_k, s_k, Vt_k = U[:, :rank], s[:rank], Vt[:rank, :]
        reconstructed = U_k @ np.diag(s_k) @ Vt_k
        mse = float(np.mean((tensor - reconstructed) ** 2))
        compressed_bytes = U_k.nbytes + s_k.nbytes + Vt_k.nbytes
        ratio = tensor.nbytes / max(compressed_bytes, 1)
        return CompressionResult(
            compressed_data={"U": U_k, "s": s_k, "Vt": Vt_k, "rank": rank},
            original_shape=tensor.shape,
            strategy=CompressionStrategy.LOW_RANK,
            compression_ratio=ratio,
            reconstruction_error=mse,
            metadata={"rank": rank, "keep_energy": keep_energy},
        )

    def _decompress_low_rank(
        self, data: Dict[str, Any], shape: Tuple[int, ...]
    ) -> np.ndarray:
        return (data["U"] @ np.diag(data["s"]) @ data["Vt"]).reshape(shape)

    def _compress_product_quantize(
        self,
        tensor: np.ndarray,
        layer_id: int,
        n_subspaces: int = 8,
        n_centroids: int = 16,
        **kwargs: Any,
    ) -> CompressionResult:
        flat = tensor.ravel().astype(np.float64)
        n = flat.size
        sub_dim = max(1, n // n_subspaces)
        n_padded = sub_dim * n_subspaces
        padded = np.pad(flat, (0, n_padded - n)) if n_padded > n else flat[:n_padded]
        subspaces = padded.reshape(n_subspaces, sub_dim)
        codebook = np.zeros((n_subspaces, n_centroids), dtype=np.float64)
        codes = np.zeros(n_subspaces, dtype=np.int32)
        for i in range(n_subspaces):
            sub = subspaces[i]
            mu, sigma = float(np.mean(sub)), max(float(np.std(sub)), 1e-10)
            scale = max(abs(mu - 4 * sigma), abs(mu + 4 * sigma), 1e-8)
            normalized = np.clip(sub / scale, -1.0, 1.0)
            centroids = np.linspace(-1.0, 1.0, n_centroids)
            for _ in range(10):
                boundaries = (centroids[1:] + centroids[:-1]) * 0.5
                idx = np.clip(np.digitize(normalized, boundaries), 0, n_centroids - 1)
                for c in range(n_centroids):
                    mask = idx == c
                    if np.any(mask):
                        centroids[c] = np.mean(normalized[mask])
            codebook[i] = centroids * scale
            idx = np.clip(np.digitize(normalized, boundaries), 0, n_centroids - 1)
            codes[i] = int(np.bincount(idx).argmax())
        reconstructed = np.zeros(n_padded, dtype=np.float64)
        for i in range(n_subspaces):
            reconstructed[i * sub_dim : (i + 1) * sub_dim] = codebook[i, codes[i]]
        mse = float(np.mean((padded - reconstructed) ** 2))
        ratio = tensor.nbytes / max(codes.nbytes + codebook.nbytes, 1)
        return CompressionResult(
            compressed_data={
                "codes": codes,
                "codebook": codebook,
                "n_subspaces": n_subspaces,
                "n_centroids": n_centroids,
                "sub_dim": sub_dim,
                "n_original": n,
            },
            original_shape=tensor.shape,
            strategy=CompressionStrategy.PRODUCT_QUANTIZE,
            compression_ratio=ratio,
            reconstruction_error=mse,
        )

    def _decompress_product_quantize(
        self, data: Dict[str, Any], shape: Tuple[int, ...]
    ) -> np.ndarray:
        codes, codebook = data["codes"], data["codebook"]
        n_subspaces, sub_dim = data["n_subspaces"], data["sub_dim"]
        n_original = data["n_original"]
        reconstructed = np.zeros(n_subspaces * sub_dim, dtype=np.float64)
        for i in range(n_subspaces):
            reconstructed[i * sub_dim : (i + 1) * sub_dim] = codebook[i, codes[i]]
        return reconstructed[:n_original].reshape(shape)

    def _compress_sparsify(
        self,
        tensor: np.ndarray,
        layer_id: int,
        sparsity_target: float = 0.90,
        **kwargs: Any,
    ) -> CompressionResult:
        flat = tensor.ravel().astype(np.float64)
        threshold = float(np.percentile(np.abs(flat), sparsity_target * 100))
        mask = np.abs(flat) >= threshold
        nonzero_values = flat[mask]
        nonzero_indices = np.where(mask)[0]
        reconstructed_flat = np.zeros_like(flat)
        reconstructed_flat[mask] = nonzero_values
        mse = float(np.mean((flat - reconstructed_flat) ** 2))
        ratio = tensor.nbytes / max(nonzero_values.nbytes + nonzero_indices.nbytes, 1)
        return CompressionResult(
            compressed_data={
                "indices": nonzero_indices,
                "values": nonzero_values,
                "threshold": threshold,
            },
            original_shape=tensor.shape,
            strategy=CompressionStrategy.SPARSIFY,
            compression_ratio=ratio,
            reconstruction_error=mse,
            metadata={
                "sparsity_achieved": 1.0 - len(nonzero_values) / max(flat.size, 1)
            },
        )

    def _decompress_sparsify(
        self, data: Dict[str, Any], shape: Tuple[int, ...]
    ) -> np.ndarray:
        flat = np.zeros(int(np.prod(shape)), dtype=np.float64)
        flat[data["indices"]] = data["values"]
        return flat.reshape(shape)

    def _compress_wavelet(
        self,
        tensor: np.ndarray,
        layer_id: int,
        keep_fraction: float = 0.25,
        **kwargs: Any,
    ) -> CompressionResult:
        flat = tensor.ravel().astype(np.float64)
        n = flat.size
        coeffs = self._haar_forward(flat)
        threshold = float(np.percentile(np.abs(coeffs), (1.0 - keep_fraction) * 100))
        mask = np.abs(coeffs) >= threshold
        nonzero_values = coeffs[mask]
        nonzero_indices = np.where(mask)[0]
        sparse_coeffs = np.zeros_like(coeffs)
        sparse_coeffs[mask] = nonzero_values
        reconstructed = self._haar_inverse(sparse_coeffs)
        mse = float(np.mean((flat - reconstructed) ** 2))
        ratio = tensor.nbytes / max(
            nonzero_values.nbytes + nonzero_indices.nbytes + 8, 1
        )
        return CompressionResult(
            compressed_data={
                "indices": nonzero_indices,
                "values": nonzero_values,
                "n_original": n,
            },
            original_shape=tensor.shape,
            strategy=CompressionStrategy.WAVELET,
            compression_ratio=ratio,
            reconstruction_error=mse,
        )

    def _decompress_wavelet(
        self, data: Dict[str, Any], shape: Tuple[int, ...]
    ) -> np.ndarray:
        n_original = data["n_original"]
        coeffs = np.zeros(n_original, dtype=np.float64)
        coeffs[data["indices"]] = data["values"]
        return self._haar_inverse(coeffs).reshape(shape)

    def _get_quantizer(self, n_bits: int) -> LloydMaxQuantizer:
        if n_bits not in self._quantizers:
            self._quantizers[n_bits] = LloydMaxQuantizer(n_bits=n_bits)
        return self._quantizers[n_bits]

    def _get_hadamard_rotator(self, dim: int) -> HadamardRotator:
        if dim not in self._hadamard_rotators:
            self._hadamard_rotators[dim] = HadamardRotator(dim=dim)
        return self._hadamard_rotators[dim]

    @staticmethod
    def _haar_forward(signal: np.ndarray) -> np.ndarray:
        n = signal.size
        if n <= 1:
            return signal.copy()
        if n & (n - 1) != 0:
            new_n = 1 << (n - 1).bit_length()
            signal = np.pad(signal, (0, new_n - n))
            n = new_n
        out = signal.copy().astype(np.float64)
        result = np.zeros(n, dtype=np.float64)
        temp = out.copy()
        h = n
        while h > 1:
            half = h // 2
            even = temp[:h:2]
            odd = temp[1:h:2]
            approx = (even + odd) * 0.5
            detail = (even - odd) * 0.5
            result[:half] = approx
            result[half:h] = detail
            temp[:h] = result[:h]
            h = half
        return result

    @staticmethod
    def _haar_inverse(coeffs: np.ndarray) -> np.ndarray:
        n = coeffs.size
        if n <= 1:
            return coeffs.copy()
        out = np.zeros(n, dtype=np.float64)
        h = 1
        temp = coeffs.copy()
        while h < n:
            approx = temp[:h]
            detail = temp[h : 2 * h]
            out[: 2 * h : 2] = approx + detail
            out[1 : 2 * h : 2] = approx - detail
            temp[: 2 * h] = out[: 2 * h]
            h *= 2
        return temp

    def _reconstruction_mse(
        self, original: np.ndarray, reconstructed: np.ndarray
    ) -> float:
        orig = original.ravel().astype(np.float64)
        recon = reconstructed.ravel()
        if len(recon) != len(orig):
            if len(recon) > len(orig):
                recon = recon[: len(orig)]
            else:
                recon = np.pad(recon, (0, len(orig) - len(recon)))
        return float(np.mean((orig - recon) ** 2))
