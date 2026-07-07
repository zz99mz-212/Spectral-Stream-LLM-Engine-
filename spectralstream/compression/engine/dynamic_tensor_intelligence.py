"""
Dynamic Tensor Intelligence — per-tensor adaptive compression
==============================================================
Analyzes tensor properties at compression time, selects optimal method,
and adapts strategy based on observed reconstruction errors.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
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
    BAND_COMPRESSION,
)

logger = logging.getLogger(__name__)


@dataclass
class TensorFeatures:
    n_elements: int = 0
    ndim: int = 0
    sparsity: float = 0.0
    mean_abs: float = 0.0
    std: float = 0.0
    kurtosis: float = 0.0
    skewness: float = 0.0
    spectral_entropy: float = 0.0
    dct_concentration: float = 0.0
    effective_rank: float = 0.0
    value_range: float = 0.0
    snr_estimate: float = 0.0

    def to_vector(self) -> np.ndarray:
        return np.array(
            [
                self.n_elements,
                self.ndim,
                self.sparsity,
                self.mean_abs,
                self.std,
                self.kurtosis,
                self.skewness,
                self.spectral_entropy,
                self.dct_concentration,
                self.effective_rank,
                self.value_range,
                self.snr_estimate,
            ],
            dtype=np.float64,
        )


@dataclass
class StrategyPerformance:
    method: str
    errors: deque = field(default_factory=lambda: deque(maxlen=100))
    ratios: deque = field(default_factory=lambda: deque(maxlen=100))
    times_ms: deque = field(default_factory=lambda: deque(maxlen=100))

    @property
    def mean_error(self) -> float:
        return float(np.mean(self.errors)) if self.errors else 1.0

    @property
    def mean_ratio(self) -> float:
        return float(np.mean(self.ratios)) if self.ratios else 1.0

    @property
    def mean_time_ms(self) -> float:
        return float(np.mean(self.times_ms)) if self.times_ms else 0.0

    @property
    def score(self) -> float:
        if not self.errors:
            return 0.5
        error_penalty = 1.0 / (1.0 + self.mean_error * 100)
        ratio_bonus = min(self.mean_ratio / 10.0, 1.0)
        time_penalty = 1.0 / (1.0 + self.mean_time_ms / 100.0)
        return 0.5 * error_penalty + 0.3 * ratio_bonus + 0.2 * time_penalty


_COMPRESSION_METHODS = [
    "spectral_dct",
    "hadamard",
    "low_rank",
    "product_quantize",
    "sparsify",
    "uniform_quantize",
]


class DynamicTensorIntelligence:
    def __init__(
        self,
        target_error: float = 0.01,
        min_ratio: float = 2.0,
        adaptation_rate: float = 0.1,
        feedback_window: int = 50,
        enable_online_learning: bool = True,
    ) -> None:
        self.target_error = target_error
        self.min_ratio = min_ratio
        self.adaptation_rate = adaptation_rate
        self.feedback_window = feedback_window
        self.enable_online_learning = enable_online_learning
        self._performance: Dict[str, StrategyPerformance] = {}
        for m in _COMPRESSION_METHODS:
            self._performance[m] = StrategyPerformance(method=m)
        self._feature_method_cache: Dict[str, str] = {}
        self._recent_errors: deque = deque(maxlen=feedback_window)
        self._quantizers: Dict[int, LloydMaxQuantizer] = {}
        self._hadamard_rotators: Dict[int, HadamardRotator] = {}

    def analyze_tensor(self, tensor: np.ndarray) -> TensorFeatures:
        tensor = np.asarray(tensor, dtype=np.float64)
        flat = tensor.ravel()
        n = flat.size
        features = TensorFeatures(n_elements=n, ndim=tensor.ndim)
        if n == 0:
            return features
        features.mean_abs = float(np.mean(np.abs(flat)))
        features.std = float(np.std(flat))
        features.value_range = float(np.max(flat) - np.min(flat))
        features.sparsity = float(np.mean(np.abs(flat) < 1e-6))
        if features.std > 1e-10:
            centered = (flat - np.mean(flat)) / features.std
            features.skewness = float(np.mean(centered**3))
            features.kurtosis = float(np.mean(centered**4) - 3.0)
        sample = flat[: min(n, 4096)]
        if len(sample) >= 4:
            features.spectral_entropy = spectral_entropy(sample)
            try:
                coeffs = dct(sample)
                energy = coeffs**2
                total_energy = float(np.sum(energy))
                if total_energy > 1e-10:
                    sorted_energy = np.sort(energy.ravel())[::-1]
                    cumulative = np.cumsum(sorted_energy) / total_energy
                    k_90 = int(np.searchsorted(cumulative, 0.90)) + 1
                    features.dct_concentration = k_90 / max(len(sample), 1)
                else:
                    features.dct_concentration = 1.0
            except (ValueError, np.linalg.LinAlgError):
                features.dct_concentration = 1.0
        if tensor.ndim == 2 and tensor.shape[0] > 1 and tensor.shape[1] > 1:
            try:
                s = np.linalg.svd(
                    tensor[: min(tensor.shape[0], 256), : min(tensor.shape[1], 256)],
                    compute_uv=False,
                )
                s_norm = s / (np.sum(s) + 1e-10)
                nonzero = s_norm[s_norm > 1e-10]
                features.effective_rank = float(
                    np.exp(-np.sum(nonzero * np.log(nonzero)))
                )
            except np.linalg.LinAlgError:
                features.effective_rank = 1.0
        if len(sample) >= 4:
            try:
                coeffs = dct(sample)
                signal_coeffs = np.zeros_like(coeffs)
                signal_coeffs[: max(1, len(coeffs) // 4)] = coeffs[
                    : max(1, len(coeffs) // 4)
                ]
                signal = idct(signal_coeffs)
                noise = sample - signal
                signal_power = float(np.sum(signal**2))
                noise_power = float(np.sum(noise**2)) + 1e-20
                features.snr_estimate = 10.0 * np.log10(signal_power / noise_power)
            except (ValueError, np.linalg.LinAlgError):
                features.snr_estimate = 20.0
        return features

    def select_method(self, features: TensorFeatures) -> str:
        feature_key = f"s{round(features.sparsity, 1)}_e{round(features.spectral_entropy, 1)}_c{round(features.dct_concentration, 1)}"
        if feature_key in self._feature_method_cache:
            cached = self._feature_method_cache[feature_key]
            if cached in self._performance:
                if self._performance[cached].mean_error < self.target_error * 2:
                    return cached
        method = self._decision_tree_select(features)
        if self.enable_online_learning and len(self._recent_errors) > 10:
            best_method = self._best_performing_method()
            if best_method and best_method != method:
                best_perf = self._performance[best_method]
                current_perf = self._performance[method]
                if best_perf.score > current_perf.score * 1.3:
                    method = best_method
        return method

    def compress(
        self,
        tensor: np.ndarray,
        method: Optional[str] = None,
        bits: int = 4,
        **kwargs: Any,
    ) -> Tuple[Any, float, float]:
        tensor = np.asarray(tensor, dtype=np.float64)
        start = time.perf_counter()
        if method is None:
            features = self.analyze_tensor(tensor)
            method = self.select_method(features)
        if method == "spectral_dct":
            compressed, ratio, mse = self._compress_dct(tensor, bits, **kwargs)
        elif method == "hadamard":
            compressed, ratio, mse = self._compress_hadamard(tensor, bits, **kwargs)
        elif method == "low_rank":
            compressed, ratio, mse = self._compress_low_rank(tensor, **kwargs)
        elif method == "product_quantize":
            compressed, ratio, mse = self._compress_pq(tensor, bits, **kwargs)
        elif method == "sparsify":
            compressed, ratio, mse = self._compress_sparsify(tensor, **kwargs)
        elif method == "uniform_quantize":
            compressed, ratio, mse = self._compress_uniform(tensor, bits, **kwargs)
        else:
            compressed = tensor.copy()
            ratio = 1.0
            mse = 0.0
        elapsed = (time.perf_counter() - start) * 1000.0
        self._performance[method].errors.append(mse)
        self._performance[method].ratios.append(ratio)
        self._performance[method].times_ms.append(elapsed)
        self._recent_errors.append(mse)
        return compressed, ratio, mse

    def decompress(
        self,
        compressed: Any,
        method: str,
        original_shape: Tuple[int, ...],
        **kwargs: Any,
    ) -> np.ndarray:
        if method == "identity":
            return np.asarray(compressed).copy()
        elif method == "spectral_dct":
            return self._decompress_dct(compressed, original_shape)
        elif method == "hadamard":
            return self._decompress_hadamard(compressed, original_shape)
        elif method == "low_rank":
            return self._decompress_low_rank(compressed, original_shape)
        elif method == "product_quantize":
            return self._decompress_pq(compressed, original_shape)
        elif method == "sparsify":
            return self._decompress_sparsify(compressed, original_shape)
        elif method == "uniform_quantize":
            return self._decompress_uniform(compressed, original_shape)
        else:
            return np.zeros(original_shape, dtype=np.float64)

    def update_feedback(
        self, method: str, error: float, ratio: float, time_ms: float = 0.0
    ) -> None:
        if method in self._performance:
            self._performance[method].errors.append(error)
            self._performance[method].ratios.append(ratio)
            self._performance[method].times_ms.append(time_ms)
        self._recent_errors.append(error)

    def get_performance_summary(self) -> Dict[str, Dict[str, float]]:
        return {
            m: {
                "mean_error": p.mean_error,
                "mean_ratio": p.mean_ratio,
                "mean_time_ms": p.mean_time_ms,
                "score": p.score,
                "n_samples": float(len(p.errors)),
            }
            for m, p in self._performance.items()
        }

    def _decision_tree_select(self, features: TensorFeatures) -> str:
        if features.sparsity > 0.85:
            return "sparsify"
        if (
            features.ndim == 2
            and features.effective_rank > 0
            and features.effective_rank < min(features.n_elements**0.5, 50) * 0.3
        ):
            return "low_rank"
        if features.dct_concentration < 0.25:
            return "spectral_dct"
        if features.spectral_entropy < 0.5 and features.dct_concentration < 0.5:
            return "hadamard"
        return "spectral_dct"

    def _best_performing_method(self) -> Optional[str]:
        best_method = None
        best_score = -1.0
        for m, p in self._performance.items():
            if len(p.errors) >= 5 and p.score > best_score:
                best_score = p.score
                best_method = m
        return best_method

    def _compress_dct(
        self, tensor: np.ndarray, bits: int, **kwargs: Any
    ) -> Tuple[Dict[str, Any], float, float]:
        flat = tensor.ravel().astype(np.float64)
        n = flat.size
        n_padded = 1 << (n - 1).bit_length() if n > 1 else 1
        padded = np.pad(flat, (0, n_padded - n)) if n_padded > n else flat
        coeffs = dct(padded)
        sorted_sq = np.sort(np.abs(coeffs) ** 2)[::-1]
        total_energy = float(np.sum(sorted_sq))
        k_keep = n_padded
        if total_energy > 1e-10:
            cumulative = np.cumsum(sorted_sq) / total_energy
            k_keep = max(1, int(np.searchsorted(cumulative, 0.95)) + 1)
        quantizer = self._get_quantizer(bits)
        kept = coeffs[:k_keep]
        if not quantizer.trained:
            quantizer.train(kept)
        indices, centroids = quantizer.compress(kept)
        dequant = centroids[indices]
        recon_coeffs = np.zeros(n_padded, dtype=np.float64)
        recon_coeffs[:k_keep] = dequant
        reconstructed = idct(recon_coeffs)[:n]
        mse = float(np.mean((flat - reconstructed) ** 2))
        ratio = tensor.nbytes / max(indices.nbytes + centroids.nbytes, 1)
        return (
            {
                "indices": indices,
                "centroids": centroids,
                "k_keep": k_keep,
                "n_padded": n_padded,
                "n_original": n,
                "bits": bits,
            },
            ratio,
            mse,
        )

    def _decompress_dct(
        self, data: Dict[str, Any], shape: Tuple[int, ...]
    ) -> np.ndarray:
        quantizer = self._get_quantizer(data["bits"])
        quantizer.centroids = data["centroids"]
        quantizer.trained = True
        coeffs = np.zeros(data["n_padded"], dtype=np.float64)
        coeffs[: data["k_keep"]] = data["centroids"][data["indices"]]
        return idct(coeffs)[: data["n_original"]].reshape(shape)

    def _compress_hadamard(
        self, tensor: np.ndarray, bits: int, **kwargs: Any
    ) -> Tuple[Dict[str, Any], float, float]:
        flat = tensor.ravel().astype(np.float32)
        n = flat.size
        n_rot = 1 << (n - 1).bit_length() if n > 1 else 1
        rotator = self._get_hadamard(n)
        padded = np.zeros(n_rot, dtype=np.float32)
        padded[:n] = flat
        rotated = rotator.rotate(padded.reshape(1, -1)).ravel()
        quantizer = self._get_quantizer(bits)
        if not quantizer.trained:
            quantizer.train(rotated)
        indices, centroids = quantizer.compress(rotated)
        dequant = centroids[indices]
        inv = rotator.inverse_rotate(dequant.reshape(1, -1)).ravel()
        mse = float(np.mean((padded - inv) ** 2))
        ratio = tensor.nbytes / max(indices.nbytes + centroids.nbytes, 1)
        return (
            {
                "indices": indices,
                "centroids": centroids,
                "n_original": n,
                "n_rotated": n_rot,
                "bits": bits,
            },
            ratio,
            mse,
        )

    def _decompress_hadamard(
        self, data: Dict[str, Any], shape: Tuple[int, ...]
    ) -> np.ndarray:
        rotator = self._get_hadamard(data["n_original"])
        dequant = data["centroids"][data["indices"]]
        inv = rotator.inverse_rotate(dequant.reshape(1, -1)).ravel()
        return inv[: data["n_original"]].reshape(shape)

    def _compress_low_rank(
        self, tensor: np.ndarray, keep_energy: float = 0.90, **kwargs: Any
    ) -> Tuple[Dict[str, Any], float, float]:
        if tensor.ndim < 2:
            return self._compress_dct(tensor, 4, **kwargs)
        U, s, Vt = np.linalg.svd(tensor, full_matrices=False)
        total = float(np.sum(s**2))
        rank = len(s)
        if total > 1e-10:
            cum = np.cumsum(s**2) / total
            rank = max(1, int(np.searchsorted(cum, keep_energy)) + 1)
        U_k, s_k, Vt_k = U[:, :rank], s[:rank], Vt[:rank, :]
        recon = U_k @ np.diag(s_k) @ Vt_k
        mse = float(np.mean((tensor - recon) ** 2))
        ratio = tensor.nbytes / max(U_k.nbytes + s_k.nbytes + Vt_k.nbytes, 1)
        return {"U": U_k, "s": s_k, "Vt": Vt_k, "rank": rank}, ratio, mse

    def _decompress_low_rank(
        self, data: Dict[str, Any], shape: Tuple[int, ...]
    ) -> np.ndarray:
        return (data["U"] @ np.diag(data["s"]) @ data["Vt"]).reshape(shape)

    def _compress_pq(
        self, tensor: np.ndarray, bits: int, n_subspaces: int = 8, **kwargs: Any
    ) -> Tuple[Dict[str, Any], float, float]:
        flat = tensor.ravel().astype(np.float64)
        n = flat.size
        sub_dim = max(1, n // n_subspaces)
        n_padded = sub_dim * n_subspaces
        padded = np.pad(flat, (0, n_padded - n)) if n_padded > n else flat[:n_padded]
        subspaces = padded.reshape(n_subspaces, sub_dim)
        n_centroids = min(1 << bits, n_subspaces)
        codebook = np.zeros((n_subspaces, n_centroids, sub_dim), dtype=np.float64)
        codes = np.zeros(n_subspaces, dtype=np.int32)
        for i in range(n_subspaces):
            sub = subspaces[i]
            mu, sigma = float(np.mean(sub)), max(float(np.std(sub)), 1e-10)
            scale = max(abs(mu - 4 * sigma), abs(mu + 4 * sigma), 1e-8)
            normed = np.clip(sub / scale, -1.0, 1.0)
            centroids = np.linspace(-1.0, 1.0, n_centroids)
            for _ in range(8):
                b = (centroids[1:] + centroids[:-1]) * 0.5
                idx = np.clip(np.digitize(normed, b), 0, n_centroids - 1)
                for c in range(n_centroids):
                    mask = idx == c
                    if np.any(mask):
                        centroids[c] = np.mean(normed[mask])
            codebook[i] = centroids * scale
            b = (centroids[1:] + centroids[:-1]) * 0.5
            codes[i] = int(
                np.bincount(
                    np.clip(np.digitize(normed, b), 0, n_centroids - 1)
                ).argmax()
            )
        recon = np.zeros(n_padded, dtype=np.float64)
        for i in range(n_subspaces):
            recon[i * sub_dim : (i + 1) * sub_dim] = codebook[i, codes[i]]
        mse = float(np.mean((padded - recon) ** 2))
        ratio = tensor.nbytes / max(codes.nbytes + codebook.nbytes, 1)
        return (
            {
                "codes": codes,
                "codebook": codebook,
                "n_subspaces": n_subspaces,
                "sub_dim": sub_dim,
                "n_original": n,
            },
            ratio,
            mse,
        )

    def _decompress_pq(
        self, data: Dict[str, Any], shape: Tuple[int, ...]
    ) -> np.ndarray:
        codes, codebook = data["codes"], data["codebook"]
        n_sub, sub_dim = data["n_subspaces"], data["sub_dim"]
        recon = np.zeros(n_sub * sub_dim, dtype=np.float64)
        for i in range(n_sub):
            recon[i * sub_dim : (i + 1) * sub_dim] = codebook[i, codes[i]]
        return recon[: data["n_original"]].reshape(shape)

    def _compress_sparsify(
        self, tensor: np.ndarray, sparsity: float = 0.90, **kwargs: Any
    ) -> Tuple[Dict[str, Any], float, float]:
        flat = tensor.ravel().astype(np.float64)
        threshold = float(np.percentile(np.abs(flat), sparsity * 100))
        mask = np.abs(flat) >= threshold
        vals = flat[mask]
        idx = np.where(mask)[0]
        recon = np.zeros_like(flat)
        recon[mask] = vals
        mse = float(np.mean((flat - recon) ** 2))
        ratio = tensor.nbytes / max(vals.nbytes + idx.nbytes, 1)
        return {"indices": idx, "values": vals}, ratio, mse

    def _decompress_sparsify(
        self, data: Dict[str, Any], shape: Tuple[int, ...]
    ) -> np.ndarray:
        flat = np.zeros(int(np.prod(shape)), dtype=np.float64)
        flat[data["indices"]] = data["values"]
        return flat.reshape(shape)

    def _compress_uniform(
        self, tensor: np.ndarray, bits: int, **kwargs: Any
    ) -> Tuple[Dict[str, Any], float, float]:
        flat = tensor.ravel().astype(np.float64)
        n_levels = 1 << bits
        lo, hi = float(np.min(flat)), float(np.max(flat))
        scale = (hi - lo) / max(n_levels - 1, 1)
        quantized = np.clip(np.round((flat - lo) / scale), 0, n_levels - 1)
        dequant = quantized * scale + lo
        mse = float(np.mean((flat - dequant) ** 2))
        ratio = tensor.nbytes / max(quantized.nbytes, 1)
        return (
            {"quantized": quantized, "lo": lo, "scale": scale, "bits": bits},
            ratio,
            mse,
        )

    def _decompress_uniform(
        self, data: Dict[str, Any], shape: Tuple[int, ...]
    ) -> np.ndarray:
        return (data["quantized"] * data["scale"] + data["lo"]).reshape(shape)

    def _get_quantizer(self, bits: int) -> LloydMaxQuantizer:
        if bits not in self._quantizers:
            self._quantizers[bits] = LloydMaxQuantizer(n_bits=bits)
        return self._quantizers[bits]

    def _get_hadamard(self, dim: int) -> HadamardRotator:
        if dim not in self._hadamard_rotators:
            self._hadamard_rotators[dim] = HadamardRotator(dim=dim)
        return self._hadamard_rotators[dim]
