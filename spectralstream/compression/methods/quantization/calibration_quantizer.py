"""
Calibration-Based Quantization Pipeline
=========================================
Data collection, quantizer selection, and pipeline orchestration
for calibration-aware quantization (GPTQ, AWQ, SqueezeLLM).

Uses existing GPTQQuant, AWQQuant, SqueezeLLMNonuniform implementations
from this package — adds calibration data collection, layer profiling,
smart quantizer selection, and end-to-end pipeline orchestration.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CalibrationData:
    weight_norms: Optional[np.ndarray] = None
    activation_scales: Optional[np.ndarray] = None
    hessian_diag: Optional[np.ndarray] = None
    weight_importance: Optional[np.ndarray] = None
    channel_importance: Optional[np.ndarray] = None
    layer_sensitivity: float = 0.5
    n_samples: int = 0


@dataclass
class QuantizedWeight:
    quantized: np.ndarray
    scales: np.ndarray
    zeros: Optional[np.ndarray] = None
    outliers: Optional[np.ndarray] = None
    outlier_indices: Optional[np.ndarray] = None
    codebook: Optional[np.ndarray] = None
    indices: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    method: str = ""
    n_bits: int = 4


@dataclass
class LayerProfile:
    name: str = ""
    shape: Tuple[int, ...] = (0,)
    n_elements: int = 0
    mean_abs: float = 0.0
    std: float = 0.0
    max_abs: float = 0.0
    sparsity: float = 0.0
    outlier_fraction: float = 0.0
    channel_energy_ratio: float = 0.0
    sensitivity: float = 0.5
    recommended_method: str = "gptq"


@dataclass
class CompressionResult:
    name: str
    method: str
    n_bits: int
    compression_ratio: float
    relative_error: float
    snr_db: float
    cosine_similarity: float
    original_shape: Tuple[int, ...]
    original_nbytes: int
    compressed_nbytes: int


LAYER_SENSITIVITY: Dict[str, float] = {
    "embed": 1.0,
    "tok_embeddings": 1.0,
    "wte": 1.0,
    "attn_q": 1.0,
    "q_proj": 1.0,
    "wq": 1.0,
    "query": 1.0,
    "attn_k": 0.92,
    "k_proj": 0.92,
    "wk": 0.92,
    "key": 0.92,
    "attn_v": 0.88,
    "v_proj": 0.88,
    "wv": 0.88,
    "value": 0.88,
    "attn_o": 1.0,
    "o_proj": 1.0,
    "wo": 1.0,
    "attn_norm": 0.7,
    "ln_1": 0.7,
    "ffn_gate": 0.55,
    "gate_proj": 0.55,
    "w1": 0.55,
    "ffn_up": 0.60,
    "up_proj": 0.60,
    "w3": 0.60,
    "ffn_down": 0.65,
    "down_proj": 0.65,
    "w2": 0.65,
    "ffn_norm": 0.50,
    "ln_2": 0.50,
    "norm": 0.50,
    "final_norm": 0.50,
    "output": 1.0,
    "lm_head": 1.0,
    "head": 1.0,
}


def _get_sensitivity(name: str) -> float:
    name_lower = name.lower()
    for key, val in LAYER_SENSITIVITY.items():
        if key in name_lower:
            return val
    if "norm" in name_lower or "bias" in name_lower:
        return 0.95
    if "weight" in name_lower:
        return 0.7
    return 0.5


def _compute_metrics(orig: np.ndarray, recon: np.ndarray) -> dict:
    o = orig.astype(np.float64).ravel()
    r = recon.astype(np.float64).ravel()
    noise = o - r
    mse = float(np.mean(noise**2))
    signal_power = float(np.mean(o**2)) + 1e-30
    snr_db = 10.0 * math.log10(signal_power / (mse + 1e-30))
    max_val = float(np.max(np.abs(o)))
    psnr_db = 10.0 * math.log10(max_val**2 / (mse + 1e-30)) if max_val > 0 else snr_db
    rel_error = float(np.linalg.norm(noise) / (np.linalg.norm(o) + 1e-30))
    cos_sim = float(np.dot(o, r) / (np.linalg.norm(o) * np.linalg.norm(r) + 1e-30))
    return {
        "mse": mse,
        "snr_db": snr_db,
        "psnr_db": psnr_db,
        "relative_error": rel_error,
        "cosine_similarity": cos_sim,
    }


class CalibrationDataCollector:
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path

    def collect(
        self,
        weight: np.ndarray,
        name: str = "",
        num_samples: int = 10,
    ) -> CalibrationData:
        w = weight.astype(np.float32)
        if w.ndim < 2:
            w_2d = w.reshape(1, -1)
        else:
            w_2d = w
        col_norms = np.sqrt(np.mean(w_2d**2, axis=0))
        hessian_diag = w.ravel() ** 2
        if w_2d.shape[0] > 1:
            channel_importance = np.sqrt(np.sum(w_2d**2, axis=1))
        else:
            channel_importance = np.sqrt(np.sum(w_2d**2, axis=0))
        if w_2d.shape[0] > 1:
            activation_scales = np.mean(np.abs(w_2d), axis=1)
        else:
            activation_scales = np.mean(np.abs(w_2d), axis=0)
        weight_importance = self._estimate_weight_importance(w_2d, name)
        sensitivity = _get_sensitivity(name)
        return CalibrationData(
            weight_norms=col_norms,
            activation_scales=activation_scales,
            hessian_diag=hessian_diag,
            weight_importance=weight_importance,
            channel_importance=channel_importance,
            layer_sensitivity=sensitivity,
            n_samples=num_samples,
        )

    def _estimate_weight_importance(
        self, w_2d: np.ndarray, name: str = ""
    ) -> np.ndarray:
        sensitivity = _get_sensitivity(name)
        magnitudes = np.abs(w_2d)
        mag_max = np.max(magnitudes)
        if mag_max > 1e-8:
            mag_norm = magnitudes / mag_max
        else:
            mag_norm = magnitudes
        importance = mag_norm * sensitivity
        if w_2d.shape[0] > 1:
            channel_mag = np.mean(magnitudes, axis=1, keepdims=True)
            ch_max = np.max(channel_mag)
            if ch_max > 1e-8:
                channel_boost = 0.5 + 0.5 * (channel_mag / ch_max)
            else:
                channel_boost = np.ones_like(channel_mag)
            importance = importance * channel_boost
        return importance.ravel().astype(np.float32)

    def collect_from_weights(
        self,
        weights: Dict[str, np.ndarray],
        num_samples: int = 10,
    ) -> Dict[str, CalibrationData]:
        return {
            name: self.collect(weight, name=name, num_samples=num_samples)
            for name, weight in weights.items()
        }


class QuantizerSelector:
    @staticmethod
    def profile(
        weight: np.ndarray,
        name: str = "",
        calibration_data: Optional[CalibrationData] = None,
    ) -> LayerProfile:
        w = weight.astype(np.float32)
        flat = w.ravel()
        n = flat.size
        profile = LayerProfile(name=name, shape=w.shape, n_elements=n)
        if n == 0:
            return profile
        profile.mean_abs = float(np.mean(np.abs(flat)))
        profile.std = float(np.std(flat))
        profile.max_abs = float(np.max(np.abs(flat)))
        profile.sparsity = float(np.mean(np.abs(flat) < 1e-10))
        profile.sensitivity = _get_sensitivity(name)
        threshold = np.percentile(np.abs(flat), 99.9)
        profile.outlier_fraction = float(np.mean(np.abs(flat) > threshold))
        if w.ndim >= 2 and w.shape[0] > 1:
            channel_energy = np.sqrt(np.sum(w**2, axis=1))
            total_energy = np.linalg.norm(channel_energy) + 1e-10
            profile.channel_energy_ratio = float(np.max(channel_energy) / total_energy)
        profile.recommended_method = QuantizerSelector._select_method(profile)
        return profile

    @staticmethod
    def _select_method(profile: LayerProfile) -> str:
        if profile.sensitivity >= 0.9:
            return "gptq"
        if profile.outlier_fraction > 0.01:
            return "squeezellm"
        if profile.sparsity > 0.5:
            return "block_int4"
        if profile.std > 0.1 and profile.sensitivity < 0.7:
            return "awq"
        return "gptq"

    @staticmethod
    def select_quantizer(method: str, bits: int = 4, group_size: int = 128):
        if method == "gptq":
            from spectralstream.compression.methods.quantization.gptq import GPTQQuant

            return GPTQQuant()
        elif method == "awq":
            from spectralstream.compression.methods.quantization.awq import AWQQuant

            return AWQQuant()
        elif method == "squeezellm":
            from spectralstream.compression.methods.quantization.squeezellm import (
                SqueezeLLMNonuniform,
            )

            return SqueezeLLMNonuniform()
        elif method == "block_int8":
            from spectralstream.compression.engine._methods import _BlockINT8

            return _BlockINT8()
        elif method == "block_int4":
            from spectralstream.compression.engine._methods import _BlockINT4

            return _BlockINT4()
        else:
            from spectralstream.compression.methods.quantization.gptq import GPTQQuant

            return GPTQQuant()


class CalibrationPipeline:
    def __init__(
        self,
        bits: int = 4,
        group_size: int = 128,
        target_error: float = 0.02,
        force_method: Optional[str] = None,
    ):
        self.bits = bits
        self.group_size = group_size
        self.target_error = target_error
        self.force_method = force_method
        self.collector = CalibrationDataCollector()

    def compress_weights(
        self,
        weights: Dict[str, np.ndarray],
        num_calib_samples: int = 10,
    ) -> Dict[str, CompressionResult]:
        results = {}
        cal_data = self.collector.collect_from_weights(weights, num_calib_samples)
        for name, weight in weights.items():
            try:
                result = self._compress_single(name, weight, cal_data.get(name))
                results[name] = result
            except Exception as e:
                logger.warning("Failed to compress %s: %s", name, e)
        return results

    def _compress_single(
        self,
        name: str,
        weight: np.ndarray,
        cal_data: Optional[CalibrationData],
    ) -> CompressionResult:
        profile = QuantizerSelector.profile(weight, name, cal_data)
        method = self.force_method or profile.recommended_method
        quantizer = QuantizerSelector.select_quantizer(
            method, self.bits, self.group_size
        )

        from spectralstream.compression.engine._methods import _BlockINT8, _BlockINT4
        from spectralstream.compression.methods.quantization.gptq import GPTQQuant
        from spectralstream.compression.methods.quantization.awq import AWQQuant
        from spectralstream.compression.methods.quantization.squeezellm import (
            SqueezeLLMNonuniform,
        )

        if method in ("block_int4", "block_int8") or method == "gptq":
            data_bytes, meta = quantizer.compress(weight)
            recon = quantizer.decompress(data_bytes, meta)
            compressed_bytes = len(data_bytes)
        elif method == "awq":
            data_bytes, meta = quantizer.compress(weight)
            recon = quantizer.decompress(data_bytes, meta)
            compressed_bytes = len(data_bytes)
        elif method == "squeezellm":
            data_bytes, meta = quantizer.compress(weight)
            recon = quantizer.decompress(data_bytes, meta)
            compressed_bytes = len(data_bytes)
        else:
            data_bytes, meta = quantizer.compress(weight)
            recon = quantizer.decompress(data_bytes, meta)
            compressed_bytes = len(data_bytes)

        metrics = _compute_metrics(weight, recon)
        original_bytes = weight.nbytes
        ratio = original_bytes / max(compressed_bytes, 1)

        return CompressionResult(
            name=name,
            method=method,
            n_bits=self.bits,
            compression_ratio=ratio,
            relative_error=metrics["relative_error"],
            snr_db=metrics["snr_db"],
            cosine_similarity=metrics["cosine_similarity"],
            original_shape=weight.shape,
            original_nbytes=original_bytes,
            compressed_nbytes=compressed_bytes,
        )

    def compress_single(
        self,
        name: str,
        weight: np.ndarray,
        method: Optional[str] = None,
        bits: Optional[int] = None,
    ) -> CompressionResult:
        cal_data = self.collector.collect(weight, name=name)
        bits = bits or self.bits
        original_force = self.force_method
        if method:
            self.force_method = method
        self.bits = bits
        try:
            return self._compress_single(name, weight, cal_data)
        finally:
            self.force_method = original_force
            self.bits = 4 if bits is None else bits
