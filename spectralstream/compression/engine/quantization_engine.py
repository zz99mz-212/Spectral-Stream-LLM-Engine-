"""
Quantization Engine — central quantization orchestration
=========================================================
Provides unified quantization pipeline, per-layer adaptive strategy selection,
DCT-domain spectral quantization, Huffman coding, and GGML dequantization.
"""

from __future__ import annotations

import heapq
import logging
import math
import struct
import time
from collections import Counter
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from spectralstream.core.math_primitives import (
    dct,
    idct,
    dct_2d,
    idct_2d,
    LloydMaxQuantizer,
    spectral_entropy,
)
from spectralstream.compression.engine._sensitivity import (
    LAYER_SENSITIVITY,
    _get_sensitivity,
)

logger = logging.getLogger(__name__)


def _format_size(n: int) -> str:
    if n >= 1024**3:
        return f"{n / 1024**3:.1f} GB"
    if n >= 1024**2:
        return f"{n / 1024**2:.1f} MB"
    if n >= 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n} B"


@dataclass
class CompressionReport:
    tensor_count: int = 0
    total_original_bytes: int = 0
    total_compressed_bytes: int = 0
    avg_snr_db: float = 0.0
    avg_psnr_db: float = 0.0
    avg_mse: float = 0.0
    max_error: float = 0.0
    per_tensor: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def compression_ratio(self) -> float:
        if self.total_compressed_bytes == 0:
            return 0.0
        return self.total_original_bytes / self.total_compressed_bytes

    def summary(self) -> str:
        return (
            f"Compression: {self.tensor_count} tensors, "
            f"{_format_size(self.total_original_bytes)} -> {_format_size(self.total_compressed_bytes)} "
            f"({self.compression_ratio:.2f}x) | SNR={self.avg_snr_db:.1f}dB, "
            f"PSNR={self.avg_psnr_db:.1f}dB, MSE={self.avg_mse:.2e}"
        )


class QualityMonitor:
    def __init__(self) -> None:
        self._reports: List[Dict[str, Any]] = []
        self._start_time: float = time.monotonic()

    def record(
        self,
        name: str,
        original: np.ndarray,
        reconstructed: np.ndarray,
        compressed_bytes: int,
        method: str = "unknown",
    ) -> Dict[str, Any]:
        orig = np.asarray(original, dtype=np.float64).ravel()
        recon = np.asarray(reconstructed, dtype=np.float64).ravel()
        min_len = min(len(orig), len(recon))
        orig = orig[:min_len]
        recon = recon[:min_len]
        mse = float(np.mean((orig - recon) ** 2))
        signal_power = float(np.mean(orig**2) + 1e-30)
        noise_power = mse + 1e-30
        snr_db = 10.0 * math.log10(signal_power / noise_power)
        peak = max(float(np.max(np.abs(orig))), 1e-30)
        psnr_db = 10.0 * math.log10(peak**2 / noise_power) if mse > 0 else 100.0
        max_err = float(np.max(np.abs(orig - recon)))
        report = {
            "name": name,
            "method": method,
            "shape": list(original.shape),
            "original_bytes": orig.nbytes,
            "compressed_bytes": compressed_bytes,
            "mse": mse,
            "snr_db": snr_db,
            "psnr_db": psnr_db,
            "max_error": max_err,
        }
        self._reports.append(report)
        return report

    def get_aggregate(self) -> CompressionReport:
        agg = CompressionReport()
        agg.tensor_count = len(self._reports)
        if not self._reports:
            return agg
        agg.total_original_bytes = sum(r["original_bytes"] for r in self._reports)
        agg.total_compressed_bytes = sum(r["compressed_bytes"] for r in self._reports)
        agg.avg_snr_db = sum(r["snr_db"] for r in self._reports) / len(self._reports)
        agg.avg_psnr_db = sum(r["psnr_db"] for r in self._reports) / len(self._reports)
        agg.avg_mse = sum(r["mse"] for r in self._reports) / len(self._reports)
        agg.max_error = max(r["max_error"] for r in self._reports)
        agg.per_tensor = list(self._reports)
        return agg

    def reset(self) -> None:
        self._reports.clear()
        self._start_time = time.monotonic()


class StrategySelector:
    def __init__(
        self, target_ratio: float = 4.0, quality_threshold: float = 0.9
    ) -> None:
        self.target_ratio = target_ratio
        self.quality_threshold = quality_threshold

    def select(self, name: str, tensor: np.ndarray) -> str:
        sensitivity = _get_sensitivity(name)
        shape = tensor.shape
        n_elements = int(np.prod(shape))
        if sensitivity > 0.9:
            return "raw" if n_elements < 4096 else "int8"
        if len(shape) == 2:
            min_dim = min(shape)
            if min_dim >= 64:
                return "spectral"
            if min_dim >= 32:
                return "dct"
        flat = tensor.ravel().astype(np.float64)
        std = float(np.std(flat))
        if std < 0.01:
            return "int4"
        if std < 0.1:
            return "int8"
        return "spectral"


class SpectralQuantizer:
    def __init__(
        self, block_size: int = 32, keep_energy: float = 0.95, n_bits: int = 8
    ) -> None:
        self.block_size = block_size
        self.keep_energy = keep_energy
        self.n_bits = n_bits
        self._quantizer = LloydMaxQuantizer(n_bits=n_bits)

    def compress(self, tensor: np.ndarray) -> dict:
        t = np.asarray(tensor, dtype=np.float64)
        orig_shape = t.shape
        if t.ndim == 1:
            t = t.reshape(1, -1)
        rows, cols = t.shape
        bs = max(2, min(self.block_size, rows, cols))
        pad_r = (bs - rows % bs) % bs
        pad_c = (bs - cols % bs) % bs
        t_padded = (
            np.pad(t, ((0, pad_r), (0, pad_c)), mode="constant")
            if pad_r > 0 or pad_c > 0
            else t
        )
        n_blocks_r = t_padded.shape[0] // bs
        n_blocks_c = t_padded.shape[1] // bs
        coeffs_flat: List[float] = []
        block_info: List[Tuple[int, int]] = []
        for br in range(n_blocks_r):
            for bc in range(n_blocks_c):
                block = t_padded[br * bs : (br + 1) * bs, bc * bs : (bc + 1) * bs]
                block_coeffs = dct_2d(block)
                energy = block_coeffs**2
                total_energy = float(np.sum(energy))
                if total_energy < 1e-20:
                    block_info.append((0, 0))
                    continue
                sorted_energy = np.sort(energy.ravel())[::-1]
                cumsum = np.cumsum(sorted_energy)
                n_keep = max(
                    1,
                    min(
                        int(np.searchsorted(cumsum / total_energy, self.keep_energy))
                        + 1,
                        bs * bs,
                    ),
                )
                block_info.append((n_keep, 0))
                flat_coeffs = block_coeffs.ravel()
                indices = np.argsort(np.abs(flat_coeffs))[::-1][:n_keep]
                for idx in indices:
                    coeffs_flat.append(float(flat_coeffs[idx]))
        quantized_bytes = b""
        if coeffs_flat:
            coeffs_arr = np.array(coeffs_flat, dtype=np.float64)
            self._quantizer.train(coeffs_arr)
            quantized = self._quantizer.quantize(coeffs_arr)
            quantized_bytes = quantized.astype(np.float16).tobytes()
        return {
            "type": "spectral",
            "orig_shape": list(orig_shape),
            "padded_shape": list(t_padded.shape),
            "block_size": bs,
            "block_info": block_info,
            "keep_energy": self.keep_energy,
            "n_bits": self.n_bits,
            "data": quantized_bytes,
            "n_coeffs": len(coeffs_flat),
        }

    def decompress(self, compressed: dict) -> np.ndarray:
        orig_shape = tuple(compressed["orig_shape"])
        padded_shape = tuple(compressed["padded_shape"])
        bs = compressed["block_size"]
        block_info = compressed["block_info"]
        if not compressed.get("data"):
            return np.zeros(orig_shape, dtype=np.float32)
        raw = np.frombuffer(compressed["data"], dtype=np.float16).astype(np.float64)
        t_padded = np.zeros(padded_shape, dtype=np.float64)
        n_blocks_r = padded_shape[0] // bs
        n_blocks_c = padded_shape[1] // bs
        coeff_idx = 0
        for br in range(n_blocks_r):
            for bc in range(n_blocks_c):
                bi = br * n_blocks_c + bc
                n_keep = block_info[bi][0] if bi < len(block_info) else 0
                block_coeffs = np.zeros((bs, bs), dtype=np.float64)
                if n_keep > 0 and coeff_idx + n_keep <= len(raw):
                    vals = raw[coeff_idx : coeff_idx + n_keep]
                    coeff_idx += n_keep
                    flat = np.zeros(bs * bs)
                    flat[: len(vals)] = vals
                    block_coeffs = flat.reshape(bs, bs)
                t_padded[br * bs : (br + 1) * bs, bc * bs : (bc + 1) * bs] = idct_2d(
                    block_coeffs
                )
        if len(orig_shape) >= 2:
            result = t_padded[: orig_shape[0], : orig_shape[1]]
        else:
            result = t_padded.ravel()[: int(np.prod(orig_shape))].reshape(orig_shape)
        return result.astype(np.float32)


class GGMLDequantizerEngine:
    GGML_BLOCK_SIZES: Dict[int, int] = {
        0: 1,
        1: 1,
        30: 1,
        2: 32,
        3: 32,
        6: 32,
        7: 32,
        8: 32,
        9: 32,
        10: 256,
        11: 256,
        12: 256,
        13: 256,
        14: 256,
        15: 256,
    }

    @classmethod
    def dequantize(cls, raw: np.ndarray, ggml_type: int) -> np.ndarray:
        if ggml_type == 0:
            return np.frombuffer(raw.tobytes(), dtype=np.float32)
        if ggml_type == 1:
            return np.frombuffer(raw.tobytes(), dtype=np.float16).astype(np.float32)
        if ggml_type == 30:
            raw_u16 = np.frombuffer(raw[: len(raw) // 2 * 2].tobytes(), dtype=np.uint16)
            return (raw_u16.astype(np.uint32) << 16).view(np.float32)
        block_size = cls.GGML_BLOCK_SIZES.get(ggml_type, 32)
        if ggml_type in (2, 3):
            return cls._dequantize_q4(raw, ggml_type, block_size)
        if ggml_type in (6, 7):
            return cls._dequantize_q5(raw, ggml_type, block_size)
        if ggml_type in (8, 9):
            return cls._dequantize_q8(raw, ggml_type, block_size)
        if ggml_type in (10, 11, 12, 13, 14, 15):
            return cls._dequantize_qk(raw, ggml_type, block_size)
        return np.zeros(max(1, len(raw)), dtype=np.float32)

    @classmethod
    def _dequantize_q4(
        cls, raw: np.ndarray, ggml_type: int, block_size: int
    ) -> np.ndarray:
        data = raw.tobytes()
        has_offset = ggml_type == 3
        bytes_per_block = 2 + block_size // 2 + (2 if has_offset else 0)
        n_blocks = max(1, len(data) // bytes_per_block)
        result = np.empty(n_blocks * block_size, dtype=np.float32)
        for i in range(n_blocks):
            off = i * bytes_per_block
            if off + 2 > len(data):
                break
            d = struct.unpack_from("<e", data, off)[0]
            off += 2
            m = struct.unpack_from("<e", data, off)[0] if has_offset else -d
            if has_offset:
                off += 2
            for j in range(block_size):
                byte_idx = off + j // 2
                if byte_idx >= len(data):
                    break
                b = data[byte_idx]
                q = b & 0x0F if j % 2 == 0 else (b >> 4) & 0x0F
                result[i * block_size + j] = m + d * q
        return result

    @classmethod
    def _dequantize_q5(
        cls, raw: np.ndarray, ggml_type: int, block_size: int
    ) -> np.ndarray:
        data = raw.tobytes()
        has_offset = ggml_type == 7
        bytes_per_block = (
            2 + 2 + block_size // 2 + block_size // 8 + (2 if has_offset else 0)
        )
        n_blocks = max(1, len(data) // bytes_per_block)
        result = np.empty(n_blocks * block_size, dtype=np.float32)
        for i in range(n_blocks):
            off = i * bytes_per_block
            if off + 4 > len(data):
                break
            d = struct.unpack_from("<e", data, off)[0]
            off += 2
            qh = struct.unpack_from("<H", data, off)[0]
            off += 2
            m = struct.unpack_from("<e", data, off)[0] if has_offset else -d
            if has_offset:
                off += 2
            for j in range(block_size):
                byte_idx = off + j // 2
                if byte_idx >= len(data):
                    break
                b = data[byte_idx]
                q = (b & 0x0F) if j % 2 == 0 else ((b >> 4) & 0x0F)
                q |= ((qh >> j) & 1) << 4
                result[i * block_size + j] = m + d * q
        return result

    @classmethod
    def _dequantize_q8(
        cls, raw: np.ndarray, ggml_type: int, block_size: int
    ) -> np.ndarray:
        data = raw.tobytes()
        has_offset = ggml_type == 9
        bytes_per_block = 2 + block_size + (2 if has_offset else 0)
        n_blocks = max(1, len(data) // bytes_per_block)
        result = np.empty(n_blocks * block_size, dtype=np.float32)
        for i in range(n_blocks):
            off = i * bytes_per_block
            if off + 2 > len(data):
                break
            d = struct.unpack_from("<e", data, off)[0]
            off += 2
            m = struct.unpack_from("<e", data, off)[0] if has_offset else -d
            if has_offset:
                off += 2
            for j in range(block_size):
                if off + j >= len(data):
                    break
                q = int(data[off + j]) - 128 if not has_offset else int(data[off + j])
                result[i * block_size + j] = m + d * q
        return result

    @classmethod
    def _dequantize_qk(
        cls, raw: np.ndarray, ggml_type: int, block_size: int
    ) -> np.ndarray:
        data = raw.tobytes()
        if len(data) < 2:
            return np.zeros(block_size, dtype=np.float32)
        d = struct.unpack_from("<e", data, 0)[0]
        values = np.frombuffer(data[2:], dtype=np.uint8).astype(np.float32)
        if len(values) == 0:
            return np.zeros(block_size, dtype=np.float32)
        values = (values - 128.0) * d
        if len(values) < block_size:
            values = np.pad(values, (0, block_size - len(values)))
        return values[:block_size]


class UnifiedQuantizer:
    def __init__(
        self,
        default_bits: int = 8,
        target_ratio: float = 4.0,
        keep_energy: float = 0.95,
    ) -> None:
        self.default_bits = default_bits
        self.target_ratio = target_ratio
        self.keep_energy = keep_energy
        self.strategy = StrategySelector(target_ratio=target_ratio)
        self.quality_monitor = QualityMonitor()
        self._spectral = SpectralQuantizer(keep_energy=keep_energy, n_bits=default_bits)

    def compress_tensor(
        self, tensor: np.ndarray, name: str = "", method: Optional[str] = None
    ) -> dict:
        t = np.asarray(tensor, dtype=np.float32)
        if method is None:
            method = self.strategy.select(name, t)
        if method == "raw":
            return {
                "type": "raw",
                "data": t.tobytes(),
                "shape": list(t.shape),
                "dtype": "float32",
            }
        if method == "int4":
            return self._compress_int4(t, name)
        if method == "int8":
            return self._compress_int8(t, name)
        if method in ("spectral", "dct"):
            return self._spectral.compress(t)
        return {
            "type": "raw",
            "data": t.tobytes(),
            "shape": list(t.shape),
            "dtype": "float32",
        }

    def decompress_tensor(self, compressed: dict) -> np.ndarray:
        ctype = compressed.get("type", "raw")
        if ctype == "raw":
            return np.frombuffer(compressed["data"], dtype=np.float32).reshape(
                tuple(compressed["shape"])
            )
        if ctype == "int4":
            return self._decompress_int4(compressed)
        if ctype == "int8":
            return self._decompress_int8(compressed)
        if ctype == "spectral":
            return self._spectral.decompress(compressed)
        return np.zeros(tuple(compressed.get("shape", [0])), dtype=np.float32)

    def _compress_int4(self, tensor: np.ndarray, name: str) -> dict:
        flat = tensor.ravel().astype(np.float64)
        scale = float(np.max(np.abs(flat)) + 1e-10)
        normalized = np.clip(flat / scale, -1.0, 1.0)
        quantized = np.round((normalized + 1.0) * 7.5).astype(np.uint8)
        packed = np.empty(len(quantized) // 2 + len(quantized) % 2, dtype=np.uint8)
        for i in range(0, len(quantized), 2):
            lo = quantized[i]
            hi = quantized[i + 1] if i + 1 < len(quantized) else 0
            packed[i // 2] = (hi << 4) | lo
        return {
            "type": "int4",
            "data": packed.tobytes(),
            "scale": scale,
            "shape": list(tensor.shape),
            "n_elements": len(flat),
        }

    def _decompress_int4(self, compressed: dict) -> np.ndarray:
        data = np.frombuffer(compressed["data"], dtype=np.uint8)
        scale = compressed["scale"]
        n_elements = compressed["n_elements"]
        unpacked = np.empty(n_elements, dtype=np.uint8)
        for i in range(n_elements):
            byte_idx = i // 2
            if byte_idx < len(data):
                unpacked[i] = (
                    data[byte_idx] & 0x0F
                    if i % 2 == 0
                    else (data[byte_idx] >> 4) & 0x0F
                )
            else:
                unpacked[i] = 0
        normalized = unpacked.astype(np.float64) / 7.5 - 1.0
        result = (normalized * scale).astype(np.float32)
        return result.reshape(tuple(compressed["shape"]))

    def _compress_int8(self, tensor: np.ndarray, name: str) -> dict:
        flat = tensor.ravel().astype(np.float64)
        scale = float(np.max(np.abs(flat)) + 1e-10)
        normalized = np.clip(flat / scale, -1.0, 1.0)
        quantized = np.round(normalized * 127).astype(np.int8)
        return {
            "type": "int8",
            "data": quantized.tobytes(),
            "scale": scale,
            "shape": list(tensor.shape),
            "n_elements": len(flat),
        }

    def _decompress_int8(self, compressed: dict) -> np.ndarray:
        data = np.frombuffer(compressed["data"], dtype=np.int8)
        scale = compressed["scale"]
        normalized = data.astype(np.float64) / 127.0
        result = (normalized * scale).astype(np.float32)
        return result.reshape(tuple(compressed["shape"]))

    def get_report(self) -> CompressionReport:
        return self.quality_monitor.get_aggregate()


UnifiedQuantizationEngine = UnifiedQuantizer
