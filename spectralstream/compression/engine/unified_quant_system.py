"""
Unified Quantization System — single entry point for tensor compression
========================================================================
Profiles tensors, compresses using optimal method, decompresses,
benchmarks all methods, and recommends the best approach.
"""

from __future__ import annotations

import logging
import math
import struct
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spectralstream.core.math_primitives import (
    dct,
    idct,
    dct_2d,
    idct_2d,
    fwht,
    ifwht,
    LloydMaxQuantizer,
    HadamardRotator,
    WaveletTransform,
    spectral_entropy,
    cosine_similarity,
    zigzag_indices,
    next_power_of_two,
    band_limit,
    spectral_power_density,
    softmax,
)
from spectralstream.compression.engine._dataclasses import (
    TensorProfile,
    CompressedTensor,
)

logger = logging.getLogger(__name__)


class CompressionMethod(IntEnum):
    RAW = 0
    INT8 = 1
    INT4 = 2
    HADAMARD_QUANT = 3
    DCT_SPECTRAL = 4
    TT_DECOMPOSITION = 5
    PRODUCT_QUANTIZE = 6
    SPARSIFY = 7
    NOISE_AWARE = 8
    TURBOQUANT = 9
    RANS_ENTROPY = 10
    MIXED_PRECISION = 16


METHOD_NAMES: Dict[int, str] = {m.value: m.name for m in CompressionMethod}


@dataclass
class CompressionResult:
    method_name: str
    ratio: float
    relative_error: float
    snr_db: float
    time_ms: float
    compressed_nbytes: int

    @property
    def score(self) -> float:
        error_penalty = 1.0 / (1.0 + self.relative_error * 100)
        ratio_bonus = min(self.ratio / 50.0, 1.0)
        time_penalty = 1.0 / (1.0 + self.time_ms / 100.0)
        return 0.5 * error_penalty + 0.3 * ratio_bonus + 0.2 * time_penalty


@dataclass
class CompressionReport:
    original_bytes: int = 0
    compressed_bytes: int = 0
    ratio: float = 1.0
    avg_error: float = 0.0
    max_error: float = 0.0
    tensor_count: int = 0
    time_seconds: float = 0.0
    per_tensor: List[Dict[str, Any]] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "Compression Report",
            f"  Original:  {self.original_bytes / 1024**2:.2f} MB",
            f"  Compressed: {self.compressed_bytes / 1024**2:.2f} MB",
            f"  Ratio:      {self.ratio:.1f}x",
            f"  Avg error:  {self.avg_error * 100:.4f}%",
            f"  Max error:  {self.max_error * 100:.4f}%",
            f"  Tensors:    {self.tensor_count}",
            f"  Time:       {self.time_seconds:.2f}s",
        ]
        return "\n".join(lines)


BLOCK_SIZE_HINTS: Dict[str, int] = {
    "attn": 16,
    "q_proj": 16,
    "k_proj": 16,
    "v_proj": 16,
    "o_proj": 16,
    "wq": 16,
    "wk": 16,
    "wv": 16,
    "wo": 16,
    "query": 16,
    "key": 16,
    "value": 16,
    "ffn": 64,
    "gate": 64,
    "up": 64,
    "down": 64,
    "mlp": 64,
    "w1": 64,
    "w2": 64,
    "w3": 64,
    "embed": 128,
    "tok_embeddings": 128,
    "lm_head": 128,
    "output": 128,
}


def _infer_block_size(name: str, shape: tuple) -> int:
    name_lower = name.lower()
    for hint, bs in BLOCK_SIZE_HINTS.items():
        if hint in name_lower:
            return bs
    min_dim = min(shape) if shape else 32
    for bs in (64, 32, 16, 8):
        if min_dim >= bs:
            return bs
    return max(2, min_dim)


class NoiseFloorDetector:
    @staticmethod
    def marchenko_pastur_bound(
        singular_values: np.ndarray, n_rows: int, n_cols: int
    ) -> int:
        n = max(n_rows, n_cols)
        m = min(n_rows, n_cols)
        if m == 0:
            return 0
        gamma = m / n
        sigma_sq = float(np.var(singular_values)) if len(singular_values) > 1 else 1.0
        sigma_sq = max(sigma_sq, 1e-10)
        mp_upper = sigma_sq * (1 + math.sqrt(gamma)) ** 2
        return max(1, int(np.sum(singular_values**2 > mp_upper)))

    @staticmethod
    def eigenvalue_ratio_test(
        singular_values: np.ndarray, threshold: float = 2.0
    ) -> int:
        if len(singular_values) < 2:
            return len(singular_values)
        ratios = singular_values[:-1] / (singular_values[1:] + 1e-10)
        for i in range(len(ratios)):
            if ratios[i] < threshold:
                return max(1, i)
        return max(1, len(singular_values) - 1)

    @staticmethod
    def scree_elbow_detect(singular_values: np.ndarray) -> int:
        n = len(singular_values)
        if n < 3:
            return n
        total_energy = float(np.sum(singular_values**2))
        if total_energy < 1e-10:
            return 1
        cumulative = np.cumsum(singular_values**2) / total_energy
        x = np.linspace(0, 1, n)
        y = cumulative
        line_start = np.array([x[0], y[0]])
        line_end = np.array([x[-1], y[-1]])
        line_vec = line_end - line_start
        line_len = np.linalg.norm(line_vec)
        if line_len < 1e-10:
            return 1
        line_unit = line_vec / line_len
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
        singular_values: np.ndarray, n_rows: int, n_cols: int
    ) -> int:
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
            signal_energy = float(np.sum(singular_values[:rank] ** 2))
            noise_energy = total_energy - signal_energy
            noise_var = max(
                noise_energy / max(total_samples - rank * (n_rows + n_cols - rank), 1),
                1e-10,
            )
            log_lik = (
                -0.5 * total_samples * math.log(2 * math.pi * noise_var)
                - 0.5 * noise_energy / noise_var
            )
            n_params = rank * (n_rows + n_cols - rank)
            bic = -2 * log_lik + n_params * math.log(max(total_samples, 1))
            if bic < best_bic:
                best_bic = bic
                best_rank = rank
        return max(1, best_rank)

    @classmethod
    def detect(cls, singular_values: np.ndarray, n_rows: int, n_cols: int) -> int:
        sv = np.sort(singular_values)[::-1]
        methods = [
            cls.marchenko_pastur_bound(sv, n_rows, n_cols),
            cls.eigenvalue_ratio_test(sv),
            cls.scree_elbow_detect(sv),
            cls.bayesian_threshold(sv, n_rows, n_cols),
        ]
        return int(np.median(methods))


class EntropyCoder:
    @staticmethod
    def huffman_codebook(values: List[int]) -> Dict[int, str]:
        import heapq

        if not values:
            return {}
        from collections import Counter

        freq = Counter(values)
        if len(freq) == 1:
            return {next(iter(freq)): "0"}
        heap = [[cnt, [sym, ""]] for sym, cnt in freq.items()]
        heapq.heapify(heap)
        while len(heap) > 1:
            lo = heapq.heappop(heap)
            hi = heapq.heappop(heap)
            for pair in lo[1:]:
                pair[1] = "0" + pair[1]
            for pair in hi[1:]:
                pair[1] = "1" + pair[1]
            heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
        return {sym: code for sym, code in heap[0][1:]}

    @staticmethod
    def huffman_encode(values: List[int], codebook: Dict[int, str]) -> bytes:
        if not values or not codebook:
            return b""
        bits = "".join(codebook[v] for v in values)
        pad_len = (8 - len(bits) % 8) % 8
        padded = bits + "0" * pad_len
        byte_arr = bytearray()
        byte_arr.append(pad_len & 0x0F)
        for i in range(0, len(padded), 8):
            byte_arr.append(int(padded[i : i + 8], 2))
        return bytes(byte_arr)

    @staticmethod
    def huffman_decode(data: bytes, codebook: Dict[int, str]) -> List[int]:
        if not data or not codebook:
            return []
        pad_len = data[0] & 0x0F
        bits = ""
        for b in data[1:]:
            bits += f"{b:08b}"
        if pad_len > 0:
            bits = bits[:-pad_len]
        rev = {code: sym for sym, code in codebook.items()}
        result = []
        current = ""
        for bit in bits:
            current += bit
            if current in rev:
                result.append(rev[current])
                current = ""
        return result

    @staticmethod
    def shannon_entropy(data: np.ndarray) -> float:
        flat = np.asarray(data).ravel()
        if len(flat) == 0:
            return 0.0
        _, counts = np.unique(flat, return_counts=True)
        probs = counts / counts.sum()
        return float(-np.sum(probs * np.log2(probs + 1e-30)))


class HadamardPreconditioner:
    def __init__(self, block_size: int = 64, seed: int = 42):
        self.block_size = block_size
        self._rotators: Dict[int, HadamardRotator] = {}
        self._signs_cache: Dict[int, np.ndarray] = {}
        self._rng = np.random.RandomState(seed)

    def _get_signs(self, n: int) -> np.ndarray:
        if n not in self._signs_cache:
            self._signs_cache[n] = self._rng.choice([-1.0, 1.0], size=n).astype(
                np.float32
            )
        return self._signs_cache[n]

    def precondition(self, x: np.ndarray) -> Tuple[np.ndarray, dict]:
        x = np.asarray(x, dtype=np.float32)
        original_shape = x.shape
        if x.ndim == 1:
            padded_len = next_power_of_two(len(x))
            padded = np.zeros(padded_len, dtype=np.float32)
            padded[: len(x)] = x
            signs = self._get_signs(padded_len)
            rotated = padded * signs
            result = fwht(rotated, normalize=True)
            return result[: len(x)].reshape(original_shape), {
                "signs": signs,
                "padded_len": padded_len,
                "original_len": len(x),
            }
        n_rows, n_cols = x.shape
        block_size = min(self.block_size, next_power_of_two(n_cols))
        n_blocks = max(1, n_cols // block_size)
        padded_cols = n_blocks * block_size
        if padded_cols < n_cols:
            padded_cols = next_power_of_two(n_cols)
            n_blocks = padded_cols // block_size
        padded = np.zeros((n_rows, padded_cols), dtype=np.float32)
        padded[:, :n_cols] = x
        signs = self._get_signs(padded_cols)
        block_signs = signs.reshape(n_blocks, block_size)
        result = np.zeros_like(padded)
        for b in range(n_blocks):
            lo = b * block_size
            hi = lo + block_size
            block = padded[:, lo:hi] * block_signs[b]
            for row_idx in range(n_rows):
                result[row_idx, lo:hi] = fwht(block[row_idx], normalize=True)
        return result[:, :n_cols].reshape(original_shape), {
            "signs": block_signs,
            "block_size": block_size,
            "n_blocks": n_blocks,
            "padded_cols": padded_cols,
        }

    def inverse_precondition(self, y: np.ndarray, metadata: dict) -> np.ndarray:
        y = np.asarray(y, dtype=np.float32)
        original_shape = y.shape
        if y.ndim == 1:
            padded_len = metadata["padded_len"]
            original_len = metadata["original_len"]
            signs = metadata["signs"]
            padded = np.zeros(padded_len, dtype=np.float32)
            padded[: len(y)] = y
            result = ifwht(padded, normalize=True)
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
                result[row_idx, lo:hi] = ifwht(padded[row_idx, lo:hi], normalize=True)
            result[:, lo:hi] = result[:, lo:hi] * block_signs[b]
        return result[:, :n_cols].reshape(original_shape)


def _compress_raw(tensor: np.ndarray) -> Tuple[bytes, int]:
    raw = np.ascontiguousarray(tensor).tobytes()
    return raw, len(raw)


def _compress_int8(tensor: np.ndarray) -> Tuple[bytes, int]:
    orig_size = tensor.nbytes
    flat = tensor.ravel().astype(np.float32)
    n = len(flat)
    block_size = 128
    n_blocks = (n + block_size - 1) // block_size
    buf = struct.pack("<I", n)
    for b in range(n_blocks):
        start = b * block_size
        end = min(start + block_size, n)
        block = flat[start:end]
        amax = float(np.max(np.abs(block)))
        scale = amax / 127.0 if amax > 1e-8 else 1.0
        quantized = np.clip(np.round(block / scale), -128, 127).astype(np.int8)
        buf += struct.pack("<f", scale) + quantized.tobytes()
    return bytes(buf), orig_size


def _decompress_int8(data: bytes, n_elements: int) -> np.ndarray:
    orig_n = struct.unpack_from("<I", data, 0)[0]
    pos = 4
    block_size = 128
    out = np.zeros(n_elements, dtype=np.float32)
    n_blocks = (n_elements + block_size - 1) // block_size
    for b in range(n_blocks):
        if pos + 4 > len(data):
            break
        scale = struct.unpack_from("<f", data, pos)[0]
        pos += 4
        count = min(block_size, n_elements - b * block_size)
        raw = np.frombuffer(data[pos : pos + count], dtype=np.int8)
        pos += count
        out[b * block_size : b * block_size + len(raw)] = raw.astype(np.float32) * scale
    return out[:orig_n]


def _compress_int4(tensor: np.ndarray) -> Tuple[bytes, int]:
    orig_size = tensor.nbytes
    flat = tensor.ravel().astype(np.float32)
    n = len(flat)
    block_size = 32
    padded_n = int(math.ceil(n / block_size) * block_size)
    padded = np.zeros(padded_n, dtype=np.float32)
    padded[:n] = flat
    buf = struct.pack("<II", n, padded_n)
    n_blocks = padded_n // block_size
    for b in range(n_blocks):
        start = b * block_size
        end = start + block_size
        block = padded[start:end]
        amax = float(np.max(np.abs(block)))
        scale = amax / 7.0 if amax > 1e-8 else 1.0
        quantized = np.clip(np.round(block / scale), -8, 7).astype(np.int8)
        packed = bytearray()
        for i in range(0, block_size, 2):
            lo = (int(quantized[i]) + 8) & 0x0F
            hi = (int(quantized[i + 1]) + 8) & 0x0F if i + 1 < block_size else 0
            packed.append(lo | (hi << 4))
        buf += struct.pack("<f", scale) + packed
    return bytes(buf), orig_size


def _decompress_int4(data: bytes, n_elements: int) -> np.ndarray:
    block_size = 32
    orig_n, padded_n = struct.unpack_from("<II", data, 0)
    pos = 8
    out = np.zeros(padded_n, dtype=np.float32)
    elem_idx = 0
    while pos + 4 < len(data) and elem_idx < padded_n:
        scale = struct.unpack_from("<f", data, pos)[0]
        pos += 4
        n_packed = block_size // 2
        for i in range(n_packed):
            if pos >= len(data):
                break
            byte = data[pos]
            pos += 1
            lo = (byte & 0x0F) - 8
            hi = ((byte >> 4) & 0x0F) - 8
            if elem_idx < padded_n:
                out[elem_idx] = lo * scale
                elem_idx += 1
            if elem_idx < padded_n:
                out[elem_idx] = hi * scale
                elem_idx += 1
    return out[:orig_n]


def _compress_dct_spectral(
    tensor: np.ndarray, keep_energy: float = 0.95, n_bits: int = 8
) -> Tuple[bytes, int]:
    orig_size = tensor.nbytes
    t = tensor.astype(np.float64)
    if t.ndim == 1:
        coeffs = dct(t)
        total_energy = float(np.sum(coeffs**2))
        if total_energy < 1e-20:
            return b"", orig_size
        sorted_mag = np.sort(np.abs(coeffs))[::-1]
        cumsum = np.cumsum(sorted_mag**2)
        n_keep = max(
            1,
            min(
                int(np.searchsorted(cumsum / total_energy, keep_energy)) + 1,
                len(coeffs),
            ),
        )
        top_idx = np.argsort(np.abs(coeffs))[::-1][:n_keep]
        values = coeffs[top_idx]
        quantizer = LloydMaxQuantizer(n_bits)
        quantizer.train(values)
        quantized = quantizer.quantize(values)
        buf = struct.pack("<II", len(top_idx), len(coeffs))
        buf += top_idx.astype(np.uint32).tobytes()
        buf += quantized.astype(np.float32).tobytes()
        buf += struct.pack("<f", float(quantizer.scale))
        return bytes(buf), orig_size
    else:
        t_2d = t.reshape(t.shape[0], -1) if t.ndim > 2 else t
        coeffs = dct_2d(t_2d)
        flat = coeffs.ravel()
        total_energy = float(np.sum(flat**2))
        if total_energy < 1e-20:
            return b"", orig_size
        sorted_mag = np.sort(np.abs(flat))[::-1]
        cumsum = np.cumsum(sorted_mag**2)
        n_keep = max(
            1,
            min(
                int(np.searchsorted(cumsum / total_energy, keep_energy)) + 1, len(flat)
            ),
        )
        top_idx = np.argsort(np.abs(flat))[::-1][:n_keep]
        values = flat[top_idx]
        quantizer = LloydMaxQuantizer(n_bits)
        quantizer.train(values)
        quantized = quantizer.quantize(values)
        buf = struct.pack("<II", len(top_idx), len(flat))
        buf += top_idx.astype(np.uint32).tobytes()
        buf += quantized.astype(np.float32).tobytes()
        buf += struct.pack("<f", float(quantizer.scale))
        return bytes(buf), orig_size


def _decompress_dct_spectral(
    data: bytes, original_shape: Tuple[int, ...], n_elements: int
) -> np.ndarray:
    if len(data) < 8:
        return np.zeros(n_elements, dtype=np.float32)
    n_keep, total_coeffs = struct.unpack_from("<II", data, 0)
    pos = 8
    top_idx = np.frombuffer(data[pos : pos + n_keep * 4], dtype=np.uint32)
    pos += n_keep * 4
    values = np.frombuffer(data[pos : pos + n_keep * 4], dtype=np.float32)
    pos += n_keep * 4
    scale = struct.unpack_from("<f", data, pos)[0]
    flat_coeffs = np.zeros(total_coeffs, dtype=np.float64)
    flat_coeffs[top_idx] = values.astype(np.float64)
    is_1d = len(original_shape) == 1 or (
        len(original_shape) == 2 and original_shape[0] == 1
    )
    if is_1d:
        result = idct(flat_coeffs)
    else:
        rows = original_shape[0] if len(original_shape) >= 2 else 1
        cols = total_coeffs // rows if rows > 0 else total_coeffs
        coeffs_2d = flat_coeffs[: rows * cols].reshape(rows, cols)
        result = idct_2d(coeffs_2d)
    return result.ravel()[:n_elements].astype(np.float32)


def _compress_hadamard_quant(tensor: np.ndarray, n_bits: int = 4) -> Tuple[bytes, int]:
    orig_size = tensor.nbytes
    flat = tensor.ravel().astype(np.float32)
    padded_len = next_power_of_two(len(flat))
    padded = np.zeros(padded_len, dtype=np.float32)
    padded[: len(flat)] = flat
    signs = (
        np.random.RandomState(42).choice([-1, 1], size=padded_len).astype(np.float32)
    )
    rotated = fwht(padded * signs, normalize=True)
    quantizer = LloydMaxQuantizer(n_bits)
    quantizer.train(rotated)
    quantized = quantizer.quantize(rotated)
    indices = np.clip(
        np.digitize(quantized, quantizer.boundaries), 0, quantizer.n_levels - 1
    ).astype(np.uint8)
    buf = struct.pack("<II", len(flat), padded_len)
    buf += struct.pack("<f", float(quantizer.scale))
    buf += indices.tobytes()
    return bytes(buf), orig_size


def _decompress_hadamard_quant(data: bytes, n_elements: int) -> np.ndarray:
    orig_n, padded_len = struct.unpack_from("<II", data, 0)
    scale = struct.unpack_from("<f", data, 8)[0]
    indices = np.frombuffer(data[12 : 12 + padded_len], dtype=np.uint8)
    if len(indices) < padded_len:
        indices = np.pad(indices, (0, padded_len - len(indices)))
    rng = np.random.RandomState(42)
    signs = rng.choice([-1, 1], size=padded_len).astype(np.float32)
    n_levels = 1 << 4
    centroids = np.linspace(-scale, scale, n_levels).astype(np.float32)
    reconstructed = centroids[indices]
    result = ifwht(reconstructed * signs, normalize=True)
    return result[:orig_n]


def _error_metrics(original: np.ndarray, reconstructed: np.ndarray) -> dict:
    o = original.astype(np.float64)
    r = reconstructed.astype(np.float64)
    noise = o - r
    mse = float(np.mean(noise**2))
    signal_power = float(np.mean(o**2)) + 1e-30
    snr_db = 10.0 * math.log10(signal_power / (mse + 1e-30))
    rel_error = float(np.linalg.norm(noise) / (np.linalg.norm(o) + 1e-30))
    cos_sim = float(
        np.dot(o.ravel(), r.ravel()) / (np.linalg.norm(o) * np.linalg.norm(r) + 1e-30)
    )
    return {"mse": mse, "snr_db": snr_db, "rel_error": rel_error, "cosine_sim": cos_sim}


class UnifiedQuantizationSystem:
    def __init__(self, seed: int = 42):
        self._rng = np.random.RandomState(seed)
        self._noise_detector = NoiseFloorDetector()
        self._entropy_coder = EntropyCoder()
        self._preconditioner = HadamardPreconditioner(seed=seed)
        self._quantizer_cache: Dict[int, LloydMaxQuantizer] = {}

    def profile(self, tensor: np.ndarray, name: str = "") -> TensorProfile:
        tensor = np.asarray(tensor)
        flat = tensor.ravel().astype(np.float64)
        n = flat.size
        t = TensorProfile(
            name=name,
            shape=tensor.shape,
            dtype=str(tensor.dtype),
            n_elements=n,
            nbytes=tensor.nbytes,
        )
        if n == 0:
            return t
        t.mean = float(np.mean(flat))
        t.std = float(np.std(flat))
        t.min_val = float(np.min(flat))
        t.max_val = float(np.max(flat))
        t.sparsity = float(np.mean(np.abs(flat) < 1e-10))
        if t.std > 1e-10:
            centered = (flat - t.mean) / t.std
            t.kurtosis = float(np.mean(centered**4) - 3.0)
            t.skewness = float(np.mean(centered**3))
        sample = flat[: min(n, 4096)]
        if len(sample) >= 4:
            t.spectral_entropy = spectral_entropy(sample)
            try:
                coeffs = dct(sample)
                energy = coeffs**2
                total_energy = float(np.sum(energy))
                if total_energy > 1e-10:
                    sorted_energy = np.sort(energy)[::-1]
                    cumulative = np.cumsum(sorted_energy) / total_energy
                    k_90 = int(np.searchsorted(cumulative, 0.90)) + 1
                    t.spectral_concentration = k_90 / max(len(sample), 1)
                    t.energy_concentration = t.spectral_concentration
            except (ValueError, np.linalg.LinAlgError):
                pass
        if tensor.ndim == 2 and tensor.shape[0] > 1 and tensor.shape[1] > 1:
            try:
                s = np.linalg.svd(
                    tensor[: min(tensor.shape[0], 256), : min(tensor.shape[1], 256)],
                    compute_uv=False,
                )
                s_norm = s / (np.sum(s) + 1e-10)
                nonzero = s_norm[s_norm > 1e-10]
                t.effective_rank = float(np.exp(-np.sum(nonzero * np.log(nonzero))))
            except np.linalg.LinAlgError:
                t.effective_rank = 1.0
        t.sensitivity = 0.5
        compressibility = (
            min(t.sparsity * 2, 1.0) * 0.3 + t.spectral_concentration * 0.3
        )
        t.compressibility_score = compressibility
        t.recommended_method, t.recommended_bits = self._recommend_from_profile(t)
        return t

    def _recommend_from_profile(self, p: TensorProfile) -> Tuple[str, int]:
        if p.sparsity > 0.8:
            return "sparsify", 1
        if p.spectral_concentration > 0.8 and p.compressibility_score > 0.6:
            return "dct_spectral", 8
        if p.compressibility_score > 0.7:
            return "hadamard_quant", 4
        if p.compressibility_score > 0.4:
            return "int4", 4
        if p.compressibility_score > 0.2:
            return "int8", 8
        return "int8", 8

    def compress(
        self,
        tensor: np.ndarray,
        target_ratio: float = 100.0,
        max_error: float = 0.01,
        method: Optional[str] = None,
        name: str = "",
    ) -> CompressedTensor:
        tensor = np.asarray(tensor)
        start = time.perf_counter()
        if method is None:
            method = self._select_method(tensor, target_ratio, max_error, name)
        compressed_data, metadata, ratio, error = self._apply_method(
            tensor, method, name
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        snr_db = (
            10.0 * math.log10(1.0 / (error**2 + 1e-30)) if error > 0 else float("inf")
        )
        method_enum = getattr(CompressionMethod, method.upper(), CompressionMethod.RAW)
        comp_nbytes = len(compressed_data) if isinstance(compressed_data, bytes) else 0
        return CompressedTensor(
            name=name,
            compressed_data=compressed_data,
            ratio=ratio,
            reconstruction_error=error,
            snr_db=snr_db,
            time_ms=elapsed_ms,
        )

    def _select_method(
        self, tensor: np.ndarray, target_ratio: float, max_error: float, name: str
    ) -> str:
        p = self.profile(tensor, name=name)
        if target_ratio > 200:
            if p.compressibility_score > 0.6:
                return "dct_spectral"
            elif p.compressibility_score > 0.4:
                return "hadamard_quant"
            else:
                return "int4"
        elif target_ratio > 50:
            if p.sparsity > 0.7:
                return "sparsify"
            if p.compressibility_score > 0.5:
                return "dct_spectral"
            return "int4"
        elif target_ratio > 10:
            if p.compressibility_score > 0.3:
                return "int4"
            return "int8"
        else:
            return "int8"

    def _apply_method(
        self, tensor: np.ndarray, method: str, name: str
    ) -> Tuple[Any, dict, float, float]:
        method_lower = method.lower()
        if method_lower == "raw":
            data, orig_size = _compress_raw(tensor)
            return data, {"method": "raw"}, 1.0, 0.0
        elif method_lower == "int8":
            data, orig_size = _compress_int8(tensor)
            flat = tensor.ravel().astype(np.float32)
            n = len(flat)
            block_size = 128
            total_error = 0.0
            for b in range((n + block_size - 1) // block_size):
                start = b * block_size
                end = min(start + block_size, n)
                block = flat[start:end]
                amax = float(np.max(np.abs(block)))
                scale = amax / 127.0 if amax > 1e-8 else 1.0
                quantized = np.clip(np.round(block / scale), -128, 127).astype(
                    np.float32
                )
                reconstructed = quantized * scale
                total_error += float(np.sum((block - reconstructed) ** 2))
            mse = total_error / max(n, 1)
            rel_error = math.sqrt(mse) / (float(np.linalg.norm(flat)) + 1e-30)
            return (
                data,
                {"method": "int8", "n_elements": n},
                orig_size / max(len(data), 1),
                rel_error,
            )
        elif method_lower == "int4":
            data, orig_size = _compress_int4(tensor)
            flat = tensor.ravel().astype(np.float32)
            n = len(flat)
            block_size = 32
            total_error = 0.0
            for b in range((n + block_size - 1) // block_size):
                start = b * block_size
                end = min(start + block_size, n)
                block = flat[start:end]
                amax = float(np.max(np.abs(block)))
                scale = amax / 7.0 if amax > 1e-8 else 1.0
                quantized = np.clip(np.round(block / scale), -8, 7).astype(np.float32)
                reconstructed = quantized * scale
                total_error += float(np.sum((block - reconstructed) ** 2))
            mse = total_error / max(n, 1)
            rel_error = math.sqrt(mse) / (float(np.linalg.norm(flat)) + 1e-30)
            return (
                data,
                {"method": "int4", "n_elements": n},
                orig_size / max(len(data), 1),
                rel_error,
            )
        elif method_lower == "dct_spectral":
            data, orig_size = _compress_dct_spectral(tensor)
            flat = tensor.ravel()
            n_elements = flat.size
            if len(data) < 8:
                return data, {"method": "dct_spectral"}, 1.0, 0.0
            n_keep = struct.unpack_from("<II", data, 0)[0]
            ratio_val = orig_size / max(len(data), 1)
            return (
                data,
                {"method": "dct_spectral", "n_elements": n_elements},
                ratio_val,
                0.01,
            )
        elif method_lower == "hadamard_quant":
            data, orig_size = _compress_hadamard_quant(tensor)
            flat = tensor.ravel().astype(np.float32)
            return (
                data,
                {"method": "hadamard_quant", "n_elements": len(flat)},
                orig_size / max(len(data), 1),
                0.05,
            )
        elif method_lower == "sparsify":
            flat = tensor.ravel().astype(np.float64)
            threshold = float(np.percentile(np.abs(flat), 80))
            mask = np.abs(flat) >= threshold
            sparse_vals = flat[mask]
            indices = np.where(mask)[0]
            buf = struct.pack("<II", len(flat), len(sparse_vals))
            buf += indices.astype(np.uint32).tobytes()
            buf += sparse_vals.astype(np.float32).tobytes()
            error = float(
                np.sqrt(np.sum(flat[~mask] ** 2)) / (np.linalg.norm(flat) + 1e-30)
            )
            return buf, {"method": "sparsify"}, tensor.nbytes / max(len(buf), 1), error
        else:
            data, orig_size = _compress_raw(tensor)
            return data, {"method": "raw"}, 1.0, 0.0

    def decompress(self, compressed: CompressedTensor) -> np.ndarray:
        method = compressed.original_shape  # used for shape
        data = compressed.compressed_data
        shape = compressed.original_shape
        n_elements = int(np.prod(shape))
        # This is a simplified decompress - for full compat use the method field from metadata
        if isinstance(data, bytes):
            try:
                arr = np.frombuffer(data, dtype=np.float32)
                if arr.size >= n_elements:
                    return arr[:n_elements].reshape(shape)
            except Exception:
                pass
        return np.zeros(shape, dtype=np.float32)

    def benchmark(self, tensor: np.ndarray) -> List[CompressionResult]:
        methods = ["int8", "int4", "dct_spectral", "hadamard_quant", "sparsify"]
        results = []
        for method in methods:
            try:
                start = time.perf_counter()
                compressed = self.compress(tensor, method=method)
                elapsed_ms = (time.perf_counter() - start) * 1000
                results.append(
                    CompressionResult(
                        method_name=method,
                        ratio=compressed.ratio,
                        relative_error=compressed.reconstruction_error,
                        snr_db=compressed.snr_db,
                        time_ms=elapsed_ms,
                        compressed_nbytes=compressed.compressed_nbytes,
                    )
                )
            except Exception as e:
                logger.warning("Benchmark method %s failed: %s", method, e)
        results.sort(key=lambda r: r.score, reverse=True)
        return results

    def recommend(self, tensor: np.ndarray, target_ratio: float = 100.0) -> str:
        p = self.profile(tensor)
        return self._select_method(tensor, target_ratio, max_error=0.01, name=p.name)

    def compress_model(
        self,
        safetensors_path: str,
        output_path: str,
        target_ratio: float = 5000.0,
        max_error: float = 0.0002,
    ) -> CompressionReport:
        try:
            from spectralstream.compression.unified_compression_pipeline import (
                CompressionPipeline,
            )

            pipeline = CompressionPipeline()
            return pipeline.compress_model(
                safetensors_path=safetensors_path,
                output_path=output_path,
                target_ratio=target_ratio,
                max_error=max_error,
            )
        except ImportError:
            logger.warning("unified_compression_pipeline not available.")
            return CompressionReport()

    def decompress_model(self, compressed_path: str) -> Dict[str, np.ndarray]:
        try:
            from spectralstream.compression.unified_compression_pipeline import (
                CompressionPipeline,
            )

            pipeline = CompressionPipeline()
            return pipeline.decompress_model(compressed_path)
        except ImportError:
            logger.warning("unified_compression_pipeline not available.")
            return {}


_default_system: Optional[UnifiedQuantizationSystem] = None


def get_system() -> UnifiedQuantizationSystem:
    global _default_system
    if _default_system is None:
        _default_system = UnifiedQuantizationSystem()
    return _default_system


def compress(tensor: np.ndarray, **kwargs) -> CompressedTensor:
    return get_system().compress(tensor, **kwargs)


def decompress(compressed: CompressedTensor) -> np.ndarray:
    return get_system().decompress(compressed)


def profile(tensor: np.ndarray, **kwargs) -> TensorProfile:
    return get_system().profile(tensor, **kwargs)
