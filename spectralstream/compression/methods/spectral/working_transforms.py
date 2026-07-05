"""
Working Transform-Domain Compression Methods
=============================================
Transform-domain techniques that actually work on real neural network weights.

Key insight: NN weights have high effective rank and don't concentrate energy
in low frequencies, so traditional DCT/Wavelet truncation fails. Instead:

1. Apply transform to decorrelate/distribute weight energy more uniformly
2. Quantize ALL transform coefficients (no truncation — that loses info)
3. The transform makes INT8/INT4/INT2 quantization more effective
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spectralstream.core.math_primitives import (
    fwht,
    next_power_of_two,
    WaveletTransform,
)

logger = logging.getLogger(__name__)

_DCT_CACHE: dict[int, np.ndarray] = {}


def _get_dct_matrix(n: int) -> np.ndarray:
    if n not in _DCT_CACHE:
        C = np.zeros((n, n), dtype=np.float64)
        C[0, :] = 1.0 / np.sqrt(n)
        s = np.sqrt(2.0 / n)
        k = np.arange(1, n, dtype=np.float64)[:, None]
        i = np.arange(n, dtype=np.float64)[None, :]
        C[1:, :] = s * np.cos(np.pi * k * (i + 0.5) / n)
        _DCT_CACHE[n] = C
    return _DCT_CACHE[n]


def _haar_decompose(x: np.ndarray, n_levels: int) -> Tuple[np.ndarray, List[int]]:
    levels = []
    current = x.copy()
    for _ in range(n_levels):
        if len(current) < 2:
            break
        approx, detail = WaveletTransform.haar_forward_1d(current)
        levels.append(detail)
        current = approx
    levels.append(current)
    sizes = [len(l) for l in reversed(levels)]
    return np.concatenate(list(reversed(levels))), sizes


def _haar_reconstruct(
    coeffs: np.ndarray, sizes: List[int], orig_len: int
) -> np.ndarray:
    idx = 0
    current = coeffs[idx : idx + sizes[0]].copy()
    idx += sizes[0]
    for sz in sizes[1:]:
        detail = coeffs[idx : idx + sz]
        idx += sz
        current = WaveletTransform.haar_inverse_1d(current[: len(detail)], detail)
    return current[:orig_len]


def quantize_int8(block: np.ndarray) -> Tuple[np.ndarray, float]:
    amax = max(float(np.max(np.abs(block))), 1e-10)
    scale = amax / 127.0
    q = np.clip(np.round(block / scale), -128, 127).astype(np.int8)
    return q, float(scale)


def dequantize_int8(q: np.ndarray, scale: float) -> np.ndarray:
    return q.astype(np.float64) * scale


def quantize_int4(block: np.ndarray) -> Tuple[np.ndarray, float]:
    amax = max(float(np.max(np.abs(block))), 1e-10)
    scale = amax / 7.0
    q = np.clip(np.round(block / scale), -8, 7).astype(np.int8)
    return q, float(scale)


def quantize_int2(block: np.ndarray) -> Tuple[np.ndarray, float]:
    amax = max(float(np.max(np.abs(block))), 1e-10)
    scale = amax / 3.0
    q = np.clip(np.round(block / scale), -4, 3).astype(np.int8)
    return q, float(scale)


def pack_int4(values: np.ndarray) -> np.ndarray:
    v = values.astype(np.int8)
    v_unsigned = (v + 8).astype(np.uint8)
    n = len(v_unsigned)
    n_packed = (n + 1) // 2
    packed = np.zeros(n_packed, dtype=np.uint8)
    for i in range(0, n - 1, 2):
        packed[i // 2] = v_unsigned[i] | (v_unsigned[i + 1] << 4)
    if n % 2 == 1:
        packed[-1] = v_unsigned[-1]
    return packed


def unpack_int4(packed: np.ndarray, n: int) -> np.ndarray:
    result = np.zeros(n, dtype=np.uint8)
    for i in range(0, n - 1, 2):
        byte = packed[i // 2]
        result[i] = byte & 0x0F
        result[i + 1] = (byte >> 4) & 0x0F
    if n % 2 == 1:
        result[-1] = packed[-1] & 0x0F
    return result.astype(np.int8) - 8


def pack_int2(values: np.ndarray) -> np.ndarray:
    v = values.astype(np.int8)
    v_unsigned = (v + 4).astype(np.uint8)
    n = len(v_unsigned)
    n_packed = (n + 3) // 4
    packed = np.zeros(n_packed, dtype=np.uint8)
    for i in range(n):
        packed[i // 4] |= v_unsigned[i] << (2 * (i % 4))
    return packed


def unpack_int2(packed: np.ndarray, n: int) -> np.ndarray:
    result = np.zeros(n, dtype=np.uint8)
    for i in range(n):
        result[i] = (packed[i // 4] >> (2 * (i % 4))) & 0x3
    return result.astype(np.int8) - 4


def compute_metrics(orig: np.ndarray, recon: np.ndarray) -> Dict[str, float]:
    o = orig.astype(np.float64).ravel()
    r = recon.astype(np.float64).ravel()
    mse = float(np.mean((o - r) ** 2))
    sp = float(np.sum(o**2))
    np_ = float(np.sum((o - r) ** 2))
    snr = 10.0 * np.log10(sp / (np_ + 1e-30))
    cos_sim = float(np.dot(o, r) / (np.linalg.norm(o) * np.linalg.norm(r) + 1e-30))
    rel_error = float(np.mean(np.abs(o - r) / (np.abs(o) + 1e-10)))
    return {
        "mse": mse,
        "snr_db": float(snr),
        "cosine_similarity": cos_sim,
        "rel_error": rel_error,
    }


class WorkingTransformCompressor(ABC):
    METHOD_NAME: str = "base"

    def __init__(self, block_size: int = 128):
        self.block_size = block_size

    def _to_blocks(self, tensor: np.ndarray) -> Tuple[np.ndarray, int]:
        flat = tensor.astype(np.float64).ravel()
        n = len(flat)
        pn = int(np.ceil(n / self.block_size) * self.block_size)
        padded = np.zeros(pn, dtype=np.float64)
        padded[:n] = flat
        return padded.reshape(-1, self.block_size), n

    @abstractmethod
    def compress(
        self, tensor: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]: ...

    @abstractmethod
    def decompress(
        self, data: Dict[str, Any], metadata: Dict[str, Any]
    ) -> np.ndarray: ...

    def evaluate(self, tensor: np.ndarray, **kwargs) -> Dict[str, Any]:
        orig_bytes = tensor.size * 4
        t0 = time.perf_counter()
        data, meta = self.compress(tensor, **kwargs)
        t_comp = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        recon = self.decompress(data, meta)
        t_decomp = (time.perf_counter() - t0) * 1000
        comp_bytes = self._compressed_size(data)
        metrics = compute_metrics(tensor, recon)
        metrics["compression_ratio"] = orig_bytes / max(comp_bytes, 1)
        metrics["compress_ms"] = t_comp
        metrics["decompress_ms"] = t_decomp
        return metrics

    @abstractmethod
    def _compressed_size(self, data: Dict[str, Any]) -> int: ...


class PlainBlockInt8(WorkingTransformCompressor):
    METHOD_NAME = "plain_block_int8"

    def __init__(self, block_size: int = 128):
        super().__init__(block_size)

    def compress(self, tensor, **kwargs):
        orig_shape = tensor.shape
        blocks, n_orig = self._to_blocks(tensor)
        n_blocks, bs = blocks.shape
        quantized = np.zeros((n_blocks, bs), dtype=np.int8)
        scales = np.zeros(n_blocks, dtype=np.float32)
        for i in range(n_blocks):
            q, s = quantize_int8(blocks[i])
            quantized[i] = q
            scales[i] = s
        data = {
            "quantized": quantized,
            "scales": scales,
            "n_orig": np.int32(n_orig),
            "block_size": np.int32(bs),
        }
        meta = {"orig_shape": orig_shape, "method": self.METHOD_NAME}
        return data, meta

    def decompress(self, data, metadata):
        flat = (
            data["quantized"].astype(np.float64) * data["scales"][:, np.newaxis]
        ).ravel()
        return (
            flat[: int(data["n_orig"])]
            .reshape(metadata["orig_shape"])
            .astype(np.float32)
        )

    def _compressed_size(self, data):
        return data["quantized"].size + data["scales"].size * 4


class HadamardBlockQuantize(WorkingTransformCompressor):
    METHOD_NAME = "hadamard_block_int8"

    def __init__(self, block_size: int = 128):
        super().__init__(block_size)

    def compress(self, tensor, **kwargs):
        orig_shape = tensor.shape
        blocks, n_orig = self._to_blocks(tensor)
        n_blocks, bs = blocks.shape
        h_blocks = fwht(blocks, normalize=True)
        quantized = np.zeros((n_blocks, bs), dtype=np.int8)
        scales = np.zeros(n_blocks, dtype=np.float32)
        for i in range(n_blocks):
            q, s = quantize_int8(h_blocks[i])
            quantized[i] = q
            scales[i] = s
        data = {
            "quantized": quantized,
            "scales": scales,
            "n_orig": np.int32(n_orig),
            "block_size": np.int32(bs),
        }
        meta = {"orig_shape": orig_shape, "method": self.METHOD_NAME}
        return data, meta

    def decompress(self, data, metadata):
        quantized, scales = data["quantized"], data["scales"]
        n_blocks, bs = quantized.shape
        h = quantized.astype(np.float64) * scales[:, np.newaxis]
        recon = fwht(h, normalize=True)
        flat = recon.ravel()
        return (
            flat[: int(data["n_orig"])]
            .reshape(metadata["orig_shape"])
            .astype(np.float32)
        )

    def _compressed_size(self, data):
        return data["quantized"].size + data["scales"].size * 4


class DCTBlockQuantize(WorkingTransformCompressor):
    METHOD_NAME = "dct_block_int8"

    def __init__(self, block_size: int = 128):
        super().__init__(block_size)

    def compress(self, tensor, **kwargs):
        orig_shape = tensor.shape
        blocks, n_orig = self._to_blocks(tensor)
        n_blocks, bs = blocks.shape
        C = _get_dct_matrix(bs)
        dct_blocks = (C @ blocks.T).T
        quantized = np.zeros((n_blocks, bs), dtype=np.int8)
        scales = np.zeros(n_blocks, dtype=np.float32)
        for i in range(n_blocks):
            q, s = quantize_int8(dct_blocks[i])
            quantized[i] = q
            scales[i] = s
        data = {
            "quantized": quantized,
            "scales": scales,
            "n_orig": np.int32(n_orig),
            "block_size": np.int32(bs),
        }
        meta = {"orig_shape": orig_shape, "method": self.METHOD_NAME}
        return data, meta

    def decompress(self, data, metadata):
        quantized, scales = data["quantized"], data["scales"]
        bs = int(data["block_size"])
        C = _get_dct_matrix(bs)
        n_blocks = quantized.shape[0]
        recon = np.zeros((n_blocks, bs), dtype=np.float64)
        for i in range(n_blocks):
            h = dequantize_int8(quantized[i], scales[i])
            recon[i] = C.T @ h
        flat = recon.ravel()
        return (
            flat[: int(data["n_orig"])]
            .reshape(metadata["orig_shape"])
            .astype(np.float32)
        )

    def _compressed_size(self, data):
        return data["quantized"].size + data["scales"].size * 4


class WaveletBlockQuantize(WorkingTransformCompressor):
    METHOD_NAME = "wavelet_block_int8"

    def __init__(self, block_size: int = 128, n_levels: int = 3):
        super().__init__(block_size)
        self.n_levels = n_levels

    def compress(self, tensor, **kwargs):
        orig_shape = tensor.shape
        blocks, n_orig = self._to_blocks(tensor)
        n_blocks, bs = blocks.shape
        w_blocks = np.zeros((n_blocks, bs), dtype=np.float64)
        all_sizes = []
        for i in range(n_blocks):
            coeffs, sizes = _haar_decompose(blocks[i], self.n_levels)
            w_blocks[i, : len(coeffs)] = coeffs
            all_sizes.append(sizes)
        quantized = np.zeros((n_blocks, bs), dtype=np.int8)
        scales = np.zeros(n_blocks, dtype=np.float32)
        for i in range(n_blocks):
            q, s = quantize_int8(w_blocks[i])
            quantized[i] = q
            scales[i] = s
        data = {
            "quantized": quantized,
            "scales": scales,
            "n_orig": np.int32(n_orig),
            "block_size": np.int32(bs),
            "n_levels": np.int32(self.n_levels),
            "sizes": np.array(
                [s for sizes in all_sizes for s in sizes], dtype=np.int32
            ).reshape(n_blocks, -1),
        }
        meta = {"orig_shape": orig_shape, "method": self.METHOD_NAME}
        return data, meta

    def decompress(self, data, metadata):
        quantized, scales = data["quantized"], data["scales"]
        bs = int(data["block_size"])
        n_blocks = quantized.shape[0]
        sizes_all = data["sizes"]
        recon = np.zeros((n_blocks, bs), dtype=np.float64)
        for i in range(n_blocks):
            h = dequantize_int8(quantized[i], scales[i])
            sizes = sizes_all[i].tolist()
            recon[i] = _haar_reconstruct(h[: sum(sizes)], sizes, bs)
        flat = recon.ravel()
        return (
            flat[: int(data["n_orig"])]
            .reshape(metadata["orig_shape"])
            .astype(np.float32)
        )

    def _compressed_size(self, data):
        return data["quantized"].size + data["scales"].size * 4 + data["sizes"].size * 4


def _pack_tiers(tier_mask: np.ndarray) -> np.ndarray:
    n = len(tier_mask)
    n_packed = (n + 3) // 4
    packed = np.zeros(n_packed, dtype=np.uint8)
    for i in range(n):
        packed[i // 4] |= (tier_mask[i] & 0x3) << (2 * (i % 4))
    return packed


def _unpack_tiers(packed: np.ndarray, n: int) -> np.ndarray:
    result = np.zeros(n, dtype=np.uint8)
    for i in range(n):
        result[i] = (packed[i // 4] >> (2 * (i % 4))) & 0x3
    return result


class HadamardMixedPrecision(WorkingTransformCompressor):
    METHOD_NAME = "hadamard_mixed_precision"

    def __init__(
        self, block_size: int = 128, int8_ratio: float = 0.5, int4_ratio: float = 0.3
    ):
        super().__init__(block_size)
        self.int8_ratio = int8_ratio
        self.int4_ratio = int4_ratio

    def compress(self, tensor, **kwargs):
        orig_shape = tensor.shape
        blocks, n_orig = self._to_blocks(tensor)
        n_blocks, bs = blocks.shape

        h_blocks = fwht(blocks, normalize=True)

        n_int8 = max(1, int(bs * self.int8_ratio))
        n_int4 = max(0, int(bs * self.int4_ratio))

        all_q = np.zeros((n_blocks, bs), dtype=np.int8)
        all_tier_packed = []
        all_scales = np.zeros((n_blocks, 3), dtype=np.float32)

        for i in range(n_blocks):
            block = h_blocks[i]
            abs_vals = np.abs(block)
            order = np.argsort(abs_vals)[::-1]

            tier = np.full(bs, 2, dtype=np.uint8)
            tier[order[:n_int8]] = 0
            if n_int4 > 0:
                tier[order[n_int8 : n_int8 + n_int4]] = 1

            for t_val, qfn in [
                (0, quantize_int8),
                (1, quantize_int4),
                (2, quantize_int2),
            ]:
                mask = tier == t_val
                if np.any(mask):
                    q, s = qfn(block[mask])
                    all_q[i, mask] = q
                    all_scales[i, t_val] = s

            all_tier_packed.append(_pack_tiers(tier))

        data = {
            "quantized": all_q,
            "scales": all_scales,
            "tier_packed": all_tier_packed,
            "n_orig": np.int32(n_orig),
            "block_size": np.int32(bs),
        }
        meta = {"orig_shape": orig_shape, "method": self.METHOD_NAME}
        return data, meta

    def decompress(self, data, metadata):
        quantized = data["quantized"]
        scales = data["scales"]
        bs = int(data["block_size"])
        n_blocks = quantized.shape[0]

        recon = np.zeros((n_blocks, bs), dtype=np.float64)
        for i in range(n_blocks):
            tier = _unpack_tiers(data["tier_packed"][i], bs)
            h = np.zeros(bs, dtype=np.float64)
            for t_val, max_val in [(0, 127.0), (1, 7.0), (2, 3.0)]:
                mask = tier == t_val
                if np.any(mask):
                    h[mask] = quantized[i, mask].astype(np.float64) * scales[i, t_val]
            recon[i] = fwht(h, normalize=True)

        flat = recon.ravel()
        return (
            flat[: int(data["n_orig"])]
            .reshape(metadata["orig_shape"])
            .astype(np.float32)
        )

    def _compressed_size(self, data):
        q = data["quantized"]
        s = data["scales"]
        total = q.size
        total += s.size * 4
        for p in data["tier_packed"]:
            total += p.size
        return total


class DCTMixedPrecision(WorkingTransformCompressor):
    METHOD_NAME = "dct_mixed_precision"

    def __init__(
        self, block_size: int = 128, int8_ratio: float = 0.5, int4_ratio: float = 0.3
    ):
        super().__init__(block_size)
        self.int8_ratio = int8_ratio
        self.int4_ratio = int4_ratio

    def compress(self, tensor, **kwargs):
        orig_shape = tensor.shape
        blocks, n_orig = self._to_blocks(tensor)
        n_blocks, bs = blocks.shape

        C = _get_dct_matrix(bs)
        dct_blocks = (C @ blocks.T).T

        n_int8 = max(1, int(bs * self.int8_ratio))
        n_int4 = max(0, int(bs * self.int4_ratio))

        all_q = np.zeros((n_blocks, bs), dtype=np.int8)
        all_scales = np.zeros((n_blocks, 3), dtype=np.float32)

        for i in range(n_blocks):
            block = dct_blocks[i]
            q, s = quantize_int8(block[:n_int8])
            all_q[i, :n_int8] = q
            all_scales[i, 0] = s
            if n_int4 > 0:
                q, s = quantize_int4(block[n_int8 : n_int8 + n_int4])
                all_q[i, n_int8 : n_int8 + n_int4] = q
                all_scales[i, 1] = s
            n_int2 = bs - n_int8 - n_int4
            if n_int2 > 0:
                q, s = quantize_int2(block[n_int8 + n_int4 :])
                all_q[i, n_int8 + n_int4 :] = q
                all_scales[i, 2] = s

        data = {
            "quantized": all_q,
            "scales": all_scales,
            "n_orig": np.int32(n_orig),
            "block_size": np.int32(bs),
            "n_int8": np.int32(n_int8),
            "n_int4": np.int32(n_int4),
        }
        meta = {"orig_shape": orig_shape, "method": self.METHOD_NAME}
        return data, meta

    def decompress(self, data, metadata):
        quantized, scales = data["quantized"], data["scales"]
        bs = int(data["block_size"])
        n_int8 = int(data["n_int8"])
        n_int4 = int(data["n_int4"])
        n_blocks = quantized.shape[0]
        C = _get_dct_matrix(bs)

        recon = np.zeros((n_blocks, bs), dtype=np.float64)
        for i in range(n_blocks):
            h = np.zeros(bs, dtype=np.float64)
            h[:n_int8] = quantized[i, :n_int8].astype(np.float64) * scales[i, 0]
            if n_int4 > 0:
                h[n_int8 : n_int8 + n_int4] = (
                    quantized[i, n_int8 : n_int8 + n_int4].astype(np.float64)
                    * scales[i, 1]
                )
            n_int2 = bs - n_int8 - n_int4
            if n_int2 > 0:
                h[n_int8 + n_int4 :] = (
                    quantized[i, n_int8 + n_int4 :].astype(np.float64) * scales[i, 2]
                )
            recon[i] = C.T @ h

        flat = recon.ravel()
        return (
            flat[: int(data["n_orig"])]
            .reshape(metadata["orig_shape"])
            .astype(np.float32)
        )

    def _compressed_size(self, data):
        return data["quantized"].size + data["scales"].size * 4


class WaveletMixedPrecision(WorkingTransformCompressor):
    METHOD_NAME = "wavelet_mixed_precision"

    def __init__(
        self, block_size: int = 128, n_levels: int = 3, detail_int4_ratio: float = 0.5
    ):
        super().__init__(block_size)
        self.n_levels = n_levels
        self.detail_int4_ratio = detail_int4_ratio

    def compress(self, tensor, **kwargs):
        orig_shape = tensor.shape
        blocks, n_orig = self._to_blocks(tensor)
        n_blocks, bs = blocks.shape

        all_q = np.zeros((n_blocks, bs), dtype=np.int8)
        all_tier_packed = []
        all_scales = np.zeros((n_blocks, 3), dtype=np.float32)
        all_sizes = []

        for i in range(n_blocks):
            coeffs, sizes = _haar_decompose(blocks[i], self.n_levels)
            all_sizes.append(sizes)
            n_coeffs = len(coeffs)
            n_approx = sizes[0]

            tier = np.zeros(bs, dtype=np.uint8)
            tier[:n_approx] = 0

            detail_coeffs = coeffs[n_approx:]
            if len(detail_coeffs) > 0:
                abs_vals = np.abs(detail_coeffs)
                n_int4 = max(0, int(len(detail_coeffs) * self.detail_int4_ratio))
                detail_tier = np.full(len(detail_coeffs), 2, dtype=np.uint8)
                if n_int4 > 0:
                    order = np.argsort(abs_vals)[::-1]
                    detail_tier[order[:n_int4]] = 1
                tier[n_approx : n_approx + len(detail_coeffs)] = detail_tier

            for t_val, qfn in [
                (0, quantize_int8),
                (1, quantize_int4),
                (2, quantize_int2),
            ]:
                mask = tier == t_val
                if np.any(mask):
                    q, s = qfn(coeffs[mask])
                    all_q[i, mask] = q
                    all_scales[i, t_val] = s

            all_tier_packed.append(_pack_tiers(tier))

        data = {
            "quantized": all_q,
            "scales": all_scales,
            "tier_packed": all_tier_packed,
            "sizes": np.array(
                [s for sizes in all_sizes for s in sizes], dtype=np.int32
            ).reshape(n_blocks, -1),
            "n_orig": np.int32(n_orig),
            "block_size": np.int32(bs),
        }
        meta = {"orig_shape": orig_shape, "method": self.METHOD_NAME}
        return data, meta

    def decompress(self, data, metadata):
        quantized = data["quantized"]
        scales = data["scales"]
        sizes_all = data["sizes"]
        bs = int(data["block_size"])
        n_blocks = quantized.shape[0]

        recon = np.zeros((n_blocks, bs), dtype=np.float64)
        for i in range(n_blocks):
            sizes = sizes_all[i].tolist()
            tier = _unpack_tiers(data["tier_packed"][i], bs)
            coeffs = np.zeros(bs, dtype=np.float64)
            for t_val in range(3):
                mask = tier == t_val
                if np.any(mask):
                    coeffs[mask] = (
                        quantized[i, mask].astype(np.float64) * scales[i, t_val]
                    )
            recon[i] = _haar_reconstruct(coeffs[: sum(sizes)], sizes, bs)

        flat = recon.ravel()
        return (
            flat[: int(data["n_orig"])]
            .reshape(metadata["orig_shape"])
            .astype(np.float32)
        )

    def _compressed_size(self, data):
        total = data["quantized"].size
        total += data["scales"].size * 4
        for p in data["tier_packed"]:
            total += p.size
        total += data["sizes"].size * 4
        return total


class HadamardDCTHybrid(WorkingTransformCompressor):
    METHOD_NAME = "hadamard_dct_hybrid"

    def __init__(self, block_size: int = 256, dct_sub_block: int = 64):
        super().__init__(block_size)
        self.dct_sub_block = dct_sub_block

    def compress(self, tensor, **kwargs):
        orig_shape = tensor.shape
        blocks, n_orig = self._to_blocks(tensor)
        n_blocks, bs = blocks.shape
        sub = self.dct_sub_block
        C = _get_dct_matrix(sub)

        h_blocks = np.zeros_like(blocks)
        for i in range(n_blocks):
            h = fwht(blocks[i], normalize=True)
            n_sub = bs // sub
            for j in range(n_sub):
                s = j * sub
                h_blocks[i, s : s + sub] = C @ h[s : s + sub]

        quantized = np.zeros((n_blocks, bs), dtype=np.int8)
        scales = np.zeros(n_blocks, dtype=np.float32)
        for i in range(n_blocks):
            q, s = quantize_int8(h_blocks[i])
            quantized[i] = q
            scales[i] = s
        data = {
            "quantized": quantized,
            "scales": scales,
            "n_orig": np.int32(n_orig),
            "block_size": np.int32(bs),
            "dct_sub_block": np.int32(sub),
        }
        meta = {"orig_shape": orig_shape, "method": self.METHOD_NAME}
        return data, meta

    def decompress(self, data, metadata):
        quantized, scales = data["quantized"], data["scales"]
        bs = int(data["block_size"])
        sub = int(data["dct_sub_block"])
        C = _get_dct_matrix(sub)
        n_blocks = quantized.shape[0]
        n_sub = bs // sub

        recon = np.zeros((n_blocks, bs), dtype=np.float64)
        for i in range(n_blocks):
            h = np.zeros(bs, dtype=np.float64)
            for j in range(n_sub):
                s = j * sub
                block_c = quantized[i, s : s + sub].astype(np.float64) * scales[i]
                h[s : s + sub] = C.T @ block_c
            recon[i] = fwht(h, normalize=True)
        flat = recon.ravel()
        return (
            flat[: int(data["n_orig"])]
            .reshape(metadata["orig_shape"])
            .astype(np.float32)
        )

    def _compressed_size(self, data):
        return data["quantized"].size + data["scales"].size * 4


class MultiResolutionQuantize(WorkingTransformCompressor):
    METHOD_NAME = "multi_resolution_quantize"

    def __init__(self, block_size: int = 256, n_levels: int = 4):
        super().__init__(block_size)
        self.n_levels = n_levels

    def compress(self, tensor, **kwargs):
        orig_shape = tensor.shape
        blocks, n_orig = self._to_blocks(tensor)
        n_blocks, bs = blocks.shape

        all_q = np.zeros((n_blocks, bs), dtype=np.int8)
        all_tier_packed = []
        all_scales = np.zeros((n_blocks, 3), dtype=np.float32)
        all_sizes = []

        for i in range(n_blocks):
            coeffs, sizes = _haar_decompose(blocks[i], self.n_levels)
            all_sizes.append(sizes)

            tier = np.zeros(bs, dtype=np.uint8)
            boundaries = [0]
            for sz in sizes:
                boundaries.append(boundaries[-1] + sz)
            n_lvls = len(sizes)
            for level_idx in range(n_lvls):
                start = boundaries[level_idx]
                end = boundaries[level_idx + 1]
                if level_idx == 0:
                    tier[start:end] = 0
                elif level_idx <= n_lvls // 2:
                    tier[start:end] = 1
                else:
                    tier[start:end] = 2

            for t_val, qfn in [
                (0, quantize_int8),
                (1, quantize_int4),
                (2, quantize_int2),
            ]:
                mask = tier == t_val
                if np.any(mask):
                    q, s = qfn(coeffs[mask])
                    all_q[i, mask] = q
                    all_scales[i, t_val] = s

            all_tier_packed.append(_pack_tiers(tier))

        data = {
            "quantized": all_q,
            "scales": all_scales,
            "tier_packed": all_tier_packed,
            "sizes": np.array(
                [s for sizes in all_sizes for s in sizes], dtype=np.int32
            ).reshape(n_blocks, -1),
            "n_orig": np.int32(n_orig),
            "block_size": np.int32(bs),
        }
        meta = {"orig_shape": orig_shape, "method": self.METHOD_NAME}
        return data, meta

    def decompress(self, data, metadata):
        quantized = data["quantized"]
        scales = data["scales"]
        sizes_all = data["sizes"]
        bs = int(data["block_size"])
        n_blocks = quantized.shape[0]

        recon = np.zeros((n_blocks, bs), dtype=np.float64)
        for i in range(n_blocks):
            sizes = sizes_all[i].tolist()
            tier = _unpack_tiers(data["tier_packed"][i], bs)
            coeffs = np.zeros(sum(sizes), dtype=np.float64)
            for t_val in range(3):
                mask = tier == t_val
                if np.any(mask):
                    coeffs[mask] = (
                        quantized[i, mask].astype(np.float64) * scales[i, t_val]
                    )
            recon[i] = _haar_reconstruct(coeffs, sizes, bs)

        flat = recon.ravel()
        return (
            flat[: int(data["n_orig"])]
            .reshape(metadata["orig_shape"])
            .astype(np.float32)
        )

    def _compressed_size(self, data):
        total = data["quantized"].size
        total += data["scales"].size * 4
        for p in data["tier_packed"]:
            total += p.size
        total += data["sizes"].size * 4
        return total


class SpectralSliceQuantize(WorkingTransformCompressor):
    METHOD_NAME = "spectral_slice_quantize"

    def __init__(
        self,
        block_size: int = 128,
        low_freq_ratio: float = 0.25,
        mid_freq_ratio: float = 0.25,
    ):
        super().__init__(block_size)
        self.low_freq_ratio = low_freq_ratio
        self.mid_freq_ratio = mid_freq_ratio

    def compress(self, tensor, **kwargs):
        orig_shape = tensor.shape
        blocks, n_orig = self._to_blocks(tensor)
        n_blocks, bs = blocks.shape
        C = _get_dct_matrix(bs)
        dct_blocks = (C @ blocks.T).T

        n_low = max(1, int(bs * self.low_freq_ratio))
        n_mid = max(0, int(bs * self.mid_freq_ratio))

        all_q = np.zeros((n_blocks, bs), dtype=np.int8)
        all_scales = np.zeros((n_blocks, 3), dtype=np.float32)

        for i in range(n_blocks):
            block = dct_blocks[i]
            q, s = quantize_int8(block[:n_low])
            all_q[i, :n_low] = q
            all_scales[i, 0] = s
            if n_mid > 0:
                q, s = quantize_int4(block[n_low : n_low + n_mid])
                all_q[i, n_low : n_low + n_mid] = q
                all_scales[i, 1] = s
            n_high = bs - n_low - n_mid
            if n_high > 0:
                q, s = quantize_int2(block[n_low + n_mid :])
                all_q[i, n_low + n_mid :] = q
                all_scales[i, 2] = s

        data = {
            "quantized": all_q,
            "scales": all_scales,
            "n_orig": np.int32(n_orig),
            "block_size": np.int32(bs),
            "n_low": np.int32(n_low),
            "n_mid": np.int32(n_mid),
        }
        meta = {"orig_shape": orig_shape, "method": self.METHOD_NAME}
        return data, meta

    def decompress(self, data, metadata):
        quantized, scales = data["quantized"], data["scales"]
        bs = int(data["block_size"])
        n_low = int(data["n_low"])
        n_mid = int(data["n_mid"])
        n_blocks = quantized.shape[0]
        C = _get_dct_matrix(bs)

        recon = np.zeros((n_blocks, bs), dtype=np.float64)
        for i in range(n_blocks):
            h = np.zeros(bs, dtype=np.float64)
            h[:n_low] = quantized[i, :n_low].astype(np.float64) * scales[i, 0]
            if n_mid > 0:
                h[n_low : n_low + n_mid] = (
                    quantized[i, n_low : n_low + n_mid].astype(np.float64)
                    * scales[i, 1]
                )
            n_high = bs - n_low - n_mid
            if n_high > 0:
                h[n_low + n_mid :] = (
                    quantized[i, n_low + n_mid :].astype(np.float64) * scales[i, 2]
                )
            recon[i] = C.T @ h

        flat = recon.ravel()
        return (
            flat[: int(data["n_orig"])]
            .reshape(metadata["orig_shape"])
            .astype(np.float32)
        )

    def _compressed_size(self, data):
        return data["quantized"].size + data["scales"].size * 4


class AdaptiveTransformSelect(WorkingTransformCompressor):
    METHOD_NAME = "adaptive_transform_select"

    def __init__(self, block_size: int = 128):
        super().__init__(block_size)

    def _apply_transform(self, block: np.ndarray, t_name: str) -> np.ndarray:
        if t_name == "hadamard":
            return fwht(block, normalize=True)
        elif t_name == "dct":
            C = _get_dct_matrix(len(block))
            return C @ block
        elif t_name == "wavelet":
            coeffs, _ = _haar_decompose(block, 3)
            return coeffs
        raise ValueError(f"Unknown transform: {t_name}")

    def _inverse_transform(
        self, h: np.ndarray, t_name: str, block_size: int, sizes=None
    ) -> np.ndarray:
        if t_name == "hadamard":
            return fwht(h, normalize=True)
        elif t_name == "dct":
            C = _get_dct_matrix(block_size)
            return C.T @ h
        elif t_name == "wavelet":
            if sizes is None:
                _, sizes = _haar_decompose(np.zeros(block_size), 3)
            return _haar_reconstruct(h, sizes, block_size)
        raise ValueError(f"Unknown transform: {t_name}")

    def compress(self, tensor, **kwargs):
        orig_shape = tensor.shape
        blocks, n_orig = self._to_blocks(tensor)
        n_blocks, bs = blocks.shape
        transforms = ["hadamard", "dct", "wavelet"]

        quantized = np.zeros((n_blocks, bs), dtype=np.int8)
        scales = np.zeros(n_blocks, dtype=np.float32)
        chosen = np.zeros(n_blocks, dtype=np.uint8)
        all_sizes = [None] * n_blocks

        for i in range(n_blocks):
            best_error = float("inf")
            best_t = 0
            best_q = None
            best_s = 0.0
            best_sizes = None

            for t_idx, t_name in enumerate(transforms):
                h = self._apply_transform(blocks[i], t_name)
                sizes = None
                if t_name == "wavelet":
                    _, sizes = _haar_decompose(blocks[i], 3)
                h_padded = np.zeros(bs, dtype=np.float64)
                h_padded[: len(h)] = h
                q, s = quantize_int8(h_padded)
                h_rec = dequantize_int8(q, s)
                h_rec_actual = h_rec[: len(h)]
                recon = self._inverse_transform(h_rec_actual, t_name, bs, sizes)
                error = float(np.sum((blocks[i] - recon) ** 2))
                if error < best_error:
                    best_error = error
                    best_t = t_idx
                    best_q = q
                    best_s = s
                    best_sizes = sizes

            quantized[i] = best_q
            scales[i] = best_s
            chosen[i] = best_t
            all_sizes[i] = best_sizes

        data = {
            "quantized": quantized,
            "scales": scales,
            "chosen_transforms": chosen,
            "n_orig": np.int32(n_orig),
            "block_size": np.int32(bs),
        }
        meta = {"orig_shape": orig_shape, "method": self.METHOD_NAME}
        return data, meta

    def decompress(self, data, metadata):
        quantized, scales, chosen = (
            data["quantized"],
            data["scales"],
            data["chosen_transforms"],
        )
        bs = int(data["block_size"])
        n_blocks = quantized.shape[0]
        transforms = ["hadamard", "dct", "wavelet"]

        _, wavelet_sizes = _haar_decompose(np.zeros(bs), 3)

        recon = np.zeros((n_blocks, bs), dtype=np.float64)
        for i in range(n_blocks):
            h = dequantize_int8(quantized[i], scales[i])
            t_name = transforms[chosen[i]]
            if t_name == "wavelet":
                recon[i] = _haar_reconstruct(h[: sum(wavelet_sizes)], wavelet_sizes, bs)
            elif t_name == "dct":
                C = _get_dct_matrix(bs)
                recon[i] = C.T @ h
            else:
                recon[i] = fwht(h, normalize=True)
        flat = recon.ravel()
        return (
            flat[: int(data["n_orig"])]
            .reshape(metadata["orig_shape"])
            .astype(np.float32)
        )

    def _compressed_size(self, data):
        return (
            data["quantized"].size
            + data["scales"].size * 4
            + data["chosen_transforms"].size
        )


ALL_WORKING_TRANSFORMS = {
    "plain_block_int8": PlainBlockInt8,
    "hadamard_block_int8": HadamardBlockQuantize,
    "dct_block_int8": DCTBlockQuantize,
    "wavelet_block_int8": WaveletBlockQuantize,
    "hadamard_mixed_precision": HadamardMixedPrecision,
    "dct_mixed_precision": DCTMixedPrecision,
    "wavelet_mixed_precision": WaveletMixedPrecision,
    "hadamard_dct_hybrid": HadamardDCTHybrid,
    "multi_resolution_quantize": MultiResolutionQuantize,
    "spectral_slice_quantize": SpectralSliceQuantize,
    "adaptive_transform_select": AdaptiveTransformSelect,
}
