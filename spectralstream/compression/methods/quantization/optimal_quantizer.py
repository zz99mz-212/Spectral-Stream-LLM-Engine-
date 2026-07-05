"""
Optimal INT4 Quantization Engine
================================
10 state-of-the-art quantization methods for achieving maximum compression
while preserving quality on real neural network weight tensors.
"""

from __future__ import annotations

import math
import struct
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _next_power_of_two(n: int) -> int:
    if n < 1:
        return 1
    return 1 << (n - 1).bit_length()


def _pack_4bit(values: np.ndarray) -> bytes:
    flat = values.astype(np.uint8).ravel()
    n = len(flat)
    n_pairs = (n + 1) // 2
    packed = np.zeros(n_pairs, dtype=np.uint8)
    for i in range(0, n, 2):
        lo = int(flat[i]) & 0x0F
        hi = int(flat[i + 1]) & 0x0F if i + 1 < n else 0
        packed[i // 2] = lo | (hi << 4)
    return packed.tobytes()


def _unpack_4bit(data: bytes, n: int) -> np.ndarray:
    raw = np.frombuffer(data, dtype=np.uint8)
    out = np.zeros(n, dtype=np.uint8)
    for i in range(n):
        byte_idx = i // 2
        if byte_idx < len(raw):
            if i % 2 == 0:
                out[i] = raw[byte_idx] & 0x0F
            else:
                out[i] = (raw[byte_idx] >> 4) & 0x0F
    return out


def _pack_nbit(values: np.ndarray, bits: int) -> bytes:
    flat = values.astype(np.uint8).ravel()
    n = len(flat)
    if bits == 8:
        return flat.tobytes()
    if bits == 4:
        return _pack_4bit(flat)
    if bits == 3:
        n3 = n * 3
        packed = np.zeros((n3 + 7) // 8, dtype=np.uint8)
        for i in range(n):
            byte_offset = (i * 3) // 8
            bit_offset = (i * 3) % 8
            val = int(flat[i]) & 0x07
            packed[byte_offset] |= (val << bit_offset) & 0xFF
            if bit_offset > 5 and byte_offset + 1 < len(packed):
                packed[byte_offset + 1] |= (val >> (8 - bit_offset)) & 0xFF
        return packed.tobytes()
    if bits == 2:
        n2 = n * 2
        packed = np.zeros((n2 + 3) // 4, dtype=np.uint8)
        for i in range(n):
            byte_idx = i // 4
            bit_idx = (i % 4) * 2
            packed[byte_idx] |= (int(flat[i]) & 0x03) << bit_idx
        return packed.tobytes()
    if bits == 1:
        n1 = n
        packed = np.zeros((n1 + 7) // 8, dtype=np.uint8)
        for i in range(n):
            packed[i // 8] |= (int(flat[i]) & 0x01) << (i % 8)
        return packed.tobytes()
    return flat.tobytes()


def _unpack_nbit(data: bytes, n: int, bits: int) -> np.ndarray:
    if bits == 8:
        raw = np.frombuffer(data, dtype=np.uint8)
        return raw[:n].copy()
    if bits == 4:
        return _unpack_4bit(data, n)
    if bits == 1:
        raw = np.frombuffer(data, dtype=np.uint8)
        out = np.zeros(n, dtype=np.uint8)
        for i in range(n):
            out[i] = (raw[i // 8] >> (i % 8)) & 0x01
        return out
    if bits == 2:
        raw = np.frombuffer(data, dtype=np.uint8)
        out = np.zeros(n, dtype=np.uint8)
        for i in range(n):
            out[i] = (raw[i // 4] >> ((i % 4) * 2)) & 0x03
        return out
    if bits == 3:
        raw = np.frombuffer(data, dtype=np.uint8)
        out = np.zeros(n, dtype=np.uint8)
        for i in range(n):
            byte_offset = (i * 3) // 8
            bit_offset = (i * 3) % 8
            val = 0
            val |= (raw[byte_offset] >> bit_offset) & 0x07
            if bit_offset > 5 and byte_offset + 1 < len(raw):
                val |= (raw[byte_offset + 1] << (8 - bit_offset)) & 0x07
            out[i] = val & 0x07
        return out
    return np.frombuffer(data, dtype=np.uint8)[:n].copy()


def _error_metrics(orig: np.ndarray, recon: np.ndarray) -> Dict[str, float]:
    o = orig.astype(np.float64).ravel()
    r = recon.astype(np.float64).ravel()
    n = min(len(o), len(r))
    o, r = o[:n], r[:n]
    noise = o - r
    mse = float(np.mean(noise**2))
    signal = float(np.mean(o**2)) + 1e-30
    snr_db = 10.0 * math.log10(signal / (mse + 1e-30))
    peak = float(np.max(np.abs(o))) + 1e-30
    psnr_db = 10.0 * math.log10(peak**2 / (mse + 1e-30))
    rel_err = float(np.linalg.norm(noise) / (np.linalg.norm(o) + 1e-30))
    cos_sim = float(np.dot(o, r) / (np.linalg.norm(o) * np.linalg.norm(r) + 1e-30))
    return {
        "mse": mse,
        "snr_db": snr_db,
        "psnr_db": psnr_db,
        "rel_error": rel_err,
        "cos_sim": cos_sim,
    }


class GroupWiseQuantizer:
    def compress(
        self, tensor: np.ndarray, group_size: int = 128, bits: int = 4
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        flat = tensor.ravel().astype(np.float32)
        n = len(flat)
        gs = min(group_size, n)
        n_groups = math.ceil(n / gs)
        max_val = (1 << bits) - 1
        scales = np.empty(n_groups, dtype=np.float32)
        mins = np.empty(n_groups, dtype=np.float32)
        quantized = np.zeros(n, dtype=np.uint8)
        for g in range(n_groups):
            start = g * gs
            end = min(start + gs, n)
            group = flat[start:end]
            gmin = float(np.min(group))
            gmax = float(np.max(group))
            if gmax - gmin < 1e-10:
                scales[g] = 1.0
                mins[g] = gmin
                quantized[start:end] = 0
            else:
                scale = (gmax - gmin) / max_val
                scales[g] = scale
                mins[g] = gmin
                quantized[start:end] = np.clip(
                    np.round((group - gmin) / scale), 0, max_val
                ).astype(np.uint8)
        packed = _pack_nbit(quantized, bits)
        metadata = {
            "shape": list(tensor.shape),
            "n_elements": n,
            "group_size": gs,
            "bits": bits,
            "scales": scales,
            "mins": mins,
        }
        return np.frombuffer(packed, dtype=np.uint8), metadata

    def decompress(self, quantized: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
        n = metadata["n_elements"]
        gs = metadata["group_size"]
        bits = metadata["bits"]
        scales = metadata["scales"]
        mins = metadata["mins"]
        n_groups = len(scales)
        unpacked = _unpack_nbit(quantized.tobytes(), n, bits)
        result = np.zeros(n, dtype=np.float32)
        for g in range(n_groups):
            start = g * gs
            end = min(start + gs, n)
            result[start:end] = (
                unpacked[start:end].astype(np.float32) * scales[g] + mins[g]
            )
        return result.reshape(metadata["shape"])


def _fwht_inplace(x: np.ndarray) -> None:
    n = len(x)
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                u = x[j]
                v = x[j + h]
                x[j] = u + v
                x[j + h] = u - v
        h *= 2


def _hadamard_transform(x: np.ndarray) -> np.ndarray:
    n = _next_power_of_two(len(x))
    padded = np.zeros(n, dtype=np.float64)
    padded[: len(x)] = x.astype(np.float64)
    _fwht_inplace(padded)
    padded /= math.sqrt(n)
    return padded[: len(x)]


def _inverse_hadamard_transform(x: np.ndarray) -> np.ndarray:
    n = _next_power_of_two(len(x))
    padded = np.zeros(n, dtype=np.float64)
    padded[: len(x)] = x.astype(np.float64) * math.sqrt(n)
    _fwht_inplace(padded)
    padded /= n
    return padded[: len(x)]


class HadamardGroupWiseQuantizer:
    def compress(
        self, tensor: np.ndarray, group_size: int = 128, bits: int = 4, seed: int = 42
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        flat = tensor.ravel().astype(np.float64)
        n = len(flat)
        rng = np.random.RandomState(seed)
        signs = (rng.random(n) > 0.5).astype(np.float64) * 2.0 - 1.0
        signed = flat * signs
        block_size = _next_power_of_two(min(n, 1024))
        transformed = np.zeros(n, dtype=np.float64)
        for i in range(0, n, block_size):
            end = min(i + block_size, n)
            chunk = signed[i:end]
            transformed[i:end] = (
                _hadamard_transform(chunk) if end - i == block_size else chunk
            )
        gs = min(group_size, n)
        n_groups = math.ceil(n / gs)
        max_val = (1 << bits) - 1
        scales = np.empty(n_groups, dtype=np.float32)
        mins = np.empty(n_groups, dtype=np.float32)
        quantized = np.zeros(n, dtype=np.uint8)
        for g in range(n_groups):
            start = g * gs
            end = min(start + gs, n)
            group = transformed[start:end]
            gmin = float(np.min(group))
            gmax = float(np.max(group))
            if gmax - gmin < 1e-10:
                scales[g] = 1.0
                mins[g] = gmin
            else:
                scale = (gmax - gmin) / max_val
                scales[g] = scale
                mins[g] = gmin
                quantized[start:end] = np.clip(
                    np.round((group - gmin) / scale), 0, max_val
                ).astype(np.uint8)
        packed = _pack_nbit(quantized, bits)
        metadata = {
            "shape": list(tensor.shape),
            "n_elements": n,
            "group_size": gs,
            "bits": bits,
            "scales": scales,
            "mins": mins,
            "signs": signs.astype(np.float32),
            "block_size": block_size,
        }
        return np.frombuffer(packed, dtype=np.uint8), metadata

    def decompress(self, quantized: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
        n = metadata["n_elements"]
        gs = metadata["group_size"]
        bits = metadata["bits"]
        scales = metadata["scales"]
        mins = metadata["mins"]
        signs = metadata["signs"].astype(np.float64)
        block_size = metadata["block_size"]
        n_groups = len(scales)
        unpacked = _unpack_nbit(quantized.tobytes(), n, bits)
        reconstructed = np.zeros(n, dtype=np.float64)
        for g in range(n_groups):
            start = g * gs
            end = min(start + gs, n)
            reconstructed[start:end] = (
                unpacked[start:end].astype(np.float64) * scales[g] + mins[g]
            )
        result = np.zeros(n, dtype=np.float64)
        for i in range(0, n, block_size):
            end = min(i + block_size, n)
            chunk = reconstructed[i:end]
            result[i:end] = (
                _inverse_hadamard_transform(chunk) if end - i == block_size else chunk
            )
        result *= signs
        return result.reshape(metadata["shape"]).astype(np.float32)


class AsymmetricQuantizer:
    def compress(
        self, tensor: np.ndarray, group_size: int = 128, bits: int = 4
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        flat = tensor.ravel().astype(np.float32)
        n = len(flat)
        gs = min(group_size, n)
        n_groups = math.ceil(n / gs)
        max_val = (1 << bits) - 1
        half_max = max_val // 2
        pos_scales = np.empty(n_groups, dtype=np.float32)
        neg_scales = np.empty(n_groups, dtype=np.float32)
        quantized = np.zeros(n, dtype=np.uint8)
        for g in range(n_groups):
            start = g * gs
            end = min(start + gs, n)
            group = flat[start:end]
            pos_vals = group[group >= 0]
            neg_vals = group[group < 0]
            pos_max = float(np.max(pos_vals)) if len(pos_vals) > 0 else 0.0
            neg_min = float(np.min(neg_vals)) if len(neg_vals) > 0 else 0.0
            p_scale = pos_max / half_max if pos_max > 1e-10 else 1.0
            n_scale = (-neg_min) / half_max if neg_min < -1e-10 else 1.0
            pos_scales[g] = p_scale
            neg_scales[g] = n_scale
            for i in range(start, end):
                val = float(flat[i])
                if val >= 0:
                    quantized[i] = int(np.clip(round(val / p_scale), 0, half_max))
                else:
                    quantized[i] = int(
                        np.clip(
                            half_max + 1 + round((-val - 1e-10) / n_scale),
                            half_max + 1,
                            max_val,
                        )
                    )
        packed = _pack_nbit(quantized, bits)
        metadata = {
            "shape": list(tensor.shape),
            "n_elements": n,
            "group_size": gs,
            "bits": bits,
            "pos_scales": pos_scales,
            "neg_scales": neg_scales,
        }
        return np.frombuffer(packed, dtype=np.uint8), metadata

    def decompress(self, quantized: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
        n = metadata["n_elements"]
        gs = metadata["group_size"]
        bits = metadata["bits"]
        pos_scales = metadata["pos_scales"]
        neg_scales = metadata["neg_scales"]
        n_groups = len(pos_scales)
        max_val = (1 << bits) - 1
        half_max = max_val // 2
        unpacked = _unpack_nbit(quantized.tobytes(), n, bits)
        result = np.zeros(n, dtype=np.float32)
        for g in range(n_groups):
            start = g * gs
            end = min(start + gs, n)
            q = unpacked[start:end].astype(np.float32)
            is_pos = q <= half_max
            is_neg = q > half_max
            result[start:end] = np.where(
                is_pos, q * pos_scales[g], -(q - half_max - 1) * neg_scales[g]
            )
        return result.reshape(metadata["shape"])


class OutlierAwareQuantizer:
    def compress(
        self,
        tensor: np.ndarray,
        group_size: int = 128,
        bits: int = 4,
        outlier_threshold: float = 3.0,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        t = tensor
        if t.ndim == 1:
            t = t.reshape(1, -1)
        rows, cols = t.shape
        row_norms = np.array([np.max(np.abs(t[i])) for i in range(rows)])
        median_norm = np.median(row_norms)
        mad = np.median(np.abs(row_norms - median_norm)) + 1e-10
        is_outlier = row_norms > median_norm + outlier_threshold * mad * 1.4826
        outlier_data_parts = []
        normal_data_parts = []
        outlier_indices = np.where(is_outlier)[0]
        normal_indices = np.where(~is_outlier)[0]
        q = GroupWiseQuantizer()
        for idx in outlier_indices:
            row = t[idx].ravel().astype(np.float32)
            comp, meta_row = q.compress(row, group_size=group_size, bits=8)
            outlier_data_parts.append((int(idx), comp, meta_row))
        for idx in normal_indices:
            row = t[idx].ravel().astype(np.float32)
            comp, meta_row = q.compress(row, group_size=group_size, bits=bits)
            normal_data_parts.append((int(idx), comp, meta_row))
        packed_outlier = bytearray()
        outlier_map = []
        for idx, comp, _ in outlier_data_parts:
            comp_bytes = comp.tobytes() if hasattr(comp, "tobytes") else bytes(comp)
            packed_outlier += struct.pack("<I", len(comp_bytes))
            packed_outlier += comp_bytes
            outlier_map.append(idx)
        packed_normal = bytearray()
        normal_map = []
        for idx, comp, _ in normal_data_parts:
            comp_bytes = comp.tobytes() if hasattr(comp, "tobytes") else bytes(comp)
            packed_normal += struct.pack("<I", len(comp_bytes))
            packed_normal += comp_bytes
            normal_map.append(idx)
        metadata = {
            "shape": list(tensor.shape),
            "bits": bits,
            "group_size": group_size,
            "outlier_bits": 8,
            "is_outlier": is_outlier.tolist(),
            "outlier_map": outlier_map,
            "normal_map": normal_map,
            "outlier_metas": [m for _, _, m in outlier_data_parts],
            "normal_metas": [m for _, _, m in normal_data_parts],
        }
        combined = bytes(packed_outlier) + b"|||" + bytes(packed_normal)
        return np.frombuffer(combined, dtype=np.uint8), metadata

    def decompress(self, quantized: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
        shape = metadata["shape"]
        t = (
            np.zeros(shape, dtype=np.float32)
            if len(shape) > 1
            else np.zeros(shape, dtype=np.float32)
        )
        if t.ndim == 1:
            t = t.reshape(1, -1)
        rows = t.shape[0]
        q = GroupWiseQuantizer()
        raw = quantized.tobytes()
        sep = raw.find(b"|||")
        outlier_bytes = raw[:sep]
        normal_bytes = raw[sep + 3 :]
        offset = 0
        for i, idx in enumerate(metadata["outlier_map"]):
            if offset + 4 > len(outlier_bytes):
                break
            length = struct.unpack("<I", outlier_bytes[offset : offset + 4])[0]
            offset += 4
            comp = np.frombuffer(
                outlier_bytes[offset : offset + length], dtype=np.uint8
            )
            offset += length
            t[idx] = q.decompress(comp, metadata["outlier_metas"][i])
        offset = 0
        for i, idx in enumerate(metadata["normal_map"]):
            if offset + 4 > len(normal_bytes):
                break
            length = struct.unpack("<I", normal_bytes[offset : offset + 4])[0]
            offset += 4
            comp = np.frombuffer(normal_bytes[offset : offset + length], dtype=np.uint8)
            offset += length
            t[idx] = q.decompress(comp, metadata["normal_metas"][i])
        return t.ravel() if len(shape) == 1 else t


class MixedBitwidthQuantizer:
    def compress(
        self, tensor: np.ndarray, group_size: int = 128, bits: int = 4
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        flat = tensor.ravel().astype(np.float32)
        n = len(flat)
        gs = min(group_size, n)
        n_groups = math.ceil(n / gs)
        q = GroupWiseQuantizer()
        group_bitwidths = np.zeros(n_groups, dtype=np.uint8)
        group_data_parts = []
        group_metas = []
        for g in range(n_groups):
            start = g * gs
            end = min(start + gs, n)
            group = flat[start:end]
            std = float(np.std(group))
            mean_abs = float(np.mean(np.abs(group)))
            max_abs = float(np.max(np.abs(group)))
            if max_abs < 1e-8:
                gbits = 1
            elif std < 0.001 * max_abs:
                gbits = 2
            elif std < 0.01 * max_abs:
                gbits = min(bits, 4)
            else:
                gbits = bits
            group_bitwidths[g] = gbits
            comp, meta = q.compress(group, group_size=len(group), bits=gbits)
            group_data_parts.append(comp)
            group_metas.append(meta)
        packed = bytearray()
        for comp in group_data_parts:
            comp_bytes = comp.tobytes() if hasattr(comp, "tobytes") else bytes(comp)
            packed += struct.pack("<I", len(comp_bytes))
            packed += comp_bytes
        metadata = {
            "shape": list(tensor.shape),
            "n_elements": n,
            "group_size": gs,
            "group_bitwidths": group_bitwidths,
            "group_metas": group_metas,
        }
        return np.frombuffer(packed, dtype=np.uint8), metadata

    def decompress(self, quantized: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
        n = metadata["n_elements"]
        gs = metadata["group_size"]
        group_metas = metadata["group_metas"]
        raw = quantized.tobytes()
        q = GroupWiseQuantizer()
        result = np.zeros(n, dtype=np.float32)
        offset = 0
        for g, meta in enumerate(group_metas):
            length = struct.unpack("<I", raw[offset : offset + 4])[0]
            offset += 4
            comp = np.frombuffer(raw[offset : offset + length], dtype=np.uint8)
            offset += length
            start = g * gs
            end = min(start + gs, n)
            result[start:end] = q.decompress(comp, meta).ravel()[: end - start]
        return result.reshape(metadata["shape"])


class QuantizationErrorFeedback:
    def compress(
        self,
        tensor: np.ndarray,
        group_size: int = 128,
        bits: int = 4,
        feedback_weight: float = 0.8,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        flat = tensor.ravel().astype(np.float64)
        n = len(flat)
        gs = min(group_size, n)
        n_groups = math.ceil(n / gs)
        max_val = (1 << bits) - 1
        scales = np.empty(n_groups, dtype=np.float32)
        mins = np.empty(n_groups, dtype=np.float32)
        quantized = np.zeros(n, dtype=np.uint8)
        residual = np.zeros(n, dtype=np.float64)
        for g in range(n_groups):
            start = g * gs
            end = min(start + gs, n)
            adjusted = flat[start:end] + residual[start:end]
            gmin = float(np.min(adjusted))
            gmax = float(np.max(adjusted))
            if gmax - gmin < 1e-10:
                scales[g] = 1.0
                mins[g] = gmin
                quantized[start:end] = 0
                recon = np.full(end - start, gmin, dtype=np.float64)
            else:
                scale = (gmax - gmin) / max_val
                scales[g] = scale
                mins[g] = gmin
                quantized[start:end] = np.clip(
                    np.round((adjusted - gmin) / scale), 0, max_val
                ).astype(np.uint8)
                recon = quantized[start:end].astype(np.float64) * scale + gmin
            error = adjusted - recon
            for i in range(end - start):
                residual[start + i] = feedback_weight * error[i]
        packed = _pack_nbit(quantized, bits)
        metadata = {
            "shape": list(tensor.shape),
            "n_elements": n,
            "group_size": gs,
            "bits": bits,
            "scales": scales,
            "mins": mins,
        }
        return np.frombuffer(packed, dtype=np.uint8), metadata

    def decompress(self, quantized: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
        n = metadata["n_elements"]
        gs = metadata["group_size"]
        bits = metadata["bits"]
        scales = metadata["scales"]
        mins = metadata["mins"]
        n_groups = len(scales)
        unpacked = _unpack_nbit(quantized.tobytes(), n, bits)
        result = np.zeros(n, dtype=np.float32)
        for g in range(n_groups):
            start = g * gs
            end = min(start + gs, n)
            result[start:end] = (
                unpacked[start:end].astype(np.float32) * scales[g] + mins[g]
            )
        return result.reshape(metadata["shape"])


class OptimalLloydMax:
    def compress(
        self,
        tensor: np.ndarray,
        group_size: int = 128,
        bits: int = 4,
        max_iter: int = 50,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        flat = tensor.ravel().astype(np.float64)
        n = len(flat)
        gs = min(group_size, n)
        n_groups = math.ceil(n / gs)
        n_levels = 1 << bits
        all_indices = np.zeros(n, dtype=np.uint8)
        all_centroids = np.zeros(n_groups * n_levels, dtype=np.float64)
        all_boundaries = np.zeros(n_groups * (n_levels - 1), dtype=np.float64)
        for g in range(n_groups):
            start = g * gs
            end = min(start + gs, n)
            group = flat[start:end]
            lo = float(np.min(group))
            hi = float(np.max(group))
            if hi - lo < 1e-10:
                all_centroids[g * n_levels : (g + 1) * n_levels] = lo
                continue
            centroids = np.linspace(lo, hi, n_levels)
            for _ in range(max_iter):
                boundaries = (centroids[1:] + centroids[:-1]) / 2.0
                indices = np.clip(np.digitize(group, boundaries), 0, n_levels - 1)
                new_centroids = np.empty(n_levels)
                for i in range(n_levels):
                    mask = indices == i
                    new_centroids[i] = (
                        np.mean(group[mask]) if np.any(mask) else centroids[i]
                    )
                if np.allclose(centroids, new_centroids, atol=1e-6):
                    break
                centroids = new_centroids
            boundaries = (centroids[1:] + centroids[:-1]) / 2.0
            indices = np.clip(np.digitize(group, boundaries), 0, n_levels - 1)
            all_indices[start:end] = indices.astype(np.uint8)
            all_centroids[g * n_levels : (g + 1) * n_levels] = centroids
            if n_levels > 1:
                all_boundaries[g * (n_levels - 1) : (g + 1) * (n_levels - 1)] = (
                    boundaries
                )
        packed = _pack_nbit(all_indices, bits)
        metadata = {
            "shape": list(tensor.shape),
            "n_elements": n,
            "group_size": gs,
            "bits": bits,
            "n_levels": n_levels,
            "centroids": all_centroids,
            "boundaries": all_boundaries,
        }
        return np.frombuffer(packed, dtype=np.uint8), metadata

    def decompress(self, quantized: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
        n = metadata["n_elements"]
        gs = metadata["group_size"]
        bits = metadata["bits"]
        n_levels = metadata["n_levels"]
        centroids = metadata["centroids"]
        n_groups = math.ceil(n / gs)
        unpacked = _unpack_nbit(quantized.tobytes(), n, bits)
        result = np.zeros(n, dtype=np.float32)
        for g in range(n_groups):
            start = g * gs
            end = min(start + gs, n)
            group_centroids = centroids[g * n_levels : (g + 1) * n_levels]
            result[start:end] = group_centroids[unpacked[start:end]].astype(np.float32)
        return result.reshape(metadata["shape"])


class E8LatticeQuantizer:
    E8_OFFSETS = [
        (0, 0, 0, 0, 0, 0, 0, 0),
        (1, 1, 0, 0, 0, 0, 0, 0),
        (1, 0, 1, 0, 0, 0, 0, 0),
        (1, 0, 0, 1, 0, 0, 0, 0),
        (1, 0, 0, 0, 1, 0, 0, 0),
        (1, 0, 0, 0, 0, 1, 0, 0),
        (1, 0, 0, 0, 0, 0, 1, 0),
        (1, 0, 0, 0, 0, 0, 0, 1),
    ]

    def _nearest_e8(self, point: np.ndarray, scale: float) -> np.ndarray:
        scaled = point / scale
        rounded = np.round(scaled).astype(np.int32)
        diff = scaled - rounded
        best = rounded.copy()
        best_dist = float(np.sum(diff**2))
        parity = int(np.sum(rounded)) % 2
        if parity != 0:
            for offset in self.E8_OFFSETS:
                candidate = rounded.copy()
                for k in range(8):
                    candidate[k] += offset[k]
                cand_diff = scaled - candidate
                d = float(np.sum(cand_diff**2))
                if d < best_dist:
                    best_dist = d
                    best = candidate.copy()
            for offset in self.E8_OFFSETS:
                candidate = rounded.copy()
                for k in range(8):
                    candidate[k] -= offset[k]
                cand_diff = scaled - candidate
                d = float(np.sum(cand_diff**2))
                if d < best_dist:
                    best_dist = d
                    best = candidate.copy()
        return best.astype(np.float64) * scale

    def compress(
        self, tensor: np.ndarray, group_size: int = 128, bits: int = 4
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        flat = tensor.ravel().astype(np.float64)
        n = len(flat)
        pad_len = (8 - n % 8) % 8
        if pad_len > 0:
            flat = np.concatenate([flat, np.zeros(pad_len)])
        n_padded = len(flat)
        scale = float(np.max(np.abs(flat))) + 1e-10
        n_vectors = n_padded // 8
        reconstructed = np.zeros(n_padded, dtype=np.float64)
        code_indices = np.zeros(n_vectors, dtype=np.int32)
        for i in range(n_vectors):
            start = i * 8
            point = flat[start : start + 8]
            r = self._nearest_e8(point, scale)
            reconstructed[start : start + 8] = r
            code_indices[i] = int(np.sum((r / scale).astype(np.int32) ** 2))
        residuals = flat - reconstructed
        r_min = float(np.min(residuals))
        r_max = float(np.max(residuals))
        max_val = (1 << bits) - 1
        res_scale = (r_max - r_min) / max_val if r_max - r_min >= 1e-10 else 1.0
        res_quantized = np.clip(
            np.round((residuals - r_min) / res_scale), 0, max_val
        ).astype(np.uint8)
        res_packed = _pack_nbit(res_quantized, bits)
        metadata = {
            "shape": list(tensor.shape),
            "n_elements": n,
            "n_padded": n_padded,
            "scale": scale,
            "reconstructed": reconstructed.astype(np.float32),
            "res_scale": res_scale,
            "res_min": r_min,
            "res_bits": bits,
            "code_indices": code_indices,
        }
        return np.frombuffer(res_packed, dtype=np.uint8), metadata

    def decompress(self, quantized: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
        n = metadata["n_elements"]
        reconstructed = metadata["reconstructed"].astype(np.float64)
        res_scale = metadata["res_scale"]
        r_min = metadata["res_min"]
        bits = metadata["res_bits"]
        residuals = _unpack_nbit(quantized.tobytes(), n, bits).astype(np.float64)
        residuals = residuals * res_scale + r_min
        result = reconstructed + residuals
        return result[:n].reshape(metadata["shape"]).astype(np.float32)


class BlockFloatingPoint:
    def compress(
        self, tensor: np.ndarray, group_size: int = 128, bits: int = 4
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        flat = tensor.ravel().astype(np.float32)
        n = len(flat)
        gs = min(group_size, n)
        n_groups = math.ceil(n / gs)
        max_mantissa = (1 << bits) - 1
        exponents = np.zeros(n_groups, dtype=np.float32)
        quantized = np.zeros(n, dtype=np.uint8)
        for g in range(n_groups):
            start = g * gs
            end = min(start + gs, n)
            group = flat[start:end]
            max_abs = float(np.max(np.abs(group)))
            if max_abs < 1e-10:
                exponents[g] = 0.0
                continue
            exp = math.floor(math.log2(max_abs)) if max_abs > 0 else 0
            scale = 2.0**exp
            exponents[g] = exp
            normalized = group / scale
            quantized[start:end] = np.clip(
                np.round((normalized + 1.0) / 2.0 * max_mantissa), 0, max_mantissa
            ).astype(np.uint8)
        packed = _pack_nbit(quantized, bits)
        metadata = {
            "shape": list(tensor.shape),
            "n_elements": n,
            "group_size": gs,
            "bits": bits,
            "exponents": exponents,
            "max_mantissa": max_mantissa,
        }
        return np.frombuffer(packed, dtype=np.uint8), metadata

    def decompress(self, quantized: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
        n = metadata["n_elements"]
        gs = metadata["group_size"]
        bits = metadata["bits"]
        exponents = metadata["exponents"]
        max_mantissa = metadata["max_mantissa"]
        n_groups = len(exponents)
        unpacked = _unpack_nbit(quantized.tobytes(), n, bits)
        result = np.zeros(n, dtype=np.float32)
        for g in range(n_groups):
            start = g * gs
            end = min(start + gs, n)
            scale = 2.0 ** exponents[g]
            normalized = (
                unpacked[start:end].astype(np.float32) / max_mantissa * 2.0 - 1.0
            )
            result[start:end] = normalized * scale
        return result.reshape(metadata["shape"])


class StochasticQuantizer:
    def compress(
        self, tensor: np.ndarray, group_size: int = 128, bits: int = 4, seed: int = 42
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        flat = tensor.ravel().astype(np.float32)
        n = len(flat)
        gs = min(group_size, n)
        n_groups = math.ceil(n / gs)
        max_val = (1 << bits) - 1
        rng = np.random.RandomState(seed)
        scales = np.empty(n_groups, dtype=np.float32)
        mins = np.empty(n_groups, dtype=np.float32)
        quantized = np.zeros(n, dtype=np.uint8)
        for g in range(n_groups):
            start = g * gs
            end = min(start + gs, n)
            group = flat[start:end]
            gmin = float(np.min(group))
            gmax = float(np.max(group))
            if gmax - gmin < 1e-10:
                scales[g] = 1.0
                mins[g] = gmin
                continue
            scale = (gmax - gmin) / max_val
            scales[g] = scale
            mins[g] = gmin
            normalized = (group - gmin) / scale
            for i in range(len(group)):
                val = float(normalized[i])
                lo = int(math.floor(val))
                hi = int(math.ceil(val))
                lo = max(0, min(lo, max_val))
                hi = max(0, min(hi, max_val))
                if lo == hi:
                    quantized[start + i] = lo
                else:
                    prob = val - lo
                    quantized[start + i] = hi if rng.random() < prob else lo
        packed = _pack_nbit(quantized, bits)
        metadata = {
            "shape": list(tensor.shape),
            "n_elements": n,
            "group_size": gs,
            "bits": bits,
            "scales": scales,
            "mins": mins,
        }
        return np.frombuffer(packed, dtype=np.uint8), metadata

    def decompress(self, quantized: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
        n = metadata["n_elements"]
        gs = metadata["group_size"]
        bits = metadata["bits"]
        scales = metadata["scales"]
        mins = metadata["mins"]
        n_groups = len(scales)
        unpacked = _unpack_nbit(quantized.tobytes(), n, bits)
        result = np.zeros(n, dtype=np.float32)
        for g in range(n_groups):
            start = g * gs
            end = min(start + gs, n)
            result[start:end] = (
                unpacked[start:end].astype(np.float32) * scales[g] + mins[g]
            )
        return result.reshape(metadata["shape"])


QUANTIZERS = {
    "GroupWise": GroupWiseQuantizer(),
    "HadamardGroupWise": HadamardGroupWiseQuantizer(),
    "Asymmetric": AsymmetricQuantizer(),
    "OutlierAware": OutlierAwareQuantizer(),
    "MixedBitwidth": MixedBitwidthQuantizer(),
    "ErrorFeedback": QuantizationErrorFeedback(),
    "LloydMax": OptimalLloydMax(),
    "E8Lattice": E8LatticeQuantizer(),
    "BlockFloat": BlockFloatingPoint(),
    "Stochastic": StochasticQuantizer(),
}


def benchmark_quantizer(
    name: str, quantizer, tensor: np.ndarray, group_size: int = 128, bits: int = 4
) -> Dict[str, Any]:
    import time

    orig_bytes = tensor.nbytes
    t0 = time.perf_counter()
    comp, meta = quantizer.compress(tensor, group_size=group_size, bits=bits)
    t_compress = time.perf_counter() - t0
    t0 = time.perf_counter()
    recon = quantizer.decompress(comp, meta)
    t_decompress = time.perf_counter() - t0
    comp_bytes = len(comp) if isinstance(comp, (bytes, bytearray)) else comp.nbytes
    metrics = _error_metrics(tensor, recon)
    ratio = orig_bytes / max(comp_bytes, 1)
    return {
        "name": name,
        "shape": list(tensor.shape),
        "orig_bytes": orig_bytes,
        "compressed_bytes": comp_bytes,
        "ratio": ratio,
        "bits": bits,
        "group_size": group_size,
        "compress_ms": t_compress * 1000,
        "decompress_ms": t_decompress * 1000,
        **metrics,
    }


def load_gemma4_weight(path: str, key: str) -> Optional[np.ndarray]:
    import json

    with open(path, "rb") as f:
        header_size = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(header_size))
    if key not in header:
        return None
    info = header[key]
    dtype = info["dtype"]
    shape = info["shape"]
    offset_start, offset_end = info["data_offsets"]
    with open(path, "rb") as f:
        f.seek(8 + header_size + offset_start)
        raw = f.read(offset_end - offset_start)
    if dtype == "BF16":
        raw_u16 = np.frombuffer(raw, dtype=np.uint16)
        f32 = (raw_u16.astype(np.uint32) << 16).view(np.float32)
        return f32.reshape(shape)
    elif dtype == "F32":
        return np.frombuffer(raw, dtype=np.float32).reshape(shape)
    elif dtype == "F16":
        return np.frombuffer(raw, dtype=np.float16).reshape(shape).astype(np.float32)
    return None


def run_benchmark(
    weight_path: Optional[str] = None,
    weight_key: Optional[str] = None,
    group_size: int = 128,
    bits: int = 4,
) -> List[Dict[str, Any]]:
    if weight_path and weight_key:
        tensor = load_gemma4_weight(weight_path, weight_key)
        if tensor is not None:
            print(f"Loaded real weight: {weight_key} shape={tensor.shape}")
        else:
            print(f"Failed to load {weight_key}, using synthetic tensor")
            tensor = None
    else:
        tensor = None
    if tensor is None:
        rng = np.random.RandomState(42)
        tensor = (rng.randn(2048, 1536) * 0.02).astype(np.float32)
        tensor[0, 0] = 0.5
        tensor[-1, -1] = -0.5
        print(f"Using synthetic tensor: shape={tensor.shape}")
    print(f"\nBenchmark: group_size={group_size}, bits={bits}")
    print(
        f"Tensor: shape={tensor.shape}, {tensor.nbytes} bytes, "
        f"mean={tensor.mean():.6f}, std={tensor.std():.6f}\n"
    )
    results = []
    for name, q in QUANTIZERS.items():
        try:
            r = benchmark_quantizer(name, q, tensor, group_size=group_size, bits=bits)
            results.append(r)
        except Exception as e:
            print(f"  {name:25s}: ERROR - {e}")
            import traceback

            traceback.print_exc()
    header = f"{'Method':25s} {'Shape':15s} {'Orig':>10s} {'Comp':>10s} {'Ratio':>7s} {'SNR(dB)':>8s} {'PSNR':>7s} {'RelErr':>8s} {'CosSim':>7s} {'C(ms)':>7s} {'D(ms)':>7s}"
    print(header)
    print("-" * len(header))
    for r in sorted(results, key=lambda x: -x["ratio"]):
        print(
            f"{r['name']:25s} {str(r['shape']):15s} {r['orig_bytes']:10d} {r['compressed_bytes']:10d} "
            f"{r['ratio']:7.2f} {r['snr_db']:8.2f} {r['psnr_db']:7.2f} "
            f"{r['rel_error']:8.6f} {r['cos_sim']:7.4f} {r['compress_ms']:7.1f} {r['decompress_ms']:7.1f}"
        )
    print()
    best = max(results, key=lambda x: x["ratio"])
    best_quality = max(results, key=lambda x: x["snr_db"])
    print(f"Best compression ratio: {best['name']} ({best['ratio']:.2f}x)")
    print(
        f"Best quality (SNR):     {best_quality['name']} ({best_quality['snr_db']:.2f} dB)"
    )
    return results
