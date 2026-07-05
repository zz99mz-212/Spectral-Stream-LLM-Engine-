from __future__ import annotations

import json
import logging
import math
import struct
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from spectralstream.core.math_primitives import (
    dct,
    dct_2d,
    idct,
    idct_2d,
    LloydMaxQuantizer,
)


def _format_size(n: int) -> str:
    if n >= 1024**3:
        return f"{n / 1024**3:.1f} GB"
    if n >= 1024**2:
        return f"{n / 1024**2:.1f} MB"
    if n >= 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n} B"


class FrequencyDomainCompressor:
    """Compress tensors by keeping only significant DCT coefficients.

    Uses energy-based pruning: only coefficients needed to retain
    `keep_energy` fraction of total energy are stored.
    """

    def __init__(
        self,
        keep_energy: float = 0.95,
        block_size: int = 32,
        n_bits: int = 8,
    ) -> None:
        """
        Args:
            keep_energy: Fraction of DCT energy to retain (0, 1].
            block_size: DCT block size.
            n_bits: Quantization bits for kept coefficients.
        """
        self.keep_energy = keep_energy
        self.block_size = block_size
        self.n_bits = n_bits

    def compress(self, tensor: np.ndarray) -> tuple:
        """Compress tensor using frequency-domain DCT.

        Returns (bytes, dict) tuple.
        """
        t = np.asarray(tensor, dtype=np.float64)
        orig_shape = t.shape

        if t.ndim == 1:
            coeffs = dct(t)
            total_energy = float(np.sum(coeffs**2))
            if total_energy < 1e-20:
                return b"", {
                    "orig_shape": list(orig_shape),
                    "keep_energy": self.keep_energy,
                    "n_bits": self.n_bits,
                }
            sorted_mag = np.sort(np.abs(coeffs.ravel()))[::-1]
            cumsum = np.cumsum(sorted_mag**2)
            n_keep = int(np.searchsorted(cumsum / total_energy, self.keep_energy)) + 1
            n_keep = max(1, min(n_keep, len(coeffs)))
            top_indices = np.argsort(np.abs(coeffs.ravel()))[::-1][:n_keep]
            values = coeffs.ravel()[top_indices]
            quantizer = LloydMaxQuantizer(self.n_bits)
            quantizer.train(values)
            quantized = quantizer.quantize(values)
            data = json.dumps(
                {
                    "indices": top_indices.tolist(),
                    "values": quantized.tolist(),
                    "scale": float(quantizer.scale),
                }
            ).encode("utf-8")
            return data, {
                "orig_shape": list(orig_shape),
                "keep_energy": self.keep_energy,
                "n_bits": self.n_bits,
                "total_coeffs": len(coeffs),
                "n_coeffs": n_keep,
            }

        # 2D: block-wise DCT
        rows, cols = t.shape
        bs = min(self.block_size, rows, cols)
        bs = max(2, bs)
        pad_r = (bs - rows % bs) % bs
        pad_c = (bs - cols % bs) % bs
        t_padded = (
            np.pad(t, ((0, pad_r), (0, pad_c)), mode="constant")
            if (pad_r or pad_c)
            else t
        )

        all_indices: List[List[int]] = []
        all_values_list: List[List[float]] = []
        n_blocks_r = t_padded.shape[0] // bs
        n_blocks_c = t_padded.shape[1] // bs
        block_sizes = []

        for br in range(n_blocks_r):
            for bc in range(n_blocks_c):
                block = t_padded[br * bs : (br + 1) * bs, bc * bs : (bc + 1) * bs]
                coeffs_2d = dct_2d(block)
                flat = coeffs_2d.ravel()
                total_e = float(np.sum(flat**2))
                if total_e < 1e-20:
                    block_sizes.append(0)
                    all_indices.append([])
                    all_values_list.append([])
                    continue
                sorted_mag = np.sort(np.abs(flat))[::-1]
                cumsum = np.cumsum(sorted_mag**2)
                n_keep = int(np.searchsorted(cumsum / total_e, self.keep_energy)) + 1
                n_keep = max(1, min(n_keep, len(flat)))
                top_idx = np.argsort(np.abs(flat))[::-1][:n_keep]
                all_indices.append(top_idx.tolist())
                all_values_list.append(flat[top_idx].tolist())
                block_sizes.append(n_keep)

        all_values_flat = [v for subl in all_values_list for v in subl]
        all_values_arr = (
            np.array(all_values_flat, dtype=np.float64)
            if all_values_flat
            else np.array([])
        )
        if len(all_values_arr) > 0:
            quantizer = LloydMaxQuantizer(self.n_bits)
            quantizer.train(all_values_arr)
            quantized_all = quantizer.quantize(all_values_arr)
            quantized_list = quantized_all.tolist()
        else:
            quantized_list = []

        data = json.dumps(
            {
                "block_sizes": block_sizes,
                "all_indices": all_indices,
                "n_blocks_r": n_blocks_r,
                "n_blocks_c": n_blocks_c,
                "block_size": bs,
                "padded_shape": list(t_padded.shape),
                "values": quantized_list,
            }
        ).encode("utf-8")

        return data, {
            "orig_shape": list(orig_shape),
            "keep_energy": self.keep_energy,
            "n_bits": self.n_bits,
            "n_coeffs": len(all_values_flat),
            "total_coeffs": rows * cols,
        }

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        """Decompress frequency-domain compressed data."""
        orig_shape = tuple(metadata["orig_shape"])
        if not data:
            return np.zeros(orig_shape, dtype=np.float32)

        meta = json.loads(data)

        if len(orig_shape) == 1:
            n_coeffs = metadata.get("total_coeffs", orig_shape[0])
            coeffs = np.zeros(n_coeffs, dtype=np.float64)
            indices = np.array(meta["indices"])
            values = np.array(meta["values"])
            coeffs[indices] = values
            return idct(coeffs).astype(np.float32)

        block_sizes = meta["block_sizes"]
        all_indices = meta["all_indices"]
        n_blocks_r = meta["n_blocks_r"]
        n_blocks_c = meta["n_blocks_c"]
        bs = meta["block_size"]
        values = meta["values"]
        val_idx = 0

        t_padded = np.zeros(meta["padded_shape"], dtype=np.float64)
        for i, n_keep in enumerate(block_sizes):
            br = i // n_blocks_c
            bc = i % n_blocks_c
            if n_keep == 0:
                continue
            block_coeffs = np.zeros(bs * bs, dtype=np.float64)
            idx_list = all_indices[i]
            if val_idx + n_keep <= len(values):
                for j in range(n_keep):
                    block_coeffs[idx_list[j]] = values[val_idx + j]
                val_idx += n_keep
            block_coeffs = block_coeffs.reshape(bs, bs)
            t_padded[br * bs : (br + 1) * bs, bc * bs : (bc + 1) * bs] = idct_2d(
                block_coeffs
            )

        result = (
            t_padded[: orig_shape[0], : orig_shape[1]]
            if len(orig_shape) >= 2
            else t_padded.ravel()[: int(np.prod(orig_shape))].reshape(orig_shape)
        )
        return result.astype(np.float32)
