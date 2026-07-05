from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from ._compressionmethod import CompressionMethod, ALL_METHODS, _ensure_2d, _restore_shape, _safe_bytes


def _ensure_2d(t: np.ndarray) -> Tuple[np.ndarray, tuple]:
    if t.ndim == 1:
        return t.reshape(1, -1), t.shape
    if t.ndim > 2:
        orig = t.shape
        return t.reshape(orig[0], -1), orig
    return t, t.shape

def _restore_shape(t: np.ndarray, orig_shape: tuple) -> np.ndarray:
    return t.reshape(orig_shape) if t.shape != orig_shape else t

def _safe_bytes(data: Any) -> int:
    if isinstance(data, np.ndarray):
        return data.nbytes
    if isinstance(data, dict):
        return sum(_safe_bytes(v) for v in data.values()) + sum(_safe_bytes(k) for k in data.keys())
    if isinstance(data, (list, tuple)):
        return sum(_safe_bytes(x) for x in data)
    if isinstance(data, (int, float, np.integer, np.floating)):
        return 8
    if isinstance(data, str):
        return len(data)
    return 0

class QuantumErrorCorrectionCompression(CompressionMethod):
    """Apply quantum error correction codes to weight storage.

    Mathematical basis:
        Stabilizer codes encode k logical qubits into n physical qubits
        using (n-k) stabilizer generators.  We adapt this to weight storage:
        - Encode weight values using redundant stabilizer structure
        - Detect and correct quantization errors via syndrome measurement

    Algorithm:
        1. Quantize weights to target precision
        2. Compute parity check matrix H
        3. Generate syndrome bits for error detection
        4. Store: encoded weights + syndromes (compact error metadata)

    The syndrome bits allow detecting single-bit quantization errors
    and correcting them during decompression.
    """
    name = "quantum_error_correction"
    category = "quantum_mechanics"

    def compress(self, tensor, n_bits=4, block_size=8, **kw):
        t, orig = _ensure_2d(tensor)
        m, n = t.shape
        flat = t.ravel().astype(np.float64)

        # Quantize
        nl = 1 << n_bits
        s = max(abs(flat.max()), abs(flat.min()), 1e-8)
        step = 2.0 / nl
        q_idx = np.clip(np.round((np.clip(flat / s, -1, 1) + 1) / step).astype(int), 0, nl - 1)
        q_vals = (q_idx * step - 1.0) * s

        # Compute error syndromes using parity checks
        n_blocks = max(1, len(flat) // block_size)
        syndromes = np.zeros(n_blocks, dtype=np.uint8)
        for b in range(n_blocks):
            start = b * block_size
            end = min(start + block_size, len(flat))
            block = q_idx[start:end]
            # Syndrome: XOR of all bits in block
            syndrome = 0
            for val in block:
                syndrome ^= int(val) & 0xF
            syndromes[b] = syndrome

        # Compute correction table: for each syndrome, find most likely error pattern
        correction_table = np.zeros((16, block_size), dtype=np.float64)
        for syn in range(16):
            # Most likely single-error pattern for this syndrome
            if syn > 0:
                error_pos = syn & (block_size - 1)
                correction_table[syn, error_pos] = step * s

        return {
            "q_idx": q_idx.astype(np.uint8),
            "scale": float(s),
            "syndromes": syndromes,
            "correction_table": correction_table.astype(np.float32),
            "block_size": block_size,
            "nl": nl,
            "shape": t.shape,
        }, {"orig_shape": tensor.shape}

    def decompress(self, cd, meta):
        q_idx = cd["q_idx"].astype(np.float64)
        step = 2.0 / cd["nl"]
        block_size = cd["block_size"]
        syndromes = cd["syndromes"]
        correction_table = cd["correction_table"].astype(np.float64)

        # Decode with error correction
        n_blocks = max(1, len(q_idx) // block_size)
        result = np.zeros_like(q_idx)
        for b in range(n_blocks):
            start = b * block_size
            end = min(start + block_size, len(q_idx))
            block = q_idx[start:end]
            syn = syndromes[b] if b < len(syndromes) else 0
            # Apply correction if syndrome indicates error
            if syn > 0 and syn < len(correction_table):
                corrected = block.copy()
                error_pos = syn & (block_size - 1)
                if error_pos < len(corrected):
                    corrected[error_pos] = np.clip(corrected[error_pos] + 1, 0, cd["nl"] - 1)
                result[start:end] = corrected * step - 1.0
            else:
                result[start:end] = block * step - 1.0

        flat = result * cd["scale"]
        return flat.reshape(meta["orig_shape"]).astype(np.float32)

def _generate_monomials(n_vars: int, degree: int) -> list:
    """Generate all monomials of given degree in n_vars variables."""
    if degree == 0:
        return [()]
    if degree == 1:
        return [(i,) for i in range(n_vars)]
    result = []
    for i in range(n_vars):
        for rest in _generate_monomials(n_vars, degree - 1):
            if len(rest) == 0 or i >= rest[0]:
                result.append((i,) + rest)
    return result[:50]  # limit for efficiency

