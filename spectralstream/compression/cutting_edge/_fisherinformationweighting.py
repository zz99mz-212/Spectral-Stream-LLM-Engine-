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

class FisherInformationWeighting(CompressionMethod):
    """Use Fisher information matrix to allocate storage by information content.

    Mathematical basis:
        The Fisher information matrix I(theta) measures how much information
        the observable data carries about parameters theta:
            I(theta) = E[ (d log p(x|theta) / d theta)^2 ]

        For Gaussian weights: I(mu) = n/sigma^2, I(sigma) = 2n/sigma^4

        Dimensions with high Fisher information are more "informative"
        and deserve more storage bits.

    Algorithm:
        1. Compute per-dimension Fisher information: I_k = n / var(w_k)
        2. Allocate bits proportional to sqrt(I_k) (optimal allocation)
        3. Quantize each dimension with its allocated precision
        4. Store: scales + quantized indices

    This achieves near-optimal rate allocation across dimensions.
    """
    name = "fisher_information_weighted"
    category = "information_theory"

    def compress(self, tensor, total_bits=4, **kw):
        t, orig = _ensure_2d(tensor)
        m, n = t.shape

        # Per-column Fisher information (for Gaussian model)
        col_var = np.var(t.astype(np.float64), axis=0) + 1e-10
        fisher_info = n / col_var  # I(mu_k) = n / sigma_k^2

        # Optimal bit allocation: b_k ~ sqrt(I_k) * total_bits / sum(sqrt(I_k))
        sqrt_fisher = np.sqrt(fisher_info)
        total_fisher = sqrt_fisher.sum() + 1e-10
        col_bits = np.clip(np.round(sqrt_fisher / total_fisher * total_bits * n).astype(int), 1, 8)

        # Quantize each column
        quantized = []
        for j in range(n):
            nb = int(col_bits[j])
            nl = 1 << nb
            col = t[:, j].astype(np.float64)
            s = max(abs(col.max()), abs(col.min()), 1e-8)
            step = 2.0 / nl
            idx = np.clip(np.round((np.clip(col / s, -1, 1) + 1) / step).astype(int), 0, nl - 1)
            quantized.append({
                "idx": idx.astype(np.uint8),
                "scale": float(s),
                "nl": nl,
            })

        return {
            "quantized": quantized,
            "col_bits": col_bits.astype(np.uint8),
            "fisher_info": fisher_info.astype(np.float32),
            "shape": t.shape,
        }, {"orig_shape": orig}

    def decompress(self, cd, meta):
        m = cd["shape"][0]
        n = cd["shape"][1]
        result = np.zeros((m, n), dtype=np.float64)

        for j, q in enumerate(cd["quantized"]):
            step = 2.0 / q["nl"]
            result[:, j] = (q["idx"].astype(np.float64) * step - 1.0) * q["scale"]

        return _restore_shape(result.astype(np.float32), meta["orig_shape"])

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

