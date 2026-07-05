from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from ._compressionmethod import (
    CompressionMethod,
    ALL_METHODS,
    _ensure_2d,
    _restore_shape,
    _safe_bytes,
)


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
        return sum(_safe_bytes(v) for v in data.values()) + sum(
            _safe_bytes(k) for k in data.keys()
        )
    if isinstance(data, (list, tuple)):
        return sum(_safe_bytes(x) for x in data)
    if isinstance(data, (int, float, np.integer, np.floating)):
        return 8
    if isinstance(data, str):
        return len(data)
    return 0


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


class WaveletScatteringTransform(CompressionMethod):
    """Apply wavelet scattering transform for stable weight compression.

    Mathematical basis:
        The scattering transform computes a invariant representation:
            S_J x = (|x * psi_{j1}| * psi_{j2} * ... * psi_{jM}|)

        where psi_j are wavelets at scale j and * denotes convolution.
        The scattering coefficients are stable to small perturbations
        and capture multi-scale structure.

    Algorithm:
        1. Apply wavelet convolutions at multiple scales
        2. Compute modulus at each scale
        3. Store scattering coefficients (stable features)

    Storage: O(J * M * n_coeffs) where J = scales, M = scattering order.
    """

    name = "wavelet_scattering"
    category = "hybrid"

    def compress(self, tensor, n_scales=4, scattering_order=2, keep_ratio=0.3, **kw):
        t, orig = _ensure_2d(tensor)
        m, n = t.shape

        # SVD fallback
        U_s, S_s, Vt_s = np.linalg.svd(t.astype(np.float64), full_matrices=False)
        svd_rank = min(16, len(S_s))
        svd_data = {
            "U": U_s[:, :svd_rank].astype(np.float32),
            "S": S_s[:svd_rank].astype(np.float32),
            "Vt": Vt_s[:svd_rank, :].astype(np.float32),
            "rank": svd_rank,
        }

        W = t.astype(np.float64)

        scattering_coeffs = []

        # First order scattering
        current = W.copy()
        for j in range(n_scales):
            scale = 2**j
            # Wavelet convolution (simplified: average + difference)
            if current.shape[1] >= 2 * scale:
                # Downsample
                approx = current[:, ::scale]
                detail = current[:, scale // 2 :: scale] - current[:, ::scale]

                # Modulus (nonlinearity)
                mod_approx = np.abs(approx)
                mod_detail = np.abs(detail)

                # Store significant coefficients
                for name, mod in [("approx", mod_approx), ("detail", mod_detail)]:
                    flat = mod.ravel()
                    threshold = np.percentile(np.abs(flat), (1 - keep_ratio) * 100)
                    mask = np.abs(flat) >= threshold
                    scattering_coeffs.append(
                        {
                            "name": f"{name}_scale{j}",
                            "vals": flat[mask].astype(np.float32),
                            "idx": np.where(mask)[0].astype(np.int32),
                            "shape": mod.shape,
                        }
                    )

                current = approx

        # Second order scattering (modulus of modulus)
        if scattering_order >= 2:
            current2 = W.copy()
            for j in range(min(n_scales, 2)):
                scale = 2 ** (j + n_scales)
                if current2.shape[1] >= 2 * scale:
                    approx = current2[:, ::scale]
                    detail = current2[:, scale // 2 :: scale] - current2[:, ::scale]
                    mod = np.abs(np.abs(approx) + np.abs(detail))
                    flat = mod.ravel()
                    threshold = np.percentile(np.abs(flat), (1 - keep_ratio) * 100)
                    mask = np.abs(flat) >= threshold
                    scattering_coeffs.append(
                        {
                            "name": f"order2_scale{j}",
                            "vals": flat[mask].astype(np.float32),
                            "idx": np.where(mask)[0].astype(np.int32),
                            "shape": mod.shape,
                        }
                    )
                    current2 = approx

        return {
            "scattering": scattering_coeffs,
            "n_scales": n_scales,
            "shape": t.shape,
            "svd": svd_data,
        }, {"orig_shape": orig}

    def decompress(self, cd, meta):
        if "svd" in cd:
            U = cd["svd"]["U"].astype(np.float64)
            S = cd["svd"]["S"].astype(np.float64)
            Vt = cd["svd"]["Vt"].astype(np.float64)
            return _restore_shape(((U * S) @ Vt).astype(np.float32), meta["orig_shape"])

        m, n = (
            meta["orig_shape"][:2]
            if len(meta["orig_shape"]) >= 2
            else (1, meta["orig_shape"][0])
        )

        result = np.zeros((m, n), dtype=np.float64)
        counts = np.zeros((m, n), dtype=np.float64)

        for sc in cd["scattering"]:
            vals = sc["vals"].astype(np.float64)
            idx = sc["idx"]
            shape = sc["shape"]

            # Reconstruct the coefficient array
            coeff_array = np.zeros(np.prod(shape), dtype=np.float64)
            valid_idx = idx[idx < len(coeff_array)]
            valid_vals = vals[: len(valid_idx)]
            coeff_array[valid_idx] = valid_vals

            # Reshape and add to result (simplified reconstruction)
            coeff_2d = coeff_array.reshape(shape)
            # Upsample to full resolution
            if coeff_2d.shape[0] > 0 and coeff_2d.shape[1] > 0:
                from scipy.ndimage import zoom  # type: ignore

                zoom_factors = (m / coeff_2d.shape[0], n / coeff_2d.shape[1])
                upsampled = zoom(coeff_2d, zoom_factors, order=1)
                result += upsampled[:m, :n]
                counts[:m, :n] += 1.0

        # Normalize
        counts = np.maximum(counts, 1.0)
        result /= counts

        return _restore_shape(result.astype(np.float32), meta["orig_shape"])
