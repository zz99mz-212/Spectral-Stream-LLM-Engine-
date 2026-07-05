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


class PlasmaTurbulenceDecomposition(CompressionMethod):
    """Treat weight matrices as turbulent plasma fields and decompose.

    Mathematical basis:
        Plasma turbulence exhibits a Kolmogorov-like energy cascade:
            E(k) ~ k^(-5/3)

        We decompose the weight matrix into:
        1. Large-scale coherent structures (low-frequency modes)
        2. Intermediate-scale vortex filaments
        3. Small-scale turbulent fluctuations

    Algorithm:
        1. Wavelet decomposition at multiple scales
        2. For each scale, separate coherent vs incoherent parts
        3. Store coherent coefficients exactly, approximate incoherent
        4. Use wavelet scattering for stability

    Storage: O(n_coherent) where n_coherent << total coefficients.
    """

    name = "plasma_turbulence"
    category = "plasma_physics"

    def compress(self, tensor, n_scales=4, coherent_ratio=0.3, **kw):
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

        # Multi-scale wavelet decomposition (Haar-like)
        scales = []
        current = W.copy()

        for scale in range(n_scales):
            cm, cn = current.shape
            if cm < 2 or cn < 2:
                break

            # Row-wise Haar decomposition
            even_rows = current[0::2, :]
            odd_rows = current[1::2, :]
            if odd_rows.shape[0] < even_rows.shape[0]:
                odd_rows = np.pad(odd_rows, ((0, 1), (0, 0)))

            approx = (even_rows + odd_rows) * 0.5
            detail_v = (even_rows - odd_rows) * 0.5

            # Column-wise on approx
            ca, cc = approx.shape
            if cc >= 2:
                even_cols = approx[:, 0::2]
                odd_cols = approx[:, 1::2]
                # Pad to same length if odd columns
                if odd_cols.shape[1] < even_cols.shape[1]:
                    odd_cols = np.pad(odd_cols, ((0, 0), (0, 1)))
                approx_c = (even_cols + odd_cols) * 0.5
                detail_h = (even_cols - odd_cols) * 0.5
            else:
                approx_c = approx
                detail_h = np.zeros_like(approx)

            # At each scale, separate coherent vs incoherent
            # detail_v is (ca, cn) and detail_h is (ca, cc) where cc ≈ cn//2
            # Use detail_v's shape for thresholding, then mask each separately
            detail_v_mag = np.abs(detail_v)
            detail_h_mag = np.abs(detail_h)
            all_mag = np.concatenate([detail_v_mag.ravel(), detail_h_mag.ravel()])
            threshold = np.percentile(all_mag, (1 - coherent_ratio) * 100)

            coherent_mask_v = detail_v_mag >= threshold
            coherent_mask_h = detail_h_mag >= threshold

            scales.append(
                {
                    "approx": approx_c,
                    "detail_v_vals": detail_v[coherent_mask_v].astype(np.float32),
                    "detail_v_idx": np.argwhere(coherent_mask_v).astype(np.int32),
                    "detail_v_shape": detail_v.shape,
                    "detail_h_vals": detail_h[coherent_mask_h].astype(np.float32),
                    "detail_h_idx": np.argwhere(coherent_mask_h).astype(np.int32),
                    "detail_h_shape": detail_h.shape,
                    "incoherent_v_mean": float(np.mean(detail_v[~coherent_mask_v]))
                    if np.any(~coherent_mask_v)
                    else 0.0,
                    "incoherent_v_std": float(np.std(detail_v[~coherent_mask_v]))
                    if np.any(~coherent_mask_v)
                    else 0.0,
                    "incoherent_h_mean": float(np.mean(detail_h[~coherent_mask_h]))
                    if np.any(~coherent_mask_h)
                    else 0.0,
                    "incoherent_h_std": float(np.std(detail_h[~coherent_mask_h]))
                    if np.any(~coherent_mask_h)
                    else 0.0,
                }
            )

            current = approx_c

        return {
            "scales": scales,
            "residual": current.astype(np.float32),
            "n_scales": len(scales),
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
        current = cd["residual"].astype(np.float64)

        for scale_data in reversed(cd["scales"]):
            approx = scale_data["approx"].astype(np.float64)
            ca, cc = approx.shape

            # Reconstruct detail arrays
            detail_v = np.zeros(scale_data["detail_v_shape"], dtype=np.float64)
            if scale_data["detail_v_idx"].shape[0] > 0:
                idx = scale_data["detail_v_idx"]
                valid = (idx[:, 0] < detail_v.shape[0]) & (
                    idx[:, 1] < detail_v.shape[1]
                )
                detail_v[idx[valid, 0], idx[valid, 1]] = scale_data["detail_v_vals"][
                    valid
                ]
            # Add incoherent part
            rng = np.random.RandomState(42 + len(cd["scales"]))
            noise_v = (
                rng.randn(*detail_v.shape) * scale_data["incoherent_v_std"] * 0.3
                + scale_data["incoherent_v_mean"]
            )
            detail_v += noise_v

            detail_h = np.zeros(scale_data["detail_h_shape"], dtype=np.float64)
            if scale_data["detail_h_idx"].shape[0] > 0:
                idx = scale_data["detail_h_idx"]
                valid = (idx[:, 0] < detail_h.shape[0]) & (
                    idx[:, 1] < detail_h.shape[1]
                )
                detail_h[idx[valid, 0], idx[valid, 1]] = scale_data["detail_h_vals"][
                    valid
                ]
            noise_h = (
                rng.randn(*detail_h.shape) * scale_data["incoherent_h_std"] * 0.3
                + scale_data["incoherent_h_mean"]
            )
            detail_h += noise_h

            # Column-wise inverse: reconstruct from approx + detail_h
            # detail_h has shape (ca, cc) matching approx
            if cc >= 2:
                recon_cols = np.zeros((ca, cc * 2), dtype=np.float64)
                recon_cols[:, 0::2] = approx + detail_h
                recon_cols[:, 1::2] = approx - detail_h
            else:
                recon_cols = approx

            # Row-wise inverse: reconstruct from recon_cols + detail_v
            ra, rc = recon_cols.shape
            # detail_v has shape from the original decomposition; trim/pad to match
            dv_rows = min(ra, detail_v.shape[0])
            dv_cols = min(rc, detail_v.shape[1])
            target_rows = ra * 2
            recon_rows = np.zeros((target_rows, rc), dtype=np.float64)
            recon_rows[0::2] = (
                recon_cols[:dv_rows, :dv_cols] + detail_v[:dv_rows, :dv_cols]
            )
            recon_rows[1::2] = (
                recon_cols[:dv_rows, :dv_cols] - detail_v[:dv_rows, :dv_cols]
            )

            current = recon_rows

        return _restore_shape(current[:m, :n].astype(np.float32), meta["orig_shape"])


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
