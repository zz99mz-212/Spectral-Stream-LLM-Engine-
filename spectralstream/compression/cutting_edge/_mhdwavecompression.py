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

class MHDWaveCompression(CompressionMethod):
    """Magnetohydrodynamic wave decomposition of weight matrices.

    Mathematical basis:
        MHD waves consist of three types:
        1. Alfvén waves: v_A = B / sqrt(mu_0 * rho), incompressible
        2. Acoustic (sound) waves: c_s = sqrt(gamma * p / rho)
        3. Entropy waves: purely advective, no pressure perturbation

        We decompose W into these three components:
            W = W_Alfven + W_acoustic + W_entropy

    Algorithm:
        1. Compute divergence-free (Alfvén) and curl-free (acoustic) parts
        2. Helmholtz decomposition: W = grad(phi) + curl(A)
        3. Store each component with its parameters

    This exploits the physical structure of weight correlations.
    """
    name = "mhd_wave"
    category = "plasma_physics"

    def compress(self, tensor, n_components=16, **kw):
        t, orig = _ensure_2d(tensor)
        m, n = t.shape
        W = t.astype(np.float64)

        # Helmholtz-like decomposition via spectral methods
        # Divergence-free part (Alfvén): rows of curl(A)
        # Curl-free part (acoustic): gradient of scalar potential

        # Compute gradients
        grad_y, grad_x = np.gradient(W)
        div_field = grad_x  # simplified: treat as 1D divergence along columns

        # Scalar potential (curl-free part) via Poisson equation in Fourier domain
        f_div = np.fft.fft2(div_field)
        kx = np.fft.fftfreq(n)[None, :]
        ky = np.fft.fftfreq(m)[:, None]
        k_sq = kx ** 2 + ky ** 2
        k_sq[0, 0] = 1.0  # avoid division by zero

        phi_hat = f_div / (1j * 2 * np.pi * k_sq + 1e-10)
        acoustic_part = np.fft.ifft2(phi_hat).real

        # Divergence-free part (Alfvén): remaining
        alfven_part = W - acoustic_part

        # Entropy part: row-wise mean (advective component)
        entropy_part = np.mean(W, axis=1, keepdims=True) * np.ones_like(W)

        # Compress each component via DFT truncation
        def compress_component(C, keep):
            fc = np.fft.fft2(C)
            flat_amp = np.abs(fc.ravel())
            top_idx = np.argsort(flat_amp)[::-1][:keep]
            return {
                "re": fc.ravel()[top_idx].real.astype(np.float32),
                "im": fc.ravel()[top_idx].imag.astype(np.float32),
                "idx": top_idx.astype(np.int32),
                "shape": fc.shape,
            }

        n_per = max(1, n_components // 3)
        alfven_c = compress_component(alfven_part, n_per)
        acoustic_c = compress_component(acoustic_part, n_per)
        entropy_c = compress_component(entropy_part, n_per)

        return {
            "alfven": alfven_c,
            "acoustic": acoustic_c,
            "entropy": entropy_c,
            "n_components": n_components,
            "shape": t.shape,
        }, {"orig_shape": orig}

    def decompress(self, cd, meta):
        m, n = meta["orig_shape"][:2] if len(meta["orig_shape"]) >= 2 else (1, meta["orig_shape"][0])

        def decompress_component(comp_data):
            fc = np.zeros(comp_data["shape"], dtype=np.complex128)
            fc.ravel()[comp_data["idx"]] = comp_data["re"].astype(np.float64) + 1j * comp_data["im"].astype(np.float64)
            return np.fft.ifft2(fc).real

        alfven = decompress_component(cd["alfven"])
        acoustic = decompress_component(cd["acoustic"])
        entropy = decompress_component(cd["entropy"])

        result = alfven + acoustic + entropy * 0.1
        return _restore_shape(result[:m, :n].astype(np.float32), meta["orig_shape"])

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

