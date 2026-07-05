from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class ShearAlfven:
    """Shear Alfvén wave basis: Bessel function J_n(k_⊥ρ_s)exp(inθ)."""

    name = "shear_alfven"
    category = "novel_physics"

    @staticmethod
    def _bessel_j(n: int, x: np.ndarray) -> np.ndarray:
        """Bessel J_n(x) via power series (NumPy only)."""
        x = np.asarray(x)
        result = np.zeros_like(x, dtype=np.float64)
        if n == 0:
            term = np.ones_like(x, dtype=np.float64)
            for k in range(40):
                result += term
                term *= -(x**2) / (4 * (k + 1) ** 2)
                if np.max(np.abs(term)) < 1e-15:
                    break
        elif n == 1:
            term = x / 2.0
            for k in range(40):
                result += term
                term *= -(x**2) / (4 * (k + 1) * (k + 2))
                if np.max(np.abs(term)) < 1e-15:
                    break
        else:
            j0 = ShearAlfven._bessel_j(0, x)
            j1 = ShearAlfven._bessel_j(1, x)
            for nu in range(1, n):
                jn = 2 * nu / (x + 1e-30) * j1 - j0
                j0, j1 = j1, jn
            result = j1
        return np.nan_to_num(result)

    def compress(self, tensor: np.ndarray, n_bessel: int = 6) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        if t.ndim < 2:
            t = t.reshape(1, -1)
        m, n = t.shape

        y = np.linspace(-5, 5, m)
        x = np.linspace(-5, 5, n)
        X, Y = np.meshgrid(x, y, indexing="ij")
        R = np.sqrt(X**2 + Y**2)
        Theta = np.arctan2(Y, X)

        basis = []
        for nu in range(n_bessel):
            J = ShearAlfven._bessel_j(nu, R)
            basis.append(J * np.cos(nu * Theta))
            basis.append(J * np.sin(nu * Theta))

        basis_arr = np.stack(basis, axis=0)
        n_basis = len(basis)
        A = basis_arr.reshape(n_basis, -1)
        vec = t.ravel()

        coeffs = A @ vec / (np.linalg.norm(A, axis=1) ** 2 + 1e-30)
        approx = (coeffs[:, None] * A).sum(axis=0)
        residual = vec - approx
        thr = np.percentile(np.abs(residual), 90)
        rmask = np.abs(residual) > thr
        ridx = np.argwhere(rmask)[:, 0]
        rvals = residual[rmask]

        meta = dict(shape=tensor.shape, n_bessel=n_bessel, n_res=len(ridx))
        data = _serialize(coeffs.astype(np.float32))
        data += _serialize(ridx.astype(np.int32)) + rvals.astype(np.float16).tobytes()
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        n_bessel = metadata["n_bessel"]
        n_res = metadata["n_res"]
        m, n = shape

        n_basis = n_bessel * 2
        coeffs = _deserialize(data[: n_basis * 4])
        pos = n_basis * 4

        y = np.linspace(-5, 5, m)
        x = np.linspace(-5, 5, n)
        X, Y = np.meshgrid(x, y, indexing="ij")
        R = np.sqrt(X**2 + Y**2)
        Theta = np.arctan2(Y, X)

        recon = np.zeros(m * n, dtype=np.float64)
        for nu in range(n_bessel):
            J = ShearAlfven._bessel_j(nu, R)
            if nu < len(coeffs):
                recon += coeffs[nu * 2] * (J * np.cos(nu * Theta)).ravel()
            if nu * 2 + 1 < len(coeffs):
                recon += coeffs[nu * 2 + 1] * (J * np.sin(nu * Theta)).ravel()

        if n_res > 0:
            ridx = _deserialize(data[pos : pos + n_res * 4]).astype(int)
            pos += n_res * 4
            rvals = np.frombuffer(data[pos : pos + n_res * 2], dtype=np.float16).astype(
                np.float64
            )
            for i, v in zip(ridx, rvals):
                if i < len(recon):
                    recon[i] += v

        return recon.reshape(shape).astype(np.float32)
