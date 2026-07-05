from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class LowerHybrid:
    """Dual Bessel: J_m(k_⊥ρ_e)J_n(k_⊥ρ_i), ω ≈ ω_LH retained."""

    name = "lower_hybrid"
    category = "novel_physics"

    def compress(self, tensor: np.ndarray, n_orders: int = 5) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        if t.ndim < 2:
            t = t.reshape(1, -1)
        m, n = t.shape

        y = np.linspace(-3, 3, m)
        x = np.linspace(-3, 3, n)
        X, Y = np.meshgrid(x, y, indexing="ij")
        R = np.sqrt(X**2 + Y**2)
        rho_e = 0.1
        rho_i = 1.0

        basis = []
        for m_idx in range(n_orders):
            for n_idx in range(n_orders):
                J_e = ShearAlfven._bessel_j(m_idx, R / max(rho_e, 1e-10))
                J_i = ShearAlfven._bessel_j(n_idx, R / max(rho_i, 1e-10))
                J_e = np.nan_to_num(J_e)
                J_i = np.nan_to_num(J_i)
                basis.append((J_e * J_i).ravel())

        basis_arr = np.stack(basis, axis=0)
        n_basis = len(basis)
        vec = t.ravel()
        A = basis_arr @ basis_arr.T + np.eye(n_basis) * 1e-8
        coeffs = np.linalg.solve(A, basis_arr @ vec)

        approx = (coeffs[:, None] * basis_arr).sum(axis=0)
        residual = vec - approx
        thr = np.percentile(np.abs(residual), 90)
        rmask = np.abs(residual) > thr
        ridx = np.argwhere(rmask)[:, 0]
        rvals = residual[rmask]

        meta = dict(shape=tensor.shape, n_orders=n_orders, n_res=len(ridx))
        data = _serialize(coeffs.astype(np.float32))
        data += _serialize(ridx.astype(np.int32)) + rvals.astype(np.float16).tobytes()
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        n_orders = metadata["n_orders"]
        n_res = metadata["n_res"]
        m, n = shape
        n_basis = n_orders * n_orders

        coeffs = _deserialize(data[: n_basis * 4])
        pos = n_basis * 4

        y = np.linspace(-3, 3, m)
        x = np.linspace(-3, 3, n)
        X, Y = np.meshgrid(x, y, indexing="ij")
        R = np.sqrt(X**2 + Y**2)
        rho_e = 0.1
        rho_i = 1.0

        recon = np.zeros(m * n, dtype=np.float64)
        idx = 0
        for m_idx in range(n_orders):
            for n_idx in range(n_orders):
                if idx < len(coeffs):
                    J_e = ShearAlfven._bessel_j(m_idx, R / max(rho_e, 1e-10))
                    J_i = ShearAlfven._bessel_j(n_idx, R / max(rho_i, 1e-10))
                    J_e = np.nan_to_num(J_e)
                    J_i = np.nan_to_num(J_i)
                    recon += coeffs[idx] * (J_e * J_i).ravel()
                    idx += 1

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
