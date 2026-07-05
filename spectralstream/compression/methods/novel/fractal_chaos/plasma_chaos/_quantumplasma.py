from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class QuantumPlasma:
    """Quantum plasmonics: ω² = ω_p² + ℏ²k⁴/4m_e² + 3k²v_th², quantum diffraction cutoff."""

    name = "quantum_plasma"
    category = "novel_physics"

    def compress(
        self, tensor: np.ndarray, hbar: float = 1.0, m_e: float = 1.0
    ) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        if t.ndim < 2:
            t = t.reshape(1, -1)
        m, n = t.shape

        F = np.fft.fft2(t)
        shifted = np.fft.fftshift(F)
        kx = np.fft.fftshift(np.fft.fftfreq(m)) * 2 * np.pi
        ky = np.fft.fftshift(np.fft.fftfreq(n)) * 2 * np.pi
        KX, KY = np.meshgrid(kx, ky, indexing="ij")
        k_sq = KX**2 + KY**2 + 1e-30

        omega_p_sq = np.mean(np.abs(shifted) ** 2)
        v_th_sq = np.var(t.ravel())
        quantum_term = hbar**2 * k_sq**2 / (4 * m_e**2)
        thermal_term = 3 * k_sq * v_th_sq
        omega_sq = omega_p_sq + quantum_term + thermal_term
        quantum_cutoff = omega_sq > np.percentile(omega_sq, 30)

        kept_v = shifted[quantum_cutoff]
        kept_i = np.argwhere(quantum_cutoff)

        meta = dict(
            shape=tensor.shape, n_kept=int(np.sum(quantum_cutoff)), hbar=hbar, m_e=m_e
        )
        data = _serialize(kept_i.astype(np.int32))
        data += kept_v.astype(np.complex64).tobytes()
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        n_kept = metadata["n_kept"]
        m, n = shape

        idx = _deserialize(data[: n_kept * 8]).reshape(-1, 2).astype(int)
        vals = np.frombuffer(data[n_kept * 8 :], dtype=np.complex64).astype(
            np.complex128
        )

        F = np.zeros((m, n), dtype=np.complex128)
        for (r, c), v in zip(idx, vals):
            if r < m and c < n:
                F[r, c] = v
                F[(m - r) % m, (n - c) % n] = np.conj(v)

        return np.fft.ifft2(F).real.astype(np.float32)
