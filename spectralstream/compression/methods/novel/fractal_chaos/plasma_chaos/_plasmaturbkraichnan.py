from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class PlasmaTurbKraichnan:
    """Kraichnan's DIA: Φ(k) = Cε^{2/3}k^{-5/3}f(kL)g(kη) with random phase."""

    name = "plasma_turb_kraichnan"
    category = "novel_physics"

    def compress(self, tensor: np.ndarray) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        if t.ndim < 2:
            t = t.reshape(1, -1)
        m, n = t.shape

        F = np.fft.fft2(t)
        shifted = np.fft.fftshift(F)
        kx = np.fft.fftshift(np.fft.fftfreq(m)) * 2 * np.pi
        ky = np.fft.fftshift(np.fft.fftfreq(n)) * 2 * np.pi
        KX, KY = np.meshgrid(kx, ky, indexing="ij")
        k_mag = np.sqrt(KX**2 + KY**2)

        power = np.abs(shifted) ** 2
        k1d = k_mag.ravel()
        p1d = power.ravel()
        bins = 64
        k_bins = np.logspace(
            np.log10(k1d[k1d > 0].min() + 1e-30), np.log10(k1d.max() + 1e-30), bins
        )
        digitized = np.digitize(k1d, k_bins)
        p_avg = np.array([p1d[digitized == i].mean() for i in range(1, bins)])

        C = float(np.mean(p_avg[:5]) / (np.mean(k_bins[:5]) ** (-5 / 3) + 1e-30))
        eps = float(np.mean(p_avg) / np.mean(k_bins ** (-5 / 3) + 1e-30))
        seed = 42

        residual = t - np.mean(t)
        thr = np.percentile(np.abs(residual), 92)
        rmask = np.abs(residual) > thr
        ridx = np.argwhere(rmask)
        rvals = residual[rmask]

        meta = dict(shape=tensor.shape, C=C, eps=eps, seed=seed, n_res=len(rvals))
        data = struct.pack("<ddi", C, eps, seed)
        data += _serialize(ridx.astype(np.int16)) + rvals.astype(np.float16).tobytes()
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        C = metadata["C"]
        eps = metadata["eps"]
        seed = metadata["seed"]
        n_res = metadata.get("n_res", 0)
        m, n = shape

        kx = np.fft.fftshift(np.fft.fftfreq(m)) * 2 * np.pi
        ky = np.fft.fftshift(np.fft.fftfreq(n)) * 2 * np.pi
        KX, KY = np.meshgrid(kx, ky, indexing="ij")
        k_mag = np.sqrt(KX**2 + KY**2)

        rng = np.random.RandomState(seed)
        phase = rng.rand(m, n) * 2 * np.pi
        spectrum = C * eps ** (2 / 3) * (k_mag ** (-5 / 3))
        spectrum[k_mag < 1e-10] = 0

        F = spectrum * np.exp(1j * phase)
        recon = np.fft.ifft2(np.fft.ifftshift(F)).real

        pos = 20
        if n_res > 0:
            ridx = np.frombuffer(data[pos : pos + n_res * 4], dtype=np.int16).reshape(
                -1, 2
            )
            pos += n_res * 4
            rvals = np.frombuffer(data[pos : pos + n_res * 2], dtype=np.float16).astype(
                np.float64
            )
            for (ii, jj), vv in zip(ridx, rvals):
                if ii < m and jj < n:
                    recon[ii, jj] += vv

        return recon.reshape(shape).astype(np.float32)
