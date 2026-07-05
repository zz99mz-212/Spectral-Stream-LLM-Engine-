"""Auto-generated from _class_wrappers.py."""

from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()


def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()


def _ensure_2d(tensor: np.ndarray) -> tuple:
    orig_ndim = tensor.ndim
    orig_shape = tensor.shape
    t = tensor.astype(np.float64)
    if t.ndim < 2:
        t = t.reshape(1, -1)
    elif t.ndim > 2:
        t = t.reshape(t.shape[0], -1)
    return t, orig_shape, orig_ndim


class PlasmaOscillation:
    """Plasma oscillation normal mode decomposition via DCT + energy-preserving thresholding."""

    name = "plasma_oscillation"
    category = "physics"

    def compress(
        self, tensor: np.ndarray, keep_frac: float = 0.5
    ) -> Tuple[bytes, dict]:
        from spectralstream.core.math_primitives import dct

        t, orig_shape, orig_ndim = _ensure_2d(tensor)
        coeffs = dct(t)
        flat = coeffs.ravel()
        n = len(flat)
        k = max(1, int(keep_frac * n))
        idx = np.argpartition(np.abs(flat), -k)[-k:]
        idx.sort()
        kept_vals = flat[idx]
        meta = dict(shape=orig_shape, ndim=orig_ndim, n=n)
        data = _serialize(idx.astype(np.int32)) + kept_vals.astype(np.float16).tobytes()
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        from spectralstream.core.math_primitives import idct

        shape = metadata["shape"]
        ndim = metadata.get("ndim", len(shape))
        n = metadata["n"]
        bytes_per_entry = 6
        max_entries = len(data) // bytes_per_entry
        k = max_entries
        if k <= 0:
            return np.zeros(shape, dtype=np.float32)
        idx = _deserialize(data[: k * 4]).astype(int)
        vals = np.frombuffer(data[k * 4 :], dtype=np.float16).astype(np.float64)
        coeffs = np.zeros(n, dtype=np.float64)
        for i, v in zip(idx, vals):
            if i < n:
                coeffs[i] = v
        if ndim < 2:
            result = idct(coeffs[: shape[0]])
        else:
            m = shape[0]
            n_cols = int(np.prod(shape[1:]))
            c2d = coeffs.reshape(m, n_cols)
            result = idct(c2d)
        return result.reshape(shape).astype(np.float32)


class DebyeShielding:
    """Debye shielding: SVD-based low-rank compression."""

    name = "debye_shielding"
    category = "physics"

    def compress(self, tensor: np.ndarray, rank: int = None) -> Tuple[bytes, dict]:
        from spectralstream.compression.methods.physics.quantum import DensityMatrix

        return DensityMatrix().compress(tensor, rank)

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        from spectralstream.compression.methods.physics.quantum import DensityMatrix

        return DensityMatrix().decompress(data, metadata)


class PlasmaTurbulence:
    """Plasma turbulence cascade via SVD-based compression."""

    name = "plasma_turbulence"
    category = "physics"

    def compress(self, tensor: np.ndarray, rank: int = None) -> Tuple[bytes, dict]:
        from spectralstream.compression.methods.physics.quantum import DensityMatrix

        return DensityMatrix().compress(tensor, rank)

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        from spectralstream.compression.methods.physics.quantum import DensityMatrix

        return DensityMatrix().decompress(data, metadata)


class PlasmaField:
    """Plasma field: SVD-based low-rank approximation."""

    name = "plasma_field"
    category = "physics"

    def compress(self, tensor: np.ndarray, rank: int = None) -> Tuple[bytes, dict]:
        from spectralstream.compression.methods.physics.quantum import DensityMatrix

        return DensityMatrix().compress(tensor, rank)

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        from spectralstream.compression.methods.physics.quantum import DensityMatrix

        return DensityMatrix().decompress(data, metadata)


class PlasmaParticleSimulation:
    """Plasma oscillation via charged particle simulation.
    Simulates particles interacting via Coulomb force, extracts oscillation
    modes via FFT, and stores mode amplitudes/frequencies for reconstruction."""

    name = "plasma_particle_simulation"
    category = "physics"

    def compress(
        self, tensor: np.ndarray, n_modes: int = 16, n_particles: int = 64
    ) -> Tuple[bytes, dict]:
        import math
        import struct

        orig_shape = tensor.shape
        mat = tensor.reshape(-1, tensor.shape[-1]).astype(np.float64)
        m, n = mat.shape

        all_modes = []
        rng = np.random.RandomState(42)
        for i in range(m):
            signal = mat[i]
            positions = rng.uniform(0, n, n_particles)
            charges = np.interp(positions, np.arange(n), signal)
            c_max = float(np.max(np.abs(charges)))
            if c_max > 1e-10:
                charges /= c_max

            velocities = np.zeros(n_particles)
            n_steps = min(30, n)
            trajectory = np.zeros((n_steps, n_particles), dtype=np.float64)

            for t in range(n_steps):
                forces = np.zeros(n_particles, dtype=np.float64)
                for p in range(n_particles):
                    for q in range(n_particles):
                        if p != q:
                            dx = positions[q] - positions[p]
                            r = abs(dx) + 1e-6
                            forces[p] += charges[q] * dx / (r**3)
                velocities += 0.05 * forces
                velocities *= 0.99
                positions += 0.05 * velocities
                trajectory[t] = positions.copy()

            fft_coeffs = np.fft.rfft(trajectory, axis=0)
            freqs = np.fft.rfftfreq(n_steps)
            magnitudes = np.abs(fft_coeffs).sum(axis=1)
            top_idx = np.argsort(magnitudes)[::-1][:n_modes]
            modes = []
            for idx in top_idx:
                modes.append((float(freqs[idx]), float(magnitudes[idx] / n_particles)))
            all_modes.append(modes)

        flat_modes = []
        for row_modes in all_modes:
            for f_val, a_val in row_modes:
                flat_modes.append(f_val)
                flat_modes.append(a_val)
        mode_arr = np.array(flat_modes, dtype=np.float32)
        header = struct.pack("<IIIII", m, n, n_particles, n_modes, len(mode_arr))
        data = header + mode_arr.tobytes()
        meta = {
            "shape": orig_shape,
            "n_modes": n_modes,
            "n_particles": n_particles,
            "n_rows": m,
        }
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        import math
        import struct

        shape = metadata["shape"]
        n_modes = metadata["n_modes"]
        n_particles = metadata["n_particles"]
        n_rows = metadata["n_rows"]
        n_cols = shape[-1]

        header_size = struct.calcsize("<IIIII")
        m, n_val, np_val, nm_val, n_vals = struct.unpack_from("<IIIII", data, 0)
        mode_arr = np.frombuffer(data[header_size:], dtype=np.float32).copy()
        result = np.zeros((n_rows, n_cols), dtype=np.float64)

        for i in range(n_rows):
            signal = np.zeros(n_cols, dtype=np.float64)
            t = np.arange(n_cols, dtype=np.float64)
            for j in range(n_modes):
                pos = (i * n_modes + j) * 2
                if pos + 1 < len(mode_arr):
                    freq = float(mode_arr[pos])
                    amp = float(mode_arr[pos + 1])
                    signal += amp * np.cos(2 * math.pi * freq * t)
            result[i] = signal

        return result.reshape(shape).astype(np.float32)
