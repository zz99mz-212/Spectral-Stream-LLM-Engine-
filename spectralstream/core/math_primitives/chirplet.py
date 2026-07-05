"""DEPRECATED — Chirplet Transform.

This module is deprecated and has been moved to _archive/core/math_primitives/chirplet.py.
Please use DCT-based spectral methods from .transforms instead.
"""

import warnings as _warnings
import numpy as np


class ChirpletTransform:
    """DEPRECATED — Chirplet transform for chirp-like signals in attention.

    This class is deprecated. Use DCT-based spectral methods instead.
    The implementation is preserved in _archive/core/math_primitives/chirplet.py.
    """

    def __init__(self, n_rates: int = 16, window_size: int = 64):
        _warnings.warn(
            "ChirpletTransform is deprecated. Use DCT-based transforms from "
            "spectralstream.core.math_primitives.transforms instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.n_rates = n_rates
        self.window_size = window_size
        self._rates = np.linspace(-0.5, 0.5, n_rates)

    def _gabor_atom(self, t: np.ndarray, freq: float, chirp_rate: float) -> np.ndarray:
        phase = 2.0 * np.pi * freq * t + np.pi * chirp_rate * t**2
        return np.exp(-0.5 * (t / (self.window_size / 4)) ** 2) * np.exp(1j * phase)

    def detect_chirp_rate(self, signal: np.ndarray) -> float:
        signal = np.asarray(signal, dtype=np.float64)
        n = len(signal)
        win = min(self.window_size, n)
        t = np.arange(win, dtype=np.float64) - win / 2.0
        best_energy = 0.0
        best_rate = 0.0
        for rate in self._rates:
            energy = 0.0
            for center in range(0, n - win + 1, max(1, win // 4)):
                segment = signal[center : center + win]
                for freq_idx in range(8):
                    freq = freq_idx / (2.0 * win)
                    atom = self._gabor_atom(t, freq, rate).real
                    proj = np.abs(np.dot(segment, atom))
                    energy += proj**2
            if energy > best_energy:
                best_energy = energy
                best_rate = rate
        return float(best_rate)

    def compress_attention(
        self, attn_matrix: np.ndarray, n_components: int = 8
    ) -> dict:
        attn = np.asarray(attn_matrix, dtype=np.float64)
        seq_len = attn.shape[0]
        diag_profile = np.array(
            [np.trace(attn, offset=k) for k in range(-seq_len + 1, seq_len)],
            dtype=np.float64,
        )
        chirp_rate = self.detect_chirp_rate(diag_profile)
        t_row = np.arange(seq_len, dtype=np.float64)
        t_col = np.arange(seq_len, dtype=np.float64)
        components = []
        for comp_idx in range(n_components):
            freq = (comp_idx + 1) / (2.0 * seq_len)
            atom_row = np.exp(
                1j * (2.0 * np.pi * freq * t_row + np.pi * chirp_rate * t_row**2)
            ).real
            atom_col = np.exp(
                1j * (2.0 * np.pi * freq * t_col + np.pi * chirp_rate * t_col**2)
            ).real
            outer = np.outer(atom_row, atom_col)
            coeff = float(np.sum(attn * outer))
            components.append(
                {"freq": freq, "chirp_rate": chirp_rate, "coefficient": coeff}
            )
        approx = np.zeros_like(attn)
        for comp in components:
            freq = comp["freq"]
            cr = comp["chirp_rate"]
            c = comp["coefficient"]
            atom_row = np.exp(
                1j * (2.0 * np.pi * freq * t_row + np.pi * cr * t_row**2)
            ).real
            atom_col = np.exp(
                1j * (2.0 * np.pi * freq * t_col + np.pi * cr * t_col**2)
            ).real
            approx += c * np.outer(atom_row, atom_col)
        norm_attn = np.linalg.norm(attn)
        norm_approx = np.linalg.norm(approx)
        if norm_approx > 1e-10:
            approx *= norm_attn / norm_approx
        return {
            "shape": attn.shape,
            "type": "chirplet",
            "n_components": n_components,
            "components": components,
            "approximation": approx,
        }

    def reconstruct_attention(self, compressed: dict) -> np.ndarray:
        return compressed["approximation"]


__all__ = ["ChirpletTransform"]
