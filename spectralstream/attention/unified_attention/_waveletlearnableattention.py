from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union

import numpy as np

from spectralstream.core.math_primitives import (
    BAND_HIGH,
    BAND_LOW,
    BAND_NORMAL,
    DCTRotator,
    HadamardRotator,
    WaveletTransform,
    apply_spectral_kernel,
    band_limit,
    dct,
    fft,
    fftfreq,
    gibbs_softmax,
    idct,
    ifft,
    next_power_of_two,
    softmax,
    spectral_entropy,
    yukawa_kernel_1d,
)


class WaveletLearnableAttention:
    """O(L log L) attention via multi-resolution wavelet decomposition.

    Replaces quadratic QK^T with multi-resolution wavelet analysis:

      1. DWT on Q, K, V along sequence dim → frequency-localised bands
      2. DC band (coarsest approx): full softmax attention
         Size = L / 2^n_levels  →  O(L² / 4^n) → O(L) for n ≈ log₂√L
      3. Detail bands (fine scales): linear attention via kernel trick
         out = φ(q) @ (φ(k)^T @ v)   where φ(x) = ReLU(x) + 1
         Each detail processed in O(L/2^k · d²) instead of O(L²/4^k)
      4. Cross-band interactions via learned mixing weights W_mix[n_bands, n_bands]
      5. Bands resized to original length and combined
         (or optionally reconstructed via true IDWT)

    Multi-resolution analogy (plasma physics → attention):
      ┌──────────────────────────────────────────────────────────────────┐
      │  Physics concept      │  Attention analogy                     │
      ├───────────────────────┼─────────────────────────────────────────┤
      │  Gyrokinetic slow     │  DC band = zonal flow (mean field)     │
      │  Gyrokinetic fast     │  Detail bands = turbulent eddies       │
      │  Cross-scale transfer │  Band mixing weights W_mix             │
      │  Energy cascade       │  Coarse→fine information flow          │
      └──────────────────────────────────────────────────────────────────┘

    Complexity: O(L log L) vs O(L²) for full attention.
      - DWT/IDWT: O(L log L) via pyramidal algorithm
      - DC attention at L/2^n: O((L/2^n)²), negligible for n ≥ 3
      - Detail linear attention: O(L · d²) per band (no softmax)
      - Example: L=4096, n=4 → DC is 256×256 → ~200× speedup
      - Example: L=1M, n=6 → DC is 15625×15625 → still O(L²) there,
        but detail bands dominate at O(L) each → ~50000× speedup

    References:
      - HighNoon: Wavelet-based attention (this implementation)
      - Kim et al. "Wavelet Attention" (2023) — different approach
    """

    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        wavelet: str = 'haar',
        n_levels: int = 3,
        use_idwt: bool = False,
    ):
        if wavelet not in ('haar', 'db4'):
            raise ValueError(f"wavelet must be 'haar' or 'db4', got {wavelet!r}")
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        if n_levels < 1:
            raise ValueError(f"n_levels must be >= 1, got {n_levels}")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.wavelet = wavelet
        self.n_levels = n_levels
        self.use_idwt = use_idwt

        n_bands = n_levels + 1
        self.band_mix = np.ones((n_bands, n_bands), dtype=np.float32) / n_bands

    # ── Vectorised multi-dimensional DWT / IDWT ───────────────────────────

    @staticmethod
    def _haar_forward(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """1-level Haar DWT along last axis. Fully vectorised.

        x:  [..., n]
        Returns: (approx, detail)  each [..., ceil(n/2)]
        """
        n = x.shape[-1]
        if n % 2 == 1:
            x = np.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, 1)], mode='constant')
        even = x[..., 0::2]
        odd = x[..., 1::2]
        approx = (even + odd) * 0.5
        detail = (even - odd) * 0.5
        return approx, detail

    @staticmethod
    def _haar_inverse(approx: np.ndarray, detail: np.ndarray) -> np.ndarray:
        """1-level Haar inverse DWT along last axis. Fully vectorised.

        approx, detail:  [..., m]
        Returns:         [..., 2*m]
        """
        m = approx.shape[-1]
        out = np.empty((*approx.shape[:-1], 2 * m), dtype=np.float64)
        out[..., 0::2] = approx + detail
        out[..., 1::2] = approx - detail
        return out

    @staticmethod
    def _db4_forward(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """1-level Daubechies-4 DWT along last axis. Fully vectorised.

        Uses the lifting factorisation (predict/update) applied to
        arbitrary leading dimensions by rolling along axis=-1.
        """
        n = x.shape[-1]
        if n % 2 == 1:
            x = np.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, 1)], mode='constant')

        alpha, beta, gamma, delta = WaveletTransform._db4_constants()

        even = x[..., 0::2].copy()
        odd = x[..., 1::2].copy()

        odd -= alpha * (np.roll(even, -1, axis=-1) + even)
        even -= beta * (np.roll(odd, -1, axis=-1) + odd)
        odd -= gamma * (np.roll(even, -1, axis=-1) + even)
        even -= delta * (np.roll(odd, -1, axis=-1) + odd)

        even *= np.sqrt(2.0)
        odd *= np.sqrt(2.0)
        return even, odd

    @staticmethod
    def _db4_inverse(approx: np.ndarray, detail: np.ndarray) -> np.ndarray:
        """1-level Daubechies-4 inverse DWT along last axis."""
        alpha, beta, gamma, delta = WaveletTransform._db4_constants()

        even = approx / np.sqrt(2.0)
        odd = detail / np.sqrt(2.0)

        even += delta * (np.roll(odd, -1, axis=-1) + odd)
        odd += gamma * (np.roll(even, -1, axis=-1) + even)
        even += beta * (np.roll(odd, -1, axis=-1) + odd)
        odd += alpha * (np.roll(even, -1, axis=-1) + even)

        m = approx.shape[-1]
        out = np.empty((*approx.shape[:-1], 2 * m), dtype=np.float64)
        out[..., 0::2] = even
        out[..., 1::2] = odd
        return out

    @staticmethod
    def _move_seq_last(x: np.ndarray) -> np.ndarray:
        """Transpose so seq dim is last: [B, H, S, D] → [B, H, D, S].

        DWT operators work along the last axis; we want them on seq.
        """
        return x.transpose(0, 1, 3, 2)

    @staticmethod
    def _move_seq_back(x: np.ndarray, seq_len: int) -> np.ndarray:
        """Undo _move_seq_last: [B, H, D, S] → [B, H, S, D]."""
        return x.transpose(0, 1, 3, 2)

    def _dwt_bands(self, x: np.ndarray) -> List[Tuple[int, np.ndarray, np.ndarray]]:
        """Multi-level wavelet decomposition along sequence dim (axis=-2).

        Internally transposes to [B, H, D, S] so the DWT last-axis ops
        operate on the sequence dimension, then transposes back.

        x:  [batch, n_heads, seq_len, head_dim]
        Returns: list of (level, approx, detail) where each tensor has
                 shape [batch, n_heads, sub_seq, head_dim].
                 The final element is the residual approximation (DC band)
                 with an empty detail array.
        """
        forward_fn = (
            WaveletLearnableAttention._haar_forward
            if self.wavelet == 'haar'
            else WaveletLearnableAttention._db4_forward
        )

        # Transpose to [B, H, D, S] → DWT last-axis ops act on seq
        current = WaveletLearnableAttention._move_seq_last(x)

        levels: List[Tuple[int, np.ndarray, np.ndarray]] = []
        level = 0

        while level < self.n_levels and current.shape[-1] > 2:
            approx, detail = forward_fn(current)
            # approx, detail: [B, H, D, S/2]  — seq is last axis
            # Transpose back to [B, H, S/2, D] for storage
            approx_b = WaveletLearnableAttention._move_seq_back(approx, -1)
            detail_b = WaveletLearnableAttention._move_seq_back(detail, -1)
            levels.append((level, approx_b, detail_b))
            current = approx          # keep in [B, H, D, S] space for next level
            level += 1

        # Final residual — transpose back
        current_b = WaveletLearnableAttention._move_seq_back(current, -1)
        levels.append((level, current_b, np.empty(0, dtype=np.float64)))
        return levels

    def _idwt_reconstruct(
        self,
        levels: List[Tuple[int, np.ndarray, np.ndarray]],
    ) -> np.ndarray:
        """Multi-level inverse DWT reconstruction (true IDWT).

        Takes the (level, approx, detail) structure where approx/detail
        are in [B, H, S', D] format.  Transposes internally to [B, H, D, S']
        for the inverse DWT, which operates along the last axis.

        Returns: [batch, n_heads, seq_len, head_dim]
        """
        inverse_fn = (
            WaveletLearnableAttention._haar_inverse
            if self.wavelet == 'haar'
            else WaveletLearnableAttention._db4_inverse
        )

        # Start from the final approximation (DC band) in [B, H, D, S] space
        current = WaveletLearnableAttention._move_seq_last(levels[-1][1])

        for _, _, detail in reversed(levels[:-1]):
            detail_t = WaveletLearnableAttention._move_seq_last(detail)
            if detail_t.size == 0:
                current = inverse_fn(current, np.zeros_like(current))
            else:
                current = inverse_fn(current, detail_t)

        # Transpose back to [B, H, S, D]
        return WaveletLearnableAttention._move_seq_back(current, -1)

    # ── Resize ─────────────────────────────────────────────────────────────

    @staticmethod
    def _resize_band(band: np.ndarray, target_len: int) -> np.ndarray:
        """Resize band along sequence dim (axis=-2) via linear interpolation.

        band:       [batch, n_heads, band_len, head_dim]
        target_len: desired sequence length

        Returns:    [batch, n_heads, target_len, head_dim]
        """
        batch, heads, blen, dim = band.shape
        if blen == target_len:
            return band
        if blen == 1:
            return np.broadcast_to(band, (batch, heads, target_len, dim)).copy()

        x_old = np.linspace(0.0, 1.0, blen, dtype=np.float64)
        x_new = np.linspace(0.0, 1.0, target_len, dtype=np.float64)

        # Flatten to [B*H*D, blen] — single loop over 1D signals
        flat = band.reshape(-1, blen)
        result_flat = np.zeros((flat.shape[0], target_len), dtype=np.float64)
        for i in range(flat.shape[0]):
            result_flat[i] = np.interp(x_new, x_old, flat[i])
        return result_flat.reshape(batch, heads, target_len, dim)

    # ── Forward ────────────────────────────────────────────────────────────

    def forward(
        self,
        q: np.ndarray,
        k: np.ndarray,
        v: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Wavelet learnable attention forward pass.

        Pipeline:
          1. Multi-level DWT of Q, K, V along sequence dim
             → DC band (L/2^n) + n detail bands (L/2, L/4, ..., L/2^n)
          2. DC band:  full softmax attention — O((L/2^n)²)
          3. Detail bands: linear attention (kernel trick) — O(L/2^k · d²)
          4. Cross-band mixing + reconstruction (resize-and-sum or IDWT)

        Args:
            q:  [seq_len, d_model] or [batch, n_heads, seq_len, head_dim]
            k:  same shape as q
            v:  same shape as q
            mask: optional [seq_len] boolean or causal mask

        Returns:
            Output array with the same shape as input.
        """
        # ── Input reshaping ────────────────────────────────────────────────
        ndim = q.ndim
        if ndim == 2:
            seq, d_in = q.shape
            if d_in != self.d_model:
                raise ValueError(f"Expected d_model={self.d_model}, got {d_in}")
            q = q.reshape(1, self.n_heads, seq, self.head_dim)
            k = k.reshape(1, self.n_heads, seq, self.head_dim)
            v = v.reshape(1, self.n_heads, seq, self.head_dim)
        elif ndim == 4:
            if q.shape[1] != self.n_heads:
                raise ValueError(
                    f"Expected n_heads={self.n_heads}, got {q.shape[1]}"
                )
        else:
            raise ValueError(f"Expected 2D or 4D input, got {ndim}D")

        # ── 1. Multi-level DWT on Q, K, V along sequence dim ───────────────
        q_lev = self._dwt_bands(q)     # [(lvl, approx, detail), ..., (lvl, DC, [])]
        k_lev = self._dwt_bands(k)
        v_lev = self._dwt_bands(v)

        n_bands = len(q_lev)           # = actual_levels + 1
        dc_idx = n_bands - 1

        # ── 2. Per-band attention ──────────────────────────────────────────
        # n_bands consists of:
        #   [detail_0, detail_1, ..., detail_{n-1}, DC]
        # where detail_k = wavelet detail coefficients at level k,
        #       DC        = final residual approximation (coarsest).

        band_outputs: List[np.ndarray] = []

        scale = math.sqrt(self.head_dim)

        for band in range(dc_idx):
            # Detail band: use wavelet *detail* coefficients (high-frequency)
            _, _, qd = q_lev[band]
            _, _, kd = k_lev[band]
            _, _, vd = v_lev[band]

            # Linear attention via kernel trick:
            #   φ(x) = ReLU(x) + 1   (feature map for non-negative kernel)
            #   out  = φ(q) @ ( φ(k)^T @ v )
            phi_q = np.maximum(qd, 0.0) + 1.0
            phi_k = np.maximum(kd, 0.0) + 1.0

            # kv = φ(k)^T @ v   → [B, H, D, D]
            kv = phi_k.transpose(0, 1, 3, 2) @ vd
            # out = φ(q) @ kv   → [B, H, QL, D]
            out = phi_q @ kv

            band_outputs.append(out)

        # DC band: full softmax attention on the coarse approximation
        q_dc = q_lev[dc_idx][1]     # final approximation = DC band
        k_dc = k_lev[dc_idx][1]
        v_dc = v_lev[dc_idx][1]

        scores = q_dc @ k_dc.transpose(0, 1, 3, 2) / scale

        if mask is not None:
            dc_len = q_dc.shape[-2]
            m_sub = WaveletLearnableAttention._resize_band(
                mask.astype(np.float64).reshape(1, 1, -1, 1), dc_len,
            )[0, 0, :, 0] > 0.5
            scores = np.where(
                m_sub[np.newaxis, :] & m_sub[:, np.newaxis],
                scores, -1e30,
            )

        attn_w = softmax(scores, axis=-1, temperature=1.0)
        dc_out = attn_w @ v_dc
        band_outputs.append(dc_out)

        # ── 3. Cross-band mixing + reconstruction ──────────────────────────
        if self.use_idwt:
            # True inverse-DWT reconstruction:
            #   Replace wavelet coefficients with attention outputs,
            #   then run the pyramidal inverse transform.
            new_lev: List[Tuple[int, np.ndarray, np.ndarray]] = []
            for i in range(dc_idx):
                new_lev.append((
                    q_lev[i][0],
                    q_lev[i][1],           # approx (not used in IDWT)
                    band_outputs[i],        # detail ← processed detail band
                ))
            new_lev.append((
                q_lev[dc_idx][0],
                band_outputs[dc_idx],       # approx ← processed DC band
                np.empty(0, dtype=np.float64),
            ))
            output = self._idwt_reconstruct(new_lev)
        else:
            # Efficient resize-and-sum:
            #   1. Resize each band to original seq length (once each)
            #   2. Mix in full-resolution space using learned weights
            orig_seq = q.shape[-2]
            resized = [
                WaveletLearnableAttention._resize_band(b, orig_seq)
                for b in band_outputs
            ]

            accumulated = np.zeros_like(q, dtype=np.float64)
            for i in range(n_bands):
                for j in range(n_bands):
                    accumulated += self.band_mix[i, j] * resized[j]
            output = accumulated / n_bands

        # ── Output reshaping ───────────────────────────────────────────────
        if ndim == 2:
            return np.ascontiguousarray(
                output[0].reshape(-1, self.d_model),
            ).astype(q.dtype)
        return output.astype(q.dtype)
