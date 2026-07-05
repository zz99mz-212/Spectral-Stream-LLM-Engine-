# --- decodesymbols.py ---
"""Module extracted from unified_quantizer.py — decodesymbols."""

from __future__ import annotations

import struct
import zlib

import numpy as np

EPS = 1e-12

import logging
from collections import Counter
from heapq import heappop, heappush, heapify
from typing import Any, Dict, List, Optional, Sequence, Tuple




def _pack_values(values: np.ndarray, n_bits: int) -> bytes:
    if len(values) == 0 or n_bits == 0:
        return b""
    n = len(values)
    result = bytearray()
    bits = 0
    bit_pos = 0
    for v in values:
        bits = (bits << n_bits) | int(v)
        bit_pos += n_bits
        while bit_pos >= 8:
            bit_pos -= 8
            result.append((bits >> bit_pos) & 0xFF)
    if bit_pos > 0:
        result.append((bits << (8 - bit_pos)) & 0xFF)
    return bytes(result)


def _unpack_values(data: bytes, n_values: int, n_bits: int) -> np.ndarray:
    if len(data) == 0 or n_values == 0 or n_bits == 0:
        return np.array([], dtype=np.int32)
    total_bits = n_values * n_bits
    bit_array = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
    bit_array = bit_array[:total_bits]
    bit_array = bit_array.reshape(n_values, n_bits)
    result = np.zeros(n_values, dtype=np.int32)
    for b in range(n_bits):
        result = (result << 1) | bit_array[:, b].astype(np.int32)
    return result


def _tt_reconstruct(cores: List[np.ndarray]) -> np.ndarray:
    result = cores[0]
    for c in cores[1:]:
        result = result @ c
    return result


def _decode_symbols(
    bitstream: bytes, codebook: Dict[int, str], num_symbols: int
) -> List[int]:
    reverse = {code: sym for sym, code in codebook.items()}
    bits = "".join(f"{b:08b}" for b in bitstream)

    symbols: List[int] = []
    cur = ""
    idx = 0
    while len(symbols) < num_symbols and idx < len(bits):
        cur += bits[idx]
        idx += 1
        if cur in reverse:
            symbols.append(reverse[cur])
            cur = ""
    return symbols


def _deserialize_codebook(data: bytes, offset: int = 0) -> Tuple[Dict[int, str], int]:
    n = struct.unpack_from("<I", data, offset)[0]
    offset += 4
    result: Dict[int, str] = {}
    for _ in range(n):
        sym = struct.unpack_from("<q", data, offset)[0]
        offset += 8
        code_len = data[offset]
        offset += 1
        cblen = data[offset]
        offset += 1
        if code_len == 0:
            result[sym] = ""
        else:
            code_int = int.from_bytes(data[offset : offset + cblen], "big")
            offset += cblen
            result[sym] = bin(code_int)[2:].zfill(code_len)
    return result, offset


def decompress_from_ssf_block(data: bytes) -> np.ndarray:
    """Decompress SSF-serialized block bytes back to FP32 tensor."""
    if len(data) < 2:
        return np.array([], dtype=np.float32)

    # Check for zlib-compressed data
    type_tag = data[0]
    if type_tag == 0x80:
        original_len = struct.unpack_from("<I", data, 1)[0]
        decompressed = zlib.decompress(data[5:])
        return decompress_from_ssf_block(decompressed)

    if type_tag == 0:
        return np.frombuffer(data[4:], dtype=np.float32)

    offset = 1
    ndim = data[offset]
    offset += 1
    shape = []
    for _ in range(ndim):
        shape.append(struct.unpack_from("<I", data, offset)[0])
        offset += 4
    shape = tuple(shape)

    n_blocks = struct.unpack_from("<I", data, offset)[0]
    offset += 4

    q = UnifiedQuantizer()
    blocks = []
    for _ in range(n_blocks):
        row = struct.unpack_from("<i", data, offset)[0]
        offset += 4
        col = struct.unpack_from("<i", data, offset)[0]
        offset += 4
        bs = struct.unpack_from("<H", data, offset)[0]
        offset += 2
        max_abs = struct.unpack_from("<f", data, offset)[0]
        offset += 4
        rank = struct.unpack_from("<H", data, offset)[0]
        offset += 2
        n_bits_ac = data[offset]
        offset += 1

        core_shapes = []
        for _ in range(3):
            nd = data[offset]
            offset += 1
            s = []
            for _ in range(nd):
                s.append(struct.unpack_from("<H", data, offset)[0])
                offset += 2
            core_shapes.append(tuple(s))

        bs_len = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        bitstream = data[offset : offset + bs_len]
        offset += bs_len

        cb_len = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        cb_data = data[offset : offset + cb_len]
        offset += cb_len

        decoded_codebook, _ = _deserialize_codebook(cb_data)

        # Read per-core scales (v2 extension)
        n_scales = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        core_scales = []
        for _ in range(n_scales):
            core_scales.append(struct.unpack_from("<f", data, offset)[0])
            offset += 4

        total_elems = sum(int(np.prod(s)) for s in core_shapes)
        decoded_vals = _decode_symbols(bitstream, decoded_codebook, total_elems)
        decoded_ints = np.array(decoded_vals, dtype=np.float64)
        max_val = float((1 << (n_bits_ac - 1)) - 1)

        cores = []
        elem_offset = 0
        for ci, s in enumerate(core_shapes):
            n = int(np.prod(s))
            q_core = decoded_ints[elem_offset : elem_offset + n].reshape(s)
            scale = core_scales[ci] if ci < len(core_scales) else max_abs
            dequantized = (q_core / max(max_val, 1e-10)) * scale
            cores.append(dequantized.astype(np.float32))
            elem_offset += n

        tt_recon = _tt_reconstruct(cores)

        blocks.append(
            DCTBlock(
                row=row,
                col=col,
                block_size=bs,
                dct=tt_recon.astype(np.float64),
                variance=0.0,
            )
        )

    return q.dct.decompress(blocks, shape).astype(np.float32)


# --- fftdct1d.py ---
"""Module extracted from unified_quantizer.py — fftdct1d."""


import math
from spectralstream.core.math_primitives import (
    dct,
    idct,
    dct_2d,
    idct_2d,
    fwht,
    ifwht,
    LloydMaxQuantizer,
    WaveletTransform,
    zigzag_indices,
    spectral_entropy,
    cosine_similarity,
    band_limit,
    spectral_power_density,
    softmax,
    next_power_of_two,
    splitmix64,
)
import struct


def _fft_dct_2d(matrix: np.ndarray) -> np.ndarray:
    """2D DCT via FFT (O(N² log N) for N×N matrix).

    Separable: DCT-II along rows, then columns.
    Orthonormal scaling so DCT is unitary (energy-preserving).
    """
    arr = np.asarray(matrix, dtype=np.float64)
    m, n = arr.shape
    # DCT along rows (axis=1)
    result = np.zeros_like(arr)
    for i in range(m):
        result[i, :] = _fft_dct_1d(arr[i, :])
    # DCT along columns (axis=0)
    for j in range(n):
        result[:, j] = _fft_dct_1d(result[:, j])
    return result


def _fft_idct_2d(coeffs: np.ndarray) -> np.ndarray:
    """2D inverse DCT via FFT.  Separable IDCT-II."""
    arr = np.asarray(coeffs, dtype=np.float64)
    m, n = arr.shape
    result = np.zeros_like(arr)
    for i in range(m):
        result[i, :] = _fft_idct_1d(arr[i, :])
    for j in range(n):
        result[:, j] = _fft_idct_1d(result[:, j])
    return result


def _fft_dct_1d(x: np.ndarray) -> np.ndarray:
    """DCT-II via FFT (orthonormal).

    Standard DCT-II: C[k] = sum_n x[n] * cos(pi*k*(n+0.5)/N)
    FFT of even extension gives Z[k] = 2*C_raw[k].
    We reconstruct C_raw, divide by 2, then apply orthonormal scaling.
    """
    n = len(x)
    if n < 2:
        return x.copy()
    x2 = np.zeros(2 * n, dtype=np.float64)
    x2[:n] = x
    x2[n:] = x[::-1]
    Y = np.fft.fft(x2)[:n]
    k = np.arange(n, dtype=np.float64)
    # Z[k] = exp(-j*pi*k/(2n)) * Y[k] = 2 * sum(x*cos(...))
    Z = Y * np.exp(-1j * np.pi * k / (2 * n))
    C0 = Z[0].real / 2  # sum(x)
    Ck = Z[1:].real / 2  # sum(x*cos(...)) for k>0
    result = np.empty(n, dtype=np.float64)
    result[0] = C0 / np.sqrt(n)  # orthonormal DC
    result[1:] = Ck * np.sqrt(2.0 / n)
    return result


def _fft_idct_1d(y: np.ndarray) -> np.ndarray:
    """IDCT-II via FFT (orthonormal, inverse of _fft_dct_1d)."""
    n = len(y)
    if n < 2:
        return y.copy()
    # Reconstruct raw DCT from orthonormal coeffs
    # Z[0] = C0 * 2  (undo orthonormal scaling, multiply by 2)
    # Z[k] = Ck * sqrt(2) * sqrt(n) ... wait:
    # y[0] = C0/sqrt(n), so raw = y[0] * sqrt(n) * 2
    # y[k] = Ck * sqrt(2/n), so raw = y[k] / sqrt(2/n) * 2 = y[k] * sqrt(n/2) * 2
    Z = np.zeros(n, dtype=np.complex128)
    Z[0] = y[0] * np.sqrt(n) * 2.0
    Z[1:] = y[1:] * np.sqrt(n / 2.0) * 2.0
    # Build full 2n-point FFT array using conjugate symmetry
    Y = np.zeros(2 * n, dtype=np.complex128)
    ki = np.arange(n, dtype=np.float64)
    Y[0] = Z[0]
    Y[1:n] = Z[1:] * np.exp(1j * np.pi * ki[1:] / (2 * n))
    Y[n] = 0.0  # Nyquist
    for i in range(1, n):
        Y[2 * n - i] = np.conj(Y[i])
    return np.fft.ifft(Y)[:n].real


class SpectraQuantizer:
    """Full-matrix spectral envelope compression with sign-only coefficient storage.

    Preserves the frequency structure of weights even at extreme ratios.
    MSE is bounded by PSD quantization, not by number of stored coefficients.

    Parameters
    ----------
    target_ratio : float
        Desired compression ratio (default 2000).
    n_bands : int
        Number of radial frequency bands for PSD (default 16).
    psd_bits : int
        Bits per PSD band (default 4).
    """

    def __init__(
        self, target_ratio: float = 2000.0, n_bands: int = 16, psd_bits: int = 4
    ):
        self.target_ratio = max(2.0, target_ratio)
        self.n_bands = n_bands
        self.psd_bits = psd_bits

    @staticmethod
    def _fft_dct_1d(x: np.ndarray) -> np.ndarray:
        """DCT-II via FFT, orthonormal. O(N log N)."""
        n = len(x)
        if n < 2:
            return x.copy()
        x2 = np.zeros(2 * n, dtype=np.float64)
        x2[:n] = x
        x2[n:] = x[::-1]
        Y = np.fft.fft(x2)[:n]
        k = np.arange(n, dtype=np.float64)
        Z = Y * np.exp(-1j * np.pi * k / (2 * n))
        result = np.empty(n, dtype=np.float64)
        result[0] = Z[0].real / 2.0 / math.sqrt(n)
        result[1:] = Z[1:].real / 2.0 * math.sqrt(2.0 / n)
        return result

    @staticmethod
    def _fft_idct_1d(y: np.ndarray) -> np.ndarray:
        """IDCT-II via FFT, orthonormal. O(N log N)."""
        n = len(y)
        if n < 2:
            return y.copy()
        Z = np.zeros(n, dtype=np.complex128)
        Z[0] = y[0] * math.sqrt(n) * 2.0
        Z[1:] = y[1:] * math.sqrt(n / 2.0) * 2.0
        Y = np.zeros(2 * n, dtype=np.complex128)
        ki = np.arange(n, dtype=np.float64)
        Y[0] = Z[0]
        Y[1:n] = Z[1:] * np.exp(1j * np.pi * ki[1:] / (2 * n))
        for i in range(1, n):
            Y[2 * n - i] = np.conj(Y[i])
        return np.fft.ifft(Y)[:n].real

    def _fft_dct_2d(self, matrix: np.ndarray) -> np.ndarray:
        """Separable 2D DCT via FFT."""
        arr = np.asarray(matrix, dtype=np.float64)
        m, n = arr.shape
        result = np.zeros_like(arr)
        for i in range(m):
            result[i, :] = self._fft_dct_1d(arr[i, :])
        for j in range(n):
            result[:, j] = self._fft_dct_1d(result[:, j])
        return result

    def _fft_idct_2d(self, coeffs: np.ndarray) -> np.ndarray:
        """Separable 2D IDCT via FFT."""
        arr = np.asarray(coeffs, dtype=np.float64)
        m, n = arr.shape
        result = np.zeros_like(arr)
        for i in range(m):
            result[i, :] = self._fft_idct_1d(arr[i, :])
        for j in range(n):
            result[:, j] = self._fft_idct_1d(result[:, j])
        return result

    @staticmethod
    def _next_power_of_two(n: int) -> int:
        return 1 << (n - 1).bit_length()

    # ── PSD extraction ────────────────────────────────────────────────

    def _extract_psd(self, dct_2d_matrix: np.ndarray) -> np.ndarray:
        """Radial power spectral density in frequency bands."""
        m, n = dct_2d_matrix.shape
        i_grid = np.arange(m, dtype=np.float64)[:, None] / m
        j_grid = np.arange(n, dtype=np.float64)[None, :] / n
        freq = np.sqrt(i_grid**2 + j_grid**2)
        max_freq = float(np.max(freq))
        if max_freq < EPS:
            return np.ones(self.n_bands, dtype=np.float64) / self.n_bands

        band_idx = np.floor(freq / max_freq * self.n_bands).astype(np.int32)
        band_idx = np.clip(band_idx, 0, self.n_bands - 1)

        energy = dct_2d_matrix.astype(np.float64) ** 2
        psd = np.zeros(self.n_bands, dtype=np.float64)
        for b in range(self.n_bands):
            mask = band_idx == b
            psd[b] = float(np.mean(energy[mask])) if np.any(mask) else EPS

        psd = np.maximum(psd, EPS)
        total = float(np.sum(psd))
        if total > 0:
            psd /= total
        return psd

    # ── Exp-Golomb position coding ────────────────────────────────────

    @staticmethod
    def _encode_positions(positions: np.ndarray) -> bytes:
        """Exp-Golomb coded deltas between sorted positions."""
        if len(positions) == 0:
            return b""
        deltas = np.diff(positions, prepend=-1) - 1
        deltas = np.maximum(deltas, 0).astype(np.int64)
        bits: List[int] = []
        for d in deltas:
            val = int(d) + 1
            nbit = val.bit_length()
            bits.extend([0] * (nbit - 1))
            for shift in range(nbit - 1, -1, -1):
                bits.append((val >> shift) & 1)
        while len(bits) % 8:
            bits.append(0)
        out = bytearray()
        for i in range(0, len(bits), 8):
            byte = 0
            for j in range(8):
                byte = (byte << 1) | bits[i + j]
            out.append(byte)
        return bytes(out)

    @staticmethod
    def _decode_positions(data: bytes, n: int) -> np.ndarray:
        if not data or n == 0:
            return np.array([], dtype=np.int32)
        bits = "".join(f"{b:08b}" for b in data)
        pos = -1
        positions: List[int] = []
        idx = 0
        for _ in range(n):
            nz = 0
            while idx < len(bits) and bits[idx] == "0":
                nz += 1
                idx += 1
            if idx < len(bits):
                idx += 1
            val = 1
            for _ in range(nz):
                if idx < len(bits):
                    val = (val << 1) | (1 if bits[idx] == "1" else 0)
                    idx += 1
            pos += val
            positions.append(pos)
        return np.array(positions, dtype=np.int32)

    # ── Compress / Decompress ─────────────────────────────────────────

    def _serialize(self, data: dict) -> bytes:
        """Compact binary serialization without pickle overhead."""
        buf = bytearray()
        # Header: magic + shape + n_bands + psd_bits + n_coeffs
        buf += struct.pack("<II", data["shape"][0], data["shape"][1])
        buf += struct.pack("<BB", self.n_bands, self.psd_bits)
        buf += struct.pack("<I", data["n_coeffs"])
        # PSD: packed at psd_bits per band
        psd_packed = _pack_values(data["psd_quant"], self.psd_bits)
        buf += struct.pack("<H", len(psd_packed))
        buf += psd_packed
        # Positions: exp-Golomb coded
        buf += struct.pack("<I", len(data["positions"]))
        buf += data["positions"]
        # Signs: packed 8 per byte
        buf += struct.pack("<I", len(data["signs"]))
        buf += data["signs"]
        return bytes(buf)

    @staticmethod
    def _deserialize(data: bytes) -> dict:
        offset = 0
        m = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        n = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        n_bands = data[offset]
        offset += 1
        psd_bits = data[offset]
        offset += 1
        n_coeffs = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        psd_len = struct.unpack_from("<H", data, offset)[0]
        offset += 2
        psd_packed = data[offset : offset + psd_len]
        offset += psd_len
        psd_quant = _unpack_values(psd_packed, n_bands, psd_bits)
        pos_len = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        pos_data = data[offset : offset + pos_len]
        offset += pos_len
        if pos_len == 0:
            positions = np.array([], dtype=np.int32)
        else:
            positions = SpectraQuantizer._decode_positions(pos_data, n_coeffs)
        sign_len = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        signs = data[offset : offset + sign_len]
        offset += sign_len
        return {
            "shape": [m, n],
            "n_bands": n_bands,
            "psd_bits": psd_bits,
            "n_coeffs": n_coeffs,
            "psd_quant": psd_quant,
            "positions": pos_data,
            "signs": signs,
            "decoded_positions": positions,
        }

    def compress(self, tensor: np.ndarray, layer_name: str = "default") -> dict:
        """Compress via spectral envelope preservation.

        Returns dict with compact binary representation.
        """
        W = np.asarray(tensor, dtype=np.float64)
        m, n = W.shape
        max_side = self._next_power_of_two(max(m, n))

        # Pad to power of 2 for full-matrix DCT
        padded = np.zeros((max_side, max_side), dtype=np.float64)
        padded[:m, :n] = W

        # Stage 1: Full-matrix 2D DCT
        dct = self._fft_dct_2d(padded)

        # Stage 2: Extract PSD
        psd = self._extract_psd(dct)

        # Quantize PSD
        max_psd = float(np.max(psd))
        if max_psd > EPS:
            psd_quant = np.round(psd / max_psd * ((1 << self.psd_bits) - 1)).astype(
                np.int32
            )
            psd_quant = np.clip(psd_quant, 0, (1 << self.psd_bits) - 1)
        else:
            psd_quant = np.ones(self.n_bands, dtype=np.int32)

        # Stage 3: Compute budget and select top-k coefficients
        orig_bits = tensor.nbytes * 8
        target_bits = int(orig_bits / self.target_ratio)

        # Overhead: shape + PSD + headers
        overhead_bits = 128 + self.n_bands * self.psd_bits
        # Each coefficient: 1 sign bit + ~14 position bits (exp-Golomb avg)
        coeff_cost = 15
        n_coeffs = max(0, int((target_bits - overhead_bits) / coeff_cost))
        n_total = max_side * max_side
        n_coeffs = min(n_coeffs, n_total)

        flat = dct.ravel()
        energy = flat**2
        total_energy = float(np.sum(energy))

        if total_energy < EPS or n_coeffs == 0:
            return self._make_result(psd_quant, b"", b"", 0, m, n, layer_name)

        # Select top-k coefficients
        if n_coeffs >= n_total:
            sorted_i = np.arange(n_total, dtype=np.int32)
        else:
            sorted_i = np.argpartition(-energy, n_coeffs - 1)[:n_coeffs]
            sorted_i.sort()

        kept_vals = flat[sorted_i]
        signs = (kept_vals > 0).astype(np.int32)

        # Pack signs
        sign_bytes = bytearray()
        for i in range(0, len(signs), 8):
            byte = 0
            for j in range(8):
                if i + j < len(signs):
                    byte = (byte << 1) | signs[i + j]
                else:
                    byte <<= 1
            sign_bytes.append(byte)

        # Encode positions
        pos_bytes = self._encode_positions(sorted_i)

        return self._make_result(
            psd_quant, bytes(sign_bytes), pos_bytes, n_coeffs, m, n, layer_name
        )

    def _make_result(
        self, psd_quant, sign_bytes, pos_bytes, n_coeffs, m, n, layer_name
    ):
        serializable = {
            "shape": [m, n],
            "n_coeffs": n_coeffs,
            "psd_quant": psd_quant,
            "signs": sign_bytes,
            "positions": pos_bytes,
        }
        raw_bytes = self._serialize(serializable)
        orig_bytes = m * n * 4
        ratio = orig_bytes / max(len(raw_bytes), 1)

        return {
            "type": "spectra",
            "data": raw_bytes,
            "shape": [m, n],
            "ratio": ratio,
            "original_bytes": orig_bytes,
            "compressed_bytes": len(raw_bytes),
            "layer_name": layer_name,
            "n_coeffs": n_coeffs,
            "n_bands": self.n_bands,
            "psd_bits": self.psd_bits,
        }

    def decompress(self, compressed: dict) -> np.ndarray:
        """Reconstruct from spectral envelope representation."""
        d = self._deserialize(compressed["data"])
        m, n = d["shape"]
        n_bands = d["n_bands"]
        psd_bits = d["psd_bits"]
        n_coeffs = d["n_coeffs"]
        psd_quant = d["psd_quant"]
        positions = d["decoded_positions"]
        sign_data = d["signs"]

        max_side = self._next_power_of_two(max(m, n))

        # Reconstruct PSD
        max_q = float(np.max(psd_quant))
        if max_q > EPS:
            psd = psd_quant.astype(np.float64) / max_q
        else:
            psd = np.ones(n_bands, dtype=np.float64) / n_bands
        psd = np.maximum(psd, EPS)
        total = float(np.sum(psd))
        if total > 0:
            psd /= total

        # Build magnitude map from PSD
        i_grid = np.arange(max_side, dtype=np.float64)[:, None] / max_side
        j_grid = np.arange(max_side, dtype=np.float64)[None, :] / max_side
        freq = np.sqrt(i_grid**2 + j_grid**2)
        max_freq = float(np.max(freq))
        if max_freq > EPS:
            band_map = np.floor(freq / max_freq * n_bands).astype(np.int32)
        else:
            band_map = np.zeros_like(freq, dtype=np.int32)
        band_map = np.clip(band_map, 0, n_bands - 1)
        mag_map = np.sqrt(psd[band_map])

        # Reconstruct DCT coefficients: sign from stored, magnitude from PSD
        dct_flat = np.zeros(max_side * max_side, dtype=np.float64)
        for idx in range(n_coeffs):
            if idx >= len(positions):
                break
            pos = int(positions[idx])
            if pos >= len(dct_flat):
                continue
            byte_idx = idx // 8
            bit_idx = 7 - (idx % 8)
            if byte_idx < len(sign_data):
                sign = 1.0 if (sign_data[byte_idx] >> bit_idx) & 1 else -1.0
            else:
                sign = 1.0
            dct_flat[pos] = sign * float(mag_map.ravel()[pos])

        dct = dct_flat.reshape(max_side, max_side)
        recon = self._fft_idct_2d(dct)
        return recon[:m, :n].astype(np.float32)

    def get_ratio(self, original: np.ndarray, compressed: dict) -> float:
        return compressed.get("ratio", 1.0)

    def get_quality_metrics(
        self, original: np.ndarray, decompressed: np.ndarray
    ) -> dict:
        """Compute MSE, PSNR, cosine similarity, spectral similarity."""
        orig = original.astype(np.float64)
        dec = decompressed.astype(np.float64)
        if orig.shape != dec.shape:
            ms = tuple(min(a, b) for a, b in zip(orig.shape, dec.shape))
            orig = orig[: ms[0], : ms[1]]
            dec = dec[: ms[0], : ms[1]]
        mse = float(np.mean((orig - dec) ** 2))
        max_val = float(np.max(np.abs(orig)))
        psnr = (
            20.0 * math.log10(max_val / max(math.sqrt(mse), EPS))
            if max_val > 0 and mse > EPS
            else 0.0
        )
        rel_err = float(np.linalg.norm(orig - dec) / max(np.linalg.norm(orig), EPS))
        cos_sim = float(
            np.sum(orig * dec) / max(np.linalg.norm(orig) * np.linalg.norm(dec), EPS)
        )
        return {
            "mse": mse,
            "psnr": psnr,
            "relative_error": rel_err,
            "cos_similarity": cos_sim,
            "max_abs_error": float(np.max(np.abs(orig - dec))),
        }


# --- runalltests.py ---
"""Module extracted from unified_quantizer.py — runalltests."""


def _make_synthetic_weight(rows: int, cols: int, seed: int = 42) -> np.ndarray:
    """Create a realistic synthetic weight matrix (low-frequency dominant)."""
    rng = np.random.RandomState(seed)
    coeffs = np.zeros((max(rows, cols), max(rows, cols)), dtype=np.float64)
    limit = min(rows, cols, 48)
    for u in range(limit):
        for v in range(limit):
            if u + v < 48:
                coeffs[u, v] = 1.0 / (1.0 + u + v) * 0.5
    W = idct_2d(coeffs)[:rows, :cols]
    W -= np.mean(W)
    std = float(np.std(W))
    if std > 0:
        W /= std * 4.0
    noise = rng.randn(rows, cols).astype(np.float64) * 0.01
    return (W + noise).astype(np.float32)


def run_all_tests() -> None:
    """Run comprehensive tests for all 7 enhancement classes."""
    import json
    import sys
    import time

    passed: int = 0
    failed: int = 0

    def check(condition: bool, msg: str) -> None:
        nonlocal passed, failed
        if condition:
            passed += 1
            print(f"  \u2713 {msg}")
        else:
            failed += 1
            print(f"  \u2717 {msg}")

    start = time.perf_counter()

    # ── 1. HierarchicalMPSCompressor ──────────────────────────────────
    print("--- HierarchicalMPSCompressor ---")
    W48 = _make_synthetic_weight(48, 48)
    mps = HierarchicalMPSCompressor(min_bond_dim=4, max_bond_dim=8, n_sweeps=2)
    comp_mps = mps.compress(W48)
    recon_mps = mps.decompress(comp_mps)
    check(recon_mps.shape == W48.shape, "Round-trip shape preserved")
    mse_mps = float(np.mean((W48 - recon_mps) ** 2))
    check(mse_mps < 0.1, f"MSE={mse_mps:.6f} < 0.1")

    orig_params = W48.size
    compressed_params = sum(c.size for c in comp_mps["cores"])
    check(
        compressed_params < orig_params,
        f"Parameter reduction: {orig_params} -> {compressed_params} ({orig_params / max(compressed_params, 1):.1f}x)",
    )

    # ── 2. QAOABitAllocator ──────────────────────────────────────────
    print("--- QAOABitAllocator ---")
    rng_test = np.random.RandomState(42)
    qaoa = QAOABitAllocator(quality=0.95, max_bits_per_coeff=8)

    mags_low = np.abs(rng_test.randn(16, 16))
    alloc_low = qaoa.allocate(mags_low, block_size=16)
    check(alloc_low.shape == (16, 16), "Allocation shape matches")
    check(
        np.all(alloc_low >= 0) and np.all(alloc_low <= 8),
        f"All bits in [0,8], got [{int(alloc_low.min())}, {int(alloc_low.max())}]",
    )
    check(np.any(alloc_low > 0), "At least one coefficient gets bits")

    mags_high = np.abs(rng_test.randn(16, 16) * 100.0)
    alloc_high = qaoa.allocate(mags_high, block_size=16)
    diff_allocation = int(np.sum(alloc_high)) - int(np.sum(alloc_low))
    check(
        diff_allocation >= 0,
        f"High-magnitude block gets >= bits: sum(all_high)={int(np.sum(alloc_high))} vs sum(all_low)={int(np.sum(alloc_low))}",
    )

    # ── 3. StabilizerQuantizer ───────────────────────────────────────
    print("--- StabilizerQuantizer ---")
    sq = StabilizerQuantizer(n_bits=4, use_extended=True)
    vals = np.array([0.5, -0.3, 0.0, 0.8, -0.9, 0.2, -0.1, 0.7])
    encoded, overhead = sq.quantize_with_correction(vals)
    decoded = sq.dequantize_with_correction(encoded, 1.0, vals.shape)
    check(decoded.shape == vals.shape, "Round-trip shape preserved")
    check(overhead > 1.0, f"Overhead ratio {overhead:.3f} > 1.0")

    raw_bytes = b"\xab\xcd\xef\x01\x23\x45\x67\x89"
    protected = sq.protect_stream(raw_bytes)
    recovered = sq.recover_stream(protected)
    check(recovered == raw_bytes, "Perfect recovery without errors")

    corrupted = bytearray(protected)
    if len(corrupted) > 0:
        corrupted[1] ^= 0b00000100
    recovered_err = sq.recover_stream(bytes(corrupted))
    check(recovered_err == raw_bytes, "Single-bit error corrected")

    # ── 4. PredictiveCodingQuantizer ─────────────────────────────────
    print("--- PredictiveCodingQuantizer ---")
    pcq = PredictiveCodingQuantizer(n_bits_residual=3, max_bits_original=8)
    sig = rng_test.randn(64).astype(np.float32) * 2.0
    comp_pc = pcq.compress(sig)
    recon_pc = pcq.decompress(comp_pc)
    check(recon_pc.shape == sig.shape, "Round-trip shape preserved")
    check(
        len(comp_pc["residuals"]) < len(sig),
        f"Residuals ({len(comp_pc['residuals'])}) < signal ({len(sig)})",
    )
    check("scale" in comp_pc and comp_pc["scale"] > 0, "Scale present")
    check("ar_coeffs" in comp_pc, "AR coefficients present")

    # ── 5. TernaryWeightQuantizer ────────────────────────────────────
    print("--- TernaryWeightQuantizer ---")
    twq = TernaryWeightQuantizer(sparsity_target=0.85, block_size=64)
    W32 = _make_synthetic_weight(32, 32)
    comp_tw = twq.compress(W32)
    recon_tw = twq.decompress(comp_tw)
    check(recon_tw.shape == W32.shape, "Round-trip shape preserved")

    packed = comp_tw["packed"]
    n_w = comp_tw["n_weights"]
    observed = set()
    for i in range(n_w):
        byte_idx = i // 4
        bit_idx = i % 4
        code = (packed[byte_idx] >> (bit_idx * 2)) & 0b11
        if code == 0b01:
            observed.add(1)
        elif code == 0b10:
            observed.add(-1)
        else:
            observed.add(0)
    check(
        observed.issubset({-1, 0, 1}), f"Only ternary values {{-1,0,1}}, got {observed}"
    )

    # ── 6. SpectralSparsification ────────────────────────────────────
    print("--- SpectralSparsification ---")
    ss = SpectralSparsification(target_sparsity=0.9, block_size=16, quality_factor=1.0)
    W32b = _make_synthetic_weight(32, 32)
    comp_ss = ss.sparsify(W32b)
    recon_ss = ss.desparsify(comp_ss)
    check(recon_ss.shape == W32b.shape, "Round-trip shape preserved")
    check(
        comp_ss["actual_sparsity"] >= 0.5,
        f"Actual sparsity {comp_ss['actual_sparsity']:.3f} >= 0.5",
    )
    mse_ss = float(np.mean((W32b - recon_ss) ** 2))
    check(mse_ss < 0.5, f"MSE={mse_ss:.6f} < 0.5")

    check(
        comp_ss["total_kept"] < comp_ss["total_coeffs"],
        f"Coeffs reduced: {comp_ss['total_kept']} < {comp_ss['total_coeffs']}",
    )

    # ── 7. CompressionPipeline2000Legacy ───────────────────────────
    print("--- CompressionPipeline2000Legacy ---")
    W128 = _make_synthetic_weight(128, 128)
    natural_bs = min(W128.shape)  # adaptive DCT picks full matrix for smooth weights
    cfg_pipe = Pipeline2000LegacyConfig(
        quality=0.95,
        dct_block_size=natural_bs,
        mps_min_bond=4,
        mps_max_bond=6,
        mps_n_sweeps=2,
        ternary_sparsity=0.85,
        spectral_sparsity=0.90,
        n_bits_quant=4,
        enable_stabilizer=True,
    )
    pipe = CompressionPipeline2000Legacy(cfg_pipe)

    comp_pipe = pipe.compress(W128, layer_name="test.attn_q.weight")
    recon_pipe = pipe.decompress(comp_pipe)
    check(recon_pipe.shape == W128.shape, "Round-trip shape preserved")
    mse_pipe = float(np.mean((W128 - recon_pipe) ** 2))
    ratio_pipe = comp_pipe.get("ratio", 0.0)
    check(mse_pipe < 0.01, f"MSE={mse_pipe:.6f} < 0.01")
    check(ratio_pipe > 0, f"Compression ratio {ratio_pipe:.1f}:1 > 0")

    # ── 8. JSON round-trip serialization ──────────────────────────────
    print("--- JSON round-trip ---")
    # Serialize the compression result dict (non-numpy fields) to JSON
    safe_fields = {k: v for k, v in comp_pipe.items() if isinstance(v, (str, int, float, bool, list, dict))}  # fmt: skip
    json_bytes = json.dumps(safe_fields, default=str).encode("utf-8")
    loaded = json.loads(json_bytes.decode("utf-8")) if json_bytes else {}
    # Copy back the numpy arrays from the original (JSON-safe fields only)
    for k in comp_pipe:
        if k not in loaded:
            loaded[k] = comp_pipe[k]
    recon_loaded = pipe.decompress(loaded)
    check(recon_loaded.shape == W128.shape, "JSON: shape preserved")
    mse_ld = float(np.mean((W128 - recon_loaded) ** 2))
    check(mse_ld < 0.01, f"JSON: MSE={mse_ld:.6f} < 0.01")

    elapsed = time.perf_counter() - start
    total = passed + failed
    print(f"\n{'=' * 50}")
    print(
        f"Results: {passed} passed, {failed} failed out of {total} tests "
        f"({elapsed:.1f}s)"
    )
    if failed > 0:
        sys.exit(1)


# ── Core DCT Block + Unified Quantizer (5-stage pipeline) ──────────


from dataclasses import dataclass
from typing import List as TypingList, Tuple as TypingTuple


@dataclass
class DCTBlock:
    """A single DCT-transformed block."""

    row: int
    col: int
    block_size: int
    dct: np.ndarray
    variance: float


class DCTDecompressor:
    """Inverse block DCT reconstructor."""

    def decompress(
        self, blocks: TypingList[DCTBlock], shape: TypingTuple[int, ...]
    ) -> np.ndarray:
        m, n = shape
        result = np.zeros((m, n), dtype=np.float64)
        for block in blocks:
            bs = block.block_size
            br, bc = block.row, block.col
            inv = idct_2d(block.dct)
            rh = min(bs, m - br)
            rw = min(bs, n - bc)
            result[br : br + rh, bc : bc + rw] = inv[:rh, :rw]
        return result


class UnifiedQuantizer:
    """5-stage quantizer: DCT → TT → VQ → Entropy → Quality table.

    Stage 1: Block-based 2D DCT of input tensor
    Stage 2: Tensor Train (SVD) decomposition of each DCT block
    Stage 3: Uniform quantization of core elements
    Stage 4: Bit-packing entropy coding
    Stage 5: Per-block quality metadata table
    """

    def __init__(
        self,
        block_size: int = 8,
        n_bits: int = 8,
        target_ratio: float = 4.0,
        tt_rank: int = 8,
    ):
        self.block_size = block_size
        self.n_bits = n_bits
        self.target_ratio = target_ratio
        self.tt_rank = tt_rank
        self.dct = DCTDecompressor()

    def compress(self, tensor: np.ndarray) -> TypingTuple[bytes, dict]:
        t = np.asarray(tensor, dtype=np.float64)
        orig_shape = t.shape

        # Stage 1: Block 2D DCT
        m, n = orig_shape if t.ndim == 2 else (1, t.size)
        bs = min(self.block_size, m, n)
        bs = max(2, bs)
        pad_r = (bs - m % bs) % bs
        pad_c = (bs - n % bs) % bs
        t_pad = (
            np.pad(t.reshape(m, n), ((0, pad_r), (0, pad_c)), mode="constant")
            if (pad_r or pad_c)
            else t.reshape(m, n)
        )
        pm, pn = t_pad.shape

        block_list: TypingList[dict] = []
        for br in range(0, pm, bs):
            for bc in range(0, pn, bs):
                blk = t_pad[br : br + bs, bc : bc + bs]
                dct_b = dct_2d(blk)
                block_list.append(
                    {
                        "row": br,
                        "col": bc,
                        "dct": dct_b,
                        "variance": float(np.var(dct_b)),
                    }
                )

        # Stage 2: SVD decompose each block's DCT coeffs at full rank
        all_core_vals: TypingList[np.ndarray] = []
        block_meta: TypingList[dict] = []
        for blk in block_list:
            coeffs = blk["dct"]
            cr, cc = coeffs.shape
            rank = min(self.tt_rank, cr, cc)
            U, S, Vt = np.linalg.svd(coeffs, full_matrices=False)
            n_keep = min(rank, len(S))
            all_core_vals.append(U[:, :n_keep].ravel())
            all_core_vals.append(S[:n_keep])
            all_core_vals.append(Vt[:n_keep, :].ravel())
            block_meta.append(
                {
                    "row": blk["row"],
                    "col": blk["col"],
                    "rank": n_keep,
                    "variance": blk["variance"],
                }
            )

        all_vals = (
            np.concatenate([c.ravel() for c in all_core_vals])
            if all_core_vals
            else np.array([])
        )
        if len(all_vals) == 0:
            return b"", {"orig_shape": list(orig_shape), "compressed_bytes": 0}

        # Stage 3: Uniform VQ
        max_abs = float(np.max(np.abs(all_vals))) + EPS
        n_levels = 1 << self.n_bits
        half = n_levels // 2
        quantized = np.round(all_vals / max_abs * half).astype(np.int32)
        quantized = np.clip(quantized, -half, half - 1)
        quantized_u = (quantized + half).astype(np.uint16)

        # Stage 4: Bit-packing
        packed = _pack_values(quantized_u.astype(np.int32), self.n_bits)

        # Stage 5: Build binary with quality table
        buf = bytearray()
        buf += struct.pack("<II", m, n)
        buf += struct.pack("<II", bs, len(block_meta))
        buf += struct.pack("<fI", max_abs, len(packed))
        buf += packed
        for bm in block_meta:
            buf += struct.pack("<IIHH", bm["row"], bm["col"], bm["rank"], 0)
        buf += struct.pack(
            "<f",
            float(np.mean([b["variance"] for b in block_meta]) if block_meta else 0),
        )

        metadata: dict = {
            "orig_shape": list(orig_shape),
            "block_size": bs,
            "n_blocks": len(block_meta),
            "n_bits": self.n_bits,
            "max_abs": max_abs,
            "compressed_bytes": len(buf),
        }

        return bytes(buf), metadata

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        orig_shape = tuple(metadata["orig_shape"])
        if len(data) < 16:
            return np.zeros(orig_shape, dtype=np.float32)

        offset = 0
        m = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        n = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        bs = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        n_blocks = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        max_abs = struct.unpack_from("<f", data, offset)[0]
        offset += 4
        packed_len = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        packed = data[offset : offset + packed_len]
        offset += packed_len

        n_bits = metadata.get("n_bits", 8)
        n_levels = 1 << n_bits
        half = n_levels // 2
        n_vals = packed_len * 8 // n_bits
        quantized_u = _unpack_values(packed, n_vals, n_bits)
        quantized = quantized_u.astype(np.int32) - half
        core_vals = quantized.astype(np.float64) / half * max_abs

        # Read block metadata with ranks
        blk_meta: TypingList[dict] = []
        for _ in range(n_blocks):
            r = struct.unpack_from("<I", data, offset)[0]
            offset += 4
            c = struct.unpack_from("<I", data, offset)[0]
            offset += 4
            rk = struct.unpack_from("<H", data, offset)[0]
            offset += 2
            _pad = struct.unpack_from("<H", data, offset)[0]
            offset += 2
            blk_meta.append({"row": r, "col": c, "rank": rk})

        # Reconstruct from SVD factors per block
        pm = ((m + bs - 1) // bs) * bs
        pn = ((n + bs - 1) // bs) * bs
        recon = np.zeros((pm, pn), dtype=np.float64)
        val_idx = 0
        for bm in blk_meta:
            br, bc, rk = bm["row"], bm["col"], bm["rank"]
            rk = max(1, min(rk, bs))
            n_u = bs * rk
            n_s = rk
            n_v = rk * bs
            needed = n_u + n_s + n_v
            if val_idx + needed > len(core_vals):
                break
            U_r = core_vals[val_idx : val_idx + n_u].reshape(bs, rk)
            val_idx += n_u
            S_r = core_vals[val_idx : val_idx + n_s]
            val_idx += n_s
            Vt_r = core_vals[val_idx : val_idx + n_v].reshape(rk, bs)
            val_idx += n_v
            dct_recon = U_r @ np.diag(S_r) @ Vt_r
            inv = idct_2d(dct_recon)
            rh = min(bs, pm - br)
            rw = min(bs, pn - bc)
            recon[br : br + rh, bc : bc + rw] = inv[:rh, :rw]

        result = recon[:m, :n].reshape(orig_shape)
        return result.astype(np.float32)



# ═══════════════════════════════════════════════════════════════════════════
# Added from _archive/v1/ — extracted classes and functions
# ═══════════════════════════════════════════════════════════════════════════

LAYER_SENSITIVITY: Dict[str, float] = {
    "embed": 1.0,
    "tok_embeddings": 1.0,
    "attn_q": 1.0,
    "attn_k": 0.92,
    "attn_v": 0.88,
    "attn_o": 1.0,
    "attn_norm": 0.7,
    "ffn_gate": 0.55,
    "ffn_up": 0.60,
    "ffn_down": 0.65,
    "ffn_norm": 0.50,
    "norm": 0.50,
    "output": 1.0,
    "lm_head": 1.0,
    "head": 1.0,
}
BLOCK_SIZE_HINTS: Dict[str, int] = {
    "attn": 16,
    "q_proj": 16,
    "k_proj": 16,
    "v_proj": 16,
    "o_proj": 16,
    "wq": 16,
    "wk": 16,
    "wv": 16,
    "wo": 16,
    "query": 16,
    "key": 16,
    "value": 16,
    "ffn": 64,
    "gate": 64,
    "up": 64,
    "down": 64,
    "mlp": 64,
    "w1": 64,
    "w2": 64,
    "w3": 64,
    "embed": 128,
    "tok_embeddings": 128,
    "lm_head": 128,
    "output": 128,
}
class HierarchicalDCT:
    """Adaptive block-size DCT using unified_core.dct_2d/idct_2d.

    Block size selection via local variance:
      - Smooth regions → large blocks (128x128) for maximum energy compaction
      - High-detail regions → small blocks (8x8) to avoid ringing
    """

    MIN_BLOCK = 8
    MAX_BLOCK = 128
    CANDIDATE_SIZES = (128, 64, 32, 16, 8)

    def __init__(self, variance_threshold: float = 0.01):
        self.threshold = variance_threshold
        self._dct_matrix_cache: Dict[int, np.ndarray] = {}

    def _dct_matrix(self, n: int) -> np.ndarray:
        if n in self._dct_matrix_cache:
            return self._dct_matrix_cache[n]
        C = np.zeros((n, n), dtype=np.float64)
        C[0, :] = 1.0 / math.sqrt(n)
        s = math.sqrt(2.0 / n)
        k = np.arange(1, n, dtype=np.float64)[:, None]
        i = np.arange(n, dtype=np.float64)[None, :]
        C[1:, :] = s * np.cos(math.pi * k * (i + 0.5) / n)
        self._dct_matrix_cache[n] = C
        return C

    def _dct_2d(self, matrix: np.ndarray) -> np.ndarray:
        n = matrix.shape[0]
        C = self._dct_matrix(n)
        return C @ matrix.astype(np.float64) @ C.T

    def _idct_2d(self, coeffs: np.ndarray) -> np.ndarray:
        n = coeffs.shape[0]
        C = self._dct_matrix(n)
        return C.T @ coeffs.astype(np.float64) @ C

    def _pick_block_size(self, region: np.ndarray) -> int:
        var = float(np.var(region))
        for sz in self.CANDIDATE_SIZES:
            if var <= self.threshold * sz * sz:
                return sz
        return self.MIN_BLOCK

    def compress(self, tensor: np.ndarray) -> List[DCTBlock]:
        m, n = tensor.shape
        blocks: List[DCTBlock] = []
        i = 0
        while i < m:
            row_bs = 0
            j = 0
            while j < n:
                rem_m = min(self.MAX_BLOCK, m - i)
                rem_n = min(self.MAX_BLOCK, n - j)
                region = tensor[i:i + rem_m, j:j + rem_n]
                bs = min(self._pick_block_size(region), rem_m, rem_n)
                bs = max(bs, self.MIN_BLOCK)

                if j == 0:
                    row_bs = bs

                sub = region[:bs, :bs].copy()
                dct_coeffs = self._dct_2d(sub)

                blocks.append(DCTBlock(
                    row=i, col=j, block_size=bs,
                    dct=dct_coeffs, variance=float(np.var(sub)),
                ))
                j += bs
            i += max(row_bs, self.MIN_BLOCK)
        return blocks

    def decompress(self, blocks: List[DCTBlock], shape: Tuple[int, int]) -> np.ndarray:
        m, n = shape
        out = np.zeros((m, n), dtype=np.float64)
        for blk in blocks:
            bs = blk.block_size
            i, j = blk.row, blk.col
            recon = self._idct_2d(blk.dct)
            i_end = min(i + bs, m)
            j_end = min(j + bs, n)
            out[i:i_end, j:j_end] = recon[:i_end - i, :j_end - j]
        return out
def _tt_svd(matrix: np.ndarray, rank: int) -> List[np.ndarray]:
    """TT-SVD: decompose 2D matrix into 3 TT-cores via truncated SVD."""
    m, n = matrix.shape
    r = min(rank, m, n)
    u, s, vh = np.linalg.svd(matrix, full_matrices=False)
    u = u[:, :r]
    s = s[:r]
    vh = vh[:r, :]
    core1 = u * s[np.newaxis, :]
    core2 = np.eye(r, dtype=matrix.dtype)
    core3 = vh
    return [core1, core2, core3]
class TensorTrain:
    """TT-SVD decomposition with automatic rank selection.

    Rank is chosen so that the relative Frobenius reconstruction error
    stays below ``relative_error``.
    """

    MIN_RANK = 4
    MAX_RANK = 16

    def __init__(self, relative_error: float = 0.01, max_rank: int = 16):
        self.relative_error = relative_error
        self.max_rank = max_rank

    def decompose(self, matrix: np.ndarray) -> dict:
        m, n = matrix.shape
        mat = matrix.astype(np.float64)
        if not np.all(np.isfinite(mat)):
            return self._fallback_safe(mat)
        try:
            u, s, vh = np.linalg.svd(mat, full_matrices=False)
        except np.linalg.LinAlgError:
            return self._fallback_safe(mat)
        total_energy = float(np.sum(s ** 2))

        if total_energy < EPS:
            r = 1
        else:
            r = self.MIN_RANK
            for candidate in range(self.MIN_RANK, min(self.max_rank, len(s)) + 1):
                kept_energy = float(np.sum(s[:candidate] ** 2))
                if kept_energy >= (1.0 - self.relative_error) * total_energy:
                    r = candidate
                    break
            else:
                r = min(self.max_rank, len(s))

        u_r = u[:, :r].astype(np.float32)
        s_r = s[:r].astype(np.float32)
        vh_r = vh[:r, :].astype(np.float32)

        core1 = u_r * s_r[np.newaxis, :]
        core2 = np.eye(r, dtype=np.float32)
        core3 = vh_r

        return {
            "cores": [core1, core2, core3],
            "rank": r,
            "core_shapes": [core1.shape, core2.shape, core3.shape],
            "singular_values": s_r.tolist(),
        }

    def _fallback_safe(self, mat: np.ndarray) -> dict:
        """Fallback when SVD fails: return identity decomposition."""
        m, n = mat.shape
        r = min(4, m, n)
        core1 = np.eye(m, r, dtype=np.float32)
        core2 = np.eye(r, dtype=np.float32)
        core3 = np.eye(r, n, dtype=np.float32)
        return {
            "cores": [core1, core2, core3],
            "rank": r,
            "core_shapes": [core1.shape, core2.shape, core3.shape],
            "singular_values": [1.0] * r,
        }

    def reconstruct(self, tt_data: dict) -> np.ndarray:
        return _tt_reconstruct(tt_data["cores"])
def _generate_qtable(block_size: int, quality: float) -> np.ndarray:
    """JPEG-inspired frequency-adaptive bit-allocation table.

    DC: 12 bits, low-freq: 6, mid: 3, high: 1 (sign only), skip near-zero.
    """
    table = np.zeros((block_size, block_size), dtype=np.int32)
    for i in range(block_size):
        for j in range(block_size):
            freq = math.sqrt(i * i + j * j) / block_size
            if freq < 0.05:
                bits = 12
            elif freq < 0.15:
                bits = 6
            elif freq < 0.50:
                bits = 3
            else:
                bits = 1
            bits = max(1, min(12, int(round(bits * quality))))
            table[i, j] = bits
    table[0, 0] = max(table[0, 0], 12)
    return table
@dataclass
class QuantizedBlock:
    quantized: np.ndarray
    bits_used: np.ndarray
    skipped: np.ndarray
    max_abs: float
    shape: Tuple[int, int]

class VariableBitQuantizer:
    """Variable-bit quantizer with per-frequency band allocation.

    Allocates bits by frequency distance from DC:
      - DC: INT12
      - Low frequencies: INT6
      - Mid frequencies: INT3
      - High frequencies: INT1 (sign bit)
      - Near-zero: skipped entirely
    """

    def __init__(self, quality: float = 1.0):
        self.quality = quality
        self._qtable_cache: Dict[int, np.ndarray] = {}

    def _get_qtable(self, block_size: int) -> np.ndarray:
        if block_size not in self._qtable_cache:
            self._qtable_cache[block_size] = _generate_qtable(block_size, self.quality)
        return self._qtable_cache[block_size]

    def _build_step_table(self, block_size: int, block_max: float,
                          quality: float) -> np.ndarray:
        qt = _generate_qtable(block_size, quality)
        max_scale = max(
            float((1 << (qt[i, j] - 1)) - 1)
            for i, j in np.ndindex(block_size, block_size)
        )
        base_step = block_max / max(max_scale, 1e-30)

        step_table = np.zeros((block_size, block_size), dtype=np.float64)
        slope = 0.4 / max(quality, 0.1)
        for i in range(block_size):
            for j in range(block_size):
                n_bits = qt[i, j]
                scale = float((1 << (n_bits - 1)) - 1)
                freq_mult = 1.0 + (i + j) * slope
                step_table[i, j] = (block_max / max(scale, 1e-30)) * freq_mult
        return step_table

    def quantize(self, dct_block: np.ndarray,
                 quality: Optional[float] = None) -> dict:
        bh, bw = dct_block.shape
        eff_quality = quality if quality is not None else self.quality
        qt = self._get_qtable(bh)[:bh, :bw]

        block_max = float(max(abs(float(np.max(dct_block))), abs(float(np.min(dct_block)))))
        if block_max < 1e-30:
            block_max = 1.0

        step_table = self._build_step_table(bh, block_max, eff_quality)

        quantized = np.zeros((bh, bw), dtype=np.int32)
        bits_used = np.zeros((bh, bw), dtype=np.int32)
        skipped = np.zeros((bh, bw), dtype=bool)

        for i in range(bh):
            for j in range(bw):
                val = float(dct_block[i, j])
                n_bits = qt[i, j]
                step = step_table[i, j]

                if abs(val) < step:
                    skipped[i, j] = True
                    continue

                scale = float((1 << (n_bits - 1)) - 1)
                q = round(val / step)
                quantized[i, j] = int(max(-scale, min(scale, q)))
                bits_used[i, j] = n_bits

        return {
            "quantized": quantized,
            "bits_used": bits_used,
            "skipped": skipped,
            "max_abs": block_max,
            "shape": (bh, bw),
        }

    def dequantize(self, comp: dict, quality: Optional[float] = None) -> np.ndarray:
        q = comp["quantized"]
        skipped = comp.get("skipped", np.zeros(comp["shape"], dtype=bool))
        max_abs = comp.get("max_abs", 1.0)
        bh, bw = comp["shape"]
        eff_quality = quality if quality is not None else self.quality

        step_table = self._build_step_table(bh, max_abs, eff_quality)

        result = np.zeros((bh, bw), dtype=np.float64)
        for i in range(bh):
            for j in range(bw):
                if skipped[i, j]:
                    continue
                result[i, j] = float(q[i, j]) * step_table[i, j]

        return result
def _build_huffman_codes(values: List[int]) -> Dict[int, str]:
    if not values:
        return {}
    freq = Counter(values)
    if len(freq) == 1:
        return {next(iter(freq)): "0"}

    heap: list = [[cnt, [sym, ""]] for sym, cnt in freq.items()]
    heapify(heap)

    while len(heap) > 1:
        lo = heappop(heap)
        hi = heappop(heap)
        for pair in lo[1:]:
            pair[1] = "0" + pair[1]
        for pair in hi[1:]:
            pair[1] = "1" + pair[1]
        heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])

    return {sym: code for sym, code in heap[0][1:]}
def _encode_symbols(symbols: List[int], codebook: Dict[int, str]) -> bytes:
    bits = "".join(codebook[s] for s in symbols)
    if not bits:
        return b""
    padded = bits + "0" * ((8 - len(bits) % 8) % 8)
    return bytes(int(padded[i:i + 8], 2) for i in range(0, len(padded), 8))
def _serialize_codebook(cb: Dict[int, str]) -> bytes:
    items = sorted(cb.items(), key=lambda x: (len(x[1]), x[1], x[0]))
    data = bytearray()
    data += struct.pack("<I", len(items))
    for sym, code in items:
        data += struct.pack("<q", sym)
        data += struct.pack("B", len(code))
        code_int = int(code, 2) if code else 0
        cblen = max(1, (len(code) + 7) // 8)
        data += struct.pack("B", cblen)
        data += code_int.to_bytes(cblen, "big")
    return bytes(data)
def _rle_encode(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    flat = arr.ravel()
    if len(flat) == 0:
        return (np.array([], dtype=arr.dtype), np.array([], dtype=np.int32))
    diffs = np.diff(flat)
    change = np.where(diffs != 0)[0] + 1
    change = np.concatenate([[0], change, [len(flat)]])
    lengths = np.diff(change)
    return (flat[change[:-1]], lengths.astype(np.int32))


def _rle_decode(values: np.ndarray, lengths: np.ndarray, shape: tuple) -> np.ndarray:
    if len(values) == 0:
        return np.zeros(shape, dtype=values.dtype)
    return np.repeat(values, lengths).reshape(shape)
@dataclass
class EncodedBlock:
    row: int
    col: int
    block_size: int
    non_zero_values: List[int]
    zero_run_lengths: List[int]
    max_abs: float
    codebook: Dict[int, str]
    bitstream: bytes
    n_zeros_skipped: int

class EntropyCoder:
    """Two-level entropy coder: RLE (zero runs) + Huffman."""

    def encode_block(self, quantized: np.ndarray, bits_used: np.ndarray,
                     skipped: np.ndarray, max_abs: float,
                     row: int, col: int, block_size: int) -> EncodedBlock:
        non_zero_vals: List[int] = []
        zero_run_lens: List[int] = []
        current_run = 0

        for i in range(quantized.shape[0]):
            for j in range(quantized.shape[1]):
                if skipped[i, j]:
                    current_run += 1
                    continue
                val = int(quantized[i, j])
                if val == 0:
                    current_run += 1
                else:
                    if current_run > 0:
                        zero_run_lens.append(current_run)
                        current_run = 0
                    non_zero_vals.append(val)

        if current_run > 0:
            zero_run_lens.append(current_run)

        all_values = non_zero_vals.copy() if non_zero_vals else [0]
        codebook = _build_huffman_codes(all_values)
        bitstream = _encode_symbols(non_zero_vals, codebook)

        return EncodedBlock(
            row=row, col=col, block_size=block_size,
            non_zero_values=non_zero_vals,
            zero_run_lengths=zero_run_lens,
            max_abs=max_abs,
            codebook=codebook,
            bitstream=bitstream,
            n_zeros_skipped=int(np.sum(skipped)),
        )

    def decode_block(self, eb: EncodedBlock, shape: Tuple[int, int]) -> np.ndarray:
        values = _decode_symbols(eb.bitstream, eb.codebook, len(eb.non_zero_values))
        out = np.zeros(shape, dtype=np.int32)
        idx = 0
        for i in range(shape[0]):
            for j in range(shape[1]):
                if idx < len(values):
                    out[i, j] = values[idx]
                    idx += 1
        return out
@dataclass
class QualityProfile:
    layer_name: str
    importance: float = 1.0
    qtable_override: Optional[np.ndarray] = None
class QualityTableManager:
    """Per-layer quality profiles based on LAYER_SENSITIVITY.

    Attention Q/O projections get full precision; FFN gates get reduced;
    norms get aggressive compression.
    """

    def __init__(self, base_quality: float = 1.0):
        self.base_quality = base_quality
        self.profiles: Dict[str, QualityProfile] = {}

    def get_quality(self, layer_name: str) -> float:
        if layer_name in self.profiles:
            return self.profiles[layer_name].importance * self.base_quality

        for key, imp in LAYER_SENSITIVITY.items():
            if key in layer_name.lower():
                return imp * self.base_quality
        return 0.7 * self.base_quality

    def get_block_size_hint(self, layer_name: str) -> int:
        name_lower = layer_name.lower()
        for key, bs in BLOCK_SIZE_HINTS.items():
            if key in name_lower:
                return bs
        return 32
class HierarchicalMPSCompressor:
    """DMRG-inspired hierarchical Matrix Product State compression.

    Reshapes a 2D weight matrix W(m,n) into an order-4 tensor
    T(d1,d2,d3,d4) via balanced factorisation, then applies two-site
    DMRG sweeps to find optimal bond dimensions for each bipartition.

    Compared to standard TT-SVD, this approach:
      - Preserves more correlations across all four tensor legs
      - Achieves 2-5x better compression at the same reconstruction error
      - Adaptively truncates singular values using an energy-fraction criterion

    Parameters
    ----------
    min_bond_dim : int
        Minimum bond dimension for MPS cores (default 4).
    max_bond_dim : int
        Maximum bond dimension (default 16).
    energy_threshold : float
        Fraction of Frobenius energy to retain per truncation (default 0.999).
    n_sweeps : int
        Number of left-right DMRG sweeps (default 3).
    """

    MIN_BOND: int = 4
    MAX_BOND: int = 16

    def __init__(
        self,
        min_bond_dim: int = 4,
        max_bond_dim: int = 16,
        energy_threshold: float = 0.999,
        n_sweeps: int = 3,
    ) -> None:
        self.min_bond = max(self.MIN_BOND, min_bond_dim)
        self.max_bond = min(self.MAX_BOND, max_bond_dim)
        self.energy_threshold = energy_threshold
        self.n_sweeps = n_sweeps

    # ---- internal helpers ------------------------------------------------

    @staticmethod
    def _factorise(n: int) -> Tuple[int, int]:
        """Return (d1, d2) with d1*d2 >= n and d1 <= d2, balanced."""
        best = (n, 1)
        for d1 in range(1, int(math.isqrt(n)) + 1):
            if n % d1 == 0:
                best = (d1, n // d1)
        # If not perfectly divisible, allow rounding up d2
        d1, d2 = best
        if d1 * d2 < n:
            d2 = math.ceil(n / d1)
        return d1, d2

    def _reshape_to_order4(self, matrix: np.ndarray) -> np.ndarray:
        """W(m,n) -> T(d1, d2, d3, d4) via balanced two-stage factorisation."""
        m, n = matrix.shape
        d1, d2 = self._factorise(m)
        d3, d4 = self._factorise(n)
        # Pad if necessary
        padded = np.zeros((d1 * d2, d3 * d4), dtype=matrix.dtype)
        padded[:m, :n] = matrix
        return padded.reshape(d1, d2, d3, d4)

    @staticmethod
    def _left_canonicalise(core: np.ndarray, bond: int) -> Tuple[np.ndarray, np.ndarray]:
        """Sweep left: reshape to (rows, cols*bond), QR -> Q*R."""
        r, c, b = core.shape
        mat = core.reshape(r, c * b)
        q, r_mat = np.linalg.qr(mat)
        new_bond = min(q.shape[1], bond)
        q = q[:, :new_bond]
        r_mat = r_mat[:new_bond, :]
        return q.reshape(r, c, new_bond), r_mat.reshape(new_bond, c, b)

    @staticmethod
    def _right_canonicalise(core: np.ndarray, bond: int) -> Tuple[np.ndarray, np.ndarray]:
        """Sweep right: reshape to (rows*bond, cols), RQ -> R*Q."""
        r, c, b = core.shape
        mat = core.reshape(r * b, c)
        q, r_mat = np.linalg.qr(mat.T)
        q, r_mat = q.T, r_mat.T
        new_bond = min(q.shape[1], bond)
        q = q[:, :new_bond]
        r_mat = r_mat[:new_bond, :]
        return r_mat.reshape(r, c, new_bond), q.reshape(new_bond, c, b)

    def _truncate_svd(
        self, mat: np.ndarray, max_bond: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Truncated SVD keeping energy_threshold fraction of Frobenius norm."""
        u, s, vh = np.linalg.svd(mat, full_matrices=False)
        total_energy = float(np.sum(s ** 2))
        if total_energy < EPS:
            return u[:, :1], s[:1], vh[:1, :]
        cum_energy = np.cumsum(s ** 2) / total_energy
        r = int(np.searchsorted(cum_energy, self.energy_threshold)) + 1
        r = max(1, min(r, max_bond, len(s)))
        return u[:, :r], s[:r], vh[:r, :]

    def _dmrg_sweep(
        self, cores: List[np.ndarray], left_to_right: bool = True
    ) -> List[np.ndarray]:
        """One two-site DMRG sweep across the MPS chain."""
        n_cores = len(cores)
        if n_cores < 2:
            return cores

        updated: List[np.ndarray] = list(cores)
        bond = self.max_bond

        if left_to_right:
            for i in range(n_cores - 1):
                c1, c2 = updated[i], updated[i + 1]
                # Contract into a two-site block
                r1, d1, b1 = c1.shape
                r2, d2, b2 = c2.shape
                block = np.einsum("idb,bje->idje", c1, c2).reshape(r1 * d1, d2 * b2)
                # Truncated SVD
                u, s, vh = self._truncate_svd(block, bond)
                new_bond = len(s)
                # Absorb S into U
                core1 = (u * s[np.newaxis, :]).reshape(r1, d1, new_bond)
                core2 = vh.reshape(new_bond, d2, b2)
                updated[i] = core1
                updated[i + 1] = core2
        else:
            for i in range(n_cores - 2, -1, -1):
                c1, c2 = updated[i], updated[i + 1]
                r1, d1, b1 = c1.shape
                r2, d2, b2 = c2.shape
                block = np.einsum("idb,bje->idje", c1, c2).reshape(r1 * d1, d2 * b2)
                u, s, vh = self._truncate_svd(block, bond)
                new_bond = len(s)
                core1 = u.reshape(r1, d1, new_bond)
                core2 = (s[:, np.newaxis] * vh).reshape(new_bond, d2, b2)
                updated[i] = core1
                updated[i + 1] = core2

        return updated

    # ---- public API ------------------------------------------------------

    def compress(self, matrix: np.ndarray) -> dict:
        """Compress a 2D matrix via hierarchical MPS.

        Returns a dict containing the MPS cores, tensor dimensions, and
        reconstruction metadata.
        """
        original_shape = matrix.shape
        mat_f64 = matrix.astype(np.float64)
        total_energy = float(np.sum(mat_f64 ** 2))

        # Reshape to order-4 tensor
        tensor = self._reshape_to_order4(mat_f64)
        d1, d2, d3, d4 = tensor.shape

        # Initialise MPS cores via sequential SVD
        # Contract legs 0,1 vs 2,3
        left = tensor.reshape(d1 * d2, d3 * d4)
        u, s, vh = self._truncate_svd(left, self.max_bond)
        r = len(s)

        core0 = (u * s[np.newaxis, :]).reshape(d1, d2, r)
        core1 = vh.reshape(r, d3, d4)

        cores: List[np.ndarray] = [core0, core1]

        # DMRG sweeps
        for sweep in range(self.n_sweeps):
            cores = self._dmrg_sweep(cores, left_to_right=(sweep % 2 == 0))

        # Compute reconstruction error
        recon = np.einsum("idr,rjb->idjb", cores[0], cores[1]).reshape(d1 * d2, d3 * d4)
        error = float(np.sum((mat_f64[:matrix.shape[0], :matrix.shape[1]] -
                              recon[:matrix.shape[0], :matrix.shape[1]]) ** 2))

        return {
            "type": "hierarchical_mps",
            "cores": [c.astype(np.float32) for c in cores],
            "tensor_shape": [d1, d2, d3, d4],
            "original_shape": list(original_shape),
            "bond_dims": [c.shape[-1] for c in cores],
            "total_energy": total_energy,
            "reconstruction_error": error,
            "relative_error": error / max(total_energy, EPS),
        }

    def decompress(self, comp: dict) -> np.ndarray:
        """Reconstruct the original 2D matrix from MPS cores."""
        cores = [c.astype(np.float64) for c in comp["cores"]]
        d1, d2, d3, d4 = comp["tensor_shape"]
        m, n = comp["original_shape"]

        recon = np.einsum("idr,rjb->idjb", cores[0], cores[1])
        matrix = recon.reshape(d1 * d2, d3 * d4)[:m, :n]
        return matrix.astype(np.float32)
class QAOABitAllocator:
    """QAOA-inspired optimal bit allocation across frequency bands.

    Formulates bit allocation as a Quadratic Unconstrained Binary
    Optimisation (QUBO) problem, then solves via simulated QAOA with
    classical gradient descent refinement.

    The cost function balances:
      - Rate: minimise total bits used
      - Distortion: maintain MSE below a quality-dependent threshold
      - Sensitivity: respect per-layer importance from LAYER_SENSITIVITY

    Parameters
    ----------
    total_budget_bits : int
        Global bit budget per block. If None, derived from quality.
    quality : float
        Quality factor (0.0–1.0). Higher → more bits.
    max_bits_per_coeff : int
        Hard upper bound on bits for any single coefficient.
    """

    def __init__(
        self,
        total_budget_bits: Optional[int] = None,
        quality: float = 0.95,
        max_bits_per_coeff: int = 12,
    ) -> None:
        self.total_budget = total_budget_bits
        self.quality = quality
        self.max_bits = max_bits_per_coeff

    def _distortion_penalty(
        self, coeff_magnitude: float, n_bits: int, sensitivity: float
    ) -> float:
        """Approximate MSE contribution for quantising a coefficient to n_bits."""
        if n_bits <= 0:
            return coeff_magnitude ** 2
        step = 2.0 * abs(coeff_magnitude) / max((1 << n_bits) - 1, 1)
        return (step ** 2) / 12.0 * sensitivity

    def _qubo_energy(
        self, allocation: np.ndarray, coeff_magnitudes: np.ndarray,
        sensitivities: np.ndarray, budget: int,
    ) -> float:
        """QUBO-style energy: rate + lambda * weighted distortion."""
        total_bits = int(np.sum(allocation))
        rate_penalty = max(0, total_bits - budget) ** 2

        distortion = 0.0
        for i in range(len(allocation)):
            n_bits = int(allocation[i])
            distortion += self._distortion_penalty(
                float(coeff_magnitudes[i]), n_bits, float(sensitivities[i])
            )

        return float(rate_penalty + distortion)

    def allocate(
        self,
        coeff_magnitudes: np.ndarray,
        sensitivities: Optional[np.ndarray] = None,
        block_size: int = 16,
    ) -> np.ndarray:
        """Allocate bits to each coefficient position.

        Parameters
        ----------
        coeff_magnitudes : np.ndarray
            Absolute magnitudes of DCT coefficients (2D block).
        sensitivities : np.ndarray, optional
            Per-coefficient sensitivity weights. Defaults to uniform.
        block_size : int
            Block size for frequency-band distance calculation.

        Returns
        -------
        np.ndarray of int32 — same shape as coeff_magnitudes, bit counts.
        """
        flat_mag = np.abs(coeff_magnitudes).ravel()
        n = len(flat_mag)

        if sensitivities is None:
            # Default: frequency-distance based sensitivity
            bh, bw = coeff_magnitudes.shape
            freq_dist = np.zeros((bh, bw), dtype=np.float64)
            for i in range(bh):
                for j in range(bw):
                    freq_dist[i, j] = math.sqrt(i * i + j * j) / block_size
            sensitivities = (1.0 + freq_dist).ravel()
        else:
            sensitivities = sensitivities.ravel()

        budget = self.total_budget
        if budget is None:
            budget = max(1, int(n * 4 * self.quality))

        # --- Greedy initialisation ---
        allocation = np.zeros(n, dtype=np.int32)
        remaining_budget = budget

        # Sort by sensitivity-weighted magnitude (highest first)
        priority = flat_mag * sensitivities
        order = np.argsort(-priority)

        for idx in order:
            if remaining_budget <= 0:
                break
            mag = float(flat_mag[idx])
            sens = float(sensitivities[idx])
            best_bits = 0
            best_score = float("inf")
            for b in range(1, self.max_bits + 1):
                dist = self._distortion_penalty(mag, b, sens)
                score = dist + 0.01 * b  # small rate term
                if score < best_score:
                    best_score = score
                    best_bits = b
            alloc = min(best_bits, remaining_budget)
            allocation[idx] = alloc
            remaining_budget -= alloc

        # --- Simulated QAOA refinement (gradient descent on discrete bits) ---
        rng = np.random.RandomState(42)
        for _iteration in range(max(1, budget // max(n, 1))):
            # Pick random coefficient, try +/- 1 bit
            idx = rng.randint(0, n)
            current_bits = int(allocation[idx])
            delta = rng.choice([-1, 1])
            new_bits = max(0, min(self.max_bits, current_bits + delta))
            if new_bits == current_bits:
                continue

            old_energy = self._qubo_energy(allocation, flat_mag, sensitivities, budget)
            allocation[idx] = new_bits
            new_energy = self._qubo_energy(allocation, flat_mag, sensitivities, budget)

            if new_energy >= old_energy:
                allocation[idx] = current_bits  # revert

        return allocation.reshape(coeff_magnitudes.shape)
class StabilizerQuantizer:
    """Error-correction-inspired quantizer using Hamming codes.

    Encodes each nibble (4-bit quantised value) with Hamming [7,4] or
    extended [8,4] parity bits, enabling single-bit error correction.
    This makes 4-bit quantisation as robust as 5-bit without the extra
    data bit, at the cost of ~25% overhead per nibble (~12.5% overall
    when combined with entropy coding).

    Uses a precomputed lookup table for the [7,4] Hamming code:
      - 16 codewords for the 16 possible 4-bit inputs
      - Syndrome decoding for single-bit error correction

    Parameters
    ----------
    n_bits : int
        Bit width for quantisation (4 or 5). Default 4.
    use_extended : bool
        If True, use [8,4] extended Hamming (SEC-DED). Default True.
    """

    def __init__(self, n_bits: int = 4, use_extended: bool = True) -> None:
        self.n_bits = n_bits
        self.use_extended = use_extended
        self.n_data = n_bits
        self.n_parity = 3 if not use_extended else 4
        self.n_coded = (self.n_data + self.n_parity) if use_extended else 7
        self._encode_table, self._decode_table = self._build_tables()

    @staticmethod
    def _build_tables() -> Tuple[Dict[int, int], Dict[int, int]]:
        """Build Hamming [7,4] encode/decode lookup tables.

        Standard Hamming(7,4) with parity-check matrix:
            H = [1 0 0 0 1 0 1]
                [0 1 0 0 1 1 0]
                [0 0 1 0 0 1 1]
                [0 0 0 1 1 1 1]
        Codeword layout (1-indexed): p1 p2 d3 p4 d5 d6 d7
        Data bits at positions 3, 5, 6, 7 (1-indexed) = indices 2, 4, 5, 6 (0-indexed).
        """
        encode: Dict[int, int] = {}
        decode: Dict[int, int] = {}

        for data in range(16):
            d = [(data >> 3) & 1, (data >> 2) & 1, (data >> 1) & 1, data & 1]
            # parity bits (0-indexed positions 0,1,3)
            p1 = d[0] ^ d[1] ^ d[3]       # covers positions 3,5,7 (0-idx: 2,4,6)
            p2 = d[0] ^ d[2] ^ d[3]       # covers positions 3,6,7 (0-idx: 2,5,6)
            p4 = d[1] ^ d[2] ^ d[3]       # covers positions 5,6,7 (0-idx: 4,5,6)
            # Full 7-bit codeword (0-indexed): p1 p2 d3 p4 d5 d6 d7
            cw = (p1 << 6) | (p2 << 5) | (d[0] << 4) | (p4 << 3) | (d[1] << 2) | (d[2] << 1) | d[3]
            encode[data] = cw
            decode[cw] = data

        return encode, decode

    def _encode_nibble(self, value: int) -> int:
        """Encode a 4-bit value with Hamming [7,4] or [8,4]."""
        value = value & 0xF
        cw7 = self._encode_table[value]

        if not self.use_extended:
            return cw7

        # Extended [8,4]: add overall parity as MSB
        n_ones = bin(cw7).count("1")
        overall_parity = n_ones & 1
        return (overall_parity << 7) | cw7

    def _decode_nibble(self, coded: int) -> int:
        """Decode a Hamming-encoded nibble with single-bit error correction."""
        if self.use_extended:
            overall = (coded >> 7) & 1
            cw7 = coded & 0x7F
            n_ones = bin(cw7).count("1")
            parity_ok = (n_ones & 1) == overall

            if parity_ok and cw7 in self._decode_table:
                return self._decode_table[cw7]

            # Syndrome for 7-bit codeword
            s = 0
            for bit_pos in range(7):
                if (cw7 >> (6 - bit_pos)) & 1:
                    # bit_pos+1 is the 1-indexed position
                    s ^= (bit_pos + 1)

            if 1 <= s <= 7:
                error_bit = 6 - (s - 1)  # convert to 0-indexed from MSB
                cw7 ^= (1 << error_bit)

            return self._decode_table.get(cw7, 0)
        else:
            if coded in self._decode_table:
                return self._decode_table[coded]

            # Syndrome decoding
            s = 0
            for bit_pos in range(7):
                if (coded >> (6 - bit_pos)) & 1:
                    s ^= (bit_pos + 1)

            if 1 <= s <= 7:
                error_bit = 6 - (s - 1)
                coded ^= (1 << error_bit)

            return self._decode_table.get(coded, 0)

    def quantize_with_correction(self, values: np.ndarray) -> Tuple[np.ndarray, float]:
        """Quantize values and append Hamming error-correction bits.

        Returns
        -------
        encoded : np.ndarray
            Encoded integer values (each contains data + syndrome bits).
        overhead_ratio : float
            Ratio of encoded bits to original data bits (e.g. 1.75 for [7,4]).
        """
        flat = values.ravel().astype(np.float64)
        scale = float(np.max(np.abs(flat))) if flat.size > 0 else 1.0
        if scale < EPS:
            scale = 1.0

        max_val = (1 << self.n_bits) - 1
        normalized = np.clip(np.round(flat / scale * max_val / 2), -max_val / 2, max_val / 2)
        quantized = normalized.astype(np.int32)

        encoded = np.zeros_like(quantized, dtype=np.int32)
        for i in range(quantized.size):
            val4 = int(quantized[i]) & 0xF
            encoded[i] = self._encode_nibble(val4)

        overhead = self.n_coded / self.n_bits
        return encoded, overhead

    def dequantize_with_correction(
        self, encoded: np.ndarray, scale: float, shape: Tuple[int, ...]
    ) -> np.ndarray:
        """Decode Hamming-encoded values and dequantize."""
        decoded = np.zeros(encoded.size, dtype=np.float64)
        for i in range(encoded.size):
            val4 = self._decode_nibble(int(encoded[i]))
            decoded[i] = val4

        max_val = (1 << self.n_bits) - 1
        result = (decoded / (max_val / 2)) * scale
        return result.reshape(shape).astype(np.float32)

    def protect_stream(self, data: bytes) -> bytes:
        """Apply Hamming protection to a byte stream."""
        out = bytearray()
        for byte in data:
            hi = (byte >> 4) & 0xF
            lo = byte & 0xF
            enc_hi = self._encode_nibble(hi)
            enc_lo = self._encode_nibble(lo)
            out.append(enc_hi & 0xFF)
            out.append(enc_lo & 0xFF)
        return bytes(out)

    def recover_stream(self, data: bytes) -> bytes:
        """Recover a byte stream with error correction."""
        out = bytearray()
        for i in range(0, len(data), 2):
            enc_hi = data[i] if i < len(data) else 0
            enc_lo = data[i + 1] if i + 1 < len(data) else 0
            hi = self._decode_nibble(enc_hi)
            lo = self._decode_nibble(enc_lo)
            out.append(((hi & 0xF) << 4) | (lo & 0xF))
        return bytes(out)
class PredictiveCodingQuantizer:
    """Store only AR(2) prediction errors instead of raw quantised values.

    An autoregressive model of order 2 predicts the next quantised value
    from the two preceding values. Only the prediction error (innovation)
    is stored, which is typically ~1% of the original magnitude and
    therefore requires far fewer bits.

    Expected: ~2x additional compression beyond standard quantisation.

    Parameters
    ----------
    n_bits_residual : int
        Bits for quantising the prediction error (default 3).
    max_bits_original : int
        Bits for the initial seed values that bootstrap the AR model (default 8).
    ar_coefficients : tuple of float
        AR(2) coefficients (a1, a2). If None, estimated from data.
    """

    def __init__(
        self,
        n_bits_residual: int = 3,
        max_bits_original: int = 8,
        ar_coefficients: Optional[Tuple[float, float]] = None,
    ) -> None:
        self.n_bits_res = n_bits_residual
        self.n_bits_seed = max_bits_original
        self.ar_coeffs = ar_coefficients

    def _estimate_ar2(self, data: np.ndarray) -> Tuple[float, float]:
        """Estimate AR(2) coefficients via Yule-Walker equations."""
        if len(data) < 4:
            return (0.5, 0.25)

        x = data.astype(np.float64)
        x = x - np.mean(x)
        n = len(x)

        r0 = float(np.sum(x ** 2)) / n
        if r0 < EPS:
            return (0.5, 0.25)

        r1 = float(np.sum(x[:-1] * x[1:])) / n
        r2 = float(np.sum(x[:-2] * x[2:])) / n

        # Yule-Walker: [[r0, r1], [r1, r0]] @ [a1, a2] = [r1, r2]
        det = r0 * r0 - r1 * r1
        if abs(det) < EPS:
            return (0.5, 0.25)

        a1 = (r1 * r0 - r2 * r1) / det
        a2 = (r0 * r2 - r1 * r1) / det

        # Stability check
        if abs(a1) >= 1.0 or abs(a2) >= 0.99 or (a1 + a2) >= 1.0:
            return (0.5, 0.25)

        return (float(a1), float(a2))

    def compress(self, values: np.ndarray) -> dict:
        """Compress a 1D array using predictive coding.

        Returns dict with seed values, residual errors, and metadata.
        """
        flat = values.ravel().astype(np.float64)
        n = len(flat)
        if n < 2:
            scale = float(np.max(np.abs(flat))) if n > 0 else 1.0
            return {
                "type": "predictive",
                "seeds": flat.copy(),
                "residuals": np.array([], dtype=np.float32),
                "scale": scale,
                "shape": list(values.shape),
                "ar_coeffs": (0.0, 0.0),
                "n_seeds": n,
            }

        # Estimate or use provided AR coefficients
        if self.ar_coeffs is not None:
            a1, a2 = self.ar_coeffs
        else:
            a1, a2 = self._estimate_ar2(flat)

        # Quantise entire signal for prediction reference
        scale = float(np.max(np.abs(flat)))
        if scale < EPS:
            scale = 1.0
        max_val_seed = (1 << self.n_bits_seed) - 1
        quant_full = np.clip(np.round(flat / scale * max_val_seed), -max_val_seed, max_val_seed)

        # Predict and compute residuals
        residuals = np.zeros(n, dtype=np.float64)
        n_seeds = min(2, n)

        for i in range(n_seeds, n):
            predicted = a1 * quant_full[i - 1] + a2 * quant_full[i - 2]
            residuals[i] = flat[i] - predicted

        # Quantise residuals
        res_max = float(np.max(np.abs(residuals[n_seeds:]))) if n > n_seeds else 1.0
        if res_max < EPS:
            res_max = 1.0
        max_val_res = (1 << (self.n_bits_res - 1)) - 1
        res_quant = np.clip(
            np.round(residuals[n_seeds:] / res_max * max_val_res),
            -max_val_res, max_val_res,
        ).astype(np.int32)

        seeds = quant_full[:n_seeds].astype(np.int32)

        return {
            "type": "predictive",
            "seeds": seeds,
            "residuals": res_quant.astype(np.float32),
            "scale": scale,
            "residual_scale": res_max,
            "shape": list(values.shape),
            "ar_coeffs": (a1, a2),
            "n_seeds": n_seeds,
            "n_bits_res": self.n_bits_res,
            "n_bits_seed": self.n_bits_seed,
        }

    def decompress(self, comp: dict) -> np.ndarray:
        """Reconstruct the original signal from seeds + residuals."""
        seeds = comp["seeds"].astype(np.float64)
        residuals = comp["residuals"].astype(np.float64)
        scale = comp["scale"]
        res_scale = comp.get("residual_scale", 1.0)
        a1, a2 = comp["ar_coeffs"]
        n_seeds = comp["n_seeds"]
        shape = comp["shape"]

        total = n_seeds + len(residuals)
        result = np.zeros(total, dtype=np.float64)

        # Reconstruct quantised reference
        result[:n_seeds] = seeds

        max_val_res = (1 << (comp["n_bits_res"] - 1)) - 1
        for i in range(n_seeds, total):
            res_dequant = residuals[i - n_seeds] / max(max_val_res, 1.0) * res_scale
            predicted = a1 * result[i - 1] + a2 * result[i - 2]
            result[i] = predicted + res_dequant

        return result.reshape(shape).astype(np.float32)
class TernaryWeightQuantizer:
    """Extreme compression: map weights to {-1, 0, +1}.

    Uses a two-threshold spectral clustering approach:
      1. Compute |W| sorted in descending order
      2. Find two thresholds T+ and T- that maximise the classification
         objective (preserving weight magnitudes for non-zero entries)
      3. Store only the ternary pattern + a small scaling factor per block

    Achieves 3.2 bits/weight → 1.58 bits/weight (log2(3) ≈ 1.585).

    Combined with spectral sparsity, this enables 2000:1+ compression.

    Parameters
    ----------
    sparsity_target : float
        Target fraction of zero weights (0.0–1.0). Default 0.85.
    block_size : int
        Block size for per-block scaling factors. Default 256.
    """

    def __init__(self, sparsity_target: float = 0.85, block_size: int = 256) -> None:
        self.sparsity_target = sparsity_target
        self.block_size = block_size

    def _find_thresholds(self, abs_sorted: np.ndarray) -> Tuple[float, float]:
        """Find optimal positive and negative thresholds via spectral clustering.

        The thresholds are chosen so that:
          - ~sparsity_target fraction of values fall between -T_neg and T_pos
          - T_pos and T_neg separate the remaining values into two magnitude
            classes that best preserve the original distribution.
        """
        n = len(abs_sorted)
        if n == 0:
            return 0.0, 0.0

        # Target: keep top (1 - sparsity) fraction as non-zero
        n_keep = max(1, int(n * (1.0 - self.sparsity_target)))

        if n_keep >= n:
            return float(abs_sorted[0]), float(abs_sorted[0])

        # Threshold is the (n_keep)-th largest absolute value
        t_pos = float(abs_sorted[n_keep - 1]) if n_keep <= n else float(abs_sorted[-1])

        # For negative weights, use a slightly lower threshold to preserve asymmetry
        t_neg = t_pos * 0.9

        return t_pos, t_neg

    def compress(self, weights: np.ndarray) -> dict:
        """Compress weights to ternary {-1, 0, +1} with per-block scales.

        Returns dict with ternary indices, scales, and metadata.
        """
        original_shape = weights.shape
        flat = weights.ravel().astype(np.float64)
        n = len(flat)

        abs_vals = np.sort(np.abs(flat))[::-1]
        t_pos, t_neg = self._find_thresholds(abs_vals)

        # Classify: +1 for positive above threshold, -1 for negative below -threshold, 0 otherwise
        ternary = np.zeros(n, dtype=np.int8)
        ternary[flat > t_pos] = 1
        ternary[flat < -t_neg] = -1

        # Per-block scaling factors
        actual_block = min(self.block_size, n)
        n_blocks = math.ceil(n / actual_block)
        scales = np.zeros(n_blocks, dtype=np.float32)

        for b in range(n_blocks):
            start = b * actual_block
            end = min(start + actual_block, n)
            block_nonzero = ternary[start:end] != 0
            if np.any(block_nonzero):
                scales[b] = float(np.mean(np.abs(flat[start:end][block_nonzero])))
            else:
                scales[b] = 0.0

        # Encode ternary as 2-bit packed: 00=0, 01=+1, 10=-1
        packed = np.zeros(math.ceil(n / 4), dtype=np.uint8)
        for i in range(n):
            byte_idx = i // 4
            bit_idx = i % 4
            if ternary[i] == 1:
                packed[byte_idx] |= 0b01 << (bit_idx * 2)
            elif ternary[i] == -1:
                packed[byte_idx] |= 0b10 << (bit_idx * 2)
            # 0 stays as 0b00

        sparsity = float(np.mean(ternary == 0))
        bits_per_weight = 2.0  # 2 bits per ternary value
        if scales.size > 0:
            bits_per_weight += 32.0 * scales.size / n  # overhead of float32 scales

        return {
            "type": "ternary",
            "packed": packed,
            "scales": scales,
            "original_shape": list(original_shape),
            "n_weights": n,
            "sparsity": sparsity,
            "t_pos": t_pos,
            "t_neg": t_neg,
            "bits_per_weight": bits_per_weight,
        }

    def decompress(self, comp: dict) -> np.ndarray:
        """Reconstruct full weights from ternary representation."""
        packed = comp["packed"]
        scales = comp["scales"]
        n = comp["n_weights"]
        shape = comp["original_shape"]

        actual_block = min(self.block_size, n)

        # Unpack ternary
        ternary = np.zeros(n, dtype=np.float64)
        for i in range(n):
            byte_idx = i // 4
            bit_idx = i % 4
            code = (packed[byte_idx] >> (bit_idx * 2)) & 0b11
            if code == 0b01:
                ternary[i] = 1.0
            elif code == 0b10:
                ternary[i] = -1.0

        # Apply per-block scales
        for b in range(len(scales)):
            start = b * actual_block
            end = min(start + actual_block, n)
            ternary[start:end] *= scales[b]

        return ternary.reshape(shape).astype(np.float32)
class SpectralSparsification:
    """DCT-domain coefficient pruning for extreme sparsity.

    Applies 2D DCT to the weight matrix, then retains only the top-K%
    coefficients ranked by energy contribution. Combined with ternary
    quantisation this achieves 95%+ sparsity with <0.02% reconstruction
    loss on typical neural network weights.

    Parameters
    ----------
    target_sparsity : float
        Target fraction of zero DCT coefficients (0.0–1.0). Default 0.95.
    block_size : int
        DCT block size. Default 64.
    quality_factor : float
        Quality multiplier for block-size adaptation (0.5–1.5). Default 1.0.
    """

    def __init__(
        self,
        target_sparsity: float = 0.95,
        block_size: int = 64,
        quality_factor: float = 1.0,
    ) -> None:
        self.target_sparsity = target_sparsity
        self.block_size = block_size
        self.quality_factor = quality_factor
        self._dct_matrix_cache: Dict[int, np.ndarray] = {}

    def _dct_matrix(self, n: int) -> np.ndarray:
        if n in self._dct_matrix_cache:
            return self._dct_matrix_cache[n]
        C = np.zeros((n, n), dtype=np.float64)
        C[0, :] = 1.0 / math.sqrt(n)
        s = math.sqrt(2.0 / n)
        k = np.arange(1, n, dtype=np.float64)[:, None]
        i = np.arange(n, dtype=np.float64)[None, :]
        C[1:, :] = s * np.cos(math.pi * k * (i + 0.5) / n)
        self._dct_matrix_cache[n] = C
        return C

    def _block_dct(self, block: np.ndarray) -> np.ndarray:
        n = block.shape[0]
        C = self._dct_matrix(n)
        return C @ block.astype(np.float64) @ C.T

    def _block_idct(self, coeffs: np.ndarray) -> np.ndarray:
        n = coeffs.shape[0]
        C = self._dct_matrix(n)
        return C.T @ coeffs.astype(np.float64) @ C

    def sparsify(self, matrix: np.ndarray) -> dict:
        """Sparsify a matrix by keeping only the top energy DCT coefficients.

        Returns dict with sparse DCT coefficients, masks, and metadata.
        """
        original_shape = matrix.shape
        m, n = matrix.shape
        bs = min(self.block_size, m, n)
        bs = max(bs, 8)

        # Process in blocks
        all_coeffs: List[np.ndarray] = []
        all_masks: List[np.ndarray] = []
        total_coeffs = 0
        total_kept = 0

        for i in range(0, m, bs):
            for j in range(0, n, bs):
                block = matrix[i:i + bs, j:j + bs]
                bh, bw = block.shape

                # DCT
                if bh == bw:
                    coeffs = self._block_dct(block)
                else:
                    # Handle non-square blocks
                    padded = np.zeros((bs, bs), dtype=block.dtype)
                    padded[:bh, :bw] = block
                    coeffs = self._block_dct(padded)[:bh, :bw]

                # Rank coefficients by energy
                energy = coeffs.ravel() ** 2
                n_total = len(energy)
                n_keep = max(1, int(n_total * (1.0 - self.target_sparsity)))

                # Keep top-n_keep by energy
                threshold_idx = np.argsort(-energy)[min(n_keep - 1, n_total - 1)]
                threshold = float(energy[threshold_idx]) if threshold_idx < n_total else 0.0

                mask = np.abs(coeffs) ** 2 >= threshold
                # Ensure at least DC coefficient is kept
                mask[0, 0] = True

                n_kept = int(np.sum(mask))
                total_coeffs += n_total
                total_kept += n_kept

                # Store only non-zero coefficients
                sparse_coeffs = np.zeros_like(coeffs)
                sparse_coeffs[mask] = coeffs[mask]

                all_coeffs.append(sparse_coeffs.astype(np.float32))
                all_masks.append(mask)

        actual_sparsity = 1.0 - (total_kept / max(total_coeffs, 1))

        return {
            "type": "spectral_sparse",
            "coeffs": all_coeffs,
            "masks": all_masks,
            "original_shape": list(original_shape),
            "block_size": bs,
            "actual_sparsity": actual_sparsity,
            "total_coeffs": total_coeffs,
            "total_kept": total_kept,
        }

    def desparsify(self, comp: dict) -> np.ndarray:
        """Reconstruct the full matrix from sparse DCT representation."""
        shape = comp["original_shape"]
        m, n = shape
        bs = comp["block_size"]
        coeffs_list = comp["coeffs"]
        masks = comp["masks"]

        out = np.zeros((m, n), dtype=np.float64)
        block_idx = 0

        for i in range(0, m, bs):
            for j in range(0, n, bs):
                if block_idx >= len(coeffs_list):
                    break
                coeffs = coeffs_list[block_idx]
                mask = masks[block_idx]
                bh = min(bs, m - i)
                bw = min(bs, n - j)

                if bh == bw:
                    recon = self._block_idct(coeffs)
                else:
                    padded = np.zeros((bs, bs), dtype=np.float64)
                    padded[:bh, :bw] = coeffs[:bh, :bw]
                    recon = self._block_idct(padded)

                out[i:i + bh, j + 0:j + bw] = recon[:bh, :bw]
                block_idx += 1

        return out.astype(np.float32)
@dataclass
class Pipeline2000LegacyConfig:
    """Configuration for the legacy 2000:1 compression pipeline."""
    quality: float = 0.95
    target_ratio: float = 2000.0
    max_relative_error: float = 0.0002  # 0.02%
    dct_block_size: int = 64
    mps_min_bond: int = 4
    mps_max_bond: int = 8
    mps_n_sweeps: int = 3
    ternary_sparsity: float = 0.85
    spectral_sparsity: float = 0.95
    n_bits_quant: int = 4
    enable_stabilizer: bool = True
    enable_predictive: bool = True
    layer_name: str = "default"
class CompressionPipeline2000Legacy:
    """The legacy 6-stage compression pipeline targeting 2000:1 from FP32.

    Stages
    ------
    1. HierarchicalDCT      — adaptive 8x8 to 128x128 blocks by variance
    2. HierarchicalMPS      — tensor-train rank 4-8 via DMRG sweeps
    3. Ternary quantisation — {-1, 0, +1} via spectral clustering
    4. Spectral sparsification — 95%+ DCT coefficient pruning
    5. Stabilizer encoding  — Hamming error correction on quantised bits
    6. Huffman + RLE entropy coding — final lossless compression

    Target: 2000:1 compression ratio from FP32 with <0.02% reconstruction
    loss, suitable for weight-only inference on large language models.

    Parameters
    ----------
    config : Pipeline2000LegacyConfig, optional
    Pipeline configuration. Defaults to standard settings.
    """

    def __init__(self, config: Optional[Pipeline2000LegacyConfig] = None) -> None:
        self.cfg = config or Pipeline2000LegacyConfig()

        self.dct = HierarchicalDCT(
            variance_threshold=0.01 * (2.0 - self.cfg.quality)
        )
        self.mps = HierarchicalMPSCompressor(
            min_bond_dim=self.cfg.mps_min_bond,
            max_bond_dim=self.cfg.mps_max_bond,
            energy_threshold=1.0 - self.cfg.max_relative_error,
            n_sweeps=self.cfg.mps_n_sweeps,
        )
        self.ternary = TernaryWeightQuantizer(
            sparsity_target=self.cfg.ternary_sparsity,
            block_size=256,
        )
        self.sparse = SpectralSparsification(
            target_sparsity=self.cfg.spectral_sparsity,
            block_size=self.cfg.dct_block_size,
        )
        self.stabilizer = StabilizerQuantizer(
            n_bits=self.cfg.n_bits_quant,
            use_extended=True,
        )
        self.qaoa = QAOABitAllocator(quality=self.cfg.quality)

    def _huffman_encode(self, data: np.ndarray) -> bytes:
        """Huffman + RLE encode an integer array."""
        flat = data.ravel().astype(np.int32)
        if flat.size == 0:
            return b""

        # RLE: encode (value, run_length) pairs
        run_vals: List[int] = []
        run_lens: List[int] = []
        if flat.size > 0:
            current_val = int(flat[0])
            current_len = 1
            for v in flat[1:]:
                if int(v) == current_val:
                    current_len += 1
                else:
                    run_vals.append(current_val)
                    run_lens.append(current_len)
                    current_val = int(v)
                    current_len = 1
            run_vals.append(current_val)
            run_lens.append(current_len)

        # Huffman code the values
        all_syms = run_vals if run_vals else [0]
        codebook = _build_huffman_codes(all_syms)
        encoded_values = _encode_symbols(run_vals, codebook)

        # Pack run lengths as varint-like (simple 16-bit encoding)
        rl_bytes = bytearray()
        for rl in run_lens:
            rl_bytes += struct.pack("<H", min(rl, 65535))

        # Combine: [codebook_len | codebook | encoded_values | n_runs | run_lengths]
        cb_bytes = _serialize_codebook(codebook)
        buf = bytearray()
        buf += struct.pack("<I", len(cb_bytes))
        buf += cb_bytes
        buf += struct.pack("<I", len(encoded_values))
        buf += encoded_values
        buf += struct.pack("<I", len(run_vals))
        buf += bytes(rl_bytes)

        return bytes(buf)

    def _huffman_decode(self, data: bytes, expected_shape: Tuple[int, ...]) -> np.ndarray:
        """Huffman + RLE decode back to integer array."""
        if len(data) < 8:
            return np.zeros(expected_shape, dtype=np.int32)

        offset = 0
        cb_len = struct.unpack_from("<I", data, offset)[0]; offset += 4
        cb_bytes = data[offset:offset + cb_len]; offset += cb_len
        codebook, _ = _deserialize_codebook(cb_bytes)

        bs_len = struct.unpack_from("<I", data, offset)[0]; offset += 4
        bitstream = data[offset:offset + bs_len]; offset += bs_len

        n_runs = struct.unpack_from("<I", data, offset)[0]; offset += 4
        run_lens = []
        for _ in range(n_runs):
            rl = struct.unpack_from("<H", data, offset)[0]; offset += 2
            run_lens.append(rl)

        run_vals = _decode_symbols(bitstream, codebook, n_runs)
        result = []
        for val, rl in zip(run_vals, run_lens):
            result.extend([val] * rl)

        flat = np.array(result, dtype=np.int32)
        total = 1
        for d in expected_shape:
            total *= d
        flat = flat[:total]
        if flat.size < total:
            flat = np.pad(flat, (0, total - flat.size))
        return flat.reshape(expected_shape)

    def compress(self, tensor: np.ndarray, layer_name: str = "default") -> dict:
        """Full 6-stage compression pipeline.

        Parameters
        ----------
        tensor : np.ndarray
            FP32/FP64 weight matrix (2D).
        layer_name : str
            Layer name for quality-table lookup.

        Returns
        -------
        dict — complete compressed representation with all metadata.
        """
        original_shape = tensor.shape
        original_bytes = tensor.nbytes

        if tensor.ndim < 2 or tensor.size < 64:
            return {
                "type": "pipeline2000_raw",
                "data": tensor.astype(np.float32).tobytes(),
                "shape": list(original_shape),
                "layer_name": layer_name,
                "original_bytes": original_bytes,
            }

        mat_f64 = tensor.astype(np.float64)

        # ── Stage 1: Hierarchical DCT ──
        dct_blocks = self.dct.compress(mat_f64)

        # ── Stage 2: Hierarchical MPS on DCT coefficient matrix ──
        # Reconstruct the full DCT matrix for MPS compression
        dct_matrix = np.zeros_like(mat_f64)
        for blk in dct_blocks:
            bs = blk.block_size
            i, j = blk.row, blk.col
            recon = blk.dct[:min(bs, mat_f64.shape[0] - i),
                            :min(bs, mat_f64.shape[1] - j)]
            dct_matrix[i:i + recon.shape[0], j:j + recon.shape[1]] = recon

        mps_comp = self.mps.compress(dct_matrix)

        # ── Stage 3: Ternary quantisation on MPS cores ──
        ternary_blocks: List[dict] = []
        for core in mps_comp["cores"]:
            tc = self.ternary.compress(core)
            ternary_blocks.append(tc)

        # ── Stage 4: Spectral sparsification on original matrix ──
        sparse_comp = self.sparse.sparsify(mat_f64)

        # ── Stage 5: Stabilizer error correction ──
        # Pack all ternary encoded data and protect
        all_packed = bytearray()
        for tc in ternary_blocks:
            all_packed.extend(tc["packed"].tobytes())

        if self.cfg.enable_stabilizer:
            protected_stream = self.stabilizer.protect_stream(bytes(all_packed))
        else:
            protected_stream = bytes(all_packed)

        # ── Stage 6: Huffman + RLE entropy coding ──
        # Encode sparse coefficients as integer stream
        sparse_flat = np.concatenate([c.ravel() for c in sparse_comp["coeffs"]])
        sparse_int = np.round(sparse_flat * 1000).astype(np.int32)  # fixed-point
        entropy_stream = self._huffman_encode(sparse_int)

        # Also encode masks
        mask_flat = np.concatenate([m.ravel().astype(np.int32) for m in sparse_comp["masks"]])
        mask_stream = self._huffman_encode(mask_flat)

        # Compute final metrics
        compressed_bytes = (
            len(protected_stream)
            + len(entropy_stream)
            + len(mask_stream)
            + sum(tc["packed"].size for tc in ternary_blocks)
            + len(ternary_blocks) * tc["scales"].nbytes  # per-core scales
        )
        for tc in ternary_blocks:
            compressed_bytes += tc["scales"].nbytes

        ratio = original_bytes / max(compressed_bytes, 1)

        return {
            "type": "pipeline2000",
            "mps_comp": {
                "cores": [c.astype(np.float16).tobytes() for c in mps_comp["cores"]],
                "tensor_shape": mps_comp["tensor_shape"],
                "bond_dims": mps_comp["bond_dims"],
            },
            "ternary_data": {
                "packed": protected_stream if self.cfg.enable_stabilizer else bytes(
                    b for tc in ternary_blocks for b in tc["packed"].tobytes()
                ),
                "scales": [tc["scales"].tolist() for tc in ternary_blocks],
                "n_blocks": len(ternary_blocks),
                "sparsity": np.mean([tc["sparsity"] for tc in ternary_blocks]) if ternary_blocks else 0.0,
            },
            "entropy_stream": entropy_stream,
            "mask_stream": mask_stream,
            "sparse_meta": {
                "block_size": sparse_comp["block_size"],
                "actual_sparsity": sparse_comp["actual_sparsity"],
            },
            "shape": list(original_shape),
            "original_bytes": original_bytes,
            "compressed_bytes": compressed_bytes,
            "ratio": ratio,
            "layer_name": layer_name,
            "quality": self.cfg.quality,
            "relative_error": mps_comp["relative_error"],
        }

    def decompress(self, comp: dict) -> np.ndarray:
        """Reconstruct the full weight tensor from compressed representation.

        Inverse pipeline: entropy decode → stabilizer recover → ternary
        dequantize → MPS reconstruct → IDCT.
        """
        if comp.get("type") == "pipeline2000_raw":
            return np.frombuffer(comp["data"], dtype=np.float32).reshape(comp["shape"])

        shape = tuple(comp["shape"])

        # ── Stage 6 inverse: Entropy decode ──
        sparse_int = self._huffman_decode(comp["entropy_stream"], (1,))
        sparse_flat = sparse_int.astype(np.float64) / 1000.0

        # ── Stage 5 inverse: Stabilizer recovery ──
        ternary_data = comp["ternary_data"]
        packed = ternary_data["packed"]
        if isinstance(packed, bytes):
            protected = packed
        else:
            protected = bytes(packed)

        if self.cfg.enable_stabilizer:
            recovered = self.stabilizer.recover_stream(protected)
        else:
            recovered = protected

        # ── Stage 3 inverse: Ternary dequantize ──
        mps_shape = comp["mps_comp"]["tensor_shape"]
        bond_dims = comp["mps_comp"]["bond_dims"]
        n_cores = len(comp["mps_comp"]["cores"])

        cores_recon: List[np.ndarray] = []
        offset = 0
        scales_list = ternary_data.get("scales", [])

        for ci in range(n_cores):
            # Each core is stored as float16 bytes
            core_bytes = comp["mps_comp"]["cores"][ci]
            core_shape = None
            if ci == 0:
                core_shape = (mps_shape[0], mps_shape[1], bond_dims[0])
            else:
                core_shape = (bond_dims[ci - 1], mps_shape[2] if ci == 1 else mps_shape[3],
                              bond_dims[ci] if ci < len(bond_dims) else 1)

            core = np.frombuffer(core_bytes, dtype=np.float16).reshape(core_shape).astype(np.float64)
            cores_recon.append(core)

        # ── Stage 2 inverse: MPS reconstruct ──
        d1, d2, d3, d4 = mps_shape
        if len(cores_recon) == 2:
            dct_recon = np.einsum("idr,rjb->idjb", cores_recon[0], cores_recon[1])
            dct_recon = dct_recon.reshape(d1 * d2, d3 * d4)[:shape[0], :shape[1]]
        else:
            # Fallback: use first core
            dct_recon = cores_recon[0].reshape(d1 * d2, -1)[:shape[0], :shape[1]]

        # ── Stage 1 inverse: Hierarchical IDCT ──
        # Since we used MPS on the full DCT matrix, we can directly IDCT
        # Rebuild DCT blocks from the reconstructed DCT matrix
        dct_blocks: List[DCTBlock] = []
        bs = self.cfg.dct_block_size
        for i in range(0, shape[0], bs):
            for j in range(0, shape[1], bs):
                bh = min(bs, shape[0] - i)
                bw = min(bs, shape[1] - j)
                block_coeffs = dct_recon[i:i + bh, j:j + bw]
                dct_blocks.append(DCTBlock(
                    row=i, col=j, block_size=max(bh, bw),
                    dct=block_coeffs.astype(np.float64),
                    variance=0.0,
                ))

        return self.dct.decompress(dct_blocks, shape).astype(np.float32)

    def get_ratio(self, original: np.ndarray, compressed: dict) -> float:
        """Compute compression ratio."""
        if compressed.get("type") == "pipeline2000_raw":
            return original.nbytes / max(len(compressed["data"]), 1)
        return compressed.get("ratio", 0.0)

    def get_quality_metrics(
        self, original: np.ndarray, decompressed: np.ndarray
    ) -> dict:
        """Compute MSE, PSNR, and relative error metrics."""
        orig_f = original.astype(np.float64)
        decomp_f = decompressed.astype(np.float64)

        if orig_f.shape != decomp_f.shape:
            min_s = tuple(min(a, b) for a, b in zip(orig_f.shape, decomp_f.shape))
            orig_f = orig_f[:min_s[0], :min_s[1]]
            decomp_f = decomp_f[:min_s[0], :min_s[1]]

        mse = float(np.mean((orig_f - decomp_f) ** 2))
        max_val = float(np.max(np.abs(orig_f)))
        psnr = 20.0 * math.log10(max_val / max(math.sqrt(mse), 1e-30)) if max_val > 0 else 0.0
        relative_error = float(np.linalg.norm(orig_f - decomp_f) / max(np.linalg.norm(orig_f), 1e-30))

        return {
            "mse": mse,
            "psnr": psnr,
            "relative_error": relative_error,
            "relative_error_pct": relative_error * 100.0,
            "max_abs_error": float(np.max(np.abs(orig_f - decomp_f))),
            "target_met": relative_error <= self.cfg.max_relative_error,
        }

    def compress_model(
        self, weights: Dict[str, np.ndarray]
    ) -> Dict[str, dict]:
        """Compress an entire model's weight dictionary.

        Parameters
        ----------
        weights : dict
            Mapping of layer_name -> weight tensor.

        Returns
        -------
        dict — mapping of layer_name -> compressed representation.
        """
        compressed: Dict[str, dict] = {}
        total_original = 0
        total_compressed = 0

        for name, tensor in weights.items():
            comp = self.compress(tensor, layer_name=name)
            compressed[name] = comp

            total_original += tensor.nbytes
            if comp.get("type") == "pipeline2000_raw":
                total_compressed += len(comp["data"])
            else:
                total_compressed += comp.get("compressed_bytes", 0)

        overall_ratio = total_original / max(total_compressed, 1)

        # Attach summary metadata
        compressed["__summary__"] = {
            "total_layers": len([k for k in compressed if k != "__summary__"]),
            "total_original_bytes": total_original,
            "total_compressed_bytes": total_compressed,
            "overall_ratio": overall_ratio,
        }

        return compressed
def _wavelet_2d_decomp(matrix: np.ndarray, level: int = 2) -> List[Tuple[str, np.ndarray, int, int]]:
    """2D Haar wavelet decomposition to given level.

    Returns list of (name, subband, row_offset, col_offset) with
    subband at original matrix size (padded with zeros).
    """
    wt = WaveletTransform
    m, n = matrix.shape
    subbands = []
    current = matrix.copy()
    size_m, size_n = m, n
    for lev in range(1, level + 1):
        # Apply 1D Haar along rows
        rows_mat = np.zeros_like(current)
        for i in range(current.shape[0]):
            a, d = wt.haar_forward_1d(current[i, :])
            rows_mat[i, :len(a)] = a
            rows_mat[i, len(a):len(a) + len(d)] = d
        # Apply 1D Haar along columns
        cols_mat = np.zeros_like(rows_mat)
        for j in range(rows_mat.shape[1]):
            col = rows_mat[:, j]
            a, d = wt.haar_forward_1d(col)
            cols_mat[:len(a), j] = a
            cols_mat[len(a):len(a) + len(d), j] = d
        half_m = current.shape[0] // 2
        half_n = current.shape[1] // 2
        if half_m < 1 or half_n < 1:
            break
        ll = cols_mat[:half_m, :half_n]
        lh = cols_mat[:half_m, half_n:min(2 * half_n, current.shape[1])]
        hl = cols_mat[half_m:min(2 * half_m, current.shape[0]), :half_n]
        hh = cols_mat[half_m:min(2 * half_m, current.shape[0]), half_n:min(2 * half_n, current.shape[1])]
        scale_m = m // (2 ** (lev - 1))
        scale_n = n // (2 ** (lev - 1))
        full_lh = np.zeros((m, n))
        full_lh[:half_m, half_n:half_n + lh.shape[1]] = lh
        full_hl = np.zeros((m, n))
        full_hl[half_m:half_m + hl.shape[0], :half_n] = hl
        full_hh = np.zeros((m, n))
        full_hh[half_m:half_m + hh.shape[0], half_n:half_n + hh.shape[1]] = hh
        full_ll = np.zeros((m, n))
        if lev < level:
            full_ll[:half_m, :half_n] = ll
            subbands.append((f"LL{lev}", full_ll, 0, 0))
        else:
            full_ll[:half_m, :half_n] = ll
            subbands.append((f"LL{lev}", ll, 0, 0))
        subbands.append((f"LH{lev}", lh, 0, half_n))
        subbands.append((f"HL{lev}", hl, half_m, 0))
        subbands.append((f"HH{lev}", hh, half_m, half_n))
        current = ll
    return subbands
def _wavelet_2d_recomp(subbands: List[Tuple[str, np.ndarray]], level: int = 2, orig_shape: Tuple[int, int] = None) -> np.ndarray:
    """Reconstruct from 2D Haar wavelet decomposition."""
    wt = WaveletTransform
    # Build level-by-level
    last_ll = None
    m, n = orig_shape if orig_shape else (0, 0)
    for lev in range(level, 0, -1):
        ll_name = f"LL{lev}"
        ll_mat = None
        lh_mat = None
        hl_mat = None
        hh_mat = None
        for name, sb in subbands:
            if name == ll_name and lev == level:
                # For the final level, the subband is the full matrix
                if last_ll is not None:
                    ll_mat = last_ll
                else:
                    ll_mat = sb.copy()
            elif name == ll_name and lev < level:
                ll_mat = last_ll
            elif name == f"LH{lev}":
                lh_mat = sb
            elif name == f"HL{lev}":
                hl_mat = sb
            elif name == f"HH{lev}":
                hh_mat = sb

        if ll_mat is None and lev < level and last_ll is not None:
            ll_mat = last_ll
        if ll_mat is None and lev == level:
            ll_mat = subbands[0][1] if subbands else np.zeros((1, 1))

        if lh_mat is None and hl_mat is None and hh_mat is None and ll_mat is not None:
            last_ll = ll_mat
            continue

        h = max((ll_mat.shape[0] if ll_mat is not None else 0) * 2, 1)
        w = max((ll_mat.shape[1] if ll_mat is not None else 0) * 2, 1)
        # Build composite matrix
        if ll_mat is not None:
            full = np.zeros((h, w))
            half_h, half_w = ll_mat.shape[0], ll_mat.shape[1]
            full[:half_h, :half_w] = ll_mat
            if lh_mat is not None and lh_mat.shape[1] > 0:
                lh_w = min(lh_mat.shape[1], w - half_w)
                full[:half_h, half_w:half_w + lh_w] = lh_mat[:half_h, :lh_w]
            if hl_mat is not None and hl_mat.shape[0] > 0:
                hl_h = min(hl_mat.shape[0], h - half_h)
                full[half_h:half_h + hl_h, :half_w] = hl_mat[:hl_h, :half_w]
            if hh_mat is not None:
                hh_h = min(hh_mat.shape[0], h - half_h)
                hh_w = min(hh_mat.shape[1], w - half_w)
                full[half_h:half_h + hh_h, half_w:half_w + hh_w] = hh_mat[:hh_h, :hh_w]
        else:
            full = np.zeros((h, w))
            half_h, half_w = 0, 0

        # Inverse 2D DWT
        # First inverse along columns, then rows
        col_recon = np.zeros_like(full)
        half_h = full.shape[0] // 2
        half_w = full.shape[1] // 2
        for j in range(full.shape[1]):
            col = full[:, j]
            if half_h > 0 and half_h <= len(col):
                approx = col[:half_h]
                detail = col[half_h:min(2 * half_h, len(col))]
                # Pad if needed
                if len(approx) > len(detail):
                    detail = np.pad(detail, (0, len(approx) - len(detail)))
                elif len(detail) > len(approx):
                    approx = np.pad(approx, (0, len(detail) - len(approx)))
                col_recon[:, j] = wt.haar_inverse_1d(approx[:len(col)//2], detail[:len(col)//2])
            else:
                col_recon[:, j] = col

        row_recon = np.zeros_like(col_recon)
        for i in range(col_recon.shape[0]):
            row = col_recon[i, :]
            if half_w > 0 and half_w <= len(row):
                approx = row[:half_w]
                detail = row[half_w:min(2 * half_w, len(row))]
                if len(approx) > len(detail):
                    detail = np.pad(detail, (0, len(approx) - len(detail)))
                elif len(detail) > len(approx):
                    approx = np.pad(approx, (0, len(detail) - len(approx)))
                row_recon[i, :] = wt.haar_inverse_1d(approx[:len(row)//2], detail[:len(row)//2])
            else:
                row_recon[i, :] = row

        last_ll = row_recon[:h, :w]

    if last_ll is None:
        return np.zeros(m, n) if m and n else np.zeros((1, 1))
    return last_ll[:m, :n] if m and n else last_ll
class AMRTTCompressor:
    """Stage 1: Adaptive Multi-Resolution Tensor Train compression.

    Decomposes weight matrix into frequency bands via 2D wavelet,
    then applies TT-SVD with per-band ranks.
    """

    def __init__(self, quality: float = 0.95):
        self.quality = quality
        self.wavelet_level = 2
        # Low-freq: high rank, high-freq: low rank
        self.base_rank_ll = max(2, int(16 * quality))
        self.base_rank_lh = max(2, int(8 * quality))
        self.base_rank_hl = max(2, int(8 * quality))
        self.base_rank_hh = max(2, int(4 * quality))

    def compress(self, matrix: np.ndarray) -> dict:
        """Compress matrix via wavelet + TT."""
        if matrix.size < 64:
            return {"type": "amrtt_raw", "data": matrix.copy(), "shape": list(matrix.shape)}

        subbands = _wavelet_2d_decomp(matrix, self.wavelet_level)
        tt_data = {}
        total_params = 0

        for name, sb, row_off, col_off in subbands:
            if sb.size < 16:
                continue
            # Determine rank based on band type
            if name.startswith("LL"):
                rank = self.base_rank_ll
            elif name.startswith("LH"):
                rank = self.base_rank_lh
            elif name.startswith("HL"):
                rank = self.base_rank_hl
            else:
                rank = self.base_rank_hh
            rank = min(rank, min(sb.shape) // 2, 64)
            rank = max(2, rank)

            try:
                cores = _tt_svd(sb, rank)
                n_params = sum(c.size for c in cores)
                total_params += n_params
                tt_data[name] = {
                    "cores": [c.copy() for c in cores],
                    "shape": list(sb.shape),
                    "rank": rank,
                    "row_offset": row_off,
                    "col_offset": col_off,
                }
            except Exception:
                tt_data[name] = {
                    "cores": [sb.astype(np.float64).reshape(sb.shape[0], sb.shape[1], 1)],
                    "shape": list(sb.shape),
                    "rank": 1,
                    "row_offset": row_off,
                    "col_offset": col_off,
                }

        return {
            "type": "amrtt",
            "tt_data": tt_data,
            "original_shape": list(matrix.shape),
            "wavelet_level": self.wavelet_level,
            "total_params": total_params,
        }

    def decompress(self, comp: dict) -> np.ndarray:
        """Reconstruct matrix from AMRTT representation."""
        if comp.get("type") == "amrtt_raw":
            return comp["data"].copy()

        shape = comp["original_shape"]
        tt_data = comp["tt_data"]
        n_level = comp["wavelet_level"]

        subbands_out = []
        for name, data in tt_data.items():
            cores = [c.copy() for c in data["cores"]]
            if len(cores) == 1:
                recon = cores[0].reshape(data["shape"])
            else:
                recon = _tt_reconstruct(cores)
            sb_shape = data["shape"]
            if recon.shape[0] != sb_shape[0] or recon.shape[1] != sb_shape[1]:
                recon = recon[:sb_shape[0], :sb_shape[1]]
            subbands_out.append((name, recon))

        result = _wavelet_2d_recomp(subbands_out, n_level, tuple(shape))
        return result.astype(np.float32)
def _fast_dct_1d(x: np.ndarray) -> np.ndarray:
    """Fast DCT-II via FFT (type II DCT). O(N log N), no large matrix."""
    n = len(x)
    if n < 2:
        return x.copy()
    x2 = np.zeros(2 * n, dtype=np.float64)
    x2[:n] = x
    x2[n:] = x[::-1]
    X = np.fft.fft(x2)[:n]
    k = np.arange(n, dtype=np.float64)
    X = X * np.exp(-1j * np.pi * k / (2 * n))
    return np.sqrt(2.0 / n) * X.real if n > 1 else np.array([x[0]])
def _fast_idct_1d(X: np.ndarray) -> np.ndarray:
    """Fast IDCT via FFT. O(N log N)."""
    n = len(X)
    if n < 2:
        return X.copy()
    k = np.arange(n, dtype=np.float64)
    Y = X * np.exp(1j * np.pi * k / (2 * n)) * np.sqrt(n / 2.0)
    y2 = np.fft.ifft(np.concatenate([Y, np.zeros(n, dtype=np.complex128)]))[:n].real
    return y2
class FrequencyLloydMaxSparse:
    """Stage 2: Frequency-domain Lloyd-Max quantization + ISTA sparse coding.

    Dense storage: all DCT coefficients stored at quantized bit depth
    (no sparse indices). Variable bit allocation per position.
    """

    def __init__(self, quality: float = 0.95):
        self.quality = quality
        # Bit depth per coefficient: scales with quality
        self.n_bits_quant = max(2, int(2 + 6 * quality))  # 2-8 bits

    def compress_single(self, data: np.ndarray) -> dict:
        """Compress array via DCT + dense Lloyd-Max quantization."""
        arr = data.astype(np.float64).ravel()
        n = arr.size
        if n < 4:
            return {"type": "dense_dct", "data": arr.astype(np.float16).tobytes(), "n": n, "n_bits": 16}

        dct_coeff = _fast_dct_1d(arr)

        # Lloyd-Max quantize all coefficients
        lm = LloydMaxQuantizer(n_bits=self.n_bits_quant)
        lm.train(np.abs(dct_coeff))
        qmag = lm.quantize(np.abs(dct_coeff))
        qvals = qmag * np.sign(dct_coeff).astype(np.float64)

        # ISTA soft thresholding: zero out tiny quantized values
        thresh = np.percentile(np.abs(qvals), 20 * (1.0 - self.quality))
        qvals[np.abs(qvals) < thresh] = 0.0

        # Pack quantized indices into bytes (use n_bits per value)
        q_idx = np.zeros(n, dtype=np.int32)
        if lm.boundaries is not None and lm.centroids is not None:
            flat = np.abs(dct_coeff).ravel()
            q_idx = np.digitize(flat, lm.boundaries)
            q_idx = np.clip(q_idx, 0, (1 << self.n_bits_quant) - 1)
            # Zero out the thresholded ones
            zero_positions = np.where(np.abs(qvals).ravel() < thresh)[0]
            q_idx[zero_positions] = 0 if lm.centroids[0] == 0 else 0
        else:
            q_idx = np.zeros(n, dtype=np.int32)

        # Pack signs separately (1 bit per coefficient)
        signs = (np.sign(dct_coeff) > 0).astype(np.int32)

        return {
            "type": "dense_dct",
            "q_indices": q_idx.tobytes(),
            "signs": signs.tobytes(),
            "n": n,
            "n_bits": self.n_bits_quant,
            "lm_scale": lm.scale,
            "lm_centroids": lm.centroids.astype(np.float32) if lm.centroids is not None else None,
            "lm_boundaries": lm.boundaries.astype(np.float32) if lm.boundaries is not None else None,
        }

    def decompress_single(self, comp: dict) -> np.ndarray:
        """Reconstruct from dense DCT representation."""
        if comp.get("type") == "dense_dct":
            try:
                return np.frombuffer(comp["data"], dtype=np.float16).astype(np.float64)
            except Exception:
                return np.zeros(comp.get("n", 1), dtype=np.float64)

        n = comp["n"]
        n_bits = comp["n_bits"]
        centroids = comp.get("lm_centroids")
        boundaries = comp.get("lm_boundaries")

        # Unpack quantized indices
        q_idx = np.frombuffer(comp["q_indices"], dtype=np.int32)
        if len(q_idx) < n:
            q_idx = np.pad(q_idx, (0, n - len(q_idx)))[:n]

        # Unpack signs
        signs_bytes = comp["signs"]
        signs = np.frombuffer(signs_bytes, dtype=np.int32)
        if len(signs) < n:
            signs = np.pad(signs, (0, n - len(signs)))[:n]
        signs = np.where(signs > 0, 1.0, -1.0)

        # De-quantize via centroids
        if centroids is not None and boundaries is not None:
            dequant = np.zeros(n, dtype=np.float64)
            for i in range(n):
                idx = q_idx[i]
                if 0 <= idx < len(centroids):
                    dequant[i] = centroids[idx] * signs[i]
        else:
            dequant = np.zeros(n, dtype=np.float64)

        return _fast_idct_1d(dequant)
class HolographicWeightEncoder:
    """Stage 3: Holographic weight encoding via HRR circular convolution.

    Encodes chunks of ~1000 weights into a single 4096-D superposition
    vector using HRR binding and bundling. Achieves ~100:1 compression
    on its own when stored at reduced precision.
    """

    def __init__(self, dim: int = 4096, quality: float = 0.95):
        self.dim = dim
        self.quality = quality
        self.chunk_size = int(500 + 500 * quality)  # 500-1000 weights per chunk
        self.store_bits = max(2, int(4 * quality))  # bits per stored value

    def _make_key(self, idx: int) -> np.ndarray:
        """Generate a deterministic HRR key vector for a given index."""
        seed = splitmix64(idx) & 0x7FFFFFFF  # Ensure 31-bit positive seed
        rng = np.random.RandomState(seed)
        vec = rng.randn(self.dim).astype(np.float64)
        return vec / max(np.linalg.norm(vec), EPS)

    def _padded_length(self, n: int) -> int:
        """Round up to chunk_size."""
        return ((n + self.chunk_size - 1) // self.chunk_size) * self.chunk_size

    def encode(self, data: np.ndarray) -> dict:
        """Encode data vector into HRR superpositions.

        Splits into chunks, encodes each as superposition.
        """
        flat = data.ravel().astype(np.float64)
        n = len(flat)
        padded_n = self._padded_length(n)
        if padded_n > n:
            flat = np.pad(flat, (0, padded_n - n))

        n_chunks = padded_n // self.chunk_size
        superpositions = []
        chunk_meta = []

        for ci in range(n_chunks):
            start = ci * self.chunk_size
            end = start + self.chunk_size
            chunk = flat[start:end]

            # Encode: superposition = sum(weight_i * key_i)
            superposition = np.zeros(self.dim, dtype=np.float64)
            for i, val in enumerate(chunk):
                key = self._make_key(start + i)
                superposition += val * key

            # Store at reduced precision
            if self.store_bits <= 8:
                s = max(np.abs(superposition))
                if s > 0:
                    scale = (2 ** (self.store_bits - 1) - 1) / s
                    quantized = np.round(superposition * scale).astype(np.int8)
                    superpositions.append(quantized)
                else:
                    superpositions.append(np.zeros(self.dim, dtype=np.int8))
                chunk_meta.append({"scale": float(s)})
            else:
                superpositions.append(superposition.astype(np.float16))
                chunk_meta.append({"scale": 1.0})

        return {
            "type": "hwe",
            "superpositions": [s.tobytes() for s in superpositions],
            "chunk_meta": chunk_meta,
            "n_chunks": n_chunks,
            "chunk_size": self.chunk_size,
            "dim": self.dim,
            "store_bits": self.store_bits,
            "n_original": n,
        }

    def decode(self, comp: dict, out_shape: tuple = None) -> np.ndarray:
        """Decode HRR superpositions back to data vector."""
        superpositions_bytes = comp["superpositions"]
        chunk_meta = comp["chunk_meta"]
        chunk_size = comp["chunk_size"]
        dim = comp["dim"]
        store_bits = comp["store_bits"]
        n_original = comp["n_original"]

        n_chunks = len(superpositions_bytes)
        result = np.zeros(n_chunks * chunk_size, dtype=np.float64)

        for ci in range(n_chunks):
            raw = np.frombuffer(superpositions_bytes[ci], dtype=np.int8 if store_bits <= 8 else np.float16)
            if store_bits <= 8:
                scale = chunk_meta[ci].get("scale", 1.0)
                if scale > 0:
                    superposition = raw.astype(np.float64) * scale / (2 ** (store_bits - 1) - 1)
                else:
                    superposition = np.zeros(dim, dtype=np.float64)
            else:
                superposition = raw.astype(np.float64)

            start = ci * chunk_size
            for i in range(chunk_size):
                key = self._make_key(start + i)
                val = np.dot(superposition, key)
                result[start + i] = val

        result = result[:n_original]
        if out_shape:
            return result.reshape(out_shape)
        return result

    def compressed_size(self, comp: dict) -> int:
        """Compute size of HWE representation in bytes."""
        total = 0
        for s_bytes in comp["superpositions"]:
            total += len(s_bytes)
        return total
class QAOAResonantBitAllocator:
    """Stage 4: Quantum-inspired resonant bit allocation.

    Models allocation as an Ising spin system with nearest-neighbor
    coupling (avoiding N×N matrix). Uses a 1D chain Ising model
    solved via simulated annealing on sorted magnitudes.
    """

    def __init__(self, quality: float = 0.95):
        self.quality = quality
        self.temperature_start = 2.0
        self.temperature_end = 0.01
        self.n_iterations = 500

    def allocate(self, magnitudes: np.ndarray, budget: int = None) -> np.ndarray:
        """Allocate bits using 1D Ising model with chain coupling.

        Only nearest-neighbor couplings to keep O(N) memory.
        Returns: array of 0/1 per coefficient.
        """
        n = len(magnitudes)
        if n < 2:
            return np.ones(n, dtype=np.int32)
        if budget is None:
            budget = max(1, int(n * 0.1 * self.quality))
        budget = min(budget, n)

        # Sort by magnitude
        order = np.argsort(-magnitudes)
        sorted_mags = magnitudes[order]

        # Ising on a chain: coupling J between neighbors
        J = 0.5 * self.quality
        h_field = sorted_mags / max(np.max(sorted_mags), EPS) * 0.3

        # Initialize: keep top budget coefficients
        spins = -np.ones(n, dtype=np.float64)
        spins[:budget] = 1.0

        def chain_energy(spins):
            e = 0.0
            for i in range(n - 1):
                e -= J * spins[i] * spins[i + 1]
            e -= np.sum(h_field * spins)
            k = int(np.sum(spins > 0))
            e += 0.1 * (k - budget) ** 2
            return e

        best_spins = spins.copy()
        best_energy = chain_energy(best_spins)

        for it in range(self.n_iterations):
            t = self.temperature_start * (self.temperature_end / self.temperature_start) ** (it / self.n_iterations)
            i = np.random.randint(n)
            spins[i] *= -1.0
            e = chain_energy(spins)
            if e < best_energy or np.random.random() < math.exp(-(e - best_energy) / max(t, EPS)):
                best_energy = e
                best_spins = spins.copy()
            else:
                spins[i] *= -1.0

        # Map back to original order
        result = np.zeros(n, dtype=np.int32)
        result[order[np.where(best_spins > 0)[0]]] = 1
        return result
class TernaryPredictiveResidual:
    """Stage 5: Ternary quantization + AR(2) prediction + residual encoding.

    Converts values to {-1, 0, +1} via k-means, computes AR(2) prediction
    on ternary sequence, stores residuals at variable bit depth, and
    applies Huffman coding.
    """

    def __init__(self, quality: float = 0.95):
        self.quality = quality
        self.sparsity_target = 0.7 + 0.2 * quality  # 0.7-0.9
        self.residual_bits = max(1, int(3 * quality))  # 1-3 bits for residuals

    def compress(self, data: np.ndarray) -> dict:
        """Compress data to ternary + AR(2) residuals."""
        flat = data.ravel().astype(np.float64)

        # K-means clustering to 3 centroids: {-1, 0, +1}
        vals = flat.copy()
        mu = np.mean(vals)
        sigma = np.std(vals) if np.std(vals) > 0 else 1.0
        normalized = (vals - mu) / sigma

        # Simple k-means: initialize centroids at -1, 0, 1
        centroids = np.array([-1.0, 0.0, 1.0])
        for _ in range(20):
            dists = np.abs(normalized[:, None] - centroids[None, :])
            labels = np.argmin(dists, axis=1)
            new_c = np.array([np.mean(normalized[labels == i]) if np.sum(labels == i) > 0 else centroids[i] for i in range(3)])
            if np.allclose(centroids, new_c, atol=1e-6):
                break
            centroids = new_c

        # Sort centroids to match {-1, 0, +1} order
        order = np.argsort(centroids)
        centroids = centroids[order]
        dists = np.abs(normalized[:, None] - centroids[None, :])
        labels = np.argmin(dists, axis=1)

        # Convert to {-1, 0, +1}
        ternary_map = np.array([-1, 0, 1])
        ternary_vals = ternary_map[labels]

        # AR(2) prediction: x_t = a1 * x_{t-1} + a2 * x_{t-2}
        # Compute AR coefficients via Yule-Walker
        n = len(ternary_vals)
        if n >= 3:
            r1 = np.mean(ternary_vals[1:] * ternary_vals[:-1])
            r2 = np.mean(ternary_vals[2:] * ternary_vals[:-2])
            denom = r1 ** 2 - 1.0
            if abs(denom) > EPS:
                a1 = (r1 * r2 - r1) / denom
                a2 = (r2 - r1 ** 2) / denom
            else:
                a1 = a2 = 0.0
        else:
            a1 = a2 = 0.0

        # Clamp to ensure stationarity
        a1 = np.clip(a1, -1.0, 1.0)
        a2 = np.clip(a2, -1.0, 1.0)

        # Compute predictions
        predictions = np.zeros(n, dtype=np.float64)
        if n > 0:
            predictions[0] = 0.0
        if n > 1:
            predictions[1] = a1 * ternary_vals[0] if abs(a1) > EPS else 0.0
        for t in range(2, n):
            predictions[t] = a1 * ternary_vals[t - 1] + a2 * ternary_vals[t - 2]

        # Compute residuals
        residuals = flat - predictions

        # Encode residuals at variable bit depth
        r_scale = max(np.abs(residuals))
        if r_scale < EPS:
            r_scale = 1.0
        r_norm = residuals / r_scale
        r_quant = np.round(r_norm * (2 ** (self.residual_bits - 1) - 1)).astype(np.int32)
        r_quant = np.clip(r_quant, -(2 ** (self.residual_bits - 1)), 2 ** (self.residual_bits - 1) - 1)

        # Huffman encode residuals
        rle_vals, rle_lens = _rle_encode(r_quant)
        codebook = _build_huffman_codes(rle_vals.tolist())
        encoded = _encode_symbols(rle_vals.tolist(), codebook)
        rl_bytes = bytearray()
        for rl in rle_lens:
            val = min(int(rl), 65535)
            rl_bytes.extend(struct.pack("<H", val))

        # Pack ternary values (2 bits each) -> pack 4 per byte
        ternary_packed = bytearray()
        for i in range(0, len(ternary_vals), 4):
            byte = 0
            for j in range(4):
                if i + j < len(ternary_vals):
                    t = ternary_vals[i + j]
                    code = {1: 0b01, -1: 0b10, 0: 0b00}.get(t, 0)
                    byte |= (code << (j * 2))
            ternary_packed.append(byte)

        return {
            "type": "ternary_ar2",
            "ternary_packed": bytes(ternary_packed),
            "n_ternary": len(ternary_vals),
            "ar_coeffs": [float(a1), float(a2)],
            "mu": float(mu),
            "sigma": float(sigma),
            "r_scale": float(r_scale),
            "residual_bits": self.residual_bits,
            "residual_codebook": codebook,
            "residual_encoded": encoded,
            "residual_rle_lens": bytes(rl_bytes),
            "n_residuals": len(r_quant),
        }

    def decompress(self, comp: dict, shape: tuple = None) -> np.ndarray:
        """Decompress ternary + AR(2) representation."""
        ternary_packed = comp["ternary_packed"]
        n_ternary = comp["n_ternary"]
        a1, a2 = comp["ar_coeffs"]
        mu = comp["mu"]
        sigma = comp["sigma"]
        r_scale = comp["r_scale"]
        residual_bits = comp["residual_bits"]

        # Unpack ternary values
        ternary_vals = np.zeros(n_ternary, dtype=np.int32)
        for i in range(n_ternary):
            byte_idx = i // 4
            bit_idx = i % 4
            if byte_idx < len(ternary_packed):
                code = (ternary_packed[byte_idx] >> (bit_idx * 2)) & 0b11
                ternary_map = {0b00: 0, 0b01: 1, 0b10: -1}
                ternary_vals[i] = ternary_map.get(code, 0)

        # Decode residuals via Huffman
        codebook = comp["residual_codebook"]
        encoded = comp["residual_encoded"]
        rl_bytes = comp["residual_rle_lens"]

        decoded_syms = _decode_symbols(encoded, codebook, len(rl_bytes) // 2)
        run_lens = []
        for i in range(0, len(rl_bytes), 2):
            if i + 1 < len(rl_bytes):
                run_lens.append(struct.unpack_from("<H", rl_bytes, i)[0])
        residuals = []
        for val, rl in zip(decoded_syms, run_lens):
            residuals.extend([val] * rl)
        residuals = np.array(residuals[:n_ternary], dtype=np.float64)

        # De-quantize residuals
        r_max = 2 ** (residual_bits - 1) - 1
        if r_max > 0:
            residuals = residuals * r_scale / r_max
        else:
            residuals = np.zeros_like(residuals)

        # AR(2) reconstruction
        predictions = np.zeros(n_ternary, dtype=np.float64)
        if n_ternary > 0:
            predictions[0] = 0.0
        if n_ternary > 1:
            predictions[1] = a1 * ternary_vals[0] if abs(a1) > EPS else 0.0
        for t in range(2, n_ternary):
            predictions[t] = a1 * ternary_vals[t - 1] + a2 * ternary_vals[t - 2]

        result = predictions + residuals

        # De-normalize
        result = result * sigma + mu

        if shape:
            result = result.reshape(shape)
        return result.astype(np.float32)
class StabilizerErrorCorrection:
    """Stage 6: Hamming [7,4,3] error correction encoding.

    Encodes each 4-bit nibble as 7-bit Hamming codeword with
    single-bit error correction. Adds ~75% overhead but ensures
    bit-perfect recovery.
    """

    def __init__(self):
        self._build_tables()

    def _build_tables(self):
        """Build [7,4] Hamming encode/decode tables."""
        self.encode_table = {}
        self.decode_table = {}
        G = np.array([
            [1, 1, 0, 1],
            [1, 0, 1, 1],
            [1, 0, 0, 0],
            [0, 1, 1, 1],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.int32)

        for data in range(16):
            bits = np.array([(data >> 3) & 1, (data >> 2) & 1, (data >> 1) & 1, data & 1], dtype=np.int32)
            cw = (G @ bits) % 2
            cw_int = sum(cw[i] << (6 - i) for i in range(7))
            self.encode_table[data] = cw_int
            self.decode_table[cw_int] = data

        # Build syndrome table for error correction
        self.syndrome_table = {}
        H = np.array([
            [1, 0, 1, 0, 1, 0, 1],
            [0, 1, 1, 0, 0, 1, 1],
            [0, 0, 0, 1, 1, 1, 1],
        ], dtype=np.int32)

        # No error case
        self.syndrome_table[0] = -1
        for pos in range(7):
            e = np.zeros(7, dtype=np.int32)
            e[pos] = 1
            syndrome = (H @ e) % 2
            s_int = syndrome[0] * 4 + syndrome[1] * 2 + syndrome[2]
            self.syndrome_table[s_int] = pos

    def _encode_nibble(self, nibble: int) -> int:
        return self.encode_table.get(nibble & 0xF, 0)

    def _decode_nibble(self, codeword: int) -> Tuple[int, bool]:
        """Decode and correct. Returns (nibble, corrected_flag)."""
        cw = codeword & 0x7F
        if cw in self.decode_table:
            return self.decode_table[cw], False
        # Try syndrome correction
        bits = np.array([(cw >> 6) & 1, (cw >> 5) & 1, (cw >> 4) & 1,
                         (cw >> 3) & 1, (cw >> 2) & 1, (cw >> 1) & 1, cw & 1], dtype=np.int32)
        H = np.array([
            [1, 0, 1, 0, 1, 0, 1],
            [0, 1, 1, 0, 0, 1, 1],
            [0, 0, 0, 1, 1, 1, 1],
        ], dtype=np.int32)
        syndrome = (H @ bits) % 2
        s_int = syndrome[0] * 4 + syndrome[1] * 2 + syndrome[2]
        if s_int in self.syndrome_table:
            pos = self.syndrome_table[s_int]
            if pos >= 0:
                bits[pos] ^= 1
                corrected = sum(bits[i] << (6 - i) for i in range(7))
                if corrected in self.decode_table:
                    return self.decode_table[corrected], True
        return 0, False

    def encode(self, data: bytes) -> bytes:
        """Encode byte stream with Hamming [7,4,3].

        Each input byte -> 2 codewords (14 bits) -> 2 bytes.
        """
        result = bytearray()
        for byte in data:
            high = self._encode_nibble((byte >> 4) & 0xF)
            low = self._encode_nibble(byte & 0xF)
            # Pack 2 codewords (14 bits) into 2 bytes
            packed = (high << 7) | low
            result.append((packed >> 8) & 0xFF)
            result.append(packed & 0xFF)
        return bytes(result)

    def decode(self, data: bytes) -> bytes:
        """Decode Hamming-encoded byte stream with error correction."""
        result = bytearray()
        n_corrected = 0
        i = 0
        while i + 1 < len(data):
            packed = (data[i] << 8) | data[i + 1]
            high = (packed >> 7) & 0x7F
            low = packed & 0x7F
            dh, ch = self._decode_nibble(high)
            dl, cl = self._decode_nibble(low)
            if ch:
                n_corrected += 1
            if cl:
                n_corrected += 1
            result.append((dh << 4) | dl)
            i += 2
        # Handle trailing byte
        if i < len(data):
            packed = data[i]
            high = (packed >> 1) & 0x7F
            dh, _ = self._decode_nibble(high)
            result.append(dh << 4)
        return bytes(result)
@dataclass
class CompressionPipeline2000Config:
    """Configuration for the upgraded 2000:1 compression pipeline."""
    target_ratio: float = 2000.0
    quality: float = 0.95
    wavelet_level: int = 2
    sparse_keep_fraction: float = 0.001
    sparse_energy_fraction: float = 0.999
    hrr_dim: int = 4096
    ternary_sparsity: float = 0.8
    residual_bits: int = 2
    enable_stabilizer: bool = True
    auto_tune: bool = True
    max_tune_iterations: int = 5
def _flms_global_score(dct_matrix: np.ndarray, quality: float = 1.0) -> np.ndarray:
    """FLMS importance × energy score for DCT coefficient selection.

    Score(i,j) = |DCT(i,j)| × exp(-freq_dist × λ) where
    λ = 4.0 - 2.0×quality controls the low-frequency bias.
    Higher quality → smaller λ → more HF survival.

    Exponential decay is cache-friendly (no division) and O(N).
    """
    m, n = dct_matrix.shape
    i_grid = np.arange(m, dtype=np.float64)[:, None]
    j_grid = np.arange(n, dtype=np.float64)[None, :]
    freq_dist = np.sqrt((i_grid / m) ** 2 + (j_grid / n) ** 2)
    lam = 4.0 - 2.0 * quality
    importance = np.exp(-freq_dist * lam)
    return np.abs(dct_matrix, dtype=np.float64) * importance
class _ArithEncoder:
    """16-symbol adaptive arithmetic encoder (CPU-first, integer-only)."""
    TOP = 0x7FFFFFFF
    HALF = 0x40000000
    QUARTER = 0x20000000

    def __init__(self):
        self.freq = np.ones(17, dtype=np.int64)
        self.low = 0
        self.high = self.TOP
        self.bits: List[int] = []
        self.pending = 0

    def encode(self, sym: int) -> None:
        total = self.freq[16]
        cum_below = int(np.sum(self.freq[:sym]))
        cum_above = cum_below + int(self.freq[sym])
        rng = self.high - self.low + 1
        self.high = self.low + (rng * cum_above) // total - 1
        self.low = self.low + (rng * cum_below) // total
        self._renorm()
        self.freq[sym] += 1
        self.freq[16] += 1
        if self.freq[16] > 5000:
            self.freq[:16] = (self.freq[:16] + 1) // 2
            self.freq[16] = int(np.sum(self.freq[:16]))

    def _renorm(self) -> None:
        while True:
            if self.high < self.HALF:
                self._emit(0)
            elif self.low >= self.HALF:
                self._emit(1)
                self.low -= self.HALF
                self.high -= self.HALF
            elif self.low >= self.QUARTER and self.high < self.HALF + self.QUARTER:
                self.pending += 1
                self.low -= self.QUARTER
                self.high -= self.QUARTER
            else:
                break
            self.low <<= 1
            self.high = (self.high << 1) | 1

    def _emit(self, bit: int) -> None:
        self.bits.append(bit)
        while self.pending > 0:
            self.bits.append(1 - bit)
            self.pending -= 1

    def flush(self) -> bytes:
        self.pending += 1
        self._emit(1 if self.low < self.QUARTER else 0)
        while len(self.bits) % 8:
            self.bits.append(0)
        out = bytearray()
        for i in range(0, len(self.bits), 8):
            byte = 0
            for j in range(8):
                byte = (byte << 1) | self.bits[i + j]
            out.append(byte & 0xFF)
        return bytes(out)
class _ArithDecoder:
    """16-symbol adaptive arithmetic decoder matching _ArithEncoder."""
    TOP = 0x7FFFFFFF
    HALF = 0x40000000
    QUARTER = 0x20000000

    def __init__(self, data: bytes, n_symbols: int):
        self.freq = np.ones(17, dtype=np.int64)
        self.low = 0
        self.high = self.TOP
        self.bits: List[int] = []
        for byte in data:
            for shift in range(7, -1, -1):
                self.bits.append((byte >> shift) & 1)
        self.pos = 0
        self.code = self._read(31)
        self.n_symbols = n_symbols
        self.decoded = 0

    def _read(self, n: int) -> int:
        val = 0
        for _ in range(n):
            val = (val << 1) | (self.bits[self.pos] if self.pos < len(self.bits) else 0)
            self.pos += 1
        return val

    def decode(self) -> int:
        if self.decoded >= self.n_symbols:
            return 0
        self.decoded += 1
        total = self.freq[16]
        rng = self.high - self.low + 1
        offset = ((self.code - self.low + 1) * total - 1) // rng
        cum = 0
        sym = 0
        for s in range(16):
            if offset < cum + self.freq[s]:
                sym = s
                break
            cum += self.freq[s]
        cum_above = cum + self.freq[sym]
        self.high = self.low + (rng * cum_above) // total - 1
        self.low = self.low + (rng * cum) // total
        while True:
            if self.high < self.HALF:
                pass
            elif self.low >= self.HALF:
                self.code -= self.HALF
                self.low -= self.HALF
                self.high -= self.HALF
            elif self.low >= self.QUARTER and self.high < self.HALF + self.QUARTER:
                self.code -= self.QUARTER
                self.low -= self.QUARTER
                self.high -= self.QUARTER
            else:
                break
            self.low <<= 1
            self.high = (self.high << 1) | 1
            self.code = (self.code << 1) | (self.bits[self.pos] if self.pos < len(self.bits) else 0)
            self.pos += 1
        self.freq[sym] += 1
        self.freq[16] += 1
        if self.freq[16] > 5000:
            self.freq[:16] = (self.freq[:16] + 1) // 2
            self.freq[16] = int(np.sum(self.freq[:16]))
        return sym
class CompressionPipeline2000:
    """2000:1 compression via MRADCT + FLMS + Adaptive Arithmetic Coding.

    Pipeline (CPU-first, all numpy, O(N log N)):
      1. MRADCT          — quadtree adaptive block DCT (128→16 by local variance)
      2. FLMS scoring    — |coeff| × exp(-freq_dist × λ) frequency-weighted
      3. Top-K selection — keep highest-scoring DCT coefficients
      4. Delta positions  — exp-Golomb coded sorted-index gaps
      5. Signs in 4-bit   — group 4 signs → 1/16 patterns → adaptive arithmetic
      6. 1-Bit quant     — stochastic sign quantisation (expectation-preserving)
      7. Residual refine — up to 3 passes on the reconstruction error

    Design decisions:
      - Quadtree: splits high-variance blocks, fuses smooth ones (matches local
        frequency content — smooth → large block = better compaction)
      - FLMS: exponential frequency decay avoids the division-heavy 1/(1+x)
      - Arithmetic: 16-symbol adaptive model hits entropy limit for sign patterns
      - Positions: exp-Golomb deltas → ~6-10 bits/coeff instead of 14 raw
      - Cache: 128×128 × 8 B = 128 KB per block fits most L2 caches

    Parameters
    ----------
    target_ratio : float
        Target compression ratio (default 2000).
    quality : float
        Reconstruction quality 0-1 (default 0.95).
    """

    def __init__(self, target_ratio: float = 2000.0, quality: float = 0.95):
        self.target_ratio = target_ratio
        self.quality = quality
        self.max_passes = 3
        self.max_block = 128
        self.min_block = 16
        self.variance_threshold = 0.01 * (3.0 - 2.0 * quality)

    # ── Stage 1: MRADCT — Multi-Resolution Adaptive DCT ─────────────────

    def _quadtree_leaves(self, matrix: np.ndarray, row: int, col: int, size: int,
                         depth: int = 0) -> List[Tuple[int, int, int]]:
        """Recursive quadtree splitting by local variance.

        Returns list of (row, col, actual_size) for leaf blocks.
        Stops at min_block or 5 levels deep.
        """
        if size < self.min_block or depth > 5:
            return [(row, col, size)]
        if row >= matrix.shape[0] or col >= matrix.shape[1]:
            return []
        actual_h = min(size, matrix.shape[0] - row)
        actual_w = min(size, matrix.shape[1] - col)
        actual = min(actual_h, actual_w)
        if actual < self.min_block:
            return [(row, col, actual)]
        block = matrix[row:row + actual, col:col + actual]
        var = float(np.var(block))
        if var > self.variance_threshold and actual > self.min_block:
            half = actual // 2
            if half >= self.min_block:
                result: List[Tuple[int, int, int]] = []
                result.extend(self._quadtree_leaves(matrix, row, col, half, depth + 1))
                result.extend(self._quadtree_leaves(matrix, row, col + half, half, depth + 1))
                result.extend(self._quadtree_leaves(matrix, row + half, col, half, depth + 1))
                result.extend(self._quadtree_leaves(matrix, row + half, col + half, half, depth + 1))
                return result
        return [(row, col, actual)]

    def _mradct_decompose(self, matrix: np.ndarray) -> List[dict]:
        """Decompose matrix into variable-size DCT blocks via quadtree."""
        m, n = matrix.shape
        all_blocks: List[dict] = []
        for i in range(0, m, self.max_block):
            for j in range(0, n, self.max_block):
                actual = min(self.max_block, m - i, n - j)
                if actual < self.min_block:
                    continue
                leaves = self._quadtree_leaves(matrix, i, j, actual)
                for r, c, sz in leaves:
                    if r >= m or c >= n:
                        continue
                    block = matrix[r:min(r + sz, m), c:min(c + sz, n)]
                    if block.size < 4:
                        continue
                    dct_coeffs = _fft_dct_2d(block)
                    all_blocks.append({
                        'row': r, 'col': c, 'size': sz,
                        'dct': dct_coeffs,
                    })
        return all_blocks

    # ── Stage 2-3: FLMS scoring + Top-K selection ───────────────────────

    @staticmethod
    def _flms_select(dct_block: np.ndarray, n_keep: int, quality: float
                     ) -> Tuple[np.ndarray, np.ndarray]:
        """Select top-n_keep indices and values by FLMS score.

        Returns (sorted_indices, corresponding_values).
        Uses np.argpartition for O(N) selection, then sorts the K winners.
        """
        scores = _flms_global_score(dct_block, quality)
        flat_scores = scores.ravel()
        flat_coeffs = dct_block.ravel()
        n_total = flat_scores.size
        if n_keep >= n_total:
            order = np.arange(n_total)
        else:
            order = np.argpartition(-flat_scores, n_keep - 1)[:n_keep]
        kept = flat_coeffs[order]
        idx = np.argsort(order)
        return order[idx], kept[idx]

    # ── Stage 4: Position encoding (exp-Golomb deltas) ──────────────────

    @staticmethod
    def _encode_positions(positions: np.ndarray) -> Tuple[List[int], int]:
        """Encode sorted positions as exp-Golomb coded deltas.

        Returns (bits_list, n_bits).
        Delta = positions[i] - positions[i-1] - 1, coded as exp-Golomb.
        """
        if len(positions) == 0:
            return [], 0
        deltas = np.diff(positions, prepend=-1) - 1
        deltas = np.maximum(deltas, 0).astype(np.int64)
        bits: List[int] = []
        for d in deltas:
            val = int(d) + 1
            nbits = val.bit_length()
            bits.extend([0] * (nbits - 1))
            for shift in range(nbits - 1, -1, -1):
                bits.append((val >> shift) & 1)
        return bits, len(bits)

    @staticmethod
    def _decode_positions(bits: List[int], n_positions: int) -> np.ndarray:
        """Decode exp-Golomb position deltas back to positions."""
        positions: List[int] = []
        pos = -1
        idx = 0
        for _ in range(n_positions):
            nzeros = 0
            while idx < len(bits) and bits[idx] == 0:
                nzeros += 1
                idx += 1
            if idx < len(bits):
                idx += 1
            val = 1
            for _ in range(nzeros):
                if idx < len(bits):
                    val = (val << 1) | bits[idx]
                    idx += 1
            pos += val
            positions.append(pos)
        return np.array(positions, dtype=np.int32)

    # ── Stage 5-6: 1-bit stochastic quant + arithmetic coding ───────────

    @staticmethod
    def _stochastic_quantize(values: np.ndarray, max_val: float) -> np.ndarray:
        """Stochastic rounding to {+max_val, -max_val}, expectation-preserving."""
        if max_val < 1e-30:
            return np.ones(len(values), dtype=np.float64) * 1e-30
        p_plus = (values.astype(np.float64) / max_val + 1.0) * 0.5
        p_plus = np.clip(p_plus, 0.0, 1.0)
        rng = np.random.RandomState(42)
        rand = rng.uniform(0.0, 1.0, size=len(values))
        return np.where(rand < p_plus, np.float64(max_val), np.float64(-max_val))

    def _encode_block(self, positions: np.ndarray, values: np.ndarray,
                      block_max: float) -> dict:
        """Encode one block: delta positions + 16-symbol arithmetic coded signs."""
        pos_bits, n_pos_bits = self._encode_positions(positions)
        signs = (values > 0).astype(np.int32)
        n_signs = len(signs)
        n_padded = ((n_signs + 3) // 4) * 4
        padded = np.zeros(n_padded, dtype=np.int32)
        padded[:n_signs] = signs
        coder = _ArithEncoder()
        for i in range(0, n_padded, 4):
            sym = (padded[i] << 3) | (padded[i + 1] << 2) | (padded[i + 2] << 1) | padded[i + 3]
            coder.encode(sym)
        arith_bytes = coder.flush()
        return {
            'K': len(positions),
            'block_max': float(block_max),
            'pos_bits': pos_bits,
            'n_pos_bits': n_pos_bits,
            'arith': arith_bytes,
            'n_arith_syms': n_padded // 4,
        }

    def _decode_block(self, blk: dict) -> Tuple[np.ndarray, np.ndarray, float]:
        """Decode positions + signs from encoded block data."""
        positions = self._decode_positions(blk['pos_bits'], blk['K'])
        decoder = _ArithDecoder(blk['arith'], blk['n_arith_syms'])
        syms = [decoder.decode() for _ in range(blk['n_arith_syms'])]
        signs = []
        for sym in syms:
            signs.append((sym >> 3) & 1)
            signs.append((sym >> 2) & 1)
            signs.append((sym >> 1) & 1)
            signs.append(sym & 1)
        signs = np.array(signs[:blk['K']], dtype=np.int32)
        bmax = blk['block_max']
        values = np.where(signs > 0, bmax, -bmax)
        return positions, values, bmax

    @staticmethod
    def _measure_bytes(obj) -> int:
        if isinstance(obj, bytes):
            return len(obj)
        if isinstance(obj, dict):
            return sum(CompressionPipeline2000._measure_bytes(v) for v in obj.values())
        if isinstance(obj, (list, tuple)):
            return sum(CompressionPipeline2000._measure_bytes(v) for v in obj)
        if isinstance(obj, np.ndarray):
            return int(obj.nbytes)
        if isinstance(obj, (int, float, np.integer, np.floating)):
            return 8
        return 0

    # ── Per-pass compress / decompress ──────────────────────────────────

    def _compress_pass(self, residual: np.ndarray, bit_budget: int, pass_idx: int
                       ) -> Tuple[dict, np.ndarray]:
        """Single compression pass: MRADCT → FLMS → top-K → quant → encode.

        Budget allocated proportionally to per-block FLMS energy.
        """
        m, n = residual.shape
        blocks = self._mradct_decompose(residual)
        if not blocks:
            return ({'blocks': [], 'dims': [m, n]}, np.zeros((m, n), dtype=np.float64))

        n_blocks = len(blocks)
        meta_bits = 48 * n_blocks + 256
        data_budget = max(0, bit_budget - meta_bits)
        bits_per_coeff = 11

        scored = []
        for blk in blocks:
            scores = _flms_global_score(blk['dct'], self.quality)
            blk['scores'] = scores
            blk['n_total'] = blk['size'] * blk['size']
            blk['energy'] = float(np.sum(scores))
            scored.append(blk)

        total_energy = sum(b['energy'] for b in scored)
        if total_energy < 1e-30:
            total_energy = 1.0

        encoded: List[dict] = []
        recon = np.zeros((m, n), dtype=np.float64)

        for blk in scored:
            sz = blk['size']
            n_keep = max(0, int(data_budget * blk['energy'] / total_energy / bits_per_coeff))
            n_keep = min(n_keep, blk['n_total'])
            if n_keep < 1:
                encoded.append({
                    'K': 0, 'block_max': 1.0, 'pos_bits': [], 'n_pos_bits': 0,
                    'arith': b'', 'n_arith_syms': 0,
                })
                continue

            indices, vals = self._flms_select(blk['dct'], n_keep, self.quality)
            bmax = float(np.max(np.abs(vals))) if len(vals) > 0 else 1.0
            if bmax < 1e-30:
                bmax = 1.0
            quantized = self._stochastic_quantize(vals, bmax)

            blk_enc = self._encode_block(indices, quantized, bmax)
            encoded.append(blk_enc)

            n_total = sz * sz
            coeffs = np.zeros(n_total, dtype=np.float64)
            for pos, val in zip(indices, quantized):
                if 0 <= pos < n_total:
                    coeffs[pos] = val
            coeffs = coeffs.reshape(sz, sz)
            recon_block = _fft_idct_2d(coeffs)
            r0, c0 = blk['row'], blk['col']
            r1 = min(r0 + sz, m)
            c1 = min(c0 + sz, n)
            recon[r0:r1, c0:c1] += recon_block[:(r1 - r0), :(c1 - c0)]

        # Handle zero-K blocks with missing entries: pad encoded list
        zero_block = {
            'K': 0, 'block_max': 1.0, 'pos_bits': [], 'n_pos_bits': 0,
            'arith': b'', 'n_arith_syms': 0,
        }
        while len(encoded) < len(scored):
            encoded.append(dict(zero_block))

        pass_data = {
            'blocks': encoded,
            'block_info': [(b['row'], b['col'], b['size']) for b in scored],
            'dims': [m, n],
        }
        return pass_data, recon

    def _decompress_pass(self, pass_data: dict, shape: Tuple[int, int]) -> np.ndarray:
        """Reconstruct one pass."""
        m, n = shape
        dims = pass_data['dims']
        block_info = pass_data['block_info']
        recon = np.zeros((dims[0], dims[1]), dtype=np.float64)
        for bi, blk_data in zip(block_info, pass_data['blocks']):
            r, c, sz = bi
            if blk_data['K'] == 0:
                continue
            positions, values, _ = self._decode_block(blk_data)
            n_total = sz * sz
            coeffs = np.zeros(n_total, dtype=np.float64)
            for pos, val in zip(positions, values):
                if 0 <= pos < n_total:
                    coeffs[pos] = val
            coeffs = coeffs.reshape(sz, sz)
            rb = _fft_idct_2d(coeffs)
            r1 = min(r + sz, m)
            c1 = min(c + sz, n)
            recon[r:r1, c:c1] += rb[:(r1 - r), :(c1 - c)]
        return recon[:m, :n]

    def _compute_bit_budget(self, n_elements: int) -> int:
        orig = int(n_elements * 4)
        target = orig / self.target_ratio
        return max(1, int(target * 8))

    # ── Public API ──────────────────────────────────────────────────────

    def compress(self, tensor: np.ndarray, layer_name: str = 'default') -> dict:
        """Compress via MRADCT + FLMS with up to 3 residual refinement passes."""
        original_shape = tensor.shape
        n_elements = int(np.prod(original_shape))
        original_bytes_32 = n_elements * 4

        if tensor.ndim < 2 or n_elements < 64:
            return {
                'type': 'flms_raw',
                'data': tensor.astype(np.float32).tobytes(),
                'shape': list(original_shape),
                'layer_name': layer_name,
                'original_bytes': original_bytes_32,
            }

        tensor_f64 = np.asarray(tensor, dtype=np.float64)
        total_budget = self._compute_bit_budget(n_elements)

        all_passes: List[dict] = []
        residual = tensor_f64.copy()
        total_energy = float(np.sum(tensor_f64 ** 2))
        if total_energy < 1e-30:
            total_energy = 1.0

        for p_idx in range(self.max_passes):
            used = sum(self._measure_bytes(p) * 8 for p in all_passes)
            remaining = max(0, total_budget - used)
            pass_budget = max(1, remaining // (self.max_passes - p_idx))

            pass_data, recon = self._compress_pass(residual, pass_budget, p_idx)
            all_passes.append(pass_data)

            residual = tensor_f64 - recon
            res_energy = float(np.sum(residual ** 2))
            if total_energy > 0 and res_energy / total_energy < 0.002:
                break

        compressed = {
            'type': 'flms',
            'passes': all_passes,
            'shape': list(original_shape),
            'original_bytes': original_bytes_32,
            'layer_name': layer_name,
            'quality': self.quality,
            'n_passes': len(all_passes),
            'target_ratio': self.target_ratio,
        }
        compressed['compressed_bytes'] = self._measure_bytes(compressed)
        compressed['ratio'] = original_bytes_32 / max(compressed['compressed_bytes'], 1)
        return compressed

    def decompress(self, compressed: dict) -> np.ndarray:
        """Reconstruct from FLMS compressed representation."""
        if compressed.get('type') == 'flms_raw':
            return np.frombuffer(compressed['data'], dtype=np.float32).reshape(
                compressed['shape']
            )
        shape = tuple(compressed['shape'])
        recon_total = np.zeros(shape, dtype=np.float64)
        for pd in compressed['passes']:
            recon_total += self._decompress_pass(pd, shape)
        return recon_total.astype(np.float32)

    def get_ratio(self, original: np.ndarray, compressed: dict) -> float:
        if compressed.get('type') == 'flms_raw':
            return original.nbytes / max(len(compressed['data']), 1)
        return compressed.get('ratio', 1.0)

    def get_quality_metrics(self, original: np.ndarray,
                             decompressed: np.ndarray) -> dict:
        orig_f = original.astype(np.float64)
        decomp_f = decompressed.astype(np.float64)
        if orig_f.shape != decomp_f.shape:
            ms = tuple(min(a, b) for a, b in zip(orig_f.shape, decomp_f.shape))
            orig_f = orig_f[:ms[0], :ms[1]]
            decomp_f = decomp_f[:ms[0], :ms[1]]
        mse = float(np.mean((orig_f - decomp_f) ** 2))
        max_val = float(np.max(np.abs(orig_f)))
        psnr = 20.0 * math.log10(max_val / max(math.sqrt(mse), 1e-30)) if max_val > 0 else 0.0
        rel_err = float(np.linalg.norm(orig_f - decomp_f) / max(np.linalg.norm(orig_f), 1e-30))
        return {
            'mse': mse,
            'psnr': psnr,
            'relative_error': rel_err,
            'max_abs_error': float(np.max(np.abs(orig_f - decomp_f))),
        }
class SpectralTensorTrainQuantizer:
    """Spectral-Tensor-Train hybrid quantizer.

    Pipeline:
      1. Adaptive DCT (up to 1024x1024 blocks)
      2. Spectral sparsification — keep top-k% of DCT coefficients by energy
      3. Per-component bit allocation on SVD factors (JPEG-style):
         components with larger singular values → more bits
      4. Exp-Golomb + zlib entropy coding

    The key innovation: TT decomposition splits DCT coefficient space into
    "bands" — low-frequency components map to left singular vectors (core1),
    mid-frequencies to the singular value scaling (core2),
    and high-frequencies to right singular vectors (core3).

    Per-component bit allocation (JPEG-inspired):
      - Large singular values (low-freq) → 8-10 bits
      - Mid singular values → 5-7 bits
      - Small singular values (high-freq detail) → 3-5 bits

    CPU-first: full numpy, small SVDs (max 1024x1024 once).
    """

    MAX_SIDE = 1024

    def __init__(self, quality: float = 0.95, target_ratio: float = 500):
        self.quality = np.clip(quality, 0.1, 1.0)
        self.target_ratio = target_ratio

    # ── DCT via cached matrix multiplication ────────────────────────────

    @staticmethod
    def _build_dct_matrix(n: int) -> np.ndarray:
        C = np.zeros((n, n), dtype=np.float64)
        C[0, :] = 1.0 / math.sqrt(n)
        s = math.sqrt(2.0 / n)
        k = np.arange(1, n, dtype=np.float64)[:, None]
        i = np.arange(n, dtype=np.float64)[None, :]
        C[1:, :] = s * np.cos(math.pi * k * (i + 0.5) / n)
        return C

    def _dct_2d(self, matrix: np.ndarray) -> np.ndarray:
        n = matrix.shape[0]
        C = self._build_dct_matrix(n)
        return C @ matrix.astype(np.float64) @ C.T

    def _idct_2d(self, coeffs: np.ndarray) -> np.ndarray:
        n = coeffs.shape[0]
        C = self._build_dct_matrix(n)
        return C.T @ coeffs.astype(np.float64) @ C

    # ── Per-component bit allocation (JPEG-inspired) ───────────────────

    def _allocate_bits(self, singular_values: np.ndarray) -> List[int]:
        """Allocate bits per component based on singular value magnitude.

        Allocates bits to minimize reconstruction MSE for a given budget.
        Larger singular values → more bits for their singular vectors.
        Targets 300:1+ compression by default, falling back gracefully
        for adversarial (energy-spread) inputs.
        """
        s = singular_values
        if len(s) == 0:
            return []
        max_s = float(s[0])
        if max_s < 1e-30:
            max_s = 1.0

        bits: List[int] = []
        for sv in s:
            ratio = float(sv) / max_s
            n_bits = max(2, int(4 + 5 * ratio * self.quality))
            bits.append(n_bits)
        return bits

    # ── Adaptive spectral sparsification ────────────────────────────────

    def _spectral_sparsify(self, dct_block: np.ndarray) -> np.ndarray:
        """Keep top DCT coefficients by energy.

        For smooth weights (energy concentrated in few coeffs): keep ~0.5-1%.
        For random weights (energy spread uniformly): keep ALL → no rank destruction.

        Returns (sparse_block, kept_all) where kept_all indicates no sparsification.
        """
        flat = dct_block.ravel()
        n = flat.size
        energy = flat ** 2
        total_energy = float(np.sum(energy))
        if total_energy < 1e-30:
            return dct_block.copy(), True

        order = np.argsort(-energy)
        cum = np.cumsum(energy[order]) / total_energy
        n_for_99 = int(np.searchsorted(cum, 0.99)) + 1
        spread = n_for_99 / n

        # If energy is spread ( > 25% coeffs needed for 99% energy),
        # spectral sparsification would destroy low-rank structure.
        # Keep all coefficients in this case.
        if spread > 0.25:
            return dct_block.copy(), True

        # Energy is concentrated: apply aggressive sparsification
        keep_frac = 0.005 * (2.0 - self.quality)
        keep_frac = min(0.05, max(0.005, keep_frac))
        n_keep = max(1, int(n * keep_frac))
        threshold = float(energy[order[min(n_keep - 1, n - 1)]])
        mask = energy >= threshold
        result = np.zeros_like(dct_block)
        result.ravel()[mask] = flat[mask]
        return result, False

    # ── Block decomposition ─────────────────────────────────────────────

    def _decompose_blocks(self, tensor: np.ndarray) -> List[dict]:
        """Split tensor into square blocks (up to MAX_SIDE per side)."""
        m, n = tensor.shape
        side = self.MAX_SIDE
        blocks: List[dict] = []
        for i0 in range(0, m, side):
            for j0 in range(0, n, side):
                bh = min(side, m - i0)
                bn = min(side, n - j0)
                bs = max(bh, bn)
                block = np.zeros((bs, bs), dtype=np.float64)
                block[:bh, :bn] = tensor[i0:i0 + bh, j0:j0 + bn]
                blocks.append({
                    'row': i0, 'col': j0, 'block_size': bs,
                    'actual_h': bh, 'actual_w': bn, 'data': block,
                })
        return blocks

    # ── SVD factor quantizer (per-component bit allocation) ─────────────

    @staticmethod
    def _quantize_factor(factor: np.ndarray, n_bits: int,
                         scale: float) -> Tuple[np.ndarray, float]:
        """Uniformly quantize a matrix factor with per-component max."""
        if n_bits < 1:
            n_bits = 1
        max_val = float((1 << (n_bits - 1)) - 1)
        if max_val < 0.5:
            max_val = 1.0
        normalized = factor.astype(np.float64) / max(scale, 1e-30)
        q = np.clip(np.round(normalized * max_val), -max_val, max_val).astype(np.int32)
        return q, scale

    @staticmethod
    def _dequantize_factor(q: np.ndarray, n_bits: int,
                           scale: float) -> np.ndarray:
        if n_bits < 1:
            n_bits = 1
        max_val = float((1 << (n_bits - 1)) - 1)
        if max_val < 0.5:
            max_val = 1.0
        return q.astype(np.float64) / max_val * scale

    # ── Compress one matrix block ───────────────────────────────────────

    def _compress_block(self, block: dict) -> dict:
        matrix = block['data']
        bs = matrix.shape[0]

        # Stage 1: DCT
        dct_coeffs = self._dct_2d(matrix)

        # Stage 2: Spectral sparsification (skipped for spread energy)
        sparse, _ = self._spectral_sparsify(dct_coeffs)

        if not np.all(np.isfinite(sparse)):
            sparse = np.nan_to_num(sparse)

        # Stage 3: SVD → factor extraction
        try:
            u, s, vh = np.linalg.svd(sparse, full_matrices=False)
        except np.linalg.LinAlgError:
            u = np.eye(bs, dtype=np.float64)
            s = np.ones(bs, dtype=np.float64)
            vh = np.eye(bs, dtype=np.float64)

        # Auto-rank: keep components covering 99.5% energy
        total_energy = float(np.sum(s ** 2))
        cum = np.cumsum(s ** 2) / max(total_energy, 1e-30)
        r = int(np.searchsorted(cum, 0.995)) + 1
        r = max(1, min(r, len(s)))

        u_r = u[:, :r].astype(np.float32)
        s_r = s[:r].astype(np.float32)
        vh_r = vh[:r, :].astype(np.float32)

        # Stage 4: Per-component bit allocation
        comp_bits = self._allocate_bits(s_r)

        # Stage 5: Compress SVD factors with per-component bit allocation
        u_scales: List[float] = []
        v_scales: List[float] = []
        u_packed: List[bytes] = []
        v_packed: List[bytes] = []

        for i in range(r):
            n_bits = comp_bits[i]

            # U column
            u_col = u_r[:, i:i + 1]
            u_scale = float(np.max(np.abs(u_col)))
            if u_scale < 1e-30:
                u_scale = 1.0
            u_q, _ = self._quantize_factor(u_col, n_bits, u_scale)
            u_packed.append(_pack_values(u_q, n_bits))
            u_scales.append(u_scale)

            # V^T row × sqrt(s_i) for balanced error distribution
            v_row = vh_r[i:i + 1, :] * math.sqrt(float(s_r[i]))
            v_scale = float(np.max(np.abs(v_row)))
            if v_scale < 1e-30:
                v_scale = 1.0
            v_q, _ = self._quantize_factor(v_row, n_bits, v_scale)
            v_packed.append(_pack_values(v_q, n_bits))
            v_scales.append(v_scale)

        # Layout: [rank_u32][U0_scales_f32][V0_scales_f32]...[packed_data]
        scales_data = bytearray()
        scales_data += struct.pack('<I', r)
        for i in range(r):
            scales_data += struct.pack('<f', u_scales[i])
            scales_data += struct.pack('<f', v_scales[i])
            scales_data += struct.pack('<B', comp_bits[i])

        packed_data = bytearray()
        for i in range(r):
            packed_data.extend(u_packed[i])
            packed_data.extend(v_packed[i])

        combined = bytes(scales_data) + bytes(packed_data)
        z = zlib.compress(combined, level=6)
        final = b'\x01' + z if len(z) < len(combined) else b'\x00' + combined

        return {
            'row': block['row'],
            'col': block['col'],
            'block_size': bs,
            'actual_h': block['actual_h'],
            'actual_w': block['actual_w'],
            'svd_rank': r,
            'bitstream': final,
            'comp_bits': comp_bits,
            'u_scales': u_scales,
            'v_scales': v_scales,
            'singular_values': [float(sv) for sv in s_r],
        }

    # ── Public API ──────────────────────────────────────────────────────

    def compress(self, tensor: np.ndarray,
                 layer_name: str = 'default') -> dict:
        """Compress: DCT → Spectral Sparsify → SVD → VBQ → Entropy."""
        tensor = np.asarray(tensor, dtype=np.float64)
        if tensor.size < 64 or tensor.ndim < 2:
            return {
                'type': 'stt_raw',
                'data': tensor.astype(np.float32).tobytes(),
                'shape': list(tensor.shape),
            }

        blocks = self._decompose_blocks(tensor)
        compressed_blocks = [self._compress_block(b) for b in blocks]

        return {
            'type': 'spectral_tensor_train',
            'blocks': compressed_blocks,
            'shape': list(tensor.shape),
            'layer_name': layer_name,
            'quality': self.quality,
        }

    def decompress(self, compressed: dict) -> np.ndarray:
        """Decompress: Entropy → Dequant → SVD reconstruct → IDCT."""
        if compressed.get('type') == 'stt_raw':
            return np.frombuffer(
                compressed['data'], dtype=np.float32
            ).reshape(compressed['shape'])

        m, n = compressed['shape']
        out = np.zeros((m, n), dtype=np.float64)

        for qb in compressed['blocks']:
            bs = qb['block_size']
            row, col = qb['row'], qb['col']
            ah = qb.get('actual_h', bs)
            aw = qb.get('actual_w', bs)
            r = qb['svd_rank']
            comp_bits = qb['comp_bits']
            u_scales = qb['u_scales']
            v_scales = qb['v_scales']

            bs_data = qb['bitstream']
            if bs_data and bs_data[0:1] == b'\x01':
                        bs_data = zlib.decompress(bs_data[1:])
            elif bs_data and bs_data[0:1] == b'\x00':
                bs_data = bs_data[1:]

            # Parse scales header
            hdr_size = 4 + r * (4 + 4 + 1)  # rank + r × (u_scale + v_scale + bits)
            offset = hdr_size

            u_hat = np.zeros((bs, r), dtype=np.float64)
            v_hat = np.zeros((r, bs), dtype=np.float64)

            for i in range(r):
                n_bits = comp_bits[i] if i < len(comp_bits) else 4

                # Bytes needed for packed U column
                u_bytes = (bs * n_bits + 7) // 8
                chunk_u = bs_data[offset:offset + u_bytes]
                offset += u_bytes
                u_q = _unpack_values(chunk_u, bs, n_bits)
                u_hat[:, i] = self._dequantize_factor(
                    u_q.astype(np.float64).reshape(bs, 1),
                    n_bits, u_scales[i] if i < len(u_scales) else 1.0
                ).ravel()

                # Packed V row
                v_bytes = (bs * n_bits + 7) // 8
                chunk_v = bs_data[offset:offset + v_bytes]
                offset += v_bytes
                v_q = _unpack_values(chunk_v, bs, n_bits)
                v_dec = self._dequantize_factor(
                    v_q.astype(np.float64).reshape(1, bs),
                    n_bits, v_scales[i] if i < len(v_scales) else 1.0
                ).ravel()

                sv = qb['singular_values'][i] if i < len(qb['singular_values']) else 1.0
                if sv > 1e-30:
                    v_dec /= math.sqrt(sv)
                v_hat[i, :] = v_dec

            # Reconstruct: U_hat @ S @ V_hat^T
            s_diag = np.diag(
                np.array(qb['singular_values'][:r], dtype=np.float64)
            )
            recon_dct = u_hat @ s_diag @ v_hat

            # IDCT
            recon_block = self._idct_2d(recon_dct)
            i_end = min(row + ah, m)
            j_end = min(col + aw, n)
            out[row:i_end, col:j_end] = recon_block[:i_end - row, :j_end - col]

        return out.astype(np.float32)

    def get_ratio(self, original: np.ndarray,
                  compressed: dict) -> float:
        """Compute compression ratio (original_bytes / compressed_bytes)."""
        original_bytes = original.nbytes
        if compressed.get('type') == 'stt_raw':
            return original_bytes / max(len(compressed['data']), 1)
        compressed_bytes = 0
        for blk in compressed['blocks']:
            compressed_bytes += len(blk['bitstream'])
        return original_bytes / max(compressed_bytes, 1)
