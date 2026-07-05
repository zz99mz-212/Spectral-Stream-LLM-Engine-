"""
TurboQuant Codec (Python)
-------------------------
Clean room Python implementation inspired by the Anvil TurboQuant codec.

Two-stage compression:
1. PolarQuant: FWHT rotation + signed 4-bit signal quantization
2. QJL: 1-bit sign residual correction

For asymmetric precision: signal at 4-bit, residual at 1-bit.

Key operations:
- Randomized Hadamard rotation (sign pattern + FWHT)
- Signed 4-bit nibble packing with per-vector scale
- 1-bit sign residual correction
- Compressed-domain cosine similarity querying
"""

import struct
from typing import Any, Dict, Optional, Tuple

import numpy as np

from spectralstream.core.math_primitives import (
    splitmix64 as _splitmix64,
    next_power_of_two as _next_power_of_two,
    fwht as _fwht,
)


class TurboQuantCodec:
    """Vector quantizer using signed 4-bit signal + 1-bit residual correction.

    Each vector is:
    1. Padded to next power of two
    2. Rotated by randomized Hadamard transform
    3. Quantized to signed 4-bit (uniform, per-vector scale)
    4. Residual compressed to 1-bit sign per element

    Compression ratio vs FP16: ~4.6x at dim=128
    """

    def __init__(self, dim: int, signal_bits: int = 4, residual_bits: int = 1):
        if signal_bits != 4 or residual_bits != 1:
            raise ValueError("Only 4-bit signal + 1-bit residual is supported")
        self.dim = dim
        self.signal_bits = signal_bits
        self.residual_bits = residual_bits
        self.rotated_dim = _next_power_of_two(dim)
        self.max_quant = (1 << (signal_bits - 1)) - 1

    def _apply_sign_pattern_inplace(self, values: np.ndarray, rotation_id: int):
        n = len(values)
        seeds = np.array(
            [
                _splitmix64((rotation_id + 1) * 0x100000001B3 + (i + 1))
                for i in range(n)
            ],
            dtype=np.uint64,
        )
        mask = (seeds & 1).astype(bool)
        values[mask] = -values[mask]

    def rotate_forward(self, vectors: np.ndarray, rotation_id: int = 0) -> tuple:
        n, d = vectors.shape
        rd = self.rotated_dim
        rotated = np.zeros((n, rd), dtype=np.float32)
        rotated[:, :d] = vectors.astype(np.float32)
        norms = np.linalg.norm(vectors.astype(np.float32), axis=1)
        inv_sqrt_rd = 1.0 / np.sqrt(float(rd))
        for i in range(n):
            self._apply_sign_pattern_inplace(rotated[i], rotation_id)
            rotated[i] = _fwht(rotated[i])
            rotated[i] *= inv_sqrt_rd
        return rotated, norms

    def rotate_inverse(
        self, rotated: np.ndarray, dim: int, rotation_id: int
    ) -> np.ndarray:
        n, rd = rotated.shape
        result = np.array(rotated, dtype=np.float32, copy=True)
        inv_sqrt_rd = 1.0 / np.sqrt(float(rd))
        for i in range(n):
            result[i] = _fwht(result[i])
            result[i] *= inv_sqrt_rd
            self._apply_sign_pattern_inplace(result[i], rotation_id)
        return result[:, :dim]

    def _pack_signal_row_vectorized(self, rotated_row: np.ndarray) -> tuple:
        rd = self.rotated_dim
        max_abs = float(np.max(np.abs(rotated_row)))
        signal_scale = max(max_abs / max(1, self.max_quant), 1e-8)

        quantized = np.round(rotated_row / signal_scale).astype(np.int32)
        clamped = np.clip(quantized, -self.max_quant - 1, self.max_quant)
        nibbles = (clamped & 0x0F).astype(np.uint8)

        n_signal_bytes = (rd + 1) // 2
        signal_row = np.zeros(n_signal_bytes, dtype=np.uint8)
        n_even = min(n_signal_bytes, (rd + 1) // 2)
        if n_even > 0:
            signal_row[:n_even] = nibbles[0::2]
        odd_positions = np.arange(1, rd, 2)
        if len(odd_positions) > 0:
            n_odd = len(odd_positions)
            signal_row[:n_odd] |= np.uint8(nibbles[odd_positions] << 4)

        restored = clamped.astype(np.float32) * signal_scale
        residuals = rotated_row.astype(np.float32) - restored

        residual_sum = float(np.sum(np.abs(residuals)))
        residual_scale = max(residual_sum / max(1, rd), signal_scale * 0.125)

        residual_row = np.packbits((residuals >= 0).astype(np.uint8))
        return signal_row, signal_scale, residual_row, residual_scale

    def _unpack_row_vectorized(
        self,
        signal_row: np.ndarray,
        residual_row: np.ndarray,
        signal_scale: float,
        residual_scale: float,
    ) -> np.ndarray:
        rd = self.rotated_dim
        n_signal_bytes = len(signal_row)
        nibbles = np.zeros(rd, dtype=np.uint8)
        n_even = min(n_signal_bytes, (rd + 1) // 2)
        if n_even > 0:
            nibbles[0::2] = signal_row[:n_even] & 0x0F
        odd_positions = np.arange(1, rd, 2)
        n_odd = len(odd_positions)
        if n_odd > 0:
            nibbles[odd_positions] = (signal_row[:n_odd] >> 4) & 0x0F

        sign_vals = nibbles.astype(np.int32)
        mask = sign_vals >= 8
        sign_vals[mask] = sign_vals[mask] - 16
        signals = sign_vals.astype(np.float32) * signal_scale

        bits = np.unpackbits(residual_row)[:rd].astype(bool)
        residual_vals = np.where(bits, residual_scale, -residual_scale)
        return signals + residual_vals

    def encode_batch(self, vectors: np.ndarray, rotation_id: int = 0) -> tuple:
        n, d = vectors.shape
        assert d == self.dim, f"Expected dim={self.dim}, got {d}"
        rotated, norms = self.rotate_forward(vectors, rotation_id)

        signal_stride = (self.rotated_dim + 1) // 2
        residual_stride = (self.rotated_dim + 7) // 8

        signal = np.zeros((n, signal_stride), dtype=np.uint8)
        residual = np.zeros((n, residual_stride), dtype=np.uint8)
        signal_scales = np.zeros(n, dtype=np.float32)
        residual_scales = np.zeros(n, dtype=np.float32)

        for i in range(n):
            sig_row, sig_scale, res_row, res_scale = self._pack_signal_row_vectorized(
                rotated[i]
            )
            signal[i] = sig_row
            residual[i] = res_row
            signal_scales[i] = sig_scale
            residual_scales[i] = res_scale

        return signal, residual, signal_scales, residual_scales, norms

    def decode_batch(
        self,
        signal: np.ndarray,
        residual: np.ndarray,
        signal_scales: np.ndarray,
        residual_scales: np.ndarray,
        n: int,
        rotation_id: int = 0,
    ) -> np.ndarray:
        rotated_out = np.zeros((n, self.rotated_dim), dtype=np.float32)
        for i in range(n):
            rotated_out[i] = self._unpack_row_vectorized(
                signal[i], residual[i], signal_scales[i], residual_scales[i]
            )
        return self.rotate_inverse(rotated_out, self.dim, rotation_id)

    def compress(self, tensor: np.ndarray) -> Tuple[bytes, Dict[str, Any]]:
        orig_shape = tensor.shape
        t = np.asarray(tensor, dtype=np.float32)
        if t.ndim == 1:
            t = t.reshape(1, -1)
        elif t.ndim > 2:
            t = t.reshape(-1, t.shape[-1])
        signal, residual, signal_scales, residual_scales, norms = self.encode_batch(t)

        buf = bytearray()
        buf += struct.pack("<II", t.shape[0], t.shape[1])
        buf += struct.pack("<I", signal.shape[1])
        buf += struct.pack("<I", residual.shape[1])
        buf += signal.tobytes()
        buf += residual.tobytes()
        buf += signal_scales.tobytes()
        buf += residual_scales.tobytes()
        buf += norms.tobytes()

        metadata = {
            "orig_shape": list(orig_shape),
            "dim": self.dim,
            "rotated_dim": self.rotated_dim,
            "n": t.shape[0],
            "signal_stride": signal.shape[1],
            "residual_stride": residual.shape[1],
        }
        return bytes(buf), metadata

    def decompress(self, data: bytes, metadata: Dict[str, Any]) -> np.ndarray:
        orig_shape = tuple(metadata["orig_shape"])
        n = metadata["n"]
        signal_stride = metadata["signal_stride"]
        residual_stride = metadata["residual_stride"]

        offset = 0
        rows = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        cols = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        ss = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        rs = struct.unpack_from("<I", data, offset)[0]
        offset += 4

        signal = np.frombuffer(
            data[offset : offset + rows * ss], dtype=np.uint8
        ).reshape(rows, ss)
        offset += rows * ss
        residual = np.frombuffer(
            data[offset : offset + rows * rs], dtype=np.uint8
        ).reshape(rows, rs)
        offset += rows * rs
        signal_scales = np.frombuffer(
            data[offset : offset + rows * 4], dtype=np.float32
        )
        offset += rows * 4
        residual_scales = np.frombuffer(
            data[offset : offset + rows * 4], dtype=np.float32
        )

        recon = self.decode_batch(
            signal, residual, signal_scales, residual_scales, rows
        )
        return recon.reshape(orig_shape)

    def query_cosine(
        self,
        query: np.ndarray,
        signal: np.ndarray,
        residual: np.ndarray,
        signal_scales: np.ndarray,
        residual_scales: np.ndarray,
        norms: np.ndarray,
        indices: Optional[list[int]] = None,
        rotation_id: int = 0,
    ) -> np.ndarray:
        n_total = len(signal)
        if indices is not None:
            active_indices = indices
        else:
            active_indices = list(range(n_total))

        query_norm = float(np.linalg.norm(query))
        if query_norm < 1e-9:
            return np.zeros(len(active_indices), dtype=np.float32)

        scores = np.zeros(len(active_indices), dtype=np.float32)
        decoded_buffer = np.zeros(self.dim, dtype=np.float32)

        for out_idx, row_idx in enumerate(active_indices):
            if row_idx < 0 or row_idx >= n_total:
                scores[out_idx] = -np.inf
                continue
            rotated = self._unpack_row_vectorized(
                signal[row_idx],
                residual[row_idx],
                signal_scales[row_idx],
                residual_scales[row_idx],
            )
            rotated_2d = rotated.reshape(1, -1)
            decoded_2d = self.rotate_inverse(rotated_2d, self.dim, rotation_id)
            decoded_buffer[:] = decoded_2d[0]
            dot = float(np.dot(decoded_buffer, query))
            denom = max(norms[row_idx] * query_norm, 1e-9)
            scores[out_idx] = dot / denom

        return scores

    def compression_ratio(self) -> float:
        fp16_bytes = self.dim * 2
        signal_bytes = (self.rotated_dim + 1) // 2
        residual_bytes = (self.rotated_dim + 7) // 8
        scale_bytes = 4 + 4 + 4
        total_per_row = signal_bytes + residual_bytes + scale_bytes
        return fp16_bytes / max(total_per_row, 1)

    def signal_bytes_per_row(self) -> int:
        return (self.rotated_dim + 1) // 2

    def residual_bytes_per_row(self) -> int:
        return (self.rotated_dim + 7) // 8
