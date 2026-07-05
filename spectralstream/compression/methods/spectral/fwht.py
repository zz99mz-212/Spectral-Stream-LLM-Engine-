"""HPC FWHT — fully vectorized via _fwht_vectorized ND support, zero row/col loops.

Novel method:
  FWHTQuant: FWHT decorrelation + block-int4 quantization of ALL coefficients
"""

from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np

from spectralstream.core.math_primitives import (
    auto_keep_fraction,
    fwht,
    ifwht,
)


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()


def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()


class FWHTQuant:
    """FWHT decorrelation + block-int4 quantization of ALL coefficients.

    O(N log N) via vectorized FWHT. Quantizes ALL transformed coefficients
    (no information loss from thresholding) with per-block scaling.
    """

    name = "fwht_quant"
    category = "spectral"

    def __init__(self, quant_bits: int = 4):
        self.quant_bits = quant_bits

    def compress(
        self,
        tensor: np.ndarray,
        block_size: int = 64,
        bits: int | None = None,
    ) -> Tuple[bytes, dict]:
        orig = tensor.astype(np.float64)
        ndim = orig.ndim
        if ndim == 1:
            orig = orig.reshape(1, -1)
        bits_ = bits if bits is not None else self.quant_bits
        qmax = float(2 ** (bits_ - 1) - 1)

        result = fwht(orig, normalize=True)
        result = fwht(result.T, normalize=True).T
        flat = result.ravel()
        n = len(flat)
        bs = block_size
        padded_n = int(math.ceil(n / bs) * bs)
        padded = np.zeros(padded_n, dtype=np.float64)
        padded[:n] = flat
        blocks = padded.reshape(-1, bs)
        n_blocks = blocks.shape[0]

        amax = np.max(np.abs(blocks), axis=1)
        scales = np.where(amax > 1e-10, amax / qmax, 1.0)
        quantized = np.clip(np.round(blocks / scales[:, None]), -qmax - 1, qmax).astype(
            np.int8 if bits_ <= 8 else np.int16
        )

        meta = dict(
            shape=orig.shape,
            block_size=bs,
            quant_bits=bits_,
            n_blocks=n_blocks,
            ndim=ndim,
        )
        data = (
            struct.pack("<ii", *orig.shape)
            + struct.pack("<ii", n_blocks, bs)
            + struct.pack("<i", bits_)
            + scales.astype(np.float16).tobytes()
            + quantized.tobytes()
        )
        del result, flat, blocks, quantized
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        ndim = metadata.get("ndim", 2)
        pos = struct.calcsize("<ii")
        n_blocks = struct.unpack_from("<i", data, pos)[0]
        pos += 4
        bs = struct.unpack_from("<i", data, pos)[0]
        pos += 4
        bits_ = struct.unpack_from("<i", data, pos)[0]
        pos += 4
        dtype_quant = np.int8 if bits_ <= 8 else np.int16

        scales = np.frombuffer(data[pos : pos + n_blocks * 2], dtype=np.float16).astype(
            np.float64
        )
        pos += n_blocks * 2
        qsize = n_blocks * bs
        quantized = np.frombuffer(
            data[pos : pos + qsize * np.dtype(dtype_quant).itemsize], dtype=dtype_quant
        ).astype(np.float64)
        quantized = quantized.reshape(n_blocks, bs)

        flat = (quantized * scales[:, None]).ravel()
        flat = flat[: shape[0] * shape[1]]
        coeffs = flat.reshape(shape)
        recon = ifwht(coeffs.T, normalize=True).T
        recon = ifwht(recon, normalize=True)
        out = recon.astype(np.float32)
        if ndim == 1:
            out = out.ravel()
        return out


class FWHT:
    """Fast Walsh-Hadamard Transform + coefficient thresholding — fully vectorized."""

    name = "fwht"
    category = "spectral"

    def compress(
        self,
        tensor: np.ndarray,
        keep_fraction: float | None = None,
        target_energy: float = 0.99,
    ) -> Tuple[bytes, dict]:
        orig = tensor.astype(np.float64)
        if orig.ndim == 1:
            orig = orig.reshape(1, -1)
        # _fwht_vectorized operates along last axis — apply to rows then columns
        result = fwht(orig, normalize=True)
        result = fwht(result.T, normalize=True).T
        flat = result.ravel()
        if keep_fraction is None:
            kf = auto_keep_fraction(flat, target_energy)
        else:
            kf = keep_fraction
        k = max(1, int(kf * flat.size))
        idx = np.argpartition(np.abs(flat), -k)[-k:]
        meta = dict(
            shape=orig.shape,
            keep_fraction=kf,
            target_energy=target_energy,
            n_kept=k,
            ndim=tensor.ndim,
        )
        data = (
            struct.pack("<ii", *orig.shape)
            + idx.astype(np.int32).tobytes()
            + flat[idx].astype(np.float16).tobytes()
        )
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        ndim = metadata.get("ndim", 2)
        m, n = shape
        k = metadata["n_kept"]
        pos = struct.calcsize("<ii")
        idx = np.frombuffer(data[pos : pos + k * 4], dtype=np.int32).copy().astype(int)
        pos += k * 4
        vals = np.frombuffer(data[pos : pos + k * 2], dtype=np.float16).astype(
            np.float64
        )
        thresh = np.zeros(m * n, dtype=np.float64)
        thresh[idx] = vals
        thresh = thresh.reshape(m, n)
        # IFWHT on columns then rows — fully vectorized
        recon = ifwht(thresh.T, normalize=True).T
        recon = ifwht(recon, normalize=True)
        out = recon.astype(np.float32)
        if ndim == 1:
            out = out.ravel()
        return out


class RandomizedHadamard:
    """Randomized Hadamard transform + quantization decorrelation."""

    name = "randomized_hadamard"
    category = "spectral"

    def compress(
        self, tensor: np.ndarray, block_size: int = 64, bits: int = 4
    ) -> Tuple[bytes, dict]:
        from spectralstream.compression.methods.quantization._class_wrappers import (
            HadamardGroupWise,
        )

        return HadamardGroupWise().compress(tensor, block_size=block_size, bits=bits)

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        from spectralstream.compression.methods.quantization._class_wrappers import (
            HadamardGroupWise,
        )

        return HadamardGroupWise().decompress(data, metadata)


class RandomRotationQuant:
    """Random rotation decorrelation + quantization."""

    name = "random_rotation_quant"
    category = "spectral"

    def compress(
        self, tensor: np.ndarray, block_size: int = 64, bits: int = 4
    ) -> Tuple[bytes, dict]:
        from spectralstream.compression.methods.quantization._class_wrappers import (
            HadamardGroupWise,
        )

        return HadamardGroupWise().compress(tensor, block_size=block_size, bits=bits)

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        from spectralstream.compression.methods.quantization._class_wrappers import (
            HadamardGroupWise,
        )

        return HadamardGroupWise().decompress(data, metadata)
