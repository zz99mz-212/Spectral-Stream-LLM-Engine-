from __future__ import annotations

import math
import struct
from typing import Any, Dict, Tuple

import numpy as np

from spectralstream.compression.methods.novel._common import (
    _block_int8_fallback,
    _block_int8_decompress,
    _svd_compress,
    _svd_decompress,
)


def _gauge_compress(tensor, method="svd", rank=0):
    if method == "svd":
        return _svd_compress(tensor, rank)
    return _block_int8_fallback(tensor)

def _gauge_decompress(data, meta):
    if meta.get("_svd"):
        return _svd_decompress(data, meta)
    return _block_int8_decompress(data, meta)

class GaugeEquivariantFFT:
    name = "gauge_equivariant_fft"
    category = "revolutionary_gauge"

    def compress(self, tensor, threshold=0.05, **params):
        flat = tensor.ravel().astype(np.float32)
        n = len(flat)
        padded = 1 << (n - 1).bit_length() if n > 0 else 1
        data = np.zeros(padded, dtype=np.float32)
        data[:n] = flat
        spec = np.fft.rfft(data)
        mag = np.abs(spec)
        th = np.percentile(mag, 100 * (1 - threshold))
        mask = mag >= th
        keep = spec[mask]
        # Gauge twist: multiply by phase factor
        phases = np.exp(2j * np.pi * np.arange(len(keep)) / max(1, len(keep)))
        keep_twisted = keep * phases
        buf = (
            struct.pack("<II", n, int(mask.sum()))
            + np.packbits(mask.astype(np.uint8)).tobytes()
            + keep_twisted.astype(np.complex64).tobytes()
        )
        return buf, {"n": n, "padded": padded}

    def decompress(self, data, metadata):
        n, n_keep = struct.unpack_from("<II", data, 0)
        pos = 8
        padded = metadata.get("padded", 1 << (n - 1).bit_length()) if n > 0 else 1
        n_bins = padded // 2 + 1
        mb = (n_bins + 7) // 8
        mask = np.unpackbits(np.frombuffer(data[pos : pos + mb], dtype=np.uint8))[
            :n_bins
        ].astype(bool)
        pos += mb
        keep = np.frombuffer(data[pos : pos + n_keep * 8], dtype=np.complex64)
        phases = np.exp(-2j * np.pi * np.arange(len(keep)) / max(1, len(keep)))
        spec = np.zeros(n_bins, dtype=np.complex64)
        spec[mask] = keep * phases
        return np.fft.irfft(spec)[:n].astype(np.float32)
