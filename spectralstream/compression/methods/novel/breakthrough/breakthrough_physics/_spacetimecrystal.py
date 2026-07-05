from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

def _next_power_of_two(n: int) -> int:
    return 1 << (n - 1).bit_length()

class SpaceTimeCrystal:
    """4D space-time crystal: fold the 2D weight into a 4D tensor and treat
    the extra two dimensions as 'space-time' coordinates. The crystal
    structure means the tensor is sparse in the 4D DCT domain.

    Real implementation: reshape to 4D, apply DCT along time-like dims,
    threshold, quantize, store sparse spectrum.
    """

    name = "space_time_crystal"
    category = "breakthrough_physics"

    def compress(
        self, tensor: np.ndarray, keep_frac: float = 0.15
    ) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        orig_shape = t.shape
        m, n = orig_shape
        # Find 4D factorization
        d1 = max(1, int(math.isqrt(m)))
        d2 = max(1, m // d1)
        d3 = max(1, int(math.isqrt(n)))
        d4 = max(1, n // d3)
        while d1 * d2 < m:
            d2 += 1
        while d3 * d4 < n:
            d4 += 1
        # Pad and reshape to 4D
        flat = t.ravel()
        total = d1 * d2 * d3 * d4
        padded = np.zeros(total, dtype=np.float64)
        padded[: len(flat)] = flat
        t4d = padded.reshape(d1, d2, d3, d4)
        # DCT along time-like dims (2 and 3)
        spec = np.fft.rfftn(t4d, axes=(2, 3))
        mag = np.abs(spec)
        thr = np.percentile(mag, (1.0 - keep_frac) * 100)
        mask = mag >= thr
        n_keep = int(np.sum(mask))
        kept = spec[mask].astype(np.complex64)
        buf = struct.pack("<IIIIII", d1, d2, d3, d4, len(flat), n_keep)
        buf += mask.astype(np.uint8).tobytes()
        buf += kept.tobytes()
        return bytes(buf), {
            "shape": orig_shape,
            "dims": (d1, d2, d3, d4),
            "n_elements": len(flat),
        }

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        d1, d2, d3, d4, n_elems, n_keep = struct.unpack_from("<IIIIII", data, 0)
        pos = 24
        mask = np.frombuffer(
            data[pos : pos + (d1 * d2 * (d3 // 2 + 1) * d4 + 7) // 8],
            dtype=np.uint8,
        )
        pos += len(mask)
        kept = np.frombuffer(data[pos : pos + n_keep * 8], dtype=np.complex64)
        spec = np.zeros(d1 * d2 * (d3 // 2 + 1) * d4, dtype=np.complex128)
        spec[np.frombuffer(mask, dtype=np.uint8).astype(bool)[: len(spec)]] = kept
        spec_4d = spec.reshape(d1, d2, d3 // 2 + 1, d4)
        recon = np.fft.irfftn(spec_4d, axes=(2, 3), s=(d3, d4))
        flat = recon.ravel()[:n_elems]
        return flat.astype(np.float32).reshape(metadata["shape"])
