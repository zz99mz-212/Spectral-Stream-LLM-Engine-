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

class DiscreteTimeCrystal:
    """Discrete time crystal via many-body localization: store a 'pulse
    sequence' of Floquet operators that builds the weight when applied
    stroboscopically.

    Real implementation: break the weight into N_x × N_y blocks. Compute
    the 'pulse' as the block-wise DCT coefficients. Store only the
    most significant pulses (drive sequence).
    """

    name = "discrete_time_crystal"
    category = "breakthrough_physics"

    def compress(self, tensor: np.ndarray, n_pulses: int = 16) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        m, n = t.shape
        n_blocks = max(2, min(m, n) // 16)
        # Divide into blocks along rows
        block_h = max(1, m // n_blocks)
        pulses = []
        for i in range(0, m, block_h):
            end = min(i + block_h, m)
            block_dct = np.fft.rfft(t[end - block_h : end].ravel())
            pulses.append(block_dct)
        pulses_arr = np.array(pulses)
        magnitudes = np.abs(pulses_arr)
        k = min(n_pulses, magnitudes.shape[1])
        thr = np.sort(magnitudes.ravel())[-k] if k > 0 else 0
        mask = magnitudes >= thr
        n_keep = int(np.sum(mask))
        kept_flat = pulses_arr[mask].astype(np.complex64)
        mask_bytes = mask.astype(np.uint8).tobytes()
        buf = struct.pack("<III", m, n, n_keep)
        buf += mask_bytes
        buf += kept_flat.tobytes()
        return bytes(buf), {
            "shape": tensor.shape,
            "m": m,
            "n": n,
            "n_pulses": len(pulses),
            "n_keep": n_keep,
            "block_h": block_h,
        }

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        m, n, n_keep = struct.unpack_from("<III", data, 0)
        pos = 12
        block_h = metadata.get("block_h", max(1, m // max(1, (m // 16))))
        n_pulses = metadata.get("n_pulses", 0)
        mask_size = n_pulses
        # Determine mask size from remaining data
        if n_pulses > 0:
            dct_len = (block_h - 1) // 2 + 1
            mask_size = n_pulses * dct_len
        mask_bytes_len = (mask_size + 7) // 8
        mask = np.frombuffer(data[pos : pos + mask_bytes_len], dtype=np.uint8)
        pos += mask_bytes_len
        # Reconstruct mask from bytes
        mask_bool = np.unpackbits(mask)[:mask_size].astype(bool)
        n_total = mask_size
        if n_keep > 0:
            kept = np.frombuffer(data[pos : pos + n_keep * 8], dtype=np.complex64)
        else:
            kept = np.array([], dtype=np.complex64)
        # Rebuild pulses
        pulses_recon = np.zeros(n_total, dtype=np.complex128)
        pulses_recon[mask_bool] = kept
        pulses_2d = pulses_recon.reshape(n_pulses, -1)
        recon = np.zeros(m * n, dtype=np.float64)
        for i in range(n_pulses):
            start = i * block_h
            end = min(start + block_h, m)
            n_elems = (end - start) * n
            if n_elems == 0:
                continue
            sig = np.fft.irfft(pulses_2d[i], n=n_elems)
            recon[start * n : end * n] = sig[:n_elems]
        return recon.astype(np.float32).reshape(m, n)
