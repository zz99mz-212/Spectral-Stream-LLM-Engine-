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

class RenormalizationGroupFlow:
    """RG flow: iteratively coarse-grain the weight matrix (Kadanoff
    blocking). Store only the fixed-point theory (smallest scale)
    and the RG flow trajectory (differences between scales).
    Different layers correspond to different RG scales.

    Real: recursive block averaging + detail coefficients (like
    wavelet decomposition).
    """

    name = "renormalization_group_flow"
    category = "breakthrough_physics"

    def compress(self, tensor: np.ndarray, n_scales: int = 3) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        orig_shape = t.shape
        scales = []
        details = []
        current = t.copy()
        for s in range(n_scales):
            m, n = current.shape
            cm, cn = max(1, m // 2), max(1, n // 2)
            # Block average (coarse-grain)
            coarse = np.zeros((cm, cn), dtype=np.float64)
            for i in range(cm):
                for j in range(cn):
                    block = current[
                        i * 2 : min((i + 1) * 2, m),
                        j * 2 : min((j + 1) * 2, n),
                    ]
                    coarse[i, j] = float(np.mean(block))
            # Detail = difference between coarse-grained and original
            up = np.kron(coarse, np.ones((2, 2)))[:m, :n]
            detail = current - up
            scales.append(coarse)
            details.append(detail)
            current = coarse
        # Store: coarsest scale + all details (quantized)
        coarsest = current.astype(np.float32)
        detail_arrays = [d.astype(np.float32) for d in details]
        buf = struct.pack("<II", orig_shape[0], orig_shape[1])
        buf += struct.pack("<I", n_scales)
        # Coarsest scale
        buf += _serialize(coarsest)
        # Details
        for d in detail_arrays:
            # Quantize detail with threshold
            thr = np.percentile(np.abs(d), 50)
            mask = np.abs(d) >= thr
            kept = d[mask].astype(np.float16).tobytes()
            mask_bytes = mask.astype(np.uint8).tobytes()
            buf += struct.pack("<II", len(mask_bytes), len(kept))
            buf += mask_bytes
            buf += kept
        return bytes(buf), {
            "shape": orig_shape,
            "n_scales": n_scales,
            "sizes": [orig_shape[0] // (2 ** (s + 1)) for s in range(n_scales)],
        }

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        orig_m, orig_n = struct.unpack_from("<II", data, 0)
        pos = 8
        n_scales = struct.unpack_from("<I", data, pos)[0]
        pos += 4
        # Read coarsest scale
        cm = max(1, orig_m // (2**n_scales))
        cn = max(1, orig_n // (2**n_scales))
        coarsest = np.frombuffer(
            data[pos : pos + cm * cn * 4], dtype=np.float32
        ).reshape(cm, cn)
        pos += cm * cn * 4
        # Read and apply details in reverse order
        recon = coarsest.copy()
        for s in range(n_scales - 1, -1, -1):
            mask_len, detail_len = struct.unpack_from("<II", data, pos)
            pos += 8
            mask_bytes = data[pos : pos + mask_len]
            pos += mask_len
            detail_bytes = data[pos : pos + detail_len]
            pos += detail_len
            # Reconstruct detail
            n_total = mask_len * 8
            mask = np.unpackbits(np.frombuffer(mask_bytes, dtype=np.uint8))[
                :n_total
            ].astype(bool)
            det = np.zeros(n_total, dtype=np.float32)
            det_vals = np.frombuffer(detail_bytes, dtype=np.float16).astype(np.float32)
            det[mask[: len(det_vals)]] = det_vals
            det_2d = det.reshape(recon.shape[0] * 2, recon.shape[1] * 2)[
                : orig_m // (2**s), : orig_n // (2**s)
            ]
            # Upsample and add detail
            up = np.kron(recon, np.ones((2, 2)))[: orig_m // (2**s), : orig_n // (2**s)]
            recon = up + det_2d
        return recon.astype(np.float32)[:orig_m, :orig_n]
