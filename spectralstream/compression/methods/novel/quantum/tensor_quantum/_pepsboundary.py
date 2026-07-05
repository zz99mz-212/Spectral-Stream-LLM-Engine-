from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()


def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()


class PEPSBoundary:
    """PEPS boundary MPS: 2D grid of truncated SVD blocks."""

    name = "peps_boundary"
    category = "tensor_quantum"

    def compress(self, tensor: np.ndarray, boundary_chi: int = 8) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        orig_shape = t.shape
        if t.ndim == 2:
            m, n = t.shape
            chi = min(boundary_chi, m, n)
            stride = max(1, min(chi, min(m, n) // 4))
            sites = []
            for i in range(0, m, stride):
                for j in range(0, n, stride):
                    i_end = min(i + stride, m)
                    j_end = min(j + stride, n)
                    block = t[i:i_end, j:j_end]
                    if block.size < 2:
                        continue
                    block_flat = block.reshape(i_end - i, -1)
                    u, s, vt = np.linalg.svd(block_flat, full_matrices=False)
                    rk = min(chi, len(s))
                    sites.append((u[:, :rk] * s[:rk]).copy().astype(np.float32))
                    sites.append(vt[:rk, :].copy().astype(np.float32))
            meta = dict(
                shape=orig_shape,
                boundary_chi=chi,
                stride=stride,
                n_sites=len(sites),
                site_shapes=[s.shape for s in sites],
            )
            data = struct.pack("<iii", chi, stride, len(sites))
            for s in sites:
                data += _serialize(s)
            return data, meta
        data = struct.pack("<iii", boundary_chi, 0, 0) + _serialize(
            t.ravel().astype(np.float32)
        )
        meta = dict(
            shape=orig_shape,
            boundary_chi=boundary_chi,
            stride=0,
            n_sites=0,
            site_shapes=[],
        )
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        chi, stride, n_sites = struct.unpack_from("<iii", data, 0)
        if n_sites == 0:
            flat = _deserialize(data[12:])
            return flat[: int(np.prod(shape))].reshape(shape).astype(np.float32)
        pos = 12
        m, n = shape
        recon = np.zeros((m, n), dtype=np.float64)
        idx = 0
        for i in range(0, m, stride):
            for j in range(0, n, stride):
                if idx >= n_sites - 1:
                    break
                us = _deserialize(
                    data[pos : pos + int(np.prod(metadata["site_shapes"][idx])) * 4]
                ).reshape(metadata["site_shapes"][idx])
                pos += int(np.prod(metadata["site_shapes"][idx])) * 4
                vt = _deserialize(
                    data[pos : pos + int(np.prod(metadata["site_shapes"][idx + 1])) * 4]
                ).reshape(metadata["site_shapes"][idx + 1])
                pos += int(np.prod(metadata["site_shapes"][idx + 1])) * 4
                idx += 2
                i_end = min(i + stride, m)
                j_end = min(j + stride, n)
                block = us @ vt
                recon[i:i_end, j:j_end] = block[: i_end - i, : j_end - j]
        return recon.astype(np.float32)
