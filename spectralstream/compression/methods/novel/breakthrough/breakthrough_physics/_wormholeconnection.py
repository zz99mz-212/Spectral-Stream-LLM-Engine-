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

class WormholeConnection:
    """Wormhole: different row blocks are connected by Einstein-Rosen
    bridges. Store only the wormhole throat parameters (radius r₀,
    charge q) and the 'bridge' connections between blocks.
    Information flows between blocks through the wormhole geometry.

    Real: block-wise compression with cross-block delta coding.
    """

    name = "wormhole_connection"
    category = "breakthrough_physics"

    def compress(self, tensor: np.ndarray, block_size: int = 32) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        m, n = t.shape
        bs = min(block_size, m, n)
        n_row_blocks = (m + bs - 1) // bs
        n_col_blocks = (n + bs - 1) // bs
        # Wormhole throats = block means
        throats = np.zeros((n_row_blocks, n_col_blocks), dtype=np.float32)
        # Bridge connections = deltas between adjacent blocks
        bridges = []
        for i in range(n_row_blocks):
            for j in range(n_col_blocks):
                r0, r1 = i * bs, min((i + 1) * bs, m)
                c0, c1 = j * bs, min((j + 1) * bs, n)
                block = t[r0:r1, c0:c1]
                throats[i, j] = float(np.mean(block))
                if j > 0:
                    prev = t[r0:r1, (j - 1) * bs : c0]
                    if prev.size > 0 and block.size > 0:
                        delta = np.mean(block) - np.mean(prev)
                        bridges.append(delta)
        # Wormhole radius and charge
        r0 = float(np.std(throats))
        q = float(np.mean(np.abs(bridges))) if bridges else 0.0
        # Store throats + bridges
        buf = struct.pack("<IIff", m, n, r0, q)
        buf += _serialize(throats)
        bridges_arr = np.array(bridges, dtype=np.float16)
        buf += bridges_arr.tobytes()
        return bytes(buf), {
            "shape": tensor.shape,
            "block_size": bs,
            "n_row_blocks": n_row_blocks,
            "n_col_blocks": n_col_blocks,
            "n_bridges": len(bridges),
            "wormhole_radius": r0,
            "charge": q,
        }

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        m, n, r0, q = struct.unpack_from("<IIff", data, 0)
        pos = 16
        bs = metadata.get("block_size", 32)
        nr = metadata.get("n_row_blocks", 1)
        nc = metadata.get("n_col_blocks", 1)
        throats = np.frombuffer(
            data[pos : pos + nr * nc * 4], dtype=np.float32
        ).reshape(nr, nc)
        pos += nr * nc * 4
        n_bridges = metadata.get("n_bridges", 0)
        if n_bridges > 0:
            bridges = np.frombuffer(
                data[pos : pos + n_bridges * 2], dtype=np.float16
            ).astype(np.float64)
        else:
            bridges = np.array([], dtype=np.float64)
        recon = np.zeros((m, n), dtype=np.float64)
        bridge_idx = 0
        for i in range(nr):
            for j in range(nc):
                r0b, r1b = i * bs, min((i + 1) * bs, m)
                c0b, c1b = j * bs, min((j + 1) * bs, n)
                val = float(throats[i, j])
                if j > 0 and bridge_idx < len(bridges):
                    val += bridges[bridge_idx]
                    bridge_idx += 1
                recon[r0b:r1b, c0b:c1b] = val
        return recon.astype(np.float32)
