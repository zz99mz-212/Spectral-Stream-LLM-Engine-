"""Auto-generated from _class_wrappers.py."""

from __future__ import annotations

import gc
import math
import struct
from typing import Any, Dict, Tuple

import numpy as np

from spectralstream.core.math_primitives import (
    dct,
    idct,
    fwht,
    ifwht,
    next_power_of_two,
    LloydMaxQuantizer,
)


class NF4:
    """Normal Float 4-bit (NF4) quantization — QLoRA-style."""

    name = "nf4"
    category = "quantization"

    def compress(self, tensor: np.ndarray, block_size: int = 64) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float32)
        s = t.shape
        ndim = t.ndim
        f = t.reshape(-1, s[-1])
        nr, nc = f.shape
        p = (block_size - nc % block_size) % block_size
        if p:
            f = np.pad(f, ((0, 0), (0, p)))
        nb = f.shape[1] // block_size
        b = f.reshape(nr, nb, block_size)
        sc = np.maximum(np.max(np.abs(b), axis=2, keepdims=True), 1e-10)
        norm = np.clip(b / sc, -1.0, 1.0)
        levels = np.array(
            [
                -1.0,
                -0.70756877,
                -0.54220910,
                -0.41681885,
                -0.31090474,
                -0.21594631,
                -0.12734098,
                -0.04209538,
                0.04209538,
                0.12734098,
                0.21594631,
                0.31090474,
                0.41681885,
                0.54220910,
                0.70756877,
                1.0,
            ],
            dtype=np.float32,
        )
        idx = np.argmin(
            np.abs(norm[:, :, :, None] - levels[None, None, None, :]), axis=3
        ).astype(np.uint8)
        even_bs = block_size if block_size % 2 == 0 else block_size - 1
        pairs = idx.reshape(nr, nb, even_bs // 2, 2)
        packed = (pairs[..., 0].astype(np.uint8) << 4) | pairs[..., 1].astype(np.uint8)
        meta = dict(
            shape=tensor.shape,
            block_size=block_size,
            nc=nc,
            n_elements=tensor.size,
            ndim=ndim,
        )
        data = (
            struct.pack("<II", nr, nb)
            + sc.astype(np.float32).tobytes()
            + packed.tobytes()
        )
        del t, f, b, sc, norm, idx, pairs
        gc.collect()
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        block_size = metadata["block_size"]
        nr, nb = struct.unpack_from("<II", data, 0)
        pos = 8
        sc = np.frombuffer(data[pos : pos + nr * nb * 4], dtype=np.float32).reshape(
            nr, nb, 1
        )
        pos += nr * nb * 4
        even_bs = block_size if block_size % 2 == 0 else block_size - 1
        packed = np.frombuffer(
            data[pos : pos + nr * nb * even_bs // 2], dtype=np.uint8
        ).reshape(nr, nb, even_bs // 2)
        levels = np.array(
            [
                -1.0,
                -0.70756877,
                -0.54220910,
                -0.41681885,
                -0.31090474,
                -0.21594631,
                -0.12734098,
                -0.04209538,
                0.04209538,
                0.12734098,
                0.21594631,
                0.31090474,
                0.41681885,
                0.54220910,
                0.70756877,
                1.0,
            ],
            dtype=np.float32,
        )
        uv_e = (packed.astype(np.uint16) >> 4).astype(np.uint8)
        uv_o = (packed & 0x0F).astype(np.uint8)
        uv_d = np.empty((nr, nb, even_bs), dtype=np.uint8)
        uv_d[..., 0::2] = uv_e
        uv_d[..., 1::2] = uv_o
        if block_size > even_bs:
            uv_d = np.pad(uv_d, ((0, 0), (0, 0), (0, 1)), mode="edge")
        d = sc * levels[uv_d]
        n_elements = metadata["n_elements"]
        flat = d.reshape(nr, nb * block_size)
        nc = metadata.get("nc", n_elements // nr if n_elements > 0 else shape[-1])
        recon = flat[:, :nc].reshape(shape)
        del sc, packed, uv_e, uv_o, uv_d, d, flat
        gc.collect()
        return recon.astype(np.float32)
