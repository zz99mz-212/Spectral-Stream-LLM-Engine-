"""Multi-stage residual quantization with progressive error compensation."""

from __future__ import annotations

import gc
from typing import Tuple

import numpy as np


class ResidualVectorQuant:
    """Multi-stage residual INT8 quantization.

    Stage 1: INT8 quantization of W -> W_q1, error1 = W - W_q1
    Stage 2: INT8 quantization of error1 -> W_q2, error2 = error1 - W_q2
    Stage 3: INT8 quantization of error2 -> W_q3
    Reconstruct: W ≈ W_q1 + W_q2 + W_q3
    """

    name = "residual_vector_quant"
    category = "quantization"

    def compress(
        self, tensor: np.ndarray, n_stages: int = 3, block_size: int = 128
    ) -> Tuple[bytes, dict]:
        flat = tensor.ravel().astype(np.float64)
        n = len(flat)
        stages = []
        residual = flat.copy()

        for stage in range(n_stages):
            n_blocks = (n + block_size - 1) // block_size
            stage_data = []
            stage_scales = []

            for b in range(n_blocks):
                start = b * block_size
                end = min(start + block_size, n)
                block = residual[start:end]
                amax = float(np.max(np.abs(block)))
                scale = amax / 127.0 if amax > 1e-8 else 1.0
                quantized = np.clip(np.round(block / scale), -128, 127).astype(np.int8)
                stage_data.append(quantized.tobytes())
                stage_scales.append(scale)
                residual[start:end] = block - quantized.astype(np.float64) * scale

            stages.append(
                {
                    "data": b"".join(stage_data),
                    "scales": np.array(stage_scales, dtype=np.float32).tobytes(),
                }
            )

        metadata = dict(
            n_elements=n,
            n_stages=n_stages,
            block_size=block_size,
            stages=stages,
            shape=tensor.shape,
        )
        return b"".join(s["data"] for s in stages), metadata

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        n = metadata["n_elements"]
        block_size = metadata["block_size"]
        stages = metadata["stages"]
        result = np.zeros(n, dtype=np.float64)

        for stage in stages:
            stage_data = stage["data"]
            scales = np.frombuffer(stage["scales"], dtype=np.float32).astype(np.float64)
            n_blocks = (n + block_size - 1) // block_size
            block_offset = 0
            for b in range(n_blocks):
                start = b * block_size
                end = min(start + block_size, n)
                block_len = end - start
                if block_offset + block_len > len(stage_data):
                    break
                raw = np.frombuffer(
                    stage_data[block_offset : block_offset + block_len], dtype=np.int8
                )
                block_offset += block_len
                result[start:end] += raw.astype(np.float64) * scales[b]

        return result.reshape(metadata["shape"]).astype(np.float32)
