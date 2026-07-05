from __future__ import annotations

import math
import struct
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class QuantumGradient:
    """Quantum gradient estimation: parameter shift rule for compression
    optimization. Estimate gradient of compression quality w.r.t. parameters
    using quantum circuit shift.
    """

    name = "quantum_gradient"
    category = "quantum_compression"

    def compress(
        self,
        tensor: np.ndarray,
        block_size: int = 64,
        n_shift_params: int = 8,
        learning_rate: float = 0.01,
        n_opt_steps: int = 10,
    ) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64).ravel()
        n = len(t)
        k = min(n_shift_params, n // block_size)
        params = np.random.randn(k).astype(np.float64) * 0.1
        padded_n = int(math.ceil(n / block_size) * block_size)
        padded = np.zeros(padded_n, dtype=np.float64)
        padded[:n] = t
        blocks = padded.reshape(-1, block_size)
        n_blocks = blocks.shape[0]
        scales = np.ones(n_blocks, dtype=np.float64)
        for _ in range(n_opt_steps):
            for i in range(min(k, n_blocks)):
                shift = math.pi / 2.0
                plus = params[i] + shift
                minus = params[i] - shift
                q_plus = blocks[i] * math.cos(plus) + blocks[
                    min(i + 1, n_blocks - 1)
                ] * math.sin(plus)
                q_minus = blocks[i] * math.cos(minus) + blocks[
                    min(i + 1, n_blocks - 1)
                ] * math.sin(minus)
                loss_plus = float(np.abs(q_plus - blocks[i].mean()).mean())
                loss_minus = float(np.abs(q_minus - blocks[i].mean()).mean())
                grad = (loss_plus - loss_minus) / (2.0 * shift)
                params[i] -= learning_rate * grad
            scales = 1.0 / (1.0 + np.exp(-params[:n_blocks]))
            # Prevent scale explosion
            scales = np.clip(scales, 0.01, 10.0)
        amax = np.max(np.abs(blocks), axis=1, keepdims=True)
        amax_scale = np.where(amax > 1e-8, amax / 127.0, 1.0)
        final_scales = (amax_scale.ravel() * scales).astype(np.float32)
        quantized = np.clip(
            np.round(blocks / final_scales[:, np.newaxis]), -128, 127
        ).astype(np.int8)
        meta = dict(
            shape=tensor.shape,
            n=n,
            block_size=block_size,
            n_blocks=n_blocks,
        )
        data = struct.pack("<II", n, block_size)
        data += _serialize(final_scales)
        data += quantized.tobytes()
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        n, block_size = struct.unpack_from("<II", data, 0)
        pos = 8
        n_blocks = int(math.ceil(n / block_size))
        scales = _deserialize(data[pos : pos + n_blocks * 4])
        pos += n_blocks * 4
        quantized = (
            np.frombuffer(data[pos : pos + n_blocks * block_size], dtype=np.int8)
            .reshape(-1, block_size)
            .astype(np.float32)
        )
        recon = (quantized * scales[:, np.newaxis]).ravel()[:n]
        return recon.reshape(shape).astype(np.float32)
