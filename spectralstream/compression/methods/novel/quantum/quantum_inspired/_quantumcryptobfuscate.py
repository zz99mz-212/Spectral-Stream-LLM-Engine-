from __future__ import annotations

import math
import struct
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class QuantumCryptObfuscate:
    """Quantum key distribution-inspired weight obfuscation. BB84 encoding:
    encode weights in conjugate bases (computational | Hadamard).
    """

    name = "quantum_crypt_obfuscate"
    category = "quantum_compression"

    def compress(
        self,
        tensor: np.ndarray,
        block_size: int = 64,
        n_bit_encodings: int = 2,
    ) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64).ravel()
        n = len(t)
        rng = np.random.RandomState(42)
        bases = rng.randint(0, 2, size=n)
        keys = rng.randint(0, 256, size=n, dtype=np.uint8)
        encoded = np.zeros(n, dtype=np.float64)
        for i in range(n):
            val = t[i]
            if bases[i] == 0:
                encoded[i] = val
            else:
                encoded[i] = val * (1.0 if keys[i] % 2 == 0 else -1.0)
        padded_n = int(math.ceil(n / block_size) * block_size)
        padded = np.zeros(padded_n, dtype=np.float64)
        padded[:n] = encoded
        blocks = padded.reshape(-1, block_size)
        amax = np.max(np.abs(blocks), axis=1, keepdims=True)
        scales = np.where(amax > 1e-8, amax / 127.0, 1.0)
        quantized = np.clip(np.round(blocks / scales), -128, 127).astype(np.int8)
        meta = dict(
            shape=tensor.shape,
            n=n,
            block_size=block_size,
        )
        data = struct.pack("<II", n, block_size)
        data += bases.astype(np.uint8).tobytes()
        data += keys.tobytes()
        data += scales.astype(np.float32).tobytes()
        data += quantized.tobytes()
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        n, block_size = struct.unpack_from("<II", data, 0)
        pos = 8
        bases = np.frombuffer(data[pos : pos + n], dtype=np.uint8)
        pos += n
        keys = np.frombuffer(data[pos : pos + n], dtype=np.uint8)
        pos += n
        n_blocks = int(math.ceil(n / block_size))
        scales = _deserialize(data[pos : pos + n_blocks * 4]).reshape(-1, 1)
        pos += n_blocks * 4
        quantized = (
            np.frombuffer(data[pos : pos + n_blocks * block_size], dtype=np.int8)
            .reshape(-1, block_size)
            .astype(np.float32)
        )
        encoded = (quantized * scales).ravel()[:n]
        decoded = np.zeros(n, dtype=np.float64)
        for i in range(n):
            val = encoded[i]
            if bases[i] == 0:
                decoded[i] = val
            else:
                decoded[i] = val * (1.0 if keys[i] % 2 == 0 else -1.0)
        return decoded.reshape(shape).astype(np.float32)
