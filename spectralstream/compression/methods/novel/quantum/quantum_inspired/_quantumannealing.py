from __future__ import annotations

import math
import struct
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class QuantumAnnealing:
    """Simulated quantum annealing for optimal bit allocation.
    Transverse field Γ(t) drives exploration. Hamiltonian H = H_0 + Γ(t) H_1.
    """

    name = "quantum_annealing"
    category = "quantum_compression"

    def compress(
        self,
        tensor: np.ndarray,
        n_bits: int = 4,
        n_anneal_steps: int = 30,
        gamma_init: float = 2.0,
    ) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64).ravel()
        n = len(t)
        t_min, t_max = float(t.min()), float(t.max())
        rng = float(t_max - t_min)
        if rng < 1e-10:
            rng = 1.0
        levels = 1 << n_bits
        scale = rng / (levels - 1)
        quantized = np.zeros(n, dtype=np.int32)
        cur = np.clip(np.round((t - t_min) / scale), 0, levels - 1).astype(np.int32)
        best = cur.copy()
        best_energy = float("inf")
        for step in range(n_anneal_steps):
            gamma = gamma_init * (1.0 - step / n_anneal_steps)
            flip_prob = 0.5 * (1.0 + math.tanh(gamma))
            for i in range(n):
                if np.random.random() < flip_prob:
                    delta = np.random.choice([-1, 1])
                    cur[i] = np.clip(cur[i] + delta, 0, levels - 1)
            recon = t_min + cur * scale
            err = float(np.mean((t - recon) ** 2))
            tunneling_term = gamma * float(
                np.mean(np.abs(np.diff(cur.astype(np.float64))))
            )
            energy = err + tunneling_term
            if energy < best_energy:
                best = cur.copy()
                best_energy = energy
        meta = dict(
            shape=tensor.shape,
            n=n,
            t_min=t_min,
            scale=scale,
            n_bits=n_bits,
            levels=levels,
        )
        packed = bytearray()
        bits_per_val = n_bits
        for i in range(0, n, 8):
            chunk = best[i : i + 8]
            for shift in range(bits_per_val):
                byte = 0
                for j, val in enumerate(chunk):
                    if val & (1 << shift):
                        byte |= 1 << j
                packed.append(byte)
        data = struct.pack("<fII", t_min, n, n_bits)
        data += bytes(packed)
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        n = metadata["n"]
        t_min = metadata["t_min"]
        scale = metadata["scale"]
        n_bits = metadata["n_bits"]
        _, _, nb = struct.unpack_from("<fII", data, 0)
        packed = data[12:]
        vals = np.zeros(n, dtype=np.int32)
        bits_per_val = n_bits
        for i in range(0, n, 8):
            chunk_size = min(8, n - i)
            for shift in range(bits_per_val):
                byte_idx = (i // 8) * bits_per_val + shift
                if byte_idx < len(packed):
                    byte_val = packed[byte_idx]
                    for j in range(chunk_size):
                        if byte_val & (1 << j):
                            vals[i + j] |= 1 << shift
        recon = t_min + vals * scale
        return recon.reshape(shape).astype(np.float32)
