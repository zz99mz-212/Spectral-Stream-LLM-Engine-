from __future__ import annotations

import math
import struct
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class QuantumGenerative:
    """Quantum circuit Born machine: p(x) = |⟨x|ψ(θ)⟩|².
    Learn variational parameters θ that generate the weight distribution.
    """

    name = "quantum_generative"
    category = "quantum_compression"

    def compress(
        self,
        tensor: np.ndarray,
        n_qubits: int = 8,
        n_layers: int = 4,
        n_samples: int = 1000,
    ) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64).ravel()
        n = len(t)
        q = min(n_qubits, int(math.log2(n)) + 1)
        d = 1 << q
        theta = np.random.randn(n_layers, q, 3).astype(np.float64) * 0.1
        hist, edges = np.histogram(t, bins=d, density=True)
        target_probs = hist / (hist.sum() + 1e-30)
        for _ in range(50):
            psi = np.ones(d, dtype=np.complex128) / math.sqrt(d)
            for layer in range(n_layers):
                for qubit in range(q):
                    angle = theta[layer, qubit, 0]
                    rot = np.array(
                        [
                            [math.cos(angle), -math.sin(angle)],
                            [math.sin(angle), math.cos(angle)],
                        ],
                        dtype=np.complex128,
                    )
                    for j in range(0, d, 2):
                        psi[[j, j + 1]] = rot @ psi[[j, j + 1]]
                for qubit in range(q - 1):
                    for j in range(0, d, 2 << qubit):
                        for k in range(1 << qubit):
                            idx1 = j + k + (1 << qubit)
                            idx2 = j + k
                            theta_l = theta[layer, qubit, 1]
                            cr = math.cos(theta_l)
                            sr = math.sin(theta_l)
                            psi[idx1], psi[idx2] = (
                                cr * psi[idx1] + sr * psi[idx2],
                                -sr * psi[idx1] + cr * psi[idx2],
                            )
            probs = (np.abs(psi) ** 2).real
            probs /= probs.sum() + 1e-30
            grad = target_probs - probs
            for layer in range(n_layers):
                theta[layer, :, 0] += 0.01 * grad[:q]
                theta[layer, :, 1] += 0.01 * grad[:q]
        theta_f32 = theta.astype(np.float32)
        meta = dict(
            shape=tensor.shape,
            n=n,
            q=q,
            d=d,
            n_layers=n_layers,
            t_min=float(t.min()),
            t_max=float(t.max()),
        )
        data = struct.pack("<IIff", q, n_layers, float(t.min()), float(t.max()))
        data += _serialize(theta_f32)
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        n = metadata["n"]
        q, n_layers, t_min, t_max = struct.unpack_from("<IIff", data, 0)
        d = 1 << q
        pos = 16
        theta = _deserialize(data[pos : pos + n_layers * q * 3 * 4]).reshape(
            n_layers, q, 3
        )
        psi = np.ones(d, dtype=np.complex128) / math.sqrt(d)
        for layer in range(n_layers):
            for qubit in range(q):
                angle = float(theta[layer, qubit, 0])
                rot = np.array(
                    [
                        [math.cos(angle), -math.sin(angle)],
                        [math.sin(angle), math.cos(angle)],
                    ],
                    dtype=np.complex128,
                )
                for j in range(0, d, 2):
                    psi[[j, j + 1]] = rot @ psi[[j, j + 1]]
            for qubit in range(q - 1):
                for j in range(0, d, 2 << qubit):
                    for k in range(1 << qubit):
                        idx1 = j + k + (1 << qubit)
                        idx2 = j + k
                        tl = float(theta[layer, qubit, 1])
                        cr = math.cos(tl)
                        sr = math.sin(tl)
                        psi[idx1], psi[idx2] = (
                            cr * psi[idx1] + sr * psi[idx2],
                            -sr * psi[idx1] + cr * psi[idx2],
                        )
        probs = (np.abs(psi) ** 2).real
        probs /= probs.sum() + 1e-30
        samples = np.random.choice(d, size=n, p=probs)
        recon = t_min + (samples / d) * (t_max - t_min)
        return recon.reshape(shape).astype(np.float32)
