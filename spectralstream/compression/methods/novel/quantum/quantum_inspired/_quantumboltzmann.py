from __future__ import annotations

import math
import struct
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class QuantumBoltzmann:
    """Restricted Boltzmann machine simulated via quantum sampling.
    Uses Gibbs sampling with quantum-inspired thermal states.
    """

    name = "quantum_boltzmann"
    category = "quantum_compression"

    def compress(
        self,
        tensor: np.ndarray,
        n_visible: int = 16,
        n_hidden: int = 8,
        n_gibbs_steps: int = 10,
        temperature: float = 1.0,
    ) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64).ravel()
        n = len(t)
        nv = min(n_visible, n)
        nh = min(n_hidden, nv)
        W = np.random.randn(nv, nh).astype(np.float64) * 0.1
        vb = np.zeros(nv, dtype=np.float64)
        hb = np.zeros(nh, dtype=np.float64)
        data_batches = t[: n * (nv // nv)].reshape(-1, nv)
        n_batches = data_batches.shape[0]
        for _ in range(n_gibbs_steps):
            pos_h = 1.0 / (1.0 + np.exp(-(data_batches @ W + hb) / temperature))
            h_samples = (np.random.random(pos_h.shape) < pos_h).astype(np.float64)
            neg_v = 1.0 / (1.0 + np.exp(-(h_samples @ W.T + vb) / temperature))
            v_samples = (np.random.random(neg_v.shape) < neg_v).astype(np.float64)
            pos_h_grad = data_batches.T @ pos_h / n_batches
            neg_h_grad = v_samples.T @ h_samples / n_batches
            W += 0.01 * (pos_h_grad - neg_h_grad)
            vb += 0.01 * (data_batches - v_samples).mean(axis=0)
            hb += 0.01 * (pos_h - h_samples).mean(axis=0)
        W_f32 = W.astype(np.float32)
        vb_f32 = vb.astype(np.float32)
        hb_f32 = hb.astype(np.float32)
        meta = dict(
            shape=tensor.shape,
            n=n,
            nv=nv,
            nh=nh,
        )
        data = struct.pack("<II", nv, nh)
        data += _serialize(W_f32)
        data += _serialize(vb_f32)
        data += _serialize(hb_f32)
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        n = metadata["n"]
        nv, nh = struct.unpack_from("<II", data, 0)
        pos = 8
        W = _deserialize(data[pos : pos + nv * nh * 4]).reshape(nv, nh)
        pos += nv * nh * 4
        vb = _deserialize(data[pos : pos + nv * 4])
        pos += nv * 4
        hb = _deserialize(data[pos : pos + nh * 4])
        h = (np.random.random(nh) < 0.5).astype(np.float64)
        for _ in range(5):
            v_prob = 1.0 / (1.0 + np.exp(-(h @ W.T + vb)))
            v = (np.random.random(nv) < v_prob).astype(np.float64)
            h_prob = 1.0 / (1.0 + np.exp(-(v @ W + hb)))
            h = (np.random.random(nh) < h_prob).astype(np.float64)
        recon = np.tile(v, math.ceil(n / nv))[:n]
        return recon.reshape(shape).astype(np.float32)
