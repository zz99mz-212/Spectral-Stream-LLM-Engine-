from __future__ import annotations

import math
import struct
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class QuantumBayes:
    """Quantum Bayesian inference for weight distributions. Density matrix
    formulation: ρ = Σ p_i |ψ_i⟩⟨ψ_i|. Estimate posterior via quantum likelihood.
    """

    name = "quantum_bayes"
    category = "quantum_compression"

    def compress(
        self,
        tensor: np.ndarray,
        n_components: int = 8,
        n_prior_samples: int = 100,
    ) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64).ravel()
        n = len(t)
        k = min(n_components, n)
        mean = float(t.mean())
        std = float(t.std()) + 1e-10
        prior_samples = np.random.randn(n_prior_samples, k).astype(np.float64)
        prior_samples = prior_samples / np.linalg.norm(
            prior_samples, axis=1, keepdims=True
        )
        likelihoods = np.zeros(n_prior_samples, dtype=np.float64)
        for s in range(n_prior_samples):
            proj = prior_samples[s] @ t[:k]
            likelihoods[s] = math.exp(-0.5 * abs(proj - mean) ** 2 / std**2)
        posterior = np.exp(likelihoods - np.max(likelihoods))
        posterior /= posterior.sum() + 1e-30
        best_idx = np.argmax(posterior)
        eigen_vals = prior_samples[best_idx].astype(np.float32)
        eigen_vecs = np.eye(k, dtype=np.float32)
        meta = dict(
            shape=tensor.shape,
            n=n,
            k=k,
            mean=mean,
            std=std,
        )
        data = struct.pack("<Iff", k, mean, std)
        data += _serialize(eigen_vals)
        data += _serialize(eigen_vecs)
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        n = metadata["n"]
        k, mean, std = struct.unpack_from("<Iff", data, 0)
        pos = 12
        eigen_vals = _deserialize(data[pos : pos + k * 4])
        pos += k * 4
        eigen_vecs = _deserialize(data[pos : pos + k * k * 4]).reshape(k, k)
        base = np.zeros(k, dtype=np.float64)
        for i in range(k):
            base += eigen_vals[i] * eigen_vecs[:, i]
        recon = np.tile(base, math.ceil(n / k))[:n]
        recon = mean + (recon - recon.mean()) / (recon.std() + 1e-10) * std
        return recon.reshape(shape).astype(np.float32)
