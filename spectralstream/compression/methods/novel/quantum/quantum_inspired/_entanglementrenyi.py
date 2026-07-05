from __future__ import annotations

import math
import struct
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class EntanglementRenyi:
    """Renyi entanglement entropy: S_α = 1/(1-α) log Tr(ρ^α).
    Store entropy profile across tensor bipartitions.
    """

    name = "entanglement_renyi"
    category = "quantum_compression"

    def _renyi_entropy(self, sv: np.ndarray, alpha: float = 2.0) -> float:
        probs = sv**2
        probs = probs / (probs.sum() + 1e-30)
        if abs(alpha - 1.0) < 1e-6:
            return -float((probs * np.log(probs + 1e-30)).sum())
        return float(1.0 / (1.0 - alpha) * math.log((probs**alpha).sum() + 1e-30))

    def compress(
        self,
        tensor: np.ndarray,
        alpha: float = 2.0,
        n_slices: int = 8,
        truncation: int = 8,
    ) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        orig_shape = t.shape
        flat = t.ravel()
        n = len(flat)
        k = min(truncation, n // n_slices)
        slices = np.array_split(flat, n_slices)
        entropies = np.zeros(n_slices, dtype=np.float64)
        for i, slc in enumerate(slices):
            if len(slc) > 1:
                _, sv, _ = np.linalg.svd(slc.reshape(1, -1), full_matrices=False)
                entropies[i] = self._renyi_entropy(sv, alpha)
        entropy_profile = entropies.astype(np.float32)
        slice_means = np.array([s.mean() for s in slices]).astype(np.float32)
        slice_stds = np.array([s.std() for s in slices]).astype(np.float32)
        meta = dict(
            shape=orig_shape,
            n=n,
            n_slices=n_slices,
            alpha=alpha,
            truncation=k,
        )
        data = struct.pack("<IfI", n_slices, alpha, k)
        data += _serialize(entropy_profile)
        data += _serialize(slice_means)
        data += _serialize(slice_stds)
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        n = metadata["n"]
        n_slices = int(metadata.get("n_slices", 8))
        alpha = metadata.get("alpha", 2.0)
        k = metadata.get("truncation", 8)
        pos = 12
        entropy_profile = _deserialize(data[pos : pos + n_slices * 4])
        pos += n_slices * 4
        slice_means = _deserialize(data[pos : pos + n_slices * 4])
        pos += n_slices * 4
        slice_stds = _deserialize(data[pos : pos + n_slices * 4])
        recon = np.zeros(n, dtype=np.float64)
        for i in range(n_slices):
            start = i * (n // n_slices)
            end = min((i + 1) * (n // n_slices), n)
            sz = end - start
            noise = (
                np.random.randn(sz) * slice_stds[i] * abs(math.sin(entropy_profile[i]))
            )
            recon[start:end] = slice_means[i] + noise
        return recon.reshape(shape).astype(np.float32)
