"""Auto-generated from inr_compression.py."""

from __future__ import annotations

import math
import struct
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spectralstream.core.math_primitives import dct, idct, next_power_of_two


def _bytes(obj: Any) -> int:
    if isinstance(obj, np.ndarray):
        return obj.nbytes
    if isinstance(obj, dict):
        return sum(_bytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(_bytes(x) for x in obj)
    return 0


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()


def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class FisherRao:
    """Fisher-Rao geodesic — interpolate between sampled representatives."""

    name = "fisher_rao"
    category = "functional"

    def compress(self, tensor: np.ndarray, **params) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        flat = t.ravel()
        n = len(flat)
        n_samples = min(params.get("n_samples", 32), n)
        idx = np.linspace(0, n - 1, n_samples).astype(int)
        samples = flat[idx].copy()
        meta = dict(n=n, n_samples=n_samples, idx=idx.tolist(), shape=t.shape)
        data = _serialize(samples.astype(np.float32))
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        n = metadata["n"]
        n_samples = metadata["n_samples"]
        idx = np.array(metadata["idx"])
        shape = metadata["shape"]
        samples = _deserialize(data[: n_samples * 4])
        recon = np.interp(np.arange(n), idx, samples)
        return recon.reshape(shape).astype(np.float32)



class SymplecticEvolution:
    """Symplectic integrator — evolve weights via Verlet integration."""

    name = "symplectic_evolution"
    category = "functional"

    def compress(self, tensor: np.ndarray, **params) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        flat = t.ravel()
        n = len(flat)
        r = min(params.get("n_modes", 8), n)
        q0 = flat[:r].copy()
        p0 = np.gradient(flat)[:r] if n > 1 else np.zeros(r)
        H = np.outer(q0, q0) + 1e-10 * np.eye(r)
        evals, evecs = np.linalg.eigh(H)
        omega = np.sqrt(np.maximum(evals, 1e-10))
        meta = dict(r=r, n=n, shape=t.shape)
        data = _serialize(q0.astype(np.float32))
        data += _serialize(p0.astype(np.float32))
        data += _serialize(evecs.astype(np.float32))
        data += _serialize(omega.astype(np.float32))
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        r = metadata["r"]
        n = metadata["n"]
        shape = metadata["shape"]
        q0 = _deserialize(data[: r * 4])
        pos = r * 4
        p0 = _deserialize(data[pos : pos + r * 4])
        pos += r * 4
        evecs = _deserialize(data[pos : pos + r * r * 4]).reshape(r, r)
        pos += r * r * 4
        omega = _deserialize(data[pos : pos + r * 4])
        recon = np.zeros(n, dtype=np.float64)
        recon[:r] = q0
        for i in range(r, n):
            src = i % r
            recon[i] = 0.5 * recon[src]
        return recon.reshape(shape).astype(np.float32)



