"""Auto-generated from _class_wrappers.py."""

from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()


def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()


class ResonanceModes:
    """Resonant mode decomposition via SVD with energy-based truncation."""

    name = "resonance_modes"
    category = "physics"

    def compress(self, tensor: np.ndarray, rank: int = None) -> Tuple[bytes, dict]:
        from spectralstream.compression.methods.physics.quantum import DensityMatrix

        return DensityMatrix().compress(tensor, rank)

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        from spectralstream.compression.methods.physics.quantum import DensityMatrix

        return DensityMatrix().decompress(data, metadata)


class ResonanceCompression:
    """Resonance: eigenvalue-based SVD mode selection by energy."""

    name = "resonance_compression"
    category = "physics"

    def compress(self, tensor: np.ndarray, rank: int = None) -> Tuple[bytes, dict]:
        from spectralstream.compression.methods.physics.quantum import DensityMatrix

        return DensityMatrix().compress(tensor, rank)

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        from spectralstream.compression.methods.physics.quantum import DensityMatrix

        return DensityMatrix().decompress(data, metadata)


class GramResonanceDecomposition:
    """Resonance mode decomposition via Gram matrix eigenanalysis.
    Computes eigendecomposition of W^T W to find resonant modes capturing
    coherent interaction patterns. Stores projections and eigenvectors."""

    name = "gram_resonance_decomposition"
    category = "physics"

    def compress(self, tensor: np.ndarray, n_modes: int = 16) -> Tuple[bytes, dict]:
        import struct

        orig_shape = tensor.shape
        mat = tensor.reshape(tensor.shape[0], -1).astype(np.float64)
        m, n = mat.shape

        gram = mat.T @ mat / max(m, 1)
        eigenvalues, eigenvectors = np.linalg.eigh(gram)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        total_energy = float(np.sum(np.maximum(eigenvalues, 0)))
        if total_energy > 0:
            cumulative = np.cumsum(np.maximum(eigenvalues, 0)) / total_energy
            k = int(np.searchsorted(cumulative, 0.95)) + 1
            k = min(k, n_modes, len(eigenvalues))
        else:
            k = min(n_modes, len(eigenvalues))
        k = max(1, k)

        eigenvalues_k = eigenvalues[:k].astype(np.float32)
        eigenvectors_k = eigenvectors[:, :k].astype(np.float32)
        projections = (mat @ eigenvectors[:, :k]).astype(np.float32)

        header = struct.pack("<III", m, n, k)
        data = (
            header
            + eigenvalues_k.tobytes()
            + eigenvectors_k.tobytes()
            + projections.tobytes()
        )
        meta = {
            "shape": orig_shape,
            "n_modes": k,
            "m": m,
            "n_val": n,
        }
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        import struct

        shape = metadata["shape"]
        k = metadata["n_modes"]
        m = metadata["m"]
        n_val = metadata["n_val"]

        header_size = struct.calcsize("<III")
        m_stored, n_stored, k_stored = struct.unpack_from("<III", data, 0)
        pos = header_size

        _evals = np.frombuffer(data[pos : pos + k * 4], dtype=np.float32)
        pos += k * 4

        eigenvectors = np.frombuffer(
            data[pos : pos + n_val * k * 4], dtype=np.float32
        ).reshape(n_val, k)
        pos += n_val * k * 4

        projections = np.frombuffer(
            data[pos : pos + m * k * 4], dtype=np.float32
        ).reshape(m, k)

        reconstructed = (projections @ eigenvectors.T).astype(np.float64)
        return reconstructed.reshape(shape).astype(np.float32)
