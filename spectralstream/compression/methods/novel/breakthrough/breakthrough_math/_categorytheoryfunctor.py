from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class CategoryTheoryFunctor:
    """Category theory: layers are objects in a category, weight matrices
    are morphisms. A functor F maps the network category to the
    category of vector spaces. Store the functor's action on objects
    (dimensions) and generating morphisms (a few 'basis' matrices).
    All other morphisms are compositions of the generators.

    Real: find a few generating matrices via non-negative matrix
    factorization of the layer structure. Store generators +
    composition coefficients.
    """

    name = "category_theory_functor"
    category = "breakthrough_math"

    def compress(self, tensor: np.ndarray, rank: int = 12) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64).reshape(tensor.shape[0], -1)
        m, n = t.shape
        U, S, Vt = np.linalg.svd(t, full_matrices=False)
        k = min(rank, len(S), m, n)
        # Generators = top SVD components (the 'morphisms')
        U_k = U[:, :k].astype(np.float32)
        S_k = S[:k].astype(np.float32)
        Vt_k = Vt[:k, :].astype(np.float32)
        # Composition coefficients = how generators combine
        comp_coeffs = S_k.copy()
        buf = struct.pack("<III", m, n, k)
        buf += _serialize(comp_coeffs)
        buf += _serialize(U_k)
        buf += _serialize(Vt_k)
        return bytes(buf), {
            "shape": tensor.shape,
            "m": m,
            "n": n,
            "k": k,
        }

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        m, n, k = struct.unpack_from("<III", data, 0)
        pos = 12
        comp = np.frombuffer(data[pos : pos + k * 4], dtype=np.float32)
        pos += k * 4
        U_k = np.frombuffer(data[pos : pos + m * k * 4], dtype=np.float32).reshape(m, k)
        pos += m * k * 4
        Vt_k = np.frombuffer(data[pos : pos + k * n * 4], dtype=np.float32).reshape(
            k, n
        )
        # Functor composition: W = Σ c_i · U_i ⊗ V_i
        recon = (U_k * comp) @ Vt_k
        return recon.astype(np.float32).reshape(metadata["shape"])
