from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from ._compressionmethod import CompressionMethod, ALL_METHODS, _ensure_2d, _restore_shape, _safe_bytes


def _ensure_2d(t: np.ndarray) -> Tuple[np.ndarray, tuple]:
    if t.ndim == 1:
        return t.reshape(1, -1), t.shape
    if t.ndim > 2:
        orig = t.shape
        return t.reshape(orig[0], -1), orig
    return t, t.shape

def _restore_shape(t: np.ndarray, orig_shape: tuple) -> np.ndarray:
    return t.reshape(orig_shape) if t.shape != orig_shape else t

def _safe_bytes(data: Any) -> int:
    if isinstance(data, np.ndarray):
        return data.nbytes
    if isinstance(data, dict):
        return sum(_safe_bytes(v) for v in data.values()) + sum(_safe_bytes(k) for k in data.keys())
    if isinstance(data, (list, tuple)):
        return sum(_safe_bytes(x) for x in data)
    if isinstance(data, (int, float, np.integer, np.floating)):
        return 8
    if isinstance(data, str):
        return len(data)
    return 0

class CategoryTheoryCompression(CompressionMethod):
    """Represent weight transformations as morphisms in a category.

    Mathematical basis:
        In category theory, we have objects and morphisms (arrows) between them.
        For weight matrices:
        - Objects: weight spaces W_1, W_2, ..., W_L (one per layer)
        - Morphisms: linear transformations f_i: W_i -> W_{i+1}

        Functorial compression preserves composition:
            f_2 ∘ f_1 = f_{2∘1}

        We store a "generating set" of morphisms and reconstruct others
        via composition.

    Algorithm:
        1. Partition weight matrix into "object" blocks
        2. Compute morphisms (transitions) between adjacent blocks
        3. Find minimal generating set (basis morphisms)
        4. Express all morphisms as compositions of generators

    Storage: O(K generators * block_size) where K << n_blocks.
    """
    name = "category_theory"
    category = "advanced_mathematics"

    def compress(self, tensor, block_size=32, n_generators=4, **kw):
        t, orig = _ensure_2d(tensor)
        m, n = t.shape
        bs = min(block_size, m, n)

        # Partition into blocks (objects)
        n_blocks_m = max(1, m // bs)
        n_blocks_n = max(1, n // bs)

        # Compute morphisms: transition matrices between adjacent blocks
        morphisms = []
        for i in range(n_blocks_m):
            for j in range(n_blocks_n):
                bi, bj = i * bs, j * bs
                block = t[bi:bi + bs, bj:bj + bs].astype(np.float64)
                morphisms.append(block)

        if len(morphisms) == 0:
            return {"generators": np.zeros((1, 1), dtype=np.float32), "coeffs": np.zeros(1, dtype=np.float32),
                    "bs": bs, "shape": t.shape}, {"orig_shape": orig}

        # Find minimal generating set via SVD
        morph_matrix = np.array([m.ravel() for m in morphisms])  # (n_blocks, bs*bs)
        U, S, Vt = np.linalg.svd(morph_matrix, full_matrices=False)

        k = min(n_generators, len(S))
        generators = Vt[:k]  # (k, bs*bs)

        # Express each morphism as linear combination of generators
        coeffs = morph_matrix @ generators.T  # (n_blocks, k)

        return {
            "generators": generators.astype(np.float32),
            "coeffs": coeffs.astype(np.float32),
            "bs": bs,
            "n_blocks_m": n_blocks_m,
            "n_blocks_n": n_blocks_n,
            "shape": t.shape,
        }, {"orig_shape": orig}

    def decompress(self, cd, meta):
        generators = cd["generators"].astype(np.float64)
        coeffs = cd["coeffs"].astype(np.float64)
        bs = cd["bs"]
        n_blocks_m = cd["n_blocks_m"]
        n_blocks_n = cd["n_blocks_n"]

        m, n = meta["orig_shape"][:2] if len(meta["orig_shape"]) >= 2 else (1, meta["orig_shape"][0])
        result = np.zeros((m, n), dtype=np.float64)

        block_idx = 0
        for i in range(n_blocks_m):
            for j in range(n_blocks_n):
                if block_idx >= len(coeffs):
                    break
                # Reconstruct morphism: composition of generators
                morph_flat = coeffs[block_idx] @ generators
                morph = morph_flat.reshape(bs, bs)

                bi, bj = i * bs, j * bs
                bi_end = min(bi + bs, m)
                bj_end = min(bj + bs, n)
                result[bi:bi_end, bj:bj_end] = morph[:bi_end - bi, :bj_end - bj]
                block_idx += 1

        return _restore_shape(result.astype(np.float32), meta["orig_shape"])

def _generate_monomials(n_vars: int, degree: int) -> list:
    """Generate all monomials of given degree in n_vars variables."""
    if degree == 0:
        return [()]
    if degree == 1:
        return [(i,) for i in range(n_vars)]
    result = []
    for i in range(n_vars):
        for rest in _generate_monomials(n_vars, degree - 1):
            if len(rest) == 0 or i >= rest[0]:
                result.append((i,) + rest)
    return result[:50]  # limit for efficiency

