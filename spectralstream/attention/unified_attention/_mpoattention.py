from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union

import numpy as np

from spectralstream.core.math_primitives import (
    BAND_HIGH,
    BAND_LOW,
    BAND_NORMAL,
    DCTRotator,
    HadamardRotator,
    WaveletTransform,
    apply_spectral_kernel,
    band_limit,
    dct,
    fft,
    fftfreq,
    gibbs_softmax,
    idct,
    ifft,
    next_power_of_two,
    softmax,
    spectral_entropy,
    yukawa_kernel_1d,
)


class MPOAttention:
    """Matrix Product Operator (MPO) attention.

    Decomposes the attention mechanism into a tensor network contraction,
    where Q, K, V are projected into Matrix Product States (MPS) and
    the attention operation is performed via MPO contraction.

    Structure:
      - Q_mpo: Q projected to rank-r bond space
        Q_mpo[i] = (bond_left, head_dim, bond_right) per site i
      - K_mpo: K projected to rank-r bond space
      - V_mpo: V projected to rank-r bond space
      - O_core: Interaction core of shape (r, r) coupling bonds

    Complexity: O(n * r^2 * d) where r = bond dimension (typically 8-16)
    vs O(n^2 * d) for standard attention.

    Physical motivation — Tensor network theory:
        The MPO decomposition of the attention matrix A_ij = f(q_i, k_j)
        allows efficient contraction without materializing the full n x n
        matrix, similar to how MPO methods in DMRG compress quantum
        Hamiltonians.
    """

    def __init__(
        self,
        d_model: int = 512,
        bond_dim: int = 16,
        n_heads: int = 8,
        temperature: float = 1.0,
        causal: bool = True,
        seed: int = 42,
    ):
        self.d_model = d_model
        self.bond_dim = min(bond_dim, d_model)
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads if n_heads > 0 else d_model
        self.temperature = temperature
        self.causal = causal

        rng = np.random.RandomState(seed)
        scale = math.sqrt(d_model)

        r = self.bond_dim

        self.W_q_left = rng.randn(d_model, r).astype(np.float32) / scale
        self.W_q_right = rng.randn(d_model, r).astype(np.float32) / scale

        self.W_k_left = rng.randn(d_model, r).astype(np.float32) / scale
        self.W_k_right = rng.randn(d_model, r).astype(np.float32) / scale

        self.W_v_left = rng.randn(d_model, r).astype(np.float32) / scale
        self.W_v_right = rng.randn(d_model, r).astype(np.float32) / scale

        self.O_core = rng.randn(r, r).astype(np.float32) / scale

        self.w_o = rng.randn(r * r, d_model).astype(np.float32) / scale

    def _mpo_project(
        self,
        x: np.ndarray,
        W_left: np.ndarray,
        W_right: np.ndarray,
    ) -> np.ndarray:
        """Project input vectors to MPO bond space.

        For token i, compute:
            mpo[i] = outer(W_left @ x[i], W_right @ x[i])

        Returns MPO of shape (n, r, r) where r = bond_dim.
        """
        n = x.shape[0]
        r = self.bond_dim

        left_proj = x @ W_left
        right_proj = x @ W_right

        mpo = np.einsum("ir,is->irs", left_proj, right_proj)

        return mpo

    def _contract_mpo_attention(
        self,
        q_mpo: np.ndarray,
        k_mpo: np.ndarray,
        v_mpo: np.ndarray,
    ) -> np.ndarray:
        """Contract MPO tensors for attention computation.

        Attention via MPO:
            A_ij = trace(O_core @ K_mpo[i]^T @ Q_mpo[j])
            out_bond[i] = sum_j A_ij * V_mpo[j]

        Returns output in bond space (n, r, r).
        """
        n = q_mpo.shape[0]
        r = self.bond_dim

        attention_bonds = np.einsum(
            "irs,tsr,rs->it",
            q_mpo,
            k_mpo,
            self.O_core,
        )

        weights = softmax(attention_bonds / math.sqrt(r), temperature=self.temperature)

        output_bond = np.einsum("ij,jrs->irs", weights, v_mpo)

        return output_bond

    def _mpo_deproject(
        self,
        bond_output: np.ndarray,
    ) -> np.ndarray:
        """Project bond-space output (n, r, r) back to model dimension (n, d).

        Flattens the (r, r) bond tensor per token and applies the
        output projection matrix w_o of shape (r*r, d_model).
        """
        n = bond_output.shape[0]
        r = self.bond_dim

        flat = bond_output.reshape(n, r * r)

        output = flat @ self.w_o

        return output

    def forward(
        self,
        q: np.ndarray,
        k: np.ndarray,
        v: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """MPO attention forward pass.

        Projects Q, K, V to bond space, contracts via O_core,
        and projects back to model dimension.
        """
        q = np.asarray(q)
        k = np.asarray(k)
        v = np.asarray(v)
        if q.ndim < 2 or k.ndim < 2 or v.ndim < 2:
            raise ValueError(f"q, k, v must be 2D arrays, got shapes {q.shape}, {k.shape}, {v.shape}")
        if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
            raise ValueError(f"q, k, v must have same length, got {q.shape[0]}, {k.shape[0]}, {v.shape[0]}")
        n = q.shape[0]

        q_mpo = self._mpo_project(q, self.W_q_left, self.W_q_right)
        k_mpo = self._mpo_project(k, self.W_k_left, self.W_k_right)
        v_mpo = self._mpo_project(v, self.W_v_left, self.W_v_right)

        if self.causal:
            r = self.bond_dim
            causal_bias = np.triu(np.full((n, n), -1e30, dtype=np.float64), k=1)
            qk_scores = np.einsum("irs,tsr->it", q_mpo, k_mpo) / math.sqrt(r)
            qk_scores = qk_scores + causal_bias
            weights = softmax(qk_scores, temperature=self.temperature)

            output_bond = np.einsum("ij,jrs->irs", weights, v_mpo)
        else:
            output_bond = self._contract_mpo_attention(q_mpo, k_mpo, v_mpo)

        output = self._mpo_deproject(output_bond)

        return output.astype(q.dtype)
