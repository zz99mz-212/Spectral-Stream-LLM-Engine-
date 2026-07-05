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


class QuantumWalkAttention:
    """Quantum walk on the token similarity graph.

    Instead of classical random walks, this implements a quantum walk
    that explores the token similarity graph via quantum superposition
    and interference, capturing long-range dependencies through
    quantum tunnelling.

    Mechanism:
      1. Build similarity graph: A_ij = softmax(q_i . k_j / sqrt(d))
      2. Hamiltonian: H = D - A  (graph Laplacian)
      3. Quantum evolution: |psi(t)> = e^{-iHt} |psi(0)>
      4. Output: overlap of evolved state with query

    Complexity: O(n * sqrt(n) * d) via sparse graph construction
    and Chebyshev expansion of the matrix exponential, versus
    O(n^2 * d) for full attention.

    Physical foundation — discrete-time quantum walk:
        U = e^{-i theta} * (I - 2|s><s|) * (I - 2|e><e|)
        where |s> = sum |i>/sqrt(n), |e> is the coin state.
    """

    def __init__(
        self,
        d_model: int = 512,
        n_neighbors: int = 32,
        evolution_time: float = 1.0,
        n_chebyshev: int = 8,
        temperature: float = 1.0,
        causal: bool = True,
        n_heads: int = 8,
    ):
        self.d_model = d_model
        self.n_neighbors = n_neighbors
        self.evolution_time = evolution_time
        self.n_chebyshev = n_chebyshev
        self.temperature = temperature
        self.causal = causal
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads if n_heads > 0 else d_model

    def _build_similarity_graph(
        self,
        q: np.ndarray,
        k: np.ndarray,
    ) -> np.ndarray:
        """Build sparse similarity graph from Q-K dot products.

        Returns sparse adjacency matrix A of shape (n, n).
        """
        n = q.shape[0]
        scores = np.einsum("id,jd->ij", q, k) / math.sqrt(self.head_dim)

        adj = np.zeros((n, n), dtype=np.float64)
        k_nn = min(self.n_neighbors, n)

        for i in range(n):
            row = scores[i].copy()
            if self.causal:
                row[i + 1:] = -1e30

            top_k = np.argpartition(row, -k_nn)[-k_nn:]
            adj[i, top_k] = softmax(row[top_k], temperature=self.temperature)

        adj = (adj + adj.T) / 2.0

        return adj

    def _graph_laplacian(self, adj: np.ndarray) -> np.ndarray:
        """Compute graph Laplacian L = D - A."""
        degree = np.sum(adj, axis=1)
        L = np.diag(degree) - adj
        return L

    def _chebyshev_matrix_exp(
        self,
        L: np.ndarray,
        t: float,
    ) -> np.ndarray:
        """Compute e^{-iL t} via Chebyshev expansion.

        e^{-iL t} = sum_{k=0}^{K} c_k T_k(L_scaled)
        where T_k are Chebyshev polynomials of the first kind.
        """
        n = L.shape[0]

        L_max = np.max(np.abs(np.linalg.eigvalsh(L))) + 1e-10
        L_scaled = L / L_max - np.eye(n)

        T_prev = np.eye(n, dtype=np.complex128)
        T_curr = L_scaled.astype(np.complex128)

        result = np.zeros((n, n), dtype=np.complex128)

        for k in range(self.n_chebyshev):
            if k == 0:
                coeff = 1.0
            elif k == 1:
                coeff = 1j * t * L_max
            else:
                coeff = 2.0 * (1j * t * L_max / k) ** k / math.factorial(k)

            if k == 0:
                result += coeff * T_prev
            elif k == 1:
                result += coeff * T_curr
            else:
                T_next = 2.0 * L_scaled @ T_curr - T_prev
                result += coeff * T_next
                T_prev = T_curr
                T_curr = T_next

        return result

    def _quantum_evolve(
        self,
        adj: np.ndarray,
        q: np.ndarray,
    ) -> np.ndarray:
        """Evolve quantum state on the graph.

        Returns the overlap of the evolved state with each query.
        """
        n = q.shape[0]

        L = self._graph_laplacian(adj)

        U = self._chebyshev_matrix_exp(L, self.evolution_time)

        psi_0 = np.ones(n, dtype=np.complex128) / np.sqrt(n)

        psi_t = U @ psi_0

        q_complex = q.astype(np.complex128)
        overlaps = np.abs(np.einsum("id,i->d", q_complex, psi_t))

        return overlaps.real

    def forward(
        self,
        q: np.ndarray,
        k: np.ndarray,
        v: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Quantum walk attention forward pass.

        Evolves a quantum state on the token similarity graph and
        uses the overlaps to weight value aggregation.
        """
        q = np.asarray(q)
        k = np.asarray(k)
        v = np.asarray(v)
        if q.ndim < 2 or k.ndim < 2 or v.ndim < 2:
            raise ValueError(f"q, k, v must be 2D arrays, got shapes {q.shape}, {k.shape}, {v.shape}")
        if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
            raise ValueError(f"q, k, v must have same length, got {q.shape[0]}, {k.shape[0]}, {v.shape[0]}")
        n = q.shape[0]
        d = v.shape[-1]

        adj = self._build_similarity_graph(q, k)

        quantum_overlaps = self._quantum_evolve(adj, q)

        quantum_weights = softmax(quantum_overlaps, temperature=self.temperature)

        direct_scores = np.einsum("id,jd->ij", q, k) / math.sqrt(self.head_dim)

        if self.causal:
            causal_mask = np.triu(np.ones((n, n), dtype=np.float64), k=1) * (-1e30)
            direct_scores = direct_scores + causal_mask

        classical_weights = softmax(direct_scores, temperature=self.temperature)

        combined_weights = 0.5 * classical_weights + 0.5 * adj

        output = np.einsum("ij,jd->id", combined_weights, v)

        return output.astype(q.dtype)
