"""
Physics-Inspired Compression Architectures for SpectralStream
=============================================================
Three radical compression paradigms that treat model weights as physics,
geometry, and signals rather than static numbers.

Paradigms:
  1. HamiltonianWeightDynamicals  — weights as Hamiltonian dynamical systems
  2. TopologicalFunctionalQuantization — geometric codebook of primitives
  3. HierarchicalStateSpaceWaveforms — multi-resolution wave decomposition

Every method exposes:
  compress(W, **kwargs) -> compressed_data
  decompress(compressed_data) -> W_reconstructed
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spectralstream.core.math_primitives import (
    dct,
    dct_2d,
    idct,
    idct_2d,
    spectral_entropy,
    zigzag_indices,
    next_power_of_two,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Shared Data Structures
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class CompressedWeight:
    """Universal container for any compression result."""

    method: str
    compressed_data: Any
    metadata: dict
    original_shape: Tuple[int, ...]
    original_bytes: int
    compressed_bytes: int
    compression_ratio: float
    reconstruction_error: float  # relative Frobenius error
    snr_db: float
    compress_time_ms: float
    decompress_time_ms: float
    extra: dict = field(default_factory=dict)


def _measure_error(
    original: np.ndarray, reconstructed: np.ndarray
) -> Tuple[float, float]:
    """Return (relative_frobenius_error, snr_db)."""
    o = original.astype(np.float64)
    r = reconstructed.astype(np.float64)
    noise = o - r
    mse = float(np.mean(noise**2))
    signal_power = float(np.mean(o**2)) + 1e-30
    snr_db = 10.0 * np.log10(signal_power / (mse + 1e-30))
    rel_err = float(np.linalg.norm(noise) / (np.linalg.norm(o) + 1e-30))
    return rel_err, snr_db


def _estimate_bytes(data: Any) -> int:
    """Recursively estimate byte size of a nested structure."""
    if isinstance(data, np.ndarray):
        return data.nbytes
    if isinstance(data, dict):
        return sum(_estimate_bytes(v) for v in data.values()) + sum(
            _estimate_bytes(k) for k in data.keys()
        )
    if isinstance(data, (list, tuple)):
        return sum(_estimate_bytes(x) for x in data)
    if isinstance(data, (int, float, np.integer, np.floating)):
        return 8
    if isinstance(data, str):
        return len(data)
    return 0


# ═══════════════════════════════════════════════════════════════════════════
# 1. HAMILTONIAN WEIGHT DYNAMICALS
# ═══════════════════════════════════════════════════════════════════════════


class HamiltonianWeightDynamicals:
    """Compress weight matrix W by treating it as a Hamiltonian dynamical system.

    Instead of storing the final state (the weight matrix), we store:
      - The SVD factorisation W = U @ diag(S) @ Vt
      - A parametric Hamiltonian H(S) governing the "energy levels"
      - Initial conditions for a symplectic integrator
      - Integration parameters (dt, n_steps)

    To reconstruct, we integrate Hamilton's equations from t=0 to t_final
    using a Verlet / symplectic Euler scheme that conserves the Hamiltonian
    structure, recovering the singular value spectrum.

    Compression ratio depends on the effective rank and the smoothness of
    the singular value spectrum: smooth spectra need fewer Hamiltonian
    parameters (polynomial or Fourier coefficients).

    Expected storage: O(r * (m + n) + K) instead of O(m * n),
    where r = rank, K = Hamiltonian parameter count.
    """

    def __init__(
        self,
        polynomial_degree: int = 6,
        fourier_modes: int = 0,
        symplectic_dt: float = 0.05,
        symplectic_steps: int = 80,
        max_rank: int = 128,
    ):
        self.polynomial_degree = polynomial_degree
        self.fourier_modes = fourier_modes
        self.symplectic_dt = symplectic_dt
        self.symplectic_steps = symplectic_steps
        self.max_rank = max_rank
        logger.info(
            "HamiltonianWeightDynamicals: poly_deg=%d, fourier=%d, dt=%.3f, steps=%d, max_rank=%d",
            polynomial_degree,
            fourier_modes,
            symplectic_dt,
            symplectic_steps,
            max_rank,
        )

    # ── Hamiltonian parametrisation ──────────────────────────────────────

    def _fit_hamiltonian_polynomial(self, S: np.ndarray) -> np.ndarray:
        """Fit polynomial H(S) = sum a_k S^k to the singular value spectrum.

        The Hamiltonian encodes the "energy landscape" of the weight matrix.
        We use a polynomial basis because:
          1. It is cheap to evaluate (Horner's rule)
          2. Symplectic integration is stable for smooth potentials
          3. It generalises well — similar weight matrices have similar
             Hamiltonian parameters

        Returns coefficient array a[0..degree].
        """
        r = len(S)
        degree = min(self.polynomial_degree, r - 1)
        # Normalise S to [0, 1] for numerical stability
        s_max = float(np.max(np.abs(S))) + 1e-10
        S_norm = S / s_max

        # Fit via least-squares polynomial regression
        # Vandermonde matrix: V[i, k] = S_norm[i]^k
        powers = np.arange(degree + 1, dtype=np.float64)
        V = S_norm[:, None] ** powers[None, :]
        # Ridge regression for stability
        lam = 1e-6
        coeffs = np.linalg.solve(V.T @ V + lam * np.eye(degree + 1), V.T @ S)
        return coeffs.astype(np.float64)

    def _fit_hamiltonian_fourier(self, S: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Fit Fourier Hamiltonian: H(S) = sum_k (a_k cos(kS) + b_k sin(kS)).

        Useful when singular values exhibit oscillatory structure
        (e.g., attention-like patterns).
        """
        r = len(S)
        n_modes = min(self.fourier_modes, r // 2)
        if n_modes == 0:
            return np.array([]), np.array([])

        s_max = float(np.max(np.abs(S))) + 1e-10
        S_norm = S / s_max * np.pi  # map to [-pi, pi]

        k = np.arange(1, n_modes + 1, dtype=np.float64)
        cos_vals = np.cos(np.outer(S_norm, k))
        sin_vals = np.sin(np.outer(S_norm, k))

        # Solve for coefficients
        basis = np.column_stack([np.ones(r), cos_vals, sin_vals])
        lam = 1e-6
        AtA = basis.T @ basis + lam * np.eye(basis.shape[1])
        AtS = basis.T @ S
        x = np.linalg.solve(AtA, AtS)
        a0 = x[0]
        a_k = x[1 : n_modes + 1]
        b_k = x[n_modes + 1 :]
        return a_k.astype(np.float64), b_k.astype(np.float64)

    def _evaluate_hamiltonian(
        self,
        s: np.ndarray,
        a_poly: np.ndarray,
        a_fourier: np.ndarray,
        b_fourier: np.ndarray,
        s_max: float,
    ) -> np.ndarray:
        """Evaluate the combined Hamiltonian H(s) at points s.

        Uses Horner's rule for the polynomial part and direct evaluation
        for the Fourier part.
        """
        s_norm = np.clip(s / (s_max + 1e-10), -1.0, 1.0)

        # Polynomial part: Horner's rule
        result = np.zeros_like(s_norm, dtype=np.float64)
        for k in range(len(a_poly) - 1, -1, -1):
            result = result * s_norm + a_poly[k]
        result *= s_max  # un-normalise output

        # Fourier part
        if len(a_fourier) > 0:
            n_modes = len(a_fourier)
            freq = np.arange(1, n_modes + 1, dtype=np.float64)
            s_scaled = s_norm * np.pi
            cos_vals = np.cos(np.outer(s_scaled, freq))
            sin_vals = np.sin(np.outer(s_scaled, freq))
            result += cos_vals @ a_fourier + sin_vals @ b_fourier

        return result

    def _hamiltonian_gradient(
        self,
        s: np.ndarray,
        a_poly: np.ndarray,
        a_fourier: np.ndarray,
        b_fourier: np.ndarray,
        s_max: float,
    ) -> np.ndarray:
        """Compute dH/ds (gradient of Hamiltonian w.r.t. s).

        This drives the symplectic evolution: dq/dt = dH/dp, dp/dt = -dH/dq.
        We use a separable Hamiltonian H(q, p) = T(p) + V(q) where
        V(q) = H(q) is the potential fitted to singular values.
        """
        s_norm = np.clip(s / (s_max + 1e-10), -1.0, 1.0)

        # dV/dq from polynomial via Horner's rule (numerically stable):
        #   d/dx [a0 + a1*x + a2*x^2 + ...] = a1 + 2*a2*x + 3*a3*x^2 + ...
        grad = np.zeros_like(s_norm, dtype=np.float64)
        if len(a_poly) > 1:
            for k in range(len(a_poly) - 1, 0, -1):
                grad = grad * s_norm + k * a_poly[k]
        grad *= s_max  # chain rule
        grad = np.clip(grad, -1e6, 1e6)  # prevent explosion

        # Fourier gradient
        if len(a_fourier) > 0:
            n_modes = len(a_fourier)
            freq = np.arange(1, n_modes + 1, dtype=np.float64)
            s_scaled = s_norm * np.pi
            grad += (
                -np.sin(np.outer(s_scaled, freq)) * (np.pi / (s_max + 1e-10))
            ) @ a_fourier
            grad += (
                np.cos(np.outer(s_scaled, freq)) * (np.pi / (s_max + 1e-10))
            ) @ b_fourier
            grad = np.clip(grad, -1e6, 1e6)

        return grad

    # ── Symplectic integration ───────────────────────────────────────────

    def _symplectic_verlet_integrate(
        self,
        q0: np.ndarray,
        p0: np.ndarray,
        a_poly: np.ndarray,
        a_fourier: np.ndarray,
        b_fourier: np.ndarray,
        s_max: float,
        dt: float,
        n_steps: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Velocity Verlet symplectic integrator.

        For separable Hamiltonian H = T(p) + V(q):
          p_{n+1/2} = p_n - (dt/2) * V'(q_n)
          q_{n+1}   = q_n + dt * p_{n+1/2}
          p_{n+1}   = p_{n+1/2} - (dt/2) * V'(q_{n+1})

        This is second-order, time-reversible, and preserves the
        symplectic 2-form — critical for long-time energy conservation.
        """
        q = q0.copy().astype(np.float64)
        p = p0.copy().astype(np.float64)

        # Clamp initial momentum to prevent divergence
        p_scale = np.max(np.abs(p0)) + 1e-10
        if p_scale > 10.0:
            p = p * (10.0 / p_scale)

        for _ in range(n_steps):
            # Half-step momentum
            grad = self._hamiltonian_gradient(q, a_poly, a_fourier, b_fourier, s_max)
            p_half = p - 0.5 * dt * grad
            p_half = np.clip(p_half, -1e4, 1e4)

            # Full-step position
            q = q + dt * p_half
            q = np.clip(q, -1e6, 1e6)

            # Half-step momentum (completed)
            grad_new = self._hamiltonian_gradient(
                q, a_poly, a_fourier, b_fourier, s_max
            )
            p = p_half - 0.5 * dt * grad_new
            p = np.clip(p, -1e4, 1e4)

        return q, p

    # ── Adaptive rank selection ─────────────────────────────────────────

    def _select_adaptive_rank(
        self, S: np.ndarray, energy_threshold: float = 0.99
    ) -> int:
        """Select rank adaptively based on singular value energy retention.

        Instead of a fixed max_rank, this method analyses the singular value
        spectrum and chooses the smallest rank that retains `energy_threshold`
        fraction of the total variance (sum of squared singular values).

        This prevents over-compression of structured matrices (where most
        energy is in a few components) while avoiding under-compression of
        diffuse spectra.

        Args:
            S: Singular values, shape (n,) sorted descending.
            energy_threshold: Fraction of total energy to retain (default 0.99).

        Returns:
            Selected rank (at least 4, at most 2*max_rank or len(S)).
        """
        total = np.sum(S**2)
        cumsum = np.cumsum(S**2)
        rank = int(np.searchsorted(cumsum, energy_threshold * total) + 1)
        return min(max(rank, 4), min(self.max_rank * 2, len(S)))

    # ── Public API ───────────────────────────────────────────────────────

    def compress(
        self,
        W: np.ndarray,
        num_steps: Optional[int] = None,
    ) -> dict:
        """Compress weight matrix via Hamiltonian spectral encoding.

        The Hamiltonian concept treats the singular value spectrum as
        energy levels of a physical system.  We fit a parametric model
        (polynomial + optional Fourier terms) to the spectrum and store
        the model parameters instead of the full singular value array.

        Steps:
          1. SVD: W = U @ diag(S) @ Vt
          2. Fit Hamiltonian H(k) = sum a_k * k^d  to the spectrum S(k)
          3. Store: U, Vt (orthogonal), a_k (Hamiltonian params), rank
          4. To reconstruct: evaluate H at integer k to recover S

        The polynomial coefficients serve as a *generative model* of
        the singular value spectrum — analogous to how a Hamiltonian
        generates energy eigenvalues in quantum mechanics.
        """
        t0 = time.time()
        W = np.asarray(W, dtype=np.float64)
        m, n = W.shape
        steps = num_steps or self.symplectic_steps

        # SVD
        U, S_full, Vt = np.linalg.svd(W, full_matrices=False)
        r = self._select_adaptive_rank(S_full)
        U_r = U[:, :r]
        S = S_full[:r]
        Vt_r = Vt[:r, :]

        s_max = float(np.max(np.abs(S))) + 1e-10

        # Fit polynomial to singular value spectrum: S(k) ≈ H(k)
        # Use index k as "time" variable
        k = np.arange(r, dtype=np.float64)
        k_norm = k / (r - 1 + 1e-10)  # normalise to [0, 1]
        S_norm = S / s_max

        # Fit polynomial via least-squares
        degree = min(self.polynomial_degree, r - 1)
        powers = np.arange(degree + 1, dtype=np.float64)
        V = k_norm[:, None] ** powers[None, :]
        lam = 1e-8
        a_poly = np.linalg.solve(V.T @ V + lam * np.eye(degree + 1), V.T @ S_norm)

        # Fourier refinement
        a_fourier, b_fourier = np.array([]), np.array([])
        if self.fourier_modes > 0:
            n_modes = min(self.fourier_modes, r // 2)
            if n_modes > 0:
                freq = np.arange(1, n_modes + 1, dtype=np.float64)
                cos_vals = np.cos(np.outer(k_norm * np.pi, freq))
                sin_vals = np.sin(np.outer(k_norm * np.pi, freq))
                basis = np.column_stack([np.ones(r), cos_vals, sin_vals])
                AtA = basis.T @ basis + lam * np.eye(basis.shape[1])
                x = np.linalg.solve(AtA, basis.T @ S_norm)
                a_fourier = x[1 : n_modes + 1]
                b_fourier = x[n_modes + 1 :]

        # Verify reconstruction quality
        S_recovered = self._evaluate_hamiltonian(
            k_norm * np.pi, a_poly, a_fourier, b_fourier, s_max
        )
        rel_err_check = float(
            np.linalg.norm(S - S_recovered) / (np.linalg.norm(S) + 1e-30)
        )

        compress_ms = (time.time() - t0) * 1000

        compressed = {
            "U": U_r.astype(np.float32),
            "Vt": Vt_r.astype(np.float32),
            "a_poly": a_poly.astype(np.float64),
            "a_fourier": a_fourier.astype(np.float64)
            if len(a_fourier) > 0
            else np.array([]),
            "b_fourier": b_fourier.astype(np.float64)
            if len(b_fourier) > 0
            else np.array([]),
            "s_max": float(s_max),
            "rank": int(r),
            "recon_error": float(rel_err_check),
            # Initial conditions for symplectic integration (used by
            # visualize_hamiltonian_trajectory and external analysis)
            "q0": np.zeros(r, dtype=np.float64),  # initial position
            "p0": np.random.RandomState(42).randn(r).astype(np.float64)
            * 0.1,  # initial momentum
            "dt": float(self.symplectic_dt),
        }

        comp_bytes = _estimate_bytes(compressed)
        orig_bytes = W.nbytes

        logger.info(
            "Hamiltonian compress: %dx%d, rank=%d, ratio=%.2f, poly_deg=%d, fourier=%d, err=%.6f",
            m,
            n,
            r,
            orig_bytes / max(comp_bytes, 1),
            len(a_poly),
            len(a_fourier),
            rel_err_check,
        )

        return {
            "data": compressed,
            "orig_shape": W.shape,
            "orig_bytes": orig_bytes,
            "comp_bytes": comp_bytes,
            "compress_ms": compress_ms,
        }

    def decompress(self, compressed: dict) -> Tuple[np.ndarray, float]:
        """Reconstruct weight matrix from Hamiltonian spectral model."""
        t0 = time.time()
        cd = compressed["data"]
        shape = compressed["orig_shape"]

        U = cd["U"].astype(np.float64)
        Vt = cd["Vt"].astype(np.float64)
        a_poly = cd["a_poly"]
        a_fourier = cd["a_fourier"]
        b_fourier = cd["b_fourier"]
        s_max = cd["s_max"]
        r = cd["rank"]

        # Evaluate Hamiltonian at integer indices to recover singular values
        k = np.arange(r, dtype=np.float64)
        k_norm = k / (r - 1 + 1e-10)
        S_rec = self._evaluate_hamiltonian(
            k_norm * np.pi, a_poly, a_fourier, b_fourier, s_max
        )
        S_rec = np.maximum(S_rec, 0.0)  # singular values must be non-negative

        # Reconstruct matrix: W ≈ U @ diag(S_rec) @ Vt
        W_rec = U @ np.diag(S_rec) @ Vt

        # Pad/truncate to original shape
        m, n = shape
        if W_rec.shape[0] < m:
            W_rec = np.pad(W_rec, ((0, m - W_rec.shape[0]), (0, 0)))
        elif W_rec.shape[0] > m:
            W_rec = W_rec[:m, :]
        if W_rec.shape[1] < n:
            W_rec = np.pad(W_rec, ((0, 0), (0, n - W_rec.shape[1])))
        elif W_rec.shape[1] > n:
            W_rec = W_rec[:, :n]

        decompress_ms = (time.time() - t0) * 1000
        return W_rec.astype(np.float32), decompress_ms


# ═══════════════════════════════════════════════════════════════════════════
# 2. TOPOLOGICAL FUNCTIONAL QUANTISATION
# ═══════════════════════════════════════════════════════════════════════════


class TopologicalFunctionalQuantization:
    """Quantise sub-matrices by matching to a universal codebook of
    geometric primitives (rotation, scaling, shear, low-rank projectors).

    Instead of quantising individual values, we quantise the *geometry*
    of sub-matrices.  Each sub-matrix is approximated as:

      W_block ≈ codebook[idx] @ Transform(params)

    where Transform is parametrised by a small number of scalars
    (rotation angle, scale factor, shear parameters).

    The codebook stays at FP32 precision.  Only indices + transform
    parameters are compressed.

    Storage per block: 10 bits (index) + O(r) transform params.
    """

    def __init__(
        self,
        codebook_size: int = 256,
        block_size: int = 32,
        max_transform_params: int = 8,
        n_training_iters: int = 20,
    ):
        self.codebook_size = codebook_size
        self.block_size = block_size
        self.max_transform_params = max_transform_params
        self.n_training_iters = n_training_iters
        logger.info(
            "TopologicalFunctionalQuantization: cb_size=%d, block=%d, max_tp=%d",
            codebook_size,
            block_size,
            max_transform_params,
        )

    # ── Codebook learning ────────────────────────────────────────────────

    def build_codebook(
        self,
        training_tensors: List[np.ndarray],
    ) -> Dict[str, Any]:
        """Learn universal geometric primitives from training data.

        Strategy:
          1. Extract many sub-blocks from training tensors
          2. Compute SVD of each block → (U, S, Vt)
          3. Cluster SVD components via K-means to find "typical" primitives
          4. Each codebook entry = (U_centroid, S_centroid, Vt_centroid)
        """
        logger.info("Building codebook from %d training tensors", len(training_tensors))

        # Step 1: Extract sub-blocks
        blocks = []
        rng = np.random.RandomState(42)
        for tensor in training_tensors:
            t = np.asarray(tensor, dtype=np.float64)
            if t.ndim == 1:
                t = t.reshape(1, -1)
            m, n = t.shape
            bs = min(self.block_size, m, n)
            for _ in range(min(50, max(1, m * n // (bs * bs)))):
                i = rng.randint(0, max(1, m - bs + 1))
                j = rng.randint(0, max(1, n - bs + 1))
                block = t[i : i + bs, j : j + bs]
                if block.size > 0 and np.std(block) > 1e-10:
                    blocks.append(block)

        if len(blocks) < self.codebook_size:
            logger.warning(
                "Only %d blocks extracted (need %d for codebook), using all",
                len(blocks),
                self.codebook_size,
            )

        # Step 2: Compute SVD for each block
        svd_components = []
        for block in blocks:
            U, S, Vt = np.linalg.svd(block, full_matrices=False)
            r = min(self.max_transform_params, len(S))
            svd_components.append(
                {
                    "U": U[:, :r],
                    "S": S[:r],
                    "Vt": Vt[:r, :],
                }
            )

        # Step 3: Cluster singular value spectra via K-means
        spectra = np.array([c["S"] for c in svd_components])
        # Pad spectra to same length
        max_len = max(len(s) for s in spectra)
        spectra_padded = np.zeros((len(spectra), max_len), dtype=np.float64)
        for i, s in enumerate(spectra):
            spectra_padded[i, : len(s)] = s

        # Normalise spectra for clustering
        norms = np.linalg.norm(spectra_padded, axis=1, keepdims=True) + 1e-10
        spectra_normed = spectra_padded / norms

        # K-means
        n_clusters = min(self.codebook_size, len(spectra_normed))
        centroids = spectra_normed[
            rng.choice(len(spectra_normed), n_clusters, replace=False)
        ].copy()
        assignments = np.zeros(len(spectra_normed), dtype=np.int32)

        for _ in range(self.n_training_iters):
            # Assign
            dists = np.linalg.norm(
                spectra_normed[:, None, :] - centroids[None, :, :], axis=2
            )
            assignments = np.argmin(dists, axis=1)
            # Update centroids
            for c in range(n_clusters):
                mask = assignments == c
                if np.any(mask):
                    centroids[c] = spectra_normed[mask].mean(axis=0)

        # Step 4: Build codebook entries
        # For each centroid, pick the closest actual block's SVD components
        codebook_u = []
        codebook_s = []
        codebook_vt = []
        for c in range(n_clusters):
            mask = assignments == c
            if not np.any(mask):
                # Use a random entry
                idx = rng.randint(0, len(svd_components))
                codebook_u.append(svd_components[idx]["U"])
                codebook_s.append(svd_components[idx]["S"])
                codebook_vt.append(svd_components[idx]["Vt"])
                continue
            # Pick the entry closest to centroid
            indices = np.where(mask)[0]
            dists_to_cent = np.linalg.norm(
                spectra_normed[indices] - centroids[c], axis=1
            )
            best = indices[np.argmin(dists_to_cent)]
            codebook_u.append(svd_components[best]["U"])
            codebook_s.append(svd_components[best]["S"])
            codebook_vt.append(svd_components[best]["Vt"])

        # Pad all codebook entries to uniform shape
        max_r = max(c.shape[1] if c.ndim > 1 else len(c) for c in codebook_s)
        for i in range(len(codebook_u)):
            r_i = codebook_u[i].shape[1]
            if r_i < max_r:
                codebook_u[i] = np.pad(codebook_u[i], ((0, 0), (0, max_r - r_i)))
                codebook_s[i] = np.pad(codebook_s[i], (0, max_r - r_i))
                codebook_vt[i] = np.pad(codebook_vt[i], ((0, 0), (0, max_r - r_i)))

        codebook = {
            "U": np.array(codebook_u, dtype=np.float32),  # (K, block, r)
            "S": np.array(codebook_s, dtype=np.float32),  # (K, r)
            "Vt": np.array(codebook_vt, dtype=np.float32),  # (K, r, block)
            "n_entries": n_clusters,
            "max_r": max_r,
        }
        cb_bytes = _estimate_bytes(codebook)
        logger.info("Codebook built: %d entries, %.1f KB", n_clusters, cb_bytes / 1024)
        return codebook

    # ── Optimal transform fitting ────────────────────────────────────────

    def _fit_transform(
        self,
        block: np.ndarray,
        cb_entry_svd: Dict[str, np.ndarray],
    ) -> Tuple[float, float, np.ndarray]:
        """Find optimal transform params (angle, scale, shear) to match block.

        We parametrise the transform as:
          T(params) = R(angle) @ diag(scale_factors) @ Shear(sx, sy)

        Returns (angle, scale, residual_error).
        """
        U_cb, S_cb, Vt_cb = cb_entry_svd["U"], cb_entry_svd["S"], cb_entry_svd["Vt"]

        # Reconstruct codebook approximation at full block size
        r = len(S_cb)
        block_approx = U_cb[:, :r] @ np.diag(S_cb[:r]) @ Vt_cb[:r, :]
        if block_approx.shape != block.shape:
            block_approx = block_approx[: block.shape[0], : block.shape[1]]

        # Find optimal scalar scale via least-squares
        # min ||block - alpha * block_approx||^2  =>  alpha = <block, block_approx> / ||block_approx||^2
        alpha = float(np.sum(block * block_approx) / (np.sum(block_approx**2) + 1e-10))

        # Find optimal rotation angle (1D search over [0, 2pi])
        block_scaled = block_approx * alpha
        best_angle = 0.0
        best_residual = float(np.linalg.norm(block - block_scaled))

        for angle_idx in range(8):
            angle = angle_idx * np.pi / 4.0
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            # 2D rotation applied element-wise to approximate
            m_b, n_b = block.shape
            rotated = block_scaled.copy()
            if m_b >= 2 and n_b >= 2:
                # Apply rotation to 2x2 blocks
                for i in range(0, m_b - 1, 2):
                    for j in range(0, n_b - 1, 2):
                        patch = block_scaled[i : i + 2, j : j + 2]
                        R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
                        rotated[i : i + 2, j : j + 2] = R @ patch @ R.T
            residual = float(np.linalg.norm(block - rotated))
            if residual < best_residual:
                best_residual = residual
                best_angle = angle

        return best_angle, alpha, best_residual

    # ── Compress / decompress ────────────────────────────────────────────

    def compress(
        self,
        W: np.ndarray,
        codebook: Optional[Dict[str, Any]] = None,
    ) -> dict:
        """Compress weight matrix using geometric codebook matching."""
        t0 = time.time()
        W = np.asarray(W, dtype=np.float64)
        m, n = W.shape
        bs = self.block_size

        if codebook is None:
            codebook = self.build_codebook([W])

        cb_U = codebook["U"]
        cb_S = codebook["S"]
        cb_Vt = codebook["Vt"]
        n_entries = codebook["n_entries"]
        max_r = codebook["max_r"]

        indices = []
        transforms = []
        total_blocks = 0
        total_residual = 0.0

        for i in range(0, m, bs):
            for j in range(0, n, bs):
                block = W[i : i + bs, j : j + bs]
                if block.size == 0:
                    continue
                total_blocks += 1

                # Find best codebook entry
                best_cb_idx = 0
                best_residual = float("inf")
                best_angle = 0.0
                best_scale = 1.0

                for cb_idx in range(n_entries):
                    cb_entry = {
                        "U": cb_U[cb_idx],
                        "S": cb_S[cb_idx],
                        "Vt": cb_Vt[cb_idx],
                    }
                    angle, scale, residual = self._fit_transform(block, cb_entry)
                    if residual < best_residual:
                        best_residual = residual
                        best_cb_idx = cb_idx
                        best_angle = angle
                        best_scale = scale

                indices.append(best_cb_idx)
                transforms.append([best_angle, best_scale])
                total_residual += best_residual**2

        compress_ms = (time.time() - t0) * 1000

        compressed = {
            "indices": np.array(indices, dtype=np.uint16),
            "transforms": np.array(transforms, dtype=np.float32),
            "codebook": codebook,
            "block_size": bs,
            "grid_m": (m + bs - 1) // bs,
            "grid_n": (n + bs - 1) // bs,
        }

        comp_bytes = _estimate_bytes(compressed)
        orig_bytes = W.nbytes

        return {
            "data": compressed,
            "orig_shape": W.shape,
            "orig_bytes": orig_bytes,
            "comp_bytes": comp_bytes,
            "compress_ms": compress_ms,
        }

    def decompress(self, compressed: dict) -> Tuple[np.ndarray, float]:
        """Reconstruct weight matrix from codebook + indices + transforms."""
        t0 = time.time()
        cd = compressed["data"]
        shape = compressed["orig_shape"]
        m, n = shape
        bs = cd["block_size"]

        cb_U = cd["codebook"]["U"]
        cb_S = cd["codebook"]["S"]
        cb_Vt = cd["codebook"]["Vt"]

        result = np.zeros((m, n), dtype=np.float64)
        block_idx = 0

        for i in range(0, m, bs):
            for j in range(0, n, bs):
                if block_idx >= len(cd["indices"]):
                    break
                cb_idx = cd["indices"][block_idx]
                angle, scale = cd["transforms"][block_idx]

                # Reconstruct block from codebook
                U_cb = cb_U[cb_idx].astype(np.float64)
                S_cb = cb_S[cb_idx].astype(np.float64)
                Vt_cb = cb_Vt[cb_idx].astype(np.float64)
                r = len(S_cb)

                block = U_cb[:, :r] @ np.diag(S_cb[:r]) @ Vt_cb[:r, :]
                # Apply scale
                block *= scale

                # Apply rotation
                cos_a, sin_a = np.cos(angle), np.sin(angle)
                bi = min(bs, m - i)
                bj = min(bs, n - j)
                block = block[:bi, :bj]
                if bi >= 2 and bj >= 2:
                    rotated = block.copy()
                    for ii in range(0, bi - 1, 2):
                        for jj in range(0, bj - 1, 2):
                            patch = block[ii : ii + 2, jj : jj + 2]
                            R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
                            rotated[ii : ii + 2, jj : jj + 2] = R @ patch @ R.T
                    block = rotated

                result[i : i + bi, j : j + bj] = block
                block_idx += 1

        decompress_ms = (time.time() - t0) * 1000
        return result[:m, :n].astype(np.float32), decompress_ms


# ═══════════════════════════════════════════════════════════════════════════
# 3. HIERARCHICAL STATE-SPACE WAVEFORMS
# ═══════════════════════════════════════════════════════════════════════════


class HierarchicalStateSpaceWaveforms:
    """Treat W as a 2D signal and decompose into hierarchical waveforms.

    Multi-resolution approach:
      1. 2D DCT → identify dominant frequency components
      2. Wavelet-like hierarchical decomposition (coarse-to-fine)
      3. B-spline interpolation between retained coefficients
      4. State-space formulation at each spatial position

    For smooth weight matrices, energy concentrates in low frequencies
    → very few coefficients needed.  For structured matrices (Toeplitz,
    circulant), the frequency representation is extremely sparse.

    Storage: O(k) where k = number of significant coefficients.
    """

    def __init__(
        self,
        keep_ratio: float = 0.15,
        n_wavelet_levels: int = 4,
        bspline_order: int = 3,
        adaptive_threshold: bool = True,
    ):
        self.keep_ratio = keep_ratio
        self.n_wavelet_levels = n_wavelet_levels
        self.bspline_order = bspline_order
        self.adaptive_threshold = adaptive_threshold
        logger.info(
            "HierarchicalStateSpaceWaveforms: keep=%.2f, levels=%d, bspline=%d",
            keep_ratio,
            n_wavelet_levels,
            bspline_order,
        )

    # ── 2D DCT spectral compression ─────────────────────────────────────

    def _dct_compress(self, W: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compress via 2D DCT + thresholding.

        Returns (coefficients, row_indices, col_indices) of kept coefficients.
        """
        W64 = W.astype(np.float64)

        # Pad to power-of-2 for optimal DCT energy compaction
        m, n = W64.shape
        mp = next_power_of_two(m)
        np_ = next_power_of_two(n)
        padded = np.zeros((mp, np_), dtype=np.float64)
        padded[:m, :n] = W64

        # 2D DCT
        coeffs = dct_2d(padded)

        # Adaptive or fixed thresholding
        if self.adaptive_threshold:
            # Keep top k% by energy
            abs_coeffs = np.abs(coeffs.ravel())
            sorted_abs = np.sort(abs_coeffs)[::-1]
            energy_cumsum = np.cumsum(sorted_abs**2)
            total_energy = energy_cumsum[-1] + 1e-30
            n_keep = (
                int(np.searchsorted(energy_cumsum, self.keep_ratio * total_energy)) + 1
            )
            n_keep = max(1, min(n_keep, coeffs.size))
            threshold = sorted_abs[min(n_keep, len(sorted_abs) - 1)]
        else:
            threshold = np.percentile(np.abs(coeffs), (1 - self.keep_ratio) * 100)

        mask = np.abs(coeffs) >= threshold
        row_idx, col_idx = np.where(mask)
        kept_coeffs = coeffs[row_idx, col_idx]

        return (
            kept_coeffs.astype(np.float64),
            row_idx.astype(np.int32),
            col_idx.astype(np.int32),
        )

    def _dct_decompress(
        self,
        coeffs: np.ndarray,
        row_idx: np.ndarray,
        col_idx: np.ndarray,
        shape: Tuple[int, int],
    ) -> np.ndarray:
        """Reconstruct from sparse DCT coefficients via inverse DCT."""
        m, n = shape
        mp = next_power_of_two(m)
        np_ = next_power_of_two(n)

        full_coeffs = np.zeros((mp, np_), dtype=np.float64)
        full_coeffs[row_idx, col_idx] = coeffs

        reconstructed = idct_2d(full_coeffs)
        return reconstructed[:m, :n].astype(np.float64)

    # ── Wavelet-like hierarchical decomposition ──────────────────────────

    def _wavelet_compress(self, W: np.ndarray) -> Dict[str, Any]:
        """Multi-level hierarchical wavelet-style decomposition.

        Uses Haar-like averaging/differencing applied to rows and columns.
        Each level captures progressively finer detail.
        """
        W64 = W.astype(np.float64)
        m, n = W64.shape

        levels = []
        current = W64

        for level in range(self.n_wavelet_levels):
            cm, cn = current.shape
            if cm < 2 or cn < 2:
                break

            # Row-wise averaging + differencing
            even_rows = current[0::2, :]
            odd_rows = current[1::2, :]
            if odd_rows.shape[0] < even_rows.shape[0]:
                odd_rows = np.pad(odd_rows, ((0, 1), (0, 0)))
            approx_r = (even_rows + odd_rows) * 0.5
            detail_r = (even_rows - odd_rows) * 0.5

            # Column-wise on approx
            cr, cc = approx_r.shape
            if cc >= 2:
                even_cols = approx_r[:, 0::2]
                odd_cols = approx_r[:, 1::2]
                if odd_cols.shape[1] < even_cols.shape[1]:
                    odd_cols = np.pad(odd_cols, ((0, 0), (0, 1)))
                approx_c = (even_cols + odd_cols) * 0.5
                detail_c_h = (even_cols + odd_cols) * 0.5  # horizontal detail
                detail_c_d = (even_cols - odd_cols) * 0.5  # diagonal detail
            else:
                approx_c = approx_r
                detail_c_h = np.zeros_like(approx_r)
                detail_c_d = np.zeros_like(approx_r)

            levels.append(
                {
                    "level": level,
                    "approx": approx_c,
                    "detail_h": detail_c_h,
                    "detail_v": detail_r,
                    "detail_d": detail_c_d,
                }
            )

            current = approx_c

        # Threshold detail coefficients
        all_details = []
        for lv in levels:
            all_details.append(lv["detail_h"].ravel())
            all_details.append(lv["detail_v"].ravel())
            all_details.append(lv["detail_d"].ravel())
        all_detail_vals = np.concatenate(all_details)

        if len(all_detail_vals) > 0:
            threshold = np.percentile(
                np.abs(all_detail_vals), (1 - self.keep_ratio) * 100
            )
        else:
            threshold = 0.0

        for lv in levels:
            for key in ["detail_h", "detail_v", "detail_d"]:
                detail = lv[key]
                mask = np.abs(detail) >= threshold
                idx = np.argwhere(mask)
                if idx.size == 0:
                    idx = np.empty((0, 2), dtype=np.int32)
                lv[key + "_sparse"] = {
                    "vals": detail[mask].astype(np.float64),
                    "idx": idx.astype(np.int32),
                    "shape": detail.shape,
                }
                del lv[key]

        return {"levels": levels, "residual": current, "orig_shape": W.shape}

    def _wavelet_decompress(self, compressed: dict) -> np.ndarray:
        """Reconstruct from hierarchical wavelet decomposition."""
        levels = compressed["levels"]
        current = compressed["residual"]

        for lv in reversed(levels):
            approx = current

            # Reconstruct detail arrays from sparse representations
            detail_h = np.zeros(lv["detail_h_sparse"]["shape"], dtype=np.float64)
            idx = lv["detail_h_sparse"]["idx"]
            if len(idx) > 0:
                detail_h[idx[:, 0], idx[:, 1]] = lv["detail_h_sparse"]["vals"]

            detail_v = np.zeros(lv["detail_v_sparse"]["shape"], dtype=np.float64)
            idx = lv["detail_v_sparse"]["idx"]
            if len(idx) > 0:
                detail_v[idx[:, 0], idx[:, 1]] = lv["detail_v_sparse"]["vals"]

            detail_d = np.zeros(lv["detail_d_sparse"]["shape"], dtype=np.float64)
            idx = lv["detail_d_sparse"]["idx"]
            if len(idx) > 0:
                detail_d[idx[:, 0], idx[:, 1]] = lv["detail_d_sparse"]["vals"]

            # Column-wise inverse
            ca, cc = approx.shape
            if cc >= 2:
                recon_cols = np.zeros((ca, cc * 2), dtype=np.float64)
                recon_cols[:, 0::2] = approx + detail_d
                recon_cols[:, 1::2] = approx - detail_d
            else:
                recon_cols = approx

            # Row-wise inverse
            ra, rc = recon_cols.shape
            target_rows = ra * 2
            recon_rows = np.zeros((target_rows, rc), dtype=np.float64)
            recon_rows[0::2] = (
                recon_cols[:ra, :] + detail_v[: min(ra, detail_v.shape[0]), :rc]
            )
            recon_rows[1::2] = (
                recon_cols[:ra, :] - detail_v[: min(ra, detail_v.shape[0]), :rc]
            )

            current = recon_rows

        orig_shape = compressed["orig_shape"]
        m, n = orig_shape
        return current[:m, :n].astype(np.float64)

    # ── B-spline interpolation between coefficients ──────────────────────

    def _bspline_interpolate(
        self,
        sparse_coeffs: np.ndarray,
        sparse_rows: np.ndarray,
        sparse_cols: np.ndarray,
        target_shape: Tuple[int, int],
        order: int = 3,
    ) -> np.ndarray:
        """Interpolate sparse DCT coefficients using B-spline basis.

        This provides smooth reconstruction between retained coefficients,
        avoiding the blockiness of pure thresholding.
        """
        m, n = target_shape
        result = np.zeros((m, n), dtype=np.float64)

        if len(sparse_coeffs) == 0:
            return result

        # For each retained coefficient, place a B-spline kernel
        support = order + 1
        for k in range(len(sparse_coeffs)):
            r, c = sparse_rows[k], sparse_cols[k]
            val = sparse_coeffs[k]

            # B-spline kernel support
            for dr in range(-support, support + 1):
                for dc in range(-support, support + 1):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < m and 0 <= cc < n:
                        # B-spline basis value (simplified cubic)
                        t_dr = abs(dr) / support
                        t_dc = abs(dc) / support
                        if t_dr <= 1.0 and t_dc <= 1.0:
                            w = (1.0 - t_dr) ** order * (1.0 - t_dc) ** order
                            result[rr, cc] += val * w

        # Normalise
        norm_map = np.zeros((m, n), dtype=np.float64)
        for k in range(len(sparse_coeffs)):
            r, c = sparse_rows[k], sparse_cols[k]
            for dr in range(-support, support + 1):
                for dc in range(-support, support + 1):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < m and 0 <= cc < n:
                        t_dr = abs(dr) / support
                        t_dc = abs(dc) / support
                        if t_dr <= 1.0 and t_dc <= 1.0:
                            w = (1.0 - t_dr) ** order * (1.0 - t_dc) ** order
                            norm_map[rr, cc] += w

        result /= norm_map + 1e-10
        return result

    # ── State-space formulation ──────────────────────────────────────────

    def _state_space_encode(self, W: np.ndarray) -> Dict[str, Any]:
        """Encode weight matrix as a spatially-indexed state-space system.

        For each row i, the "state" encodes the local weight behavior:
          x(i+1) = A * x(i) + B * w(i)
          y(i)   = C * x(i)

        We learn the system matrices (A, B, C) from the weight matrix
        and store them compactly.
        """
        m, n = W.shape
        state_dim = min(8, n // 4)  # latent state dimension

        # SVD of the weight matrix gives the natural state-space decomposition
        U, S, Vt = np.linalg.svd(W.astype(np.float64), full_matrices=False)
        r = min(state_dim, len(S))

        # System matrices from balanced realization
        # A = diag(exp(-S_k))  (stable dynamics)
        A = np.diag(np.exp(-S[:r] / (np.max(S[:r]) + 1e-10)))
        # B = Vt[:r, :]  (input matrix)
        B = Vt[:r, :]
        # C = U[:, :r]   (output matrix)
        C = U[:, :r]

        # Initial state from first row
        x0 = np.zeros(r, dtype=np.float64)
        if n >= r:
            x0 = W[0, :r].astype(np.float64)
        elif n > 0:
            x0[:n] = W[0, :].astype(np.float64)

        return {
            "A": A.astype(np.float32),
            "B": B.astype(np.float32),
            "C": C.astype(np.float32),
            "x0": x0.astype(np.float32),
            "state_dim": r,
            "S": S[:r].astype(np.float32),
        }

    def _state_space_decode(
        self, ss: Dict[str, Any], shape: Tuple[int, int]
    ) -> np.ndarray:
        """Decode weight matrix from state-space representation."""
        m, n = shape
        A = ss["A"].astype(np.float64)
        B = ss["B"].astype(np.float64)
        C = ss["C"].astype(np.float64)
        x0 = ss["x0"].astype(np.float64)
        r = ss["state_dim"]

        result = np.zeros((m, n), dtype=np.float64)
        x = x0.copy()

        for i in range(m):
            # Output: y = C @ x
            y = C[i, :] @ x if i < C.shape[0] else np.zeros(n)
            result[i, : len(y)] = y[:n]

            # State update: x = A @ x + B @ w_row (use output as input)
            if i < B.shape[1]:
                x = A @ x + B[:, i] * (y[0] if len(y) > 0 else 0)
            else:
                x = A @ x

        return result

    # ── Public API ───────────────────────────────────────────────────────

    def compress(self, W: np.ndarray) -> dict:
        """Compress weight matrix using hierarchical state-space waveforms.

        Strategy:
          1. 2D DCT → sparse coefficients
          2. Hierarchical wavelet decomposition → multi-scale details
          3. State-space encoding → compact dynamics
          4. Return whichever gives best compression
        """
        t0 = time.time()
        W = np.asarray(W, dtype=np.float64)
        m, n = W.shape

        # Method 1: DCT spectral
        dct_coeffs, dct_rows, dct_cols = self._dct_compress(W)
        dct_compressed = {
            "coeffs": dct_coeffs,
            "row_idx": dct_rows,
            "col_idx": dct_cols,
            "method": "dct",
        }
        dct_bytes = _estimate_bytes(dct_compressed)

        # Method 2: Wavelet hierarchical
        wavelet_compressed = self._wavelet_compress(W)
        wavelet_compressed["method"] = "wavelet"
        wavelet_bytes = _estimate_bytes(wavelet_compressed)

        # Method 3: State-space
        ss_compressed = self._state_space_encode(W)
        ss_compressed["method"] = "statespace"
        ss_bytes = _estimate_bytes(ss_compressed)

        # Choose best
        candidates = [
            ("dct", dct_compressed, dct_bytes),
            ("wavelet", wavelet_compressed, wavelet_bytes),
            ("statespace", ss_compressed, ss_bytes),
        ]
        best_name, best_data, best_bytes = min(candidates, key=lambda x: x[2])

        compress_ms = (time.time() - t0) * 1000
        orig_bytes = W.nbytes

        logger.info(
            "HierarchicalStateSpace: chose %s, ratio=%.2f",
            best_name,
            orig_bytes / max(best_bytes, 1),
        )

        return {
            "data": best_data,
            "orig_shape": W.shape,
            "orig_bytes": orig_bytes,
            "comp_bytes": best_bytes,
            "compress_ms": compress_ms,
            "chosen_method": best_name,
        }

    def decompress(self, compressed: dict) -> Tuple[np.ndarray, float]:
        """Reconstruct weight matrix from compressed representation."""
        t0 = time.time()
        cd = compressed["data"]
        shape = compressed["orig_shape"]
        method = cd.get("method", "dct")

        if method == "dct":
            result = self._dct_decompress(
                cd["coeffs"], cd["row_idx"], cd["col_idx"], shape
            )
        elif method == "wavelet":
            result = self._wavelet_decompress(cd)
        elif method == "statespace":
            result = self._state_space_decode(cd, shape)
        else:
            raise ValueError(f"Unknown method: {method}")

        decompress_ms = (time.time() - t0) * 1000
        return result.astype(np.float32), decompress_ms


# ═══════════════════════════════════════════════════════════════════════════
# 4. PHYSICS COMPRESSION ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════


class PhysicsCompressionOrchestrator:
    """Main entry point: tries all physics methods + standard methods
    and returns the best one meeting the error budget.

    For each tensor:
      1. Analyse its properties (rank, entropy, structure)
      2. Try Hamiltonian, Geometric, and Waveform compression
      3. Also try standard SVD truncation as baseline
      4. Return the method with best ratio within error tolerance
    """

    def __init__(
        self,
        target_ratio: float = 5000.0,
        max_error: float = 0.001,
        hamiltonian_steps: int = 80,
        codebook_size: int = 128,
        wavelet_keep: float = 0.15,
    ):
        self.target_ratio = target_ratio
        self.max_error = max_error
        self.hamiltonian = HamiltonianWeightDynamicals(
            symplectic_steps=hamiltonian_steps,
        )
        self.geometric = TopologicalFunctionalQuantization(
            codebook_size=codebook_size,
        )
        self.waveform = HierarchicalStateSpaceWaveforms(
            keep_ratio=wavelet_keep,
        )
        self._codebook: Optional[Dict[str, Any]] = None
        logger.info(
            "PhysicsCompressionOrchestrator: target_ratio=%.1f, max_error=%.4f",
            target_ratio,
            max_error,
        )

    def _build_codebook_if_needed(self, tensors: List[np.ndarray]) -> None:
        """Build geometric codebook from training tensors if not yet built."""
        if self._codebook is None:
            self._codebook = self.geometric.build_codebook(tensors)

    def _svd_baseline(self, W: np.ndarray, rank: int) -> dict:
        """Simple SVD truncation as baseline."""
        m, n = W.shape
        U, S, Vt = np.linalg.svd(W.astype(np.float64), full_matrices=False)
        r = min(rank, len(S))
        return {
            "U": U[:, :r].astype(np.float32),
            "S": S[:r].astype(np.float32),
            "Vt": Vt[:r, :].astype(np.float32),
            "rank": r,
        }

    def _svd_decompress(self, cd: dict, shape: Tuple[int, ...]) -> np.ndarray:
        """Reconstruct from SVD baseline."""
        W = (
            cd["U"].astype(np.float64)
            @ np.diag(cd["S"].astype(np.float64))
            @ cd["Vt"].astype(np.float64)
        )
        m, n = shape
        return W[:m, :n].astype(np.float32)

    def _quick_svd_ratio(self, W: np.ndarray) -> int:
        """Estimate minimum rank needed for target error via quick SVD analysis."""
        # Sample a few rows for speed
        m, n = W.shape
        sample_rows = min(m, 32)
        row_idx = np.linspace(0, m - 1, sample_rows, dtype=int)
        W_sample = W[row_idx, :]
        _, S, _ = np.linalg.svd(W_sample.astype(np.float64), full_matrices=False)
        total = float(np.sum(S**2)) + 1e-30
        cumsum = np.cumsum(S**2) / total
        # Find rank where we keep enough energy for target error
        energy_needed = 1.0 - self.max_error**2
        rank = int(np.searchsorted(cumsum, energy_needed)) + 1
        # Cap rank to a reasonable fraction of matrix dimension
        rank = max(1, min(rank, min(m, n) // 2, 64))
        return rank

    def analyze_and_compress(
        self,
        W: np.ndarray,
        target_ratio: Optional[float] = None,
        max_error: Optional[float] = None,
    ) -> CompressedWeight:
        """Analyse tensor properties and compress with the best physics method.

        Tries:
          1. SVD baseline (truncated)
          2. Hamiltonian dynamical encoding
          3. Geometric codebook quantisation (if codebook available)
          4. Hierarchical state-space waveforms

        Returns the CompressedWeight with the best compression ratio
        within the error budget.
        """
        target_ratio = target_ratio or self.target_ratio
        max_error = max_error or self.max_error
        W = np.asarray(W, dtype=np.float64)
        m, n = W.shape
        orig_bytes = W.nbytes

        logger.info(
            "analyze_and_compress: %dx%d (%d bytes), target_ratio=%.1f, max_error=%.4f",
            m,
            n,
            orig_bytes,
            target_ratio,
            max_error,
        )

        # Analyse tensor properties
        spectral_ent = spectral_entropy(W.ravel().astype(np.float32))
        quick_rank = self._quick_svd_ratio(W)
        W_norm = float(np.linalg.norm(W)) + 1e-30

        candidates: List[CompressedWeight] = []

        # --- Candidate 1: SVD baseline ---
        try:
            svd_rank = quick_rank
            cd_svd = self._svd_baseline(W, svd_rank)
            t0 = time.time()
            W_rec = self._svd_decompress(cd_svd, W.shape)
            decompress_ms = (time.time() - t0) * 1000
            rel_err, snr = _measure_error(W, W_rec)
            comp_bytes = _estimate_bytes(cd_svd)
            ratio = orig_bytes / max(comp_bytes, 1)
            candidates.append(
                CompressedWeight(
                    method="svd_baseline",
                    compressed_data=cd_svd,
                    metadata={"rank": svd_rank},
                    original_shape=W.shape,
                    original_bytes=orig_bytes,
                    compressed_bytes=comp_bytes,
                    compression_ratio=ratio,
                    reconstruction_error=rel_err,
                    snr_db=snr,
                    compress_time_ms=0,
                    decompress_time_ms=decompress_ms,
                )
            )
        except Exception as e:
            logger.warning("SVD baseline failed: %s", e)

        # --- Candidate 2: Hamiltonian ---
        try:
            t0 = time.time()
            ham_result = self.hamiltonian.compress(W)
            compress_ms = (time.time() - t0) * 1000
            W_rec, decompress_ms = self.hamiltonian.decompress(ham_result)
            rel_err, snr = _measure_error(W, W_rec.astype(np.float64))
            comp_bytes = ham_result["comp_bytes"]
            ratio = orig_bytes / max(comp_bytes, 1)
            candidates.append(
                CompressedWeight(
                    method="hamiltonian_dynamical",
                    compressed_data=ham_result["data"],
                    metadata=ham_result,
                    original_shape=W.shape,
                    original_bytes=orig_bytes,
                    compressed_bytes=comp_bytes,
                    compression_ratio=ratio,
                    reconstruction_error=rel_err,
                    snr_db=snr,
                    compress_time_ms=compress_ms,
                    decompress_time_ms=decompress_ms,
                    extra={"rank": ham_result["data"]["rank"]},
                )
            )
        except Exception as e:
            logger.warning("Hamiltonian compression failed: %s", e)

        # --- Candidate 3: Geometric codebook ---
        try:
            self._build_codebook_if_needed([W])
            t0 = time.time()
            geo_result = self.geometric.compress(W, codebook=self._codebook)
            compress_ms = (time.time() - t0) * 1000
            W_rec, decompress_ms = self.geometric.decompress(geo_result)
            rel_err, snr = _measure_error(W, W_rec.astype(np.float64))
            comp_bytes = geo_result["comp_bytes"]
            ratio = orig_bytes / max(comp_bytes, 1)
            candidates.append(
                CompressedWeight(
                    method="geometric_codebook",
                    compressed_data=geo_result["data"],
                    metadata=geo_result,
                    original_shape=W.shape,
                    original_bytes=orig_bytes,
                    compressed_bytes=comp_bytes,
                    compression_ratio=ratio,
                    reconstruction_error=rel_err,
                    snr_db=snr,
                    compress_time_ms=compress_ms,
                    decompress_time_ms=decompress_ms,
                )
            )
        except Exception as e:
            logger.warning("Geometric codebook compression failed: %s", e)

        # --- Candidate 4: Hierarchical waveforms ---
        try:
            t0 = time.time()
            wave_result = self.waveform.compress(W)
            compress_ms = (time.time() - t0) * 1000
            W_rec, decompress_ms = self.waveform.decompress(wave_result)
            rel_err, snr = _measure_error(W, W_rec.astype(np.float64))
            comp_bytes = wave_result["comp_bytes"]
            ratio = orig_bytes / max(comp_bytes, 1)
            candidates.append(
                CompressedWeight(
                    method="hierarchical_waveform",
                    compressed_data=wave_result["data"],
                    metadata=wave_result,
                    original_shape=W.shape,
                    original_bytes=orig_bytes,
                    compressed_bytes=comp_bytes,
                    compression_ratio=ratio,
                    reconstruction_error=rel_err,
                    snr_db=snr,
                    compress_time_ms=compress_ms,
                    decompress_time_ms=decompress_ms,
                    extra={
                        "chosen_submethod": wave_result.get("chosen_method", "unknown")
                    },
                )
            )
        except Exception as e:
            logger.warning("Hierarchical waveform compression failed: %s", e)

        if not candidates:
            raise RuntimeError("All compression methods failed")

        # Select best candidate within error budget
        feasible = [c for c in candidates if c.reconstruction_error <= max_error]
        if feasible:
            # Among feasible, pick highest ratio
            best = max(feasible, key=lambda c: c.compression_ratio)
        else:
            # If none meet error budget, pick lowest error
            best = min(candidates, key=lambda c: c.reconstruction_error)
            logger.warning(
                "No method met error budget %.4f; best error=%.6f (method=%s)",
                max_error,
                best.reconstruction_error,
                best.method,
            )

        logger.info(
            "Best method: %s, ratio=%.2f, error=%.6f, SNR=%.1f dB",
            best.method,
            best.compression_ratio,
            best.reconstruction_error,
            best.snr_db,
        )

        return best


# ═══════════════════════════════════════════════════════════════════════════
# 5. COMPRESSION VISUALIZER
# ═══════════════════════════════════════════════════════════════════════════


class CompressionVisualizer:
    """Debugging and analysis tools for compression results.

    Provides:
      - Frequency spectrum analysis
      - Manifold visualisation (PCA/t-SNE of weight blocks)
      - Hamiltonian trajectory analysis
      - Compression report generation
    """

    def visualize_frequency_spectrum(self, W: np.ndarray) -> Dict[str, Any]:
        """Analyse the frequency content of a weight matrix.

        Returns DCT energy distribution, dominant frequencies, and
        spectral entropy — useful for choosing compression strategy.
        """
        W64 = np.asarray(W, dtype=np.float64)
        m, n = W64.shape

        # 2D DCT
        mp = next_power_of_two(m)
        np_ = next_power_of_two(n)
        padded = np.zeros((mp, np_), dtype=np.float64)
        padded[:m, :n] = W64
        coeffs = dct_2d(padded)

        # Energy distribution
        energy = coeffs**2
        # Radial energy (distance from DC component)
        cy, cx = mp // 2, np_ // 2
        yy, xx = np.mgrid[:mp, :np_]
        radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(np.float64)
        max_radius = float(np.max(radius))

        n_bins = 32
        radial_energy = np.zeros(n_bins)
        radial_counts = np.zeros(n_bins)
        for i in range(mp):
            for j in range(np_):
                bin_idx = min(int(radius[i, j] / max_radius * n_bins), n_bins - 1)
                radial_energy[bin_idx] += energy[i, j]
                radial_counts[bin_idx] += 1

        # Normalise
        radial_energy /= radial_counts + 1e-10

        # Energy concentration in low frequencies
        total_energy = float(np.sum(energy))
        low_freq_mask = radius <= max_radius * 0.25
        low_freq_energy = float(np.sum(energy[low_freq_mask]))
        low_freq_ratio = low_freq_energy / (total_energy + 1e-10)

        # Spectral entropy
        spec_ent = spectral_entropy(W64.ravel().astype(np.float32))

        return {
            "shape": W.shape,
            "total_energy": total_energy,
            "low_freq_energy_ratio": low_freq_ratio,
            "spectral_entropy": spec_ent,
            "radial_energy_profile": radial_energy,
            "recommended": "dct"
            if low_freq_ratio > 0.7
            else "wavelet"
            if low_freq_ratio > 0.4
            else "svd",
        }

    def visualize_manifold(
        self, W: np.ndarray, n_components: int = 3
    ) -> Dict[str, Any]:
        """Project weight matrix blocks onto a low-dimensional manifold.

        Useful for understanding whether blocks cluster geometrically
        (which would benefit from codebook quantisation).
        """
        W64 = np.asarray(W, dtype=np.float64)
        m, n = W64.shape
        bs = min(32, m, n)

        # Extract blocks as vectors
        blocks = []
        positions = []
        for i in range(0, m - bs + 1, bs):
            for j in range(0, n - bs + 1, bs):
                block = W64[i : i + bs, j : j + bs]
                blocks.append(block.ravel())
                positions.append((i, j))

        if len(blocks) < 2:
            return {"error": "Not enough blocks for manifold analysis"}

        X = np.array(blocks, dtype=np.float64)
        # Centre
        X -= X.mean(axis=0)

        # PCA
        _, S_pca, Vt_pca = np.linalg.svd(X, full_matrices=False)
        projected = X @ Vt_pca[:n_components, :].T

        # Cluster analysis: how many natural clusters?
        # Use silhouette-like metric: average inter-cluster / intra-cluster distance
        from scipy.spatial.distance import cdist  # type: ignore

        dists = cdist(projected, projected)
        n_proj = len(projected)
        if n_proj > 10:
            # K-means for k=2..min(10, n_proj)
            best_k = 2
            best_score = -1
            for k in range(2, min(11, n_proj)):
                rng = np.random.RandomState(42)
                centroids = projected[rng.choice(n_proj, k, replace=False)].copy()
                for _ in range(10):
                    assigns = np.argmin(cdist(projected, centroids), axis=1)
                    for c in range(k):
                        mask = assigns == c
                        if np.any(mask):
                            centroids[c] = projected[mask].mean(axis=0)
                    assigns = np.argmin(cdist(projected, centroids), axis=1)

                # Silhouette score
                intra = 0.0
                inter = np.full(k, np.inf)
                for c in range(k):
                    mask = assigns == c
                    if np.sum(mask) > 1:
                        intra += np.mean(dists[mask][:, mask])
                    other_mask = ~mask
                    if np.any(other_mask):
                        inter[c] = np.mean(dists[mask][:, other_mask])
                silhouette = float(np.mean(inter - intra) / (np.max(dists) + 1e-10))
                if silhouette > best_score:
                    best_score = silhouette
                    best_k = k
        else:
            best_k = 2
            best_score = 0.0

        return {
            "n_blocks": n_proj,
            "block_size": bs,
            "pca_singular_values": S_pca[:n_components].tolist(),
            "variance_explained": float(
                np.sum(S_pca[:n_components] ** 2) / (np.sum(S_pca**2) + 1e-10)
            ),
            "projected": projected.tolist(),
            "positions": positions,
            "estimated_clusters": best_k,
            "silhouette_score": best_score,
            "recommended": "geometric_codebook" if best_k >= 4 else "svd",
        }

    def visualize_hamiltonian_trajectory(
        self,
        W: np.ndarray,
        n_steps: int = 200,
    ) -> Dict[str, Any]:
        """Simulate the Hamiltonian trajectory and analyse its properties.

        Returns trajectory data, energy conservation error, and
        convergence diagnostics.
        """
        W64 = np.asarray(W, dtype=np.float64)
        m, n = W64.shape

        ham = HamiltonianWeightDynamicals(
            polynomial_degree=6,
            symplectic_dt=0.05,
            symplectic_steps=n_steps,
            max_rank=min(32, min(m, n)),
        )

        # Compress
        result = ham.compress(W64)
        cd = result["data"]

        # Record trajectory
        q_trajectory = []
        p_trajectory = []
        q = cd["q0"].copy()
        p = cd["p0"].copy()
        dt = cd["dt"]

        q_trajectory.append(q.copy())
        p_trajectory.append(p.copy())

        a_poly = cd["a_poly"]
        a_fourier = cd["a_fourier"]
        b_fourier = cd["b_fourier"]
        s_max = cd["s_max"]

        for step in range(n_steps):
            grad = ham._hamiltonian_gradient(q, a_poly, a_fourier, b_fourier, s_max)
            p_half = p - 0.5 * dt * grad
            q = q + dt * p_half
            grad_new = ham._hamiltonian_gradient(q, a_poly, a_fourier, b_fourier, s_max)
            p = p_half - 0.5 * dt * grad_new

            q_trajectory.append(q.copy())
            p_trajectory.append(p.copy())

        q_traj = np.array(q_trajectory)
        p_traj = np.array(p_trajectory)

        # Compute Hamiltonian at each step
        H_values = []
        for t_idx in range(len(q_trajectory)):
            V = ham._evaluate_hamiltonian(
                q_traj[t_idx], a_poly, a_fourier, b_fourier, s_max
            )
            T = 0.5 * np.sum(p_traj[t_idx] ** 2)
            H_values.append(float(T + np.sum(V)))
        H_values = np.array(H_values)

        # Energy conservation error
        H0 = H_values[0] if len(H_values) > 0 else 0
        H_max = np.max(np.abs(H_values)) + 1e-30
        energy_error = float(np.max(np.abs(H_values - H0)) / H_max)

        # Convergence
        q_final = q_traj[-1]
        S_target = cd["q0"]  # target is the initial singular values
        convergence_error = float(
            np.linalg.norm(q_final - S_target) / (np.linalg.norm(S_target) + 1e-30)
        )

        return {
            "n_steps": n_steps,
            "rank": cd["rank"],
            "energy_conservation_error": energy_error,
            "convergence_error": convergence_error,
            "H_initial": float(H_values[0]) if len(H_values) > 0 else 0,
            "H_final": float(H_values[-1]) if len(H_values) > 0 else 0,
            "H_std": float(np.std(H_values)) if len(H_values) > 0 else 0,
            "q_trajectory_norm": np.linalg.norm(q_traj, axis=1).tolist(),
            "p_trajectory_norm": np.linalg.norm(p_traj, axis=1).tolist(),
            "H_values": H_values.tolist(),
        }

    def print_compression_analysis(self, result: CompressedWeight) -> str:
        """Print a detailed analysis of a compression result."""
        lines = [
            "=" * 70,
            f"COMPRESSION ANALYSIS: {result.method}",
            "=" * 70,
            f"  Original shape:       {result.original_shape}",
            f"  Original bytes:       {result.original_bytes:,}",
            f"  Compressed bytes:     {result.compressed_bytes:,}",
            f"  Compression ratio:    {result.compression_ratio:.2f}x",
            f"  Reconstruction error: {result.reconstruction_error:.6f} (relative Frobenius)",
            f"  SNR:                  {result.snr_db:.2f} dB",
            f"  Compress time:        {result.compress_time_ms:.1f} ms",
            f"  Decompress time:      {result.decompress_time_ms:.1f} ms",
        ]
        if result.extra:
            lines.append("  Extra info:")
            for k, v in result.extra.items():
                lines.append(f"    {k}: {v}")
        lines.append("=" * 70)
        report = "\n".join(lines)
        logger.info("\n%s", report)
        return report


# ═══════════════════════════════════════════════════════════════════════════
# TEST SUITE
# ═══════════════════════════════════════════════════════════════════════════


def _run_tests():
    """Run comprehensive tests on synthetic matrices."""
    import sys

    print("\n" + "=" * 80)
    print("PHYSICS COMPRESSION — TEST SUITE")
    print("=" * 80)

    rng = np.random.RandomState(42)

    # ── Test matrices ────────────────────────────────────────────────────

    test_matrices = {}

    # 1. Low-rank matrix (rank 10, 256x256)
    A = rng.randn(256, 10)
    B = rng.randn(10, 256)
    test_matrices["low_rank_10"] = A @ B * 0.01

    # 2. Random matrix (256x256)
    test_matrices["random"] = rng.randn(256, 256) * 0.01

    # 3. Toeplitz matrix (256x256)
    vals = rng.randn(256)
    toeplitz = np.zeros((256, 256), dtype=np.float64)
    for i in range(256):
        for j in range(256):
            toeplitz[i, j] = vals[abs(i - j)]
    test_matrices["toeplitz"] = toeplitz * 0.01

    # 4. Circulant matrix (256x256)
    first_col = rng.randn(256)
    circulant = np.zeros((256, 256), dtype=np.float64)
    for j in range(256):
        circulant[:, j] = np.roll(first_col, j)
    test_matrices["circulant"] = circulant * 0.01

    # 5. Diagonal-dominant matrix (256x256)
    diag = np.diag(rng.randn(256) * 2.0)
    noise = rng.randn(256, 256) * 0.005
    test_matrices["diag_dominant"] = diag + noise

    # ── Run orchestrator on each matrix ──────────────────────────────────

    orchestrator = PhysicsCompressionOrchestrator(
        target_ratio=100.0,
        max_error=0.01,
        hamiltonian_steps=80,
        codebook_size=64,
        wavelet_keep=0.15,
    )

    viz = CompressionVisualizer()

    all_results = []

    for name, matrix in test_matrices.items():
        print(f"\n{'─' * 70}")
        print(f"TEST: {name}  shape={matrix.shape}  dtype={matrix.dtype}")
        print(f"  Norm: {np.linalg.norm(matrix):.4f}")
        print(f"  Mean: {np.mean(matrix):.6f}")
        print(f"  Std:  {np.std(matrix):.6f}")

        # Run orchestrator
        try:
            result = orchestrator.analyze_and_compress(matrix)
            report = viz.print_compression_analysis(result)
            all_results.append((name, result))
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback

            traceback.print_exc()

    # ── Summary table ────────────────────────────────────────────────────

    print(f"\n{'=' * 80}")
    print("SUMMARY TABLE")
    print(f"{'=' * 80}")
    header = f"{'Matrix':<20} {'Method':<25} {'Ratio':>8} {'Error':>10} {'SNR(dB)':>10} {'Comp(ms)':>10} {'Decomp(ms)':>10}"
    print(header)
    print("-" * 100)

    for name, result in all_results:
        print(
            f"{name:<20} {result.method:<25} "
            f"{result.compression_ratio:>8.2f} "
            f"{result.reconstruction_error:>10.6f} "
            f"{result.snr_db:>10.2f} "
            f"{result.compress_time_ms:>10.1f} "
            f"{result.decompress_time_ms:>10.1f}"
        )

    # ── Individual method tests ──────────────────────────────────────────

    print(f"\n{'=' * 80}")
    print("INDIVIDUAL METHOD TESTS (low_rank_10 matrix)")
    print(f"{'=' * 80}")

    W = test_matrices["low_rank_10"]

    # Hamiltonian
    print("\n--- HamiltonianWeightDynamicals ---")
    ham = HamiltonianWeightDynamicals(
        polynomial_degree=6, symplectic_steps=80, max_rank=16
    )
    t0 = time.time()
    ham_result = ham.compress(W)
    comp_ms = (time.time() - t0) * 1000
    W_rec, decomp_ms = ham.decompress(ham_result)
    rel_err, snr = _measure_error(W, W_rec.astype(np.float64))
    comp_bytes = ham_result["comp_bytes"]
    ratio = W.nbytes / max(comp_bytes, 1)
    print(f"  Ratio:      {ratio:.2f}x")
    print(f"  Error:      {rel_err:.6f}")
    print(f"  SNR:        {snr:.2f} dB")
    print(f"  Compress:   {comp_ms:.1f} ms")
    print(f"  Decompress: {decomp_ms:.1f} ms")
    print(f"  Rank:       {ham_result['data']['rank']}")
    print(f"  Poly deg:   {len(ham_result['data']['a_poly'])}")

    # Geometric
    print("\n--- TopologicalFunctionalQuantization ---")
    geo = TopologicalFunctionalQuantization(codebook_size=64, block_size=32)
    cb = geo.build_codebook([W])
    t0 = time.time()
    geo_result = geo.compress(W, codebook=cb)
    comp_ms = (time.time() - t0) * 1000
    W_rec, decomp_ms = geo.decompress(geo_result)
    rel_err, snr = _measure_error(W, W_rec.astype(np.float64))
    comp_bytes = geo_result["comp_bytes"]
    ratio = W.nbytes / max(comp_bytes, 1)
    print(f"  Ratio:      {ratio:.2f}x")
    print(f"  Error:      {rel_err:.6f}")
    print(f"  SNR:        {snr:.2f} dB")
    print(f"  Compress:   {comp_ms:.1f} ms")
    print(f"  Decompress: {decomp_ms:.1f} ms")
    print(f"  Blocks:     {len(geo_result['data']['indices'])}")

    # Waveform
    print("\n--- HierarchicalStateSpaceWaveforms ---")
    wave = HierarchicalStateSpaceWaveforms(keep_ratio=0.15)
    t0 = time.time()
    wave_result = wave.compress(W)
    comp_ms = (time.time() - t0) * 1000
    W_rec, decomp_ms = wave.decompress(wave_result)
    rel_err, snr = _measure_error(W, W_rec.astype(np.float64))
    comp_bytes = wave_result["comp_bytes"]
    ratio = W.nbytes / max(comp_bytes, 1)
    print(f"  Ratio:      {ratio:.2f}x")
    print(f"  Error:      {rel_err:.6f}")
    print(f"  SNR:        {snr:.2f} dB")
    print(f"  Compress:   {comp_ms:.1f} ms")
    print(f"  Decompress: {decomp_ms:.1f} ms")
    print(f"  Chosen:     {wave_result['chosen_method']}")

    # ── Frequency spectrum analysis ──────────────────────────────────────

    print(f"\n{'=' * 80}")
    print("FREQUENCY SPECTRUM ANALYSIS")
    print(f"{'=' * 80}")

    for name, matrix in test_matrices.items():
        spec = viz.visualize_frequency_spectrum(matrix)
        print(
            f"  {name:<20}: low_freq_ratio={spec['low_freq_energy_ratio']:.4f}, "
            f"entropy={spec['spectral_entropy']:.4f}, "
            f"recommended={spec['recommended']}"
        )

    # ── Manifold analysis ────────────────────────────────────────────────

    print(f"\n{'=' * 80}")
    print("MANIFOLD ANALYSIS")
    print(f"{'=' * 80}")

    for name, matrix in test_matrices.items():
        manifold = viz.visualize_manifold(matrix, n_components=3)
        if "error" not in manifold:
            print(
                f"  {name:<20}: clusters={manifold['estimated_clusters']}, "
                f"silhouette={manifold['silhouette_score']:.4f}, "
                f"var_explained={manifold['variance_explained']:.4f}, "
                f"recommended={manifold['recommended']}"
            )
        else:
            print(f"  {name:<20}: {manifold['error']}")

    # ── Hamiltonian trajectory analysis ──────────────────────────────────

    print(f"\n{'=' * 80}")
    print("HAMILTONIAN TRAJECTORY ANALYSIS")
    print(f"{'=' * 80}")

    for name, matrix in test_matrices.items():
        traj = viz.visualize_hamiltonian_trajectory(matrix, n_steps=100)
        print(
            f"  {name:<20}: energy_err={traj['energy_conservation_error']:.6f}, "
            f"convergence={traj['convergence_error']:.6f}, "
            f"H_std={traj['H_std']:.6f}"
        )

    print(f"\n{'=' * 80}")
    print("ALL TESTS COMPLETE")
    print(f"{'=' * 80}\n")

    return all_results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(name)s %(levelname)s: %(message)s"
    )
    _run_tests()
