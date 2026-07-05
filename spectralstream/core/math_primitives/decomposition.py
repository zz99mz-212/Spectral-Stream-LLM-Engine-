"""Symmetric/Anti-symmetric decomposition and truncated SVD."""

from typing import Any, Dict, Optional

import numpy as np


class SymAntiSymDecomposition:
    """Decompose W = W_sym + W_anti, compress each with optimal basis."""

    @staticmethod
    def decompose(matrix: np.ndarray) -> tuple:
        matrix = np.asarray(matrix, dtype=np.float64)
        W_sym = (matrix + matrix.T) * 0.5
        W_anti = (matrix - matrix.T) * 0.5
        return W_sym, W_anti

    @staticmethod
    def compress(matrix: np.ndarray, keep_energy: float = 0.99) -> dict:
        matrix = np.asarray(matrix, dtype=np.float64)
        W_sym, W_anti = SymAntiSymDecomposition.decompose(matrix)
        eigenvalues, eigenvectors = np.linalg.eigh(W_sym)
        order = np.argsort(np.abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        total_energy = float(np.sum(eigenvalues**2))
        cumulative = np.cumsum(eigenvalues**2) / (total_energy + 1e-30)
        n_keep_sym = max(
            1, min(int(np.searchsorted(cumulative, keep_energy)) + 1, len(eigenvalues))
        )
        m = W_anti.shape[0]
        if m % 2 == 1:
            W_anti_padded = np.pad(W_anti, ((0, 1), (0, 1)))
        else:
            W_anti_padded = W_anti
        iA = 1j * W_anti_padded
        eigvals_iA = np.linalg.eigvalsh(iA)
        skew_sv = np.sort(np.abs(eigvals_iA.imag))[::-1]
        total_skew = float(np.sum(skew_sv**2))
        if total_skew > 1e-30:
            cum_skew = np.cumsum(skew_sv**2) / total_skew
            n_keep_anti = max(
                1, min(int(np.searchsorted(cum_skew, keep_energy)) + 1, len(skew_sv))
            )
        else:
            n_keep_anti = 0
        return {
            "shape": matrix.shape,
            "type": "sym_antisym",
            "sym_eigenvalues": eigenvalues[:n_keep_sym],
            "sym_eigenvectors": eigenvectors[:, :n_keep_sym],
            "sym_n_total": len(eigenvalues),
            "anti_skew_singular_values": skew_sv[:n_keep_anti],
            "anti_shape": W_anti_padded.shape,
            "keep_energy": keep_energy,
        }

    @staticmethod
    def decompress(compressed: dict) -> np.ndarray:
        ev = compressed["sym_eigenvalues"]
        V = compressed["sym_eigenvectors"]
        n = compressed["shape"][0]
        W_sym = V @ np.diag(ev) @ V.T
        skew_sv = compressed["anti_skew_singular_values"]
        m_padded = compressed["anti_shape"][0]
        m = compressed["shape"][0]
        n_blocks = len(skew_sv)
        J = np.zeros((m_padded, m_padded), dtype=np.float64)
        for k in range(n_blocks):
            J[2 * k, 2 * k + 1] = skew_sv[k]
            J[2 * k + 1, 2 * k] = -skew_sv[k]
        rng = np.random.RandomState(42)
        Q, _ = np.linalg.qr(rng.randn(m_padded, m_padded))
        W_anti_padded = Q @ J @ Q.T
        return W_sym + W_anti_padded[:m, :m]


def truncated_svd(
    matrix: np.ndarray, rank: Optional[int] = None, energy_threshold: float = 0.99
) -> Dict[str, Any]:
    mat = np.asarray(matrix, dtype=np.float64)
    if mat.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got {mat.ndim}D")
    m, n = mat.shape
    if m == 0 or n == 0:
        return {
            "U": np.zeros((m, 0)),
            "S": np.zeros(0),
            "Vt": np.zeros((0, n)),
            "rank": 0,
            "energy_retained": 0.0,
        }
    U, S, Vt = np.linalg.svd(mat, full_matrices=False)
    total_energy = float(np.sum(S**2))
    if rank is not None:
        r = min(rank, len(S))
    else:
        if total_energy < 1e-30:
            r = 1
        else:
            cum = np.cumsum(S**2) / total_energy
            r = min(int(np.searchsorted(cum, energy_threshold)) + 1, len(S))
        r = max(1, r)
    energy_retained = float(np.sum(S[:r] ** 2)) / (total_energy + 1e-30)
    return {
        "U": U[:, :r].astype(np.float32),
        "S": S[:r].astype(np.float32),
        "Vt": Vt[:r, :].astype(np.float32),
        "rank": r,
        "energy_retained": energy_retained,
    }
