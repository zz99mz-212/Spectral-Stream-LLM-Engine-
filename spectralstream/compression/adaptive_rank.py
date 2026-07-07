"""Adaptive rank estimation and matrix structure detection for real weight distributions.

Real neural network weights have specific statistical properties:
- SVD singular values decay but NOT at a fixed rate
- Kronecker/Toeplitz/Hankel structural assumptions rarely hold
- Adaptive rank selection via knee detection works well
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np


def _randomized_svd(
    X: np.ndarray,
    n_components: int,
    n_oversamples: int = 10,
    n_iter: int = 2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fast randomized SVD for large matrices."""
    m, n = X.shape
    actual_rank = min(m, n)
    k = min(n_components, actual_rank - 1) if actual_rank > 1 else 1
    k = max(k, 1)

    if X.size < 10000 or actual_rank < 10 or k >= actual_rank // 2:
        try:
            U, S, Vh = np.linalg.svd(X, full_matrices=False)
        except np.linalg.LinAlgError:
            k = min(actual_rank, k)
            U = np.eye(m, k, dtype=X.dtype)
            S = np.ones(k, dtype=X.dtype)
            Vh = np.eye(k, n, dtype=X.dtype)
            return U[:, :k], S[:k], Vh[:k, :]
        k = min(n_components, len(S))
        return (
            U[:, :k].astype(X.dtype),
            S[:k].astype(X.dtype),
            Vh[:k, :].astype(X.dtype),
        )

    try:
        rng = np.random.default_rng(random_state)
        oversampled = min(k + n_oversamples, n)
        O = rng.normal(0.0, 1.0, (n, oversampled)).astype(X.dtype, copy=False)
        Y = X @ O
        for _ in range(n_iter):
            Y = X @ (X.T @ Y)
        Q, _ = np.linalg.qr(Y)
        B = Q.T @ X
        U_hat, S, Vh = np.linalg.svd(B, full_matrices=False)
        k_actual = min(k, len(S))
        U = Q @ U_hat[:, :k_actual]
        S = S[:k_actual].copy()
        Vh = Vh[:k_actual, :].copy()
    except np.linalg.LinAlgError:
        try:
            U, S, Vh = np.linalg.svd(X, full_matrices=False)
        except np.linalg.LinAlgError:
            U = np.eye(m, k, dtype=X.dtype)
            S = np.ones(k, dtype=X.dtype)
            Vh = np.eye(k, n, dtype=X.dtype)
            return U[:, :k], S[:k], Vh[:k, :]
        k = min(n_components, len(S))
        return (
            U[:, :k].astype(X.dtype),
            S[:k].astype(X.dtype),
            Vh[:k, :].astype(X.dtype),
        )
    return U, S, Vh


def estimate_adaptive_rank(
    tensor: np.ndarray,
    energy_threshold: float = 0.999,
    max_rank: Optional[int] = None,
    svd_samples: int = 1024,
) -> int:
    """Find optimal rank from SVD singular value decay.

    Uses randomized SVD for large matrices and finds the "knee" in the
    singular value curve using cumulative energy or elbow detection.

    Parameters
    ----------
    tensor : np.ndarray
        Input matrix (will be reshaped to 2D if necessary).
    energy_threshold : float
        Fraction of spectral energy to preserve (default 0.999 = 99.9%).
    max_rank : int, optional
        Maximum allowed rank. Defaults to min(shape) // 2.
    svd_samples : int
        Number of singular values to compute for large matrices.

    Returns
    -------
    int
        Selected rank (always >= 2).
    """
    t = tensor.reshape(tensor.shape[0], -1) if tensor.ndim > 2 else tensor
    m, n = t.shape
    k = min(m, n)
    if k < 4 or t.size < 1024:
        return max(1, k // 8)
    if max_rank is None:
        max_rank = max(2, k // 2)
    compute_k = min(max_rank, k - 1, svd_samples)
    compute_k = max(compute_k, 1)
    try:
        U, S, Vh = np.linalg.svd(t, full_matrices=False)
        limit = min(max_rank, len(S))
        S = S[:limit]
        total_energy = float(np.sum(S**2))
    except np.linalg.LinAlgError:
        return max(1, k // 10)

    if len(S) < 2 or S[0] <= 0:
        return max(1, k // 10)

    if total_energy < 1e-30:
        return max(1, k // 10)

    cumsum = np.cumsum(S**2)
    rank_by_energy = int(np.searchsorted(cumsum / total_energy, energy_threshold) + 1)

    # Knee detection: very conservative — only use when there is an EXTREME drop
    # where one singular value is > 100x larger than the next meaningful one
    if len(S) >= 4 and not use_randomized:
        log_s = np.log(np.maximum(S, 1e-30))
        diffs = np.diff(log_s)
        if len(diffs) > 1:
            abs_diffs = np.abs(diffs)
            max_drop_idx = int(np.argmin(diffs))
            max_drop_val = abs_diffs[max_drop_idx]
            # Require the maximum drop to be > 10x the mean of ALL other drops
            other_diffs = np.concatenate(
                [abs_diffs[:max_drop_idx], abs_diffs[max_drop_idx + 1 :]]
            )
            mean_other = float(np.mean(other_diffs)) if len(other_diffs) > 0 else 0.0
            if mean_other > 0 and max_drop_val > 10.0 * mean_other:
                # Only use knee if it's clearly below the energy-based rank
                if max_drop_idx + 1 < rank_by_energy:
                    rank_by_energy = max_drop_idx + 1

    selected = max(2, min(rank_by_energy, max_rank, k - 1))
    return selected


def estimate_kronecker_fit_error(tensor: np.ndarray) -> float:
    """Compute ||W - A⊗B|| / ||W|| to measure Kronecker structure fit.

    Returns relative Frobenius norm error (0 = perfect Kronecker, 1 = no fit).

    Uses subsampled matrix for speed and tests only likely factorizations
    near sqrt(m) and sqrt(n).
    """
    t = np.asarray(tensor, dtype=np.float64)
    if t.ndim != 2 or min(t.shape) < 4:
        return 1.0
    m, n = t.shape
    t_norm = float(np.linalg.norm(t, "fro")) + 1e-30

    # For small matrices (<=128), use full matrix — preserves Kronecker structure
    # For large matrices (>128), real neural net weights are NEVER Kronecker,
    # so the fit error will be ~1.0 regardless. Use a strided subsample for speed.
    if m > 128 or n > 128:
        step_m = max(1, m // 64)
        step_n = max(1, n // 64)
        t = t[::step_m, ::step_n]
        m, n = t.shape

    best_err = 1.0

    # Find all divisors of m and n up to a reasonable limit
    candidates_a = set()
    # Add divisors near sqrt
    sqrt_m = int(math.isqrt(m))
    sqrt_n = int(math.isqrt(n))
    for base in (sqrt_m, sqrt_n):
        for offset in range(-2, 3):
            val = max(2, base + offset)
            if m % val == 0:
                candidates_a.add(val)
            if n % val == 0:
                candidates_a.add(val)
    # Add small divisors (2, 3, 4, ...), these catch cases like (5x5)⊗(20x20)
    for d in range(2, min(17, m, n)):
        if m % d == 0:
            candidates_a.add(d)
        if n % d == 0:
            candidates_a.add(d)

    # Limit to 10 candidates maximum for speed
    candidates = sorted(candidates_a)[:10]
    for a in candidates:
        if m % a != 0:
            continue
        b = m // a
        for c_div in sorted(candidates_a):
            if n % c_div != 0:
                continue
            d = n // c_div
            W_r = (
                t.reshape(a, b, c_div, d)
                .transpose(0, 2, 1, 3)
                .reshape(a * c_div, b * d)
            )
            try:
                U, s, Vt = np.linalg.svd(W_r, full_matrices=False)
                if len(s) < 1 or s[0] <= 0:
                    continue
                scale = math.sqrt(max(s[0], 0.0))
                A_mat = scale * U[:, 0].reshape(a, c_div)
                B_mat = scale * Vt[0, :].reshape(b, d)
                recon = np.kron(A_mat, B_mat)
                err = float(np.linalg.norm(t - recon, "fro") / t_norm)
                if err < best_err:
                    best_err = err
                if best_err < 0.01:
                    return best_err
            except np.linalg.LinAlgError:
                continue

    return best_err


def estimate_toeplitz_fit_error(tensor: np.ndarray) -> float:
    """Compute ||W - Toeplitz(W)|| / ||W|| for Toeplitz structure.

    Returns relative Frobenius norm error (0 = perfect Toeplitz, 1 = no fit).
    """
    t = np.asarray(tensor, dtype=np.float64)
    if t.ndim != 2 or min(t.shape) < 4:
        return 1.0
    m, n = t.shape
    w = np.zeros(m + n - 1)
    for k in range(-(m - 1), n):
        diag = np.diag(t, k=k)
        w[k + m - 1] = np.mean(diag) if len(diag) > 0 else 0.0
    i = np.arange(m)[:, None]
    j = np.arange(n)[None, :]
    recon = w[j - i + m - 1]
    err = float(np.linalg.norm(t - recon, "fro") / (np.linalg.norm(t, "fro") + 1e-30))
    return err


def estimate_hankel_fit_error(tensor: np.ndarray) -> float:
    """Compute ||W - Hankel(W)|| / ||W|| for Hankel structure.

    Returns relative Frobenius norm error (0 = perfect Hankel, 1 = no fit).
    """
    t = np.asarray(tensor, dtype=np.float64)
    if t.ndim != 2 or min(t.shape) < 4:
        return 1.0
    m, n = t.shape
    w = np.zeros(m + n - 1)
    for k in range(m + n - 1):
        i0 = max(0, k - n + 1)
        i1 = min(m, k + 1)
        if i0 < i1:
            w[k] = float(np.mean(t[np.arange(i0, i1), k - np.arange(i0, i1)]))
        else:
            w[k] = 0.0
    i = np.arange(m)[:, None]
    j = np.arange(n)[None, :]
    recon = w[i + j]
    err = float(np.linalg.norm(t - recon, "fro") / (np.linalg.norm(t, "fro") + 1e-30))
    return err


def detect_matrix_structure(tensor: np.ndarray) -> Dict[str, float]:
    """Analyze matrix structure and return fitness scores.

    Returns dict with keys:
    - 'low_rank': how well SVD low-rank approximation works (0-1, higher = better)
    - 'kronecker': Kronecker product fit score (0-1, higher = better fit)
    - 'toeplitz': Toeplitz structure score (0-1, higher = better fit)
    - 'sparse': sparsity score (0-1, higher = sparser)
    - 'block_sparse': block sparsity score (0-1)
    - 'circulant': circulant structure score (0-1)
    """
    t = np.asarray(tensor, dtype=np.float64)
    if t.ndim != 2 or min(t.shape) < 4:
        return {
            "low_rank": 0.0,
            "kronecker": 0.0,
            "toeplitz": 0.0,
            "sparse": 0.0,
            "block_sparse": 0.0,
            "circulant": 0.0,
        }

    m, n = t.shape
    k = min(m, n)
    t_norm = float(np.linalg.norm(t, "fro")) + 1e-30

    # Low-rank score: what fraction of energy is captured by top min(k, 64) singular values?
    low_rank_score = 0.0
    try:
        compute_k = min(k, 64)
        U, S, Vh = _randomized_svd(t, compute_k, n_oversamples=5, n_iter=1)
        total_energy = float(np.sum(S**2))
        # Check how many singular values needed for 99% energy
        if total_energy > 1e-30:
            cum = np.cumsum(S**2) / total_energy
            rank_99 = int(np.searchsorted(cum, 0.99) + 1)
            # Score = how concentrated: 1 - (rank_99 / compute_k)
            low_rank_score = max(0.0, 1.0 - (rank_99 / max(compute_k, 1)))
            # Bonus if first singular value dominates
            if len(S) > 0 and S[0] > 0:
                first_ratio = float(S[0] ** 2) / total_energy
                low_rank_score = min(1.0, low_rank_score * 0.5 + first_ratio * 0.5)
    except np.linalg.LinAlgError:
        low_rank_score = 0.5

    # Kronecker fit score
    kronecker_err = estimate_kronecker_fit_error(tensor)
    kronecker_score = max(0.0, 1.0 - kronecker_err)
    # Penalize: random matrices with subsampled factorizations can appear
    # to have moderate Kronecker fit. Require very low error (< 5%) to count.
    if kronecker_err > 0.05:
        kronecker_score *= 0.1  # Heavily penalize non-perfect Kronecker fit

    # Toeplitz fit score
    toeplitz_err = estimate_toeplitz_fit_error(tensor)
    toeplitz_score = max(0.0, 1.0 - toeplitz_err)

    # Sparsity score
    abs_t = np.abs(t)
    sparse_score = float(np.mean(abs_t < 0.001))

    # Block sparsity score (check 16x16 blocks)
    block_sparse_score = 0.0
    if m >= 16 and n >= 16:
        blocks_m = m // 16
        blocks_n = n // 16
        if blocks_m >= 1 and blocks_n >= 1:
            block_norms = np.zeros((blocks_m, blocks_n), dtype=np.float64)
            for bi in range(blocks_m):
                for bj in range(blocks_n):
                    b = t[bi * 16 : (bi + 1) * 16, bj * 16 : (bj + 1) * 16]
                    block_norms[bi, bj] = float(np.linalg.norm(b, "fro"))
            threshold = float(np.percentile(block_norms, 50))
            block_sparse_score = (
                float(np.mean(block_norms < threshold)) if block_norms.size > 0 else 0.0
            )

    # Circulant structure score
    circulant_score = 0.0
    if m == n and n >= 4:
        sub = t[: min(32, n), : min(32, n)]
        first_col = sub[:, 0]
        sz = min(len(first_col), sub.shape[1])
        if sz > 1:
            scores = np.zeros(sz, dtype=np.float64)
            for j in range(sz):
                expected = np.roll(first_col, j)[:sz]
                actual = sub[:sz, j]
                corr = np.corrcoef(expected, actual)
                scores[j] = (
                    float(np.abs(corr[0, 1])) if not np.isnan(corr[0, 1]) else 0.0
                )
            circulant_score = float(np.mean(scores)) if sz > 0 else 0.0

    return {
        "low_rank": float(low_rank_score),
        "kronecker": float(kronecker_score),
        "toeplitz": float(toeplitz_score),
        "sparse": float(sparse_score),
        "block_sparse": float(block_sparse_score),
        "circulant": float(circulant_score),
    }


def suggest_methods_by_structure(tensor: np.ndarray) -> List[Dict[str, any]]:
    """Suggest compression methods based on matrix structure analysis.

    Returns ranked list of method suggestions with scores.
    """
    scores = detect_matrix_structure(tensor)
    suggestions = []

    # Low-rank: recommend SVD-based methods
    if scores["low_rank"] > 0.7:
        suggestions.append(
            {
                "method_type": "svd",
                "score": scores["low_rank"],
                "reason": f"low_rank_structure={scores['low_rank']:.2f}",
                "methods": ["svd_compress", "svd_truncated"],
            }
        )
    elif scores["low_rank"] > 0.5:
        suggestions.append(
            {
                "method_type": "svd",
                "score": scores["low_rank"] * 0.8,
                "reason": f"moderate_low_rank={scores['low_rank']:.2f}",
                "methods": ["svd_compress"],
            }
        )

    # Kronecker: only recommend if fit is good
    if scores["kronecker"] > 0.9:
        suggestions.append(
            {
                "method_type": "kronecker",
                "score": scores["kronecker"],
                "reason": f"kronecker_fit={scores['kronecker']:.2f}",
                "methods": ["kronecker"],
            }
        )

    # Toeplitz/Hankel: only if fit is very good (< 10% error)
    if scores["toeplitz"] > 0.9:
        suggestions.append(
            {
                "method_type": "toeplitz",
                "score": scores["toeplitz"] * 0.9,
                "reason": f"toeplitz_fit={scores['toeplitz']:.2f}",
                "methods": ["toeplitz"],
            }
        )

    # Sparsity-based methods
    if scores["sparse"] > 0.5:
        suggestions.append(
            {
                "method_type": "sparsity",
                "score": scores["sparse"],
                "reason": f"sparsity={scores['sparse']:.2f}",
                "methods": ["sparsity_int4", "unstructured_pruning"],
            }
        )

    # Quantization works on ANY distribution — always include as fallback
    suggestions.append(
        {
            "method_type": "quantization",
            "score": 0.3,
            "reason": "universal_fallback",
            "methods": ["block_int8", "block_int4", "hadamard_int8"],
        }
    )

    suggestions.sort(key=lambda x: -x["score"])
    return suggestions
