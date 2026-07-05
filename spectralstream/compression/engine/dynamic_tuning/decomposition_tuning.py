# --- safek.py ---
"""Module extracted from decomposition_tuning.py — safek."""

from __future__ import annotations

import math

def _mat_shape(shape: Tuple[int, ...]) -> Tuple[int, int]:
    """Flatten trailing dims -> (m, n)."""
    m = shape[0]
    n = max(1, int(np.prod(shape[1:]))) if len(shape) > 1 else 1
    return m, n
def _safe_k(k: int, min_dim: int) -> int:
    return max(1, min(k, min_dim))
def _energy_error(singular_values: np.ndarray, k: int) -> float:
    """Relative Frobenius-norm error from truncating to rank k.

    error = sqrt(sum(sigma[k:]^2) / sum(sigma^2))
    """
    total = np.sum(singular_values**2)
    if total < 1e-30:
        return 0.0
    kept = np.sum(singular_values[:k] ** 2) if k > 0 else 0.0
    discarded = max(0.0, total - kept)
    return float(math.sqrt(discarded / total))
def _estimate_singular_values(tensor: np.ndarray, n_components: int = 32) -> np.ndarray:
    """Fast singular-value estimation via randomized SVD.

    Uses power iteration (2 passes) for stable estimation of the
    top singular values without a full SVD.
    """
    mat = tensor.reshape(tensor.shape[0], -1).astype(np.float64)
    m, n = mat.shape
    k = min(n_components, min(m, n))
    if k < 1:
        return np.array([0.0])
    if k >= min(m, n) // 2:
        _, sv, _ = np.linalg.svd(mat, full_matrices=False)
        return sv

    rng = np.random.RandomState(42)
    Q = rng.randn(n, k + 5)
    for _ in range(2):
        Q = mat.T @ (mat @ Q)
        Q, _ = np.linalg.qr(Q)
    sv = np.linalg.svd(mat @ Q, compute_uv=False)
    return sv
def _randomized_svd_error(tensor: np.ndarray, k: int) -> float:
    """Predict SVD truncation error at rank k via randomized SVD."""
    sv = _estimate_singular_values(tensor)
    return _energy_error(sv, k)
def tune_svd(tensor: np.ndarray, target_ratio: float) -> TuningResult:
    """Convenience: tune SVD to target ratio."""
    shape = tensor.shape
    m, n = _mat_shape(shape)
    k = DecompositionTuning.svd_rank_for_ratio(shape, target_ratio)
    k = _safe_k(k, min(m, n))
    ratio = m * n / (k * (m + n + 1))
    error = _randomized_svd_error(tensor, k)
    return TuningResult(
        method="svd_truncated",
        rank=k,
        params={"rank": k},
        predicted_ratio=ratio,
        predicted_error=error,
        feasible=ratio >= target_ratio * 0.9,
    )
def tune_tensor_train(
    tensor: np.ndarray, target_ratio: float, d: int = 3
) -> TuningResult:
    """Convenience: tune Tensor Train to target ratio."""
    shape = tensor.shape
    m, n = _mat_shape(shape)
    r = DecompositionTuning.tt_rank_for_ratio(shape, target_ratio, d)
    n_i = _tt_shape(m, n, d)
    prod_ni = int(np.prod(n_i))
    sum_ni = int(np.sum(n_i))
    r = _safe_k(r, min(n_i))
    ratio = prod_ni / (r * r * sum_ni)
    error = _randomized_svd_error(tensor, r)
    return TuningResult(
        method="tensor_train",
        rank=r,
        params={"rank": r, "d": d},
        predicted_ratio=ratio,
        predicted_error=error,
        feasible=ratio >= target_ratio * 0.9,
    )
def tune_cp(tensor: np.ndarray, target_ratio: float) -> TuningResult:
    """Convenience: tune CP to target ratio."""
    shape = tensor.shape
    m, n = _mat_shape(shape)
    R = DecompositionTuning.cp_rank_for_ratio(shape, target_ratio)
    R = _safe_k(R, min(m, n))
    ratio = m * n / (R * (m + n))
    error = _randomized_svd_error(tensor, R)
    return TuningResult(
        method="cp_decomposition",
        rank=R,
        params={"rank": R},
        predicted_ratio=ratio,
        predicted_error=error,
        feasible=ratio >= target_ratio * 0.9,
    )
def tune_tucker(tensor: np.ndarray, target_ratio: float) -> TuningResult:
    """Convenience: tune Tucker to target ratio."""
    shape = tensor.shape
    ranks = DecompositionTuning.tucker_ranks_for_ratio(shape, target_ratio)
    r = ranks[0]
    m, n = _mat_shape(shape)
    r = _safe_k(r, min(m, n))
    core_size = r * r
    factor_size = m * r + n * r
    total_params = core_size + factor_size
    ratio = m * n / max(total_params, 1)
    error = _randomized_svd_error(tensor, r)
    return TuningResult(
        method="tucker_decomposition",
        rank=r,
        params={"ranks": ranks},
        predicted_ratio=ratio,
        predicted_error=error,
        feasible=ratio >= target_ratio * 0.9,
    )
def tune_butterfly(tensor: np.ndarray, target_ratio: float) -> TuningResult:
    """Convenience: tune Butterfly to target ratio."""
    shape = tensor.shape
    m, n = _mat_shape(shape)
    levels = max(1, int(math.log2(min(m, n))))
    k = int(m * n / (target_ratio * levels * (m + n)))
    k = _safe_k(k, min(m, n))
    ratio = m * n / (k * levels * (m + n))
    error = _randomized_svd_error(tensor, k)
    return TuningResult(
        method="butterfly",
        rank=k,
        params={"n_levels": levels, "rank": k},
        predicted_ratio=ratio,
        predicted_error=error,
        feasible=ratio >= target_ratio * 0.9,
    )
def tune_kronecker(tensor: np.ndarray, target_ratio: float) -> TuningResult:
    """Convenience: tune Kronecker to target ratio."""
    shape = tensor.shape
    m, n_val = _mat_shape(shape)
    k = int(math.sqrt(m * n_val / target_ratio / 2))
    k = _safe_k(k, min(m, n_val))
    total_params = 2 * k * k
    ratio = m * n_val / max(total_params, 1)
    error = _randomized_svd_error(tensor, k)
    return TuningResult(
        method="kronecker",
        rank=k,
        params={"shape_a": (k, k)},
        predicted_ratio=ratio,
        predicted_error=error,
        feasible=ratio >= target_ratio * 0.9,
    )
def tune_cur(tensor: np.ndarray, target_ratio: float) -> TuningResult:
    """Convenience: tune CUR to target ratio."""
    shape = tensor.shape
    m, n_val = _mat_shape(shape)
    k = DecompositionTuning.cur_rank_for_ratio(shape, target_ratio)
    k = _safe_k(k, min(m, n_val))
    total_params = k * (m + n_val + k)
    ratio = m * n_val / max(total_params, 1)
    error = _randomized_svd_error(tensor, k)
    return TuningResult(
        method="cur_decomposition",
        rank=k,
        params={"rank": k},
        predicted_ratio=ratio,
        predicted_error=error,
        feasible=ratio >= target_ratio * 0.9,
    )
def tune_nystrom(tensor: np.ndarray, target_ratio: float) -> TuningResult:
    """Convenience: tune Nystrom to target ratio."""
    shape = tensor.shape
    m, n_val = _mat_shape(shape)
    k = DecompositionTuning.nystrom_rank_for_ratio(shape, target_ratio)
    k = _safe_k(k, min(m, n_val))
    total_params = k * (m + n_val + k)
    ratio = m * n_val / max(total_params, 1)
    error = _randomized_svd_error(tensor, k)
    return TuningResult(
        method="nystrom",
        rank=k,
        params={"rank": k},
        predicted_ratio=ratio,
        predicted_error=error,
        feasible=ratio >= target_ratio * 0.9,
    )
def tune_random_feature(tensor: np.ndarray, target_ratio: float) -> TuningResult:
    """Convenience: tune Random Feature to target ratio."""
    shape = tensor.shape
    m, n_val = _mat_shape(shape)
    k = DecompositionTuning.random_feature_count(shape, target_ratio)
    k = _safe_k(k, min(m, n_val))
    total_params = k * (m + n_val)
    ratio = m * n_val / max(total_params, 1)
    error = _randomized_svd_error(tensor, k)
    return TuningResult(
        method="random_feature",
        rank=k,
        params={"n_features": k},
        predicted_ratio=ratio,
        predicted_error=error,
        feasible=ratio >= target_ratio * 0.9,
    )
def tune_structured(
    method: str, tensor: np.ndarray, target_ratio: float
) -> TuningResult:
    """Convenience: tune structured matrix to target ratio."""
    shape = tensor.shape
    d = shape[0]
    if method == "block_diagonal":
        b = DecompositionTuning.block_diagonal_block_size(shape, target_ratio)
        total_params = d * b
        ratio = d * d / max(total_params, 1)
        error = _randomized_svd_error(tensor, max(1, b))
        return TuningResult(
            method="block_diagonal",
            rank=b,
            params={"block_size": b},
            predicted_ratio=ratio,
            predicted_error=error,
            feasible=True,
        )
    elif method == "toeplitz":
        ratio = DecompositionTuning.toeplitz_ratio(shape)
        return TuningResult(
            method="toeplitz",
            rank=1,
            params={"method": "toeplitz"},
            predicted_ratio=ratio,
            predicted_error=0.5,
            feasible=ratio >= target_ratio,
        )
    elif method == "hankel":
        ratio = DecompositionTuning.hankel_ratio(shape)
        return TuningResult(
            method="hankel",
            rank=1,
            params={"method": "hankel"},
            predicted_ratio=ratio,
            predicted_error=0.5,
            feasible=ratio >= target_ratio,
        )
    else:
        raise ValueError(f"Unknown structured method: {method}")
def _tt_shape(m: int, n: int, d: int) -> List[int]:
    """Reshape m x n matrix into d-dimensional TT shape."""
    if d < 2:
        return [m, n]
    if d == 2:
        return [m, n]
    factors = _factorize(m * n, d)
    return factors
def _factorize(n: int, d: int) -> List[int]:
    """Partition n into d roughly equal integer factors."""
    if d <= 1:
        return [n]
    factors = [1] * d
    remaining = n
    for i in range(d - 1, 0, -1):
        factors[i] = max(1, int(round(remaining ** (1.0 / (i + 1)))))
        while factors[i] > 1 and remaining % factors[i] != 0:
            factors[i] -= 1
        remaining //= factors[i]
    factors[0] = remaining
    factors = [max(1, f) for f in factors]
    return factors
def _build_method_map() -> Dict[str, Any]:
    """Build map from method name to tuning handler."""

    def _svd_handler(tensor, shape, target_ratio):
        return tune_svd(tensor, target_ratio)

    def _tt_handler(tensor, shape, target_ratio):
        return tune_tensor_train(tensor, target_ratio)

    def _cp_handler(tensor, shape, target_ratio):
        return tune_cp(tensor, target_ratio)

    def _tucker_handler(tensor, shape, target_ratio):
        return tune_tucker(tensor, target_ratio)

    def _block_tucker_handler(tensor, shape, target_ratio):
        m, n = _mat_shape(shape)
        params = DecompositionTuning.block_tucker_params(shape, target_ratio)
        nb = params["n_blocks"]
        rf = params["rank_frac"]
        block_dim = max(m // nb, 1)
        block_r = max(1, int(rf * block_dim))
        total = nb * (block_r * block_r + 2 * block_dim * block_r)
        ratio = m * n / max(total, 1)
        error = _randomized_svd_error(tensor, block_r)
        return TuningResult(
            method="block_tucker",
            rank=block_r,
            params=params,
            predicted_ratio=ratio,
            predicted_error=error,
            feasible=ratio >= target_ratio * 0.9,
        )

    def _ht_handler(tensor, shape, target_ratio):
        r = DecompositionTuning.hierarchical_tucker_rank_for_ratio(shape, target_ratio)
        m, n = _mat_shape(shape)
        r = _safe_k(r, min(m, n))
        ratio = m * n / (r * (m + n + 1))
        error = _randomized_svd_error(tensor, r)
        return TuningResult(
            method="hierarchical_tucker",
            rank=r,
            params={"rank": r},
            predicted_ratio=ratio,
            predicted_error=error,
            feasible=ratio >= target_ratio * 0.9,
        )

    def _butterfly_handler(tensor, shape, target_ratio):
        return tune_butterfly(tensor, target_ratio)

    def _monarch_handler(tensor, shape, target_ratio):
        bs = DecompositionTuning.monarch_block_size(shape, target_ratio)
        m, n = _mat_shape(shape)
        ratio = m * n / (bs * (m + n))
        error = _randomized_svd_error(tensor, bs)
        return TuningResult(
            method="monarch",
            rank=bs,
            params={"block_size": bs},
            predicted_ratio=ratio,
            predicted_error=error,
            feasible=ratio >= target_ratio * 0.9,
        )

    def _kronecker_handler(tensor, shape, target_ratio):
        return tune_kronecker(tensor, target_ratio)

    def _cur_handler(tensor, shape, target_ratio):
        return tune_cur(tensor, target_ratio)

    def _nystrom_handler(tensor, shape, target_ratio):
        return tune_nystrom(tensor, target_ratio)

    def _rf_handler(tensor, shape, target_ratio):
        return tune_random_feature(tensor, target_ratio)

    def _h_matrix_handler(tensor, shape, target_ratio):
        bs = DecompositionTuning.h_matrix_block_size(shape, target_ratio)
        m, n = _mat_shape(shape)
        ratio = m * n / (bs * (m + n))
        error = _randomized_svd_error(tensor, bs)
        return TuningResult(
            method="h_matrix",
            rank=bs,
            params={"block_size": bs, "eps": 0.01},
            predicted_ratio=ratio,
            predicted_error=error,
            feasible=ratio >= target_ratio * 0.9,
        )

    def _bd_handler(tensor, shape, target_ratio):
        return tune_structured("block_diagonal", tensor, target_ratio)

    def _toeplitz_handler(tensor, shape, target_ratio):
        return tune_structured("toeplitz", tensor, target_ratio)

    def _hankel_handler(tensor, shape, target_ratio):
        return tune_structured("hankel", tensor, target_ratio)

    return {
        "svd_truncated": _svd_handler,
        "tensor_train": _tt_handler,
        "tensor_ring": _tt_handler,
        "tt_orthogonal": _tt_handler,
        "tt_svd": _tt_handler,
        "cp_decomposition": _cp_handler,
        "tucker_decomposition": _tucker_handler,
        "block_tucker": _block_tucker_handler,
        "hierarchical_tucker": _ht_handler,
        "butterfly": _butterfly_handler,
        "monarch": _monarch_handler,
        "kronecker": _kronecker_handler,
        "cur_decomposition": _cur_handler,
        "nystrom": _nystrom_handler,
        "random_feature": _rf_handler,
        "h_matrix": _h_matrix_handler,
        "block_diagonal": _bd_handler,
        "toeplitz": _toeplitz_handler,
        "hankel": _hankel_handler,
    }