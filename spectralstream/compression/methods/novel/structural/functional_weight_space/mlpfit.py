"""Module extracted from functional_weight_space.py — mlpfit."""

from __future__ import annotations

import math
import struct
from typing import Callable, Tuple

import numpy as np

ActType = Callable[[np.ndarray], np.ndarray]
from spectralstream.compression.methods.novel._wrap import _BlockINT8Wrapper


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()


def _deserialize(data: bytes, dtype: type = np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()


def _should_fallback(
    tensor: np.ndarray, n_params: int, ratio_thresh: float = 1.5
) -> bool:
    """Use BlockINT8 if functional params are too large relative to tensor."""
    func_bytes = n_params * 4  # float32
    orig_bytes = tensor.nbytes
    return func_bytes * ratio_thresh >= orig_bytes


def _siren_fit(
    flat: np.ndarray,
    coords: np.ndarray,
    hidden: int,
    n_layers: int,
    n_epochs: int,
    act: ActType,
    freq: float = 1.0,
    lr: float = 0.005,
) -> tuple:
    """Fit SIREN using random features + least-squares output layer (chunked).

    Freeze all hidden layers (random init) and learn only the output linear
    layer via closed-form chunked least squares.
    """
    rng = np.random.RandomState(42)
    rng_j = np.random.RandomState(7)
    n = len(flat)
    w1 = (rng.randn(1, hidden) * 1.0 / freq).astype(np.float32)
    b1 = (rng.uniform(-np.pi, np.pi, hidden)).astype(np.float32)
    weights = [(w1, b1)]
    for _ in range(n_layers - 1):
        w = (rng_j.randn(hidden, hidden) * np.sqrt(2.0 / hidden)).astype(np.float32)
        b = (rng_j.uniform(-np.pi, np.pi, hidden)).astype(np.float32)
        weights.append((w, b))

    def _features_aug(sl: slice) -> np.ndarray:
        c = coords[sl].reshape(-1, 1)
        h = c
        for w, b in weights:
            h = act(h @ w + b)
        h = h.astype(np.float64)
        return np.column_stack([h, np.ones(len(h), dtype=np.float64)])

    n_dims = hidden + 1
    x = _chunked_lstsq(_features_aug, flat, n_dims)
    wo = x[:hidden].astype(np.float32)
    bo = float(x[hidden])
    w1_out, b1_out = weights[0]
    w_mid_out = [w for w, b in weights[1:]]
    b_mid_out = [b for w, b in weights[1:]]
    return w1_out, b1_out, w_mid_out, b_mid_out, wo, bo


def _siren_eval(
    coords: np.ndarray,
    w1: np.ndarray,
    b1: np.ndarray,
    w_mid: List[np.ndarray],
    b_mid: List[np.ndarray],
    wo: np.ndarray,
    bo: float,
    act: ActType,
) -> np.ndarray:
    x = coords.reshape(-1, 1)
    h = act(x @ w1 + b1)
    for wi, bi in zip(w_mid, b_mid):
        h = act(h @ wi + bi)
    return (h @ wo + bo).ravel()


def _serialize_siren(
    w1, b1, w_mid, b_mid, wo, bo, n, shape, hidden, n_layers
) -> Tuple[bytes, dict]:
    data = _serialize(w1) + _serialize(b1) + _serialize(wo) + _serialize(np.array([bo]))
    for w, b in zip(w_mid, b_mid):
        data += _serialize(w) + _serialize(b)
    meta = dict(
        hidden_dim=hidden,
        n_layers=n_layers,
        n=n,
        shape=tuple(shape),
    )
    return data, meta


def _deserialize_siren(data: bytes, meta: dict) -> tuple:
    hidden = meta["hidden_dim"]
    n_layers = meta["n_layers"]
    pos = 0
    w1 = _deserialize(data[pos : pos + hidden * 4]).reshape(1, hidden)
    pos += hidden * 4
    b1 = _deserialize(data[pos : pos + hidden * 4])
    pos += hidden * 4
    wo = _deserialize(data[pos : pos + hidden * 4])
    pos += hidden * 4
    bo = float(_deserialize(data[pos : pos + 4])[0])
    pos += 4
    w_mid, b_mid = [], []
    for _ in range(n_layers - 1):
        w_mid.append(
            _deserialize(data[pos : pos + hidden * hidden * 4]).reshape(hidden, hidden)
        )
        pos += hidden * hidden * 4
        b_mid.append(_deserialize(data[pos : pos + hidden * 4]))
        pos += hidden * 4
    return w1, b1, w_mid, b_mid, wo, bo


def _siren_compress(
    tensor: np.ndarray,
    act: ActType,
    hidden: int = 32,
    n_layers: int = 3,
    n_epochs: int = 100,
    freq: float = 1.0,
) -> Tuple[bytes, dict]:
    flat = tensor.ravel().astype(np.float64)
    n = len(flat)
    coords = np.linspace(-1, 1, n).astype(np.float32)
    n_params = 3 * hidden + 1 + (n_layers - 1) * hidden * (hidden + 1)
    if _should_fallback(tensor, n_params):
        return _BlockINT8Wrapper.compress(tensor)
    w1, b1, w_mid, b_mid, wo, bo = _siren_fit(
        flat, coords, hidden, n_layers, n_epochs, act, freq
    )
    return _serialize_siren(
        w1, b1, w_mid, b_mid, wo, bo, n, tensor.shape, hidden, n_layers
    )


def _siren_decompress(data: bytes, meta: dict, act: ActType) -> np.ndarray:
    if meta.get("_fallback"):
        return _BlockINT8Wrapper.decompress(data, meta)
    w1, b1, w_mid, b_mid, wo, bo = _deserialize_siren(data, meta)
    n = meta["n"]
    coords = np.linspace(-1, 1, n).astype(np.float32)
    result = _siren_eval(coords, w1, b1, w_mid, b_mid, wo, bo, act)
    return result[:n].reshape(meta["shape"]).astype(np.float32)


def _chunked_lstsq(
    basis_fn, flat: np.ndarray, n_dims: int, chunk_size: int = 1 << 20
) -> np.ndarray:
    """Solve min_x ||A x - b||^2 via chunked normal equations + regularized solve.

    Builds A^T A (d×d) and A^T b (d×1) incrementally using float32 accumulation
    to avoid O(n×d) memory.  Converts to float64 only for the final d×d solve.
    Adds Tikhonov regularization for numerical stability.
    """
    n = len(flat)
    target_f32 = flat.astype(np.float32)
    ATA = np.zeros((n_dims, n_dims), dtype=np.float64)
    ATb = np.zeros(n_dims, dtype=np.float64)
    _eye = np.eye(n_dims, dtype=np.float64)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        A_chunk = basis_fn(slice(start, end))  # (m, d) float32
        # BLAS syrk for A^T A in float32
        ata_f32 = A_chunk.T @ A_chunk  # (d, d) float32
        ATA += ata_f32.astype(np.float64)
        # A^T b
        atb_f32 = A_chunk.T @ target_f32[start:end]  # (d,) float32
        ATb += atb_f32.astype(np.float64)
    # Regularized solve: (A^T A + λI)^{-1} A^T b
    lam = 1e-8 * float(np.trace(ATA)) / max(n_dims, 1)
    x, _, _, _ = np.linalg.lstsq(ATA + lam * _eye, ATb, rcond=None)
    return x.astype(np.float32)


def _fourier_fit(
    flat: np.ndarray,
    coords: np.ndarray,
    n_feats: int,
    sigma: float,
) -> Tuple[np.ndarray, np.ndarray, float]:
    n = len(flat)
    rng = np.random.RandomState(42)
    B = (rng.randn(n_feats) * sigma).astype(np.float32)

    def _phi(sl: slice) -> np.ndarray:
        c = coords[sl]
        proj = c[:, None] * B[None, :]
        return np.concatenate([np.sin(proj), np.cos(proj)], axis=1)

    n_dims = 2 * n_feats + 1

    def _basis_aug(sl: slice) -> np.ndarray:
        ph = _phi(sl)
        return np.column_stack([ph, np.ones(len(ph), dtype=np.float32)])

    x = _chunked_lstsq(_basis_aug, flat, n_dims)
    w = x[: 2 * n_feats].ravel()
    b = float(x[2 * n_feats])
    return B, w, b


def _fourier_eval(
    coords: np.ndarray, B: np.ndarray, w: np.ndarray, b: float
) -> np.ndarray:
    proj = coords[:, None] * B[None, :]
    phi = np.concatenate([np.sin(proj), np.cos(proj)], axis=1)
    return phi @ w + b


def _fourier_compress(
    tensor: np.ndarray,
    n_feats: int = 64,
    sigma: float = 1.0,
) -> Tuple[bytes, dict]:
    flat = tensor.ravel().astype(np.float64)
    n = len(flat)
    n_params = n_feats * 2 + 1
    if _should_fallback(tensor, n_params):
        return _BlockINT8Wrapper.compress(tensor)
    coords = np.linspace(-1, 1, n).astype(np.float32)
    B, w, bv = _fourier_fit(flat, coords, n_feats, sigma)
    data = _serialize(B) + _serialize(w) + struct.pack("<f", bv)
    meta = dict(n_feats=n_feats, sigma=sigma, n=n, shape=tensor.shape)
    return data, meta


def _fourier_decompress(data: bytes, meta: dict) -> np.ndarray:
    if meta.get("_fallback"):
        return _BlockINT8Wrapper.decompress(data, meta)
    n_feats = meta["n_feats"]
    B = _deserialize(data[: n_feats * 4])
    w = _deserialize(data[n_feats * 4 : n_feats * 12])
    bv = struct.unpack_from("<f", data, n_feats * 12)[0]
    n = meta["n"]
    coords = np.linspace(-1, 1, n).astype(np.float32)
    result = _fourier_eval(coords, B, w, bv)
    return result[:n].reshape(meta["shape"]).astype(np.float32)


def _hash_encode(coords: np.ndarray, grid_values: np.ndarray) -> np.ndarray:
    """1D linear interpolation on a lookup table."""
    n = len(coords)
    g = len(grid_values)
    idx_f = (coords + 1.0) * 0.5 * (g - 1)
    idx0 = np.clip(np.floor(idx_f).astype(np.int32), 0, g - 2)
    frac = idx_f - idx0
    return grid_values[idx0] * (1 - frac) + grid_values[idx0 + 1] * frac


def _hash_fit(flat: np.ndarray, coords: np.ndarray, n_grid: int) -> np.ndarray:
    """Fit grid values via least squares given the hash encoding — chunked."""
    g = n_grid
    n_dims = g

    def _encode(sl: slice) -> np.ndarray:
        c = coords[sl]
        m = len(c)
        idx_f = (c + 1.0) * 0.5 * (g - 1)
        idx0 = np.clip(np.floor(idx_f).astype(np.int32), 0, g - 2)
        frac = idx_f - idx0
        A = np.zeros((m, g), dtype=np.float32)
        rows = np.arange(m)
        A[rows, idx0] = 1 - frac
        A[rows, idx0 + 1] = frac
        return A

    vals = _chunked_lstsq(_encode, flat, n_dims)
    return vals.ravel()


def _hash_compress(tensor: np.ndarray, n_grid: int = 64) -> Tuple[bytes, dict]:
    flat = tensor.ravel().astype(np.float64)
    n = len(flat)
    if _should_fallback(tensor, n_grid, 2.0):
        return _BlockINT8Wrapper.compress(tensor)
    coords = np.linspace(-1, 1, n).astype(np.float32)
    grid_vals = _hash_fit(flat, coords, n_grid)
    data = _serialize(grid_vals)
    meta = dict(n_grid=n_grid, n=n, shape=tensor.shape)
    return data, meta


def _hash_decompress(data: bytes, meta: dict) -> np.ndarray:
    if meta.get("_fallback"):
        return _BlockINT8Wrapper.decompress(data, meta)
    n_grid = meta["n_grid"]
    grid_vals = _deserialize(data[: n_grid * 4])
    n = meta["n"]
    coords = np.linspace(-1, 1, n).astype(np.float32)
    result = _hash_encode(coords, grid_vals)
    return result[:n].reshape(meta["shape"]).astype(np.float32)


def _mlp_fit(
    flat: np.ndarray,
    coords: np.ndarray,
    widths: List[int],
    act: ActType,
    epochs: int = 100,
    lr: float = 0.01,
) -> tuple:
    """MLP with random frozen features + least-squares output layer (chunked)."""
    rng = np.random.RandomState(42)
    layers = []
    dims = [1] + widths + [1]
    for i in range(len(dims) - 1):
        s = np.sqrt(2.0 / (dims[i] + dims[i + 1]))
        w = (rng.randn(dims[i], dims[i + 1]) * s).astype(np.float32)
        b = (rng.randn(dims[i + 1]) * 0.0).astype(np.float32)
        layers.append((w, b))
    w_out = layers[-1][0]

    def _features_aug(sl: slice) -> np.ndarray:
        c = coords[sl].reshape(-1, 1)
        h = c
        for w, b in layers[:-1]:
            h = act(h @ w + b)
        h = h.astype(np.float64)
        return np.column_stack([h, np.ones(len(h), dtype=np.float64)])

    hdim = widths[-1] if widths else 1
    n_dims = hdim + 1
    x = _chunked_lstsq(_features_aug, flat, n_dims)
    wo = x[:hdim].astype(np.float32).reshape(w_out.shape)
    bo = np.array([float(x[hdim])], dtype=np.float32)
    layers[-1] = (wo, bo)
    return layers


def _mlp_eval(coords: np.ndarray, layers: list, act: ActType) -> np.ndarray:
    h = coords.reshape(-1, 1)
    for w, b in layers[:-1]:
        h = act(h @ w + b)
    return (h @ layers[-1][0] + layers[-1][1]).ravel()


def _mlp_compress(
    tensor: np.ndarray,
    widths: List[int],
    act: ActType,
) -> Tuple[bytes, dict]:
    flat = tensor.ravel().astype(np.float64)
    n = len(flat)
    dims = [1] + widths + [1]
    n_params = sum(dims[i] * dims[i + 1] + dims[i + 1] for i in range(len(dims) - 1))
    if _should_fallback(tensor, n_params):
        return _BlockINT8Wrapper.compress(tensor)
    coords = np.linspace(-1, 1, n).astype(np.float32)
    layers = _mlp_fit(flat, coords, widths, act)
    data = b""
    for w, b in layers:
        data += _serialize(w) + _serialize(b)
    meta = dict(widths=widths, n=n, shape=tensor.shape)
    return data, meta


def _mlp_decompress(data: bytes, meta: dict, act: ActType) -> np.ndarray:
    if meta.get("_fallback"):
        return _BlockINT8Wrapper.decompress(data, meta)
    widths = meta["widths"]
    dims = [1] + widths + [1]
    pos = 0
    layers = []
    for i in range(len(dims) - 1):
        w = _deserialize(data[pos : pos + dims[i] * dims[i + 1] * 4]).reshape(
            dims[i], dims[i + 1]
        )
        pos += dims[i] * dims[i + 1] * 4
        b = _deserialize(data[pos : pos + dims[i + 1] * 4])
        pos += dims[i + 1] * 4
        layers.append((w, b))
    n = meta["n"]
    coords = np.linspace(-1, 1, n).astype(np.float32)
    result = _mlp_eval(coords, layers, act)
    return result[:n].reshape(meta["shape"]).astype(np.float32)


def _poly_features(coords: np.ndarray, degree: int) -> np.ndarray:
    """Chebyshev polynomials T_k(x) up to degree."""
    x = coords.ravel()
    n = len(x)
    basis = np.zeros((n, degree + 1), dtype=np.float32)
    basis[:, 0] = 1.0
    if degree >= 1:
        basis[:, 1] = x
    for k in range(2, degree + 1):
        basis[:, k] = 2 * x * basis[:, k - 1] - basis[:, k - 2]
    return basis


def _poly_fit(flat: np.ndarray, coords: np.ndarray, degree: int) -> np.ndarray:
    n_dims = degree + 1

    def _basis(sl: slice) -> np.ndarray:
        return _poly_features(coords[sl], degree)

    x = _chunked_lstsq(_basis, flat, n_dims)
    return x.ravel()


def _poly_eval(coords: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    basis = _poly_features(coords, len(coeffs) - 1)
    return (basis @ coeffs).ravel()


def _poly_compress(tensor: np.ndarray, degree: int = 10) -> Tuple[bytes, dict]:
    flat = tensor.ravel().astype(np.float64)
    n = len(flat)
    n_coeffs = degree + 1
    if _should_fallback(tensor, n_coeffs):
        return _BlockINT8Wrapper.compress(tensor)
    coords = np.linspace(-1, 1, n).astype(np.float32)
    coeffs = _poly_fit(flat, coords, degree)
    data = _serialize(coeffs)
    meta = dict(degree=degree, n=n, shape=tensor.shape)
    return data, meta


def _poly_decompress(data: bytes, meta: dict) -> np.ndarray:
    if meta.get("_fallback"):
        return _BlockINT8Wrapper.decompress(data, meta)
    degree = meta["degree"]
    coeffs = _deserialize(data[: (degree + 1) * 4])
    n = meta["n"]
    coords = np.linspace(-1, 1, n).astype(np.float32)
    result = _poly_eval(coords, coeffs)
    return result[:n].reshape(meta["shape"]).astype(np.float32)


def _spline_fit(
    flat: np.ndarray, coords: np.ndarray, n_knots: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Uniform cubic B-spline fit via least squares — fully vectorized."""
    x = coords.ravel()
    xs = (x + 1.0) * 0.5
    t = np.linspace(0, 1, n_knots, dtype=np.float32)
    d = np.abs(xs[:, None] - t[None, :]) * n_knots
    basis = np.maximum(0, 1 - d) ** 3
    coeffs, _, _, _ = np.linalg.lstsq(basis, flat.astype(np.float64), rcond=None)
    return t, coeffs.astype(np.float32).ravel()


def _spline_eval(
    coords: np.ndarray, knots: np.ndarray, coeffs: np.ndarray
) -> np.ndarray:
    x = coords.ravel()
    xs = (x + 1.0) * 0.5
    n_knots = len(knots)
    d = np.abs(xs[:, None] - knots[None, :]) * n_knots
    basis = np.maximum(0, 1 - d) ** 3
    return (basis @ coeffs).ravel()


def _spline_compress(tensor: np.ndarray, n_knots: int = 32) -> Tuple[bytes, dict]:
    flat = tensor.ravel().astype(np.float64)
    n = len(flat)
    if _should_fallback(tensor, n_knots * 2):
        return _BlockINT8Wrapper.compress(tensor)
    coords = np.linspace(-1, 1, n).astype(np.float32)
    knots, coeffs = _spline_fit(flat, coords, n_knots)
    data = _serialize(knots) + _serialize(coeffs)
    meta = dict(n_knots=n_knots, n=n, shape=tensor.shape)
    return data, meta


def _spline_decompress(data: bytes, meta: dict) -> np.ndarray:
    if meta.get("_fallback"):
        return _BlockINT8Wrapper.decompress(data, meta)
    nk = meta["n_knots"]
    knots = _deserialize(data[: nk * 4])
    coeffs = _deserialize(data[nk * 4 : nk * 8])
    n = meta["n"]
    coords = np.linspace(-1, 1, n).astype(np.float32)
    result = _spline_eval(coords, knots, coeffs)
    return result[:n].reshape(meta["shape"]).astype(np.float32)


def _rbf_fit(
    flat: np.ndarray,
    coords: np.ndarray,
    centers: np.ndarray,
    kernel: Callable,
) -> np.ndarray:
    n_centers = len(centers)
    n_dims = n_centers

    def _phi(sl: slice) -> np.ndarray:
        c = coords[sl]
        dists = np.abs(c.ravel()[:, None] - centers[None, :])
        return kernel(dists)

    x = _chunked_lstsq(_phi, flat, n_dims)
    return x.ravel()


def _rbf_eval(
    coords: np.ndarray, centers: np.ndarray, weights: np.ndarray, kernel: Callable
) -> np.ndarray:
    dists = np.abs(coords.ravel()[:, None] - centers[None, :])
    phi = kernel(dists)
    return (phi @ weights).ravel()


def _rbf_compress(
    tensor: np.ndarray, n_centers: int, kernel: Callable, center_type: str = "uniform"
) -> Tuple[bytes, dict]:
    flat = tensor.ravel().astype(np.float64)
    n = len(flat)
    if _should_fallback(tensor, n_centers * 2):
        return _BlockINT8Wrapper.compress(tensor)
    coords = np.linspace(-1, 1, n).astype(np.float32)
    if center_type == "uniform":
        centers = np.linspace(-1, 1, n_centers).astype(np.float32)
    else:
        rng = np.random.RandomState(
            int(center_type.split("_")[-1]) if center_type.startswith("seed") else 42
        )
        centers = np.sort(rng.uniform(-1, 1, n_centers)).astype(np.float32)
    weights = _rbf_fit(flat, coords, centers, kernel)
    data = _serialize(centers) + _serialize(weights)
    meta = dict(n_centers=n_centers, n=n, shape=tensor.shape)
    return data, meta


def _rbf_decompress(data: bytes, meta: dict, kernel: Callable) -> np.ndarray:
    if meta.get("_fallback"):
        return _BlockINT8Wrapper.decompress(data, meta)
    nc = meta["n_centers"]
    centers = _deserialize(data[: nc * 4])
    weights = _deserialize(data[nc * 4 : nc * 8])
    n = meta["n"]
    coords = np.linspace(-1, 1, n).astype(np.float32)
    result = _rbf_eval(coords, centers, weights, kernel)
    return result[:n].reshape(meta["shape"]).astype(np.float32)


def _sparse_dct_basis(n: int, n_basis: int) -> np.ndarray:
    """DCT basis matrix (n x n_basis) — fully vectorized."""
    k = np.arange(n_basis, dtype=np.float32)
    x_idx = np.arange(n, dtype=np.float32) + 0.5
    basis = np.cos(np.pi * k[None, :] * x_idx[:, None] / n)
    return (basis * math.sqrt(2.0 / n)).astype(np.float32)


def _sparse_fwht_basis(n: int, n_basis: int) -> np.ndarray:
    """FWHT-like basis (first n_basis columns) — full matrix (may OOM for large n)."""
    rng = np.random.RandomState(42)
    basis = rng.randn(n, n_basis).astype(np.float32)
    basis /= np.linalg.norm(basis, axis=0, keepdims=True) + 1e-10
    return basis


def _sparse_dct_basis_chunk(n_full: int, sl: slice, n_basis: int) -> np.ndarray:
    """DCT basis for one chunk — no full n×n_basis allocation."""
    n_chunk = sl.stop - sl.start
    k = np.arange(n_basis, dtype=np.float32)
    x_idx = np.arange(sl.start, sl.stop, dtype=np.float32) + 0.5
    basis = np.cos(np.pi * k[None, :] * x_idx[:, None] / n_full)
    return (basis * math.sqrt(2.0 / n_full)).astype(np.float32)


def _sparse_fwht_basis_chunk(n_full: int, sl: slice, n_basis: int) -> np.ndarray:
    """FWHT-like basis for one chunk — generated on-the-fly."""
    n_chunk = sl.stop - sl.start
    rng = np.random.RandomState(42)
    # Deterministic sub-sample of the full random matrix
    basis = rng.randn(n_chunk, n_basis).astype(np.float32)
    norm = np.sqrt(n_full).astype(np.float32)
    basis /= norm
    return basis


def _sparse_omp_fit(
    flat: np.ndarray, basis_fn, n_basis: int, n_nonzero: int, chunk_size: int = 1 << 18
) -> Tuple[np.ndarray, np.ndarray]:
    """Orthogonal Matching Pursuit — chunked to avoid O(n×m) memory."""
    n = len(flat)
    m = n_basis
    target = flat.astype(np.float64)
    residual = target.copy()
    idx: list[int] = []

    def _correlation(vec: np.ndarray) -> np.ndarray:
        corr = np.zeros(m, dtype=np.float64)
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            A_chunk = basis_fn(slice(start, end)).astype(np.float64)
            corr += A_chunk.T @ vec[start:end]
        return corr

    def _lstsq_sub(columns: list[int], b: np.ndarray) -> np.ndarray:
        ATA = np.zeros((len(columns), len(columns)), dtype=np.float64)
        ATb = np.zeros(len(columns), dtype=np.float64)
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            A_chunk = basis_fn(slice(start, end)).astype(np.float64)[:, columns]
            ATA += A_chunk.T @ A_chunk
            ATb += A_chunk.T @ b[start:end]
        try:
            L = np.linalg.cholesky(ATA)
            return np.linalg.solve(L, np.linalg.solve(L.T, ATb))
        except np.linalg.LinAlgError:
            c, _, _, _ = np.linalg.lstsq(ATA, ATb, rcond=None)
            return c

    for _ in range(min(n_nonzero, m)):
        corr = _correlation(residual)
        i = int(np.argmax(np.abs(corr)))
        if i in idx:
            break
        idx.append(i)
        c = _lstsq_sub(idx, flat.astype(np.float64))
        A_res = np.zeros(n, dtype=np.float64)
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            A_chunk = basis_fn(slice(start, end)).astype(np.float64)[:, idx]
            A_res[start:end] = A_chunk @ c
        residual = target - A_res

    full = np.zeros(m, dtype=np.float32)
    for j, i in enumerate(idx):
        full[i] = float(c[j]) if j < len(c) else 0.0
    return full, np.array(idx, dtype=np.int32)


def _sparse_compress(
    tensor: np.ndarray, n_basis: int, n_nonzero: int, basis_type: str = "dct"
) -> Tuple[bytes, dict]:
    flat = tensor.ravel().astype(np.float64)
    n = len(flat)
    n_params = n_basis + n_nonzero
    if _should_fallback(tensor, n_params):
        return _BlockINT8Wrapper.compress(tensor)

    if basis_type == "dct":

        def _basis_fn(sl: slice) -> np.ndarray:
            return _sparse_dct_basis_chunk(n, sl, n_basis)
    else:

        def _basis_fn(sl: slice) -> np.ndarray:
            return _sparse_fwht_basis_chunk(n, sl, n_basis)

    coeffs, indices = _sparse_omp_fit(flat, _basis_fn, n_basis, n_nonzero)
    data = (
        struct.pack("<II", n_basis, len(indices))
        + indices.astype(np.int32).tobytes()
        + coeffs.astype(np.float32).tobytes()
    )
    meta = dict(
        n_basis=n_basis,
        n_nonzero=len(indices),
        basis_type=basis_type,
        n=n,
        shape=tensor.shape,
    )
    return data, meta


def _sparse_decompress(data: bytes, meta: dict) -> np.ndarray:
    if meta.get("_fallback"):
        return _BlockINT8Wrapper.decompress(data, meta)
    n_basis, nz = struct.unpack_from("<II", data, 0)
    pos = 8
    indices = np.frombuffer(data[pos : pos + nz * 4], dtype=np.int32)
    pos += nz * 4
    coeffs = np.frombuffer(data[pos : pos + nz * 4], dtype=np.float32)
    n = meta["n"]
    shape = meta["shape"]
    basis_type = meta.get("basis_type", "dct")

    full = np.zeros(n_basis, dtype=np.float32)
    full[indices] = coeffs

    if n <= 1 << 16:  # Small: direct reconstruction
        if basis_type == "dct":
            basis = _sparse_dct_basis(n, n_basis)
        else:
            basis = _sparse_fwht_basis(n, n_basis)
        result = basis @ full
    else:  # Large: chunked reconstruction
        result = np.zeros(n, dtype=np.float64)
        chunk_size = 1 << 18
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            sl = slice(start, end)
            if basis_type == "dct":
                A_chunk = _sparse_dct_basis_chunk(n, sl, n_basis)
            else:
                A_chunk = _sparse_fwht_basis_chunk(n, sl, n_basis)
            result[start:end] = A_chunk @ full
        result = result.astype(np.float32)

    return result[:n].reshape(shape).astype(np.float32)


def _hierarchical_fit(
    flat: np.ndarray,
    coords: np.ndarray,
    n_levels: int,
    base_hidden: int,
    act: ActType,
) -> List[tuple]:
    """Fit hierarchical SIRENs at different scales (chunked evaluation)."""
    levels = []
    residual = flat.astype(np.float64).copy()
    for level in range(n_levels):
        hidden = max(base_hidden // (2**level), 4)
        w1, b1, w_mid, b_mid, wo, bo = _siren_fit(residual, coords, hidden, 2, 50, act)
        pred = _siren_eval(coords, w1, b1, w_mid, b_mid, wo, bo, act)
        residual = residual - pred
        levels.append((w1, b1, w_mid, b_mid, wo, bo, hidden))
    return levels


def _hierarchical_eval(coords: np.ndarray, levels: list, act: ActType) -> np.ndarray:
    result = np.zeros(len(coords), dtype=np.float64)
    for w1, b1, w_mid, b_mid, wo, bo, _ in levels:
        result += _siren_eval(coords, w1, b1, w_mid, b_mid, wo, bo, act)
    return result


def _hierarchical_compress(
    tensor: np.ndarray, n_levels: int = 3, base_hidden: int = 16, act: ActType = np.sin
) -> Tuple[bytes, dict]:
    flat = tensor.ravel().astype(np.float64)
    n = len(flat)
    total_params = sum(max(base_hidden // (2**l), 4) * 4 for l in range(n_levels))
    if _should_fallback(tensor, total_params):
        return _BlockINT8Wrapper.compress(tensor)
    coords = np.linspace(-1, 1, n).astype(np.float32)
    levels = _hierarchical_fit(flat, coords, n_levels, base_hidden, act)
    data = struct.pack("<I", n_levels)
    level_meta = []
    for w1, b1, w_mid, b_mid, wo, bo, hid in levels:
        data += (
            _serialize(w1)
            + _serialize(b1)
            + _serialize(wo)
            + _serialize(np.array([bo]))
        )
        for w, b in zip(w_mid, b_mid):
            data += _serialize(w) + _serialize(b)
        level_meta.append(hid)
    meta = dict(n_levels=n_levels, n=n, shape=tensor.shape, level_hidden=level_meta)
    return data, meta


def _hierarchical_decompress(data: bytes, meta: dict, act: ActType) -> np.ndarray:
    if meta.get("_fallback"):
        return _BlockINT8Wrapper.decompress(data, meta)
    n_levels = meta["n_levels"]
    level_hidden = meta["level_hidden"]
    pos = 4
    levels = []
    for l in range(n_levels):
        hid = level_hidden[l]
        w1 = _deserialize(data[pos : pos + hid * 4]).reshape(1, hid)
        pos += hid * 4
        b1 = _deserialize(data[pos : pos + hid * 4])
        pos += hid * 4
        wo = _deserialize(data[pos : pos + hid * 4])
        pos += hid * 4
        bo = float(_deserialize(data[pos : pos + 4])[0])
        pos += 4
        w_mid, b_mid = [], []
        for _ in range(1):  # 2-layer
            w_mid.append(
                _deserialize(data[pos : pos + hid * hid * 4]).reshape(hid, hid)
            )
            pos += hid * hid * 4
            b_mid.append(_deserialize(data[pos : pos + hid * 4]))
            pos += hid * 4
        levels.append((w1, b1, w_mid, b_mid, wo, bo, hid))
    n = meta["n"]
    coords = np.linspace(-1, 1, n).astype(np.float32)
    result = _hierarchical_eval(coords, levels, act)
    return result[:n].reshape(meta["shape"]).astype(np.float32)


def _symbolic_basis(coords: np.ndarray, funcs: List[Callable]) -> np.ndarray:
    x = coords.ravel()
    n = len(x)
    m = len(funcs)
    basis = np.zeros((n, m), dtype=np.float32)
    for i, fn in enumerate(funcs):
        basis[:, i] = fn(x)
    return basis


def _symbolic_fit(
    flat: np.ndarray, coords: np.ndarray, funcs: List[Callable]
) -> np.ndarray:
    n_dims = len(funcs)

    def _basis(sl: slice) -> np.ndarray:
        return _symbolic_basis(coords[sl], funcs)

    x = _chunked_lstsq(_basis, flat, n_dims)
    return x.ravel()


def _symbolic_eval(
    coords: np.ndarray, funcs: List[Callable], coeffs: np.ndarray
) -> np.ndarray:
    basis = _symbolic_basis(coords, funcs)
    return (basis @ coeffs).ravel()


_SYMBOLIC_FUNCS: dict = {}


def _build_symbolic_funcs() -> None:
    np_mod = __import__("numpy", fromlist=["np"])
    _SYMBOLIC_FUNCS["trig"] = [np_mod.sin, np_mod.cos]
    _SYMBOLIC_FUNCS["poly"] = [lambda x: x, lambda x: x**2, lambda x: x**3]
    _SYMBOLIC_FUNCS["mixed"] = [
        np_mod.sin,
        np_mod.cos,
        lambda x: np_mod.exp(-np.abs(x)),
    ]
    _SYMBOLIC_FUNCS["trig_poly"] = [np_mod.sin, np_mod.cos, lambda x: x, lambda x: x**2]
    _SYMBOLIC_FUNCS["exp"] = [np_mod.exp, lambda x: np_mod.exp(-x)]
    _SYMBOLIC_FUNCS["all"] = [
        np_mod.sin,
        np_mod.cos,
        np_mod.tanh,
        lambda x: x,
        lambda x: x**2,
        lambda x: np_mod.exp(-np.abs(x)),
    ]


_build_symbolic_funcs()


def _symbolic_compress(tensor: np.ndarray, func_set_name: str) -> Tuple[bytes, dict]:
    flat = tensor.ravel().astype(np.float64)
    n = len(flat)
    funcs = _SYMBOLIC_FUNCS[func_set_name]
    n_coeffs = len(funcs)
    if _should_fallback(tensor, n_coeffs):
        return _BlockINT8Wrapper.compress(tensor)
    coords = np.linspace(-1, 1, n).astype(np.float32)
    coeffs = _symbolic_fit(flat, coords, funcs)
    data = struct.pack("<I", len(funcs)) + _serialize(coeffs)
    meta = dict(func_set=func_set_name, n_coeffs=n_coeffs, n=n, shape=tensor.shape)
    return data, meta


def _symbolic_decompress(data: bytes, meta: dict) -> np.ndarray:
    if meta.get("_fallback"):
        return _BlockINT8Wrapper.decompress(data, meta)
    func_set = meta["func_set"]
    n_coeffs = meta["n_coeffs"]
    coeffs = _deserialize(data[4 : 4 + n_coeffs * 4])
    funcs = _SYMBOLIC_FUNCS[func_set]
    n = meta["n"]
    coords = np.linspace(-1, 1, n).astype(np.float32)
    result = _symbolic_eval(coords, funcs, coeffs)
    return result[:n].reshape(meta["shape"]).astype(np.float32)


def _adaptive_select_best(tensor: np.ndarray) -> Tuple[bytes, dict, str]:
    """Try multiple methods and pick the one with best ratio/error trade-off."""
    candidates = [
        ("siren", lambda: _siren_compress(tensor, np.sin, 24, 2, 80)),
        ("fourier", lambda: _fourier_compress(tensor, 48, 1.0)),
        ("hash", lambda: _hash_compress(tensor, 48)),
        ("poly", lambda: _poly_compress(tensor, 8)),
    ]
    best_data, best_meta, best_score = None, None, -float("inf")
    for name, fn in candidates:
        try:
            d, m = fn()
            ratio = tensor.nbytes / len(d) if len(d) > 0 else 1.0
            score = ratio if m.get("_fallback") else ratio * 0.7
            if score > best_score:
                best_data, best_meta, best_score = d, m, score
        except Exception:
            continue
    if best_data is None:
        return _BlockINT8Wrapper.compress(tensor)
    return best_data, best_meta


def _make_siren_class(name_str: str, act: ActType, hidden=32, n_layers=3, freq=1.0):
    name_stem = "".join(p.title() for p in name_str.split("_"))
    cname = f"SIREN{name_stem}"

    def compress(self, tensor, **params):
        return _siren_compress(
            tensor,
            act,
            hidden=params.get("hidden_dim", hidden),
            n_layers=params.get("n_layers", n_layers),
            n_epochs=params.get("n_epochs", 100),
            freq=freq,
        )

    def decompress(self, data, metadata):
        return _siren_decompress(data, metadata, act)

    return type(
        cname,
        (object,),
        {
            "name": name_str,
            "category": "functional_weight_space",
            "compress": compress,
            "decompress": decompress,
            "__doc__": f"Functional weight space via SIREN ({name_str}).",
        },
    )


def _make_mlp_class(name_str: str, act: ActType, widths: List[int]):
    name_stem = "".join(p.title() for p in name_str.split("_"))
    cname = f"NeuralField{name_stem}"

    def compress(self, tensor, **params):
        return _mlp_compress(tensor, widths, act)

    def decompress(self, data, metadata):
        return _mlp_decompress(data, metadata, act)

    return type(
        cname,
        (object,),
        {
            "name": name_str,
            "category": "functional_weight_space",
            "compress": compress,
            "decompress": decompress,
            "__doc__": f"Functional weight space via MLP ({name_str}).",
        },
    )


def _make_fourier_class(name_str: str, n_feats: int, sigma: float):
    name_stem = "".join(p.title() for p in name_str.split("_"))
    cname = f"Fourier{name_stem}"

    def compress(self, tensor, **params):
        return _fourier_compress(tensor, n_feats=n_feats, sigma=sigma)

    def decompress(self, data, metadata):
        return _fourier_decompress(data, metadata)

    return type(
        cname,
        (object,),
        {
            "name": name_str,
            "category": "functional_weight_space",
            "compress": compress,
            "decompress": decompress,
            "__doc__": f"Functional weight space via Fourier features ({name_str}).",
        },
    )


def _make_hash_class(name_str: str, n_grid: int):
    name_stem = "".join(p.title() for p in name_str.split("_"))
    cname = f"Hash{name_stem}"

    def compress(self, tensor, **params):
        return _hash_compress(tensor, n_grid=n_grid)

    def decompress(self, data, metadata):
        return _hash_decompress(data, metadata)

    return type(
        cname,
        (object,),
        {
            "name": name_str,
            "category": "functional_weight_space",
            "compress": compress,
            "decompress": decompress,
            "__doc__": f"Functional weight space via hash encoding ({name_str}).",
        },
    )


def _make_poly_class(name_str: str, degree: int):
    name_stem = "".join(p.title() for p in name_str.split("_"))
    cname = f"Poly{name_stem}"

    def compress(self, tensor, **params):
        return _poly_compress(tensor, degree=degree)

    def decompress(self, data, metadata):
        return _poly_decompress(data, metadata)

    return type(
        cname,
        (object,),
        {
            "name": name_str,
            "category": "functional_weight_space",
            "compress": compress,
            "decompress": decompress,
            "__doc__": f"Functional weight space via polynomial ({name_str}).",
        },
    )


def _make_spline_class(name_str: str, n_knots: int):
    name_stem = "".join(p.title() for p in name_str.split("_"))
    cname = f"Spline{name_stem}"

    def compress(self, tensor, **params):
        return _spline_compress(tensor, n_knots=n_knots)

    def decompress(self, data, metadata):
        return _spline_decompress(data, metadata)

    return type(
        cname,
        (object,),
        {
            "name": name_str,
            "category": "functional_weight_space",
            "compress": compress,
            "decompress": decompress,
            "__doc__": f"Functional weight space via spline ({name_str}).",
        },
    )


def _make_rbf_class(name_str: str, n_centers: int, kernel: Callable, ctype="uniform"):
    name_stem = "".join(p.title() for p in name_str.split("_"))
    cname = f"RBF{name_stem}"

    def compress(self, tensor, **params):
        return _rbf_compress(tensor, n_centers, kernel, ctype)

    def decompress(self, data, metadata):
        return _rbf_decompress(data, metadata, kernel)

    return type(
        cname,
        (object,),
        {
            "name": name_str,
            "category": "functional_weight_space",
            "compress": compress,
            "decompress": decompress,
            "__doc__": f"Functional weight space via RBF ({name_str}).",
        },
    )


def _make_sparse_class(name_str: str, n_basis: int, n_nonzero: int, basis_type="dct"):
    name_stem = "".join(p.title() for p in name_str.split("_"))
    cname = f"Sparse{name_stem}"

    def compress(self, tensor, **params):
        return _sparse_compress(tensor, n_basis, n_nonzero, basis_type)

    def decompress(self, data, metadata):
        return _sparse_decompress(data, metadata)

    return type(
        cname,
        (object,),
        {
            "name": name_str,
            "category": "functional_weight_space",
            "compress": compress,
            "decompress": decompress,
            "__doc__": f"Functional weight space via sparse coding ({name_str}).",
        },
    )


def _make_hierarchical_class(
    name_str: str, n_levels: int, base_hidden: int, act: ActType
):
    name_stem = "".join(p.title() for p in name_str.split("_"))
    cname = f"Hierarchical{name_stem}"

    def compress(self, tensor, **params):
        return _hierarchical_compress(tensor, n_levels, base_hidden, act)

    def decompress(self, data, metadata):
        return _hierarchical_decompress(data, metadata, act)

    return type(
        cname,
        (object,),
        {
            "name": name_str,
            "category": "functional_weight_space",
            "compress": compress,
            "decompress": decompress,
            "__doc__": f"Functional weight space via hierarchical ensemble ({name_str}).",
        },
    )


def _make_symbolic_class(name_str: str, func_set: str):
    name_stem = "".join(p.title() for p in name_str.split("_"))
    cname = f"Symbolic{name_stem}"

    def compress(self, tensor, **params):
        return _symbolic_compress(tensor, func_set)

    def decompress(self, data, metadata):
        return _symbolic_decompress(data, metadata)

    return type(
        cname,
        (object,),
        {
            "name": name_str,
            "category": "functional_weight_space",
            "compress": compress,
            "decompress": decompress,
            "__doc__": f"Functional weight space via symbolic ({name_str}).",
        },
    )
