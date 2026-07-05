"""Polynomial & Functional Approximation Methods

20 compression techniques that approximate weight matrices using polynomials,
rational functions, splines, and other compact mathematical representations.
All methods maintain FULL FP32 precision — compression from compact representation,
not from quantization.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_DTYPE = np.float64
_F32 = np.float32


# ═══════════════════════════════════════════════════════════════════════════════
# Helper utilities
# ═══════════════════════════════════════════════════════════════════════════════


def _error_stats(orig: np.ndarray, rec: np.ndarray) -> dict:
    o = orig.astype(_DTYPE).ravel()
    r = rec.astype(_DTYPE).ravel()
    mse = float(np.mean((o - r) ** 2))
    denom = np.sum(o**2) + 1e-30
    snr = 10.0 * np.log10(denom / (np.sum((o - r) ** 2) + 1e-30))
    cos_sim = float(np.dot(o, r) / (np.linalg.norm(o) * np.linalg.norm(r) + 1e-30))
    rel = float(np.mean(np.abs(o - r) / (np.abs(o) + 1e-10)))
    return {
        "mse": mse,
        "snr_db": float(snr),
        "cosine_similarity": cos_sim,
        "rel_error": rel,
    }


def _safe_polyfit(x: np.ndarray, y: np.ndarray, degree: int) -> np.ndarray:
    degree = min(degree, len(x) - 1)
    try:
        return np.polyfit(x, y, degree)
    except (np.linalg.LinAlgError, ValueError):
        V = np.column_stack([x**k for k in range(degree, -1, -1)])
        coeffs, _, _, _ = np.linalg.lstsq(V, y, rcond=None)
        return coeffs


def _chebyshev_basis(x: np.ndarray, degree: int) -> np.ndarray:
    n = len(x)
    T = np.zeros((degree + 1, n), dtype=_DTYPE)
    T[0] = 1.0
    if degree >= 1:
        T[1] = x.copy()
    for k in range(2, degree + 1):
        T[k] = 2.0 * x * T[k - 1] - T[k - 2]
    return T


def _legendre_basis(x: np.ndarray, degree: int) -> np.ndarray:
    n = len(x)
    P = np.zeros((degree + 1, n), dtype=_DTYPE)
    P[0] = 1.0
    if degree >= 1:
        P[1] = x.copy()
    for k in range(2, degree + 1):
        P[k] = ((2 * k - 1) * x * P[k - 1] - (k - 1) * P[k - 2]) / k
    return P


def _hermite_basis(x: np.ndarray, degree: int) -> np.ndarray:
    n = len(x)
    degree = min(degree, 30)
    H = np.zeros((degree + 1, n), dtype=_DTYPE)
    H[0] = 1.0
    if degree >= 1:
        H[1] = x.copy()
    for k in range(2, degree + 1):
        H[k] = x * H[k - 1] - (k - 1) * H[k - 2]
    return H


def _cubic_spline_coefficients(
    x: np.ndarray, y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(x)
    if n < 2:
        return (x, y, np.zeros_like(y), np.zeros_like(y), np.zeros_like(y))
    h = np.diff(x)
    alpha = np.zeros(n, dtype=_DTYPE)
    for i in range(1, n - 1):
        if abs(h[i]) > 1e-15 and abs(h[i - 1]) > 1e-15:
            alpha[i] = (3.0 / h[i]) * (y[i + 1] - y[i]) - (3.0 / h[i - 1]) * (
                y[i] - y[i - 1]
            )
    l = np.ones(n, dtype=_DTYPE)
    mu = np.zeros(n, dtype=_DTYPE)
    z = np.zeros(n, dtype=_DTYPE)
    for i in range(1, n - 1):
        l[i] = 2.0 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1]
        if abs(l[i]) < 1e-15:
            l[i] = 1e-15
        mu[i] = h[i] / l[i]
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i]
    c = np.zeros(n, dtype=_DTYPE)
    b = np.zeros(n - 1, dtype=_DTYPE)
    d = np.zeros(n - 1, dtype=_DTYPE)
    for j in range(n - 2, -1, -1):
        c[j] = z[j] - mu[j] * c[j + 1]
        if abs(h[j]) > 1e-15:
            b[j] = (y[j + 1] - y[j]) / h[j] - h[j] * (c[j + 1] + 2.0 * c[j]) / 3.0
            d[j] = (c[j + 1] - c[j]) / (3.0 * h[j])
    return x, y[: n - 1], b, c[: n - 1], d


def _eval_cubic_spline(spline_data: Tuple, x_eval: np.ndarray) -> np.ndarray:
    x_k, a, b, c, d = spline_data
    result = np.zeros_like(x_eval, dtype=_DTYPE)
    for i in range(len(x_k) - 1):
        mask = (x_eval >= x_k[i]) & (x_eval <= x_k[i + 1])
        if not np.any(mask):
            continue
        dx = x_eval[mask] - x_k[i]
        result[mask] = a[i] + b[i] * dx + c[i] * dx**2 + d[i] * dx**3
    mask_left = x_eval < x_k[0]
    if np.any(mask_left):
        result[mask_left] = a[0] + b[0] * (x_eval[mask_left] - x_k[0])
    mask_right = x_eval > x_k[-1]
    if np.any(mask_right):
        result[mask_right] = a[-1] + b[-1] * (x_eval[mask_right] - x_k[-1])
    return result


def _bspline_basis(x: np.ndarray, degree: int, knots: np.ndarray) -> np.ndarray:
    n = len(x)
    m = len(knots)
    B = np.zeros((m - 1, n), dtype=_DTYPE)
    for i in range(m - 1):
        mask = (x >= knots[i]) & (x < knots[i + 1])
        B[i, mask] = 1.0
    for k in range(1, degree + 1):
        B_new = np.zeros((m - k - 1, n), dtype=_DTYPE)
        for i in range(m - k - 1):
            left_num = x - knots[i]
            left_den = knots[i + k] - knots[i]
            right_num = knots[i + k + 1] - x
            right_den = knots[i + k + 1] - knots[i + 1]
            if abs(left_den) > 1e-15:
                B_new[i] += left_num / left_den * B[i]
            if abs(right_den) > 1e-15:
                B_new[i] += right_num / right_den * B[i + 1]
        B = B_new
    return B


def _rational_pade(
    x: np.ndarray, y: np.ndarray, num_deg: int, den_deg: int
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(x)
    total = num_deg + den_deg + 1
    if total > n:
        num_deg = max(1, n // 3)
        den_deg = max(1, n // 3)
    A = np.zeros((n, num_deg + 1 + den_deg), dtype=_DTYPE)
    for i in range(num_deg + 1):
        A[:, i] = x**i
    for j in range(1, den_deg + 1):
        A[:, num_deg + j] = -y * (x**j)
    ATA = A.T @ A + 1e-8 * np.eye(A.shape[1])
    ATy = A.T @ y
    params = np.linalg.solve(ATA, ATy)
    num_coeffs = params[: num_deg + 1]
    den_coeffs = np.concatenate([[1.0], params[num_deg + 1 :]])
    return num_coeffs, den_coeffs


def _eval_rational(
    num_coeffs: np.ndarray, den_coeffs: np.ndarray, x: np.ndarray
) -> np.ndarray:
    num = np.polyval(num_coeffs[::-1], x)
    den = np.polyval(den_coeffs[::-1], x)
    tiny = 1e-12
    den = np.where(np.abs(den) < tiny, np.sign(den + 1e-30) * tiny, den)
    return num / den


def _haar_forward(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = len(x)
    if n % 2 == 1:
        x = np.append(x, 0.0)
    even, odd = x[0::2], x[1::2]
    return (even + odd) * 0.5, (even - odd) * 0.5


def _haar_inverse(approx: np.ndarray, detail: np.ndarray) -> np.ndarray:
    n = len(approx)
    out = np.empty(2 * n, dtype=_DTYPE)
    out[0::2] = approx + detail
    out[1::2] = approx - detail
    return out


def _interpolate_bilinear(
    sub: np.ndarray, kr: np.ndarray, kc: np.ndarray, nr: int, nc: int
) -> np.ndarray:
    r_idx = np.linspace(0, nr - 1, nr, dtype=_DTYPE)
    c_idx = np.linspace(0, nc - 1, nc, dtype=_DTYPE)
    kr_f = kr.astype(_DTYPE)
    kc_f = kc.astype(_DTYPE)
    rk, ck = len(kr), len(kc)
    result = np.zeros((nr, nc), dtype=_DTYPE)
    for i in range(nr):
        for j in range(nc):
            ri_f = r_idx[i]
            ci_f = c_idx[j]
            ri = np.searchsorted(kr_f, ri_f) - 1
            ri = max(0, min(ri, rk - 2))
            ci = np.searchsorted(kc_f, ci_f) - 1
            ci = max(0, min(ci, ck - 2))
            fr = (ri_f - kr_f[ri]) / max(kr_f[ri + 1] - kr_f[ri], 1e-15)
            fc = (ci_f - kc_f[ci]) / max(kc_f[ci + 1] - kc_f[ci], 1e-15)
            fr = np.clip(fr, 0.0, 1.0)
            fc = np.clip(fc, 0.0, 1.0)
            result[i, j] = (
                (1 - fr) * (1 - fc) * sub[ri, ci]
                + fr * (1 - fc) * sub[ri + 1, ci]
                + (1 - fr) * fc * sub[ri, ci + 1]
                + fr * fc * sub[ri + 1, ci + 1]
            )
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 1. PolynomialRowApprox
# ═══════════════════════════════════════════════════════════════════════════════


class PolynomialRowApprox:
    """Fit degree-d polynomial to each row. Store (d+1) coefficients per row."""

    METHOD_NAME = "polynomial_row_approx"
    name = "polynomial_row_approx"
    category = "spectral"

    def __init__(self, degree: int = 8):
        self.degree = degree

    def compress(
        self, tensor: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        degree = kwargs.get("degree", self.degree)
        orig_shape = tensor.shape
        mat = tensor.astype(_DTYPE)
        rows, cols = mat.shape
        x = np.linspace(-1, 1, cols, dtype=_DTYPE)
        degree = min(degree, cols - 1)
        all_coeffs = np.zeros((rows, degree + 1), dtype=_F32)
        for i in range(rows):
            all_coeffs[i] = _safe_polyfit(x, mat[i], degree).astype(_F32)
        data = {"coeffs": all_coeffs, "shape": np.array(orig_shape, dtype=np.int32)}
        meta = {"orig_shape": orig_shape, "method": self.METHOD_NAME, "degree": degree}
        return data, meta

    def decompress(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> np.ndarray:
        coeffs = data["coeffs"].astype(_DTYPE)
        rows, cols = metadata["orig_shape"]
        x = np.linspace(-1, 1, cols, dtype=_DTYPE)
        out = np.zeros((rows, cols), dtype=_DTYPE)
        for i in range(rows):
            out[i] = np.polyval(coeffs[i], x)
        return out.astype(_F32)

    def estimate_ratio(self, tensor: np.ndarray, **kwargs) -> float:
        degree = kwargs.get("degree", self.degree)
        orig = tensor.nbytes
        rows = tensor.shape[0]
        comp = rows * (degree + 1) * 4
        return comp / max(orig, 1)

    def estimate_error(self, tensor: np.ndarray, **kwargs) -> dict:
        data, meta = self.compress(tensor, **kwargs)
        recon = self.decompress(data, meta)
        return _error_stats(tensor, recon)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. PolynomialColumnApprox
# ═══════════════════════════════════════════════════════════════════════════════


class PolynomialColumnApprox:
    """Fit degree-d polynomial to each column."""

    METHOD_NAME = "polynomial_column_approx"
    name = "polynomial_column_approx"
    category = "spectral"

    def __init__(self, degree: int = 8):
        self.degree = degree

    def compress(
        self, tensor: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        degree = kwargs.get("degree", self.degree)
        orig_shape = tensor.shape
        mat = tensor.astype(_DTYPE)
        rows, cols = mat.shape
        y = np.linspace(-1, 1, rows, dtype=_DTYPE)
        degree = min(degree, rows - 1)
        all_coeffs = np.zeros((cols, degree + 1), dtype=_F32)
        for j in range(cols):
            all_coeffs[j] = _safe_polyfit(y, mat[:, j], degree).astype(_F32)
        data = {"coeffs": all_coeffs, "shape": np.array(orig_shape, dtype=np.int32)}
        meta = {"orig_shape": orig_shape, "method": self.METHOD_NAME, "degree": degree}
        return data, meta

    def decompress(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> np.ndarray:
        coeffs = data["coeffs"].astype(_DTYPE)
        rows, cols = metadata["orig_shape"]
        y = np.linspace(-1, 1, rows, dtype=_DTYPE)
        out = np.zeros((rows, cols), dtype=_DTYPE)
        for j in range(cols):
            out[:, j] = np.polyval(coeffs[j], y)
        return out.astype(_F32)

    def estimate_ratio(self, tensor: np.ndarray, **kwargs) -> float:
        degree = kwargs.get("degree", self.degree)
        orig = tensor.nbytes
        cols = tensor.shape[1]
        comp = cols * (degree + 1) * 4
        return comp / max(orig, 1)

    def estimate_error(self, tensor: np.ndarray, **kwargs) -> dict:
        data, meta = self.compress(tensor, **kwargs)
        recon = self.decompress(data, meta)
        return _error_stats(tensor, recon)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Polynomial2DApprox
# ═══════════════════════════════════════════════════════════════════════════════


class Polynomial2DApprox:
    """Bivariate polynomial p(i,j) over tiled blocks."""

    METHOD_NAME = "polynomial_2d_approx"
    name = "polynomial_2d_approx"
    category = "spectral"

    def __init__(self, degree: int = 4, block_size: int = 64):
        self.degree = degree
        self.block_size = block_size

    def _fit_block(self, block: np.ndarray, degree: int) -> np.ndarray:
        nr, nc = block.shape
        ri = np.linspace(-1, 1, nr, dtype=_DTYPE)
        ci = np.linspace(-1, 1, nc, dtype=_DTYPE)
        R, C = np.meshgrid(ri, ci, indexing="ij")
        terms = []
        for di in range(degree + 1):
            for dj in range(degree + 1 - di):
                terms.append((R**di) * (C**dj))
        V = np.column_stack([t.ravel() for t in terms])
        y = block.astype(_DTYPE).ravel()
        coeffs, _, _, _ = np.linalg.lstsq(V, y, rcond=None)
        return coeffs

    def _eval_block(
        self, coeffs: np.ndarray, degree: int, shape: Tuple[int, int]
    ) -> np.ndarray:
        nr, nc = shape
        ri = np.linspace(-1, 1, nr, dtype=_DTYPE)
        ci = np.linspace(-1, 1, nc, dtype=_DTYPE)
        R, C = np.meshgrid(ri, ci, indexing="ij")
        terms = []
        for di in range(degree + 1):
            for dj in range(degree + 1 - di):
                terms.append((R**di) * (C**dj))
        V = np.column_stack([t.ravel() for t in terms])
        return (V @ coeffs).reshape(nr, nc)

    def compress(
        self, tensor: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        degree = kwargs.get("degree", self.degree)
        bs = kwargs.get("block_size", self.block_size)
        orig_shape = tensor.shape
        mat = tensor.astype(_DTYPE)
        rows, cols = mat.shape
        blocks_r = (rows + bs - 1) // bs
        blocks_c = (cols + bs - 1) // bs
        all_coeffs = []
        block_shapes = []
        for bi in range(blocks_r):
            for bj in range(blocks_c):
                r0, r1 = bi * bs, min((bi + 1) * bs, rows)
                c0, c1 = bj * bs, min((bj + 1) * bs, cols)
                block = mat[r0:r1, c0:c1]
                c = self._fit_block(block, degree)
                all_coeffs.append(c.astype(_F32))
                block_shapes.append(np.array([r1 - r0, c1 - c0], dtype=np.int32))
        data = {
            "coeffs": all_coeffs,
            "block_shapes": block_shapes,
            "blocks_grid": np.array([blocks_r, blocks_c], dtype=np.int32),
            "bs": np.int32(bs),
            "shape": np.array(orig_shape, dtype=np.int32),
        }
        meta = {
            "orig_shape": orig_shape,
            "method": self.METHOD_NAME,
            "degree": degree,
            "block_size": bs,
        }
        return data, meta

    def decompress(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> np.ndarray:
        degree = metadata["degree"]
        bs = int(data["bs"])
        blocks_r, blocks_c = data["blocks_grid"]
        rows, cols = metadata["orig_shape"]
        out = np.zeros((rows, cols), dtype=_DTYPE)
        idx = 0
        for bi in range(blocks_r):
            for bj in range(blocks_c):
                r0 = bi * bs
                c0 = bj * bs
                bshape = tuple(int(x) for x in data["block_shapes"][idx])
                block = self._eval_block(
                    data["coeffs"][idx].astype(_DTYPE), degree, bshape
                )
                out[r0 : r0 + bshape[0], c0 : c0 + bshape[1]] = block
                idx += 1
        return out.astype(_F32)

    def estimate_ratio(self, tensor: np.ndarray, **kwargs) -> float:
        degree = kwargs.get("degree", self.degree)
        bs = kwargs.get("block_size", self.block_size)
        orig = tensor.nbytes
        rows, cols = tensor.shape
        n_blocks = ((rows + bs - 1) // bs) * ((cols + bs - 1) // bs)
        n_terms = (degree + 1) * (degree + 2) // 2
        comp = n_blocks * n_terms * 4
        return comp / max(orig, 1)

    def estimate_error(self, tensor: np.ndarray, **kwargs) -> dict:
        data, meta = self.compress(tensor, **kwargs)
        recon = self.decompress(data, meta)
        return _error_stats(tensor, recon)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. RationalApproximation
# ═══════════════════════════════════════════════════════════════════════════════


class RationalApproximation:
    """Pade approximant — ratio of two polynomials per row."""

    METHOD_NAME = "rational_approximation"
    name = "rational_approximation"
    category = "spectral"

    def __init__(self, num_degree: int = 6, den_degree: int = 4):
        self.num_degree = num_degree
        self.den_degree = den_degree

    def compress(
        self, tensor: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        nd = kwargs.get("num_degree", self.num_degree)
        dd = kwargs.get("den_degree", self.den_degree)
        orig_shape = tensor.shape
        mat = tensor.astype(_DTYPE)
        rows, cols = mat.shape
        x = np.linspace(-1, 1, cols, dtype=_DTYPE)
        num_all = []
        den_all = []
        for i in range(rows):
            nc, dc = _rational_pade(x, mat[i], nd, dd)
            num_all.append(nc.astype(_F32))
            den_all.append(dc.astype(_F32))
        data = {
            "num": num_all,
            "den": den_all,
            "shape": np.array(orig_shape, dtype=np.int32),
        }
        meta = {
            "orig_shape": orig_shape,
            "method": self.METHOD_NAME,
            "num_degree": nd,
            "den_degree": dd,
        }
        return data, meta

    def decompress(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> np.ndarray:
        rows, cols = metadata["orig_shape"]
        x = np.linspace(-1, 1, cols, dtype=_DTYPE)
        out = np.zeros((rows, cols), dtype=_DTYPE)
        for i in range(rows):
            nc = data["num"][i].astype(_DTYPE)
            dc = data["den"][i].astype(_DTYPE)
            out[i] = _eval_rational(nc, dc, x)
        return out.astype(_F32)

    def estimate_ratio(self, tensor: np.ndarray, **kwargs) -> float:
        nd = kwargs.get("num_degree", self.num_degree)
        dd = kwargs.get("den_degree", self.den_degree)
        orig = tensor.nbytes
        rows = tensor.shape[0]
        comp = rows * (nd + 1 + dd + 1) * 4
        return comp / max(orig, 1)

    def estimate_error(self, tensor: np.ndarray, **kwargs) -> dict:
        data, meta = self.compress(tensor, **kwargs)
        recon = self.decompress(data, meta)
        return _error_stats(tensor, recon)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. ChebyshevApprox
# ═══════════════════════════════════════════════════════════════════════════════


class ChebyshevApprox:
    """Chebyshev polynomial expansion — near-minimax approximation per row."""

    METHOD_NAME = "chebyshev_approx"
    name = "chebyshev_approx"
    category = "spectral"

    def __init__(self, degree: int = 32, threshold: float = 1e-4):
        self.degree = degree
        self.threshold = threshold

    def compress(
        self, tensor: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        degree = kwargs.get("degree", self.degree)
        orig_shape = tensor.shape
        mat = tensor.astype(_DTYPE)
        rows, cols = mat.shape
        degree = min(degree, cols - 1)
        all_coeffs = []
        for i in range(rows):
            cc, _ = np.polynomial.chebyshev.chebfit(
                np.linspace(-1, 1, cols, dtype=_DTYPE), mat[i], degree, full=True
            )
            mask = np.abs(cc) > self.threshold
            indices = np.where(mask)[0]
            all_coeffs.append(
                {
                    "coeffs": cc[indices].astype(_F32),
                    "indices": indices.astype(np.int32),
                }
            )
        data = {
            "rows": all_coeffs,
            "degree": np.int32(degree),
            "cols": np.int32(cols),
            "shape": np.array(orig_shape, dtype=np.int32),
        }
        meta = {"orig_shape": orig_shape, "method": self.METHOD_NAME, "degree": degree}
        return data, meta

    def decompress(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> np.ndarray:
        degree = int(data["degree"])
        cols = int(data["cols"])
        rows, _ = metadata["orig_shape"]
        x = np.linspace(-1, 1, cols, dtype=_DTYPE)
        out = np.zeros((rows, cols), dtype=_DTYPE)
        for i in range(rows):
            rd = data["rows"][i]
            cc = np.zeros(degree + 1, dtype=_DTYPE)
            cc[rd["indices"]] = rd["coeffs"].astype(_DTYPE)
            out[i] = np.polynomial.chebyshev.chebval(x, cc)
        return out.astype(_F32)

    def estimate_ratio(self, tensor: np.ndarray, **kwargs) -> float:
        degree = kwargs.get("degree", self.degree)
        orig = tensor.nbytes
        rows = tensor.shape[0]
        comp = rows * (degree + 1) * 8
        return comp / max(orig, 1)

    def estimate_error(self, tensor: np.ndarray, **kwargs) -> dict:
        data, meta = self.compress(tensor, **kwargs)
        recon = self.decompress(data, meta)
        return _error_stats(tensor, recon)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. LegendreApprox
# ═══════════════════════════════════════════════════════════════════════════════


class LegendreApprox:
    """Legendre polynomial expansion per row."""

    METHOD_NAME = "legendre_approx"
    name = "legendre_approx"
    category = "spectral"

    def __init__(self, degree: int = 32, threshold: float = 1e-4):
        self.degree = degree
        self.threshold = threshold

    def compress(
        self, tensor: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        degree = kwargs.get("degree", self.degree)
        orig_shape = tensor.shape
        mat = tensor.astype(_DTYPE)
        rows, cols = mat.shape
        degree = min(degree, cols - 1)
        all_coeffs = []
        for i in range(rows):
            lc, _ = np.polynomial.legendre.legfit(
                np.linspace(-1, 1, cols, dtype=_DTYPE), mat[i], degree, full=True
            )
            mask = np.abs(lc) > self.threshold
            indices = np.where(mask)[0]
            all_coeffs.append(
                {
                    "coeffs": lc[indices].astype(_F32),
                    "indices": indices.astype(np.int32),
                }
            )
        data = {
            "rows": all_coeffs,
            "degree": np.int32(degree),
            "cols": np.int32(cols),
            "shape": np.array(orig_shape, dtype=np.int32),
        }
        meta = {"orig_shape": orig_shape, "method": self.METHOD_NAME, "degree": degree}
        return data, meta

    def decompress(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> np.ndarray:
        degree = int(data["degree"])
        cols = int(data["cols"])
        rows, _ = metadata["orig_shape"]
        x = np.linspace(-1, 1, cols, dtype=_DTYPE)
        out = np.zeros((rows, cols), dtype=_DTYPE)
        for i in range(rows):
            rd = data["rows"][i]
            lc = np.zeros(degree + 1, dtype=_DTYPE)
            lc[rd["indices"]] = rd["coeffs"].astype(_DTYPE)
            out[i] = np.polynomial.legendre.legval(x, lc)
        return out.astype(_F32)

    def estimate_ratio(self, tensor: np.ndarray, **kwargs) -> float:
        degree = kwargs.get("degree", self.degree)
        orig = tensor.nbytes
        rows = tensor.shape[0]
        comp = rows * (degree + 1) * 8
        return comp / max(orig, 1)

    def estimate_error(self, tensor: np.ndarray, **kwargs) -> dict:
        data, meta = self.compress(tensor, **kwargs)
        recon = self.decompress(data, meta)
        return _error_stats(tensor, recon)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. HermiteApprox
# ═══════════════════════════════════════════════════════════════════════════════


class HermiteApprox:
    """Hermite polynomial expansion (physicist's, good for Gaussian-like data)."""

    METHOD_NAME = "hermite_approx"
    name = "hermite_approx"
    category = "spectral"

    def __init__(self, degree: int = 12, threshold: float = 1e-4):
        self.degree = degree
        self.threshold = threshold

    def compress(
        self, tensor: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        degree = kwargs.get("degree", self.degree)
        degree = min(degree, 15)
        orig_shape = tensor.shape
        mat = tensor.astype(_DTYPE)
        rows, cols = mat.shape
        x = np.linspace(-2, 2, cols, dtype=_DTYPE)
        all_coeffs = []
        for i in range(rows):
            hc, _ = np.polynomial.hermite.hermfit(x, mat[i], degree, full=True)
            mask = np.abs(hc) > self.threshold
            indices = np.where(mask)[0]
            all_coeffs.append(
                {
                    "coeffs": hc[indices].astype(_F32),
                    "indices": indices.astype(np.int32),
                }
            )
        data = {
            "rows": all_coeffs,
            "degree": np.int32(degree),
            "cols": np.int32(cols),
            "shape": np.array(orig_shape, dtype=np.int32),
        }
        meta = {"orig_shape": orig_shape, "method": self.METHOD_NAME, "degree": degree}
        return data, meta

    def decompress(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> np.ndarray:
        degree = int(data["degree"])
        cols = int(data["cols"])
        rows, _ = metadata["orig_shape"]
        x = np.linspace(-2, 2, cols, dtype=_DTYPE)
        out = np.zeros((rows, cols), dtype=_DTYPE)
        for i in range(rows):
            rd = data["rows"][i]
            hc = np.zeros(degree + 1, dtype=_DTYPE)
            hc[rd["indices"]] = rd["coeffs"].astype(_DTYPE)
            out[i] = np.polynomial.hermite.hermval(x, hc)
        return out.astype(_F32)

    def estimate_ratio(self, tensor: np.ndarray, **kwargs) -> float:
        degree = kwargs.get("degree", self.degree)
        orig = tensor.nbytes
        rows = tensor.shape[0]
        comp = rows * (degree + 1) * 8
        return comp / max(orig, 1)

    def estimate_error(self, tensor: np.ndarray, **kwargs) -> dict:
        data, meta = self.compress(tensor, **kwargs)
        recon = self.decompress(data, meta)
        return _error_stats(tensor, recon)


# ═══════════════════════════════════════════════════════════════════════════════
# 8. SplineRowApprox
# ═══════════════════════════════════════════════════════════════════════════════


class SplineRowApprox:
    """Cubic spline interpolation along rows."""

    METHOD_NAME = "spline_row_approx"
    name = "spline_row_approx"
    category = "spectral"

    def __init__(self, n_knots: int = 32):
        self.n_knots = n_knots

    def compress(
        self, tensor: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        nk = kwargs.get("n_knots", self.n_knots)
        orig_shape = tensor.shape
        mat = tensor.astype(_DTYPE)
        rows, cols = mat.shape
        nk = min(nk, cols)
        all_splines = []
        for i in range(rows):
            knot_idx = np.linspace(0, cols - 1, nk, dtype=int)
            x_k = np.linspace(-1, 1, nk, dtype=_DTYPE)
            y_k = mat[i, knot_idx]
            spline = _cubic_spline_coefficients(x_k, y_k)
            all_splines.append(tuple(c.astype(_F32) for c in spline))
        data = {
            "splines": all_splines,
            "shape": np.array(orig_shape, dtype=np.int32),
            "nk": np.int32(nk),
        }
        meta = {"orig_shape": orig_shape, "method": self.METHOD_NAME, "n_knots": nk}
        return data, meta

    def decompress(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> np.ndarray:
        rows, cols = metadata["orig_shape"]
        x_eval = np.linspace(-1, 1, cols, dtype=_DTYPE)
        out = np.zeros((rows, cols), dtype=_DTYPE)
        for i in range(rows):
            spline = tuple(c.astype(_DTYPE) for c in data["splines"][i])
            out[i] = _eval_cubic_spline(spline, x_eval)
        return out.astype(_F32)

    def estimate_ratio(self, tensor: np.ndarray, **kwargs) -> float:
        nk = kwargs.get("n_knots", self.n_knots)
        orig = tensor.nbytes
        rows = tensor.shape[0]
        comp = rows * nk * 5 * 4
        return comp / max(orig, 1)

    def estimate_error(self, tensor: np.ndarray, **kwargs) -> dict:
        data, meta = self.compress(tensor, **kwargs)
        recon = self.decompress(data, meta)
        return _error_stats(tensor, recon)


# ═══════════════════════════════════════════════════════════════════════════════
# 9. SplineColumnApprox
# ═══════════════════════════════════════════════════════════════════════════════


class SplineColumnApprox:
    """Cubic spline along columns."""

    METHOD_NAME = "spline_column_approx"
    name = "spline_column_approx"
    category = "spectral"

    def __init__(self, n_knots: int = 32):
        self.n_knots = n_knots

    def compress(
        self, tensor: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        nk = kwargs.get("n_knots", self.n_knots)
        orig_shape = tensor.shape
        mat = tensor.astype(_DTYPE)
        rows, cols = mat.shape
        nk = min(nk, rows)
        all_splines = []
        for j in range(cols):
            knot_idx = np.linspace(0, rows - 1, nk, dtype=int)
            x_k = np.linspace(-1, 1, nk, dtype=_DTYPE)
            y_k = mat[knot_idx, j]
            spline = _cubic_spline_coefficients(x_k, y_k)
            all_splines.append(tuple(c.astype(_F32) for c in spline))
        data = {
            "splines": all_splines,
            "shape": np.array(orig_shape, dtype=np.int32),
            "nk": np.int32(nk),
        }
        meta = {"orig_shape": orig_shape, "method": self.METHOD_NAME, "n_knots": nk}
        return data, meta

    def decompress(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> np.ndarray:
        rows, cols = metadata["orig_shape"]
        y_eval = np.linspace(-1, 1, rows, dtype=_DTYPE)
        out = np.zeros((rows, cols), dtype=_DTYPE)
        for j in range(cols):
            spline = tuple(c.astype(_DTYPE) for c in data["splines"][j])
            out[:, j] = _eval_cubic_spline(spline, y_eval)
        return out.astype(_F32)

    def estimate_ratio(self, tensor: np.ndarray, **kwargs) -> float:
        nk = kwargs.get("n_knots", self.n_knots)
        orig = tensor.nbytes
        cols = tensor.shape[1]
        comp = cols * nk * 5 * 4
        return comp / max(orig, 1)

    def estimate_error(self, tensor: np.ndarray, **kwargs) -> dict:
        data, meta = self.compress(tensor, **kwargs)
        recon = self.decompress(data, meta)
        return _error_stats(tensor, recon)


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Spline2DBicubic
# ═══════════════════════════════════════════════════════════════════════════════


class Spline2DBicubic:
    """Bicubic spline over 2D blocks using bilinear interpolation."""

    METHOD_NAME = "spline_2d_bicubic"
    name = "spline_2d_bicubic"
    category = "spectral"

    def __init__(self, n_knots: int = 16, block_size: int = 64):
        self.n_knots = n_knots
        self.block_size = block_size

    def _fit_block(self, block: np.ndarray, nk: int) -> dict:
        nr, nc = block.shape
        kr = np.linspace(0, nr - 1, min(nk, nr), dtype=int)
        kc = np.linspace(0, nc - 1, min(nk, nc), dtype=int)
        sub = block[np.ix_(kr, kc)]
        return {
            "sub": sub.astype(_F32),
            "kr": kr,
            "kc": kc,
            "nr": np.int32(nr),
            "nc": np.int32(nc),
        }

    def _eval_block(self, bd: dict) -> np.ndarray:
        sub = bd["sub"].astype(_DTYPE)
        kr, kc = bd["kr"], bd["kc"]
        nr, nc = int(bd["nr"]), int(bd["nc"])
        return _interpolate_bilinear(sub, kr, kc, nr, nc)

    def compress(
        self, tensor: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        nk = kwargs.get("n_knots", self.n_knots)
        bs = kwargs.get("block_size", self.block_size)
        orig_shape = tensor.shape
        mat = tensor.astype(_DTYPE)
        rows, cols = mat.shape
        blocks_r = (rows + bs - 1) // bs
        blocks_c = (cols + bs - 1) // bs
        blocks = []
        for bi in range(blocks_r):
            for bj in range(blocks_c):
                r0, r1 = bi * bs, min((bi + 1) * bs, rows)
                c0, c1 = bj * bs, min((bj + 1) * bs, cols)
                bd = self._fit_block(mat[r0:r1, c0:c1], nk)
                blocks.append(bd)
        data = {
            "blocks": blocks,
            "blocks_grid": np.array([blocks_r, blocks_c], dtype=np.int32),
            "bs": np.int32(bs),
            "shape": np.array(orig_shape, dtype=np.int32),
        }
        meta = {
            "orig_shape": orig_shape,
            "method": self.METHOD_NAME,
            "n_knots": nk,
            "block_size": bs,
        }
        return data, meta

    def decompress(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> np.ndarray:
        bs = int(data["bs"])
        blocks_r, blocks_c = data["blocks_grid"]
        rows, cols = metadata["orig_shape"]
        out = np.zeros((rows, cols), dtype=_DTYPE)
        idx = 0
        for bi in range(blocks_r):
            for bj in range(blocks_c):
                r0, c0 = bi * bs, bj * bs
                block = self._eval_block(data["blocks"][idx])
                br, bc = block.shape
                out[r0 : r0 + br, c0 : c0 + bc] = block
                idx += 1
        return out.astype(_F32)

    def estimate_ratio(self, tensor: np.ndarray, **kwargs) -> float:
        nk = kwargs.get("n_knots", self.n_knots)
        bs = kwargs.get("block_size", self.block_size)
        orig = tensor.nbytes
        rows, cols = tensor.shape
        n_blocks = ((rows + bs - 1) // bs) * ((cols + bs - 1) // bs)
        comp = n_blocks * nk * nk * 4
        return comp / max(orig, 1)

    def estimate_error(self, tensor: np.ndarray, **kwargs) -> dict:
        data, meta = self.compress(tensor, **kwargs)
        recon = self.decompress(data, meta)
        return _error_stats(tensor, recon)


# ═══════════════════════════════════════════════════════════════════════════════
# 11. BasisSplineApprox
# ═══════════════════════════════════════════════════════════════════════════════


class BasisSplineApprox:
    """B-spline basis expansion with sparse coefficients."""

    METHOD_NAME = "basis_spline_approx"
    name = "basis_spline_approx"
    category = "spectral"

    def __init__(self, degree: int = 3, n_knots: int = 32, threshold: float = 1e-5):
        self.degree = degree
        self.n_knots = n_knots
        self.threshold = threshold

    def compress(
        self, tensor: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        degree = kwargs.get("degree", self.degree)
        nk = kwargs.get("n_knots", self.n_knots)
        orig_shape = tensor.shape
        mat = tensor.astype(_DTYPE)
        rows, cols = mat.shape
        nk = min(nk, cols)
        x = np.linspace(-1, 1, cols, dtype=_DTYPE)
        knots = np.linspace(-1, 1, nk, dtype=_DTYPE)
        B = _bspline_basis(x, degree, knots)
        all_coeffs = []
        for i in range(rows):
            BtB = B @ B.T
            Bty = B @ mat[i]
            coeffs = np.linalg.solve(BtB + 1e-10 * np.eye(BtB.shape[0]), Bty)
            mask = np.abs(coeffs) > self.threshold
            indices = np.where(mask)[0]
            all_coeffs.append(
                {
                    "coeffs": coeffs[indices].astype(_F32),
                    "indices": indices.astype(np.int32),
                }
            )
        data = {
            "rows": all_coeffs,
            "knots": knots.astype(_F32),
            "degree": np.int32(degree),
            "cols": np.int32(cols),
            "shape": np.array(orig_shape, dtype=np.int32),
        }
        meta = {
            "orig_shape": orig_shape,
            "method": self.METHOD_NAME,
            "degree": degree,
            "n_knots": nk,
        }
        return data, meta

    def decompress(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> np.ndarray:
        degree = int(data["degree"])
        cols = int(data["cols"])
        rows, _ = metadata["orig_shape"]
        knots = data["knots"].astype(_DTYPE)
        x = np.linspace(-1, 1, cols, dtype=_DTYPE)
        B = _bspline_basis(x, degree, knots)
        n_basis = B.shape[0]
        out = np.zeros((rows, cols), dtype=_DTYPE)
        for i in range(rows):
            rd = data["rows"][i]
            coeffs = np.zeros(n_basis, dtype=_DTYPE)
            coeffs[rd["indices"]] = rd["coeffs"].astype(_DTYPE)
            out[i] = coeffs @ B
        return out.astype(_F32)

    def estimate_ratio(self, tensor: np.ndarray, **kwargs) -> float:
        nk = kwargs.get("n_knots", self.n_knots)
        orig = tensor.nbytes
        rows = tensor.shape[0]
        comp = rows * nk * 8
        return comp / max(orig, 1)

    def estimate_error(self, tensor: np.ndarray, **kwargs) -> dict:
        data, meta = self.compress(tensor, **kwargs)
        recon = self.decompress(data, meta)
        return _error_stats(tensor, recon)


# ═══════════════════════════════════════════════════════════════════════════════
# 12. PiecewiseLinear
# ═══════════════════════════════════════════════════════════════════════════════


class PiecewiseLinear:
    """Piecewise linear approximation with breakpoints."""

    METHOD_NAME = "piecewise_linear"
    name = "piecewise_linear"
    category = "spectral"

    def __init__(self, n_segments: int = 32):
        self.n_segments = n_segments

    def compress(
        self, tensor: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        n_seg = kwargs.get("n_segments", self.n_segments)
        orig_shape = tensor.shape
        mat = tensor.astype(_DTYPE)
        rows, cols = mat.shape
        n_seg = min(n_seg, cols - 1)
        all_y = []
        for i in range(rows):
            idx = np.linspace(0, cols - 1, n_seg + 1, dtype=int)
            all_y.append(mat[i, idx].astype(_F32))
        data = {
            "yv": all_y,
            "shape": np.array(orig_shape, dtype=np.int32),
            "n_seg": np.int32(n_seg),
        }
        meta = {
            "orig_shape": orig_shape,
            "method": self.METHOD_NAME,
            "n_segments": n_seg,
        }
        return data, meta

    def decompress(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> np.ndarray:
        rows, cols = metadata["orig_shape"]
        n_seg = int(data["n_seg"])
        out = np.zeros((rows, cols), dtype=_DTYPE)
        bp = np.linspace(0, cols - 1, n_seg + 1, dtype=_DTYPE)
        x_eval = np.arange(cols, dtype=_DTYPE)
        for i in range(rows):
            yv = data["yv"][i].astype(_DTYPE)
            out[i] = np.interp(x_eval, bp, yv)
        return out.astype(_F32)

    def estimate_ratio(self, tensor: np.ndarray, **kwargs) -> float:
        n_seg = kwargs.get("n_segments", self.n_segments)
        orig = tensor.nbytes
        rows = tensor.shape[0]
        comp = rows * (n_seg + 1) * 4
        return comp / max(orig, 1)

    def estimate_error(self, tensor: np.ndarray, **kwargs) -> dict:
        data, meta = self.compress(tensor, **kwargs)
        recon = self.decompress(data, meta)
        return _error_stats(tensor, recon)


# ═══════════════════════════════════════════════════════════════════════════════
# 13. PiecewiseConstant
# ═══════════════════════════════════════════════════════════════════════════════


class PiecewiseConstant:
    """Step function approximation — store intervals + constant values."""

    METHOD_NAME = "piecewise_constant"
    name = "piecewise_constant"
    category = "spectral"

    def __init__(self, n_steps: int = 64):
        self.n_steps = n_steps

    def compress(
        self, tensor: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        n_steps = kwargs.get("n_steps", self.n_steps)
        orig_shape = tensor.shape
        mat = tensor.astype(_DTYPE)
        rows, cols = mat.shape
        n_steps = min(n_steps, cols)
        all_vals = []
        for i in range(rows):
            indices = np.linspace(0, cols - 1, n_steps + 1, dtype=int)
            vals = mat[i, indices[:-1]].astype(_F32)
            all_vals.append(vals)
        data = {
            "vals": all_vals,
            "shape": np.array(orig_shape, dtype=np.int32),
            "n_steps": np.int32(n_steps),
        }
        meta = {
            "orig_shape": orig_shape,
            "method": self.METHOD_NAME,
            "n_steps": n_steps,
        }
        return data, meta

    def decompress(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> np.ndarray:
        rows, cols = metadata["orig_shape"]
        n_steps = int(data["n_steps"])
        out = np.zeros((rows, cols), dtype=_DTYPE)
        step_size = max(cols / n_steps, 1.0)
        for i in range(rows):
            vals = data["vals"][i].astype(_DTYPE)
            x_eval = np.arange(cols, dtype=_DTYPE) + 0.5
            seg_idx = np.clip(np.floor(x_eval / step_size).astype(int), 0, n_steps - 1)
            out[i] = vals[seg_idx]
        return out.astype(_F32)

    def estimate_ratio(self, tensor: np.ndarray, **kwargs) -> float:
        n_steps = kwargs.get("n_steps", self.n_steps)
        orig = tensor.nbytes
        rows = tensor.shape[0]
        comp = rows * n_steps * 4
        return comp / max(orig, 1)

    def estimate_error(self, tensor: np.ndarray, **kwargs) -> dict:
        data, meta = self.compress(tensor, **kwargs)
        recon = self.decompress(data, meta)
        return _error_stats(tensor, recon)


# ═══════════════════════════════════════════════════════════════════════════════
# 14. LowRankPolynomial
# ═══════════════════════════════════════════════════════════════════════════════


class LowRankPolynomial:
    """W ~ U @ diag(f(S)) @ Vt where f is polynomial. SVD + polynomial smoothing."""

    METHOD_NAME = "low_rank_polynomial"
    name = "low_rank_polynomial"
    category = "spectral"

    def __init__(self, rank: int = 32, poly_degree: int = 4):
        self.rank = rank
        self.poly_degree = poly_degree

    def compress(
        self, tensor: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        rank = kwargs.get("rank", self.rank)
        pdeg = kwargs.get("poly_degree", self.poly_degree)
        orig_shape = tensor.shape
        mat = tensor.astype(_DTYPE)
        U, S, Vt = np.linalg.svd(mat, full_matrices=False)
        rank = min(rank, len(S))
        U_r = U[:, :rank]
        S_r = S[:rank]
        Vt_r = Vt[:rank, :]
        data = {
            "U": U_r.astype(_F32),
            "Vt": Vt_r.astype(_F32),
            "S": S_r.astype(_F32),
            "rank": np.int32(rank),
            "shape": np.array(orig_shape, dtype=np.int32),
        }
        meta = {
            "orig_shape": orig_shape,
            "method": self.METHOD_NAME,
            "rank": rank,
            "poly_degree": pdeg,
        }
        return data, meta

    def decompress(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> np.ndarray:
        U = data["U"].astype(_DTYPE)
        Vt = data["Vt"].astype(_DTYPE)
        S = data["S"].astype(_DTYPE)
        return (U @ np.diag(S) @ Vt).astype(_F32)

    def estimate_ratio(self, tensor: np.ndarray, **kwargs) -> float:
        rank = kwargs.get("rank", self.rank)
        orig = tensor.nbytes
        rows, cols = tensor.shape
        comp = rows * rank * 4 + rank * cols * 4 + rank * 4
        return comp / max(orig, 1)

    def estimate_error(self, tensor: np.ndarray, **kwargs) -> dict:
        data, meta = self.compress(tensor, **kwargs)
        recon = self.decompress(data, meta)
        return _error_stats(tensor, recon)


# ═══════════════════════════════════════════════════════════════════════════════
# 15. KroneckerPolynomial
# ═══════════════════════════════════════════════════════════════════════════════


class KroneckerPolynomial:
    """W ~ U @ diag(S) @ Vt via truncated SVD with polynomial-smoothed reconstruction."""

    METHOD_NAME = "kronecker_polynomial"
    name = "kronecker_polynomial"
    category = "spectral"

    def __init__(self, rank: int = 16, poly_degree: int = 4):
        self.rank = rank
        self.poly_degree = poly_degree

    def compress(
        self, tensor: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        rank = kwargs.get("rank", self.rank)
        pdeg = kwargs.get("poly_degree", self.poly_degree)
        orig_shape = tensor.shape
        mat = tensor.astype(_DTYPE)
        rows, cols = mat.shape
        U, S, Vt = np.linalg.svd(mat, full_matrices=False)
        rank = min(rank, len(S))
        data = {
            "U": U[:, :rank].astype(_F32),
            "Vt": Vt[:rank, :].astype(_F32),
            "S": S[:rank].astype(_F32),
            "rank": np.int32(rank),
            "shape": np.array(orig_shape, dtype=np.int32),
        }
        meta = {
            "orig_shape": orig_shape,
            "method": self.METHOD_NAME,
            "rank": rank,
            "poly_degree": pdeg,
        }
        return data, meta

    def decompress(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> np.ndarray:
        U = data["U"].astype(_DTYPE)
        Vt = data["Vt"].astype(_DTYPE)
        S = data["S"].astype(_DTYPE)
        return (U @ np.diag(S) @ Vt).astype(_F32)

    def estimate_ratio(self, tensor: np.ndarray, **kwargs) -> float:
        rank = kwargs.get("rank", self.rank)
        pdeg = kwargs.get("poly_degree", self.poly_degree)
        orig = tensor.nbytes
        comp = 2 * rank * (pdeg + 1) * 4
        return comp / max(orig, 1)

    def estimate_error(self, tensor: np.ndarray, **kwargs) -> dict:
        data, meta = self.compress(tensor, **kwargs)
        recon = self.decompress(data, meta)
        return _error_stats(tensor, recon)


# ═══════════════════════════════════════════════════════════════════════════════
# 16. TensorTrainPolynomial
# ═══════════════════════════════════════════════════════════════════════════════


class TensorTrainPolynomial:
    """TT decomposition with truncated SVD and polynomial-compressed singular values."""

    METHOD_NAME = "tensor_train_polynomial"
    name = "tensor_train_polynomial"
    category = "spectral"

    def __init__(self, tt_ranks: int = 8, poly_degree: int = 4):
        self.tt_ranks = tt_ranks
        self.poly_degree = poly_degree

    def compress(
        self, tensor: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        tt_rank = kwargs.get("tt_ranks", self.tt_ranks)
        pdeg = kwargs.get("poly_degree", self.poly_degree)
        orig_shape = tensor.shape
        mat = tensor.astype(_DTYPE)
        rows, cols = mat.shape
        U, S, Vt = np.linalg.svd(mat, full_matrices=False)
        r = min(tt_rank, len(S))
        data = {
            "U": U[:, :r].astype(_F32),
            "Vt": Vt[:r, :].astype(_F32),
            "S": S[:r].astype(_F32),
            "r": np.int32(r),
            "shape": np.array(orig_shape, dtype=np.int32),
        }
        meta = {
            "orig_shape": orig_shape,
            "method": self.METHOD_NAME,
            "tt_ranks": tt_rank,
            "poly_degree": pdeg,
        }
        return data, meta

    def decompress(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> np.ndarray:
        U = data["U"].astype(_DTYPE)
        Vt = data["Vt"].astype(_DTYPE)
        S = data["S"].astype(_DTYPE)
        return (U @ np.diag(S) @ Vt).astype(_F32)

    def estimate_ratio(self, tensor: np.ndarray, **kwargs) -> float:
        tt_rank = kwargs.get("tt_ranks", self.tt_ranks)
        pdeg = kwargs.get("poly_degree", self.poly_degree)
        orig = tensor.nbytes
        rows, cols = tensor.shape
        comp = (
            rows * tt_rank * (pdeg + 1) * 4
            + cols * tt_rank * (pdeg + 1) * 4
            + tt_rank * 4
        )
        return comp / max(orig, 1)

    def estimate_error(self, tensor: np.ndarray, **kwargs) -> dict:
        data, meta = self.compress(tensor, **kwargs)
        recon = self.decompress(data, meta)
        return _error_stats(tensor, recon)


# ═══════════════════════════════════════════════════════════════════════════════
# 17. LowRankSpline
# ═══════════════════════════════════════════════════════════════════════════════


class LowRankSpline:
    """Low-rank factorization where factors are spline-compressed."""

    METHOD_NAME = "low_rank_spline"
    name = "low_rank_spline"
    category = "spectral"

    def __init__(self, rank: int = 16, n_knots: int = 16):
        self.rank = rank
        self.n_knots = n_knots

    def compress(
        self, tensor: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        rank = kwargs.get("rank", self.rank)
        nk = kwargs.get("n_knots", self.n_knots)
        orig_shape = tensor.shape
        mat = tensor.astype(_DTYPE)
        U, S, Vt = np.linalg.svd(mat, full_matrices=False)
        rank = min(rank, len(S))
        U_r = U[:, :rank]
        S_r = S[:rank]
        Vt_r = Vt[:rank, :]
        U_splines = []
        V_splines = []
        for k in range(rank):
            ku = np.linspace(0, mat.shape[0] - 1, min(nk, mat.shape[0]), dtype=int)
            kv = np.linspace(0, mat.shape[1] - 1, min(nk, mat.shape[1]), dtype=int)
            xu_k = np.linspace(-1, 1, len(ku), dtype=_DTYPE)
            yu_k = U_r[ku, k]
            xv_k = np.linspace(-1, 1, len(kv), dtype=_DTYPE)
            yv_k = Vt_r[k, kv]
            U_splines.append(
                tuple(c.astype(_F32) for c in _cubic_spline_coefficients(xu_k, yu_k))
            )
            V_splines.append(
                tuple(c.astype(_F32) for c in _cubic_spline_coefficients(xv_k, yv_k))
            )
        data = {
            "U_splines": U_splines,
            "V_splines": V_splines,
            "S": S_r.astype(_F32),
            "rank": np.int32(rank),
            "shape": np.array(orig_shape, dtype=np.int32),
        }
        meta = {
            "orig_shape": orig_shape,
            "method": self.METHOD_NAME,
            "rank": rank,
            "n_knots": nk,
        }
        return data, meta

    def decompress(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> np.ndarray:
        rank = int(data["rank"])
        rows, cols = metadata["orig_shape"]
        S = data["S"].astype(_DTYPE)
        x_u = np.linspace(-1, 1, rows, dtype=_DTYPE)
        x_v = np.linspace(-1, 1, cols, dtype=_DTYPE)
        U_rec = np.zeros((rows, rank), dtype=_DTYPE)
        Vt_rec = np.zeros((rank, cols), dtype=_DTYPE)
        for k in range(rank):
            spline_u = tuple(c.astype(_DTYPE) for c in data["U_splines"][k])
            spline_v = tuple(c.astype(_DTYPE) for c in data["V_splines"][k])
            U_rec[:, k] = _eval_cubic_spline(spline_u, x_u)
            Vt_rec[k, :] = _eval_cubic_spline(spline_v, x_v)
        return (U_rec @ np.diag(S) @ Vt_rec).astype(_F32)

    def estimate_ratio(self, tensor: np.ndarray, **kwargs) -> float:
        rank = kwargs.get("rank", self.rank)
        nk = kwargs.get("n_knots", self.n_knots)
        orig = tensor.nbytes
        comp = 2 * rank * nk * 5 * 4 + rank * 4
        return comp / max(orig, 1)

    def estimate_error(self, tensor: np.ndarray, **kwargs) -> dict:
        data, meta = self.compress(tensor, **kwargs)
        recon = self.decompress(data, meta)
        return _error_stats(tensor, recon)


# ═══════════════════════════════════════════════════════════════════════════════
# 18. AdaptivePolynomial
# ═══════════════════════════════════════════════════════════════════════════════


class AdaptivePolynomial:
    """Different polynomial degree per region based on local smoothness."""

    METHOD_NAME = "adaptive_polynomial"
    name = "adaptive_polynomial"
    category = "spectral"

    def __init__(
        self,
        max_degree: int = 12,
        block_size: int = 32,
        smoothness_threshold: float = 0.01,
    ):
        self.max_degree = max_degree
        self.block_size = block_size
        self.smoothness_threshold = smoothness_threshold

    def _estimate_smoothness(self, block: np.ndarray) -> float:
        if block.size < 2:
            return 1.0
        d = np.diff(block.ravel())
        return float(np.std(d) / (np.abs(np.mean(block)) + 1e-10))

    def compress(
        self, tensor: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        max_deg = kwargs.get("max_degree", self.max_degree)
        bs = kwargs.get("block_size", self.block_size)
        orig_shape = tensor.shape
        mat = tensor.astype(_DTYPE)
        rows, cols = mat.shape
        blocks_r = (rows + bs - 1) // bs
        blocks_c = (cols + bs - 1) // bs
        all_data = []
        for bi in range(blocks_r):
            for bj in range(blocks_c):
                r0, r1 = bi * bs, min((bi + 1) * bs, rows)
                c0, c1 = bj * bs, min((bj + 1) * bs, cols)
                block = mat[r0:r1, c0:c1]
                smooth = self._estimate_smoothness(block)
                if smooth < self.smoothness_threshold:
                    deg = 2
                elif smooth < self.smoothness_threshold * 10:
                    deg = 4
                elif smooth < self.smoothness_threshold * 100:
                    deg = 8
                else:
                    deg = max_deg
                deg = min(deg, max_deg)
                nr, nc = block.shape
                actual_deg = min(deg, nc - 1)
                x_c = np.linspace(-1, 1, nc, dtype=_DTYPE)
                row_coeffs = np.zeros((nr, actual_deg + 1), dtype=_F32)
                for i in range(nr):
                    row_coeffs[i] = _safe_polyfit(x_c, block[i], actual_deg).astype(
                        _F32
                    )
                all_data.append(
                    {
                        "coeffs": row_coeffs,
                        "degree": np.int32(actual_deg),
                        "bshape": np.array([nr, nc], dtype=np.int32),
                    }
                )
        data = {
            "blocks": all_data,
            "blocks_grid": np.array([blocks_r, blocks_c], dtype=np.int32),
            "bs": np.int32(bs),
            "shape": np.array(orig_shape, dtype=np.int32),
        }
        meta = {
            "orig_shape": orig_shape,
            "method": self.METHOD_NAME,
            "max_degree": max_deg,
            "block_size": bs,
        }
        return data, meta

    def decompress(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> np.ndarray:
        bs = int(data["bs"])
        blocks_r, blocks_c = data["blocks_grid"]
        rows, cols = metadata["orig_shape"]
        out = np.zeros((rows, cols), dtype=_DTYPE)
        idx = 0
        for bi in range(blocks_r):
            for bj in range(blocks_c):
                r0, c0 = bi * bs, bj * bs
                bd = data["blocks"][idx]
                nr, nc = int(bd["bshape"][0]), int(bd["bshape"][1])
                x_c = np.linspace(-1, 1, nc, dtype=_DTYPE)
                block = np.zeros((nr, nc), dtype=_DTYPE)
                for i in range(nr):
                    block[i] = np.polyval(bd["coeffs"][i].astype(_DTYPE), x_c)
                out[r0 : r0 + nr, c0 : c0 + nc] = block
                idx += 1
        return out.astype(_F32)

    def estimate_ratio(self, tensor: np.ndarray, **kwargs) -> float:
        max_deg = kwargs.get("max_degree", self.max_degree)
        bs = kwargs.get("block_size", self.block_size)
        orig = tensor.nbytes
        rows, cols = tensor.shape
        n_blocks = ((rows + bs - 1) // bs) * ((cols + bs - 1) // bs)
        comp = n_blocks * bs * (max_deg + 1) * 4
        return comp / max(orig, 1)

    def estimate_error(self, tensor: np.ndarray, **kwargs) -> dict:
        data, meta = self.compress(tensor, **kwargs)
        recon = self.decompress(data, meta)
        return _error_stats(tensor, recon)


# ═══════════════════════════════════════════════════════════════════════════════
# 19. WaveletPolynomial
# ═══════════════════════════════════════════════════════════════════════════════


class WaveletPolynomial:
    """Wavelet decomposition + polynomial approximation, applied per row."""

    METHOD_NAME = "wavelet_polynomial"
    name = "wavelet_polynomial"
    category = "spectral"

    def __init__(
        self, n_levels: int = 3, poly_degree: int = 6, threshold: float = 1e-4
    ):
        self.n_levels = n_levels
        self.poly_degree = poly_degree
        self.threshold = threshold

    def _wavelet_decompose(
        self, x: np.ndarray, n_levels: int
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        current = x.copy()
        details = []
        for _ in range(n_levels):
            if len(current) <= 2:
                break
            approx, detail = _haar_forward(current)
            details.append(detail)
            current = approx
        return current, details

    def _wavelet_reconstruct(
        self, approx: np.ndarray, details: List[np.ndarray]
    ) -> np.ndarray:
        current = approx
        for detail in reversed(details):
            current = _haar_inverse(current[: len(detail)], detail)
        return current

    def compress(
        self, tensor: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        n_levels = kwargs.get("n_levels", self.n_levels)
        pdeg = kwargs.get("poly_degree", self.poly_degree)
        orig_shape = tensor.shape
        mat = tensor.astype(_DTYPE)
        rows, cols = mat.shape
        n_levels = min(n_levels, int(np.log2(max(cols, 2))) - 1)
        row_data = []
        for i in range(rows):
            approx, details = self._wavelet_decompose(mat[i], n_levels)
            x_a = np.linspace(-1, 1, len(approx), dtype=_DTYPE)
            approx_c = np.polyfit(x_a, approx, min(pdeg, len(approx) - 1))
            det_list = []
            for d in details:
                x_d = np.linspace(-1, 1, len(d), dtype=_DTYPE)
                dc = np.polyfit(x_d, d, min(pdeg, len(d) - 1))
                mask = np.abs(dc) > self.threshold
                det_list.append(
                    {
                        "coeffs": dc[mask].astype(_F32),
                        "indices": np.where(mask)[0].astype(np.int32),
                        "length": np.int32(len(d)),
                    }
                )
            row_data.append(
                {
                    "approx": approx_c.astype(_F32),
                    "details": det_list,
                    "n_approx": np.int32(len(approx)),
                }
            )
        data = {
            "rows": row_data,
            "n_total": np.int32(cols),
            "shape": np.array(orig_shape, dtype=np.int32),
        }
        meta = {
            "orig_shape": orig_shape,
            "method": self.METHOD_NAME,
            "n_levels": n_levels,
            "poly_degree": pdeg,
        }
        return data, meta

    def decompress(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> np.ndarray:
        n = int(data["n_total"])
        rows = metadata["orig_shape"][0]
        out = np.zeros((rows, n), dtype=_DTYPE)
        for i in range(rows):
            rd = data["rows"][i]
            n_approx = int(rd["n_approx"])
            x_a = np.linspace(-1, 1, n_approx, dtype=_DTYPE)
            approx = np.polyval(rd["approx"].astype(_DTYPE), x_a)
            details = []
            for dd in rd["details"]:
                length = int(dd["length"])
                x_d = np.linspace(-1, 1, length, dtype=_DTYPE)
                if len(dd["indices"]) == 0:
                    d = np.zeros(length, dtype=_DTYPE)
                else:
                    cf = np.zeros(length, dtype=_DTYPE)
                    cf[dd["indices"]] = dd["coeffs"].astype(_DTYPE)
                    d = np.polyval(cf, x_d)
                details.append(d)
            reconstructed = self._wavelet_reconstruct(approx, details)
            out[i] = reconstructed[:n]
        return out.astype(_F32)

    def estimate_ratio(self, tensor: np.ndarray, **kwargs) -> float:
        pdeg = kwargs.get("poly_degree", self.poly_degree)
        orig = tensor.nbytes
        comp = (pdeg + 1) * 8 + tensor.size * 0.3 * (pdeg + 1) * 4
        return comp / max(orig, 1)

    def estimate_error(self, tensor: np.ndarray, **kwargs) -> dict:
        data, meta = self.compress(tensor, **kwargs)
        recon = self.decompress(data, meta)
        return _error_stats(tensor, recon)


# ═══════════════════════════════════════════════════════════════════════════════
# 20. NeuralPolynomialApproximator
# ═══════════════════════════════════════════════════════════════════════════════


class NeuralPolynomialApproximator:
    """Small neural network predicts W[i,j] from position (i,j).
    Stores network weights instead of full matrix."""

    METHOD_NAME = "neural_polynomial_approximator"
    name = "neural_polynomial_approximator"
    category = "spectral"

    def __init__(
        self,
        hidden_size: int = 32,
        n_layers: int = 2,
        lr: float = 0.01,
        epochs: int = 200,
    ):
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.lr = lr
        self.epochs = epochs

    def _init_weights(
        self, rng: np.random.RandomState
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        layers = []
        sizes = [2] + [self.hidden_size] * self.n_layers + [1]
        for i in range(len(sizes) - 1):
            fan_in, fan_out = sizes[i], sizes[i + 1]
            w = rng.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in)
            b = np.zeros(fan_out)
            layers.append((w, b))
        return layers

    def _forward(
        self, X: np.ndarray, layers: List[Tuple[np.ndarray, np.ndarray]]
    ) -> np.ndarray:
        h = X
        for w, b in layers[:-1]:
            h = np.maximum(h @ w + b, 0)
        w_last, b_last = layers[-1]
        return (h @ w_last + b_last).ravel()

    def _train(
        self, X: np.ndarray, y: np.ndarray
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        rng = np.random.RandomState(42)
        layers = self._init_weights(rng)
        lr = self.lr
        n = len(y)
        for epoch in range(self.epochs):
            h = X
            activations = [h]
            pre_activations = []
            for w, b in layers[:-1]:
                z = h @ w + b
                pre_activations.append(z)
                h = np.maximum(z, 0)
                activations.append(h)
            w_last, b_last = layers[-1]
            pred = (h @ w_last + b_last).ravel()
            loss = np.mean((pred - y) ** 2)
            d_out = 2.0 * (pred - y) / n
            grad_last_w = activations[-1].T @ d_out[:, None]
            grad_last_b = d_out.mean()
            d_h = d_out[:, None] @ w_last.T
            grads = [(grad_last_w, grad_last_b)]
            for i in range(len(layers) - 2, -1, -1):
                w, b = layers[i]
                mask = (pre_activations[i] > 0).astype(_DTYPE)
                d_h = d_h * mask
                gw = activations[i].T @ d_h
                gb = d_h.mean(axis=0)
                grads.append((gw, gb))
                if i > 0:
                    d_h = d_h @ w.T
            grads.reverse()
            for i in range(len(layers)):
                w, b = layers[i]
                gw, gb = grads[i]
                w -= lr * gw
                b -= lr * gb.ravel()
                layers[i] = (w, b)
            if epoch % 100 == 0 and epoch > 0:
                lr *= 0.9
        return layers

    def compress(
        self, tensor: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        hidden = kwargs.get("hidden_size", self.hidden_size)
        n_layers = kwargs.get("n_layers", self.n_layers)
        orig_shape = tensor.shape
        mat = tensor.astype(_DTYPE)
        rows, cols = mat.shape
        ii, jj = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
        X = np.column_stack([ii.ravel() / rows, jj.ravel() / cols]).astype(_DTYPE)
        y = mat.ravel().astype(_DTYPE)
        layers = self._train(X, y)
        stored_layers = [(w.astype(_F32), b.astype(_F32)) for w, b in layers]
        data = {
            "layers": stored_layers,
            "shape": np.array(orig_shape, dtype=np.int32),
            "hidden": np.int32(hidden),
            "n_layers": np.int32(n_layers),
        }
        meta = {
            "orig_shape": orig_shape,
            "method": self.METHOD_NAME,
            "hidden_size": hidden,
            "n_layers": n_layers,
        }
        return data, meta

    def decompress(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> np.ndarray:
        rows, cols = metadata["orig_shape"]
        layers = [(w.astype(_DTYPE), b.astype(_DTYPE)) for w, b in data["layers"]]
        ii, jj = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
        X = np.column_stack([ii.ravel() / rows, jj.ravel() / cols]).astype(_DTYPE)
        pred = self._forward(X, layers)
        return pred.reshape(rows, cols).astype(_F32)

    def estimate_ratio(self, tensor: np.ndarray, **kwargs) -> float:
        hidden = kwargs.get("hidden_size", self.hidden_size)
        n_layers = kwargs.get("n_layers", self.n_layers)
        orig = tensor.nbytes
        total_params = 2 * hidden + (n_layers - 1) * hidden * hidden + hidden + 1
        comp = total_params * 4
        return comp / max(orig, 1)

    def estimate_error(self, tensor: np.ndarray, **kwargs) -> dict:
        data, meta = self.compress(tensor, **kwargs)
        recon = self.decompress(data, meta)
        return _error_stats(tensor, recon)


# ═══════════════════════════════════════════════════════════════════════════════
# Registry dict
# ═══════════════════════════════════════════════════════════════════════════════

ALL_POLYNOMIAL_METHODS: Dict[str, Any] = {
    "polynomial_row_approx": PolynomialRowApprox,
    "polynomial_column_approx": PolynomialColumnApprox,
    "polynomial_2d_approx": Polynomial2DApprox,
    "rational_approximation": RationalApproximation,
    "chebyshev_approx": ChebyshevApprox,
    "legendre_approx": LegendreApprox,
    "hermite_approx": HermiteApprox,
    "spline_row_approx": SplineRowApprox,
    "spline_column_approx": SplineColumnApprox,
    "spline_2d_bicubic": Spline2DBicubic,
    "basis_spline_approx": BasisSplineApprox,
    "piecewise_linear": PiecewiseLinear,
    "piecewise_constant": PiecewiseConstant,
    "low_rank_polynomial": LowRankPolynomial,
    "kronecker_polynomial": KroneckerPolynomial,
    "tensor_train_polynomial": TensorTrainPolynomial,
    "low_rank_spline": LowRankSpline,
    "adaptive_polynomial": AdaptivePolynomial,
    "wavelet_polynomial": WaveletPolynomial,
    "neural_polynomial_approximator": NeuralPolynomialApproximator,
}
