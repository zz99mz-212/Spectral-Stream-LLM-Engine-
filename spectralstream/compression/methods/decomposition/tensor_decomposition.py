"""
Tensor decomposition methods for n-dimensional tensors.
All methods return: (compressed: dict, ratio: float, snr_db: float)
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np
from scipy.linalg import svd


def _bytes(obj) -> int:
    if isinstance(obj, np.ndarray):
        return obj.nbytes
    if isinstance(obj, dict):
        return sum(_bytes(v) for _, v in obj.items())
    if isinstance(obj, (list, tuple)):
        return sum(_bytes(x) for x in obj)
    return 8


def _snr(orig: np.ndarray, recon: np.ndarray) -> float:
    o = orig.astype(np.float64).ravel()
    r = recon.astype(np.float64).ravel()
    mse = np.mean((o - r) ** 2)
    return float(10.0 * np.log10(np.mean(o**2) / (mse + 1e-30)))


def _cp_reconstruct(factors: List[np.ndarray]) -> np.ndarray:
    ndim = len(factors)
    if ndim == 2:
        return factors[0] @ factors[1].T
    labels = [chr(97 + i) for i in range(ndim)]
    sub = ",".join(f"{l}r" for l in labels) + "->" + "".join(labels)
    return np.einsum(sub, *factors)


def _tucker_reconstruct(core: np.ndarray, factors: List[np.ndarray]) -> np.ndarray:
    ndim = len(factors)
    if ndim == 2:
        return factors[0] @ core @ factors[1].T
    labels_core = [chr(97 + i) for i in range(ndim)]
    labels_out = [chr(97 + ndim + i) for i in range(ndim)]
    subs = (
        "".join(labels_core)
        + ","
        + ",".join(f"{labels_out[i]}{labels_core[i]}" for i in range(ndim))
        + "->"
        + "".join(labels_out)
    )
    return np.einsum(subs, core, *factors)


def _tt_reconstruct(cores: List[np.ndarray], shape: Tuple[int, ...]) -> np.ndarray:
    z = cores[0].astype(np.float64)
    for k in range(1, len(cores)):
        if k == len(cores) - 1:
            z = z.reshape(-1, cores[k].shape[0]) @ cores[k].astype(np.float64)
        else:
            z = np.tensordot(z, cores[k].astype(np.float64), axes=[-1, 0])
    return z.reshape(shape)


def _tr_reconstruct(cores: List[np.ndarray], shape: Tuple[int, ...]) -> np.ndarray:
    ndim = len(shape)
    if ndim == 2:
        return cores[0].astype(np.float64) @ cores[1].astype(np.float64)
    z = cores[0].astype(np.float64)
    for k in range(1, ndim - 1):
        z = np.tensordot(z, cores[k].astype(np.float64), axes=[-1, 0])
    z = np.tensordot(z, cores[-1].astype(np.float64), axes=[-1, 0])
    z = np.trace(z, axis1=0, axis2=z.ndim - 1)
    return z.reshape(shape)


def cp_decomposition(
    tensor: np.ndarray, rank: int = 8, max_iters: int = 50, tol: float = 1e-6
) -> Tuple[Dict, float, float]:
    """CANDECOMP/PARAFAC (CP) decomposition.
    Factorizes tensor into sum of rank-1 tensors.
    W ~ sum_{r=1}^R a_r^{(1)} circ a_r^{(2)} circ ... circ a_r^{(N)}
    Uses ALS (Alternating Least Squares).
    """
    t = np.asarray(tensor, dtype=np.float64)
    shape = t.shape
    ndim = t.ndim
    rank = min(rank, max(shape))
    rank = max(1, rank)

    rng = np.random.RandomState(42)
    factors = [rng.randn(s, rank) for s in shape]

    for _ in range(max_iters):
        old = [f.copy() for f in factors]
        for n in range(ndim):
            V = np.ones((rank, rank))
            for j in range(ndim):
                if j != n:
                    V *= factors[j].T @ factors[j]

            kr = None
            for j in range(ndim):
                if j == n:
                    continue
                if kr is None:
                    kr = factors[j]
                else:
                    kr = (kr[:, None, :] * factors[j][None, :, :]).reshape(-1, rank)

            unfold = np.moveaxis(t, n, 0).reshape(shape[n], -1)
            factors[n] = unfold @ kr @ np.linalg.pinv(V)

        diff = sum(np.linalg.norm(factors[i] - old[i], ord="fro") for i in range(ndim))
        tot = sum(np.linalg.norm(f, ord="fro") for f in factors)
        if diff < tol * max(tot, 1e-30):
            break

    c = {f"factor_{i}": f.astype(np.float32) for i, f in enumerate(factors)}
    c["shape"] = shape
    c["rank"] = rank

    recon = _cp_reconstruct(factors)
    return c, tensor.nbytes / max(_bytes(c), 1), _snr(t, recon)


def tucker_decomposition(
    tensor: np.ndarray, ranks: List[int] = None
) -> Tuple[Dict, float, float]:
    """Tucker decomposition (higher-order SVD).
    W ~ G x_1 U_1 x_2 U_2 x_3 ... x_N U_N
    G is the core tensor, U_i are factor matrices.
    """
    t = np.asarray(tensor, dtype=np.float64)
    shape = t.shape
    ndim = t.ndim

    if ranks is None:
        ranks = [max(1, s // 2) for s in shape]
    ranks = [min(r, s) for r, s in zip(ranks, shape)]

    factors = []
    for n in range(ndim):
        unfold = np.moveaxis(t, n, 0).reshape(shape[n], -1)
        U, s, _ = svd(unfold, full_matrices=False)
        rn = min(ranks[n], len(s))
        factors.append(U[:, :rn])

    labels_t = [chr(97 + i) for i in range(ndim)]
    labels_c = [chr(97 + ndim + i) for i in range(ndim)]
    subs = (
        "".join(labels_t)
        + ","
        + ",".join(f"{labels_t[i]}{labels_c[i]}" for i in range(ndim))
        + "->"
        + "".join(labels_c)
    )
    core = np.einsum(subs, t, *factors)

    c = {"core": core.astype(np.float32), "shape": shape, "ranks": ranks}
    for i, f in enumerate(factors):
        c[f"factor_{i}"] = f.astype(np.float32)

    recon = _tucker_reconstruct(core, factors)
    return c, tensor.nbytes / max(_bytes(c), 1), _snr(t, recon)


def tensor_train_decompose(
    tensor: np.ndarray, rank: int = 8
) -> Tuple[Dict, float, float]:
    """Tensor Train decomposition via sequential SVD.
    W(i1,...,id) ~ G1(i1) @ G2(i2) @ ... @ Gd(id)
    Each core has shape [r_{k-1}, n_k, r_k].
    """
    t = np.asarray(tensor, dtype=np.float64)
    shape = t.shape
    d = len(shape)
    rank = max(1, rank)

    cores = []
    cur = t.copy()
    for k in range(d - 1):
        rp = cores[-1].shape[-1] if cores else 1
        mat = cur.reshape(rp * shape[k], -1)
        U, s, Vt = svd(mat, full_matrices=False)
        rk = min(rank, len(s))
        if k == 0:
            cores.append(U[:, :rk])
        else:
            cores.append(U[:, :rk].reshape(rp, shape[k], rk))
        cur = (s[:rk, None] * Vt[:rk, :]).T

    cores.append(cur.T)

    c = {"cores": [cr.astype(np.float32) for cr in cores], "shape": shape}
    recon = _tt_reconstruct(cores, shape)
    return c, tensor.nbytes / max(_bytes(c), 1), _snr(t, recon)


def tensor_ring_decompose(
    tensor: np.ndarray, rank: int = 8
) -> Tuple[Dict, float, float]:
    """Tensor Ring decomposition (closed loop of TT cores).
    Similar to TT but with trace constraint for cyclic structure.
    """
    t = np.asarray(tensor, dtype=np.float64)
    shape = t.shape
    d = len(shape)
    rank = max(1, rank)

    if d == 2:
        U, s, Vt = svd(t, full_matrices=False)
        r = min(rank, len(s))
        c0 = U[:, :r]
        c1 = s[:r, None] * Vt[:r, :]
        cores = [c0, c1]
        c = {"cores": [cr.astype(np.float32) for cr in cores], "shape": shape}
        recon = c0 @ c1
        return c, tensor.nbytes / max(_bytes(c), 1), _snr(t, recon)

    cores = []
    cur = t.copy()
    for k in range(d - 1):
        rp = cores[-1].shape[-1] if cores else 1
        mat = cur.reshape(rp * shape[k], -1)
        U, s, Vt = svd(mat, full_matrices=False)
        rk = min(rank, len(s))
        cores.append(U[:, :rk].reshape(rp, shape[k], rk))
        cur = (s[:rk, None] * Vt[:rk, :]).T

    rl = cores[-1].shape[-1]
    rf = cores[0].shape[0] if cores else 1
    cores.append(cur.T.reshape(rl, shape[-1], rf))

    c = {"cores": [cr.astype(np.float32) for cr in cores], "shape": shape}
    recon = _tr_reconstruct(cores, shape)
    return c, tensor.nbytes / max(_bytes(c), 1), _snr(t, recon)


if __name__ == "__main__":
    t = np.random.randn(128, 128).astype(np.float32)
    data, ratio, snr = cp_decomposition(t, rank=16)
    print(f"CP: {ratio:.2f}x, SNR={snr:.1f}dB")
    data, ratio, snr = tucker_decomposition(t, ranks=[32, 32])
    print(f"Tucker: {ratio:.2f}x, SNR={snr:.1f}dB")
    data, ratio, snr = tensor_train_decompose(t, rank=16)
    print(f"TT: {ratio:.2f}x, SNR={snr:.1f}dB")
    data, ratio, snr = tensor_ring_decompose(t, rank=16)
    print(f"TR: {ratio:.2f}x, SNR={snr:.1f}dB")
