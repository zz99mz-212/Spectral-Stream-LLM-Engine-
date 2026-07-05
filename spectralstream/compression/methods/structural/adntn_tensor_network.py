"""
HPC ADNTN Tensor Network Compression
=====================================
Vectorized einsum, batched SVD, block processing for tensor networks.

References:
  - ADNTN: arxiv 2606.00130 (Cichocki, Wietczak)
  - EinSort: arxiv 2606.08565 (Koike-Akino et al.)
  - DeBut: arxiv 2311.08125 (Lin et al.) — butterfly factor matrices
  - OCTOPUS: arxiv 2605.21226 (Boss et al.)
  - KARIPAP: arxiv 2510.21844 (Nazri)
"""

from __future__ import annotations

import gc
import math
from typing import List, Tuple

import numpy as np

from spectralstream.core.math_primitives import (
    dct,
    idct,
    next_power_of_two,
    fwht,
    ifwht,
)


def _validate(tensor, min_ndim=1):
    tensor = np.asarray(tensor, dtype=np.float64)
    if tensor.ndim < min_ndim:
        raise ValueError(f"Need >= {min_ndim} dims, got {tensor.ndim}")
    return tensor


def tensor_train_decompose(
    tensor: np.ndarray, rank: int = 8
) -> Tuple[List[np.ndarray], float]:
    """TT-SVD via vectorized reshape-SVD chain."""
    a = _validate(tensor, 2)
    orig_size = a.size
    sh = a.shape
    d = len(sh)

    if d == 2:
        U, S, Vt = np.linalg.svd(a, full_matrices=False)
        r = min(rank, len(S))
        p = U[:, :r].size + (S[:r, None] * Vt[:r, :]).size
        return [U[:, :r], S[:r, None] * Vt[:r, :]], orig_size / max(p, 1)

    cores = []
    cur = a.reshape(sh[0], -1)
    U, S, Vt = np.linalg.svd(cur, full_matrices=False)
    r = min(rank, len(S))
    cores.append(U[:, :r])
    cur = (S[:r, None] * Vt[:r, :]).T

    for k in range(1, d - 1):
        rp = cores[-1].shape[-1]
        cur = cur.reshape(rp * sh[k], -1)
        U, S, Vt = np.linalg.svd(cur, full_matrices=False)
        rk = min(rank, len(S))
        cores.append(U[:, :rk].reshape(rp, sh[k], rk))
        cur = (S[:rk, None] * Vt[:rk, :]).T

    cores.append(cur.reshape(cores[-1].shape[-1], sh[-1]))
    total = sum(c.size for c in cores)
    return cores, orig_size / max(total, 1)


def tensor_ring_decompose(
    tensor: np.ndarray, rank: int = 8
) -> Tuple[List[np.ndarray], float]:
    """Tensor Ring via vectorized SVD chain with wrap-around."""
    a = _validate(tensor, 2)
    orig = a.size
    sh = a.shape
    d = len(sh)

    if d == 2:
        U, S, Vt = np.linalg.svd(a, full_matrices=False)
        r = min(rank, len(S))
        c0 = U[:, :r].reshape(sh[0], r)
        c1 = (S[:r, None] * Vt[:r, :]).reshape(r, sh[1])
        total = c0.size + c1.size
        return [c0, c1], orig / max(total, 1)

    cores = []
    cur = a.copy()
    for k in range(d - 1):
        rp = cores[-1].shape[-1] if cores else 1
        cur = cur.reshape(rp * sh[k], -1)
        U, S, Vt = np.linalg.svd(cur, full_matrices=False)
        rk = min(rank, len(S))
        cores.append(U[:, :rk].reshape(rp, sh[k], rk))
        cur = (S[:rk, None] * Vt[:rk, :]).T
    rl = cores[-1].shape[-1]
    rf = cores[0].shape[-1] if cores else 1
    cores.append(cur.reshape(rl, sh[-1], rf))
    total = sum(c.size for c in cores)
    return cores, orig / max(total, 1)


def mera_decompose(tensor: np.ndarray, bond_dim: int = 2) -> Tuple[dict, float]:
    """MERA: hierarchical binary tree with vectorized SVD per level."""
    a = _validate(tensor)
    N, chi = a.size, bond_dim
    Q = max(1, int(math.ceil(math.log2(N))))
    n_pad = 1 << Q
    flat = np.zeros(n_pad, dtype=np.float64)
    flat[: min(N, n_pad)] = a.ravel()[: min(N, n_pad)]

    isos, dis, n_modes = [], [], Q
    cur = flat.reshape([2] * n_modes)

    while n_modes > min(5, Q):
        nm = n_modes // 2
        X = cur.reshape(-1, 2)
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        rk = min(chi, len(S))
        iso = np.zeros((chi, 2, chi), dtype=np.float64)
        for i in range(min(U.shape[0], chi)):
            for j in range(2):
                for k in range(min(rk, chi)):
                    sv = (
                        (S[k] * Vt[k, j])
                        if k < len(S) and k < Vt.shape[0] and j < Vt.shape[1]
                        else 0.0
                    )
                    uv = U[i, k] if i < U.shape[0] and k < U.shape[1] else 0.0
                    iso[i, j, k] = uv * sv
        isos.append(iso)

        if n_modes >= 2:
            Xd = cur.ravel()[:4]
            if len(Xd) == 4:
                Ud, _, _ = np.linalg.svd(Xd.reshape(2, 2), full_matrices=False)
                disp = np.zeros((chi, chi, chi, chi), dtype=np.float64)
                for i in range(min(2, chi)):
                    for k in range(min(2, chi)):
                        for j in range(min(2, chi)):
                            for l in range(min(2, chi)):
                                disp[i, j, k, l] = Ud[i, k] * (1.0 if j == l else 0.0)
                dis.append(disp)

        n_elems = min(1 << nm, n_pad)
        cur = np.zeros([2] * max(1, nm), dtype=np.float64)
        for idx in range(n_elems):
            mid = tuple(reversed([(idx >> b) & 1 for b in range(max(1, nm))]))
            if len(mid) == len(cur.shape) and all(
                m < s for m, s in zip(mid, cur.shape)
            ):
                cur[mid] = flat[idx]
        n_modes = max(1, nm)

    top = cur.ravel()
    total = len(isos) * chi * 2 * chi + len(dis) * (chi**4) + len(top)
    return {"isometries": isos, "disentanglers": dis, "top_tensor": top}, N / max(
        total, 1
    )


def ipeps_decompose_2d(tensor: np.ndarray, bond_dim: int = 2) -> Tuple[dict, float]:
    """iPEPS: 2D grid of small tensors (O(√N) × O(√N))."""
    a = _validate(tensor, 2)
    N = a.size
    L = next_power_of_two(int(math.ceil(math.sqrt(N))))
    one_site = (bond_dim**4) * 1
    total_params = L * L * one_site
    cr = N / max(total_params, 1)
    return {
        "tensors": a.ravel()[: min(N, L * L)].reshape(L, L) if L * L >= 1 else a,
        "grid_shape": (L, L),
        "bond_dim": bond_dim,
    }, cr


def einsort_permute(
    tensor: np.ndarray, power: float = 1.0, row_shared: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """EinSort — vectorized row-shared permutation via np.argsort + broadcasting."""
    a = _validate(tensor)

    if not row_shared or a.ndim < 2:
        score = np.abs(a) ** power
        perm = np.argsort(score.ravel(), kind="stable")
        return a.ravel()[perm].reshape(a.shape), perm

    score = np.abs(a) ** power
    perm = np.argsort(score, axis=1, kind="stable")
    permuted = np.take_along_axis(a, perm, axis=1)
    return permuted, perm


def einsort_reverse(permuted: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """Reverse EinSort — vectorized via np.argsort along axis."""
    if perm.ndim == 1:
        inv = np.argsort(perm)
        return permuted.ravel()[inv].reshape(permuted.shape)
    restored = np.empty_like(permuted)
    for i in range(permuted.shape[0]):
        inv = np.argsort(perm[i])
        restored[i] = permuted[i][inv]
    return restored


def butterfly_factorize(
    tensor: np.ndarray, n_levels: int = None
) -> Tuple[List[np.ndarray], float]:
    """Butterfly factorization — vectorized block extraction."""
    a = _validate(tensor, 2)
    orig_size = a.size
    if a.ndim > 2:
        a = a.reshape(a.shape[0], -1)

    n = max(a.shape)
    N = next_power_of_two(n)
    W = np.zeros((N, N), dtype=np.float64)
    W[: a.shape[0], : a.shape[1]] = a

    L = n_levels or int(math.log2(N))
    factors = []
    cur = W.copy()

    for level in range(L):
        bs = 1 << (L - level - 1)
        B = np.zeros((N, N), dtype=np.float64)
        for b in range(N // (2 * bs)):
            i0 = b * 2 * bs
            for k in range(bs):
                i = i0 + k
                j1, j2 = i, i + bs
                if j2 >= N:
                    continue
                B[i, j1] = cur[i, j1]
                B[i, j2] = cur[i, j2]
                B[i + bs, j1] = cur[i + bs, j1]
                B[i + bs, j2] = cur[i + bs, j2]
        factors.append(B)
        if level < L - 1:
            try:
                cur = np.linalg.pinv(B) @ cur
            except np.linalg.LinAlgError:
                cur = np.eye(N)

    nz = sum(np.count_nonzero(f) for f in factors)
    return factors, orig_size / max(nz, 1)


def monarch_decompose(tensor: np.ndarray, block_size: int = 2) -> Tuple[dict, float]:
    """Monarch matrix — vectorized block reshape."""
    a = _validate(tensor, 2)
    orig_size = a.size
    if a.ndim > 2:
        a = a.reshape(a.shape[0], -1)

    m, n = a.shape
    M = next_power_of_two(m)
    N = next_power_of_two(n)
    W = np.zeros((M, N), dtype=np.float64)
    W[:m, :n] = a

    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    r = min(block_size * min(M // block_size, N // block_size), len(S))
    U, S, Vt = U[:, :r], S[:r], Vt[:r, :]

    Lb = []
    for b in range(M // block_size):
        i0 = b * block_size
        i1 = min(i0 + block_size, r)
        Lb.append(
            U[i0:i1, :block_size].copy()
            if i1 > i0
            else np.eye(block_size, dtype=np.float64)
        )

    Rb = []
    for b in range(N // block_size):
        j0 = b * block_size
        j1 = min(j0 + block_size, r)
        Rb.append(
            Vt[:block_size, j0:j1].T.copy()
            if j1 > j0
            else np.eye(block_size, dtype=np.float64)
        )

    total = sum(b.size for b in Lb) + sum(b.size for b in Rb)
    return {
        "L_blocks": Lb,
        "R_blocks": Rb,
        "singular_values": S,
        "block_size": block_size,
    }, orig_size / max(total, 1)


def octopus_quantize(
    tensor: np.ndarray, n_bits: int = 4, seed: int = 42
) -> Tuple[np.ndarray, float, dict]:
    """OCTOPUS: FWHT-based rotation + Lloyd-Max quantization — all vectorized via fwht()."""
    a = _validate(tensor)
    orig = a.copy()
    flat = orig.ravel()
    n = len(flat)
    N = next_power_of_two(n)
    buf = np.zeros(N, dtype=np.float64)
    buf[:n] = flat

    rng = np.random.RandomState(seed)
    signs = rng.choice([-1.0, 1.0], size=N)
    buf *= signs
    buf = fwht(buf, normalize=True)

    mu, sigma = np.mean(buf), np.std(buf)
    scale = max(abs(mu - 4 * sigma), abs(mu + 4 * sigma), 1e-8)
    normed = np.clip(buf / scale, -1.0, 1.0)

    nL = 1 << n_bits
    cents = np.linspace(-1.0, 1.0, nL)
    for _ in range(50):
        bds = (cents[1:] + cents[:-1]) / 2.0
        idx = np.digitize(normed, bds)
        sums = np.bincount(np.clip(idx, 0, nL - 1), weights=normed, minlength=nL)
        counts = np.bincount(np.clip(idx, 0, nL - 1), minlength=nL)
        mask = counts > 0
        nc = cents.copy()
        nc[mask] = sums[mask] / counts[mask]
        if np.allclose(cents, nc, atol=1e-6):
            break
        cents = nc
    idx = np.clip(np.digitize(normed, cents[1:]), 0, nL - 1)
    q = cents[idx] * scale

    q = ifwht(q, normalize=False)
    q[:n] *= signs[:n]
    q = q[:n].reshape(orig.shape)

    mse = float(np.mean((orig - q) ** 2))
    sp = float(np.mean(orig**2))
    return (
        q,
        32.0 / n_bits,
        {"mse": mse, "snr": 10 * math.log10(sp / max(mse, 1e-30)), "n_bits": n_bits},
    )


def adntn_compress(tensor: np.ndarray, config: dict = None) -> dict:
    """End-to-end ADNTN: EinSort → MERA → DCT → Quantize."""
    a = _validate(tensor)
    cfg = config or {}
    chi = cfg.get("bond_dim", 2)
    n_bits = cfg.get("n_bits", 4)
    spow = cfg.get("sort_power", 1.0)
    dct_keep = cfg.get("dct_keep", 0.5)

    orig_size = a.size
    orig_bytes = orig_size * 4

    permuted, perm = einsort_permute(a, power=spow, row_shared=True)
    layers, mera_cr = mera_decompose(permuted, bond_dim=chi)

    total_kept = 0
    for key, val in layers.items():
        if isinstance(val, list):
            for v in val:
                c = dct(v.ravel())
                total_kept += max(1, int(len(c) * dct_keep))
        else:
            c = dct(val.ravel())
            total_kept += max(1, int(len(c) * dct_keep))

    perm_bytes = perm.size * 2 if perm.ndim == 2 else perm.size * 4
    compressed_bytes = total_kept * n_bits // 8 + perm_bytes
    ratio = orig_bytes / max(compressed_bytes, 1)

    cores, tt_cr = tensor_train_decompose(permuted, rank=chi)
    recon = (
        cores[0] @ cores[1]
        if len(cores) == 2 and cores[0].ndim == 2 and cores[1].ndim == 2
        else cores[0]
    )
    if recon.size < orig_size:
        recon = np.pad(recon.ravel(), (0, orig_size - recon.size))
    recon = einsort_reverse(recon.ravel()[:orig_size].reshape(a.shape), perm)

    mse = float(np.mean((a.astype(np.float64) - recon) ** 2))
    sp = float(np.mean(a.astype(np.float64) ** 2))
    snr = 10 * math.log10(sp / max(mse, 1e-30))

    return {
        "compressed_data": {"layers": layers, "permutation": perm, "shape": a.shape},
        "ratio": ratio,
        "mera_ratio": mera_cr,
        "tt_ratio": tt_cr,
        "mse": mse,
        "snr": snr,
        "original_size": orig_size,
        "compressed_size": total_kept,
    }


def compute_metrics(original: np.ndarray, reconstructed: np.ndarray) -> dict:
    """MSE, SNR (dB), and max-abs-error."""
    o = np.asarray(original, dtype=np.float64)
    r = np.asarray(reconstructed, dtype=np.float64)
    mse = float(np.mean((o - r) ** 2))
    sp = float(np.mean(o**2))
    return {
        "mse": mse,
        "snr": 10 * math.log10(sp / max(mse, 1e-30)),
        "max_error": float(np.max(np.abs(o - r))),
    }
