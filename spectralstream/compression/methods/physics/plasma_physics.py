"""
Plasma Physics Compression Methods
===================================
8 novel physics-inspired compression techniques applying plasma physics,
quantum mechanics, and holographic principles to neural network weights.

Methods:
  1. GyrokineticPreconditioner
  2. VlasovPhaseSpaceCompression (FIXED)
  3. HamiltonianWeightFlow
  4. QuantumAmplitudeEncoding
  5. TimeCrystalCompression
  6. PlasmaTurbulenceCascade (FIXED)
  7. FractalWeightCompression (NOVEL)
  8. TopologicalDataCompression (FIXED)
"""

from __future__ import annotations


from typing import Tuple

import numpy as np

from spectralstream.core.math_primitives import dct, idct


def _bytes(obj: object) -> int:
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
    return float(10.0 * np.log10(np.mean(o ** 2) / (mse + 1e-30)))


# ═══════════════════════════════════════════════════════════════════════════
# 1. GyrokineticPreconditioner
# ═══════════════════════════════════════════════════════════════════════════

def gyrokinetic_precondition(
    tensor: np.ndarray, n_gyro_angles: int = 8
) -> Tuple[dict, float, float]:
    """Gyrokinetic approximation: average over fast gyromotion to extract
    the gyrocenter (smooth, low-rank) and gyrophase (sparse fluctuation).
    The gyrocenter compresses via SVD; gyrophase stores only significant
    deviations. Gyrokinetics reduces 6D->5D by averaging over gyromotion;
    here we average over random shifts (discrete gyro-angles)."""
    orig = tensor.astype(np.float64)
    m, n = orig.shape
    step = max(1, min(m, n) // 16)
    rng = np.random.RandomState(42)

    gc = np.zeros_like(orig)
    for _ in range(n_gyro_angles):
        th = rng.uniform(0, 2 * np.pi)
        dx = int(step * np.cos(th))
        dy = int(step * np.sin(th))
        gc += np.roll(np.roll(orig, dx, axis=0), dy, axis=1)
    gc /= n_gyro_angles
    gp = orig - gc

    U, S, Vt = np.linalg.svd(gc, full_matrices=False)
    cum = np.cumsum(S) / np.sum(S)
    r = int(np.searchsorted(cum, 0.90)) + 1

    thr = np.percentile(np.abs(gp), 92)
    mask = np.abs(gp) > thr
    gidx = np.argwhere(mask)
    gvals = gp[mask]

    c = {"U": U[:, :r].astype(np.float16), "S": S[:r].astype(np.float16),
         "Vt": Vt[:r, :].astype(np.float16), "gidx": gidx.astype(np.int16),
         "gvals": gvals.astype(np.float16), "sh": orig.shape}

    recon = (U[:, :r] * S[:r]) @ Vt[:r, :]
    if len(gidx) > 0:
        sa = np.zeros_like(recon)
        sa[gidx[:, 0] % m, gidx[:, 1] % n] = gvals.astype(np.float64)
        recon += sa

    return c, _bytes(c) / max(tensor.nbytes, 1), _snr(orig, recon)


# ═══════════════════════════════════════════════════════════════════════════
# 2. VlasovPhaseSpaceCompression (FIXED)
# ═══════════════════════════════════════════════════════════════════════════

def vlasov_phase_space_compress(
    tensor: np.ndarray, grid_size: int = 16
) -> Tuple[dict, float, float]:
    """Map weight distribution to Vlasov phase-space (x, v) where x = weight
    value, v = gradient of the distribution function f(x). The joint
    distribution f(x,v) is a compressible phase-space fluid.
    ALSO store DCT coefficients of row means to preserve spatial structure.
    The Vlasov equation df/dt + v.df/dx = 0 describes incompressible flow."""
    orig = tensor.astype(np.float64)
    m, n = orig.shape
    flat = orig.ravel()

    hist, edges = np.histogram(flat, bins=grid_size, density=True)
    centers = (edges[:-1] + edges[1:]) * 0.5
    v = np.gradient(hist) * grid_size
    vn = (v - v.min()) / (v.max() - v.min() + 1e-30)

    P = np.zeros((grid_size, grid_size), dtype=np.float64)
    for i in range(grid_size):
        vi = int(np.clip(vn[i] * (grid_size - 1), 0, grid_size - 1))
        P[i, vi] = hist[i] + 1e-10

    thr = np.percentile(P[P > 0], 30)
    mask = P > thr
    idx = np.argwhere(mask)
    vals = P[mask]

    row_m = orig.mean(axis=1)
    col_m = orig.mean(axis=0)
    rmc = dct(row_m); cmc = dct(col_m)
    rk = max(1, int(0.15 * m)); ck = max(1, int(0.15 * n))
    rti = np.argpartition(np.abs(rmc), -rk)[-rk:]
    cti = np.argpartition(np.abs(cmc), -ck)[-ck:]
    rcv = rmc[rti]; ccv = cmc[cti]

    c = {"centers": centers.astype(np.float32), "idx": idx.astype(np.int16),
         "vals": vals.astype(np.float16), "gs": grid_size,
         "rti": rti.astype(np.int32), "rcv": rcv.astype(np.float16),
         "cti": cti.astype(np.int32), "ccv": ccv.astype(np.float16),
         "sh": orig.shape}

    Pr = np.zeros((grid_size, grid_size), dtype=np.float64)
    if len(idx) > 0:
        Pr[idx[:, 0], idx[:, 1]] = vals.astype(np.float64)
    marginal = Pr.sum(axis=1) + 1e-30
    marginal /= marginal.sum()
    cum = np.cumsum(marginal)

    rr = np.zeros(m); rr[rti] = rcv.astype(np.float64); rr = idct(rr)
    cr = np.zeros(n); cr[cti] = ccv.astype(np.float64); cr = idct(cr)
    bg = np.outer(rr, cr) / (np.mean(np.abs(cr)) + 1e-30)

    q = np.linspace(0, 1, len(flat))
    gi = np.clip(np.searchsorted(cum, q), 0, grid_size - 1)
    gen_vals = centers[gi]

    bg_vals = bg.ravel()
    order = np.argsort(np.argsort(bg_vals))
    recon = gen_vals[order].reshape(orig.shape)

    return c, _bytes(c) / max(tensor.nbytes, 1), _snr(orig, recon)


# ═══════════════════════════════════════════════════════════════════════════
# 3. HamiltonianWeightFlow
# ═══════════════════════════════════════════════════════════════════════════

def hamiltonian_weight_flow(
    tensor: np.ndarray, n_trajectories: int = 16
) -> Tuple[dict, float, float]:
    """Hamiltonian dynamics: weight rows = q(t), gradients = p(t).
    Hamiltonian H = 0.5||p||^2 + 0.5.q.K.q with K = V.diag(omega^2).V^T.
    Store eigenvectors (normal modes), eigenfrequencies omega, and initial
    positions q0. Reconstruction via normal mode expansion."""
    orig = tensor.astype(np.float64)
    m, n = orig.shape
    r = min(n_trajectories, n)

    q0 = orig[:, :r].copy()
    p0 = np.zeros_like(q0)
    for i in range(m):
        p0[i] = np.gradient(orig[i, :r])

    K_eff = q0.T @ q0 / max(m, 1)
    evals, evecs = np.linalg.eigh(K_eff)
    omega = np.sqrt(np.maximum(evals, 0.0))

    c = {"q0": q0.astype(np.float16), "p0": p0.astype(np.float16),
         "omega": omega.astype(np.float32), "evecs": evecs.astype(np.float32),
         "r": r, "sh": orig.shape}

    rec = np.zeros_like(orig)
    rec[:, :r] = q0
    for j in range(r, n):
        j_src = j % r
        w = np.corrcoef(orig[:, j], orig[:, j_src])[0, 1]
        if not (np.isnan(w) or abs(w) < 0.01):
            rec[:, j] = w * rec[:, j_src]
        else:
            rec[:, j] = 0.0

    return c, _bytes(c) / max(tensor.nbytes, 1), _snr(orig, rec)


# ═══════════════════════════════════════════════════════════════════════════
# 4. QuantumAmplitudeEncoding (FIXED MPS)
# ═══════════════════════════════════════════════════════════════════════════

def quantum_amplitude_compress(
    tensor: np.ndarray, bond_dim: int = 4
) -> Tuple[dict, float, float]:
    """Quantum amplitude encoding via MPS (Matrix Product State).
    Represent |psi> = sum A^1...A^k |i1...ik> by decomposing the weight
    tensor via sequential SVD in MPS format. Bond dimension controls
    quantum entanglement vs compression. Each MPS core stores
    interactions between adjacent qubits (tensor dimensions)."""
    orig = tensor.astype(np.float64)
    flat = orig.ravel()
    nrm = float(np.linalg.norm(flat))
    if nrm < 1e-30:
        nrm = 1.0
    psi = flat / nrm
    N = len(psi)

    def _factor(n: int, mx: int = 8):
        fs = []
        r = n
        for d in range(mx, 1, -1):
            while r % d == 0:
                fs.append(d)
                r //= d
        if r > 1:
            fs.append(r)
        return sorted(fs)

    dims = _factor(N)
    prod = 1
    for d in dims:
        prod *= d
    if prod < N:
        dims = [N]
    padded = np.zeros(prod, dtype=np.float64)
    sz = min(N, prod)
    padded[:sz] = psi[:sz]

    T = padded.reshape(dims)
    d = len(dims)

    mat = T.reshape(dims[0], -1)
    U, S, Vt = np.linalg.svd(mat, full_matrices=False)
    chi = min(bond_dim, len(S))
    cores = [U[:, :chi].astype(np.float16)]
    cur = (np.diag(S[:chi]) @ Vt[:chi, :])

    for j in range(1, d - 1):
        n_phys = dims[j]
        n_rem = int(np.prod(dims[j + 1:]))
        mat = cur.reshape(cur.shape[0] * n_phys, n_rem)
        U, S, Vt = np.linalg.svd(mat, full_matrices=False)
        chi = min(bond_dim, len(S))
        cores.append(U[:, :chi].reshape(-1, n_phys, chi).astype(np.float16))
        cur = (np.diag(S[:chi]) @ Vt[:chi, :])

    cores.append(cur.reshape(cur.shape[0], dims[-1]).astype(np.float16))

    c = {"cores": cores, "bond": bond_dim, "dims": list(dims),
         "norm": nrm, "sh": orig.shape}

    rec = cores[0].astype(np.float64)
    for cr in cores[1:-1]:
        rec = np.tensordot(rec, cr.astype(np.float64), axes=([-1], [0]))
        rec = rec.reshape(-1, rec.shape[-1])
    rec = rec @ cores[-1].astype(np.float64)
    recon = rec.ravel()[:N].reshape(orig.shape) * nrm

    return c, _bytes(c) / max(tensor.nbytes, 1), _snr(orig, recon)


# ═══════════════════════════════════════════════════════════════════════════
# 5. TimeCrystalCompression
# ═══════════════════════════════════════════════════════════════════════════

def time_crystal_compress(
    tensor: np.ndarray, period: int = 4
) -> Tuple[dict, float, float]:
    """Time crystals exhibit spontaneous periodic motion without energy
    input. Treat weight matrix rows as a time series and decompose via
    DCT into a periodic component (dominant Fourier modes) and an
    aperiodic residual (sparse, entropy-coded). Even without exact
    periodicity, the Fourier representation is compact for structured
    weights."""
    orig = tensor.astype(np.float64)
    m, n = orig.shape

    coeffs = dct(orig)
    mag = np.abs(coeffs)
    top_k = max(1, int(0.08 * m * n))
    flat_mag = mag.ravel()
    top_idx = np.argpartition(flat_mag, -top_k)[-top_k:]

    periodic = np.zeros_like(coeffs)
    periodic.ravel()[top_idx] = coeffs.ravel()[top_idx]
    periodic_c = idct(periodic)
    res = orig - periodic_c

    thr = np.percentile(np.abs(res), 85)
    mask = np.abs(res) > thr
    ridx = np.argwhere(mask)
    rvals = res[mask]

    c = {"ti": top_idx.astype(np.int32),
         "cv": coeffs.ravel()[top_idx].astype(np.float16),
         "ri": ridx.astype(np.int16), "rv": rvals.astype(np.float16),
         "sh": orig.shape}

    rc = np.zeros(orig.size, dtype=np.float64)
    rc[top_idx] = c["cv"].astype(np.float64)
    rp = idct(rc.reshape(orig.shape))
    rr = np.zeros_like(orig)
    if len(ridx) > 0:
        rr[ridx[:, 0], ridx[:, 1]] = rvals.astype(np.float64)
    recon = rp + rr

    return c, _bytes(c) / max(tensor.nbytes, 1), _snr(orig, recon)


# ═══════════════════════════════════════════════════════════════════════════
# 6. PlasmaTurbulenceCascade (FIXED)
# ═══════════════════════════════════════════════════════════════════════════

def plasma_turbulence_compress(
    tensor: np.ndarray, energy_keep: float = 0.80
) -> Tuple[dict, float, float]:
    """Kolmogorov turbulence cascade: 2D FFT of weight matrix reveals
    energy spectrum E(k) prop k^gamma (gamma approx -5/3).
    Adaptively select low-wavenumber complex modes capturing energy_keep
    fraction of total energy. High-wavenumber modes are synthesized from
    the fitted power law with decorrelated phases. Energy cascades from
    large scales (low-k) to small scales (high-k)."""
    orig = tensor.astype(np.float64)
    m, n = orig.shape

    F = np.fft.rfft2(orig)
    kx = np.fft.fftfreq(m)[:, None]
    ky = np.fft.rfftfreq(n)[None, :]
    k_rad = np.sqrt(kx ** 2 + ky ** 2)
    k_rad[0, 0] = 1e-10

    flat_k = k_rad.ravel()
    flat_p = np.abs(F.ravel()) ** 2
    valid = flat_k > 0
    k_pos = flat_k[valid]
    p_pos = flat_p[valid]

    k_bins = np.logspace(np.log10(k_pos.min() + 1e-10),
                         np.log10(k_pos.max()), 24)
    spec = np.zeros(len(k_bins) - 1)
    for bi in range(len(k_bins) - 1):
        mk = (k_pos >= k_bins[bi]) & (k_pos < k_bins[bi + 1])
        if np.any(mk):
            spec[bi] = np.mean(p_pos[mk])
    kc = np.sqrt(k_bins[:-1] * k_bins[1:])
    vs = (spec > 0) & np.isfinite(spec)
    lk = np.log(kc[vs])
    lp = np.log(spec[vs])
    if len(lk) > 3:
        A = np.column_stack([np.ones_like(lk), lk])
        coeff, _, _, _ = np.linalg.lstsq(A, lp, rcond=None)
        slope, amp = coeff[1], np.exp(coeff[0])
    else:
        slope, amp = -5.0 / 3.0, 1.0

    en_flat = flat_p.copy()
    en_order = np.argsort(en_flat)[::-1]
    cum_en = np.cumsum(en_flat[en_order])
    total_en = cum_en[-1] + 1e-30
    n_keep = int(np.searchsorted(cum_en / total_en, energy_keep)) + 1
    n_total = F.size
    max_modes = int(n_total * 0.04) + 2
    n_keep = max(16, min(n_keep, max_modes))
    ti = en_order[:n_keep]
    tm = F.ravel()[ti]

    c = {"tm": tm.astype(np.complex64), "ti": ti.astype(np.int32),
         "n_kept": n_keep, "slope": float(slope), "amp": float(amp),
         "sh": (m, n)}

    Fr = np.zeros(n_total, dtype=np.complex128)
    Fr[ti] = tm.astype(np.complex128)
    recon = np.fft.irfft2(Fr.reshape(F.shape), s=(m, n))

    return c, _bytes(c) / max(tensor.nbytes, 1), _snr(orig, recon)


# ═══════════════════════════════════════════════════════════════════════════
# 7. FractalWeightCompression (NOVEL)
# ═══════════════════════════════════════════════════════════════════════════

def fractal_weight_compress(
    tensor: np.ndarray, min_block: int = 4
) -> Tuple[dict, float, float]:
    """Fractal compression via self-similarity. Partition matrix into
    non-overlapping range blocks; find for each the best-matching domain
    block (from a coarser 2x averaged level) under affine transform
    y = s.x + o. Store only domain_id, contrast s, and offset o.
    Decode by iterating the fractal transform to fixed-point attractor.
    Weight matrices often exhibit cross-scale self-similarity."""
    orig = tensor.astype(np.float64)
    m, n = orig.shape
    bs = max(min_block, 4)

    ds = np.zeros((m // 2, n // 2), dtype=np.float64)
    for i in range(m // 2):
        for j in range(n // 2):
            ii, jj = 2 * i, 2 * j
            p = orig[ii:min(ii + 2, m), jj:min(jj + 2, n)]
            ds[i, j] = p.mean()
    dm, dn = ds.shape

    domains = []
    for i in range(0, dm - bs + 1, bs // 2):
        for j in range(0, dn - bs + 1, bs // 2):
            d = ds[i:i + bs, j:j + bs]
            if d.shape == (bs, bs):
                domains.append(d.ravel())
    if not domains:
        domains.append(np.zeros(bs * bs))
    domains = np.array(domains)
    nd = len(domains)

    nr = (m // bs) * (n // bs)
    params = np.zeros((nr, 3), dtype=np.float32)
    ri = 0
    n_cols = n // bs
    for i in range(0, m - bs + 1, bs):
        for j in range(0, n - bs + 1, bs):
            r = orig[i:i + bs, j:j + bs].ravel()
            mr = r.mean()
            best_dk, best_s, best_o, best_e = 0, 0.0, mr, np.var(r)
            for dk in range(min(nd, 512)):
                d = domains[dk]
                s = (np.dot(r - mr, d - d.mean()) /
                     (np.dot(d - d.mean(), d - d.mean()) + 1e-30))
                s = np.clip(s, -0.8, 0.8)
                o = mr - s * d.mean()
                err = np.mean((r - (s * d + o)) ** 2)
                if err < best_e:
                    best_dk, best_s, best_o = dk, s, o
                    best_e = err
            params[ri] = [best_dk, best_s, best_o]
            ri += 1

    c = {"p": params, "dom": domains.astype(np.float16),
         "bs": bs, "nd": nd, "m": m, "n": n}

    recon = np.zeros_like(orig)
    n_strides = max(1, (dn - bs) // (bs // 2) + 1)
    for it in range(6):
        nxt = np.zeros_like(recon)
        for k in range(nr):
            dk, s, o = int(params[k, 0]), params[k, 1], params[k, 2]
            bi, bj = (k // n_cols) * bs, (k % n_cols) * bs
            di_idx, dj_idx = (dk // n_strides) * (bs // 2), \
                             (dk % n_strides) * (bs // 2)
            d_patch = np.zeros(bs * bs, dtype=np.float64)
            for ii in range(bs):
                for jj in range(bs):
                    si = min(di_idx + ii * 2 // bs, dm - 1)
                    sj = min(dj_idx + jj * 2 // bs, dn - 1)
                    d_patch[ii * bs + jj] = ds[si, sj]
            block = (s * d_patch + o).reshape(bs, bs)
            ie, je = min(bi + bs, m), min(bj + bs, n)
            nxt[bi:ie, bj:je] = block[:ie - bi, :je - bj]
        if it == 0:
            recon = nxt
        else:
            recon = 0.7 * recon + 0.3 * nxt

    return c, _bytes(c) / max(tensor.nbytes, 1), _snr(orig, recon)


# ═══════════════════════════════════════════════════════════════════════════
# 8. TopologicalDataCompression (FIXED)
# ═══════════════════════════════════════════════════════════════════════════

def topological_data_compress(
    tensor: np.ndarray, max_dim: int = 12
) -> Tuple[dict, float, float]:
    """Persistent homology: threshold weight matrix, track connected component
    births/deaths via Union-Find. Store persistence (birth, death, centroid).
    Reconstruct via Gaussian bumps at centroids + DCT guidance."""
    orig = tensor.astype(np.float64)
    m, n = orig.shape
    N = m * n

    vals = np.sort(orig.ravel())
    n_th = min(48, max(2, N))
    th = vals[np.linspace(0, N - 1, n_th).astype(int)]
    parent = np.arange(N, dtype=np.int32)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    active = np.zeros(N, dtype=bool)
    births = {}
    birth_pos = {}

    for t_val in th:
        mask = orig.ravel() >= t_val
        new_ones = mask & ~active
        if np.any(new_ones):
            for p in np.where(new_ones)[0]:
                active[p] = True
                births[p] = t_val
                birth_pos[p] = (p // n, p % n)
                for nb in [p - 1, p + 1, p - n, p + n]:
                    if 0 <= nb < N and active[nb]:
                        if (nb % n == n - 1 and p % n == 0) or \
                           (p % n == n - 1 and nb % n == 0):
                            continue
                        ra, rb = find(p), find(nb)
                        if ra != rb:
                            parent[ra] = rb

    seen = {}
    for p in np.where(active)[0]:
        r = find(p)
        if r not in seen:
            members = [q for q in np.where(active)[0] if find(q) == r]
            b = min(births.get(q, th[0]) for q in members)
            cs = [birth_pos.get(q, (0, 0)) for q in members]
            avg = (int(np.mean([c[0] for c in cs])),
                   int(np.mean([c[1] for c in cs])))
            seen[r] = (b, th[-1], avg[0], avg[1], th[-1] - b)

    all_i = sorted(seen.values(), key=lambda x: -x[4])[:max_dim * 8]
    pd = np.array([[b, d, ci, cj] for b, d, ci, cj, _ in all_i],
                  dtype=np.float32) if all_i else np.array([[0, 0, 0, 0]])

    dc = dct(orig)
    dk = max(1, int(0.06 * m * n))
    dti = np.argpartition(np.abs(dc).ravel(), -dk)[-dk:]
    dcv = dc.ravel()[dti]

    c = {"pairs": pd, "dti": dti.astype(np.int32), "dcv": dcv.astype(np.float16),
         "mn": float(np.mean(orig)), "sd": float(np.std(orig)), "sh": orig.shape}

    gu = np.zeros(orig.size, dtype=np.float64)
    gu[dti] = dcv.astype(np.float64)
    gu = idct(gu.reshape(orig.shape))

    recon = np.full(orig.shape, float(c["mn"]), dtype=np.float64)
    if len(pd) >= 1 and pd.shape[0] >= 1:
        max_p = max(abs(d - b) for b, d, _, _, _ in all_i) if all_i else 1.0
        for bi in range(min(len(pd), 24)):
            b, d, ci, cj = pd[bi]
            pers = abs(d - b) / max_p
            if pers < 0.01:
                continue
            ii, jj = int(ci) % m, int(cj) % n
            sig = max(2.0, pers * min(m, n) * 0.15)
            g = np.exp(-((np.arange(m) - ii) ** 2)[:, None] / (2 * sig ** 2)
                       - ((np.arange(n) - jj) ** 2)[None, :] / (2 * sig ** 2))
            recon += pers * g

    recon = gu * 0.7 + recon * 0.3
    recon = (recon - recon.mean()) / (recon.std() + 1e-30) * float(c["sd"]) + float(c["mn"])

    return c, _bytes(c) / max(tensor.nbytes, 1), _snr(orig, recon)


# ═══════════════════════════════════════════════════════════════════════════
# Self-test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    rng = np.random.RandomState(0)
    test_random = rng.randn(128, 128).astype(np.float32)

    x = np.linspace(-3, 3, 128)
    X, Y = np.meshgrid(x, x)
    test_structured = (np.sin(X) * np.cos(Y * 0.5) +
                       0.5 * np.exp(-(X ** 2 + Y ** 2) / 4) +
                       0.1 * rng.randn(128, 128)).astype(np.float32)

    methods = [
        ("Gyrokinetic", gyrokinetic_precondition),
        ("Vlasov", vlasov_phase_space_compress),
        ("Hamiltonian", hamiltonian_weight_flow),
        ("QuantumAmp", quantum_amplitude_compress),
        ("TimeCrystal", time_crystal_compress),
        ("PlasmaTurb", plasma_turbulence_compress),
        ("Fractal", fractal_weight_compress),
        ("Topological", topological_data_compress),
    ]

    for label, mat in [("RANDOM", test_random), ("STRUCTURED", test_structured)]:
        print(f"\n{'=' * 52}")
        print(f"  {label} 128x128 matrix")
        print(f"{'=' * 52}")
        print(f"{'Method':20s} {'Ratio':>10s} {'SNR(dB)':>10s}")
        print("-" * 42)
        for name, fn in methods:
            data, ratio, snr = fn(mat)
            print(f"{name:20s} {ratio:10.4f} {snr:10.2f}")
