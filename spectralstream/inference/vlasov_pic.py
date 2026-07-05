from __future__ import annotations

import time
import math
import numpy as np
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field

from spectralstream.core.math_primitives import softmax

try:
    from scipy.fft import dct, idct

    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


def _cross_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = a.shape[-1]
    if d >= 3:
        n_groups = d // 3
        rem = d % 3
        out = np.empty_like(a)
        for g in range(n_groups):
            i = g * 3
            a3 = a[..., i : i + 3]
            b3 = b[..., i : i + 3]
            cx = a3[..., 1] * b3[..., 2] - a3[..., 2] * b3[..., 1]
            cy = a3[..., 2] * b3[..., 0] - a3[..., 0] * b3[..., 2]
            cz = a3[..., 0] * b3[..., 1] - a3[..., 1] * b3[..., 0]
            out[..., i : i + 3] = np.stack([cx, cy, cz], axis=-1)
        if rem > 0:
            out[..., -rem:] = a[..., -rem:] * b[..., -rem:].sum(
                axis=-1, keepdims=True
            ) - b[..., -rem:] * a[..., -rem:].sum(axis=-1, keepdims=True)
        return out
    return a * b.sum(axis=-1, keepdims=True) - b * a.sum(axis=-1, keepdims=True)


def _tsc_weights(xi: float) -> Tuple[int, int, int, float, float, float]:
    i_base = int(np.floor(xi))
    u = xi - i_base
    if u < 0.5:
        w0 = 0.5 * (0.5 - u) ** 2
        w1 = 0.75 - u**2
        w2 = 0.5 * (0.5 + u) ** 2
        return i_base - 1, i_base, i_base + 1, w0, w1, w2
    else:
        u = u - 1.0
        w0 = 0.5 * (0.5 - u) ** 2
        w1 = 0.75 - u**2
        w2 = 0.5 * (0.5 + u) ** 2
        return i_base, i_base + 1, i_base + 2, w0, w1, w2


def tsc_deposit(
    positions: np.ndarray,
    field: np.ndarray,
    n_grid: int,
    x_min: float = 0.0,
    x_max: float = 1.0,
) -> np.ndarray:
    n = len(positions)
    nf = field.shape[-1] if field.ndim > 1 else 1
    f = np.asarray(field, dtype=np.float64).reshape(n, nf)
    grid_field = np.zeros((n_grid, nf), dtype=np.float64)

    xi = (positions - x_min) / (x_max - x_min) * n_grid
    xi = np.clip(xi, 0.0, float(n_grid))

    i_base = np.floor(xi).astype(np.int64)
    u = xi - i_base

    mask_l = u < 0.5
    u_l = u[mask_l]
    w0_l = 0.5 * (0.5 - u_l) ** 2
    w1_l = 0.75 - u_l**2
    w2_l = 0.5 * (0.5 + u_l) ** 2
    i0_l = np.clip(i_base[mask_l] - 1, 0, n_grid - 1)
    i1_l = np.clip(i_base[mask_l], 0, n_grid - 1)
    i2_l = np.clip(i_base[mask_l] + 1, 0, n_grid - 1)

    mask_r = ~mask_l
    u_r = u[mask_r] - 1.0
    w0_r = 0.5 * (0.5 - u_r) ** 2
    w1_r = 0.75 - u_r**2
    w2_r = 0.5 * (0.5 + u_r) ** 2
    i0_r = np.clip(i_base[mask_r], 0, n_grid - 1)
    i1_r = np.clip(i_base[mask_r] + 1, 0, n_grid - 1)
    i2_r = np.clip(i_base[mask_r] + 2, 0, n_grid - 1)

    i0 = np.concatenate([i0_l, i0_r])
    i1 = np.concatenate([i1_l, i1_r])
    i2 = np.concatenate([i2_l, i2_r])
    w0 = np.concatenate([w0_l, w0_r])
    w1 = np.concatenate([w1_l, w1_r])
    w2 = np.concatenate([w2_l, w2_r])
    f_all = np.concatenate([f[mask_l], f[mask_r]], axis=0)

    np.add.at(grid_field, i0, (w0[:, None] * f_all))
    np.add.at(grid_field, i1, (w1[:, None] * f_all))
    np.add.at(grid_field, i2, (w2[:, None] * f_all))

    return grid_field


def tsc_interpolate(
    positions: np.ndarray,
    grid_field: np.ndarray,
    x_min: float = 0.0,
    x_max: float = 1.0,
) -> np.ndarray:
    n = len(positions)
    n_grid = grid_field.shape[0]
    nf = grid_field.shape[-1] if grid_field.ndim > 1 else 1
    g = np.asarray(grid_field, dtype=np.float64).reshape(n_grid, nf)

    xi = (positions - x_min) / (x_max - x_min) * n_grid
    xi = np.clip(xi, 0.0, float(n_grid))

    i_base = np.floor(xi).astype(np.int64)
    u = xi - i_base

    interpolated = np.zeros((n, nf), dtype=np.float64)

    idx = np.arange(n)
    mask_l = u < 0.5
    mask_r = ~mask_l

    u_l = u[mask_l]
    w0 = 0.5 * (0.5 - u_l) ** 2
    w1 = 0.75 - u_l**2
    w2 = 0.5 * (0.5 + u_l) ** 2
    i0 = np.clip(i_base[mask_l] - 1, 0, n_grid - 1)
    i1 = np.clip(i_base[mask_l], 0, n_grid - 1)
    i2 = np.clip(i_base[mask_l] + 1, 0, n_grid - 1)
    interpolated[idx[mask_l]] = (
        w0[:, None] * g[i0] + w1[:, None] * g[i1] + w2[:, None] * g[i2]
    )

    u_r = u[mask_r] - 1.0
    w0 = 0.5 * (0.5 - u_r) ** 2
    w1 = 0.75 - u_r**2
    w2 = 0.5 * (0.5 + u_r) ** 2
    i0 = np.clip(i_base[mask_r], 0, n_grid - 1)
    i1 = np.clip(i_base[mask_r] + 1, 0, n_grid - 1)
    i2 = np.clip(i_base[mask_r] + 2, 0, n_grid - 1)
    interpolated[idx[mask_r]] = (
        w0[:, None] * g[i0] + w1[:, None] * g[i1] + w2[:, None] * g[i2]
    )

    return interpolated


@dataclass
class RefinedPatch:
    level: int
    i_start: int
    i_end: int
    x_min: float
    x_max: float


class AMRGrid:
    def __init__(
        self,
        n_base: int = 64,
        refine_threshold: float = 2.0,
        x_min: float = 0.0,
        x_max: float = 1.0,
    ):
        self.n_base = n_base
        self.refine_threshold = refine_threshold
        self.x_min = x_min
        self.x_max = x_max
        self.dx_base = (x_max - x_min) / n_base
        self.patches: List[RefinedPatch] = []
        self._refined_flag = np.zeros(n_base, dtype=bool)

    def build_from_density(self, density: np.ndarray):
        mean_d = np.mean(density)
        threshold = self.refine_threshold * mean_d
        hot = np.where(density.ravel() > threshold)[0]
        self._refined_flag[:] = False
        self._refined_flag[hot] = True

        self.patches = []
        if len(hot) == 0:
            return
        groups = np.split(hot, np.where(np.diff(hot) != 1)[0] + 1)
        for grp in groups:
            i_s = int(grp[0])
            i_e = int(grp[-1]) + 1
            x_s = self.x_min + i_s * self.dx_base
            x_e = self.x_min + i_e * self.dx_base
            self.patches.append(RefinedPatch(1, i_s, i_e, x_s, x_e))

    @property
    def n_refined_cells(self) -> int:
        return sum(2 * (p.i_end - p.i_start) for p in self.patches)

    def deposit(self, positions: np.ndarray, field: np.ndarray) -> np.ndarray:
        n_grid = self.n_base * 2
        nf = field.shape[-1] if field.ndim > 1 else 1
        f = np.asarray(field, dtype=np.float64).reshape(-1, nf)
        composite = np.zeros((n_grid, nf), dtype=np.float64)

        for patch in self.patches:
            mask = (positions >= patch.x_min) & (positions < patch.x_max)
            if not mask.any():
                continue
            pos_sub = positions[mask]
            field_sub = f[mask]
            n_fine = 2 * (patch.i_end - patch.i_start)
            fine_start = 2 * patch.i_start
            fine_end = fine_start + n_fine
            x_s = self.x_min + fine_start * (self.x_max - self.x_min) / n_grid
            xi_sub = (pos_sub - x_s) / (self.x_max - self.x_min) * n_grid
            xi_sub = np.clip(xi_sub, float(fine_start), float(fine_end - 1))
            ib = np.floor(xi_sub).astype(np.int64)
            u = xi_sub - ib
            ml = u < 0.5
            mr = ~ml
            for branch_mask, shift in [(ml, -1), (mr, 0)]:
                if not branch_mask.any():
                    continue
                u_b = u[branch_mask]
                if shift == -1:
                    u_b = u_b
                    w0 = 0.5 * (0.5 - u_b) ** 2
                    w1 = 0.75 - u_b**2
                    w2 = 0.5 * (0.5 + u_b) ** 2
                    idx0 = np.clip(ib[branch_mask] - 1, fine_start, fine_end - 1)
                    idx1 = np.clip(ib[branch_mask], fine_start, fine_end - 1)
                    idx2 = np.clip(ib[branch_mask] + 1, fine_start, fine_end - 1)
                else:
                    u_b = u_b - 1.0
                    w0 = 0.5 * (0.5 - u_b) ** 2
                    w1 = 0.75 - u_b**2
                    w2 = 0.5 * (0.5 + u_b) ** 2
                    idx0 = np.clip(ib[branch_mask], fine_start, fine_end - 1)
                    idx1 = np.clip(ib[branch_mask] + 1, fine_start, fine_end - 1)
                    idx2 = np.clip(ib[branch_mask] + 2, fine_start, fine_end - 1)
                fb = field_sub[branch_mask]
                np.add.at(composite, idx0, w0[:, None] * fb)
                np.add.at(composite, idx1, w1[:, None] * fb)
                np.add.at(composite, idx2, w2[:, None] * fb)

        if len(self.patches) > 0:
            in_patch = np.zeros(len(positions), dtype=bool)
            for patch in self.patches:
                in_patch |= (positions >= patch.x_min) & (positions < patch.x_max)
            base_only = ~in_patch
        else:
            base_only = np.ones(len(positions), dtype=bool)

        if base_only.any():
            xi_base = (
                (positions[base_only] - self.x_min)
                / (self.x_max - self.x_min)
                * self.n_base
            )
            xi_base = np.clip(xi_base, 0.0, float(self.n_base))
            ib = np.floor(xi_base).astype(np.int64)
            u = xi_base - ib
            ml = u < 0.5
            mr = ~ml
            f_b = f[base_only]
            for branch_mask, shift in [(ml, -1), (mr, 0)]:
                if not branch_mask.any():
                    continue
                u_b = u[branch_mask]
                if shift == -1:
                    w0 = 0.5 * (0.5 - u_b) ** 2
                    w1 = 0.75 - u_b**2
                    w2 = 0.5 * (0.5 + u_b) ** 2
                    idx0 = np.clip(2 * (ib[branch_mask] - 1), 0, n_grid - 1)
                    idx1 = np.clip(2 * ib[branch_mask], 0, n_grid - 1)
                    idx2 = np.clip(2 * (ib[branch_mask] + 1), 0, n_grid - 1)
                else:
                    u_b = u_b - 1.0
                    w0 = 0.5 * (0.5 - u_b) ** 2
                    w1 = 0.75 - u_b**2
                    w2 = 0.5 * (0.5 + u_b) ** 2
                    idx0 = np.clip(2 * ib[branch_mask], 0, n_grid - 1)
                    idx1 = np.clip(2 * (ib[branch_mask] + 1), 0, n_grid - 1)
                    idx2 = np.clip(2 * (ib[branch_mask] + 2), 0, n_grid - 1)
                fb = f_b[branch_mask]
                np.add.at(composite, idx0, w0[:, None] * fb)
                np.add.at(composite, idx1, w1[:, None] * fb)
                np.add.at(composite, idx2, w2[:, None] * fb)

        return composite

    def interpolate(self, positions: np.ndarray, grid_field: np.ndarray) -> np.ndarray:
        n_grid = grid_field.shape[0]
        nf = grid_field.shape[-1] if grid_field.ndim > 1 else 1
        g = np.asarray(grid_field, dtype=np.float64).reshape(n_grid, nf)
        n = len(positions)
        result = np.zeros((n, nf), dtype=np.float64)

        xi = (positions - self.x_min) / (self.x_max - self.x_min) * n_grid
        xi = np.clip(xi, 0.0, float(n_grid))
        ib = np.floor(xi).astype(np.int64)
        u = xi - ib

        ml = u < 0.5
        mr = ~ml
        idx = np.arange(n)

        if ml.any():
            u_l = u[ml]
            w0 = 0.5 * (0.5 - u_l) ** 2
            w1 = 0.75 - u_l**2
            w2 = 0.5 * (0.5 + u_l) ** 2
            i0 = np.clip(ib[ml] - 1, 0, n_grid - 1)
            i1 = np.clip(ib[ml], 0, n_grid - 1)
            i2 = np.clip(ib[ml] + 1, 0, n_grid - 1)
            result[idx[ml]] = (
                w0[:, None] * g[i0] + w1[:, None] * g[i1] + w2[:, None] * g[i2]
            )

        if mr.any():
            u_r = u[mr] - 1.0
            w0 = 0.5 * (0.5 - u_r) ** 2
            w1 = 0.75 - u_r**2
            w2 = 0.5 * (0.5 + u_r) ** 2
            i0 = np.clip(ib[mr], 0, n_grid - 1)
            i1 = np.clip(ib[mr] + 1, 0, n_grid - 1)
            i2 = np.clip(ib[mr] + 2, 0, n_grid - 1)
            result[idx[mr]] = (
                w0[:, None] * g[i0] + w1[:, None] * g[i1] + w2[:, None] * g[i2]
            )

        return result


def boris_push(
    vel: np.ndarray,
    E: np.ndarray,
    B_field: np.ndarray,
    dt: float,
    charge: float = 1.0,
    mass: float = 1.0,
) -> np.ndarray:
    qm = charge / mass
    t = qm * B_field * (0.5 * dt)
    tsq = (t * t).sum(axis=-1, keepdims=True) + 1e-30

    v_minus = vel + 0.5 * dt * qm * E
    v_prime = _cross_product(v_minus, t)
    s = 2.0 * t / (1.0 + tsq)
    v_plus = v_minus + _cross_product(v_prime, s)
    v_new = v_plus + 0.5 * dt * qm * E
    return v_new


def monte_carlo_collisions(
    vel: np.ndarray,
    nu: float,
    dt: float,
    rng: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    if rng is None:
        rng = np.random.RandomState()
    n, d = vel.shape
    prob = min(nu * dt, 0.999)
    mask = rng.uniform(size=n) < prob

    if not mask.any():
        return vel

    result = vel.copy()
    n_coll = int(mask.sum())

    if d >= 3:
        n_groups = d // 3
        rem = d % 3
        for idx in np.where(mask)[0]:
            v = vel[idx].copy()
            for g in range(n_groups):
                i = g * 3
                v3 = v[i : i + 3]
                a = rng.randn(3)
                a /= np.linalg.norm(a) + 1e-30
                ca = rng.uniform(-1, 1)
                sa = np.sqrt(max(0.0, 1.0 - ca * ca)) * rng.choice([-1, 1])
                v_rot = v3 * ca + np.cross(v3, a) * sa + a * np.dot(v3, a) * (1.0 - ca)
                result[idx, i : i + 3] = v_rot
            if rem > 0:
                result[idx, -rem:] *= rng.choice([-1.0, 1.0], size=rem)
    else:
        for idx in np.where(mask)[0]:
            result[idx] *= rng.choice([-1.0, 1.0])

    return result


def langevin_thermostat(
    vel: np.ndarray,
    target_temp: float,
    friction: float,
    dt: float,
    rng: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    if rng is None:
        rng = np.random.RandomState()
    n, d = vel.shape

    friction_factor = np.exp(-friction * dt)
    noise_amplitude = np.sqrt(target_temp * (1.0 - friction_factor**2))

    drift = friction_factor * vel
    noise = noise_amplitude * rng.randn(n, d)
    return drift + noise


class FieldDiagnostics:
    def __init__(self):
        self.history: Dict[str, List[float]] = {
            "field_energy": [],
            "particle_energy": [],
            "total_energy": [],
            "field_energy_change": [],
            "charge_conservation": [],
        }
        self.step = 0

    def update(
        self,
        E_grid: np.ndarray,
        dx: float,
        particle_vel: np.ndarray,
        density: np.ndarray,
    ):
        field_energy = 0.5 * np.sum(E_grid**2) * dx
        particle_energy = 0.5 * np.sum(particle_vel**2)
        total_energy = field_energy + particle_energy
        charge = np.sum(density) * dx

        self.history["field_energy"].append(float(field_energy))
        self.history["particle_energy"].append(float(particle_energy))
        self.history["total_energy"].append(float(total_energy))
        self.history["charge_conservation"].append(float(charge))

        if self.step > 0:
            delta_field = field_energy - self.history["field_energy"][-2]
            self.history["field_energy_change"].append(float(delta_field))

        self.step += 1

    @property
    def energy_conservation_error(self) -> float:
        totals = np.array(self.history["total_energy"])
        if len(totals) < 2:
            return 0.0
        mean_tot = np.mean(totals)
        if mean_tot < 1e-30:
            return 0.0
        return float(np.std(totals) / mean_tot)

    def summary(self) -> Dict[str, float]:
        return {
            "energy_conservation_error": self.energy_conservation_error,
            "field_energy": self.history["field_energy"][-1]
            if self.history["field_energy"]
            else 0.0,
            "particle_energy": self.history["particle_energy"][-1]
            if self.history["particle_energy"]
            else 0.0,
            "total_energy": self.history["total_energy"][-1]
            if self.history["total_energy"]
            else 0.0,
            "charge": self.history["charge_conservation"][-1]
            if self.history["charge_conservation"]
            else 0.0,
        }

    def reset(self):
        for k in self.history:
            self.history[k].clear()
        self.step = 0


def identify_fast_particles(vel: np.ndarray, percentile: float = 80.0) -> np.ndarray:
    speed = np.linalg.norm(vel, axis=-1)
    threshold = np.percentile(speed, percentile)
    return speed >= threshold


def sub_cycle_push(
    vel: np.ndarray,
    E: np.ndarray,
    B_field: np.ndarray,
    dt: float,
    fast_mask: np.ndarray,
    n_sub_steps: int = 3,
    charge: float = 1.0,
    mass: float = 1.0,
) -> np.ndarray:
    v_new = vel.copy()
    slow_mask = ~fast_mask

    if slow_mask.any():
        v_new[slow_mask] = boris_push(
            vel[slow_mask],
            E[slow_mask],
            B_field[slow_mask],
            dt,
            charge,
            mass,
        )

    if fast_mask.any():
        sub_dt = dt / n_sub_steps
        v_fast = vel[fast_mask].copy()
        for _ in range(n_sub_steps):
            v_fast = boris_push(
                v_fast,
                E[fast_mask],
                B_field[fast_mask],
                sub_dt,
                charge,
                mass,
            )
        v_new[fast_mask] = v_fast

    return v_new


def filter_current_density(J: np.ndarray, filter_strength: float = 0.3) -> np.ndarray:
    w = np.clip(filter_strength, 0.0, 0.49)
    n_grid = J.shape[0]
    J_filtered = J.copy()

    for c in range(J.shape[1]):
        Jf = J_filtered[:, c]
        prev = np.roll(Jf, 1)
        next_ = np.roll(Jf, -1)
        Jf[:] = w * prev + (1.0 - 2.0 * w) * Jf + w * next_

    return J_filtered


def spectral_filter_current(J: np.ndarray, cutoff_mode: float = 0.5) -> np.ndarray:
    n_grid = J.shape[0]
    n_keep = int(n_grid * cutoff_mode)
    J_fft = np.fft.fft(J, axis=0)
    J_fft[n_keep + 1 :] = 0.0
    return np.fft.ifft(J_fft, axis=0).real


def apply_perfect_conductor_bc(
    grid_field: np.ndarray,
    x_min: float = 0.0,
    x_max: float = 1.0,
) -> np.ndarray:
    g = grid_field.copy()
    g[0] = g[1]
    g[-1] = g[-2]
    return g


def reflect_particles(
    positions: np.ndarray,
    x_min: float = 0.0,
    x_max: float = 1.0,
) -> np.ndarray:
    p = positions.copy()
    below = p < x_min
    above = p >= x_max
    p[below] = 2.0 * x_min - p[below]
    p[above] = 2.0 * x_max - p[above]
    return np.clip(p, x_min, x_max)


def solve_screened_poisson_1d(
    rho: np.ndarray,
    dx: float,
    mu: float = 1.0,
    bc_type: str = "periodic",
) -> np.ndarray:
    N = rho.shape[0]
    d = rho.shape[1] if rho.ndim > 1 else 1
    rho_2d = rho.reshape(N, -1)

    if bc_type == "periodic":
        rho_fft = np.fft.fft(rho_2d, axis=0)
        k = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
        k_sq = k[:, None] ** 2
        denom = k_sq + mu**2 + 1e-30
        denom[0] = 1.0
        phi_fft = 4.0 * np.pi * rho_fft / denom
        phi_fft[0] = 0.0
        phi = np.fft.ifft(phi_fft, axis=0).real
    elif bc_type == "neumann":
        if not _HAS_SCIPY:
            raise ImportError(
                "scipy.fft.dct required for Neumann BCs; falling back to periodic"
            )
        rho_dct = dct(rho_2d, type=2, axis=0, norm="ortho")
        k_dct = np.arange(N, dtype=np.float64)
        k_sq_dct = (k_dct * np.pi / (N * dx))[:, None] ** 2
        denom = k_sq_dct + mu**2 + 1e-30
        denom[0] = 1.0
        phi_dct = 4.0 * np.pi * rho_dct / denom
        phi_dct[0] = 0.0
        phi = idct(phi_dct, type=3, axis=0, norm="ortho")
    else:
        raise ValueError(f"Unknown bc_type: {bc_type}")

    return phi.reshape(rho.shape).astype(np.float64)


class VlasovPICSolverV2:
    def __init__(
        self,
        d_model: int,
        n_grid: int = 64,
        n_grid_coarse: int = 16,
        n_grid_fine: int = 256,
        dt: float = 0.5,
        screening_length: float = 1.0,
        use_amr: bool = True,
        use_multi_scale: bool = True,
        use_boris: bool = True,
        use_quadratic_cic: bool = True,
        use_collisions: bool = True,
        use_thermostat: bool = True,
        use_diagnostics: bool = True,
        use_sub_cycling: bool = True,
        use_current_filtering: bool = True,
        use_bcs: bool = True,
        collision_freq: float = 0.1,
        temp_target: float = 1.0,
        friction: float = 0.1,
        refine_threshold: float = 2.0,
        bc_type: str = "neumann",
    ):
        self.d_model = d_model
        self.n_grid = n_grid
        self.n_grid_coarse = n_grid_coarse
        self.n_grid_fine = n_grid_fine
        self.dt = dt
        self.screening_length = screening_length
        self.bc_type = bc_type

        self.use_amr = use_amr
        self.use_multi_scale = use_multi_scale
        self.use_boris = use_boris
        self.use_quadratic_cic = use_quadratic_cic
        self.use_collisions = use_collisions
        self.use_thermostat = use_thermostat
        self.use_diagnostics = use_diagnostics
        self.use_sub_cycling = use_sub_cycling
        self.use_current_filtering = use_current_filtering
        self.use_bcs = use_bcs

        self.collision_freq = collision_freq
        self.temp_target = temp_target
        self.friction = friction
        self.refine_threshold = refine_threshold

        self.diagnostics = FieldDiagnostics() if use_diagnostics else None
        self._rng = np.random.RandomState(42)

    def _deposit(
        self,
        positions: np.ndarray,
        field: np.ndarray,
        n_grid: int,
        x_min: float = 0.0,
        x_max: float = 1.0,
    ) -> np.ndarray:
        if self.use_quadratic_cic:
            return tsc_deposit(positions, field, n_grid, x_min, x_max)
        dx = (x_max - x_min) / n_grid
        grid_field = np.zeros((n_grid, field.shape[-1]), dtype=np.float64)
        xi = (positions - x_min) / dx
        xi = np.clip(xi, 0.0, n_grid)
        left_idx = np.floor(xi).astype(np.int64)
        left_idx = np.clip(left_idx, 0, n_grid - 1)
        right_idx = np.clip(left_idx + 1, 0, n_grid - 1)
        w_left = 1.0 - (xi - left_idx)
        w_right = xi - left_idx
        np.add.at(grid_field, left_idx, w_left[:, None] * field)
        np.add.at(grid_field, right_idx, w_right[:, None] * field)
        return grid_field

    def _interpolate(
        self,
        positions: np.ndarray,
        grid_field: np.ndarray,
        x_min: float = 0.0,
        x_max: float = 1.0,
    ) -> np.ndarray:
        if self.use_quadratic_cic:
            return tsc_interpolate(positions, grid_field, x_min, x_max)
        n_grid = grid_field.shape[0]
        dx = (x_max - x_min) / n_grid
        xi = (positions - x_min) / dx
        xi = np.clip(xi, 0.0, n_grid)
        left_idx = np.floor(xi).astype(np.int64)
        left_idx = np.clip(left_idx, 0, n_grid - 1)
        right_idx = np.clip(left_idx + 1, 0, n_grid - 1)
        w_left = 1.0 - (xi - left_idx)
        w_right = xi - left_idx
        return (
            w_left[:, None] * grid_field[left_idx]
            + w_right[:, None] * grid_field[right_idx]
        )

    def _multi_scale_solve(
        self,
        positions: np.ndarray,
        k: np.ndarray,
        v: np.ndarray,
        valid: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        d = k.shape[-1]
        n = len(positions)

        scales = [
            ("coarse", self.n_grid_coarse, 0.2),
            ("medium", self.n_grid, 0.5),
            ("fine", self.n_grid_fine, 0.3),
        ]

        E_total = np.zeros((n, d), dtype=np.float64)
        v_total = np.zeros((n, d), dtype=np.float64)

        for name, n_g, weight in scales:
            rho = self._deposit(positions, valid[:, None], n_g)
            rho_k = self._deposit(positions, k * valid[:, None], n_g)
            rho_v = self._deposit(positions, v * valid[:, None], n_g)

            density = np.maximum(rho, 1e-10)
            k_mean = rho_k / density
            v_mean = rho_v / density

            mu = 1.0 / max(self.screening_length, 1e-8)
            dx = 1.0 / n_g
            phi = solve_screened_poisson_1d(k_mean, dx, mu, self.bc_type)

            E_grid = -np.gradient(phi, dx, axis=0)
            if self.use_bcs:
                E_grid = apply_perfect_conductor_bc(E_grid)

            E_part = self._interpolate(positions, E_grid)
            v_part = self._interpolate(positions, v_mean)

            E_total += weight * E_part
            v_total += weight * v_part

            if self.use_diagnostics and name == "medium":
                self.diagnostics.update(E_grid, dx, v, density)

        return E_total, v_total

    def _amr_solve(
        self,
        positions: np.ndarray,
        k: np.ndarray,
        v: np.ndarray,
        valid: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        d = k.shape[-1]
        amr = AMRGrid(self.n_grid, self.refine_threshold)
        rho = self._deposit(positions, valid[:, None], self.n_grid)
        amr.build_from_density(rho)

        rho_k = amr.deposit(positions, k * valid[:, None])
        rho_v = amr.deposit(positions, v * valid[:, None])

        density = np.maximum(rho_k[..., :1], 1e-10)
        k_mean = rho_k / density
        v_mean = rho_v / density

        n_g_eff = rho_k.shape[0]
        mu = 1.0 / max(self.screening_length, 1e-8)
        dx = 1.0 / n_g_eff
        phi = solve_screened_poisson_1d(k_mean, dx, mu, self.bc_type)

        E_grid = -np.gradient(phi, dx, axis=0)
        if self.use_bcs:
            E_grid = apply_perfect_conductor_bc(E_grid)

        E_part = amr.interpolate(positions, E_grid)
        v_part = amr.interpolate(positions, v_mean)

        return E_part, v_part

    def compute_attention(
        self,
        q: np.ndarray,
        k: np.ndarray,
        v: np.ndarray,
        mask: Optional[np.ndarray] = None,
        causal: bool = False,
        return_diagnostics: bool = False,
    ) -> np.ndarray:
        n = q.shape[0]
        d = q.shape[-1]
        positions = np.linspace(0.0, 1.0 - 1.0 / max(n, 2), n)

        if mask is not None:
            valid = mask.astype(np.float64)
        else:
            valid = np.ones(n, dtype=np.float64)

        if causal:
            return self._causal_attention(q, k, v, positions, valid, return_diagnostics)

        if self.use_bcs:
            positions = reflect_particles(positions)

        k_scaled = k * valid[:, None]
        v_scaled = v * valid[:, None]

        if self.use_multi_scale:
            E_at_q, v_field = self._multi_scale_solve(
                positions, k_scaled, v_scaled, valid
            )
        elif self.use_amr:
            E_at_q, v_field = self._amr_solve(positions, k_scaled, v_scaled, valid)
        else:
            rho = self._deposit(positions, valid[:, None], self.n_grid)
            rho_k = self._deposit(positions, k_scaled, self.n_grid)
            rho_v = self._deposit(positions, v_scaled, self.n_grid)

            density = np.maximum(rho, 1e-10)
            k_mean = rho_k / density
            v_mean = rho_v / density

            mu = 1.0 / max(self.screening_length, 1e-8)
            dx = 1.0 / self.n_grid

            if self.use_current_filtering:
                J = rho_k.copy()
                J = spectral_filter_current(J, cutoff_mode=0.7)
                k_mean = J / density

            phi = solve_screened_poisson_1d(k_mean, dx, mu, self.bc_type)
            E_grid = -np.gradient(phi, dx, axis=0)

            if self.use_bcs:
                E_grid = apply_perfect_conductor_bc(E_grid)

            E_at_q = self._interpolate(positions, E_grid)
            v_field = self._interpolate(positions, v_mean)

        q_norm = np.linalg.norm(q, axis=-1)
        E_norm = np.linalg.norm(E_at_q, axis=-1)
        cos_sim = np.sum(q * E_at_q, axis=-1) / (q_norm * E_norm + 1e-30)
        F = np.clip(cos_sim, -1.0, 1.0)

        if self.use_boris:
            if self.use_multi_scale:
                rho_v = self._deposit(positions, v * valid[:, None], self.n_grid)
                rho = self._deposit(positions, valid[:, None], self.n_grid)
                density = np.maximum(rho, 1e-10)
                v_mean_grid = rho_v / density
            else:
                v_mean_grid = None

            if v_mean_grid is not None:
                dx_B = 1.0 / self.n_grid
                B_grid = np.gradient(v_mean_grid, dx_B, axis=0)
                if self.use_bcs:
                    B_grid = apply_perfect_conductor_bc(B_grid)
                B_field = self._interpolate(positions, B_grid)
            else:
                B_field = np.zeros_like(q)

            fast_mask = (
                identify_fast_particles(v, percentile=80.0)
                if self.use_sub_cycling
                else np.zeros(n, dtype=bool)
            )

            if self.use_sub_cycling and fast_mask.any():
                v_new = sub_cycle_push(
                    v,
                    E_at_q,
                    B_field,
                    self.dt,
                    fast_mask,
                    n_sub_steps=3,
                )
            else:
                v_new = boris_push(v, E_at_q, B_field, self.dt)
        else:
            alpha = 1.0 / (1.0 + np.exp(-F / max(self.dt, 1e-8)))
            v_new = alpha[:, None] * v_field + (1.0 - alpha[:, None]) * v

        if self.use_thermostat:
            v_new = langevin_thermostat(
                v_new,
                self.temp_target,
                self.friction,
                self.dt,
                self._rng,
            )

        if self.use_collisions:
            v_new = monte_carlo_collisions(
                v_new,
                self.collision_freq,
                self.dt,
                self._rng,
            )

        output = v_new.astype(q.dtype)

        if return_diagnostics and self.diagnostics is not None:
            return output, self.diagnostics.summary()

        return output

    def _causal_attention(
        self,
        q: np.ndarray,
        k: np.ndarray,
        v: np.ndarray,
        positions: np.ndarray,
        valid: np.ndarray,
        return_diagnostics: bool = False,
    ) -> np.ndarray:
        n = q.shape[0]
        d = q.shape[-1]
        output = np.zeros_like(q)

        rho_acc = np.zeros((self.n_grid, 1), dtype=np.float64)
        k_acc = np.zeros((self.n_grid, d), dtype=np.float64)
        v_acc = np.zeros((self.n_grid, d), dtype=np.float64)

        for i in range(n):
            if valid[i] < 0.5 and i > 0:
                output[i] = output[i - 1]
                continue

            pos_i = positions[i : i + 1]

            if self.use_quadratic_cic:
                rho_contrib = self._deposit(
                    pos_i, np.ones((1, 1), dtype=np.float64), self.n_grid
                )
                k_contrib = self._deposit(pos_i, k[i : i + 1] * valid[i], self.n_grid)
                v_contrib = self._deposit(pos_i, v[i : i + 1] * valid[i], self.n_grid)
            else:
                xi = (positions[i] - 0.0) * self.n_grid
                xi = np.clip(xi, 0.0, self.n_grid)
                left = int(np.floor(xi))
                left = np.clip(left, 0, self.n_grid - 1)
                right = np.clip(left + 1, 0, self.n_grid - 1)
                wl = 1.0 - (xi - left)
                wr = xi - left
                rho_contrib = np.zeros((self.n_grid, 1), dtype=np.float64)
                k_contrib = np.zeros((self.n_grid, d), dtype=np.float64)
                v_contrib = np.zeros((self.n_grid, d), dtype=np.float64)
                np.add.at(rho_contrib, left, wl * valid[i])
                np.add.at(rho_contrib, right, wr * valid[i])
                np.add.at(k_contrib, left, wl * valid[i] * k[i])
                np.add.at(k_contrib, right, wr * valid[i] * k[i])
                np.add.at(v_contrib, left, wl * valid[i] * v[i])
                np.add.at(v_contrib, right, wr * valid[i] * v[i])

            rho_acc += rho_contrib
            k_acc += k_contrib
            v_acc += v_contrib

            density = np.maximum(rho_acc, 1e-10)
            k_mean = k_acc / density
            v_mean = v_acc / density

            mu = 1.0 / max(self.screening_length, 1e-8)
            dx = 1.0 / self.n_grid

            if self.use_current_filtering:
                J = k_mean.copy()
                J = spectral_filter_current(J, cutoff_mode=0.7)
                k_mean = J

            phi = solve_screened_poisson_1d(k_mean, dx, mu, self.bc_type)
            E_grid = -np.gradient(phi, dx, axis=0)

            if self.use_bcs:
                E_grid = apply_perfect_conductor_bc(E_grid)

            E_i = self._interpolate(pos_i, E_grid)[0]
            v_field_i = self._interpolate(pos_i, v_mean)[0]

            q_norm = np.linalg.norm(q[i]) + 1e-30
            E_norm = np.linalg.norm(E_i) + 1e-30
            F_i = float(np.dot(q[i], E_i)) / (q_norm * E_norm)
            F_i = np.clip(F_i, -1.0, 1.0)

            alpha = 1.0 / (1.0 + np.exp(-F_i / max(self.dt, 1e-8)))
            output[i] = alpha * v_field_i + (1.0 - alpha) * v[i]

        return output.astype(q.dtype)


class V2PICAttentionLayer:
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_grid: int = 64,
        **solver_kwargs,
    ):
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.n_grid = n_grid

        rng = np.random.RandomState(42)
        scale = float(np.sqrt(d_model))
        self.w_q = rng.randn(d_model, d_model).astype(np.float32) / scale
        self.w_k = rng.randn(d_model, d_model).astype(np.float32) / scale
        self.w_v = rng.randn(d_model, d_model).astype(np.float32) / scale
        self.w_o = rng.randn(d_model, d_model).astype(np.float32) / scale

        self.solvers = [
            VlasovPICSolverV2(d_model=self.head_dim, n_grid=n_grid, **solver_kwargs)
            for _ in range(n_heads)
        ]

    def forward(
        self,
        x: np.ndarray,
        causal: bool = True,
        mask: Optional[np.ndarray] = None,
        return_diagnostics: bool = False,
    ) -> np.ndarray:
        n = x.shape[0]

        q = x @ self.w_q
        k = x @ self.w_k
        v = x @ self.w_v

        q = q.reshape(n, self.n_heads, self.head_dim)
        k = k.reshape(n, self.n_heads, self.head_dim)
        v = v.reshape(n, self.n_heads, self.head_dim)

        outputs = []
        all_diags = []
        for h in range(self.n_heads):
            if return_diagnostics:
                out_h, diag = self.solvers[h].compute_attention(
                    q[:, h, :],
                    k[:, h, :],
                    v[:, h, :],
                    mask=mask,
                    causal=causal,
                    return_diagnostics=True,
                )
                all_diags.append(diag)
            else:
                out_h = self.solvers[h].compute_attention(
                    q[:, h, :],
                    k[:, h, :],
                    v[:, h, :],
                    mask=mask,
                    causal=causal,
                )
            outputs.append(out_h)

        out = np.concatenate(outputs, axis=-1)
        out = out @ self.w_o

        if return_diagnostics:
            return out, all_diags
        return out

    def prefill(self, prompt: np.ndarray):
        n = prompt.shape[0]
        k = prompt @ self.w_k
        v = prompt @ self.w_v
        k = k.reshape(n, self.n_heads, self.head_dim)
        v = v.reshape(n, self.n_heads, self.head_dim)

        for h in range(self.n_heads):
            solver = self.solvers[h]
            grid = solver.n_grid
            positions = np.linspace(0.0, 1.0 - 1.0 / max(n, 2), n)
            if solver.use_quadratic_cic:
                _ = solver._deposit(positions, k[:, h, :], grid)
                _ = solver._deposit(positions, v[:, h, :], grid)
