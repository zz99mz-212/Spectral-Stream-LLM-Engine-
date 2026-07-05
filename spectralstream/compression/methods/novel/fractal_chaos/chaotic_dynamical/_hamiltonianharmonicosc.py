from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _ser(a):
    return a.astype(np.float32).tobytes()


def _svd_core(t, r=8):
    tt = t.astype(np.float64).reshape(t.shape[0], -1)
    m, n = tt.shape
    U, S, Vt = np.linalg.svd(tt, full_matrices=False)
    k = min(r, len(S), m, n)
    return (
        m,
        n,
        k,
        U[:, :k].astype(np.float32),
        S[:k].astype(np.float32),
        Vt[:k, :].astype(np.float32),
    )


def _hdr(m, n, k):
    return struct.pack("<III", m, n, k)


def _rhdr(d, p=0):
    m, n, k = struct.unpack_from("<III", d, p)
    return m, n, k, p + 12


def _recon(d, p, m, n, k, s):
    Uk = np.frombuffer(d[p : p + m * k * 4], dtype=np.float32).reshape(m, k)
    p += m * k * 4
    Sk = np.frombuffer(d[p : p + k * 4], dtype=np.float32)
    p += k * 4
    Vk = np.frombuffer(d[p : p + k * n * 4], dtype=np.float32).reshape(k, n)
    return ((Uk * Sk) @ Vk).astype(np.float32).reshape(s)


def _lorenz(s, r, b, dt, st):
    x = y = z = 1.0
    o = np.zeros(st)
    for i in range(st):
        dx = s * (y - x)
        dy = x * (r - z) - y
        dz = x * y - b * z
        x += dt * dx
        y += dt * dy
        z += dt * dz
        o[i] = x
    return o


def _rossler(a, b, c, dt, st):
    x = y = z = 0.1
    o = np.zeros(st)
    for i in range(st):
        dx = -y - z
        dy = x + a * y
        dz = b + z * (x - c)
        x += dt * dx
        y += dt * dy
        z += dt * dz
        o[i] = x
    return o


def _henon(a, b, st):
    x = y = 0.0
    o = np.zeros(st)
    for i in range(st):
        xn = 1.0 - a * x * x + y
        yn = b * x
        x, y = xn, yn
        o[i] = x
    return o


def _logistic(r, x0, st):
    x = x0
    o = np.zeros(st)
    for i in range(st):
        x = r * x * (1.0 - x)
        o[i] = x
    return o


def _extract_chaos_param(tensor):
    return float(np.abs(tensor.ravel()).mean() % 1.0)


def _modulate_by_attractor(S, attractor, phase=0.0):
    k = min(len(S), len(attractor))
    alpha = attractor[:k]
    mn, mx = alpha.min(), alpha.max()
    alpha = (alpha - mn) / (mx - mn + 1e-30)
    return np.array(S[:k]) * (1.0 + 0.1 * (alpha - 0.5) + 0.05 * math.sin(phase))


class HamiltonianHarmonicOsc:
    """Harmonic osc"""

    name = "hamiltonian_harmonic_osc"
    category = "novel_chaotic"

    def compress(self, tensor: np.ndarray, rank: int = 8) -> Tuple[bytes, dict]:
        m, n, k, Uk, Sk, Vk = _svd_core(tensor, rank)
        fq = _extract_chaos_param(tensor) * 10.0
        q, p = 0.0, 0.0
        traj = np.zeros(256)
        for i in range(256):
            dH_dp = p
            dH_dq = -fq * fq * q
            p += 0.01 * (-dH_dq)
            q += 0.01 * dH_dp
            traj[i] = q
        Smod = _modulate_by_attractor(Sk, traj)
        return _hdr(m, n, k) + struct.pack("<f", fq) + _ser(Uk) + _ser(Smod) + _ser(
            Vk
        ), {"shape": tensor.shape}

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        m, n, k, pos = _rhdr(data)
        pos += 4
        return _recon(data, pos, m, n, k, metadata["shape"])
