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

class MultiStableLorenzCompetition:
    """Competing Lorenz"""

    name = "multistable_lorenz_competition"
    category = "novel_chaotic"

    def compress(self, tensor: np.ndarray, rank: int = 8) -> Tuple[bytes, dict]:
        m, n, k, Uk, Sk, Vk = _svd_core(tensor, rank)
        comp_l = _extract_chaos_param(tensor) * 10.0
        x1, y1, z1 = 1.0, 1.0, 1.0
        x2, y2, z2 = -1.0, -1.0, 1.0
        traj = np.zeros(128)
        for i in range(128):
            dx1 = 10 * (y1 - x1) + comp_l * (x2 - x1)
            dy1 = x1 * (28 - z1) - y1
            dz1 = x1 * y1 - 8 / 3 * z1
            dx2 = 10 * (y2 - x2) + comp_l * (x1 - x2)
            dy2 = x2 * (28 - z2) - y2
            dz2 = x2 * y2 - 8 / 3 * z2
            x1 += 0.01 * dx1
            y1 += 0.01 * dy1
            z1 += 0.01 * dz1
            x2 += 0.01 * dx2
            y2 += 0.01 * dy2
            z2 += 0.01 * dz2
            traj[i] = x1 - x2
        Smod = _modulate_by_attractor(Sk, traj, comp_l)
        return _hdr(m, n, k) + struct.pack("<f", comp_l) + _ser(Uk) + _ser(Smod) + _ser(
            Vk
        ), {"shape": tensor.shape}

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        m, n, k, pos = _rhdr(data)
        pos += 4
        return _recon(data, pos, m, n, k, metadata["shape"])
