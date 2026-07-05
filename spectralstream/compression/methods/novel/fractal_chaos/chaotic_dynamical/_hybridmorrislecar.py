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

class HybridMorrisLecar:
    """Morris-Lecar"""

    name = "hybrid_morris_lecar"
    category = "novel_chaotic"

    def compress(self, tensor: np.ndarray, rank: int = 8) -> Tuple[bytes, dict]:
        m, n, k, Uk, Sk, Vk = _svd_core(tensor, rank)
        I_ml = _extract_chaos_param(tensor) * 2.0
        V_ml = 0.0
        n_ml = 0.0
        traj = np.zeros(256)
        for i in range(256):
            m_inf = 0.5 * (1 + math.tanh((V_ml + 1.2) / 0.18))
            n_inf = 0.5 * (1 + math.tanh((V_ml - 12) / 0.17))
            tau_n = 1.0 + 5.0 / math.cosh((V_ml - 12) / 0.34)
            dv = (
                I_ml
                - 2.0 * (V_ml + 0.7)
                - 4.0 * m_inf * (V_ml - 1.0)
                - 0.5 * n_ml * (V_ml + 0.7)
            )
            dn = (n_inf - n_ml) / tau_n
            V_ml += 0.01 * dv
            n_ml += 0.01 * dn
            traj[i] = V_ml
        Smod = _modulate_by_attractor(Sk, traj, I_ml)
        return _hdr(m, n, k) + struct.pack("<f", I_ml) + _ser(Uk) + _ser(Smod) + _ser(
            Vk
        ), {"shape": tensor.shape}

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        m, n, k, pos = _rhdr(data)
        pos += 4
        return _recon(data, pos, m, n, k, metadata["shape"])
