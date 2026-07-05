from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class PersistentHomologyRank:
    """C1. PERSISTENT-HOMOLOGY-RANK: keep persistence pairs above threshold."""

    name = "persistent_homology_rank"
    category = "novel_topological"

    def compress(
        self, tensor: np.ndarray, epsilon: float = 0.1, max_features: int = 64
    ) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        m, n = t.shape
        flat = t.ravel()
        N = len(flat)

        idx_sorted = np.argsort(flat)
        parent = np.arange(N, dtype=np.int32)
        rank = np.zeros(N, dtype=np.int32)

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int) -> None:
            rx, ry = find(x), find(y)
            if rx == ry:
                return
            if rank[rx] < rank[ry]:
                parent[rx] = ry
            elif rank[rx] > rank[ry]:
                parent[ry] = rx
            else:
                parent[ry] = rx
                rank[rx] += 1

        active = np.zeros(N, dtype=bool)
        births: dict = {}
        deaths: dict = {}
        value_range = float(flat[-1] - flat[0]) if N > 1 else 1.0
        persistence_pairs = []

        for idx in idx_sorted:
            val = flat[idx]
            active[idx] = True
            if idx not in births:
                births[idx] = val
            neighbors = []
            if idx > 0 and active[idx - 1] and ((idx % n) != 0):
                neighbors.append(idx - 1)
            if idx < N - 1 and active[idx + 1] and ((idx + 1) % n != 0):
                neighbors.append(idx + 1)
            if idx >= n and active[idx - n]:
                neighbors.append(idx - n)
            if idx < N - n and active[idx + n]:
                neighbors.append(idx + n)

            for nb in neighbors:
                if find(idx) != find(nb):
                    root_idx = find(idx)
                    root_nb = find(nb)
                    death_val = val
                    if births.get(root_idx, val) <= births.get(root_nb, val):
                        dying = root_nb
                        surviving = root_idx
                    else:
                        dying = root_idx
                        surviving = root_nb
                    pers = death_val - births.get(dying, val)
                    if pers > epsilon * value_range:
                        persistence_pairs.append(
                            (births.get(dying, val), death_val, pers)
                        )
                    union(idx, nb)

        persistence_pairs.sort(key=lambda x: -x[2])
        persistence_pairs = persistence_pairs[:max_features]

        if not persistence_pairs:
            persistence_pairs = [(float(np.mean(flat)), float(np.mean(flat)), 0.0)]

        pairs_arr = np.array(persistence_pairs, dtype=np.float32)

        bg_mean = float(np.mean(t))
        bg_std = float(np.std(t))

        meta = dict(
            shape=t.shape,
            bg_mean=bg_mean,
            bg_std=bg_std,
            n_pairs=len(persistence_pairs),
        )
        data = _serialize(pairs_arr)
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        bg_mean = metadata["bg_mean"]
        bg_std = metadata["bg_std"]
        n_pairs = metadata["n_pairs"]
        pairs = (
            _deserialize(data).reshape(-1, 3) if n_pairs > 0 else np.array([[0, 0, 0]])
        )
        pairs = pairs[:n_pairs]

        m, n = shape
        recon = np.full(shape, bg_mean, dtype=np.float64)

        max_pers = max(np.abs(p[2]) for p in pairs) if len(pairs) > 0 else 1.0
        if max_pers < 1e-10:
            max_pers = 1.0

        for b, d, p in pairs:
            pers = abs(p) / max_pers
            cx, cy = m // 2, n // 2
            sig = max(2.0, pers * min(m, n) * 0.15)
            xs = np.arange(m)
            ys = np.arange(n)
            g = np.exp(
                -((xs - cx) ** 2)[:, None] / (2 * sig**2)
                - ((ys - cy) ** 2)[None, :] / (2 * sig**2)
            )
            recon += pers * g * (d - b)

        recon = (recon - np.mean(recon)) / (np.std(recon) + 1e-30) * bg_std + bg_mean
        return recon.astype(np.float32)
