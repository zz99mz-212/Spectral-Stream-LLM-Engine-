from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spectralstream.core.math_primitives import dct, idct, fwht, zigzag_indices, next_power_of_two

from ._compressionmethod import CompressionMethod, _ensure_2d, _restore_shape, _safe_bytes

class HierarchicalClusteredPQ(CompressionMethod):
    """Hierarchical Clustered Product Quantization."""
    name = "hierarchical_pq"; category = "quantization"

    def compress(self, tensor, n_clusters=8, n_sub=4, n_bits=4, **kw):
        t, orig = _ensure_2d(tensor)
        m, n = t.shape
        rng = np.random.RandomState(42)
        rc = t[rng.choice(m, min(n_clusters, m), replace=False)].copy()
        for _ in range(10):
            d = np.linalg.norm(t[:,None,:] - rc[None,:,:], axis=2)
            a = np.argmin(d, axis=1)
            for c in range(n_clusters):
                mask = a == c
                if np.any(mask): rc[c] = t[mask].mean(axis=0)
        d = np.linalg.norm(t[:,None,:] - rc[None,:,:], axis=2)
        a = np.argmin(d, axis=1)
        sub_dim = max(1, n // n_sub)
        nl = 1 << n_bits
        c_cbs, c_codes = {}, {}
        for c in range(n_clusters):
            mask = a == c
            if not np.any(mask): continue
            cd = t[mask]
            cbs, codes = [], []
            for s in range(n_sub):
                sd = cd[:, s*sub_dim:(s+1)*sub_dim]
                cb = sd[rng.choice(len(sd), min(nl, len(sd)), replace=False)].copy()
                for _ in range(5):
                    dd = np.linalg.norm(sd[:,None,:] - cb[None,:,:], axis=2)
                    aa = np.argmin(dd, axis=1)
                    for k in range(nl):
                        mm = aa == k
                        if np.any(mm): cb[k] = sd[mm].mean(axis=0)
                dd = np.linalg.norm(sd[:,None,:] - cb[None,:,:], axis=2)
                codes.append(np.argmin(dd, axis=1).astype(np.uint8))
                cbs.append(cb.astype(np.float32))
            c_cbs[c] = cbs; c_codes[c] = codes
        return {"rc": rc.astype(np.float32), "a": a.astype(np.uint8), "cbs": c_cbs,
                "codes": c_codes, "sub_dim": sub_dim, "n_sub": n_sub, "shape": t.shape}, {"orig_shape": orig}

    def decompress(self, cd, meta):
        m, n = meta["orig_shape"][:2]
        result = np.zeros((m, n), dtype=np.float32)
        for c in range(len(cd["rc"])):
            mask = cd["a"] == c
            if not np.any(mask) or c not in cd["cbs"]: continue
            indices = np.where(mask)[0]
            cbs, codes = cd["cbs"][c], cd["codes"][c]
            for i, ri in enumerate(indices):
                row = np.zeros(n, dtype=np.float32)
                for s in range(cd["n_sub"]):
                    st = s * cd["sub_dim"]
                    if s < len(cbs) and i < len(codes[s]):
                        row[st:st+cd["sub_dim"]] = cbs[s][codes[s][i]][:cd["sub_dim"]]
                result[ri] = row
        return _restore_shape(result, meta["orig_shape"])