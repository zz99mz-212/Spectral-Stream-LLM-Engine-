from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spectralstream.core.math_primitives import dct, idct, fwht, zigzag_indices, next_power_of_two

from ._compressionmethod import CompressionMethod, _ensure_2d, _restore_shape, _safe_bytes

class TimeCrystalPhase(CompressionMethod):
    """TimeCrystal alternating-phase compression."""
    name = "timecrystal"; category = "novel"

    def compress(self, tensor, n_phases=4, **kw):
        t, orig = _ensure_2d(tensor)
        m, n = t.shape
        rng = np.random.RandomState(42)
        phases = np.linspace(0, 2*np.pi, n_phases, endpoint=False)
        phase_data = []
        for phi in phases:
            rot = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
            if m >= 2 and n >= 2:
                for i in range(0, m, 2):
                    for j in range(0, n, 2):
                        block = t[i:i+2, j:j+2].astype(np.float64)
                        phase_data.append((rot @ block @ rot.T).astype(np.float32))
        return {"phases": phases.astype(np.float32), "data": phase_data,
                "n_phases": n_phases, "shape": t.shape}, {"orig_shape": orig}

    def decompress(self, cd, meta):
        m = meta["orig_shape"][0] if len(meta["orig_shape"]) >= 2 else 1
        n = meta["orig_shape"][-1]
        result = np.zeros((m, n), dtype=np.float64)
        idx = 0
        for phi in cd["phases"]:
            rot = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
            for i in range(0, m, 2):
                for j in range(0, n, 2):
                    if idx < len(cd["data"]):
                        block = cd["data"][idx].astype(np.float64)
                        restored = rot.T @ block @ rot
                        bi, bj = min(2, m-i), min(2, n-j)
                        result[i:i+bi, j:j+bj] += restored[:bi, :bj]
                        idx += 1
        result /= max(len(cd["phases"]), 1)
        return _restore_shape(result.astype(np.float32), meta["orig_shape"])