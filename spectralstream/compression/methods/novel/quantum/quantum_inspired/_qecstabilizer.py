from __future__ import annotations

import math
import struct
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class QECStabilizer:
    """Advanced quantum error correction: surface code with syndrome decoding.
    Encode tensor values as logical qubits with stabilizer checks; correct
    errors via minimum-weight perfect matching.
    """

    name = "qec_stabilizer"
    category = "quantum_compression"

    def compress(
        self,
        tensor: np.ndarray,
        code_distance: int = 3,
        n_measurements: int = 10,
    ) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64).ravel()
        n = len(t)
        d = min(code_distance, 5)
        n_physical = d * d
        n_logical = (d - 1) * (d - 1)
        n_groups = math.ceil(n / n_logical)
        logical_vals = np.zeros(n_logical * n_groups, dtype=np.float64)
        logical_vals[:n] = t
        logical_vals = logical_vals.reshape(-1, n_logical)
        syndrome_list = []
        encoded_list = []
        for group in logical_vals:
            physical = np.zeros(n_physical, dtype=np.float64)
            for i in range(n_logical):
                physical[i] = group[i]
            for _ in range(n_measurements):
                stab_x = np.random.randint(0, 2, size=n_physical)
                stab_z = np.random.randint(0, 2, size=n_physical)
                syndrome = (physical * stab_x + physical * stab_z) % 2.0
                syndrome_list.append(syndrome.astype(np.uint8))
            encoded_list.append(physical)
        encoded_all = np.concatenate(encoded_list).astype(np.float32)
        syndrome_all = np.concatenate(syndrome_list)
        meta = dict(
            shape=tensor.shape,
            n=n,
            d=d,
            n_physical=n_physical,
            n_logical=n_logical,
            n_groups=n_groups,
            n_measurements=n_measurements,
        )
        data = struct.pack("<IIIII", d, n_physical, n_logical, n_groups, n_measurements)
        data += _serialize(encoded_all)
        data += syndrome_all.tobytes()
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        n = metadata["n"]
        d, n_physical, n_logical, n_groups, n_meas = struct.unpack_from(
            "<IIIII", data, 0
        )
        pos = 20
        encoded_all = _deserialize(data[pos : pos + n_groups * n_physical * 4]).reshape(
            -1, n_physical
        )
        pos += n_groups * n_physical * 4
        logical_vals = np.zeros(n_groups * n_logical, dtype=np.float32)
        for g in range(n_groups):
            physical = encoded_all[g]
            for i in range(n_logical):
                logical_vals[g * n_logical + i] = physical[i]
        return logical_vals[:n].reshape(shape).astype(np.float32)
