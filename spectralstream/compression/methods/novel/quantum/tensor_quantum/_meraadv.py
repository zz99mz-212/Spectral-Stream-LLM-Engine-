from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class MERAAdv:
    name = "mera_adv"
    category = "tensor_quantum"

    def compress(self, tensor: np.ndarray, bond_dim: int = 8) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        orig_shape = t.shape
        if t.ndim == 2:
            u_list = []
            sv_list = []
            cur = t.copy()
            for _ in range(6):
                if min(cur.shape) <= bond_dim:
                    break
                U, S, Vt = np.linalg.svd(cur, full_matrices=False)
                chi = min(bond_dim, len(S))
                ut = U[:, :chi].copy()
                u_list.append(ut)
                sv_list.append((S[:chi, None] * Vt[:chi, :]).copy())
                cur = ut @ sv_list[-1]
                pm = max(bond_dim, cur.shape[0] // 2)
                pn = max(bond_dim, cur.shape[1] // 2)
                cur = cur[:pm, :pn]
            if not u_list:
                meta = dict(
                    shape=orig_shape,
                    n_layers=1,
                    layer_shapes=[t.shape],
                    bond_dim=bond_dim,
                )
                data = struct.pack("<i", 1) + _serialize(t.astype(np.float32))
                return data, meta
            layers = []
            for ut_i, svt_i in zip(u_list, sv_list):
                layers.append(ut_i)
                layers.append(svt_i)
            meta = dict(
                shape=orig_shape,
                n_layers=len(layers),
                layer_shapes=[l.shape for l in layers],
                bond_dim=bond_dim,
            )
            data = struct.pack("<i", len(layers))
            for lay in layers:
                data += _serialize(lay.astype(np.float32))
            return data, meta
        meta = dict(shape=orig_shape, n_layers=0, layer_shapes=[], bond_dim=bond_dim)
        data = struct.pack("<i", 0) + _serialize(t.ravel().astype(np.float32))
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        n_layers = metadata["n_layers"]
        if n_layers <= 1:
            flat = _deserialize(data[4:])
            return flat[: int(np.prod(shape))].reshape(shape).astype(np.float32)
        pos = 4
        layers = []
        for ls in metadata["layer_shapes"]:
            sz = int(np.prod(ls))
            layers.append(_deserialize(data[pos : pos + sz * 4]).reshape(ls))
            pos += sz * 4
        n_pairs = n_layers // 2
        recon = layers[-1].astype(np.float64)
        for i in range(n_pairs - 1, -1, -1):
            ut = layers[2 * i].astype(np.float64)
            svt = layers[2 * i + 1].astype(np.float64)
            if i < n_pairs - 1:
                target_n = ut.shape[0]
                if recon.shape[0] < target_n or recon.shape[1] < target_n:
                    tmp = np.zeros((target_n, target_n), dtype=np.float64)
                    h = min(recon.shape[0], target_n)
                    w = min(recon.shape[1], target_n)
                    tmp[:h, :w] = recon[:h, :w]
                    recon = tmp
            recon = ut @ svt
        return recon[: shape[0], : shape[1]].astype(np.float32)
