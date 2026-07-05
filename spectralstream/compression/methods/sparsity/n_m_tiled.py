"""
N:M Tiled Sparsity
====================
Tiled implementation of N:M sparsity with cache-friendly block access
patterns for efficient hardware utilization on large tensors.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class NMTiled:
    """Tiled N:M sparsity with cache-friendly block processing."""

    name = "n_m_tiled"
    category = "structural"

    def __init__(self, n: int = 2, m: int = 4, tile_size: int = 256):
        self.n = n
        self.m = m
        self.tile_size = tile_size

    def compress(
        self, tensor: np.ndarray, n: int | None = None, m: int | None = None, **kwargs
    ) -> Tuple[bytes, Dict[str, Any]]:
        n_val = n if n is not None else self.n
        m_val = m if m is not None else self.m
        orig_shape = tensor.shape
        flat = tensor.astype(np.float32).ravel()
        n_total = len(flat)
        pad = -n_total % m_val
        if pad:
            flat = np.pad(flat, (0, pad), mode="constant")
        n_padded = len(flat)

        n_groups = n_padded // m_val
        tile_size = min(self.tile_size, n_groups)
        n_tiles = (n_groups + tile_size - 1) // tile_size

        all_values = []
        mask_accum = np.zeros(n_padded, dtype=bool)

        for t in range(n_tiles):
            start = t * tile_size
            end = min(start + tile_size, n_groups)
            tile_len = (end - start) * m_val
            tile_offset = start * m_val

            tile = flat[tile_offset : tile_offset + tile_len].reshape(
                end - start, m_val
            )
            sort_idx = np.argpartition(-np.abs(tile), n_val - 1, axis=1)
            tile_mask = np.zeros(tile_len, dtype=bool)
            row_offsets = np.repeat(np.arange(end - start), n_val)
            col_pos = sort_idx[:, :n_val].ravel()
            tile_mask[row_offsets * m_val + col_pos] = True

            mask_accum[tile_offset : tile_offset + tile_len] = tile_mask
            all_values.append(tile.ravel()[tile_mask])

        values = (
            np.concatenate(all_values).astype(np.float32)
            if all_values
            else np.array([], dtype=np.float32)
        )
        mask = mask_accum[:n_total]

        if np.sum(mask) < 1:
            mask[:1] = True
            values = flat[:1].astype(np.float32)

        mask_packed = np.packbits(mask)
        meta: Dict[str, Any] = {
            "shape": orig_shape,
            "n": n_val,
            "m": m_val,
            "n_kept": int(np.sum(mask)),
        }
        data = mask_packed.tobytes() + values.tobytes()
        return data, meta

    def decompress(self, data: bytes, metadata: Dict[str, Any]) -> np.ndarray:
        shape = metadata["shape"]
        n_kept = metadata["n_kept"]
        n = int(np.prod(shape))
        mask_bytes = (n + 7) // 8
        mask_packed = np.frombuffer(data[:mask_bytes], dtype=np.uint8)
        mask = np.unpackbits(mask_packed)[:n].astype(bool)
        kept = np.frombuffer(
            data[mask_bytes : mask_bytes + n_kept * 4], dtype=np.float32
        )
        recon = np.zeros(n, dtype=np.float32)
        recon[mask] = kept[: np.sum(mask)]
        return recon.reshape(shape).astype(np.float32)

    def estimate_ratio(self, tensor: np.ndarray, **kwargs) -> float:
        n_val = kwargs.get("n", self.n)
        m_val = kwargs.get("m", self.m)
        sparsity = 1.0 - n_val / m_val
        orig = tensor.nbytes
        n_kept = int(tensor.size * (1.0 - sparsity))
        mask_bytes = (tensor.size + 7) // 8
        comp = n_kept * 4 + mask_bytes
        return comp / max(orig, 1)

    def estimate_error(self, tensor: np.ndarray, **kwargs) -> Dict[str, float]:
        data, meta = self.compress(tensor, **kwargs)
        recon = self.decompress(data, meta)
        orig = tensor.astype(np.float64).ravel()
        rec = recon.astype(np.float64).ravel()
        mse = float(np.mean((orig - rec) ** 2))
        snr = 10 * np.log10(np.sum(orig**2) / (np.sum((orig - rec) ** 2) + 1e-30))
        cos_sim = float(
            np.dot(orig, rec) / (np.linalg.norm(orig) * np.linalg.norm(rec) + 1e-30)
        )
        return {
            "mse": mse,
            "snr_db": float(snr),
            "cosine_similarity": cos_sim,
            "rel_error": float(np.mean(np.abs(orig - rec) / (np.abs(orig) + 1e-10))),
        }
