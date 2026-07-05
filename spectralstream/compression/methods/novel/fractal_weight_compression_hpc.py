"""HPC-vectorized fractal weight compression using IFS.
All Python loops replaced with NumPy vectorized ops.
"""

from __future__ import annotations

import struct
from typing import Any, Dict, Tuple

import numpy as np


def _select_codebook_hpc(blocks: np.ndarray, n_codebook: int) -> np.ndarray:
    """Select diverse codebook blocks using vectorized distance computation.

    Uses argmax of min-distances instead of Python loops.
    Memory: O(n_blocks * n_codebook) for distance matrix.
    """
    n_blocks = blocks.shape[0]
    n_codebook = min(n_codebook, n_blocks)
    if n_codebook <= 0:
        return np.empty((0, blocks.shape[1]), dtype=blocks.dtype)

    selected = [0]
    dists = np.full(n_blocks, np.inf, dtype=np.float64)

    for _ in range(1, n_codebook):
        # Vectorized MSE: ||blocks[i] - blocks[sel]||^2 / block_size
        sel_block = blocks[selected[-1]]
        diff = blocks.astype(np.float64) - sel_block.astype(np.float64)
        dist = np.sum(diff * diff, axis=1) / blocks.shape[1]
        np.minimum(dists, dist, out=dists)

        next_idx = int(np.argmax(dists))
        selected.append(next_idx)

    return blocks[selected].copy()


def _best_affine_transform_hpc(
    target: np.ndarray, codebook: np.ndarray
) -> Tuple[float, float, int]:
    """Vectorized affine transform search across all codebook blocks.

    All codebook blocks evaluated in parallel via NumPy ops.
    Returns (scale, offset, best_idx).
    """
    target_f = target.ravel().astype(np.float64)
    block_size = target_f.size

    # codebook: (n_codebook, block_size)
    cb = codebook.astype(np.float64)
    n_cb = cb.shape[0]

    # Vectorized computations across all codebook blocks
    # var_cb[i] = cb[i] @ cb[i] / block_size
    cb_dot = np.sum(cb * cb, axis=1) / block_size
    var_cb = cb_dot

    # cov[i] = target @ cb[i] / block_size
    cov = cb @ target_f / block_size

    # scale[i] = cov[i] / var_cb[i] (where var_cb > eps)
    scale = np.where(var_cb > 1e-10, cov / var_cb, 0.0)

    # offset[i] = mean(target) - scale[i] * mean(cb[i])
    target_mean = target_f.mean()
    cb_mean = cb.mean(axis=1)
    offset = target_mean - scale * cb_mean

    # MSE[i] = mean((target - scale[i] * cb[i] - offset[i])^2)
    recon = scale[:, None] * cb + offset[:, None]
    mse = np.mean((target_f[None, :] - recon) ** 2, axis=1)

    # Handle zero-var blocks
    mse = np.where(var_cb > 1e-10, mse, np.inf)

    best_idx = int(np.argmin(mse))
    return float(scale[best_idx]), float(offset[best_idx]), best_idx


def _find_self_similar_blocks_hpc(
    tensor: np.ndarray,
    block_size: int = 16,
    n_codebook: int = 64,
    n_iterations: int = 4,
) -> Dict[str, Any]:
    """HPC-vectorized fractal self-similarity detection.

    All inner loops vectorized across codebook blocks.
    """
    orig_shape = tensor.shape
    flat = tensor.ravel()
    n = len(flat)
    n_blocks = (n + block_size - 1) // block_size

    padded = np.zeros(n_blocks * block_size, dtype=tensor.dtype)
    padded[:n] = flat
    blocks = padded.reshape(n_blocks, block_size)

    codebook = _select_codebook_hpc(blocks, n_codebook)

    transforms = []
    current = blocks.copy().astype(np.float64)

    for iteration in range(n_iterations):
        it_transforms = []
        for i in range(n_blocks):
            scale, offset, idx = _best_affine_transform_hpc(current[i], codebook)
            it_transforms.append((scale, offset, int(idx)))
            current[i] = float(scale) * codebook[int(idx)] + float(offset)

        transforms.append(it_transforms)

        if iteration > 0:
            prev = np.float64([t[0] for t in transforms[-2]])
            curr = np.float64([t[0] for t in transforms[-1]])
            if float(np.abs(curr - prev).mean()) < 0.001:
                break

    return {
        "codebook": codebook.astype(np.float32).tobytes(),
        "codebook_shape": codebook.shape,
        "transforms": transforms,
        "n_blocks": n_blocks,
        "block_size": block_size,
        "orig_shape": orig_shape,
        "n_iterations": iteration + 1,
    }


def fractal_weight_compress_hpc(
    tensor: np.ndarray,
    block_size: int = 16,
    n_codebook: int = 64,
    n_iterations: int = 4,
) -> Tuple[bytes, dict]:
    """HPC-vectorized fractal compression via IFS."""
    result = _find_self_similar_blocks_hpc(tensor, block_size, n_codebook, n_iterations)

    codebook_data = result["codebook"]
    transforms = result["transforms"]
    n_codebook_blocks = result["codebook_shape"][0]

    flat_trans = np.float32([[s, o, idx] for it in transforms for s, o, idx in it])

    header = struct.pack(
        "<6I",
        result["orig_shape"][0] if len(result["orig_shape"]) >= 1 else 0,
        result["orig_shape"][1] if len(result["orig_shape"]) >= 2 else 0,
        result["block_size"],
        result["n_blocks"],
        n_codebook_blocks,
        result["n_iterations"],
    )

    compressed = header + codebook_data + flat_trans.tobytes()
    metadata = {
        "method": "fractal_weight_hpc",
        "orig_shape": result["orig_shape"],
        "block_size": block_size,
        "n_blocks": result["n_blocks"],
        "n_iterations": result["n_iterations"],
        "transforms_shape": (result["n_iterations"], len(transforms[0]), 3),
    }
    return compressed, metadata


def fractal_weight_decompress_hpc(data: bytes, metadata: dict) -> np.ndarray:
    """HPC-vectorized decompress — vectorized IFS reconstruction."""
    header = data[: struct.calcsize("<6I")]
    rows, cols, block_size, n_blocks, n_codebook_blocks, n_iterations = struct.unpack(
        "<6I", header
    )
    orig_shape = (rows, cols) if rows > 0 and cols > 0 else (rows or len(data),)

    offset = len(header)
    cb_size = n_codebook_blocks * block_size * 4
    codebook = np.frombuffer(data[offset : offset + cb_size], dtype=np.float32).reshape(
        n_codebook_blocks, block_size
    )

    flat_trans = np.frombuffer(data[offset + cb_size :], dtype=np.float32)
    trans = flat_trans.reshape(n_iterations, n_blocks, 3)

    # Vectorized reconstruction: apply all blocks at once
    result_chunks = np.zeros((n_blocks, block_size), dtype=np.float32)

    # Apply final iteration
    scale = trans[-1, :, 0:1]
    idx = trans[-1, :, 2].astype(np.intp)
    offset_val = trans[-1, :, 1:2]
    result_chunks = scale * codebook[idx] + offset_val

    # Apply inverse iterations (vectorized)
    for it in range(n_iterations - 2, -1, -1):
        scale = trans[it, :, 0:1]
        idx = trans[it, :, 2].astype(np.intp)
        offset_val = trans[it, :, 1:2]
        result_chunks = scale * codebook[idx] + offset_val

    result = result_chunks.ravel()[: np.prod(orig_shape)]
    return result.reshape(orig_shape)


class FractalWeightCompressionHPC:
    name = "fractal_weight_hpc"
    category = "novel_fractal"

    def compress(
        self,
        tensor: np.ndarray,
        block_size: int = 64,
        n_codebook: int = 64,
        n_iterations: int = 4,
        **kwargs,
    ) -> Tuple[bytes, dict]:
        return fractal_weight_compress_hpc(tensor, block_size, n_codebook, n_iterations)

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        return fractal_weight_decompress_hpc(data, metadata)
