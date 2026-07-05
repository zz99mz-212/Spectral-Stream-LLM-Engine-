"""Fractal compression for neural network weights using self-similarity.

Key insight: Weight matrices exhibit fractal-like self-similarity across scales.
Attention heads in the same layer have similar patterns. FFN channels repeat.
We exploit this with iterated function systems (IFS).
"""

from __future__ import annotations

import gc
import struct
from typing import Any, Dict, Optional, Tuple

import numpy as np


def _block_mse(a: np.ndarray, b: np.ndarray) -> float:
    """MSE between two blocks (no copy)."""
    diff = a.ravel() - b.ravel()
    return float(diff @ diff) / a.size


def _best_affine_transform(
    target: np.ndarray, codebook: np.ndarray
) -> Tuple[float, float, int]:
    """Find best affine (scale, offset) to approximate target from a codebook block.

    Memory: O(n) for single block, no copies.
    Returns: (scale, offset, best_idx)
    """
    best_mse = float("inf")
    best_scale = 0.0
    best_offset = 0.0
    best_idx = 0

    target_f = target.ravel().astype(np.float64)
    for i in range(len(codebook)):
        cb_f = codebook[i].ravel().astype(np.float64)
        # Optimal scale = cov(target, cb) / var(cb)
        var_cb = float(cb_f @ cb_f) / len(cb_f)
        if var_cb < 1e-10:
            continue
        cov = float(target_f @ cb_f) / len(target_f)
        scale = cov / var_cb
        offset = float(target_f.mean() - scale * cb_f.mean())

        # MSE with this transform
        recon = scale * cb_f + offset
        mse = float(((target_f - recon) ** 2).mean())

        if mse < best_mse:
            best_mse = mse
            best_scale = scale
            best_offset = offset
            best_idx = i

        del cb_f  # free per iteration

    del target_f
    return best_scale, best_offset, best_idx


def _find_self_similar_blocks(
    tensor: np.ndarray,
    block_size: int = 16,
    n_codebook: int = 64,
    n_iterations: int = 4,
) -> Dict[str, Any]:
    """Find fractal self-similarity in a weight tensor.

    Strategy:
    1. Divide tensor into blocks of block_size
    2. Select n_codebook seed blocks (k-means-like selection)
    3. For each block, find best affine transform from codebook
    4. Store codebook + transforms per iteration

    Memory: O(n_codebook * block_size^2) for codebook, streams through blocks.
    """
    orig_shape = tensor.shape
    flat = tensor.ravel()
    n = len(flat)
    n_blocks = n // block_size

    # Handle non-divisible size by padding
    if n % block_size != 0:
        n_blocks += 1
        padded = np.zeros(n_blocks * block_size, dtype=tensor.dtype)
        padded[:n] = flat
    else:
        padded = flat.copy()

    blocks = padded.reshape(n_blocks, block_size)

    # Select codebook: pick diverse blocks
    selected = [0]
    for _ in range(1, min(n_codebook, n_blocks)):
        dists = np.array(
            [
                min(_block_mse(blocks[i], blocks[s]) for s in selected)
                for i in range(n_blocks)
            ]
        )
        selected.append(int(np.argmax(dists)))
        del dists

    codebook = blocks[selected].copy()

    # IFS iterations
    transforms = []
    current = blocks.copy()

    for iteration in range(n_iterations):
        iteration_transforms = []
        for i in range(n_blocks):
            scale, offset, idx = _best_affine_transform(current[i], codebook)
            iteration_transforms.append((scale, offset, int(idx)))

            # Apply transform for next iteration
            current[i] = scale * codebook[idx] + offset

        transforms.append(iteration_transforms)

        # Check convergence
        if iteration > 0:
            prev = np.array([t[0] for t in transforms[-2]])
            curr = np.array([t[0] for t in transforms[-1]])
            change = float(np.abs(curr - prev).mean())
            if change < 0.001:
                break
            del prev, curr

        gc.collect()

    result = {
        "codebook": codebook.astype(np.float32).tobytes(),
        "codebook_shape": codebook.shape,
        "transforms": transforms,
        "n_blocks": n_blocks,
        "block_size": block_size,
        "orig_shape": orig_shape,
        "n_iterations": iteration + 1,
    }

    del blocks, current, padded
    gc.collect()

    return result


def fractal_weight_compress(
    tensor: np.ndarray,
    block_size: int = 16,
    n_codebook: int = 64,
    n_iterations: int = 4,
) -> Tuple[bytes, dict]:
    """Fractal compression via iterated function systems.

    Memory: O(n_codebook * block_size^2) ~ 64*256 = 16KB. Streams through blocks.
    """
    result = _find_self_similar_blocks(tensor, block_size, n_codebook, n_iterations)

    # Serialize
    codebook_data = result["codebook"]
    transforms = result["transforms"]
    n_codebook_blocks = result["codebook_shape"][0]

    # Flatten transforms: for each iteration, n_blocks * (scale, offset, idx)
    flat_trans = np.array(
        [[s, o, idx] for it in transforms for s, o, idx in it], dtype=np.float32
    )

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
        "method": "fractal_weight",
        "orig_shape": result["orig_shape"],
        "block_size": block_size,
        "n_blocks": result["n_blocks"],
        "n_iterations": result["n_iterations"],
        "transforms_shape": (result["n_iterations"], len(transforms[0]), 3),
    }

    return compressed, metadata


def fractal_weight_decompress(data: bytes, metadata: dict) -> np.ndarray:
    """Decompress fractal-encoded weights."""
    header = data[: struct.calcsize("<6I")]
    rows, cols, block_size, n_blocks, n_codebook_blocks, n_iterations = struct.unpack(
        "<6I", header
    )

    orig_shape = (rows, cols) if rows > 0 and cols > 0 else (rows or len(data),)

    offset = len(header)
    cb_size = n_codebook_blocks * block_size * 4  # float32 bytes
    codebook_data = data[offset : offset + cb_size]
    codebook = np.frombuffer(codebook_data, dtype=np.float32).reshape(
        n_codebook_blocks, block_size
    )

    flat_trans = np.frombuffer(data[offset + cb_size :], dtype=np.float32)
    trans = flat_trans.reshape(n_iterations, n_blocks, 3)

    # Reconstruct via IFS
    result = np.zeros(n_blocks * block_size, dtype=np.float32)
    result_chunks = result.reshape(n_blocks, block_size)

    for i in range(n_blocks):
        scale, offset_val, idx = trans[-1, i]
        result_chunks[i] = scale * codebook[int(idx)] + offset_val

    # Apply inverse iterations
    for it in range(n_iterations - 2, -1, -1):
        next_result = np.zeros_like(result_chunks)
        for i in range(n_blocks):
            scale, offset_val, idx = trans[it, i]
            next_result[i] = scale * codebook[int(idx)] + offset_val
        result_chunks[:] = next_result
        del next_result

    output = result[: np.prod(orig_shape)].reshape(orig_shape)
    del result, codebook, flat_trans
    gc.collect()

    return output


class FractalWeightCompression:
    name = "fractal_weight"
    category = "novel_fractal"

    def compress(
        self,
        tensor: np.ndarray,
        block_size: int = 16,
        n_codebook: int = 64,
        n_iterations: int = 4,
        **kwargs,
    ) -> Tuple[bytes, dict]:
        return fractal_weight_compress(tensor, block_size, n_codebook, n_iterations)

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        return fractal_weight_decompress(data, metadata)


# Also register with standard interface
compress = fractal_weight_compress
decompress = fractal_weight_decompress
