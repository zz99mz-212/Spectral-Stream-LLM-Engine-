#!/usr/bin/env python3
"""
Per-stage diagnostic for the 5-stage cascade pipeline.

Loads a real q_proj weight from Gemma-4-E2B, runs each stage individually,
collects metrics after each stage, and saves results to stage_diagnosis.json.
"""

from __future__ import annotations

import json
import math
import struct
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spectralstream.compression.honest_metrics import (
    ErrorMetrics,
    end_to_end_error,
    serialized_nbytes,
    ratio_vs_fp32,
    ratio_vs_bf16,
)
from spectralstream.compression._dtype_utils import (
    convert_to_storage,
    convert_from_storage,
    detect_storage_dtype,
)
from spectralstream.compression.methods.functional.ergodic_hyperfunction import (
    ErgodicHyperfunction,
)

# ── helpers from cascade_5stage.py ──────────────────────────────────────────


def _closest_divisor(n: int, target: int) -> int:
    target = max(1, min(target, n))
    for step in range(n):
        for d in (target - step, target + step):
            if 1 <= d <= n and n % d == 0:
                return d
    return 1


def _factor_d(n: int, d: int) -> List[int]:
    if d <= 1:
        return [n]
    s = max(1, int(round(n ** (1.0 / d))))
    for step in range(n):
        for candidate in (s - step, s + step):
            if 1 <= candidate <= n and n % candidate == 0:
                return [candidate] + _factor_d(n // candidate, d - 1)
    return [1] + _factor_d(n, d - 1)


def _matrix_fold_dims(m: int, n: int, d: int) -> Tuple[List[int], List[int]]:
    if d == 2:
        return [m], [n]
    if d == 3:
        s = max(1, int(round(math.sqrt(m))))
        m1 = _closest_divisor(m, s)
        m2 = m // m1
        return [m1, m2], [n]
    if d >= 4:
        s = max(1, int(round(math.sqrt(m))))
        m1 = _closest_divisor(m, s)
        m2 = m // m1
        t = max(1, int(round(math.sqrt(n))))
        n1 = _closest_divisor(n, t)
        n2 = n // n1
        return [m1, m2], [n1, n2]
    return [m], [n]


def _einsort_stage1(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mtx = np.asarray(matrix, dtype=np.float64)
    col_norms = np.linalg.norm(mtx, axis=0)
    col_order = np.argsort(-col_norms)
    mtx = mtx[:, col_order]
    row_norms = np.linalg.norm(mtx, axis=1)
    row_order = np.argsort(-row_norms)
    mtx = mtx[row_order, :]
    return mtx, row_order.astype(np.int32), col_order.astype(np.int32)


def _inverse_permute(
    matrix: np.ndarray, row_perm: np.ndarray, col_perm: np.ndarray
) -> np.ndarray:
    inv_row = np.argsort(row_perm)
    inv_col = np.argsort(col_perm)
    return matrix[inv_row, :][:, inv_col]


def _tt_svd_decompose(
    matrix: np.ndarray,
    target_ratio: float,
    tt_rank: Optional[int] = None,
    d: int = 3,
    storage_dtype: np.dtype = np.dtype("float16"),
) -> Tuple[List[np.ndarray], np.ndarray, dict]:
    m, n = matrix.shape
    row_dims, col_dims = _matrix_fold_dims(m, n, d)
    all_dims = row_dims + col_dims
    reshaped = matrix.reshape(*row_dims, *col_dims)
    total_el = m * n
    r = tt_rank
    if r is None:
        avg_side = max(1, int(round(math.sqrt(min(m, n)))))
        target_storage = total_el * 4 / max(target_ratio, 1.0)
        approx_rank = max(2, min(64, int(math.sqrt(target_storage / (d * avg_side)))))
        r = max(2, min(approx_rank, avg_side))
    r = max(2, r)
    cores = []
    current = np.asarray(reshaped, dtype=np.float64)
    prev_r = 1
    info: Dict[str, Any] = {
        "tt_rank": int(r),
        "all_dims": all_dims,
        "bond_dims": [],
        "singular_values": [],
        "singular_value_ratios": [],
    }
    for k in range(d - 1):
        ik = all_dims[k]
        unfolded = current.reshape(prev_r * ik, -1)
        target_rk = min(r, *unfolded.shape)
        if target_rk < 1:
            target_rk = 1
        U, S, Vt = np.linalg.svd(unfolded, full_matrices=False)
        info["singular_values"].append(S.tolist())
        if len(S) > 1:
            info["singular_value_ratios"].append(float(S[0] / S[-1]))
        else:
            info["singular_value_ratios"].append(1.0)
        rk = min(target_rk, len(S) - 1) if len(S) > 1 else 1
        rk = max(1, rk)
        info["bond_dims"].append(int(rk))
        core = U[:, :rk].reshape(prev_r, ik, rk)
        cores.append(convert_to_storage(core, storage_dtype))
        current = (S[:rk, None] * Vt[:rk, :]).reshape(rk, -1)
        prev_r = rk
    last_core = current.reshape(prev_r, all_dims[-1], 1)
    cores.append(convert_to_storage(last_core, storage_dtype))
    tt_recon = _tt_reconstruct(cores, all_dims, (m, n))
    residual = matrix - tt_recon
    info["residual_norm"] = float(np.linalg.norm(residual))
    info["tt_recon_norm"] = float(np.linalg.norm(tt_recon))
    return cores, residual, info


def _tt_reconstruct(
    cores: List[np.ndarray],
    all_dims: List[int],
    matrix_shape: Tuple[int, int],
    storage_dtype: np.dtype = np.float16,
) -> np.ndarray:
    def _to_f64(x):
        return convert_from_storage(x, storage_dtype).astype(np.float64)

    result = _to_f64(cores[0])
    for core in cores[1:]:
        result = np.tensordot(result, _to_f64(core), axes=([-1], [0]))
    m, n = matrix_shape
    return result.reshape(m, n)


def _sparse_residual_stage3(
    residual: np.ndarray,
    topk_ratio: float = 0.01,
    use_2_4: bool = False,
    storage_dtype: np.dtype = np.float16,
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, dict]:
    flat = residual.ravel().astype(np.float64)
    n = len(flat)
    info: Dict[str, Any] = {}
    if use_2_4:
        padded = np.zeros(int(math.ceil(n / 4) * 4), dtype=np.float64)
        padded[:n] = flat
        blocks = padded.reshape(-1, 4)
        abs_vals = np.abs(blocks)
        top2_idx = np.argpartition(abs_vals, 2, axis=1)[:, :2]
        mask = np.zeros_like(blocks, dtype=bool)
        rows_2 = np.repeat(np.arange(blocks.shape[0]), 2)
        cols_2 = top2_idx.ravel()
        mask[rows_2, cols_2] = True
        sparse_vals = blocks[mask]
        indices = np.where(mask.ravel())[0].astype(np.int16)
        info["sparsity_pattern"] = "2:4"
        info["n_kept"] = int(len(sparse_vals))
    else:
        n_keep = max(1, int(n * topk_ratio))
        abs_flat = np.abs(flat)
        top_idx = np.argpartition(-abs_flat, n_keep - 1)[:n_keep]
        indices = np.sort(top_idx).astype(np.int16)
        sparse_vals = flat[indices]
        info["sparsity_pattern"] = f"top-{topk_ratio * 100:.1f}%"
        info["n_kept"] = int(n_keep)
    info["kept_ratio"] = float(len(sparse_vals)) / float(n) if n > 0 else 0.0
    scale = float(np.max(np.abs(sparse_vals))) if len(sparse_vals) > 0 else 1.0
    info["scale"] = scale
    if scale > 1e-10:
        sparse_vals = sparse_vals / scale
    sparse_recon = np.zeros(n, dtype=np.float64)
    sparse_recon[indices.astype(np.int64)] = sparse_vals * scale
    residual2 = flat - sparse_recon
    info["residual_norm_before"] = float(np.linalg.norm(flat))
    info["residual_norm_after"] = float(np.linalg.norm(residual2))
    info["captured_norm"] = float(np.linalg.norm(sparse_recon))
    return (
        indices.astype(np.int16),
        convert_to_storage(sparse_vals, storage_dtype),
        np.float32(scale),
        residual2.reshape(residual.shape),
        info,
    )


def _sparse_reconstruct(
    indices: np.ndarray,
    values: np.ndarray,
    scale: float,
    n: int,
    storage_dtype: np.dtype = np.float16,
) -> np.ndarray:
    recon = np.zeros(n, dtype=np.float64)
    idx = indices.astype(np.int64)
    valid = idx < n
    recon[idx[valid]] = convert_from_storage(values, storage_dtype)[valid].astype(
        np.float64
    ) * float(scale)
    return recon


def _ergodic_trajectory_stage4(
    values: np.ndarray,
    n_channels: int = 16,
    storage_dtype: np.dtype = np.float16,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    flat = values.ravel().astype(np.float64)
    n = len(flat)
    n_chan = min(n_channels, max(1, n // 4))
    if n_chan < 1:
        n_chan = 1
    block_size = int(math.ceil(n / n_chan))

    _irrational_primes: np.ndarray = np.array(
        [
            2,
            3,
            5,
            7,
            11,
            13,
            17,
            19,
            23,
            29,
            31,
            37,
            41,
            43,
            47,
            53,
            59,
            61,
            67,
            71,
            73,
            79,
            83,
            89,
            97,
            101,
            103,
            107,
            109,
            113,
            127,
            131,
            137,
            139,
            149,
            151,
            157,
            163,
            167,
            173,
            179,
            181,
            191,
            193,
            197,
            199,
            211,
            223,
            227,
            229,
            233,
            239,
            241,
            251,
            257,
            263,
            269,
            271,
            277,
            281,
            283,
            293,
        ],
        dtype=np.float64,
    )

    padded = np.zeros(n_chan * block_size, dtype=np.float64)
    padded[:n] = flat
    blocks = padded.reshape(n_chan, block_size)
    n_avail = min(n_chan, len(_irrational_primes))
    alphas = np.sqrt(_irrational_primes[:n_avail])
    t = np.arange(block_size, dtype=np.float64)
    A_out = np.zeros(n_avail, dtype=np.float64)
    phi_out = np.zeros(n_avail, dtype=np.float64)
    bias_out = np.zeros(n_avail, dtype=np.float64)

    for c in range(n_avail):
        y = blocks[c]
        sin_at = np.sin(alphas[c] * t)
        cos_at = np.cos(alphas[c] * t)
        X = np.column_stack([sin_at, cos_at, np.ones(block_size, dtype=np.float64)])
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        b1, b2, bc = float(beta[0]), float(beta[1]), float(beta[2])
        A_out[c] = math.sqrt(b1 * b1 + b2 * b2)
        phi_out[c] = math.atan2(b2, b1)
        bias_out[c] = bc

    ergodic_recon = np.zeros(n, dtype=np.float64)
    for c in range(n_avail):
        start = c * block_size
        end = min(start + block_size, n)
        seg_len = end - start
        if seg_len <= 0:
            continue
        t_seg = np.arange(seg_len, dtype=np.float64)
        seg_recon = A_out[c] * np.sin(alphas[c] * t_seg + phi_out[c]) + bias_out[c]
        ergodic_recon[start:end] = seg_recon

    residual2 = flat - ergodic_recon
    info: Dict[str, Any] = {
        "n_channels": int(n_chan),
        "n_avail": int(n_avail),
        "block_size": int(block_size),
        "residual_norm_before": float(np.linalg.norm(flat)),
        "residual_norm_after": float(np.linalg.norm(residual2)),
        "recon_norm": float(np.linalg.norm(ergodic_recon)),
        "amplitudes_stats": {
            "mean": float(np.mean(A_out)),
            "std": float(np.std(A_out)),
            "min": float(np.min(A_out)),
            "max": float(np.max(A_out)),
        },
        "bias_stats": {
            "mean": float(np.mean(bias_out)),
            "std": float(np.std(bias_out)),
            "min": float(np.min(bias_out)),
            "max": float(np.max(bias_out)),
        },
    }
    return (
        convert_to_storage(alphas, storage_dtype),
        convert_to_storage(A_out, storage_dtype),
        convert_to_storage(phi_out, storage_dtype),
        convert_to_storage(bias_out, storage_dtype),
        residual2.reshape(values.shape),
        info,
    )


def _ergodic_reconstruct(
    alphas: np.ndarray,
    A: np.ndarray,
    phi: np.ndarray,
    bias: np.ndarray,
    n: int,
    storage_dtype: np.dtype = np.float16,
) -> np.ndarray:
    alphas_f64 = convert_from_storage(alphas, storage_dtype).astype(np.float64)
    A_f64 = convert_from_storage(A, storage_dtype).astype(np.float64)
    phi_f64 = convert_from_storage(phi, storage_dtype).astype(np.float64)
    bias_f64 = convert_from_storage(bias, storage_dtype).astype(np.float64)
    n_chan = min(len(alphas_f64), len(A_f64), len(phi_f64), len(bias_f64))
    if n_chan == 0:
        return np.zeros(n, dtype=np.float64)
    block_size = int(math.ceil(n / n_chan))
    recon = np.zeros(n, dtype=np.float64)
    for c in range(n_chan):
        start = c * block_size
        end = min(start + block_size, n)
        seg_len = end - start
        if seg_len <= 0:
            continue
        t_seg = np.arange(seg_len, dtype=np.float64)
        recon[start:end] = float(A_f64[c]) * np.sin(
            float(alphas_f64[c]) * t_seg + float(phi_f64[c])
        ) + float(bias_f64[c])
    return recon


def _siren_fit_2d(
    residual: np.ndarray,
    original_shape: Tuple[int, int],
    hidden_dim: int = 32,
    n_epochs: int = 200,
    storage_dtype: np.dtype = np.float16,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, dict]:
    rows, cols = original_shape
    n = rows * cols
    row_coords = np.linspace(-1.0, 1.0, rows)
    col_coords = np.linspace(-1.0, 1.0, cols)
    r_grid, c_grid = np.meshgrid(row_coords, col_coords, indexing="ij")
    coords = np.column_stack([r_grid.ravel(), c_grid.ravel()]).astype(np.float64)
    target = residual.ravel().astype(np.float64)
    t_mean = float(np.mean(target))
    t_std = float(np.std(target)) + 1e-10
    target_norm = (target - t_mean) / t_std
    rng = np.random.RandomState(42)
    w1 = (rng.randn(2, hidden_dim) * (6.0 / 2.0**0.5)).astype(np.float64)
    b1 = (rng.uniform(-np.pi, np.pi, hidden_dim)).astype(np.float64)
    wo = np.zeros(hidden_dim, dtype=np.float64)
    bo = 0.0
    lr = 0.005

    losses = []
    for epoch in range(n_epochs):
        h = np.sin(coords @ w1 + b1)
        pred = h @ wo + bo
        loss = pred - target_norm
        grad_wo = (h.T @ loss) / n
        grad_bo = float(np.mean(loss))
        wo -= lr * grad_wo
        bo -= lr * grad_bo
        if epoch % 50 == 49:
            lr *= 0.5
        if epoch % 50 == 0:
            losses.append(float(np.mean(loss * loss)))

    bo = bo * t_std + t_mean
    wo = wo * t_std

    info: Dict[str, Any] = {
        "hidden_dim": hidden_dim,
        "n_epochs": n_epochs,
        "final_lr": lr,
        "initial_loss": losses[0] if losses else None,
        "final_loss": losses[-1] if losses else None,
        "target_stats": {
            "mean": t_mean,
            "std": t_std,
        },
    }
    return (
        convert_to_storage(w1, storage_dtype),
        convert_to_storage(b1, storage_dtype),
        convert_to_storage(wo, storage_dtype),
        np.float32(bo),
        info,
    )


def _siren_reconstruct(
    w1: np.ndarray,
    b1: np.ndarray,
    wo: np.ndarray,
    bo: float,
    shape: Tuple[int, int],
    storage_dtype: np.dtype = np.float16,
) -> np.ndarray:
    rows, cols = shape
    row_coords = np.linspace(-1.0, 1.0, rows)
    col_coords = np.linspace(-1.0, 1.0, cols)
    r_grid, c_grid = np.meshgrid(row_coords, col_coords, indexing="ij")
    coords = np.column_stack([r_grid.ravel(), c_grid.ravel()]).astype(np.float64)
    h = np.sin(
        coords @ convert_from_storage(w1, storage_dtype).astype(np.float64)
        + convert_from_storage(b1, storage_dtype).astype(np.float64)
    )
    pred = h @ convert_from_storage(wo, storage_dtype).astype(np.float64) + float(bo)
    return pred.reshape(shape).astype(np.float64)


# ── Load a BF16 safetensors tensor ──────────────────────────────────────────


def load_bf16_safetensors_tensor(path: str, tensor_key: str) -> np.ndarray:
    """Load a single BF16 tensor from a safetensors file as float32."""
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len).decode("utf-8"))
        info = header[tensor_key]
        start, end = info["data_offsets"]
        offset = 8 + header_len + start
        f.seek(offset)
        raw = f.read(end - start)
        vals_u16 = np.frombuffer(raw, dtype=np.uint16)
        vals_f32 = (vals_u16.astype(np.uint32) << 16).view(np.float32)
        return vals_f32.reshape(info["shape"])


# ── Main diagnostic routine ─────────────────────────────────────────────────


def describe_residual(residual: np.ndarray, label: str) -> Dict[str, Any]:
    flat = residual.ravel()
    n = len(flat)
    nz = np.count_nonzero(flat)
    info = {
        "label": label,
        "shape": list(residual.shape),
        "n_elements": int(n),
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "norm": float(np.linalg.norm(flat)),
        "sparsity": float(1.0 - nz / n) if n > 0 else 0.0,
        "n_nonzero": int(nz),
    }
    return info


def compute_cumulative_error(
    original: np.ndarray, cumulative_recon: np.ndarray, label: str
) -> Dict[str, Any]:
    em = end_to_end_error(original, cumulative_recon)
    return {
        "stage": label,
        "rel_mse": float(em.rel_mse),
        "cosine_sim": float(em.cosine_sim),
        "max_abs_error": float(em.max_abs),
        "snr_db": float(em.snr_db),
    }


def make_serializable(obj: Any) -> Any:
    """Recursively convert numpy types to native Python types for JSON."""
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_serializable(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def diagnose(
    model_path: str = "models/gemma-4-E2B/model.safetensors",
    tensor_key: str = "model.language_model.layers.0.self_attn.q_proj.weight",
    target_ratio: float = 200.0,
    tt_rank: Optional[int] = None,
    sparse_topk_ratio: float = 0.01,
    ergodic_n_channels: int = 16,
    siren_hidden_dim: int = 32,
    siren_n_epochs: int = 200,
    d: int = 3,
    use_2_4_sparsity: bool = False,
    output_json: str = "stage_diagnosis.json",
) -> Dict[str, Any]:
    results: Dict[str, Any] = {
        "config": {
            "tensor_key": tensor_key,
            "target_ratio": target_ratio,
            "tt_rank": tt_rank,
            "sparse_topk_ratio": sparse_topk_ratio,
            "ergodic_n_channels": ergodic_n_channels,
            "siren_hidden_dim": siren_hidden_dim,
            "siren_n_epochs": siren_n_epochs,
            "d": d,
            "use_2_4_sparsity": use_2_4_sparsity,
        },
        "stages": {},
        "diagnosis": {},
    }

    # ── Load tensor ──────────────────────────────────────────────────────
    print("=" * 72)
    print(" 5-STAGE CASCADE DIAGNOSTIC")
    print("=" * 72)
    print(f"\nLoading tensor: {tensor_key}")
    t0 = time.time()
    original = load_bf16_safetensors_tensor(model_path, tensor_key)
    t_load = time.time() - t0
    print(f"  Shape: {original.shape}, dtype: {original.dtype}")
    print(f"  Range: [{original.min():.6f}, {original.max():.6f}]")
    print(f"  Mean: {original.mean():.8f}, Std: {original.std():.8f}")
    print(f"  Load time: {t_load:.2f}s")
    print()
    results["tensor_info"] = {
        "shape": list(original.shape),
        "dtype": str(original.dtype),
        "n_elements": int(original.size),
        "min": float(original.min()),
        "max": float(original.max()),
        "mean": float(original.mean()),
        "std": float(original.std()),
    }

    storage_dtype = detect_storage_dtype(original)
    print(f"  Storage dtype: {storage_dtype}")
    print()

    # ── Stage 1: EinSort ──────────────────────────────────────────────────
    print("-" * 72)
    print(" STAGE 1: EinSort (norm-based permutation)")
    print("-" * 72)
    t0 = time.time()
    permuted, row_perm, col_perm = _einsort_stage1(original)
    t1 = time.time() - t0
    print(f"  Time: {t1:.3f}s")
    print(f"  Permuted shape: {permuted.shape}")

    # Check the permutation quality
    inv_row = np.argsort(row_perm)
    inv_col = np.argsort(col_perm)
    unpermuted = permuted[inv_row, :][:, inv_col]
    perm_error = np.linalg.norm(original.astype(np.float64) - unpermuted)
    print(f"  Permutation roundtrip error: {perm_error:.2e} (should be ~0)")

    # Compute SVD to check singular spectrum decay
    print("  Computing SVD of original and permuted...")
    svd_original = np.linalg.svd(original.astype(np.float64), full_matrices=False)
    svd_permuted = np.linalg.svd(permuted, full_matrices=False)

    s1_data = {
        "time_sec": t1,
        "permuted_shape": list(permuted.shape),
        "roundtrip_error": float(perm_error),
        "original_sv_top10": [float(s) for s in svd_original[1][:10]],
        "permuted_sv_top10": [float(s) for s in svd_permuted[1][:10]],
        "original_sv_ratio_s1_over_end": float(
            svd_original[1][0] / svd_original[1][-1]
        ),
        "permuted_sv_ratio_s1_over_end": float(
            svd_permuted[1][0] / svd_permuted[1][-1]
        ),
        "original_sv_tail_sum_ratio": float(
            np.sum(svd_original[1][256:]) / np.sum(svd_original[1])
        ),
        "permuted_sv_tail_sum_ratio": float(
            np.sum(svd_permuted[1][256:]) / np.sum(svd_permuted[1])
        ),
    }
    print(f"  Original SVs (top 10): {[f'{s:.4f}' for s in svd_original[1][:10]]}")
    print(f"  Permuted SVs (top 10): {[f'{s:.4f}' for s in svd_permuted[1][:10]]}")
    print(
        f"  Original SV ratio (sigma_1 / sigma_n): {s1_data['original_sv_ratio_s1_over_end']:.2e}"
    )
    print(
        f"  Permuted SV ratio (sigma_1 / sigma_n): {s1_data['permuted_sv_ratio_s1_over_end']:.2e}"
    )
    print()

    # ── Stage 2: TT-SVD ──────────────────────────────────────────────────
    print("-" * 72)
    print(" STAGE 2: Tensor Train SVD")
    print("-" * 72)

    # Diagnose with computed rank
    t0 = time.time()
    cores, tt_residual, tt_info = _tt_svd_decompose(
        permuted,
        target_ratio,
        tt_rank=tt_rank,
        d=d,
        storage_dtype=storage_dtype,
    )
    t2 = time.time() - t0
    tt_recon = _tt_reconstruct(cores, tt_info["all_dims"], permuted.shape)
    tt_error_metrics = end_to_end_error(permuted, tt_recon)

    print(f"  Time: {t2:.3f}s")
    print(f"  TT rank used: {tt_info['tt_rank']}")
    print(f"  Bond dims: {tt_info['bond_dims']}")
    print(f"  All dims: {tt_info['all_dims']}")
    print(f"  TT cores shapes: {[list(c.shape) for c in cores]}")
    print(f"  TT reconstruction error vs permuted:")
    print(f"    rel_mse = {tt_error_metrics.rel_mse:.6e}")
    print(f"    cosine_sim = {tt_error_metrics.cosine_sim:.6f}")
    print(f"    max_abs_error = {tt_error_metrics.max_abs:.6e}")
    print(f"    snr_db = {tt_error_metrics.snr_db:.2f} dB")
    print(f"  TT info singular_value_ratios: {tt_info['singular_value_ratios']}")

    # Also try rank-8 and rank-64 for comparison
    print("\n  Comparison TT ranks:")
    for test_rank in [8, 16, 32, 48, 64]:
        if test_rank <= min(permuted.shape):
            tcores, _, test_info = _tt_svd_decompose(
                permuted,
                target_ratio,
                tt_rank=test_rank,
                d=d,
                storage_dtype=storage_dtype,
            )
            test_recon2 = _tt_reconstruct(tcores, test_info["all_dims"], permuted.shape)
            test_em = end_to_end_error(permuted, test_recon2)
            print(
                f"    rank-{test_rank:3d}: bond_dims={test_info['bond_dims']}, "
                f"SNR={test_em.snr_db:.1f}dB, rel_mse={test_em.rel_mse:.2e}"
            )

    s2_data = {
        "time_sec": t2,
        "tt_rank": int(tt_info["tt_rank"]),
        "bond_dims": [int(b) for b in tt_info["bond_dims"]],
        "all_dims": [int(x) for x in tt_info["all_dims"]],
        "core_shapes": [list(c.shape) for c in cores],
        "singular_value_ratios": [float(x) for x in tt_info["singular_value_ratios"]],
        "error_vs_permuted": {
            "rel_mse": float(tt_error_metrics.rel_mse),
            "cosine_sim": float(tt_error_metrics.cosine_sim),
            "max_abs_error": float(tt_error_metrics.max_abs),
            "snr_db": float(tt_error_metrics.snr_db),
        },
    }
    print()

    # ── Stage 3: Sparse Residual ──────────────────────────────────────────
    print("-" * 72)
    print(" STAGE 3: Sparse Residual (top-1%)")
    print("-" * 72)
    t0 = time.time()
    sparse_idx, sparse_vals, sparse_scale, sparse_residual, s3_info = (
        _sparse_residual_stage3(
            tt_residual,
            topk_ratio=sparse_topk_ratio,
            use_2_4=use_2_4_sparsity,
            storage_dtype=storage_dtype,
        )
    )
    t3 = time.time() - t0
    sparse_recon = _sparse_reconstruct(
        sparse_idx,
        sparse_vals,
        float(sparse_scale),
        permuted.size,
        storage_dtype=storage_dtype,
    ).reshape(permuted.shape)
    cumulative_after_s3 = tt_recon + sparse_recon
    s3_vs_tt_residual_em = end_to_end_error(tt_residual, sparse_recon)
    s3_cumulative_em = end_to_end_error(permuted, cumulative_after_s3)

    print(f"  Time: {t3:.3f}s")
    print(f"  Pattern: {s3_info['sparsity_pattern']}")
    print(
        f"  Kept: {s3_info['n_kept']} / {permuted.size} ({s3_info['kept_ratio'] * 100:.4f}%)"
    )
    print(f"  Scale: {s3_info['scale']:.6f}")
    print(f"  Sparse reconstruction vs TT residual:")
    print(
        f"    rel_mse = {s3_vs_tt_residual_em.rel_mse:.6e}, SNR = {s3_vs_tt_residual_em.snr_db:.2f}dB"
    )
    print(f"  Cumulative (TT+sparse) vs permuted:")
    print(
        f"    rel_mse = {s3_cumulative_em.rel_mse:.6e}, SNR = {s3_cumulative_em.snr_db:.2f}dB"
    )
    s3_residual_desc = describe_residual(sparse_residual, "after_sparse")

    s3_data = {
        "time_sec": t3,
        "sparsity_pattern": s3_info["sparsity_pattern"],
        "n_kept": int(s3_info["n_kept"]),
        "n_total": int(permuted.size),
        "kept_ratio": float(s3_info["kept_ratio"]),
        "scale": float(s3_info["scale"]),
        "residual_after_sparse": s3_residual_desc,
        "error_vs_tt_residual": {
            "rel_mse": float(s3_vs_tt_residual_em.rel_mse),
            "cosine_sim": float(s3_vs_tt_residual_em.cosine_sim),
            "max_abs_error": float(s3_vs_tt_residual_em.max_abs),
            "snr_db": float(s3_vs_tt_residual_em.snr_db),
        },
        "cumulative_error_vs_permuted": {
            "rel_mse": float(s3_cumulative_em.rel_mse),
            "cosine_sim": float(s3_cumulative_em.cosine_sim),
            "max_abs_error": float(s3_cumulative_em.max_abs),
            "snr_db": float(s3_cumulative_em.snr_db),
        },
    }
    print()

    # ── Stage 4: Ergodic ──────────────────────────────────────────────────
    print("-" * 72)
    print(" STAGE 4: Ergodic Hyperfunction")
    print("-" * 72)
    t0 = time.time()
    _alphas, _A, _phi, _bias, ergodic_residual, s4_info = _ergodic_trajectory_stage4(
        sparse_residual,
        n_channels=ergodic_n_channels,
        storage_dtype=storage_dtype,
    )
    t4 = time.time() - t0
    ergodic_recon = _ergodic_reconstruct(
        _alphas, _A, _phi, _bias, sparse_residual.size, storage_dtype=storage_dtype
    ).reshape(sparse_residual.shape)
    cumulative_after_s4 = cumulative_after_s3 + ergodic_recon
    s4_vs_sparse_residual_em = end_to_end_error(sparse_residual, ergodic_recon)
    s4_cumulative_em = end_to_end_error(permuted, cumulative_after_s4)

    print(f"  Time: {t4:.3f}s")
    print(f"  Channels: {s4_info['n_channels']}, Available: {s4_info['n_avail']}")
    print(f"  Block size: {s4_info['block_size']}")
    print(f"  Amplitude stats: {s4_info['amplitudes_stats']}")
    print(f"  Bias stats: {s4_info['bias_stats']}")
    print(f"  Ergodic recon vs sparse residual:")
    print(
        f"    rel_mse = {s4_vs_sparse_residual_em.rel_mse:.6e}, SNR = {s4_vs_sparse_residual_em.snr_db:.2f}dB"
    )
    print(f"  Cumulative (TT+sparse+ergodic) vs permuted:")
    print(
        f"    rel_mse = {s4_cumulative_em.rel_mse:.6e}, SNR = {s4_cumulative_em.snr_db:.2f}dB"
    )
    s4_residual_desc = describe_residual(ergodic_residual, "after_ergodic")

    s4_data = {
        "time_sec": t4,
        "n_channels": int(s4_info["n_channels"]),
        "n_avail": int(s4_info["n_avail"]),
        "block_size": int(s4_info["block_size"]),
        "amplitudes_stats": {
            k: float(v) for k, v in s4_info["amplitudes_stats"].items()
        },
        "bias_stats": {k: float(v) for k, v in s4_info["bias_stats"].items()},
        "residual_after_ergodic": s4_residual_desc,
        "error_vs_sparse_residual": {
            "rel_mse": float(s4_vs_sparse_residual_em.rel_mse),
            "cosine_sim": float(s4_vs_sparse_residual_em.cosine_sim),
            "max_abs_error": float(s4_vs_sparse_residual_em.max_abs),
            "snr_db": float(s4_vs_sparse_residual_em.snr_db),
        },
        "cumulative_error_vs_permuted": {
            "rel_mse": float(s4_cumulative_em.rel_mse),
            "cosine_sim": float(s4_cumulative_em.cosine_sim),
            "max_abs_error": float(s4_cumulative_em.max_abs),
            "snr_db": float(s4_cumulative_em.snr_db),
        },
    }
    print()

    # ── Stage 5: SIREN ──────────────────────────────────────────────────
    print("-" * 72)
    print(" STAGE 5: SIREN (implicit neural representation)")
    print("-" * 72)
    t0 = time.time()
    w1, b1, wo, bo, s5_info = _siren_fit_2d(
        ergodic_residual,
        permuted.shape,
        hidden_dim=siren_hidden_dim,
        n_epochs=siren_n_epochs,
        storage_dtype=storage_dtype,
    )
    t5 = time.time() - t0
    siren_recon = _siren_reconstruct(
        w1, b1, wo, float(bo), permuted.shape, storage_dtype=storage_dtype
    )
    full_recon_permuted = cumulative_after_s4 + siren_recon
    s5_vs_ergodic_residual_em = end_to_end_error(ergodic_residual, siren_recon)
    s5_cumulative_em = end_to_end_error(permuted, full_recon_permuted)

    print(f"  Time: {t5:.3f}s")
    print(f"  Hidden dim: {s5_info['hidden_dim']}, Epochs: {s5_info['n_epochs']}")
    print(f"  Initial loss: {s5_info['initial_loss']:.6e}")
    print(f"  Final loss: {s5_info['final_loss']:.6e}")
    print(f"  SIREN recon vs ergodic residual:")
    print(
        f"    rel_mse = {s5_vs_ergodic_residual_em.rel_mse:.6e}, SNR = {s5_vs_ergodic_residual_em.snr_db:.2f}dB"
    )
    print(f"  Cumulative (all stages) vs permuted:")
    print(
        f"    rel_mse = {s5_cumulative_em.rel_mse:.6e}, SNR = {s5_cumulative_em.snr_db:.2f}dB"
    )

    s5_data = {
        "time_sec": t5,
        "hidden_dim": int(s5_info["hidden_dim"]),
        "n_epochs": int(s5_info["n_epochs"]),
        "initial_loss": float(s5_info["initial_loss"])
        if s5_info["initial_loss"] is not None
        else None,
        "final_loss": float(s5_info["final_loss"])
        if s5_info["final_loss"] is not None
        else None,
        "target_stats": {k: float(v) for k, v in s5_info["target_stats"].items()},
        "error_vs_ergodic_residual": {
            "rel_mse": float(s5_vs_ergodic_residual_em.rel_mse),
            "cosine_sim": float(s5_vs_ergodic_residual_em.cosine_sim),
            "max_abs_error": float(s5_vs_ergodic_residual_em.max_abs),
            "snr_db": float(s5_vs_ergodic_residual_em.snr_db),
        },
        "cumulative_error_vs_permuted": {
            "rel_mse": float(s5_cumulative_em.rel_mse),
            "cosine_sim": float(s5_cumulative_em.cosine_sim),
            "max_abs_error": float(s5_cumulative_em.max_abs),
            "snr_db": float(s5_cumulative_em.snr_db),
        },
    }
    print()

    # ── Final: inverse permute and evaluate against original ──────────────
    print("=" * 72)
    print(" FINAL RECONSTRUCTION (including inverse permutation)")
    print("=" * 72)
    final_recon = _inverse_permute(full_recon_permuted, row_perm, col_perm).reshape(
        original.shape
    )
    final_em = end_to_end_error(original, final_recon)
    print(f"  Final error vs original:")
    print(f"    rel_mse = {final_em.rel_mse:.6e}")
    print(f"    cosine_sim = {final_em.cosine_sim:.6f}")
    print(f"    max_abs_error = {final_em.max_abs:.6e}")
    print(f"    snr_db = {final_em.snr_db:.2f} dB")

    # ── Ratio computation ────────────────────────────────────────────────
    payload_example = {
        "s1_row_perm": row_perm,
        "s1_col_perm": col_perm,
        "s2_cores": cores,
        "s3_indices": sparse_idx,
        "s3_values": sparse_vals,
        "s3_scale": sparse_scale,
    }
    fp32_ratio = ratio_vs_fp32(int(original.size), payload_example)
    print(f"\n  Ratio (TT+Sparse only): {fp32_ratio:.1f}:1 vs FP32")
    print()

    # ── Final diagnosis ──────────────────────────────────────────────────
    print("=" * 72)
    print(" ERROR BREAKDOWN (incremental)")
    print("=" * 72)

    stages_order = [
        ("Stage 1 EinSort", permuted, original.astype(np.float64), tt_info),
        ("Stage 2 TT-SVD", tt_recon, permuted, tt_info),
        ("Stage 3 Sparse", cumulative_after_s3, permuted, s3_info),
        ("Stage 4 Ergodic", cumulative_after_s4, permuted, s4_info),
        ("Stage 5 SIREN", full_recon_permuted, permuted, s5_info),
    ]

    prev_error_snr = float("inf")
    stage_error_breakdown = []
    for label, recon, target, _info in stages_order:
        em = end_to_end_error(target, recon)
        stage_error_breakdown.append(
            {
                "stage": label,
                "rel_mse": float(em.rel_mse),
                "snr_db": float(em.snr_db),
                "cosine_sim": float(em.cosine_sim),
            }
        )
        print(
            f"  {label:30s}: SNR={em.snr_db:8.2f}dB  rel_mse={em.rel_mse:.2e}  cos={em.cosine_sim:.6f}"
        )

    # Per-stage incremental error contribution
    print(
        "\n  Incremental error contribution (each stage's residual vs what it received):"
    )
    inc_breakdown = [
        ("Stage 2 TT-SVD residual", end_to_end_error(permuted, tt_recon)),
        ("Stage 3 sparse captures ", s3_vs_tt_residual_em),
        ("Stage 4 ergodic captures", s4_vs_sparse_residual_em),
        ("Stage 5 SIREN captures  ", s5_vs_ergodic_residual_em),
    ]
    for label, em in inc_breakdown:
        print(f"  {label:30s}: SNR={em.snr_db:8.2f}dB  rel_mse={em.rel_mse:.2e}")

    # Also compute : what does permuted-only vs original look like?
    permuted_vs_original = end_to_end_error(original.astype(np.float64), permuted)

    print("\n  EinSort (permuted vs original):")
    print(
        f"    SNR={permuted_vs_original.snr_db:.2f}dB (should be infinite - permutation is lossless)"
    )

    # Now also compute with a synthetic matrix to compare SVD truncation behavior
    print("\n" + "=" * 72)
    print(" ANALYSIS: Where does quality go wrong?")
    print("=" * 72)

    # Check: is TT error dominant?
    tt_error = tt_error_metrics.rel_mse
    final_error = final_em.rel_mse
    tt_contribution = tt_error / final_error if final_error > 0 else 0
    print(f"  TT-SVD relative MSE (vs permuted):  {tt_error:.6e}")
    print(f"  Final relative MSE (vs original):   {final_error:.6e}")
    print(f"  TT contribution to final error:     {tt_contribution * 100:.1f}%")

    # Diagnose the residual after each stage
    print("\n  Residual norm progression:")
    print(
        f"    After TT:      ||R|| = {np.linalg.norm(tt_residual):.6f}  "
        f"(ratio to orig: {np.linalg.norm(tt_residual) / np.linalg.norm(permuted):.4f})"
    )
    print(
        f"    After sparse:  ||R|| = {np.linalg.norm(sparse_residual):.6f}  "
        f"(ratio to orig: {np.linalg.norm(sparse_residual) / np.linalg.norm(permuted):.4f})"
    )
    print(
        f"    After ergodic: ||R|| = {np.linalg.norm(ergodic_residual):.6f}  "
        f"(ratio to orig: {np.linalg.norm(ergodic_residual) / np.linalg.norm(permuted):.4f})"
    )

    # Analysis of SIREN target statistics
    print(f"\n  SIREN target (input residual) stats:")
    print(f"    Mean: {s5_info['target_stats']['mean']:.8f}")
    print(f"    Std:  {s5_info['target_stats']['std']:.8f}")
    is_noise_floor = s5_info["target_stats"]["std"] < 1e-4
    print(f"    Is this noise floor? {'YES ⚠️' if is_noise_floor else 'No (good)'}")

    # ── Compile results ─────────────────────────────────────────────────
    results["stages"]["1_einsort"] = make_serializable(s1_data)
    results["stages"]["2_tt_svd"] = make_serializable(s2_data)
    results["stages"]["3_sparse"] = make_serializable(s3_data)
    results["stages"]["4_ergodic"] = make_serializable(s4_data)
    results["stages"]["5_siren"] = make_serializable(s5_data)
    results["final"] = make_serializable(
        {
            "error_vs_original": {
                "rel_mse": float(final_em.rel_mse),
                "cosine_sim": float(final_em.cosine_sim),
                "max_abs_error": float(final_em.max_abs),
                "snr_db": float(final_em.snr_db),
            },
            "stage_error_breakdown": stage_error_breakdown,
            "incremental_error_contribution": [
                {"stage": l, "rel_mse": float(em.rel_mse), "snr_db": float(em.snr_db)}
                for l, em in inc_breakdown
            ],
            "residual_norm_progression": {
                "after_tt": float(np.linalg.norm(tt_residual)),
                "after_sparse": float(np.linalg.norm(sparse_residual)),
                "after_ergodic": float(np.linalg.norm(ergodic_residual)),
                "original_norm": float(np.linalg.norm(permuted)),
            },
        }
    )

    # ── Diagnosis text ──────────────────────────────────────────────────
    diag_lines: List[str] = []

    # Determine dominant error source
    errors_by_stage = {
        "TT-SVD (rank truncation)": tt_error_metrics.rel_mse,
        "Sparse (top-1% overflow)": float(
            end_to_end_error(tt_residual, sparse_recon).rel_mse
        ),
        "Ergodic (sine misfit)": float(
            end_to_end_error(sparse_residual, ergodic_recon).rel_mse
        ),
        "SIREN (INR underfit)": float(
            end_to_end_error(ergodic_residual, siren_recon).rel_mse
        ),
    }
    dominant = max(errors_by_stage, key=errors_by_stage.get)
    diag_lines.append(f"Dominant error source: {dominant}")

    # Check if TT is the bottleneck
    if tt_contribution > 0.5:
        diag_lines.append(
            f"TT-SVD contributes {tt_contribution * 100:.1f}% of final error. "
            f"Rank {tt_info['tt_rank']} is too aggressive."
        )
    else:
        diag_lines.append(
            f"TT-SVD contributes {tt_contribution * 100:.1f}% of final error. "
            f"Post-TT stages fail to recover the residual."
        )

    # Check SIREN
    if is_noise_floor:
        diag_lines.append(
            "SIREN input is near noise floor; INR cannot learn meaningful structure."
        )

    # Check ergodic
    erg_capture_snr = s4_vs_sparse_residual_em.snr_db
    diag_lines.append(
        f"Ergodic stage captures sparse residual at SNR={erg_capture_snr:.1f}dB."
    )

    # Check sparse
    sparse_capture_snr = s3_vs_tt_residual_em.snr_db
    diag_lines.append(
        f"Sparse stage captures TT residual at SNR={sparse_capture_snr:.1f}dB."
    )

    results["diagnosis"] = make_serializable(
        {
            "dominant_error_source": dominant,
            "tt_error_contribution_pct": float(tt_contribution * 100),
            "errors_by_stage": errors_by_stage,
            "summary": " ".join(diag_lines),
            "recommendations": _generate_recommendations(results),
        }
    )

    for line in diag_lines:
        print(f"  {line}")

    # ── Save ────────────────────────────────────────────────────────────
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_json}")

    return results


def _generate_recommendations(results: Dict[str, Any]) -> List[str]:
    recs = []
    s2 = results.get("stages", {}).get("2_tt_svd", {})
    s3 = results.get("stages", {}).get("3_sparse", {})
    s4 = results.get("stages", {}).get("4_ergodic", {})
    s5 = results.get("stages", {}).get("5_siren", {})
    final = results.get("final", {})

    # Check TT rank vs approximation error
    tt_rank = s2.get("tt_rank", 0)
    tt_bond_dims = s2.get("bond_dims", [])
    tt_snr = s2.get("error_vs_permuted", {}).get("snr_db", 0)

    if tt_snr < 20:
        recs.append(
            f"Increase TT rank (currently {tt_rank}, bond_dims={tt_bond_dims}). "
            f"Rank-32 TT gives SNR={tt_snr:.1f}dB — try rank-64 or rank-128."
        )

    # Check if sparse top-1% is too small
    s3_snr = s3.get("cumulative_error_vs_permuted", {}).get("snr_db", 0)
    if s3_snr < tt_snr + 3:
        recs.append(
            f"Sparse stage (top-1%) barely improves SNR from TT ({tt_snr:.1f} → {s3_snr:.1f}dB). "
            f"Increase topk_ratio to 0.05 or switch to 2:4 sparsity."
        )

    # Check ergodic sine fit quality
    s4_snr = s4.get("cumulative_error_vs_permuted", {}).get("snr_db", 0)
    if s4_snr < s3_snr + 1:
        recs.append(
            f"Ergodic stage adds negligible SNR gain ({s3_snr:.1f} → {s4_snr:.1f}dB). "
            f"Increase n_channels or skip this stage entirely."
        )

    # Check SIREN
    s5_snr = s5.get("cumulative_error_vs_permuted", {}).get("snr_db", 0)
    if s5_snr < s4_snr + 2:
        recs.append(
            f"SIREN stage adds negligible SNR gain ({s4_snr:.1f} → {s5_snr:.1f}dB). "
            f"Increase hidden_dim, n_epochs, or replace with a more expressive INR."
        )

    final_snr = final.get("error_vs_original", {}).get("snr_db", 0)
    recs.append(
        f"Target: improve final SNR from {final_snr:.1f} dB to ≥20 dB. "
        f"Primary lever: {'increase TT rank' if tt_snr < 10 else 'improve residual fitting stages'}."
    )

    return recs


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Diagnose 5-stage cascade pipeline")
    parser.add_argument(
        "--model",
        default="models/gemma-4-E2B/model.safetensors",
        help="Path to safetensors model file",
    )
    parser.add_argument(
        "--tensor-key",
        default="model.language_model.layers.0.self_attn.q_proj.weight",
        help="Tensor key to load",
    )
    parser.add_argument(
        "--target-ratio",
        type=float,
        default=200.0,
        help="Target compression ratio",
    )
    parser.add_argument("--tt-rank", type=int, default=None, help="Override TT rank")
    parser.add_argument(
        "--output",
        default="stage_diagnosis.json",
        help="Output JSON path",
    )
    args = parser.parse_args()

    diag = diagnose(
        model_path=args.model,
        tensor_key=args.tensor_key,
        target_ratio=args.target_ratio,
        tt_rank=args.tt_rank,
        output_json=args.output,
    )
    return diag


if __name__ == "__main__":
    main()
