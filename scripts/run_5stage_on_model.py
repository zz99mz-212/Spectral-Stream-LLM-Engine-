from __future__ import annotations

import json
import math
import pickle
import struct
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spectralstream.compression.cascade_5stage import (
    FiveStageCascade,
    compress_cascade,
    decompress_cascade,
    _einsort_stage1,
    _tt_svd_decompose,
    _sparse_residual_stage3,
    _ergodic_trajectory_stage4,
    _siren_fit_2d,
)
from spectralstream.compression.honest_metrics import (
    serialized_nbytes,
    end_to_end_error,
    dual_ratio,
    ErrorMetrics,
)

FP32_BYTES = 4
BF16_BYTES = 2


def load_safetensors_bf16(path: str) -> Dict[str, np.ndarray]:
    """Load safetensors file, converting BF16 to float32 on the fly."""
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header_json = f.read(header_len).decode("utf-8")
        header = json.loads(header_json)

    tensors: Dict[str, np.ndarray] = {}
    body_offset = 8 + header_len
    for name, info in header.items():
        if name == "__metadata__":
            continue
        dtype_str = info["dtype"]
        shape = tuple(info["shape"])
        off = tuple(info["data_offsets"])
        with open(path, "rb") as f:
            f.seek(body_offset + off[0])
            raw = f.read(off[1] - off[0])
        if dtype_str == "BF16":
            t = np.frombuffer(raw, dtype=np.uint16).reshape(shape).astype(np.float32)
        elif dtype_str == "F32":
            t = np.frombuffer(raw, dtype=np.float32).reshape(shape)
        elif dtype_str == "F16":
            t = np.frombuffer(raw, dtype=np.float16).reshape(shape).astype(np.float32)
        elif dtype_str == "F8_E4M3":
            t = np.frombuffer(raw, dtype=np.uint8).reshape(shape).astype(np.float32)
        else:
            t = np.frombuffer(raw, dtype=np.dtype(dtype_str.lower())).reshape(shape)
        tensors[name] = t
    return tensors


def synthetic_llm_tensor(rows: int, cols: int, rng=None) -> np.ndarray:
    """Generate a realistic weight matrix matching LLM distributions."""
    if rng is None:
        rng = np.random.RandomState(42)
    t = rng.randn(rows, cols).astype(np.float32) * 0.02
    n_outliers = max(1, rows * cols // 500)
    outlier_rows = rng.randint(0, rows, n_outliers)
    outlier_cols = rng.randint(0, cols, n_outliers)
    t[outlier_rows, outlier_cols] *= rng.choice([-1, 1], n_outliers) * 15.0
    low_rank = (
        rng.randn(rows, min(16, rows)).astype(np.float32)
        @ rng.randn(min(16, rows), cols).astype(np.float32)
        * 0.3
    )
    t += low_rank
    return t


def per_stage_contribution(
    original: np.ndarray,
    cascade: FiveStageCascade,
    target_ratio: float,
) -> Dict[str, Any]:
    """Run cascade stages individually and measure each contribution."""
    m, n = original.shape
    total_el = m * n
    results: Dict[str, Any] = {}

    # Stage 1: EinSort permutation
    t1 = time.perf_counter()
    permuted, row_perm, col_perm = _einsort_stage1(original)
    results["s1_perm_time_s"] = time.perf_counter() - t1
    results["s1_row_perm_bytes"] = row_perm.nbytes
    results["s1_col_perm_bytes"] = col_perm.nbytes
    s1_recon = permuted
    s1_em = end_to_end_error(original, s1_recon)
    results["s1"] = {
        "rel_mse": s1_em.rel_mse,
        "cosine_sim": s1_em.cosine_sim,
        "snr_db": s1_em.snr_db,
        "perm_bytes": int(row_perm.nbytes + col_perm.nbytes),
    }

    # Stage 2: TT-SVD on permuted
    t2 = time.perf_counter()
    cores, residual_tt = _tt_svd_decompose(
        permuted, target_ratio, tt_rank=cascade.tt_rank, d=cascade.d
    )
    results["s2_time_s"] = time.perf_counter() - t2
    tt_el = sum(c.nbytes for c in cores)
    results["s2_cores_bytes"] = tt_el
    s2_recon = permuted - residual_tt
    s2_total = _inverse_permute_fn(s2_recon, row_perm, col_perm)
    s2_em = end_to_end_error(original, s2_total)
    results["s2"] = {
        "rel_mse": s2_em.rel_mse,
        "cosine_sim": s2_em.cosine_sim,
        "snr_db": s2_em.snr_db,
        "cores_bytes": int(tt_el),
        "residual_norm": float(np.linalg.norm(residual_tt)),
        "residual_std": float(np.std(residual_tt)),
    }

    # Stage 3: Sparse on TT residual
    t3 = time.perf_counter()
    sp_idx, sp_vals, sp_scale, residual_sparse = _sparse_residual_stage3(
        residual_tt,
        topk_ratio=cascade.sparse_topk_ratio,
        use_2_4=cascade.use_2_4_sparsity,
    )
    results["s3_time_s"] = time.perf_counter() - t3
    s3_bytes = int(sp_idx.nbytes + sp_vals.nbytes + 4)
    results["s3_sparse_bytes"] = s3_bytes
    s3_recon = s2_recon + (residual_tt - residual_sparse)
    s3_total = _inverse_permute_fn(s3_recon, row_perm, col_perm)
    s3_em = end_to_end_error(original, s3_total)
    results["s3"] = {
        "rel_mse": s3_em.rel_mse,
        "cosine_sim": s3_em.cosine_sim,
        "snr_db": s3_em.snr_db,
        "nnz": int(len(sp_idx)),
        "sparsity": float(len(sp_idx) / total_el),
        "sparse_bytes": s3_bytes,
        "residual_norm": float(np.linalg.norm(residual_sparse)),
        "residual_std": float(np.std(residual_sparse)),
    }

    # Stage 4: Ergodic on sparse residual
    t4 = time.perf_counter()
    alphas, A, phi, bias, residual_ergodic = _ergodic_trajectory_stage4(
        residual_sparse, n_channels=cascade.ergodic_n_channels
    )
    results["s4_time_s"] = time.perf_counter() - t4
    s4_bytes = int(alphas.nbytes + A.nbytes + phi.nbytes + bias.nbytes)
    results["s4_ergodic_bytes"] = s4_bytes
    s4_recon = s3_recon + (residual_sparse - residual_ergodic)
    s4_total = _inverse_permute_fn(s4_recon, row_perm, col_perm)
    s4_em = end_to_end_error(original, s4_total)
    results["s4"] = {
        "rel_mse": s4_em.rel_mse,
        "cosine_sim": s4_em.cosine_sim,
        "snr_db": s4_em.snr_db,
        "n_channels": int(cascade.ergodic_n_channels),
        "ergodic_bytes": s4_bytes,
        "residual_norm": float(np.linalg.norm(residual_ergodic)),
        "residual_std": float(np.std(residual_ergodic)),
    }

    # Stage 5: SIREN on ergodic residual
    t5 = time.perf_counter()
    w1, b1, wo, bo = _siren_fit_2d(
        residual_ergodic,
        permuted.shape,
        hidden_dim=cascade.siren_hidden_dim,
        n_epochs=cascade.siren_n_epochs,
    )
    results["s5_time_s"] = time.perf_counter() - t5
    s5_bytes = int(w1.nbytes + b1.nbytes + wo.nbytes + 4)
    results["s5_siren_bytes"] = s5_bytes
    s5_recon = s4_recon + (residual_ergodic - np.zeros_like(residual_ergodic))
    s5_total = _inverse_permute_fn(s5_recon, row_perm, col_perm)
    s5_em = end_to_end_error(original, s5_total)
    results["s5"] = {
        "rel_mse": s5_em.rel_mse,
        "cosine_sim": s5_em.cosine_sim,
        "snr_db": s5_em.snr_db,
        "hidden_dim": int(cascade.siren_hidden_dim),
        "n_epochs": cascade.siren_n_epochs,
        "siren_bytes": s5_bytes,
    }

    total_bytes = int(
        row_perm.nbytes + col_perm.nbytes + tt_el + s3_bytes + s4_bytes + s5_bytes
    )
    results["total_stored_bytes"] = total_bytes
    results["stage_breakdown"] = {
        "s1_perm": int(row_perm.nbytes + col_perm.nbytes),
        "s2_tt_cores": int(tt_el),
        "s3_sparse": s3_bytes,
        "s4_ergodic": s4_bytes,
        "s5_siren": s5_bytes,
    }
    stage_total = sum(results["stage_breakdown"].values())
    results["stage_pct"] = {
        k: round(v / max(stage_total, 1) * 100, 2)
        for k, v in results["stage_breakdown"].items()
    }
    results["total_time_s"] = sum(
        results.get(k, 0)
        for k in ["s1_perm_time_s", "s2_time_s", "s3_time_s", "s4_time_s", "s5_time_s"]
    )
    return results


def _inverse_permute_fn(matrix, row_perm, col_perm):
    inv_row = np.argsort(row_perm)
    inv_col = np.argsort(col_perm)
    return matrix[inv_row, :][:, inv_col]


def process_tensor(
    name: str,
    tensor: np.ndarray,
    target_ratio: float,
    cascade_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "tensor_name": name,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "n_elements": int(tensor.size),
        "fp32_bytes": int(tensor.size * FP32_BYTES),
        "bf16_bytes": int(tensor.size * BF16_BYTES),
    }

    if tensor.ndim == 1:
        result["skipped"] = "1D tensor"
        return result
    if tensor.ndim != 2:
        result["skipped"] = f"{tensor.ndim}D tensor, not 2D"
        return result
    if tensor.shape[0] < 8 or tensor.shape[1] < 8:
        result["skipped"] = "too small"
        return result

    tensor_f32 = np.asarray(tensor, dtype=np.float32)
    cascade = FiveStageCascade(**cascade_kwargs)

    # Per-stage contribution (more detailed)
    t_full_start = time.perf_counter()
    try:
        contrib = per_stage_contribution(tensor_f32, cascade, target_ratio)
        result["per_stage"] = contrib
    except Exception as e:
        result["per_stage_error"] = str(e)
        contrib = {}

    # Full end-to-end
    t_compress = time.perf_counter()
    try:
        payload, metadata = compress_cascade(
            tensor_f32, target_ratio=target_ratio, **cascade_kwargs
        )
        compress_time = time.perf_counter() - t_compress
        result["compress_time_s"] = compress_time

        serialized = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        comp_bytes = len(serialized)
        result["compressed_bytes_pickle"] = comp_bytes
        result["serialized_nbytes"] = serialized_nbytes(payload)

        dr = dual_ratio(tensor_f32.size, payload)
        result["ratio_vs_fp32"] = dr["ratio_vs_fp32"]
        result["ratio_vs_bf16"] = dr["ratio_vs_bf16"]

        t_decompress = time.perf_counter()
        reconstructed = decompress_cascade(payload, metadata)
        decompress_time = time.perf_counter() - t_decompress
        result["decompress_time_s"] = decompress_time

        em = end_to_end_error(tensor_f32, reconstructed)
        result["rel_mse"] = em.rel_mse
        result["cosine_sim"] = em.cosine_sim
        result["max_abs_error"] = em.max_abs
        result["snr_db"] = em.snr_db

        result["total_time_s"] = compress_time + decompress_time
    except Exception as e:
        result["e2e_error"] = str(e)

    result["status"] = "ok" if "rel_mse" in result else "error"
    return result


def load_best_model() -> Tuple[Dict[str, np.ndarray], str]:
    paths_to_try = [
        "models/gemma-4-E2B/model.safetensors",
    ]
    for p in paths_to_try:
        if Path(p).exists():
            print(f"Loading model: {p}")
            tensors = load_safetensors_bf16(p)
            return tensors, p
    return {}, ""


def main():
    target_ratio = 200.0
    max_tensors = 5
    cascade_kwargs = {
        "tt_rank": 16,
        "sparse_topk_ratio": 0.01,
        "ergodic_n_channels": 16,
        "siren_hidden_dim": 32,
        "siren_n_epochs": 100,
        "d": 3,
        "use_2_4_sparsity": False,
    }

    print("=" * 80)
    print("5-STAGE CASCADE COMPRESSION — REAL MODEL WEIGHTS")
    print("=" * 80)

    # Try loading real model
    tensors, model_path = load_best_model()
    use_synthetic = len(tensors) == 0

    if use_synthetic:
        print("\nNo real model found. Generating synthetic LLM-weight tensors...")
        shapes = [
            ("synthetic_linear_4096x4096", (4096, 4096)),
            ("synthetic_linear_4096x1024", (4096, 1024)),
            ("synthetic_linear_1024x4096", (1024, 4096)),
            ("synthetic_linear_2048x2048", (2048, 2048)),
            ("synthetic_linear_8192x1024", (8192, 1024)),
        ]
        rng = np.random.RandomState(42)
        for name, shape in shapes:
            tensors[name] = synthetic_llm_tensor(shape[0], shape[1], rng)
        source_desc = "synthetic"
    else:
        # Filter to 2D weight matrices
        targets = {}
        for k, t in tensors.items():
            if t.ndim == 2 and t.shape[0] > 32 and t.shape[1] > 32:
                targets[k] = t
        tensors = targets
        source_desc = model_path
        print(f"\nFound {len(tensors)} 2D weight matrices")

    # Filter to medium-sized 2D matrices (2K-2M elements, manageable for SVD)
    tensors = {
        k: v
        for k, v in tensors.items()
        if 1000 < v.size < 2_000_000 and max(v.shape) < 5000
    }
    print(f"After size filter (1K-2M elements, max_dim<5000): {len(tensors)} tensors")

    # Sort by size (largest first)
    sorted_names = sorted(tensors.keys(), key=lambda k: -tensors[k].size)
    selected = sorted_names[:max_tensors]
    print(f"Processing {len(selected)} tensors (largest first)\n")

    all_results: List[Dict[str, Any]] = []
    agg_fp32_ratios: List[float] = []
    agg_bf16_ratios: List[float] = []
    agg_rel_mses: List[float] = []
    agg_cosine_sims: List[float] = []
    agg_snrs: List[float] = []
    total_compress_time = 0.0
    total_decompress_time = 0.0

    for i, name in enumerate(selected):
        tensor = tensors[name]
        print(
            f"[{i + 1}/{len(selected)}] {name}  shape={tensor.shape}  size={tensor.size * 4 / 1e6:.1f}MB(fp32)"
        )
        sys.stdout.flush()

        result = process_tensor(name, tensor, target_ratio, cascade_kwargs)

        if result.get("status") == "ok":
            print(
                f"  Ratio fp32: {result['ratio_vs_fp32']:.1f}:1  bf16: {result['ratio_vs_bf16']:.1f}:1"
            )
            print(
                f"  rel_mse={result['rel_mse']:.6e}  cos_sim={result['cosine_sim']:.6f}  SNR={result['snr_db']:.1f}dB"
            )
            print(
                f"  time: {result.get('compress_time_s', 0):.2f}s + {result.get('decompress_time_s', 0):.2f}s"
            )

            if "per_stage" in result:
                ps = result["per_stage"]
                stg = ps.get("stage_breakdown", {})
                pct = ps.get("stage_pct", {})
                print(
                    f"  Stage breakdown (bytes): s1_perm={stg.get('s1_perm', 0)}  s2_tt={stg.get('s2_tt_cores', 0)}  s3_sparse={stg.get('s3_sparse', 0)}  s4_ergodic={stg.get('s4_ergodic', 0)}  s5_siren={stg.get('s5_siren', 0)}"
                )
                print(f"  Stage pct: {pct}")
                if "s2" in ps:
                    print(
                        f"  S2 resid_norm={ps['s2']['residual_norm']:.4f}  resid_std={ps['s2']['residual_std']:.4f}"
                    )
                if "s3" in ps:
                    print(
                        f"  S3 sparsity={ps['s3']['sparsity']:.6f}  resid_norm={ps['s3']['residual_norm']:.4f}"
                    )
                if "s4" in ps:
                    print(
                        f"  S4 resid_norm={ps['s4']['residual_norm']:.4f}  resid_std={ps['s4']['residual_std']:.4f}"
                    )

            agg_fp32_ratios.append(result["ratio_vs_fp32"])
            agg_bf16_ratios.append(result["ratio_vs_bf16"])
            agg_rel_mses.append(result["rel_mse"])
            agg_cosine_sims.append(result["cosine_sim"])
            agg_snrs.append(result["snr_db"])
            total_compress_time += result.get("compress_time_s", 0)
            total_decompress_time += result.get("decompress_time_s", 0)
        else:
            reason = (
                result.get("skipped")
                or result.get("e2e_error")
                or result.get("per_stage_error")
                or "unknown"
            )
            print(f"  SKIPPED: {reason}")

        all_results.append(result)
        print()

    # Aggregates
    print("=" * 80)
    print("AGGREGATE STATISTICS")
    print("=" * 80)
    print(f"Source: {source_desc}")
    print(
        f"Tensors processed: {len([r for r in all_results if r.get('status') == 'ok'])}/{len(all_results)}"
    )

    if agg_fp32_ratios:
        print(f"\nCompression ratio vs FP32:")
        print(f"  Mean: {np.mean(agg_fp32_ratios):.1f}:1")
        print(f"  Median: {np.median(agg_fp32_ratios):.1f}:1")
        print(f"  Min: {min(agg_fp32_ratios):.1f}:1")
        print(f"  Max: {max(agg_fp32_ratios):.1f}:1")

        print(f"\nCompression ratio vs BF16:")
        print(f"  Mean: {np.mean(agg_bf16_ratios):.1f}:1")
        print(f"  Median: {np.median(agg_bf16_ratios):.1f}:1")
        print(f"  Min: {min(agg_bf16_ratios):.1f}:1")
        print(f"  Max: {max(agg_bf16_ratios):.1f}:1")

        print(f"\nError metrics:")
        print(
            f"  rel_mse — Mean: {np.mean(agg_rel_mses):.6e}  Median: {np.median(agg_rel_mses):.6e}"
        )
        print(
            f"  cosine_sim — Mean: {np.mean(agg_cosine_sims):.6f}  Median: {np.median(agg_cosine_sims):.6f}"
        )
        print(
            f"  SNR (dB) — Mean: {np.mean(agg_snrs):.1f}  Median: {np.median(agg_snrs):.1f}"
        )

        print(f"\nTiming:")
        print(f"  Total compress time: {total_compress_time:.2f}s")
        print(f"  Total decompress time: {total_decompress_time:.2f}s")
        print(f"  Total wall time: {total_compress_time + total_decompress_time:.2f}s")

    # Save results
    output = {
        "source": source_desc,
        "config": cascade_kwargs,
        "target_ratio": target_ratio,
        "results": all_results,
        "aggregate": {
            "n_processed": len(agg_fp32_ratios),
            "mean_ratio_fp32": float(np.mean(agg_fp32_ratios))
            if agg_fp32_ratios
            else 0,
            "median_ratio_fp32": float(np.median(agg_fp32_ratios))
            if agg_fp32_ratios
            else 0,
            "mean_ratio_bf16": float(np.mean(agg_bf16_ratios))
            if agg_bf16_ratios
            else 0,
            "median_ratio_bf16": float(np.median(agg_bf16_ratios))
            if agg_bf16_ratios
            else 0,
            "mean_rel_mse": float(np.mean(agg_rel_mses)) if agg_rel_mses else 0,
            "median_rel_mse": float(np.median(agg_rel_mses)) if agg_rel_mses else 0,
            "mean_cosine_sim": float(np.mean(agg_cosine_sims))
            if agg_cosine_sims
            else 0,
            "median_cosine_sim": float(np.median(agg_cosine_sims))
            if agg_cosine_sims
            else 0,
            "mean_snr_db": float(np.mean(agg_snrs)) if agg_snrs else 0,
            "median_snr_db": float(np.median(agg_snrs)) if agg_snrs else 0,
            "total_compress_time_s": total_compress_time,
            "total_decompress_time_s": total_decompress_time,
        },
    }

    out_path = "scripts/5stage_results.json"
    with open(out_path, "w") as f:
        # Convert numpy types for JSON
        class NpEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.integer,)):
                    return int(obj)
                if isinstance(obj, (np.floating,)):
                    return float(obj)
                if isinstance(obj, np.bool_):
                    return bool(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)

        json.dump(output, f, indent=2, cls=NpEncoder)

    print(f"\nResults saved to: {out_path}")
    return output


if __name__ == "__main__":
    import sys

    main()
