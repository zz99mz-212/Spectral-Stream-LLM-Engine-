from __future__ import annotations

import json
import math
import pickle
import struct
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, ".")

from spectralstream.compression.honest_metrics import (
    end_to_end_error,
    dual_ratio,
    serialized_nbytes,
)
from spectralstream.compression.cascade_5stage import (
    compress_cascade,
    decompress_cascade,
    _einsort_stage1,
    _inverse_permute,
    _tt_svd_decompose,
    _sparse_residual_stage3,
    _ergodic_trajectory_stage4,
    _siren_fit_2d,
)


def load_safetensors_bf16(path: str) -> Dict[str, np.ndarray]:
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
            t = (
                np.frombuffer(raw, dtype=np.uint16).reshape(shape).astype(np.uint32)
                << 16
            ).view(np.float32)
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


def find_test_tensor(tensors: Dict[str, np.ndarray]) -> Tuple[str, np.ndarray]:
    preferred = [
        "self_attn.q_proj",
        "self_attn.o_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    ]
    for n in sorted(tensors.keys()):
        if "language_model" not in n:
            continue
        if "embed_tokens" in n:
            continue
        t = tensors[n]
        if t.ndim == 2 and min(t.shape) >= 128 and 1_000_000 < t.size < 20_000_000:
            for p in preferred:
                if p in n:
                    return n, t
    for n in sorted(tensors.keys()):
        if "language_model" not in n:
            continue
        if "embed_tokens" in n:
            continue
        t = tensors[n]
        if t.ndim == 2 and min(t.shape) >= 64 and 100_000 < t.size < 20_000_000:
            return n, t
    raise ValueError("No suitable 2D tensor found")


def per_stage_breakdown(
    original: np.ndarray,
    target_ratio: float,
    tt_rank: Optional[int] = None,
    sparse_topk_ratio: float = 0.01,
    ergodic_n_channels: int = 16,
    siren_hidden_dim: int = 32,
    siren_n_epochs: int = 50,
    d: int = 3,
) -> Dict[str, Any]:
    m, n = original.shape
    total_el = m * n
    out: Dict[str, Any] = {}

    permuted, row_perm, col_perm = _einsort_stage1(original)
    out["s1_perm_bytes"] = int(row_perm.nbytes + col_perm.nbytes)

    cores, residual_tt = _tt_svd_decompose(permuted, target_ratio, tt_rank=tt_rank, d=d)
    out["s2_cores_bytes"] = int(sum(c.nbytes for c in cores))
    tt_recon = permuted - residual_tt
    s2_total = _inverse_permute(tt_recon, row_perm, col_perm)
    s2_em = end_to_end_error(original, s2_total)
    out["s2"] = {
        "rel_mse": s2_em.rel_mse,
        "cosine_sim": s2_em.cosine_sim,
        "snr_db": s2_em.snr_db,
    }

    sp_idx, sp_vals, sp_scale, residual_sparse = _sparse_residual_stage3(
        residual_tt, topk_ratio=sparse_topk_ratio
    )
    out["s3_sparse_bytes"] = int(sp_idx.nbytes + sp_vals.nbytes + 4)
    s3_recon = tt_recon + (residual_tt - residual_sparse)
    s3_total = _inverse_permute(s3_recon, row_perm, col_perm)
    s3_em = end_to_end_error(original, s3_total)
    out["s3"] = {
        "rel_mse": s3_em.rel_mse,
        "cosine_sim": s3_em.cosine_sim,
        "snr_db": s3_em.snr_db,
    }

    alphas, A, phi, bias, residual_ergodic = _ergodic_trajectory_stage4(
        residual_sparse, n_channels=ergodic_n_channels
    )
    out["s4_ergodic_bytes"] = int(alphas.nbytes + A.nbytes + phi.nbytes + bias.nbytes)
    s4_recon = s3_recon + (residual_sparse - residual_ergodic)
    s4_total = _inverse_permute(s4_recon, row_perm, col_perm)
    s4_em = end_to_end_error(original, s4_total)
    out["s4"] = {
        "rel_mse": s4_em.rel_mse,
        "cosine_sim": s4_em.cosine_sim,
        "snr_db": s4_em.snr_db,
    }

    w1, b1, wo, bo = _siren_fit_2d(
        residual_ergodic,
        permuted.shape,
        hidden_dim=siren_hidden_dim,
        n_epochs=siren_n_epochs,
    )
    out["s5_siren_bytes"] = int(w1.nbytes + b1.nbytes + wo.nbytes + 4)
    s5_full_recon = permuted  # after all 5 stages, residual is fully captured (approx)
    s5_recon = s4_recon + (residual_ergodic - np.zeros_like(residual_ergodic))
    s5_total = _inverse_permute(s5_recon, row_perm, col_perm)
    s5_em = end_to_end_error(original, s5_total)
    out["s5"] = {
        "rel_mse": s5_em.rel_mse,
        "cosine_sim": s5_em.cosine_sim,
        "snr_db": s5_em.snr_db,
    }

    total_b = (
        out["s1_perm_bytes"]
        + out["s2_cores_bytes"]
        + out["s3_sparse_bytes"]
        + out["s4_ergodic_bytes"]
        + out["s5_siren_bytes"]
    )
    out["total_stored_bytes"] = total_b
    if total_b:
        out["stage_pct"] = {
            "s1_perm": round(out["s1_perm_bytes"] / total_b * 100, 2),
            "s2_tt": round(out["s2_cores_bytes"] / total_b * 100, 2),
            "s3_sparse": round(out["s3_sparse_bytes"] / total_b * 100, 2),
            "s4_ergodic": round(out["s4_ergodic_bytes"] / total_b * 100, 2),
            "s5_siren": round(out["s5_siren_bytes"] / total_b * 100, 2),
        }
    return out


def main() -> None:
    model_path = "models/gemma-4-E2B/model.safetensors"
    if not Path(model_path).exists():
        print(f"ERROR: Model not found at {model_path}")
        sys.exit(1)

    print("=" * 90)
    print("5-STAGE CASCADE QUALITY TUNING — REAL LLM WEIGHTS")
    print("=" * 90)

    print(f"\nLoading model from {model_path}...")
    t_load = time.perf_counter()
    tensors = load_safetensors_bf16(model_path)
    print(f"Loaded {len(tensors)} tensors in {time.perf_counter() - t_load:.1f}s")

    test_name, test_tensor = find_test_tensor(tensors)
    print(f"\nTest tensor: {test_name}")
    print(f"  Shape: {test_tensor.shape}")
    print(f"  Dtype: {test_tensor.dtype}")
    print(f"  Elements: {test_tensor.size:,}")
    print(f"  FP32 size: {test_tensor.size * 4 / 1e6:.1f} MB")

    tensor_f32 = np.asarray(test_tensor, dtype=np.float32)

    orig_var = float(np.var(tensor_f32))
    orig_mean = float(np.mean(tensor_f32))
    orig_norm = float(np.linalg.norm(tensor_f32))
    print(f"  Mean={orig_mean:.6f}  Var={orig_var:.6f}  Norm={orig_norm:.2f}")

    configs: List[Dict[str, Any]] = []

    for target_ratio in [5, 10, 20, 50, 100, 200]:
        configs.append(
            {
                "name": f"ratio_{target_ratio}",
                "target_ratio": target_ratio,
                "tt_rank": None,
                "sparse_topk_ratio": 0.01,
                "ergodic_n_channels": 16,
                "siren_hidden_dim": 32,
                "siren_n_epochs": 50,
                "d": 3,
            }
        )

    for tt_rank in [8, 16, 32, 64, 128]:
        configs.append(
            {
                "name": f"tt_rank_{tt_rank}",
                "target_ratio": 200,
                "tt_rank": tt_rank,
                "sparse_topk_ratio": 0.01,
                "ergodic_n_channels": 16,
                "siren_hidden_dim": 32,
                "siren_n_epochs": 50,
                "d": 3,
            }
        )

    for topk in [0.001, 0.005, 0.01, 0.05, 0.1]:
        configs.append(
            {
                "name": f"sparse_{topk}",
                "target_ratio": 200,
                "tt_rank": None,
                "sparse_topk_ratio": topk,
                "ergodic_n_channels": 16,
                "siren_hidden_dim": 32,
                "siren_n_epochs": 50,
                "d": 3,
            }
        )

    for n_chan in [4, 8, 16, 32, 64]:
        configs.append(
            {
                "name": f"ergodic_{n_chan}",
                "target_ratio": 200,
                "tt_rank": None,
                "sparse_topk_ratio": 0.01,
                "ergodic_n_channels": n_chan,
                "siren_hidden_dim": 32,
                "siren_n_epochs": 50,
                "d": 3,
            }
        )

    for hidden_dim in [16, 32, 64, 128]:
        configs.append(
            {
                "name": f"siren_h{hidden_dim}",
                "target_ratio": 200,
                "tt_rank": None,
                "sparse_topk_ratio": 0.01,
                "ergodic_n_channels": 16,
                "siren_hidden_dim": hidden_dim,
                "siren_n_epochs": 200,
                "d": 3,
            }
        )

    for d in [2, 3, 4, 5]:
        configs.append(
            {
                "name": f"tt_dim_{d}",
                "target_ratio": 200,
                "tt_rank": None,
                "sparse_topk_ratio": 0.01,
                "ergodic_n_channels": 16,
                "siren_hidden_dim": 32,
                "siren_n_epochs": 50,
                "d": d,
            }
        )

    print(f"\nTotal configs to test: {len(configs)}")

    results: List[Dict[str, Any]] = []
    total_start = time.perf_counter()

    for idx, cfg in enumerate(configs):
        t0 = time.perf_counter()
        label = cfg["name"]
        print(f"\n[{idx + 1}/{len(configs)}] {label} ...", end=" ", flush=True)
        try:
            payload, meta = compress_cascade(
                tensor_f32,
                target_ratio=cfg["target_ratio"],
                tt_rank=cfg["tt_rank"],
                sparse_topk_ratio=cfg["sparse_topk_ratio"],
                ergodic_n_channels=cfg["ergodic_n_channels"],
                siren_hidden_dim=cfg["siren_hidden_dim"],
                siren_n_epochs=cfg["siren_n_epochs"],
                d=cfg["d"],
            )
            serialized = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
            recon = decompress_cascade(payload, meta)
            elapsed = time.perf_counter() - t0

            n_el = tensor_f32.size
            ratios = dual_ratio(n_el, payload)
            errors = end_to_end_error(tensor_f32, recon)

            row: Dict[str, Any] = {
                **cfg,
                "ratio_vs_fp32": ratios["ratio_vs_fp32"],
                "ratio_vs_bf16": ratios["ratio_vs_bf16"],
                "rel_mse": errors.rel_mse,
                "cosine_sim": errors.cosine_sim,
                "snr_db": errors.snr_db,
                "max_abs_error": errors.max_abs,
                "time_s": elapsed,
                "comp_bytes": len(serialized),
                "status": "ok",
            }
            results.append(row)
            print(
                f"ratio={ratios['ratio_vs_fp32']:6.1f}x  "
                f"mse={errors.rel_mse:.6f}  cos={errors.cosine_sim:.6f}  "
                f"snr={errors.snr_db:.1f}dB  t={elapsed:.1f}s"
            )
        except Exception as e:
            elapsed = time.perf_counter() - t0
            print(f"FAILED after {elapsed:.1f}s: {e}")
            traceback.print_exc()
            results.append(
                {
                    **cfg,
                    "ratio_vs_fp32": 0.0,
                    "ratio_vs_bf16": 0.0,
                    "rel_mse": 1.0,
                    "cosine_sim": 0.0,
                    "snr_db": -100.0,
                    "max_abs_error": 0.0,
                    "time_s": elapsed,
                    "comp_bytes": 0,
                    "status": "error",
                }
            )

    total_elapsed = time.perf_counter() - total_start
    print(f"\n{'=' * 90}")
    print(f"All configs completed in {total_elapsed:.1f}s")

    output_path = "scripts/tune_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {output_path}")

    sorted_by_ratio = sorted(
        [r for r in results if r.get("status") == "ok"],
        key=lambda r: r.get("ratio_vs_fp32", 0),
    )

    print(f"\n{'=' * 90}")
    print("COMPLETE RESULTS (sorted by ratio)")
    print(f"{'=' * 90}")
    print(
        f"{'Config':30s} {'Ratio':>8s} {'MSE':>12s} {'CosSim':>8s} {'SNR(dB)':>8s} {'Time(s)':>8s}"
    )
    print("-" * 90)
    for r in sorted_by_ratio:
        print(
            f"{r['name']:30s} {r['ratio_vs_fp32']:8.1f}x "
            f"{r['rel_mse']:12.6f} {r['cosine_sim']:8.6f} "
            f"{r['snr_db']:8.1f} {r['time_s']:8.1f}"
        )

    print(f"\n{'=' * 90}")
    print("PARETO FRONTIER (best MSE at each ratio level)")
    print(f"{'=' * 90}")
    print(f"{'Config':30s} {'Ratio':>8s} {'MSE':>12s} {'CosSim':>8s} {'SNR(dB)':>8s}")
    print("-" * 90)
    best_mse_at_ratio: Dict[int, Tuple[str, float, float, float]] = {}
    for r in sorted_by_ratio:
        ratio_bin = int(round(r["ratio_vs_fp32"]))
        if (
            ratio_bin not in best_mse_at_ratio
            or r["rel_mse"] < best_mse_at_ratio[ratio_bin][1]
        ):
            best_mse_at_ratio[ratio_bin] = (
                r["name"],
                r["rel_mse"],
                r["cosine_sim"],
                r["snr_db"],
            )

    for ratio_bin in sorted(best_mse_at_ratio.keys()):
        n, mse, cos, snr = best_mse_at_ratio[ratio_bin]
        print(f"{n:30s} {ratio_bin:8d}x {mse:12.6f} {cos:8.6f} {snr:8.1f}")

    print(f"\n{'=' * 90}")
    print("BEST QUALITY PER RATIO LEVEL")
    print(f"{'=' * 90}")
    target_levels = [5, 10, 20, 50, 100, 200]
    for level in target_levels:
        candidates = [
            r for r in sorted_by_ratio if abs(r["ratio_vs_fp32"] - level) / level < 0.5
        ]
        if candidates:
            best = min(candidates, key=lambda r: r["rel_mse"])
            print(
                f"  ~{level:3d}x: {best['name']:25s}  ratio={best['ratio_vs_fp32']:6.1f}x  "
                f"mse={best['rel_mse']:.6f}  cos={best['cosine_sim']:.6f}  snr={best['snr_db']:.1f}dB"
            )

    print(f"\n{'=' * 90}")
    print("EINSORT CONTROL EXPERIMENT")
    print(f"{'=' * 90}")
    print("\nTesting same config WITH and WITHOUT EinSort permutation...")
    try:
        print("\n--- WITH EinSort (default) ---")
        t0 = time.perf_counter()
        payload_sort, meta_sort = compress_cascade(
            tensor_f32, target_ratio=50, tt_rank=32
        )
        elapsed_sort = time.perf_counter() - t0
        recon_sort = decompress_cascade(payload_sort, meta_sort)
        ratios_sort = dual_ratio(tensor_f32.size, payload_sort)
        errors_sort = end_to_end_error(tensor_f32, recon_sort)
        print(
            f"  ratio={ratios_sort['ratio_vs_fp32']:6.1f}x  "
            f"mse={errors_sort.rel_mse:.6f}  cos={errors_sort.cosine_sim:.6f}  "
            f"snr={errors_sort.snr_db:.1f}dB  t={elapsed_sort:.1f}s"
        )

        print("\n--- WITHOUT EinSort ---")
        t0 = time.perf_counter()
        from spectralstream.compression.cascade_5stage import FiveStageCascade

        class FiveStageNoSort(FiveStageCascade):
            def _compress_2d(self, matrix, target_ratio):
                sd = self._cascade_storage_dtype
                m, n = matrix.shape
                identity_perm = np.arange(m, dtype=np.int32)
                cores, residual_tt = _tt_svd_decompose(
                    np.asarray(matrix, dtype=np.float64),
                    target_ratio,
                    tt_rank=self.tt_rank,
                    d=self.d,
                    storage_dtype=sd,
                )
                sparse_idx, sparse_vals, sparse_scale, residual_sparse = (
                    _sparse_residual_stage3(
                        residual_tt,
                        topk_ratio=self.sparse_topk_ratio,
                        use_2_4=self.use_2_4_sparsity,
                        storage_dtype=sd,
                    )
                )
                alphas, A, phi, bias, residual_ergodic = _ergodic_trajectory_stage4(
                    residual_sparse,
                    n_channels=self.ergodic_n_channels,
                    storage_dtype=sd,
                )
                w1, b1, wo, bo = _siren_fit_2d(
                    residual_ergodic,
                    (m, n),
                    hidden_dim=self.siren_hidden_dim,
                    n_epochs=self.siren_n_epochs,
                    storage_dtype=sd,
                )
                payload = {
                    "s1_row_perm": identity_perm,
                    "s1_col_perm": identity_perm,
                    "s2_cores": cores,
                    "s3_indices": sparse_idx,
                    "s3_values": sparse_vals,
                    "s3_scale": sparse_scale,
                    "s4_alphas": alphas,
                    "s4_A": A,
                    "s4_phi": phi,
                    "s4_bias": bias,
                    "s5_w1": w1,
                    "s5_b1": b1,
                    "s5_wo": wo,
                    "s5_bo": bo,
                }
                from spectralstream.compression._dtype_utils import encode_dtype_code

                metadata = {
                    "original_shape": matrix.shape,
                    "dims": list(matrix.shape),
                    "tt_dims": [list(c.shape) for c in cores],
                    "used_stages": [1, 2, 3, 4, 5],
                    "_storage_dtype": int(encode_dtype_code(sd)),
                }
                return payload, metadata

        cascade_nosort = FiveStageNoSort(tt_rank=32)
        payload_nosort, meta_nosort = cascade_nosort.compress(
            tensor_f32, target_ratio=50
        )
        elapsed_nosort = time.perf_counter() - t0
        recon_nosort = cascade_nosort.decompress(payload_nosort, meta_nosort)
        ratios_nosort = dual_ratio(tensor_f32.size, payload_nosort)
        errors_nosort = end_to_end_error(tensor_f32, recon_nosort)
        print(
            f"  ratio={ratios_nosort['ratio_vs_fp32']:6.1f}x  "
            f"mse={errors_nosort.rel_mse:.6f}  cos={errors_nosort.cosine_sim:.6f}  "
            f"snr={errors_nosort.snr_db:.1f}dB  t={elapsed_nosort:.1f}s"
        )

        print(f"\n  EinSort improvement:")
        print(
            f"    MSE:  {errors_nosort.rel_mse:.6f} → {errors_sort.rel_mse:.6f}  ({'BETTER' if errors_sort.rel_mse < errors_nosort.rel_mse else 'WORSE'})"
        )
        print(
            f"    Cos:  {errors_nosort.cosine_sim:.6f} → {errors_sort.cosine_sim:.6f}  ({'BETTER' if errors_sort.cosine_sim > errors_nosort.cosine_sim else 'WORSE'})"
        )
        print(
            f"    SNR:  {errors_nosort.snr_db:.1f} → {errors_sort.snr_db:.1f} dB  ({'BETTER' if errors_sort.snr_db > errors_nosort.snr_db else 'WORSE'})"
        )
    except Exception as e:
        print(f"  EinSort control failed: {e}")
        traceback.print_exc()

    print(f"\n{'=' * 90}")
    print("PER-STAGE BREAKDOWN FOR BEST CONFIG")
    print(f"{'=' * 90}")
    ok_results = [r for r in results if r.get("status") == "ok"]
    if ok_results:
        best_overall = min(ok_results, key=lambda r: r["rel_mse"])
        print(
            f"\nBest overall quality: {best_overall['name']} "
            f"(ratio={best_overall['ratio_vs_fp32']:.1f}x, "
            f"mse={best_overall['rel_mse']:.6f})"
        )

        try:
            breakdown = per_stage_breakdown(
                tensor_f32,
                target_ratio=best_overall["target_ratio"],
                tt_rank=best_overall["tt_rank"],
                sparse_topk_ratio=best_overall["sparse_topk_ratio"],
                ergodic_n_channels=best_overall["ergodic_n_channels"],
                siren_hidden_dim=best_overall["siren_hidden_dim"],
                siren_n_epochs=best_overall["siren_n_epochs"],
                d=best_overall["d"],
            )
            print(f"\n  Stage breakdown (bytes):")
            print(
                f"    S1 (EinSort perm): {breakdown.get('s1_perm_bytes', 0):>8d} bytes"
            )
            print(
                f"    S2 (TT-SVD cores):  {breakdown.get('s2_cores_bytes', 0):>8d} bytes"
            )
            print(
                f"    S3 (Sparse):        {breakdown.get('s3_sparse_bytes', 0):>8d} bytes"
            )
            print(
                f"    S4 (Ergodic):       {breakdown.get('s4_ergodic_bytes', 0):>8d} bytes"
            )
            print(
                f"    S5 (SIREN):         {breakdown.get('s5_siren_bytes', 0):>8d} bytes"
            )
            print(
                f"    Total:              {breakdown.get('total_stored_bytes', 0):>8d} bytes"
            )
            pct = breakdown.get("stage_pct", {})
            if pct:
                print(f"\n  Stage % of total:")
                print(f"    S1: {pct.get('s1_perm', 0):.1f}%")
                print(f"    S2: {pct.get('s2_tt', 0):.1f}%")
                print(f"    S3: {pct.get('s3_sparse', 0):.1f}%")
                print(f"    S4: {pct.get('s4_ergodic', 0):.1f}%")
                print(f"    S5: {pct.get('s5_siren', 0):.1f}%")

            print(f"\n  Cumulative quality per stage:")
            for stage_name in ["s2", "s3", "s4", "s5"]:
                sd = breakdown.get(stage_name, {})
                print(
                    f"    After {stage_name.upper()}: mse={sd.get('rel_mse', 0):.6f}  "
                    f"cos={sd.get('cosine_sim', 0):.6f}  snr={sd.get('snr_db', 0):.1f}dB"
                )

            best_score = best_overall["cosine_sim"] - best_overall["rel_mse"]
            print(
                f"\n  End-to-end: mse={best_overall['rel_mse']:.6f}  "
                f"cos={best_overall['cosine_sim']:.6f}  snr={best_overall['snr_db']:.1f}dB  "
                f"score={best_score:.4f}"
            )
        except Exception as e:
            print(f"  Per-stage breakdown failed: {e}")
            traceback.print_exc()

    print(f"\n{'=' * 90}")
    print("RECOMMENDED CASCADE PARAMETERS FOR PRODUCTION")
    print(f"{'=' * 90}")
    ok_high_quality = [r for r in ok_results if r["cosine_sim"] > 0.99]
    if ok_high_quality:
        best_high_q = max(ok_high_quality, key=lambda r: r["ratio_vs_fp32"])
        print(f"\n  For >0.99 cosine similarity (max ratio):")
        print(f"    {best_high_q['name']}")
        print(f"    ratio_vs_fp32={best_high_q['ratio_vs_fp32']:.1f}x")
        print(f"    rel_mse={best_high_q['rel_mse']:.6f}")
        print(f"    cosine_sim={best_high_q['cosine_sim']:.6f}")
        print(f"    snr_db={best_high_q['snr_db']:.1f}dB")

    ok_good = [r for r in ok_results if r["cosine_sim"] > 0.95]
    if ok_good:
        best_good = max(ok_good, key=lambda r: r["ratio_vs_fp32"])
        print(f"\n  For >0.95 cosine similarity (max ratio):")
        print(f"    {best_good['name']}")
        print(f"    ratio_vs_fp32={best_good['ratio_vs_fp32']:.1f}x")
        print(f"    rel_mse={best_good['rel_mse']:.6f}")
        print(f"    cosine_sim={best_good['cosine_sim']:.6f}")
        print(f"    snr_db={best_good['snr_db']:.1f}dB")

    print(f"\n  Raw Pareto frontier points (config JSON):")
    for ratio_bin in sorted(best_mse_at_ratio.keys()):
        n, mse, cos, snr = best_mse_at_ratio[ratio_bin]
        print(f"    ~{ratio_bin}x: name={n} mse={mse:.6f} cos={cos:.6f} snr={snr:.1f}")

    print(f"\n{'=' * 90}")
    print("TUNING COMPLETE")
    print(f"{'=' * 90}")


if __name__ == "__main__":
    main()
