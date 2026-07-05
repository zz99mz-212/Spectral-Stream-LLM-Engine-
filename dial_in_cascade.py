"""
DIAL IN: Multiplicative Cascade Compression on Real Gemma-4 Weights
Targets: 500:1, 1200:1, 5000:1
Cascade chain: Decomposition → Spectral → Structural → Entropy → Quantization (last resort)
Memory-safe: max 2048x2048 float32 (16MB), gc.collect() between tests, no spectral_angle on large.
"""

import gc
import sys
import time
import json
import math
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict

import numpy as np

sys.path.insert(0, ".")

from spectralstream.compression.engine import (
    CompressionIntelligenceEngine,
    CompressionConfig,
    MemoryMappedTensorEngine,
)
from spectralstream.compression.engine.dynamic_tuning.multiplicative_stacking import (
    MultiplicativeStackingEngine,
    StackingPlan,
    StackingStage,
)

# ── Config ──────────────────────────────────────────────────────────────
MODEL_PATH = (
    "/home/mike/Documents/Github/SpectralStream/models/gemma-4-E2B/model.safetensors"
)
SAMPLE_SIZE = 2048
TARGETS = [500, 1200, 5000]
MAX_ERROR = 0.01
np.random.seed(42)


# ── Fast quality metrics (avoids spectral_angle ~128 TiB for 4M elements) ──


def fast_assess(orig: np.ndarray, recon: np.ndarray) -> Dict[str, float]:
    """Compute core quality metrics without spectral_angle (O(n) not O(n^2))."""
    o = orig.ravel().astype(np.float64)
    r = recon.ravel().astype(np.float64)
    diff = o - r
    mse = float(np.mean(diff**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(diff)))
    orig_norm = float(np.linalg.norm(o))
    rel_err = float(np.linalg.norm(diff) / (orig_norm + 1e-30))
    snr = float(20 * np.log10(orig_norm / (np.linalg.norm(diff) + 1e-30)))
    cos_sim = float(np.dot(o, r) / (np.linalg.norm(o) * np.linalg.norm(r) + 1e-30))
    max_abs = float(np.max(np.abs(diff)))
    corr = float(np.corrcoef(o, r)[0, 1]) if len(o) > 1 else 0.0
    grade = (
        "S"
        if rel_err < 0.0002
        else "A"
        if rel_err < 0.001
        else "B"
        if rel_err < 0.005
        else "C"
        if rel_err < 0.01
        else "D"
        if rel_err < 0.05
        else "F"
    )
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "relative_error": rel_err,
        "snr_db": snr,
        "cosine_similarity": cos_sim,
        "max_abs_error": max_abs,
        "correlation": corr,
        "grade": grade,
    }


# ── Memory-Safe Sampling ─────────────────────────────────────────────────


def sample_real_weights() -> Tuple[np.ndarray, str, int]:
    mmap = MemoryMappedTensorEngine(MODEL_PATH)
    try:
        big_tensors = [(n, mmap.get_nbytes(n)) for n in mmap.get_tensor_names()]
        big_tensors.sort(key=lambda x: -x[1])

        print(f"Model: {MODEL_PATH}")
        print(f"Total tensors: {len(big_tensors)}")
        print("Top 10 tensors:")
        for name, nb in big_tensors[:10]:
            print(f"  {name}: {nb:,} bytes ({nb / 1e6:.1f} MB)")

        # Pick the largest 2D+ weight matrix
        for name, nb in big_tensors:
            view = mmap.get_tensor(name)
            if view.ndim >= 2:
                size0 = min(SAMPLE_SIZE, view.shape[0])
                size1 = min(SAMPLE_SIZE, view.shape[-1])
                weight_sample = np.array(view[:size0, :size1], dtype=np.float32)
                print(
                    f"\nSampled: {name} shape={view.shape} → {weight_sample.shape} ({weight_sample.nbytes / 1e6:.1f} MB)"
                )
                del view
                gc.collect()
                mmap.close()
                return weight_sample, name, nb

        name, nb = big_tensors[0]
        view = mmap.get_tensor(name)
        size0 = min(SAMPLE_SIZE * SAMPLE_SIZE, view.shape[0])
        weight_sample = np.array(view[:size0], dtype=np.float32)
        print(
            f"\nSampled (1D): {name} shape={view.shape} → {weight_sample.shape} ({weight_sample.nbytes / 1e6:.1f} MB)"
        )
        del view
        gc.collect()
        mmap.close()
        return weight_sample, name, nb
    except Exception as e:
        mmap.close()
        raise e


# ── Individual Tier Tests ───────────────────────────────────────────────


def test_individual_tiers(tensor: np.ndarray) -> Dict[str, Any]:
    print("\n" + "=" * 72)
    print("PHASE 1: INDIVIDUAL TIER TESTS ON REAL WEIGHTS")
    print("=" * 72)

    results = {}
    config = CompressionConfig(target_ratio=5000.0, max_error=MAX_ERROR)
    engine = CompressionIntelligenceEngine(config=config)
    try:
        tier_methods = {
            "decomposition": ["svd_compress"],
            "spectral": ["dct_spectral", "fwht_compress"],
            "quantization": [
                "block_int8",
                "block_int4",
                "hadamard_int8",
                "hadamard_int4",
            ],
        }

        for tier_name, method_names in tier_methods.items():
            if method_names is None:
                continue
            for mname in method_names:
                inst = engine._methods.get(mname)
                if not inst:
                    continue
                try:
                    t0 = time.time()
                    if mname == "svd_compress":
                        rank = max(4, min(128, tensor.shape[0] // 50))
                        data, meta = inst.compress(tensor, rank=rank)
                    elif "dct" in mname:
                        keep = max(0.01, 1.0 / 50.0)
                        data, meta = inst.compress(tensor, keep_ratio=keep)
                    elif "fwht" in mname:
                        data, meta = inst.compress(tensor)
                    elif "int4" in mname:
                        data, meta = inst.compress(tensor, block_size=64)
                    elif "int8" in mname:
                        data, meta = inst.compress(tensor, block_size=256)
                    else:
                        data, meta = inst.compress(tensor)
                    t1 = time.time()

                    recon = inst.decompress(data, meta)
                    q = fast_assess(tensor, recon)
                    ratio = tensor.nbytes / max(len(data), 1)
                    results[mname] = {
                        "tier": tier_name,
                        "ratio": ratio,
                        "error": q["relative_error"],
                        "snr": q["snr_db"],
                        "grade": q["grade"],
                        "time_s": t1 - t0,
                    }
                    print(
                        f"  {mname:25s} | ratio={ratio:>8.1f}:1 | error={q['relative_error']:.6f} | "
                        f"SNR={q['snr_db']:6.1f}dB | grade={q['grade']} | {t1 - t0:.3f}s"
                    )
                except Exception as e:
                    err_str = str(e)[:80]
                    print(f"  {mname:25s} | FAILED: {err_str}")
                gc.collect()
        return results
    finally:
        engine.close()
        gc.collect()


@dataclass
class StageDiagnostic:
    stage_index: int
    method_type: str
    method_name: str
    sub_ratio: float
    sub_error: float
    cumulative_ratio: float
    cumulative_error: float
    residual_norm: float
    time_s: float


@dataclass
class CascadeDiagnostic:
    target: float
    actual_ratio: float
    actual_error: float
    snr_db: float
    grade: str
    n_stages: int
    stages: List[StageDiagnostic]
    status: str
    total_time: float = 0.0


# ── Cascade Stage-by-Stage ──────────────────────────────────────────────


def diagnose_cascade_stages(
    tensor: np.ndarray,
    target_ratio: float,
    max_error: float,
    tensor_name: str = "",
) -> Optional[CascadeDiagnostic]:
    t_start = time.time()
    config = CompressionConfig(target_ratio=target_ratio, max_error=max_error)
    engine = CompressionIntelligenceEngine(config=config)
    try:
        stacking = MultiplicativeStackingEngine(engine)
        stages_config = stacking.build_cascade_config(target_ratio)

        if not stages_config:
            print(f"  [SKIP] No cascade config for target {target_ratio}")
            return None

        print(
            f"\n  Stages ({len(stages_config)}): {[s['method_type'] for s in stages_config]}"
        )

        plan = StackingPlan(tensor_name=tensor_name)
        reconstructed = np.zeros_like(tensor, dtype=np.float32)
        tensor_f32 = tensor.astype(np.float32)
        stage_diagnostics = []
        total_stage_time = 0.0

        for i, stage_config in enumerate(stages_config):
            method_type = stage_config["method_type"]
            t_stage_start = time.time()

            residual = tensor_f32 - reconstructed

            best, candidates = stacking.select_method_for_stage(
                residual,
                method_type,
                previous_methods=[s.method_name for s in plan.stages],
            )

            method_name = None
            method_inst = None
            if best and best.get("method"):
                method_name = best["method"]
                method_inst = engine._methods.get(method_name)

            if not method_name or not method_inst:
                method_name, method_inst = stacking._get_method(method_type)

            if not method_name or not method_inst:
                print(f"    Stage {i + 1} ({method_type}): NO METHOD — skip")
                continue

            n_stages = len(stages_config)
            sub_ratio_target = max(1.5, target_ratio ** (1.0 / max(n_stages, 1)))
            stage_target_error = max_error / max(n_stages - i, 1)
            tuned_params = stacking._tune_stage_params(
                method_inst, method_name, residual, sub_ratio_target, stage_target_error
            )

            config_params = stage_config.get("params", {})
            for k, v in config_params.items():
                tuned_params.setdefault(k, v)

            # SVD always needs rank — compute from rank_frac
            if method_type == "decomposition" and "rank_frac" in config_params:
                rf = config_params["rank_frac"]
                m, n_ = (
                    residual.shape[0],
                    max(1, np.prod(residual.shape[1:]) if residual.ndim > 1 else 1),
                )
                k = int(m * n_ / (sub_ratio_target * (m + n_ + 1)))
                k = max(1, min(k, min(m, n_)))
                tuned_params["rank"] = k

            try:
                compressed, meta = method_inst.compress(residual, **tuned_params)
            except Exception:
                compressed, meta = method_inst.compress(residual)

            try:
                decompressed = method_inst.decompress(compressed, meta)
                if decompressed.shape != residual.shape:
                    decompressed = decompressed.reshape(residual.shape)

                stage_ratio = residual.nbytes / max(len(compressed), 1)
                reconstructed += decompressed

                cumulative_error = float(
                    np.linalg.norm((tensor_f32 - reconstructed).ravel())
                    / (np.linalg.norm(tensor_f32.ravel()) + 1e-30)
                )

                total_compressed = sum(
                    len(s.compressed_data) for s in plan.stages if s.compressed_data
                ) + len(compressed)
                cum_ratio = tensor.nbytes / max(total_compressed, 1)
                residual_norm = float(
                    np.linalg.norm((tensor_f32 - reconstructed).ravel())
                )

                t_stage_end = time.time()
                stage_time = t_stage_end - t_stage_start
                total_stage_time += stage_time

                # Sub-error relative to original (not residual)
                sub_error = cumulative_error

                diag = StageDiagnostic(
                    stage_index=i + 1,
                    method_type=method_type,
                    method_name=method_name,
                    sub_ratio=stage_ratio,
                    sub_error=sub_error,
                    cumulative_ratio=cum_ratio,
                    cumulative_error=cumulative_error,
                    residual_norm=residual_norm,
                    time_s=stage_time,
                )
                stage_diagnostics.append(diag)

                print(
                    f"    Stage {i + 1}: {method_name:20s} | "
                    f"sub_ratio={stage_ratio:>8.1f}:1 | "
                    f"cum_ratio={cum_ratio:>8.1f}:1 | "
                    f"cum_err={cumulative_error:.6f} | "
                    f"resid_norm={residual_norm:.4f} | {stage_time:.3f}s"
                )

                plan.stages.append(
                    StackingStage(
                        method_name=method_name,
                        category=getattr(method_inst, "category", ""),
                        tier=0,
                        params=tuned_params,
                        sub_ratio=stage_ratio,
                        sub_error=sub_error,
                        compressed_data=compressed,
                        metadata=meta,
                    )
                )

                if cumulative_error <= max_error:
                    print(f"    → Quality gate PASSED after stage {i + 1}")
                    break

            except Exception as e:
                err_str = str(e)[:80]
                print(f"    Stage {i + 1} ({method_type}): FAILED — {err_str}")
                continue

        q = fast_assess(tensor_f32, reconstructed)
        total_time = time.time() - t_start
        final_ratio = tensor.nbytes / max(
            sum(len(s.compressed_data) for s in plan.stages if s.compressed_data), 1
        )
        status = "MET" if q["relative_error"] <= max_error else "FAIL"

        print(
            f"\n  → FINAL: ratio={final_ratio:.1f}:1, error={q['relative_error']:.6f}, "
            f"SNR={q['snr_db']:.1f}dB, grade={q['grade']} [{status}] in {total_time:.2f}s"
        )

        return CascadeDiagnostic(
            target=target_ratio,
            actual_ratio=final_ratio,
            actual_error=q["relative_error"],
            snr_db=q["snr_db"],
            grade=q["grade"],
            n_stages=len(stage_diagnostics),
            stages=stage_diagnostics,
            status=status,
            total_time=total_time,
        )
    finally:
        engine.close()
        gc.collect()


# ── Parameter Sweep ──────────────────────────────────────────────────────


def tune_cascade_parameters(
    tensor: np.ndarray, target_ratio: float, tensor_name: str = ""
) -> Dict[str, Any]:
    print(f"\n{'=' * 72}")
    print(f"PHASE 3: PARAMETER SWEEP FOR TARGET={target_ratio}:1")
    print("=" * 72)

    results = []
    best_result = None

    rank_fracs = [0.005, 0.01, 0.02, 0.04]
    keep_fracs = [0.05, 0.1, 0.2, 0.3]
    quant_bits = [4, 8]
    struct_block_sizes = [64, 128]

    configs_tested = 0
    for rf in rank_fracs:
        for kf in keep_fracs:
            stages = [
                {"method_type": "decomposition", "params": {"rank_frac": rf}},
                {"method_type": "spectral", "params": {"keep_frac": kf}},
            ]
            if target_ratio >= 2000:
                stages.append(
                    {
                        "method_type": "structural",
                        "params": {"block_size": np.random.choice(struct_block_sizes)},
                    }
                )
            stages.append({"method_type": "entropy", "params": {"method": "rans"}})
            if target_ratio >= 3000:
                stages.append(
                    {
                        "method_type": "quantization",
                        "params": {"bits": np.random.choice(quant_bits)},
                    }
                )

            config = CompressionConfig(target_ratio=target_ratio, max_error=MAX_ERROR)
            engine = CompressionIntelligenceEngine(config=config)
            try:
                stacking = MultiplicativeStackingEngine(engine)
                plan = stacking._plan_from_config(tensor, stages, tensor_name)
                if plan is None or plan.total_ratio < 1.0:
                    engine.close()
                    gc.collect()
                    continue

                compressed, meta = stacking.execute_stacking(plan, tensor)
                try:
                    recon = stacking.unstack(compressed, meta, tensor.shape)
                except Exception:
                    engine.close()
                    gc.collect()
                    continue

                q = fast_assess(tensor, recon)
                actual_ratio = tensor.nbytes / max(len(compressed), 1)

                error_penalty = (
                    0.0
                    if q["relative_error"] <= MAX_ERROR
                    else 10.0 * (q["relative_error"] / MAX_ERROR)
                )
                score = (
                    actual_ratio / max(q["relative_error"], 1e-10)
                    - error_penalty * 1000
                )

                result = {
                    "rank_frac": rf,
                    "keep_frac": kf,
                    "n_stages": len(stages),
                    "actual_ratio": actual_ratio,
                    "actual_error": q["relative_error"],
                    "snr": q["snr_db"],
                    "grade": q["grade"],
                    "score": score,
                    "status": "MET" if q["relative_error"] <= MAX_ERROR else "FAIL",
                    "stages": [s["method_type"] for s in stages],
                }
                results.append(result)
                if best_result is None or score > best_result["score"]:
                    best_result = result
                configs_tested += 1
                if configs_tested % 5 == 0:
                    print(
                        f"  Tested {configs_tested}... best: ratio={best_result['actual_ratio']:.1f}:1 "
                        f"error={best_result['actual_error']:.6f} {best_result['grade']} [{best_result['status']}]"
                    )
                del recon, compressed, meta
            except Exception as e:
                err_str = str(e)[:60]
                if configs_tested % 10 == 0:
                    print(f"  Config fail: {err_str}")
            finally:
                engine.close()
                gc.collect()

    print(f"\n  Total configs tested: {configs_tested}")
    if best_result:
        print(
            f"  BEST: rank_frac={best_result['rank_frac']} keep_frac={best_result['keep_frac']} "
            f"ratio={best_result['actual_ratio']:.1f}:1 error={best_result['actual_error']:.6f} "
            f"SNR={best_result['snr']:.1f}dB {best_result['grade']} [{best_result['status']}]"
        )
    return {"results": results, "best": best_result, "configs_tested": configs_tested}


# ── Bottleneck Analysis ─────────────────────────────────────────────────


def analyze_bottlenecks(diagnostics: List[CascadeDiagnostic]) -> Dict[str, Any]:
    print("\n" + "=" * 72)
    print("PHASE 4: BOTTLENECK ANALYSIS")
    print("=" * 72)

    findings = {}
    for diag in diagnostics:
        if diag is None or not diag.stages:
            continue

        print(
            f"\n  Target {diag.target:5d}:1 → actual {diag.actual_ratio:>8.1f}:1 "
            f"error {diag.actual_error:.6f} [{diag.status}]"
        )

        for j, stage in enumerate(diag.stages):
            ratio_share = stage.sub_ratio / max(diag.actual_ratio, 1)
            print(
                f"    Stage {stage.stage_index} ({stage.method_name:25s}): "
                f"ratio={stage.sub_ratio:>8.1f}:1 ({ratio_share * 100:5.1f}%) | "
                f"cum_error={stage.cumulative_error:.6f}"
            )

        additive_approx = sum(s.sub_error for s in diag.stages) / len(diag.stages)
        multiplicative_approx = 1.0 - np.prod([1.0 - s.sub_error for s in diag.stages])
        print(
            f"    Error model: actual={diag.actual_error:.6f}, "
            f"additive_approx={additive_approx:.6f}, "
            f"multiplicative_approx={multiplicative_approx:.6f}"
        )

        findings[f"target_{int(diag.target)}"] = {
            "n_stages": diag.n_stages,
            "stage_ratios": [s.sub_ratio for s in diag.stages],
            "stage_errors": [s.sub_error for s in diag.stages],
            "actual_ratio": diag.actual_ratio,
            "actual_error": diag.actual_error,
            "status": diag.status,
        }

    return findings


# ── Main ─────────────────────────────────────────────────────────────────


def main():
    print("=" * 72)
    print("CASCADE COMPRESSION DIAL-IN ON REAL GEMMA-4 WEIGHTS")
    print("=" * 72)

    tensor, name, total_bytes = sample_real_weights()
    print(f"Tensor: {tensor.shape}, {tensor.dtype}, {tensor.nbytes / 1e6:.1f} MB")

    # Phase 1: Individual tier tests
    tier_results = test_individual_tiers(tensor)

    # Phase 2: Cascade diagnostics
    print("\n" + "=" * 72)
    print("PHASE 2: CASCADE DIAGNOSTICS")
    print("=" * 72)

    cascade_diagnostics = []
    for target in TARGETS:
        print(f"\n--- Target {target}:1 ---")
        diag = diagnose_cascade_stages(tensor, float(target), MAX_ERROR, name)
        if diag:
            cascade_diagnostics.append(diag)

    # Phase 3: Parameter sweep
    sweep_results = {}
    for target in [1200, 5000]:
        sweep = tune_cascade_parameters(tensor, float(target), name)
        sweep_results[target] = sweep

    # Phase 4: Bottleneck analysis
    findings = analyze_bottlenecks(cascade_diagnostics)

    # Phase 5: Summary
    print("\n" + "=" * 72)
    print("PHASE 5: SUMMARY & OPTIMAL CONFIGURATION")
    print("=" * 72)

    print("\n--- Tier Performance ---")
    for mname, r in sorted(tier_results.items(), key=lambda x: -x[1]["ratio"]):
        s = "✓" if r["error"] <= MAX_ERROR else "✗"
        print(
            f"  {s} {mname:25s} ratio={r['ratio']:>8.1f}:1  error={r['error']:.6f}  "
            f"SNR={r['snr']:6.1f}dB  grade={r['grade']}"
        )

    print("\n--- Cascade Performance ---")
    for diag in cascade_diagnostics:
        if diag:
            print(
                f"  [{diag.status}] Target {diag.target:5d}:1 → "
                f"actual {diag.actual_ratio:>8.1f}:1  "
                f"error={diag.actual_error:.6f}  "
                f"SNR={diag.snr_db:.1f}dB  {diag.n_stages} stages  "
                f"{diag.total_time:.2f}s"
            )

    print("\n--- Optimal Configurations ---")
    for target, sweep in sweep_results.items():
        best = sweep.get("best")
        if best:
            print(f"  Target {target}:1:")
            print(f"    rank_frac={best['rank_frac']}, keep_frac={best['keep_frac']}")
            print(f"    stages={best['stages']}")
            print(
                f"    ratio={best['actual_ratio']:.1f}:1, error={best['actual_error']:.6f} [{best['status']}]"
            )

    print("\n--- Bottleneck Findings ---")
    for tk, f in findings.items():
        print(
            f"  {tk}: status={f['status']}, stages={f['n_stages']}, "
            f"ratios={[f'{r:.1f}' for r in f['stage_ratios']]}"
        )
        if f["actual_error"] > MAX_ERROR:
            worst = int(np.argmax(f["stage_errors"]))
            print(
                f"    BOTTLENECK: Stage {worst + 1} error={f['stage_errors'][worst]:.6f}"
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
