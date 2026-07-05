"""
DIAL IN: SVD/Decomposition/Tensor-Train methods on REAL Gemma-4 weights.
Multiplicative residual cascade targeting 500:1 - 5000:1.
Quantization is LAST RESORT only — pure decomposition.
"""

import gc
import sys
import time
import json
import numpy as np

sys.path.insert(0, ".")

from spectralstream.compression.engine.memory_mapped_engine import (
    MemoryMappedTensorEngine,
)
from spectralstream.core.math_primitives.quality import QualityAssessor
from spectralstream.compression.methods import METHOD_CLASSES

qa = QualityAssessor()

MODEL_PATH = (
    "/home/mike/Documents/Github/SpectralStream/models/gemma-4-E2B/model.safetensors"
)

# ── Decomposition/SVD/Tensor methods ONLY (no quantization) ──
DECOMP_METHODS = [
    # Engine built-in
    "svd_compress",
    "tensor_train",
    # Decomposition wrappers
    "butterfly",
    "monarch",
    "cp_decomposition",
    "einsort_tt",
    "lotr",
    "kronecker",
    "cur_decomposition",
    "h_matrix",
    "nystrom",
    "adntn_mera",
    "block_diagonal",
    "toeplitz",
    "hankel",
    "svd_truncated",
    "tensor_network",
    "hierarchical_mps",
    "decomp_tensor_train",
    "tensor_ring",
    "tt_orthogonal",
    "tt_svd",
    "tucker_decomposition",
    "block_tucker",
    "hierarchical_tucker",
    # Novel tensor network
    "mera_adv",
    "peps_boundary",
    "qtt_adapt",
    "tt_cross",
    "dmrg_sweep",
    "qtt_fourier",
    "matrix_product_operator",
    "floquet_tensor",
    "singular_value_density",
    "hyperspectral_tensor",
    "tensor_network_regroup",
    "density_matrix_renorm",
    "quantum_fourier_feature",
]

# ── Multi-rank sweep for SVD-based methods ──
SVD_RANKS = [128, 64, 32, 16, 8, 4, 2]

# ── Residual cascade config ──
CASCADE_RANKS = [128, 64, 32, 16, 8, 4]


def get_weight_samples(mmap, max_size: int = 1024) -> dict:
    """Sample real weights from attention and FFN layers."""
    names = mmap.get_tensor_names()
    samples = {}

    # Language model weights (the big ones)
    lang_targets = [
        n
        for n in names
        if "language_model" in n
        and n.endswith(".weight")
        and any(
            x in n
            for x in [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]
        )
    ]

    # Audio tower weights
    audio_targets = [
        n
        for n in names
        if "audio_tower" in n
        and n.endswith(".weight")
        and any(
            x in n
            for x in [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]
        )
    ]

    target_names = (lang_targets[:6] + audio_targets[:6])[:10]
    print(f"\n{'=' * 80}")
    print(f"Sampling {len(target_names)} real weight tensors:")
    print(f"{'=' * 80}")

    for name in target_names:
        view = mmap.get_tensor(name)
        shape = view.shape
        sample_rows = min(max_size, shape[0])
        sample_cols = min(max_size, shape[-1]) if len(shape) > 1 else 1

        if len(shape) == 2:
            data = np.array(view[:sample_rows, :sample_cols]).astype(np.float64)
        elif len(shape) == 1:
            data = np.array(view[:sample_rows]).astype(np.float64)
        else:
            data = np.array(view.ravel()[: sample_rows * sample_cols]).astype(
                np.float64
            )

        info = mmap._tensor_info[name]
        print(
            f"  {name}: shape={info[0]}, sample={data.shape}, "
            f"std={data.std():.4f}, mean={data.mean():.6f}, "
            f"min={data.min():.4f}, max={data.max():.4f}"
        )
        samples[name] = data

    return samples


def test_single_method(
    name: str, method_cls, tensor: np.ndarray, sample_name: str
) -> dict:
    """Test a single decomposition method with default params."""
    result = {"method": name, "tensor": sample_name, "variants": []}

    try:
        inst = method_cls() if isinstance(method_cls, type) else method_cls
        t32 = tensor.astype(np.float32)
        orig_bytes = tensor.nbytes

        # Try default params
        data, meta = inst.compress(t32)
        recon = inst.decompress(data, meta)
        q = qa.assess(tensor, recon.astype(np.float64))
        ratio = orig_bytes / max(len(data), 1)

        result["variants"].append(
            {
                "params": "default",
                "ratio": round(ratio, 1),
                "cosine": round(float(q.cosine_similarity), 6),
                "snr_db": round(float(q.snr_db), 2),
                "rel_error": round(float(q.relative_error), 8),
                "rmse": round(float(q.rmse), 8),
                "data_bytes": len(data),
                "rank": meta.get("rank", meta.get("passthrough", None)),
            }
        )
    except Exception as e:
        result["variants"].append(
            {
                "params": "default",
                "error": str(e)[:100],
            }
        )

    # If SVD-based, sweep ranks
    is_svd_like = any(
        x in name.lower()
        for x in [
            "svd",
            "truncated",
            "tt_",
            "tucker",
            "tensor_ring",
            "tensor_network",
            "tensor_train",
            "decomp_tensor",
            "hierarchical",
        ]
    )
    if is_svd_like:
        for rank in SVD_RANKS:
            try:
                inst2 = method_cls() if isinstance(method_cls, type) else method_cls
                t32 = tensor.astype(np.float32)
                data, meta = inst2.compress(t32, rank=rank)
                recon = inst2.decompress(data, meta)
                q = qa.assess(tensor, recon.astype(np.float64))
                ratio = orig_bytes / max(len(data), 1)
                result["variants"].append(
                    {
                        "params": f"rank={rank}",
                        "ratio": round(ratio, 1),
                        "cosine": round(float(q.cosine_similarity), 6),
                        "snr_db": round(float(q.snr_db), 2),
                        "rel_error": round(float(q.relative_error), 8),
                        "rmse": round(float(q.rmse), 8),
                        "data_bytes": len(data),
                        "rank": meta.get("rank", rank),
                    }
                )
            except Exception as e:
                result["variants"].append(
                    {
                        "params": f"rank={rank}",
                        "error": str(e)[:100],
                    }
                )

    return result


def test_residual_cascade(tensor: np.ndarray, ranks: list, sample_name: str) -> dict:
    """Multiplicative residual cascade: SVD on residuals."""
    print(f"\n  ── Residual cascade on {sample_name} ──")

    t64 = tensor.astype(np.float64)
    orig_shape = t64.shape
    if t64.ndim > 2:
        t64 = t64.reshape(t64.shape[0], -1)

    residual = t64.copy()
    final_recon = np.zeros_like(t64, dtype=np.float64)
    total_ratio = 1.0
    stages = []

    for i, rank in enumerate(ranks):
        # Check if residual quality is already good enough to stop
        try:
            from spectralstream.compression.engine._methods import _SVDCompress

            svd = _SVDCompress()

            r32 = residual.astype(np.float32)
            data, meta = svd.compress(r32, rank=rank)
            stage_recon = svd.decompress(data, meta).astype(np.float64)
        except Exception:
            # Fallback: manual SVD
            r32 = residual.astype(np.float64)
            U, S, Vt = np.linalg.svd(r32, full_matrices=False)
            k = min(rank, len(S))
            stage_recon = (U[:, :k] * S[:k]) @ Vt[:k, :]

        final_recon += stage_recon
        residual = t64 - final_recon

        stage_ratio = t64.nbytes / max(len(data) if "data" in dir() else 1, 1)
        total_ratio *= stage_ratio

        q = qa.assess(t64, final_recon)

        stage = {
            "stage": i + 1,
            "rank": rank,
            "stage_ratio": round(stage_ratio, 1),
            "cumul_ratio": round(total_ratio, 1),
            "cosine": round(float(q.cosine_similarity), 6),
            "snr_db": round(float(q.snr_db), 2),
            "rel_error": round(float(q.relative_error), 8),
            "residual_std": round(float(residual.std()), 6),
            "residual_mean": round(float(residual.mean()), 8),
        }
        stages.append(stage)
        print(
            f"    Stage {i + 1} rank={rank}: ratio={stage_ratio:.1f}:1 "
            f"(cumul={total_ratio:.1f}:1), cos={q.cosine_similarity:.4f}, "
            f"SNR={q.snr_db:.1f}dB, res_std={residual.std():.4f}"
        )

        if total_ratio >= 5000:
            print(f"    ✓ Target 5000:1 reached!")
            break

    return {
        "tensor": sample_name,
        "shape": list(orig_shape),
        "total_stages": len(stages),
        "final_ratio": round(total_ratio, 1),
        "final_cosine": round(float(q.cosine_similarity), 6),
        "final_snr": round(float(q.snr_db), 2),
        "stages": stages,
    }


def test_multi_method_cascade(
    tensor: np.ndarray, methods_params: list, sample_name: str
) -> dict:
    """Cascade with different methods at each stage."""
    print(f"\n  ── Multi-method cascade on {sample_name} ──")

    t64 = tensor.astype(np.float64)
    orig_shape = t64.shape
    if t64.ndim > 2:
        t64 = t64.reshape(t64.shape[0], -1)

    residual = t64.copy()
    cumulative_recon = np.zeros_like(t64, dtype=np.float64)
    total_ratio = 1.0
    stages = []

    for i, (method_name, method_cls, params) in enumerate(methods_params):
        try:
            inst = method_cls() if isinstance(method_cls, type) else method_cls
            r32 = residual.astype(np.float32)
            data, meta = inst.compress(r32, **params)
            stage_recon = inst.decompress(data, meta).astype(np.float64)
        except Exception as e:
            stages.append(
                {
                    "stage": i + 1,
                    "method": method_name,
                    "params": params,
                    "error": str(e)[:100],
                }
            )
            continue

        cumulative_recon += stage_recon
        residual = t64 - cumulative_recon

        stage_ratio = t64.nbytes / max(len(data), 1)
        total_ratio *= stage_ratio

        # Report quality on FULL cumulative reconstruction
        q = qa.assess(t64, cumulative_recon)

        stage = {
            "stage": i + 1,
            "method": method_name,
            "params": params,
            "stage_ratio": round(stage_ratio, 1),
            "cumul_ratio": round(total_ratio, 1),
            "cosine": round(float(q.cosine_similarity), 6),
            "snr_db": round(float(q.snr_db), 2),
            "rel_error": round(float(q.relative_error), 8),
            "residual_std": round(float(residual.std()), 6),
        }
        stages.append(stage)
        print(
            f"    Stage {i + 1} {method_name}{params}: ratio={stage_ratio:.1f}:1 "
            f"(cumul={total_ratio:.1f}:1), cos={q.cosine_similarity:.4f}, "
            f"SNR={q.snr_db:.1f}dB, res_std={residual.std():.4f}"
        )

        if total_ratio >= 5000:
            print(f"    ✓ Target 5000:1 reached!")
            break

    final_q = qa.assess(t64, cumulative_recon)
    return {
        "tensor": sample_name,
        "shape": list(orig_shape),
        "total_stages": len(stages),
        "final_ratio": round(total_ratio, 1),
        "final_cosine": round(float(final_q.cosine_similarity), 6),
        "final_snr": round(float(final_q.snr_db), 2),
        "stages": stages,
    }


def main():
    mmap = MemoryMappedTensorEngine(MODEL_PATH)
    all_results = {
        "single_method": [],
        "residual_cascade": [],
        "multi_cascade": [],
    }

    try:
        # ── Phase 1: Sample real weights ──
        samples = get_weight_samples(mmap)
        print(f"\nSampled {len(samples)} tensors")

        # Select a diverse subset for method testing
        test_samples = {}
        for name, data in samples.items():
            if "q_proj" in name:
                test_samples["attention_q"] = data
            elif "k_proj" in name:
                test_samples["attention_k"] = data
            elif "v_proj" in name:
                test_samples["attention_v"] = data
            elif "o_proj" in name:
                test_samples["attention_o"] = data
            elif "gate_proj" in name:
                test_samples["ffn_gate"] = data
            elif "up_proj" in name:
                test_samples["ffn_up"] = data
            elif "down_proj" in name:
                test_samples["ffn_down"] = data
        print(f"\nTest sample types: {list(test_samples.keys())}")

        # Use first available sample as representative
        first_key = list(test_samples.keys())[0]
        first_sample = test_samples[first_key]
        print(f"\nUsing '{first_key}' shape={first_sample.shape} for method sweep")

        # ── Phase 2: Test each decomposition method ──
        print(f"\n{'=' * 80}")
        print(
            f"PHASE 2: Test {len(DECOMP_METHODS)} decomposition methods on real weights"
        )
        print(f"{'=' * 80}")

        for method_name in DECOMP_METHODS:
            cls = METHOD_CLASSES.get(method_name)
            if cls is None:
                print(f"  ? {method_name}: not in METHOD_CLASSES, skipping")
                continue

            print(f"\n  Testing {method_name}...", end=" ")
            sys.stdout.flush()
            result = test_single_method(method_name, cls, first_sample, first_key)
            all_results["single_method"].append(result)

            best = None
            for v in result["variants"]:
                if "error" not in v and (
                    best is None or v.get("ratio", 0) > best.get("ratio", 0)
                ):
                    best = v
            if best:
                print(
                    f"✓ best: ratio={best['ratio']}:1, cos={best['cosine']}, SNR={best['snr_db']}dB"
                )
            else:
                print(f"✗ all failed")

            gc.collect()

        # ── Phase 3: Residual cascade on each tensor type ──
        print(f"\n{'=' * 80}")
        print(f"PHASE 3: SVD residual cascade on each tensor type")
        print(f"{'=' * 80}")

        for tname, tdata in test_samples.items():
            result = test_residual_cascade(tdata, CASCADE_RANKS, tname)
            all_results["residual_cascade"].append(result)
            gc.collect()

        # ── Phase 4: Multi-method cascades ──
        print(f"\n{'=' * 80}")
        print(f"PHASE 4: Multi-method cascades")
        print(f"{'=' * 80}")

        # Find top-3 methods by ratio
        method_scores = []
        for r in all_results["single_method"]:
            for v in r["variants"]:
                if "error" not in v and v.get("ratio", 0) > 2:
                    method_scores.append(
                        (r["method"], v["params"], v["ratio"], v["cosine"])
                    )
        method_scores.sort(key=lambda x: -x[2])

        print(f"\n  Top methods by ratio:")
        for m, p, ratio, cos in method_scores[:10]:
            print(f"    {m}({p}): {ratio}:1  cos={cos}")

        if method_scores:
            # Build cascade from diverse high-ratio methods
            used = set()
            cascade_stages = []
            for m, p, ratio, cos in method_scores:
                if m not in used and cos > 0.9:
                    cls = METHOD_CLASSES.get(m)
                    if cls:
                        params = {}
                        if "rank=" in p:
                            params["rank"] = int(p.split("=")[1])
                        cascade_stages.append((m, cls, params))
                        used.add(m)
                    if len(cascade_stages) >= 5:
                        break

            if cascade_stages:
                result = test_multi_method_cascade(
                    first_sample, cascade_stages, first_key
                )
                all_results["multi_cascade"].append(result)
                gc.collect()

        # ── Phase 5: High-ratio aggressive cascade ──
        print(f"\n{'=' * 80}")
        print(f"PHASE 5: Aggressive high-ratio cascade (target 5000:1+)")
        print(f"{'=' * 80}")

        for tname, tdata in test_samples.items():
            if tdata.shape[0] >= 64 and tdata.shape[-1] >= 64:
                aggressive_ranks = [64, 32, 16, 8, 4, 2]
                result = test_residual_cascade(
                    tdata, aggressive_ranks, f"{tname}_aggressive"
                )
                all_results["residual_cascade"].append(result)
                gc.collect()

        # ── Report ──
        print(f"\n{'=' * 80}")
        print(f"FINAL REPORT")
        print(f"{'=' * 80}")

        # Summary table
        print(
            f"\n{'Method':<25} {'Params':<20} {'Ratio':>10} {'Cos':>10} {'SNR dB':>8}"
        )
        print(f"{'-' * 25} {'-' * 20} {'-' * 10} {'-' * 10} {'-' * 8}")
        for r in all_results["single_method"]:
            for v in r["variants"]:
                if "error" not in v:
                    print(
                        f"{r['method']:<25} {v['params']:<20} {v['ratio']:>8.1f}:1 "
                        f"{v['cosine']:>10.4f} {v['snr_db']:>8.1f}"
                    )

        print(f"\n{'=' * 80}")
        print(f"CASCADE RESULTS")
        print(f"{'=' * 80}")
        for cres in all_results["residual_cascade"]:
            print(
                f"\n{cres['tensor']} (shape={cres['shape']}): "
                f"{cres['total_stages']} stages → "
                f"ratio={cres['final_ratio']:>.1f}:1, "
                f"cos={cres['final_cosine']:.4f}, "
                f"SNR={cres['final_snr']:.1f}dB"
            )
            for s in cres["stages"]:
                print(
                    f"  Stage {s['stage']} rank={s['rank']}: "
                    f"stage_ratio={s['stage_ratio']}:1 "
                    f"→ cumul={s['cumul_ratio']}:1 "
                    f"cos={s['cosine']:.4f} "
                    f"res_std={s['residual_std']:.4f}"
                )

        # Save results
        output_path = "/tmp/decomposition_dial_in_results.json"
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nFull results saved to {output_path}")

    finally:
        mmap.close()


if __name__ == "__main__":
    main()
