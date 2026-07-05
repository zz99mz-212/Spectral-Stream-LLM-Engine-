#!/usr/bin/env python3
"""
Multi-tensor validation of top-performing methods across FFN and attention weights.
Tests 3 tensor types × 2 shapes each = 6 configurations per method.
"""

import gc, json, os, signal, sys, time
from collections import defaultdict, OrderedDict
from typing import Any, Dict, List
import numpy as np

sys.path.insert(0, ".")


class TimeoutError(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutError("Timeout")


# Top methods discovered from profiling (representative subset)
TOP_METHODS = OrderedDict(
    [
        # Best quality/ratio balance (sorted by score)
        ("delta_int4", "delta_quant"),
        ("delta_int4_sparse_ratio2", "delta_quant"),
        ("delta_int4_sparse_ratio5", "delta_quant"),
        ("sparsity_int4", "sparsity_quant"),
        ("sparse_quant", "sparsity_quant"),
        ("sparsity_int4_ratio2", "sparsity_quant"),
        ("sparsity_int4_ratio10", "sparsity_quant"),
        ("block_int8", "quantization"),
        ("block_int4", "quantization"),
        ("awq_quant", "quantization"),
        ("hadamard_int4", "transform_quant"),
        ("cascade_2_stage", "hybrid"),
        ("cascade_full_1200", "cascade"),
        ("cascade_stage1_structural", "cascade"),
        ("cascade_stage2_delta", "cascade"),
        ("boltzmann_encoding", "functional"),
        ("information_bottleneck", "functional"),
        ("svd_compress", "decomposition"),
        ("dct_spectral", "spectral"),
        ("quantum_control", "quantum_engine"),
        ("active_inference_code", "information_theory_2"),
        ("algorithmic_mutual_info", "information_theory_2"),
        ("cp_decomposition", "decomposition"),
        ("density_matrix_renorm", "novel"),
        ("einsort_tt", "decomposition"),
        ("tt_cross", "novel"),
        ("tensor_train", "tensor_network"),
        ("topological_skeleton", "revolutionary_topological"),
        # Perfect quality methods
        ("lossless_zstd", "lossless"),
        ("mera_adv", "novel"),
        ("peps_boundary", "novel"),
        # Extreme ratio (unlikely to generalize well but good to check)
        ("siren_sin_h16_l2_f1.0", "functional_weight_space"),
        ("tt_rank4", "tensor_network"),
        ("chaos_predictability", "novel_chaos"),
    ]
)


def test_one(method_name: str, tensor: np.ndarray, timeout_s: int = 90):
    from spectralstream.compression.methods import METHOD_CLASSES
    from spectralstream.core.math_primitives.quality import QualityAssessor

    result = {"method": method_name}
    cls = METHOD_CLASSES.get(method_name)
    if cls is None:
        result["status"] = "NOT_FOUND"
        return result
    try:
        inst = cls() if isinstance(cls, type) else cls
    except Exception as e:
        result["status"] = "INST_ERR"
        result["error"] = str(e)[:200]
        return result
    result["category"] = getattr(inst, "category", "unknown")
    result["method"] = getattr(inst, "name", method_name)

    if not callable(getattr(inst, "compress", None)):
        result["status"] = "NO_COMPRESS"
        return result
    gc.collect()

    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_s)
        t0 = time.time()
        data, meta = inst.compress(tensor)
        tc = time.time() - t0
        signal.alarm(0)
    except TimeoutError:
        result["status"] = "TIMEOUT"
        return result
    except Exception as e:
        signal.alarm(0)
        result["status"] = "CMP_ERR"
        result["error"] = str(e)[:200]
        return result

    if data is None:
        result["status"] = "NULL_DATA"
        return result

    csize = (
        len(data)
        if isinstance(data, (bytes, bytearray))
        else (
            sum(len(d) if isinstance(d, (bytes, bytearray)) else 0 for d in data)
            if isinstance(data, list)
            else sys.getsizeof(data)
        )
    )
    if csize <= 0:
        csize = tensor.nbytes

    if not callable(getattr(inst, "decompress", None)):
        result["status"] = "NO_DECOMP"
        result["cs"] = csize
        return result

    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_s)
        t0 = time.time()
        recon = inst.decompress(data, meta)
        td = time.time() - t0
        signal.alarm(0)
    except TimeoutError:
        result["status"] = "DECOMP_TMO"
        result["cs"] = csize
        return result
    except Exception as e:
        signal.alarm(0)
        result["status"] = "DECOMP_ERR"
        result["error"] = str(e)[:200]
        result["cs"] = csize
        return result

    if recon is None:
        result["status"] = "NULL_RECON"
        result["cs"] = csize
        return result

    try:
        qa = QualityAssessor()
        q = qa.assess(tensor, recon)
    except Exception as e:
        result["status"] = "QUAL_ERR"
        result["error"] = str(e)[:200]
        result["cs"] = csize
        return result

    ratio = tensor.nbytes / max(csize, 1)
    del recon, data
    gc.collect()

    result.update(
        {
            "status": "OK",
            "ratio": float(ratio),
            "cos": float(q.cosine_similarity),
            "snr": float(q.snr_db),
            "psnr": float(q.psnr_db),
            "ssim": float(q.ssim),
            "corr": float(q.correlation_coefficient),
            "ber": float(q.bit_error_rate),
            "rel": float(q.relative_error),
            "ho": float(q.histogram_overlap),
            "tc": float(tc),
            "td": float(td),
            "cs": int(csize),
        }
    )
    return result


def load_tensor(mmap_path: str, tensor_name: str) -> np.ndarray:
    from spectralstream.compression.engine.memory_mapped_engine import (
        MemoryMappedTensorEngine,
    )

    mmap = MemoryMappedTensorEngine(mmap_path)
    view = mmap.get_tensor(tensor_name)
    arr = np.array(view, copy=True)
    mmap.close()
    return arr


def get_test_tensors(mmap_path: str):
    from spectralstream.compression.engine.memory_mapped_engine import (
        MemoryMappedTensorEngine,
    )

    mmap = MemoryMappedTensorEngine(mmap_path)
    # Select 3 representative tensors: FFN gate (rectangular), FFN down (tall), attention Q
    selected = OrderedDict()
    for name in mmap.get_tensor_names():
        shape, dtype_str, offset, nbytes = mmap.get_tensor_info(name)
        if 10_000_000 < nbytes < 200_000_000 and "weight" in name:
            if "mlp.gate_proj" in name:
                selected["gate_proj"] = (name, shape, dtype_str, nbytes)
            elif "mlp.down_proj" in name:
                selected["down_proj"] = (name, shape, dtype_str, nbytes)
            elif "self_attn.q_proj" in name:
                selected["q_proj"] = (name, shape, dtype_str, nbytes)
            elif "self_attn.o_proj" in name:
                selected["o_proj"] = (name, shape, dtype_str, nbytes)
        if len(selected) >= 4:
            break
    mmap.close()
    return selected


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="/home/mike/Documents/Github/SpectralStream/models/gemma-4-E2B/model.safetensors",
    )
    parser.add_argument("--output", type=str, default="/tmp/profile_multitensor.json")
    args = parser.parse_args()

    print("=" * 120)
    print("  MULTI-TENSOR VALIDATION — Top Methods Across 4 Tensor Types")
    print(f"  Methods: {len(TOP_METHODS)}")
    print("=" * 120)

    # Get test tensors
    test_tensors = get_test_tensors(args.model)
    print(f"\n  Test tensors ({len(test_tensors)}):")
    for key, (name, shape, dtype, nbytes) in test_tensors.items():
        print(
            f"    {key:>12s}: {name.split('.')[-3] + '.' + name.split('.')[-2] + '.' + name.split('.')[-1]:<35s} "
            f"shape={shape} {nbytes / 1e6:.1f}MB {dtype}"
        )

    all_results = []
    start_time = time.time()
    total_tests = len(TOP_METHODS) * len(test_tensors)

    for t_idx, (tkey, (tname, tshape, tdtype, tnbytes)) in enumerate(
        test_tensors.items()
    ):
        print(f"\n[{tkey}] Loading tensor ({tnbytes / 1e6:.1f}MB)...")
        tensor = load_tensor(args.model, tname)
        print(
            f"  Loaded: {tensor.shape} {tensor.nbytes / 1e6:.1f}MB [{tensor.min():.4f}, {tensor.max():.4f}]"
        )

        for m_idx, (mname, mcat) in enumerate(TOP_METHODS.items()):
            idx = t_idx * len(TOP_METHODS) + m_idx + 1
            elapsed = time.time() - start_time

            # Show progress
            rr = f"[{idx:3d}/{total_tests}] {mname:<32s} on {tkey:<12s}"
            print(f"\r{rr} ({elapsed:.0f}s)...", end="")
            sys.stdout.flush()

            result = test_one(mname, tensor, timeout_s=90)
            result["tensor_key"] = tkey
            result["tensor_name"] = tname
            result["tensor_shape"] = str(tensor.shape)
            all_results.append(result)

            if result["status"] == "OK":
                print(
                    f"\r{rr} r={result['ratio']:>8.1f}:1 cos={result['cos']:.4f} "
                    f"SNR={result['snr']:.1f}dB BER={result.get('ber', 0):.4f}"
                )
            elif result["status"] == "TIMEOUT":
                print(f"\r{rr} TIMEOUT")
            else:
                print(f"\r{rr} {result['status']}: {result.get('error', '')[:40]}")
            sys.stdout.flush()
            gc.collect()

        del tensor
        gc.collect()

    total_elapsed = time.time() - start_time
    print(f"\n\n  Total: {total_elapsed:.0f}s ({total_elapsed / 60:.1f}min)")

    # Print per-method cross-tensor summary
    print(f"\n{'=' * 120}")
    print(f"  CROSS-TENSOR VALIDATION SUMMARY")
    print(f"{'=' * 120}")

    # Group by method
    by_method = defaultdict(list)
    for r in all_results:
        by_method[r["method"]].append(r)

    print(
        f"\n  {'METHOD':<28s} {'CATEGORY':<22s} {'MIN_R':>7s} {'AVG_R':>7s} "
        f"{'MIN_COS':>7s} {'AVG_COS':>7s} {'MIN_SNR':>7s} {'AVG_SNR':>7s}"
    )
    print(
        f"  {'-' * 28} {'-' * 22} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 7}"
    )

    method_stats = []
    for mname, results in by_method.items():
        ok_results = [r for r in results if r["status"] == "OK"]
        if not ok_results:
            method_stats.append(
                (mname, results[0].get("category", "?"), 0, 0, 0, 0, 0, 0, 0)
            )
            continue
        ratios = [r["ratio"] for r in ok_results]
        coss = [r["cos"] for r in ok_results]
        snrs = [r["snr"] for r in ok_results]
        berrs = [r.get("ber", 1) for r in ok_results]
        cat = ok_results[0].get("category", "?")
        min_r = min(ratios)
        avg_r = np.mean(ratios)
        min_c = min(coss)
        avg_c = np.mean(coss)
        min_s = min(snrs)
        avg_s = np.mean(snrs)
        avg_ber = np.mean(berrs)
        avg_score = avg_r * avg_c**2
        method_stats.append(
            (mname, cat, min_r, avg_r, min_c, avg_c, min_s, avg_s, avg_score, avg_ber)
        )
        print(
            f"  {mname:<28s} {cat:<22s} {min_r:>7.1f}:1 {avg_r:>7.1f}:1 "
            f"{min_c:>7.4f} {avg_c:>7.4f} {min_s:>7.1f}dB {avg_s:>7.1f}dB"
        )

    # Best overall by average score
    print(f"\n  >>> TOP 15 BY AVERAGE SCORE (avg_ratio × avg_cos²):")
    method_stats.sort(key=lambda x: -x[8])
    for ms in method_stats[:15]:
        mname, cat, min_r, avg_r, min_c, avg_c, min_s, avg_s, avg_score, avg_ber = ms
        print(
            f"    {mname:<28s} cat={cat:<22s} avg_r={avg_r:>8.1f}:1 avg_cos={avg_c:.4f} "
            f"avg_SNR={avg_s:.1f}dB min_cos={min_c:.4f} BER={avg_ber:.4f}"
        )

    # Best quality methods (cos consistently > 0.99)
    print(f"\n  >>> CONSISTENT HIGH-QUALITY METHODS (min_cos > 0.95):")
    high_q = [ms for ms in method_stats if ms[4] > 0.95 and ms[2] > 1]
    high_q.sort(key=lambda x: -x[2] * x[4] ** 2)
    for ms in high_q:
        mname, cat, min_r, avg_r, min_c, avg_c, min_s, avg_s, avg_score, avg_ber = ms
        print(
            f"    {mname:<28s} avg_r={avg_r:>8.1f}:1 cos_range=[{min_c:.4f},{avg_c:.4f}] "
            f"SNR_range=[{min_s:.1f},{avg_s:.1f}]dB BER={avg_ber:.4f}"
        )

    # Save
    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\n  Saved {len(all_results)} results to {args.output}")

    total_ok = sum(1 for r in all_results if r["status"] == "OK")
    total_to = sum(1 for r in all_results if r["status"] == "TIMEOUT")
    total_err = sum(1 for r in all_results if r["status"] not in ("OK", "TIMEOUT"))
    print(f"\n{'=' * 120}")
    print(
        f"  FINAL: OK: {total_ok}/{len(all_results)} | Timeout: {total_to} | Errors: {total_err}"
    )
    print(f"{'=' * 120}")


if __name__ == "__main__":
    main()
