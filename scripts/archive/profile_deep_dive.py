#!/usr/bin/env python3
"""
Deep-dive profile: test 5-10 methods per high-potential category on full-scale weights.
Focus on categories that showed quality potential in the initial sweep.
"""

import gc, json, os, signal, sys, time, traceback
from collections import defaultdict
from typing import Any, Dict, List
import numpy as np

sys.path.insert(0, ".")


class TimeoutError(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutError("Timeout")


# Categories that showed real compression quality
HIGH_PRIORITY_CATS = [
    "quantization",  # block_int8: 3.9:1 cos=1.0, block_int4: 4.0:1 cos=0.998
    "delta_quant",  # delta_int4: 5.3:1 cos=0.997
    "sparsity_quant",  # sparsity_int4: 5.8:1 cos=0.984
    "transform_quant",  # hadamard_int4: 2.2:1 cos=0.999
    "decomposition",  # svd_compress: 9.8:1 cos=0.599
    "spectral",  # dct_spectral: 6.7:1 cos=0.663
    "hybrid",  # cascade_2_stage: 3.3:1 cos=0.996
    "cascade",  # cascade_stage1: 16.1:1 cos=0.521
    "functional",  # boltzmann: 4.0:1 cos=0.988
    "novel",  # density_matrix_renorm: 38.4:1 cos=0.326
    "novel_chaos",  # bifurcation: 8.3:1 cos=0.504
    "novel_signal",  # adaptive_filter: 4.0:1 cos=0.781
    "novel_info",  # channel_capacity: 4.0:1 cos=0.770
    "fractal_holographic",  # chaotic_adaptive: 4.0:1 cos=0.746
    "novel_biological",  # predictive_coding: unknown
    "novel_fractal",  # fractal_weight: unknown
    "novel_physics",  # drift_wave: 3.8:1 cos=-0.0002
    "novel_chaotic",  # double_pendulum: unknown
]

# Categories with extreme ratio potential (even if quality is low)
EXTREME_RATIO_CATS = [
    "tensor_network",  # tensor_train: 682:1
    "revolutionary_topological",  # topological_skeleton: 77:1
    "functional_weight_space",  # siren: 29399:1
    "quantum_engine",  # quantum_budget_allocator: 496694:1
    "quantum_compression",  # adiabatic: unknown
    "unified_physics_quantum2",  # category_adjunction: 154:1
    "information_theory_2",  # active_inference_code: 3.9:1 cos=1.0
]


def discover_methods():
    from spectralstream.compression.methods import METHOD_CLASSES

    cat_methods = defaultdict(list)
    for name in list(METHOD_CLASSES.keys()):
        cls = METHOD_CLASSES[name]
        try:
            inst = cls() if isinstance(cls, type) else cls
            cat = getattr(inst, "category", "unknown")
            cat_methods[cat].append(name)
        except Exception:
            pass
    return dict(cat_methods)


def test_one(method_name: str, tensor: np.ndarray, timeout_s: int = 60):
    from spectralstream.compression.methods import METHOD_CLASSES
    from spectralstream.core.math_primitives.quality import QualityAssessor

    result = {"method": method_name, "shape": str(tensor.shape)}

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

    cat = getattr(inst, "category", "unknown")
    result["category"] = cat
    mname = getattr(inst, "name", method_name)
    result["method"] = mname

    if not callable(getattr(inst, "compress", None)):
        result["status"] = "NO_COMPRESS"
        return result

    gc.collect()

    # Compress
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

    # Decompress
    if not callable(getattr(inst, "decompress", None)):
        result["status"] = "NO_DECOMPRESS"
        result["cs"] = csize
        return result

    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_s // 2 + 10)
        t0 = time.time()
        recon = inst.decompress(data, meta)
        td = time.time() - t0
        signal.alarm(0)
    except TimeoutError:
        result["status"] = "DECOMP_TIMEOUT"
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
            "cs": int(csize),
            "cos": float(q.cosine_similarity),
            "snr": float(q.snr_db),
            "psnr": float(q.psnr_db),
            "mse": float(q.mse),
            "mae": float(q.mae),
            "rel": float(q.relative_error),
            "ssim": float(q.ssim),
            "corr": float(q.correlation_coefficient),
            "ber": float(q.bit_error_rate),
            "tc": float(tc),
            "td": float(td),
            "ho": float(q.histogram_overlap),
        }
    )
    return result


def load_tensor(mmap_path: str, tensor_name: str):
    from spectralstream.compression.engine.memory_mapped_engine import (
        MemoryMappedTensorEngine,
    )

    mmap = MemoryMappedTensorEngine(mmap_path)
    view = mmap.get_tensor(tensor_name)
    arr = np.array(view, copy=True)
    mmap.close()
    return arr


def scan_tensors(mmap_path: str):
    from spectralstream.compression.engine.memory_mapped_engine import (
        MemoryMappedTensorEngine,
    )

    mmap = MemoryMappedTensorEngine(mmap_path)
    result = []
    for name in mmap.get_tensor_names():
        shape, dtype_str, offset, nbytes = mmap.get_tensor_info(name)
        if 10_000_000 < nbytes < 200_000_000 and "weight" in name and "mlp" in name:
            result.append((name, shape, dtype_str, nbytes))
    mmap.close()
    return result


def print_results(all_results):
    by_status = defaultdict(list)
    for r in all_results:
        by_status[r.get("status", "UNKNOWN")].append(r)

    ok = by_status.get("OK", [])
    timeout = by_status.get("TIMEOUT", [])
    errors = [r for r in all_results if r["status"] not in ("OK", "TIMEOUT")]

    print(f"\n{'=' * 110}")
    print(f"  DEEP DIVE RESULTS")
    print(f"{'=' * 110}")
    print(
        f"  Total: {len(all_results)} | OK: {len(ok)} | Timeout: {len(timeout)} | Errors: {len(errors)}\n"
    )

    if ok:
        ok_s = sorted(ok, key=lambda r: -r.get("ratio", 0))
        print(
            f"  {'METHOD':<32s} {'CATEGORY':<28s} {'RATIO':>8s} {'COS':>7s} {'SNR':>7s} "
            f"{'SSIM':>7s} {'CORR':>7s} {'BER':>7s} {'TIME':>6s}"
        )
        print(
            f"  {'-' * 32} {'-' * 28} {'-' * 8} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 6}"
        )
        for r in ok_s[:80]:
            print(
                f"  {r['method']:<32s} {r['category']:<28s} {r['ratio']:>8.1f}:1 "
                f"{r.get('cos', 0):>7.4f} {r.get('snr', 0):>7.1f}dB "
                f"{r.get('ssim', 0):>7.4f} {r.get('corr', 0):>7.4f} "
                f"{r.get('ber', 0):>7.4f} {r.get('tc', 0):>5.1f}s"
            )

        # High quality, good ratio
        print(f"\n  >>> RATIO > 5:1, COS > 0.95 (THE SWEET SPOT):")
        sweet = [r for r in ok if r["ratio"] > 5 and r.get("cos", 0) > 0.95]
        for r in sorted(sweet, key=lambda x: -x["ratio"] * x.get("cos", 0) ** 2):
            print(
                f"    {r['method']:<32s} {r['category']:<28s} {r['ratio']:>8.1f}:1 "
                f"cos={r['cos']:.4f} SNR={r['snr']:.1f}dB BER={r.get('ber', 0):.6f}"
            )
        if not sweet:
            print("    (none) — need to push harder on quantization")

        # High ratio with moderate quality
        print(f"\n  >>> RATIO > 20:1, COS > 0.3:")
        mod = [r for r in ok if r["ratio"] > 20 and r.get("cos", 0) > 0.3]
        for r in sorted(mod, key=lambda x: -x["ratio"] * x.get("cos", 0)):
            print(
                f"    {r['method']:<32s} {r['category']:<28s} {r['ratio']:>8.1f}:1 "
                f"cos={r['cos']:.4f} SNR={r['snr']:.1f}dB"
            )

        # Extreme ratio
        print(f"\n  >>> EXTREME RATIO (> 500:1):")
        extreme = [r for r in ok if r["ratio"] > 500]
        for r in sorted(extreme, key=lambda x: -x["ratio"]):
            print(
                f"    {r['method']:<32s} {r['category']:<28s} {r['ratio']:>10.1f}:1 "
                f"cos={r['cos']:.4f} SNR={r['snr']:.1f}dB BER={r.get('ber', 0):.4f}"
            )

        # Best per category
        print(f"\n  >>> BEST PER CATEGORY (by score = ratio × cos²):")
        per_cat = {}
        for r in ok:
            c = r["category"]
            sc = r["ratio"] * r.get("cos", 0) ** 2
            if c not in per_cat or sc > per_cat[c][0]:
                per_cat[c] = (sc, r)
        for cat in sorted(per_cat.keys()):
            sc, r = per_cat[cat]
            print(
                f"    {cat:<30s} {r['method']:<25s} score={sc:>8.1f}  "
                f"r={r['ratio']:.1f}:1  cos={r['cos']:.4f}  SNR={r['snr']:.1f}dB"
            )

        # Best overall top 20
        print(f"\n  >>> TOP 20 OVERALL (by score):")
        scored = [(r["ratio"] * r.get("cos", 0) ** 2, r) for r in ok]
        scored.sort(key=lambda x: -x[0])
        for score, r in scored[:20]:
            print(
                f"    {r['method']:<32s} score={score:>8.1f}  "
                f"r={r['ratio']:>8.1f}:1  cos={r['cos']:.4f}  SNR={r['snr']:.1f}dB  "
                f"BER={r.get('ber', 0):.6f}  TIME={r.get('tc', 0):.1f}s"
            )

    if errors:
        print(f"\n  Errors ({len(errors)}):")
        for r in errors[:20]:
            print(
                f"    {r.get('method', '?'):<32s} [{r.get('status', '?')}] {r.get('error', '')[:100]}"
            )


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="/home/mike/Documents/Github/SpectralStream/models/gemma-4-E2B/model.safetensors",
    )
    parser.add_argument("--max-methods", type=int, default=10)
    parser.add_argument("--cat", type=str, default=None)
    parser.add_argument("--output", type=str, default="/tmp/profile_deep.json")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument(
        "--tensor-size",
        type=str,
        default="large",
        choices=["small", "medium", "large", "huge"],
    )
    args = parser.parse_args()

    print("=" * 110)
    print("  DEEP DIVE — Full-Scale Compression Profiler")
    print(f"  Methods per category: {args.max_methods}, Timeout: {args.timeout}s")
    print("=" * 110)

    cat_methods = discover_methods()
    print(
        f"\n  Discovered {sum(len(v) for v in cat_methods.values())} methods across {len(cat_methods)} categories"
    )

    # Select categories
    if args.cat:
        cats_to_test = [args.cat]
    else:
        cats_to_test = HIGH_PRIORITY_CATS + EXTREME_RATIO_CATS

    print(f"\n  Target categories ({len(cats_to_test)}):")
    for c in cats_to_test:
        n = len(cat_methods.get(c, []))
        print(f"    {c:<35s} {n} methods")

    # Scan tensors
    tensors = scan_tensors(args.model)
    print(f"\n  Found {len(tensors)} qualifying tensors")

    # Pick tensor by size
    size_map = {
        "small": (10_000_000, 30_000_000),
        "medium": (30_000_000, 50_000_000),
        "large": (50_000_000, 100_000_000),
        "huge": (100_000_000, 500_000_000),
    }
    lo, hi = size_map.get(args.tensor_size, (50_000_000, 100_000_000))

    candidates = [(n, s, d, b) for n, s, d, b in tensors if lo <= b <= hi] or tensors
    primary = candidates[0]
    print(
        f"\n  Using: {primary[0]} shape={primary[1]} {primary[3] / 1e6:.1f}MB {primary[2]}"
    )

    # Load tensor
    tensor = load_tensor(args.model, primary[0])
    print(f"  Loaded: {tensor.shape} {tensor.nbytes / 1e6:.1f}MB {tensor.dtype}")
    print(f"  Value range: [{tensor.min():.4f}, {tensor.max():.4f}]")

    # Test methods
    start_time = time.time()
    all_results = []
    cat_ok_ct = defaultdict(int)
    cat_tested_ct = defaultdict(int)

    for cat_idx, cat in enumerate(cats_to_test):
        methods = cat_methods.get(cat, [])[: args.max_methods]
        if not methods:
            print(f"\n  [{cat_idx + 1}/{len(cats_to_test)}] {cat} — NO METHODS FOUND")
            continue

        for mi, mname in enumerate(methods):
            cat_tested_ct[cat] += 1
            idx = sum(cat_tested_ct.values())
            total_est = args.max_methods * len(cats_to_test)
            elapsed = time.time() - start_time

            print(
                f"\r  [{idx:4d}/{total_est}] {cat:<30s} {mname:<30s} "
                f"({elapsed:.0f}s elapsed)...",
                end="",
            )
            sys.stdout.flush()

            result = test_one(mname, tensor, timeout_s=args.timeout)
            all_results.append(result)

            if result["status"] == "OK":
                cat_ok_ct[cat] += 1
                # Show result inline
                print(
                    f"\r  [{idx:4d}/{total_est}] {cat:<30s} {mname:<30s} "
                    f"r={result['ratio']:.1f}:1 cos={result['cos']:.4f} "
                    f"SNR={result['snr']:.1f}dB BER={result.get('ber', 0):.4f}     "
                )
            elif result["status"] in ("TIMEOUT",):
                print(
                    f"\r  [{idx:4d}/{total_est}] {cat:<30s} {mname:<30s} TIMEOUT              "
                )
            elif result["status"] in ("CMP_ERR", "DECOMP_ERR", "QUAL_ERR", "INST_ERR"):
                print(
                    f"\r  [{idx:4d}/{total_est}] {cat:<30s} {mname:<30s} "
                    f"ERR: {result.get('error', result['status'])[:50]}"
                )
            else:
                print(
                    f"\r  [{idx:4d}/{total_est}] {cat:<30s} {mname:<30s} {result['status']}"
                )
            sys.stdout.flush()
            gc.collect()

    total_elapsed = time.time() - start_time
    print(
        f"\n\n  Deep dive completed in {total_elapsed:.0f}s ({total_elapsed / 60:.1f}min)"
    )

    print_results(all_results)

    # Save
    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\n  Saved {len(all_results)} results to {args.output}")

    ok_count = sum(1 for r in all_results if r["status"] == "OK")
    to_count = sum(1 for r in all_results if r["status"] == "TIMEOUT")
    err_count = sum(1 for r in all_results if r["status"] not in ("OK", "TIMEOUT"))
    print(f"\n{'=' * 110}")
    print(
        f"  FINAL: {len(all_results)} | OK: {ok_count} | Timeout: {to_count} | Errors: {err_count}"
    )
    print(f"  Time: {total_elapsed:.0f}s")
    print(f"{'=' * 110}")


if __name__ == "__main__":
    main()
