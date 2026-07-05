#!/usr/bin/env python3
"""
Profile compression methods on REAL full-scale FFN weights (12288x1536).
All in-process with signal-based timeout.
"""

import gc, json, os, signal, sys, time, traceback
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

sys.path.insert(0, ".")


class TimeoutError(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutError("Method timed out")


CORE_CATEGORIES = [
    "quantization",
    "transform_quant",
    "sparsity_quant",
    "delta_quant",
    "decomposition",
    "spectral",
    "structural",
    "entropy",
    "hybrid",
    "cascade",
    "lossless",
    "functional",
    "physics",
    "novel",
    "tensor_network",
    "novel_fractal",
    "novel_chaos",
    "novel_chaotic",
    "novel_physics",
    "novel_info",
    "novel_signal",
    "novel_biological",
    "quantum_compression",
    "quantum_engine",
    "functional_weight_space",
    "topological_biological",
    "geometric_topological_manifold",
    "fractal_holographic",
    "information_theory_2",
    "unified_physics_quantum2",
    "revolutionary",
    "revolutionary_gauge",
    "revolutionary_topological",
    "breakthrough_decomposition",
    "breakthrough_hybrid",
    "breakthrough_info",
    "breakthrough_math",
    "breakthrough_signal",
]

RISKY_CATEGORIES = {
    "breakthrough_decomposition",
    "breakthrough_hybrid",
    "breakthrough_info",
    "breakthrough_math",
    "breakthrough_signal",
    "revolutionary",
    "revolutionary_gauge",
    "revolutionary_topological",
    "unified_physics_quantum2",
    "information_theory_2",
    "fractal_holographic",
    "quantum_compression",
    "quantum_engine",
    "functional_weight_space",
}


def discover_methods() -> Dict[str, List[str]]:
    from spectralstream.compression.methods import METHOD_CLASSES

    cat_methods: Dict[str, List[str]] = defaultdict(list)
    for name in list(METHOD_CLASSES.keys()):
        cls = METHOD_CLASSES[name]
        try:
            inst = cls() if isinstance(cls, type) else cls
            cat = getattr(inst, "category", "unknown")
            cat_methods[cat].append(name)
        except Exception:
            cat_methods["unknown"].append(name)
    return dict(cat_methods)


def test_one(
    method_name: str, tensor: np.ndarray, timeout_s: int = 60
) -> Dict[str, Any]:
    from spectralstream.compression.methods import METHOD_CLASSES
    from spectralstream.core.math_primitives.quality import QualityAssessor

    result = {
        "method": method_name,
        "tensor": str(tensor.shape),
        "shape": str(tensor.shape),
    }

    cls = METHOD_CLASSES.get(method_name)
    if cls is None:
        result["status"] = "NOT_FOUND"
        return result

    try:
        inst = cls() if isinstance(cls, type) else cls
    except Exception as e:
        result["status"] = "INSTANTIATE_ERROR"
        result["error"] = str(e)[:200]
        return result

    cat = getattr(inst, "category", "unknown")
    result["category"] = cat
    mname = getattr(inst, "name", method_name)
    if mname:
        result["method"] = mname

    if not hasattr(inst, "compress") or not callable(getattr(inst, "compress", None)):
        result["status"] = "NO_COMPRESS_METHOD"
        return result

    gc.collect()

    # ── Compress ──
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
        result["status"] = "COMPRESS_ERROR"
        result["error"] = str(e)[:200]
        result["traceback"] = traceback.format_exc()[-300:]
        return result

    if data is None:
        result["status"] = "NULL_DATA"
        return result

    # Compressed size
    if isinstance(data, (bytes, bytearray)):
        csize = len(data)
    elif isinstance(data, list):
        csize = sum(len(d) if isinstance(d, (bytes, bytearray)) else 0 for d in data)
    else:
        csize = sys.getsizeof(data)
    if csize <= 0:
        csize = tensor.nbytes

    # ── Decompress ──
    if not hasattr(inst, "decompress") or not callable(
        getattr(inst, "decompress", None)
    ):
        result["status"] = "NO_DECOMPRESS_METHOD"
        result["compressed_size"] = csize
        return result

    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_s // 2 + 10)
        t0 = time.time()
        recon = inst.decompress(data, meta)
        td = time.time() - t0
        signal.alarm(0)
    except TimeoutError:
        result["status"] = "DECOMPRESS_TIMEOUT"
        result["compressed_size"] = csize
        return result
    except Exception as e:
        signal.alarm(0)
        result["status"] = "DECOMPRESS_ERROR"
        result["error"] = str(e)[:200]
        result["compressed_size"] = csize
        return result

    if recon is None:
        result["status"] = "NULL_RECON"
        result["compressed_size"] = csize
        return result

    # ── Quality ──
    try:
        qa = QualityAssessor()
        q = qa.assess(tensor, recon)
    except Exception as e:
        result["status"] = "QUALITY_ERROR"
        result["error"] = str(e)[:200]
        result["compressed_size"] = csize
        return result

    ratio = tensor.nbytes / max(csize, 1)

    del recon, data
    gc.collect()

    result.update(
        {
            "status": "OK",
            "ratio": float(ratio),
            "cosine_similarity": float(q.cosine_similarity),
            "snr_db": float(q.snr_db),
            "psnr_db": float(q.psnr_db),
            "mse": float(q.mse),
            "mae": float(q.mae),
            "max_abs_error": float(q.max_abs_error),
            "relative_error": float(q.relative_error),
            "ssim": float(q.ssim),
            "histogram_overlap": float(q.histogram_overlap),
            "correlation_coefficient": float(q.correlation_coefficient),
            "effective_rank_ratio": float(q.effective_rank_ratio),
            "bit_error_rate": float(q.bit_error_rate),
            "compressed_size": int(csize),
            "original_size": int(tensor.nbytes),
            "t_compress": float(tc),
            "t_decompress": float(td),
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


def scan_tensors(mmap_path: str) -> List[Tuple[str, tuple, str, int]]:
    from spectralstream.compression.engine.memory_mapped_engine import (
        MemoryMappedTensorEngine,
    )

    mmap = MemoryMappedTensorEngine(mmap_path)
    result = []
    for name in mmap.get_tensor_names():
        shape, dtype_str, offset, nbytes = mmap.get_tensor_info(name)
        if 10_000_000 < nbytes < 200_000_000 and "weight" in name:
            if "mlp" in name or "self_attn" in name or "attention" in name:
                result.append((name, shape, dtype_str, nbytes))
    mmap.close()
    return result


def print_results(all_results: List[Dict[str, Any]]):
    by_status = defaultdict(list)
    for r in all_results:
        by_status[r.get("status", "UNKNOWN")].append(r)

    ok = by_status.get("OK", [])
    timeout = by_status.get("TIMEOUT", [])
    errors = [r for r in all_results if r.get("status") not in ("OK", "TIMEOUT")]

    print(f"\n{'=' * 100}")
    print(f"  RESULTS — All Categories")
    print(f"{'=' * 100}")
    print(
        f"  Tested: {len(all_results)} | OK: {len(ok)} | Timeout: {len(timeout)} | Errors: {len(errors)}"
    )

    if ok:
        ok_sorted = sorted(ok, key=lambda r: r.get("ratio", 0), reverse=True)
        print(
            f"\n  {'METHOD':<30s} {'CATEGORY':<25s} {'RATIO':>8s} {'COS':>7s} {'SNR':>7s} {'SSIM':>7s} {'TIME':>6s}"
        )
        print(
            f"  {'-' * 30} {'-' * 25} {'-' * 8} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 6}"
        )
        for r in ok_sorted[:60]:
            tc = r.get("t_compress", 0)
            print(
                f"  {r['method']:<30s} {r['category']:<25s} {r['ratio']:>8.1f}:1 "
                f"{r.get('cosine_similarity', 0):>7.4f} {r.get('snr_db', 0):>7.1f}dB "
                f"{r.get('ssim', 0):>7.4f} {tc:>5.1f}s"
            )

        print(f"\n  >>> HIGH RATIO (ratio > 10:1, cos > 0.9):")
        high = [r for r in ok if r["ratio"] > 10 and r["cosine_similarity"] > 0.9]
        for r in sorted(high, key=lambda x: -x["ratio"]):
            print(
                f"    {r['method']:<30s} {r['ratio']:>8.1f}:1 cos={r['cosine_similarity']:.4f} "
                f"SNR={r['snr_db']:.1f}dB SSIM={r['ssim']:.4f}"
            )

        print(f"\n  >>> VERY HIGH RATIO (ratio > 100:1, cos > 0.5):")
        high2 = [r for r in ok if r["ratio"] > 100 and r["cosine_similarity"] > 0.5]
        for r in sorted(high2, key=lambda x: -x["ratio"]):
            print(
                f"    {r['method']:<30s} {r['ratio']:>8.1f}:1 cos={r['cosine_similarity']:.4f} "
                f"SNR={r['snr_db']:.1f}dB"
            )

        print(f"\n  >>> EXTREME RATIO (ratio > 1000:1, any quality):")
        high3 = [r for r in ok if r["ratio"] > 1000]
        for r in sorted(high3, key=lambda x: -x["ratio"]):
            print(
                f"    {r['method']:<30s} {r['ratio']:>8.1f}:1 cos={r['cosine_similarity']:.4f} "
                f"SNR={r['snr_db']:.1f}dB BER={r.get('bit_error_rate', 'N/A')}"
            )

        print(f"\n  >>> BEST SCORE (ratio × cos²):")
        scored = [(r["ratio"] * r["cosine_similarity"] ** 2, r) for r in ok]
        scored.sort(key=lambda x: -x[0])
        for score, r in scored[:20]:
            print(
                f"    {r['method']:<30s} score={score:>8.1f}  ratio={r['ratio']:>8.1f}:1  "
                f"cos={r['cosine_similarity']:.4f}  SNR={r['snr_db']:.1f}dB"
            )

        print(f"\n  >>> BEST PER CATEGORY:")
        per_cat = {}
        for r in ok:
            c = r["category"]
            sc = r["ratio"] * r["cosine_similarity"] ** 2
            if c not in per_cat or sc > per_cat[c][0]:
                per_cat[c] = (sc, r)
        for cat in sorted(per_cat.keys()):
            sc, r = per_cat[cat]
            print(
                f"    {cat:<35s} {r['method']:<25s} score={sc:.1f}  ratio={r['ratio']:.1f}:1  "
                f"cos={r['cosine_similarity']:.4f}  SNR={r['snr_db']:.1f}dB"
            )

    if errors:
        print(f"\n  --- Errors ({len(errors)}) ---")
        for r in errors[:20]:
            print(
                f"    {r.get('method', '?'):<30s} [{r.get('status', '?')}] {r.get('error', '')[:100]}"
            )

    if timeout:
        print(f"\n  --- Timeouts ({len(timeout)}) ---")
        for r in timeout[:10]:
            print(f"    {r.get('method', '?'):<30s} TIMEOUT")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="/home/mike/Documents/Github/SpectralStream/models/gemma-4-E2B/model.safetensors",
    )
    parser.add_argument("--max-methods", type=int, default=3)
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--output", type=str, default="/tmp/profile_full_scale.json")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--second-tensor", action="store_true")
    parser.add_argument("--skip-risky", action="store_true")
    args = parser.parse_args()

    print("=" * 100)
    print("  SPECTRALSTREAM FULL-SCALE COMPRESSION PROFILER")
    print(f"  Model: {args.model}")
    print(f"  Methods per category: {args.max_methods}, Timeout: {args.timeout}s")
    print("=" * 100)

    # Discover
    print("\n[1/4] Discovering methods...")
    t0 = time.time()
    cat_methods = discover_methods()
    total = sum(len(v) for v in cat_methods.values())
    print(
        f"  {total} methods across {len(cat_methods)} categories ({time.time() - t0:.1f}s)"
    )

    filtered = {}
    skipped_risky = 0
    for cat in CORE_CATEGORIES:
        if cat in cat_methods:
            if args.skip_risky and cat in RISKY_CATEGORIES:
                skipped_risky += len(cat_methods[cat])
                continue
            filtered[cat] = cat_methods[cat]

    n_testable = sum(len(v) for v in filtered.values())
    print(f"  Testing {n_testable} methods across {len(filtered)} categories")
    if skipped_risky:
        print(f"  (skipped {skipped_risky} risky methods)")
    for cat in sorted(filtered.keys()):
        n = len(filtered[cat])
        mark = " *RISKY*" if cat in RISKY_CATEGORIES else ""
        print(f"    {cat:<35s} {n:>5d} methods{mark}")

    # Scan tensors
    print(f"\n[2/4] Scanning model...")
    tensors = scan_tensors(args.model)
    shapes = {}
    for name, shape, dtype, nbytes in tensors:
        key = str(shape)
        if key not in shapes:
            shapes[key] = (dtype, nbytes, name)
            print(f"    {name}: shape={shape}, {nbytes / 1e6:.1f}MB, dtype={dtype}")
    if not tensors:
        print("  FATAL: No tensors found")
        sys.exit(1)

    # Pick primary + secondary
    primary_t = None
    secondary_t = None
    tnames = list(dict.fromkeys([t[0] for t in tensors]))
    for t in tnames:
        shape, dtype_str, _, nbytes = [x for x in tensors if x[0] == t][0]
        if primary_t is None:
            primary_t = t
        elif (
            secondary_t is None
            and shape != [x[1] for x in tensors if x[0] == primary_t][0]
        ):
            secondary_t = t
        if primary_t and secondary_t:
            break
    if primary_t is None:
        primary_t = tnames[0]

    pshape = [x[1] for x in tensors if x[0] == primary_t][0]
    ps = f"{pshape} = {[x[3] for x in tensors if x[0] == primary_t][0] / 1e6:.1f}MB"
    print(f"\n  Primary:   {primary_t.split('.')[-1]}  {ps}")
    if secondary_t:
        sshape = [x[1] for x in tensors if x[0] == secondary_t][0]
        ss = f"{sshape} = {[x[3] for x in tensors if x[0] == secondary_t][0] / 1e6:.1f}MB"
        print(f"  Secondary: {secondary_t.split('.')[-1]}  {ss}")

    # Load primary tensor
    print(f"\n[3/4] Loading primary tensor...")
    tensor = load_tensor(args.model, primary_t)
    print(
        f"  Loaded: {tensor.shape}, {tensor.nbytes / 1e6:.1f}MB, dtype={tensor.dtype}"
    )

    # Test each category
    start_time = time.time()
    all_results: List[Dict[str, Any]] = []
    cat_ok = defaultdict(int)
    cat_tested = defaultdict(int)

    limit_per_cat = args.max_methods
    cat_order = sorted(filtered.keys())
    # Sort so quantized categories come first (most likely to work)
    cat_order.sort(
        key=lambda c: 0
        if c
        in (
            "quantization",
            "transform_quant",
            "sparsity_quant",
            "delta_quant",
            "decomposition",
            "spectral",
            "structural",
            "entropy",
            "lossless",
            "hybrid",
            "cascade",
            "novel",
        )
        else 1
    )

    for cat_idx, cat in enumerate(cat_order):
        if args.category and args.category != cat:
            continue
        methods = filtered[cat]
        test_methods = methods[:limit_per_cat]

        for mi, mname in enumerate(test_methods):
            cat_tested[cat] += 1
            idx = sum(cat_tested.values())
            elapsed = time.time() - start_time
            total_planned = limit_per_cat * len(filtered)

            print(
                f"\r  [{idx:4d}/{total_planned}] {cat:<35s} {mname:<30s} "
                f"({elapsed:.0f}s)...",
                end="",
            )
            sys.stdout.flush()

            result = test_one(mname, tensor, timeout_s=args.timeout)
            result["tensor_name"] = primary_t
            all_results.append(result)

            if result["status"] == "OK":
                cat_ok[cat] += 1
            elif result["status"] == "TIMEOUT":
                pass

            # Per-category best summary
            if cat_ok[cat] > 0:
                cr = [r for r in all_results[-cat_tested[cat] :] if r["status"] == "OK"]
                if cr:
                    best = max(
                        cr, key=lambda r: r["ratio"] * r["cosine_similarity"] ** 2
                    )
                    print(
                        f"\r  [{idx:4d}/{total_planned}] {cat:<35s} BEST: {best['method']:<25s} "
                        f"r={best['ratio']:.1f}:1 cos={best['cosine_similarity']:.4f} SNR={best['snr_db']:.1f}dB   "
                    )
            sys.stdout.flush()
            gc.collect()

    print(f"\n  Primary tensor profiling done in {time.time() - start_time:.1f}s")

    # Second tensor
    if args.second_tensor and secondary_t:
        print(f"\n[3b/4] Re-testing best method per category on second tensor...")
        cat_best = {}
        for r in all_results:
            if r.get("status") == "OK":
                c = r.get("category", "")
                if c not in cat_best:
                    cat_best[c] = (
                        r["method"],
                        r["ratio"] * r["cosine_similarity"] ** 2,
                    )
                else:
                    curr_score = r["ratio"] * r["cosine_similarity"] ** 2
                    if curr_score > cat_best[c][1]:
                        cat_best[c] = (r["method"], curr_score)

        tensor2 = load_tensor(args.model, secondary_t)
        for cat in sorted(cat_best.keys()):
            method = cat_best[cat][0]
            idx = sum(cat_tested.values()) + 1
            print(f"\r  [{idx}] {cat:<35s} {method:<30s} on secondary...", end="")
            sys.stdout.flush()
            result = test_one(method, tensor2, timeout_s=args.timeout)
            result["tensor_name"] = secondary_t
            all_results.append(result)
            if result["status"] == "OK":
                print(
                    f"\r  [{idx}] {cat:<35s} {method:<30s} r={result['ratio']:.1f}:1 "
                    f"cos={result['cosine_similarity']:.4f} SNR={result['snr_db']:.1f}dB      "
                )
            else:
                print(
                    f"\r  [{idx}] {cat:<35s} {method:<30s} {result['status']}                     "
                )
            gc.collect()
        del tensor2
        gc.collect()

    del tensor
    gc.collect()

    # Print results
    print_results(all_results)

    # Save
    if args.output:
        print(f"\n  Saving results to {args.output}...")
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"  Saved {len(all_results)} results")

    ok_count = sum(1 for r in all_results if r["status"] == "OK")
    to_count = sum(1 for r in all_results if r["status"] == "TIMEOUT")
    err_count = sum(
        1 for r in all_results if r["status"] not in ("OK", "TIMEOUT", "NOT_FOUND")
    )
    print(f"\n{'=' * 100}")
    print(
        f"  FINAL: {len(all_results)} tests | OK: {ok_count} | Timeout: {to_count} | Errors: {err_count}"
    )
    print(f"  Time: {time.time() - start_time:.0f}s")
    print(f"{'=' * 100}")


if __name__ == "__main__":
    main()
