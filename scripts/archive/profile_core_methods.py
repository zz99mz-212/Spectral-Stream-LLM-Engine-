#!/usr/bin/env python3
"""
Profile compression methods on REAL full-scale FFN weights (12288x1536).
Tests each method in a subprocess with timeout to prevent hanging.
"""

import gc
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, ".")

# ── Core categories to test (skip breakthrough/massive which hang) ─────────

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

# Categories that are known to have slow/hanging methods - test fewer
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
    """Map all methods to categories by category attribute."""
    from spectralstream.compression.methods import METHOD_CLASSES

    cat_methods: Dict[str, List[str]] = defaultdict(list)
    for name in list(METHOD_CLASSES.keys()):
        try:
            cls = METHOD_CLASSES[name]
            if isinstance(cls, type):
                try:
                    inst = cls.__new__(cls)
                    inst.__init__()
                except Exception:
                    try:
                        inst = cls()
                    except Exception:
                        continue
            else:
                inst = cls
            cat = getattr(inst, "category", "unknown")
            cat_methods[cat].append(name)
        except Exception:
            cat_methods["unknown"].append(name)
    return dict(cat_methods)


def get_tensor_info(mmap_path: str) -> List[Tuple[str, tuple, str, int]]:
    """Get tensor info without loading."""
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


def write_method_test_script(
    method_name: str,
    tensor_path: str,
    tensor_name: str,
    output_path: str,
    timeout_s: int = 120,
) -> str:
    """Write a standalone test script for a single method on a tensor."""
    script = f'''#!/usr/bin/env python3
"""Test {method_name} on {tensor_name}."""
import gc, json, os, sys, time, traceback
import numpy as np

sys.path.insert(0, '.')

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

def run_test():
    from spectralstream.compression.engine.memory_mapped_engine import MemoryMappedTensorEngine
    from spectralstream.compression.methods import METHOD_CLASSES
    from spectralstream.core.math_primitives.quality import QualityAssessor

    # Load tensor
    mmap = MemoryMappedTensorEngine({json.dumps(tensor_path)})
    view = mmap.get_tensor({json.dumps(tensor_name)})
    tensor = np.array(view, copy=True)
    mmap.close()

    # Get method
    cls = METHOD_CLASSES.get({json.dumps(method_name)})
    if cls is None:
        return {{"status": "NOT_FOUND"}}

    try:
        inst = cls.__new__(cls)
        inst.__init__()
    except Exception:
        try:
            inst = cls()
        except Exception as e:
            return {{"status": "INSTANTIATE_ERROR", "error": str(e)[:200]}}

    cat = getattr(inst, 'category', 'unknown')
    mname = getattr(inst, 'name', method_name)

    # Compress
    try:
        t0 = time.time()
        data, meta = inst.compress(tensor)
        tc = time.time() - t0
    except Exception as e:
        return {{"status": "COMPRESS_ERROR", "method": mname, "category": cat,
                 "error": str(e)[:200], "traceback": traceback.format_exc()[-300:]}}

    if data is None:
        return {{"status": "NULL_DATA", "method": mname, "category": cat}}

    # Measure compressed size
    if isinstance(data, (bytes, bytearray)):
        csize = len(data)
    elif isinstance(data, list):
        csize = sum(len(d) if isinstance(d, (bytes, bytearray)) else 0 for d in data)
    else:
        csize = sys.getsizeof(data)
    if csize == 0:
        csize = tensor.nbytes

    # Decompress
    try:
        t0 = time.time()
        recon = inst.decompress(data, meta)
        td = time.time() - t0
    except Exception as e:
        return {{"status": "DECOMPRESS_ERROR", "method": mname, "category": cat,
                 "error": str(e)[:200], "compressed_size": csize,
                 "t_compress": tc}}

    if recon is None:
        return {{"status": "NULL_RECON", "method": mname, "category": cat,
                 "compressed_size": csize, "t_compress": tc}}

    # Quality
    try:
        qa = QualityAssessor()
        q = qa.assess(tensor, recon)
    except Exception as e:
        return {{"status": "QUALITY_ERROR", "method": mname, "category": cat,
                 "error": str(e)[:200], "compressed_size": csize,
                 "t_compress": tc, "t_decompress": td}}

    ratio = tensor.nbytes / max(csize, 1)
    del recon, data, tensor
    gc.collect()

    return {{
        "status": "OK",
        "method": mname,
        "category": cat,
        "tensor": {json.dumps(tensor_name)},
        "shape": str(tensor.shape),
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
    }}

result = run_test()
with open({json.dumps(output_path)}, 'w') as f:
    json.dump(result, f)
'''
    return script


def test_method(
    method_name: str, tensor_path: str, tensor_name: str, timeout_s: int = 120
) -> Optional[Dict[str, Any]]:
    """Test a single method in a subprocess with timeout."""
    output_path = f"/tmp/_method_test_{method_name}.json"
    script = write_method_test_script(
        method_name, tensor_path, tensor_name, output_path, timeout_s
    )

    # Write script
    script_path = f"/tmp/_run_{method_name}.py"
    with open(script_path, "w") as f:
        f.write(script)

    try:
        proc = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            timeout=timeout_s,
            env={
                **os.environ,
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
                "PYTHONWARNINGS": "ignore",
            },
        )
    except subprocess.TimeoutExpired:
        # Cleanup
        try:
            os.remove(script_path)
        except OSError:
            pass
        try:
            os.remove(output_path)
        except OSError:
            pass
        return {
            "method": method_name,
            "status": "TIMEOUT",
            "tensor": tensor_name,
        }
    except Exception as e:
        return {
            "method": method_name,
            "status": "SUBPROCESS_ERROR",
            "error": str(e)[:200],
            "tensor": tensor_name,
        }

    # Read result
    result = None
    if os.path.exists(output_path):
        try:
            with open(output_path, "r") as f:
                result = json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
        try:
            os.remove(output_path)
        except OSError:
            pass

    try:
        os.remove(script_path)
    except OSError:
        pass

    if result is None:
        stderr = proc.stderr.decode()[-500:] if proc.stderr else ""
        return {
            "method": method_name,
            "status": "NO_OUTPUT",
            "stderr": stderr,
            "tensor": tensor_name,
        }

    result.setdefault("method", method_name)
    result.setdefault("tensor", tensor_name)
    return result


def print_results(results: List[Dict[str, Any]], title: str):
    """Print formatted results."""
    by_status = defaultdict(list)
    for r in results:
        by_status[r.get("status", "UNKNOWN")].append(r)

    ok = by_status.get("OK", [])
    timeout = by_status.get("TIMEOUT", [])
    not_found = by_status.get("NOT_FOUND", [])
    errors = [
        r for r in results if r.get("status") not in ("OK", "TIMEOUT", "NOT_FOUND")
    ]

    print(f"\n{'=' * 100}")
    print(f"  {title}")
    print(f"{'=' * 100}")
    print(
        f"  Tested: {len(results)} | OK: {len(ok)} | Timeout: {len(timeout)} | "
        f"Not found: {len(not_found)} | Errors: {len(errors)}"
    )

    if ok:
        # Sort by ratio
        ok_sorted = sorted(ok, key=lambda r: r.get("ratio", 0), reverse=True)
        print(
            f"\n  {'METHOD':<35s} {'CATEGORY':<30s} {'RATIO':>10s} {'COS':>8s} {'SNR':>8s} "
            f"{'SSIM':>8s} {'TIME':>8s}"
        )
        print(
            f"  {'-' * 35} {'-' * 30} {'-' * 10} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8}"
        )
        for r in ok_sorted[:60]:
            tc = r.get("t_compress", 0)
            print(
                f"  {r['method']:<35s} {r['category']:<30s} {r['ratio']:>10.1f}:1 "
                f"{r.get('cosine_similarity', 0):>8.4f} {r.get('snr_db', 0):>8.1f}dB "
                f"{r.get('ssim', 0):>8.4f} {tc:>7.1f}s"
            )

        # High performers
        print(f"\n  >>> Methods with ratio > 10:1 AND cos > 0.9:")
        high = [
            r
            for r in ok
            if r.get("ratio", 0) > 10 and r.get("cosine_similarity", 0) > 0.9
        ]
        if high:
            for r in sorted(high, key=lambda x: -x["ratio"]):
                print(
                    f"    {r['method']:<35s} {r['ratio']:>10.1f}:1 cos={r['cosine_similarity']:.4f} "
                    f"SNR={r['snr_db']:.1f}dB SSIM={r['ssim']:.4f}"
                )
        else:
            print("    (none)")

        print(f"\n  >>> Methods with ratio > 100:1 AND cos > 0.5:")
        high2 = [
            r
            for r in ok
            if r.get("ratio", 0) > 100 and r.get("cosine_similarity", 0) > 0.5
        ]
        if high2:
            for r in sorted(high2, key=lambda x: -x["ratio"]):
                print(
                    f"    {r['method']:<35s} {r['ratio']:>10.1f}:1 cos={r['cosine_similarity']:.4f} "
                    f"SNR={r['snr_db']:.1f}dB"
                )
        else:
            print("    (none)")

        print(f"\n  >>> Methods with ratio > 1000:1 (any quality):")
        high3 = [r for r in ok if r.get("ratio", 0) > 1000]
        if high3:
            for r in sorted(high3, key=lambda x: -x["ratio"]):
                print(
                    f"    {r['method']:<35s} {r['ratio']:>10.1f}:1 cos={r['cosine_similarity']:.4f} "
                    f"SNR={r['snr_db']:.1f}dB BER={r.get('bit_error_rate', 'N/A')}"
                )
        else:
            print("    (none)")

        print(f"\n  >>> Methods with ratio > 10:1 AND SNR > 20dB:")
        high4 = [r for r in ok if r.get("ratio", 0) > 10 and r.get("snr_db", 0) > 20]
        if high4:
            for r in sorted(high4, key=lambda x: -x["snr_db"]):
                print(
                    f"    {r['method']:<35s} {r['ratio']:>10.1f}:1 cos={r['cosine_similarity']:.4f} "
                    f"SNR={r['snr_db']:.1f}dB BER={r.get('bit_error_rate', 'N/A')}"
                )
        else:
            print("    (none)")

        # Per tensor best
        print(f"\n  >>> Best method overall (highest ratio * cos^2):")
        scored = [
            (r.get("ratio", 0) * r.get("cosine_similarity", 0) ** 2, r) for r in ok
        ]
        scored.sort(key=lambda x: -x[0])
        for score, r in scored[:10]:
            print(
                f"    {r['method']:<35s} score={score:.1f} ratio={r['ratio']:.1f}:1 "
                f"cos={r['cosine_similarity']:.4f} SNR={r['snr_db']:.1f}dB"
            )

    # Errors
    if errors:
        print(f"\n  --- Errors ({len(errors)}) ---")
        for r in errors[:15]:
            print(
                f"    {r.get('method', '?'):<35s} [{r.get('status', '?')}] {r.get('error', '')[:100]}"
            )


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Profile core compression methods on real weights"
    )
    parser.add_argument(
        "--model",
        default="/home/mike/Documents/Github/SpectralStream/models/gemma-4-E2B/model.safetensors",
    )
    parser.add_argument(
        "--methods-per-category",
        type=int,
        default=3,
        help="Methods per category to test (default: 3)",
    )
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--output", type=str, default="/tmp/profile_results.json")
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Timeout per method in seconds (default: 120)",
    )
    parser.add_argument(
        "--second-tensor", action="store_true", help="Also test on second tensor type"
    )
    args = parser.parse_args()

    print("=" * 100)
    print("  SPECTRALSTREAM FULL-SCALE COMPRESSION PROFILER")
    print(f"  Model: {args.model}")
    print(f"  Testing: {args.methods_per_category} methods/category")
    print("=" * 100)

    # Step 1: Discover methods
    print("\n[1/4] Discovering methods and categories...")
    t0 = time.time()
    cat_methods = discover_methods()
    print(
        f"  Found {sum(len(v) for v in cat_methods.values())} methods across "
        f"{len(cat_methods)} categories ({time.time() - t0:.1f}s)"
    )

    # Filter to core categories
    filtered = {}
    for cat in CORE_CATEGORIES:
        if cat in cat_methods:
            filtered[cat] = cat_methods[cat]
        else:
            print(f"  WARNING: Category '{cat}' not found in discovered methods")

    # Print method counts
    for cat in sorted(filtered.keys()):
        n = len(filtered[cat])
        mark = " *RISKY*" if cat in RISKY_CATEGORIES else ""
        print(f"    {cat:<40s} {n:>5d} methods{mark}")

    # Step 2: Get tensor info
    print(f"\n[2/4] Scanning model for test tensors...")
    tensors = get_tensor_info(args.model)
    print(f"  Found {len(tensors)} qualifying tensors")

    # Show unique shapes
    shapes_seen = {}
    for name, shape, dtype, nbytes in tensors:
        key = str(shape)
        if key not in shapes_seen:
            shapes_seen[key] = (dtype, nbytes, name)
            print(f"    {name}: shape={shape}, {nbytes / 1e6:.1f}MB, dtype={dtype}")

    if not tensors:
        print("  FATAL: No tensors found")
        sys.exit(1)

    # Pick test tensors: one rectangular (gate/up_proj) and one tall (down_proj)
    primary = None
    secondary = None
    for name, shape, dtype, nbytes in tensors:
        if primary is None:
            primary = (name, shape, dtype, nbytes)
        elif secondary is None and shape != primary[1]:
            secondary = (name, shape, dtype, nbytes)
        if primary and secondary:
            break

    if primary is None:
        primary = tensors[0]
    print(
        f"\n  Primary tensor:  {primary[0]} ({primary[1]} = {primary[3] / 1e6:.1f}MB)"
    )
    if secondary:
        print(
            f"  Secondary tensor: {secondary[0]} ({secondary[1]} = {secondary[3] / 1e6:.1f}MB)"
        )

    # Step 3: Test methods
    print(f"\n[3/4] Testing up to {args.methods_per_category} methods per category...")
    print(f"  Timeout: {args.timeout}s per method (subprocess isolated)")
    sys.stdout.flush()

    start_time = time.time()
    all_results: List[Dict[str, Any]] = []
    cat_ok = defaultdict(int)
    cat_tested = defaultdict(int)

    for cat in sorted(filtered.keys()):
        if args.category and args.category != cat:
            continue

        methods = filtered[cat]
        test_methods = methods[: args.methods_per_category]

        for i, method_name in enumerate(test_methods):
            cat_tested[cat] += 1
            elapsed = time.time() - start_time
            idx = sum(cat_tested.values())
            total_planned = args.methods_per_category * len(filtered)

            print(
                f"\r  [{idx:4d}/{total_planned}] {cat:<35s} {method_name:<30s} "
                f"({elapsed:.0f}s elapsed)...",
                end="",
            )
            sys.stdout.flush()

            result = test_method(
                method_name, args.model, primary[0], timeout_s=args.timeout
            )
            all_results.append(result)

            if result.get("status") == "OK":
                cat_ok[cat] += 1
            elif result.get("status") == "TIMEOUT":
                pass  # already counted as tested
            del result
            gc.collect()

        # Per-category summary
        ok_in_cat = cat_ok[cat]
        tested_in_cat = cat_tested[cat]
        if ok_in_cat > 0:
            cat_results = [
                r for r in all_results[-tested_in_cat:] if r.get("status") == "OK"
            ]
            if cat_results:
                best = max(
                    cat_results,
                    key=lambda r: r.get("ratio", 0)
                    * r.get("cosine_similarity", 0) ** 2,
                )
                print(
                    f"\r  [{idx:4d}] {cat:<35s} BEST: {best['method']:<25s} "
                    f"ratio={best['ratio']:.1f}:1 cos={best['cosine_similarity']:.4f}    "
                )

    total_elapsed = time.time() - start_time
    print(f"\n  Core profiling completed in {total_elapsed:.0f}s")

    # Step 3b: Re-test best method from each category on second tensor
    if args.second_tensor and secondary:
        print(
            f"\n[3b/4] Re-testing best method per category on {secondary[0].split('.')[-1]}..."
        )
        cat_best = {}
        for r in all_results:
            if r.get("status") == "OK":
                c = r.get("category", "")
                if c not in cat_best:
                    cat_best[c] = r["method"]

        for cat, method in sorted(cat_best.items()):
            idx = sum(cat_tested.values()) + 1
            print(f"\r  [{idx}] {cat:<35s} {method:<30s} on secondary...", end="")
            sys.stdout.flush()
            result = test_method(
                method, args.model, secondary[0], timeout_s=args.timeout
            )
            all_results.append(result)
            if result.get("status") == "OK":
                print(
                    f"\r  [{idx}] {cat:<35s} {method:<30s} "
                    f"ratio={result['ratio']:.1f}:1 cos={result['cosine_similarity']:.4f} "
                    f"SNR={result['snr_db']:.1f}dB    "
                )
            else:
                print(
                    f"\r  [{idx}] {cat:<35s} {method:<30s} FAILED ({result.get('status')})"
                )
            gc.collect()

    # Step 4: Print results
    print(f"\n[4/4] Results...")
    print_results(all_results, "FULL-SCALE PROFILING RESULTS")

    # Save
    if args.output:
        print(f"\n  Saving results to {args.output}...")
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"  Saved {len(all_results)} results to {args.output}")

    # Summary
    ok_count = sum(1 for r in all_results if r.get("status") == "OK")
    timeout_count = sum(1 for r in all_results if r.get("status") == "TIMEOUT")
    err_count = sum(
        1 for r in all_results if r.get("status") not in ("OK", "TIMEOUT", "NOT_FOUND")
    )
    print(f"\n{'=' * 100}")
    print(f"  FINAL SUMMARY")
    print(
        f"  Total tests: {len(all_results)} | OK: {ok_count} | Timeout: {timeout_count} | Errors: {err_count}"
    )
    print(f"  Total time: {time.time() - start_time:.0f}s")
    print(f"{'=' * 100}")


if __name__ == "__main__":
    main()
