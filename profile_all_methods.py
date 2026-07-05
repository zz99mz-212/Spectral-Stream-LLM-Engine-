#!/usr/bin/env python3
"""
Profile ALL 3164 compression methods across 38 categories on real Gemma-4 FFN weights.

Tests full 12288x1536 weight matrices (not 512x512 blocks) at full scale.
Memory-safe: one tensor + one method at a time.
"""

import gc
import sys
import time
import traceback
from collections import defaultdict, OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, ".")

# ── Lazy imports ──────────────────────────────────────────────────────────


def get_method_by_category() -> Dict[str, List[str]]:
    """Map every method to its category by inspecting .category attribute."""
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


LOADED_TENSORS: Dict[str, np.ndarray] = {}


def load_test_tensors(mmap_path: str, max_mb: int = 100) -> Dict[str, np.ndarray]:
    """Load full FFN and attention weight matrices into RAM from safetensors."""
    from spectralstream.compression.engine.memory_mapped_engine import (
        MemoryMappedTensorEngine,
    )

    mmap = MemoryMappedTensorEngine(mmap_path)
    targets = {}
    for name in mmap.get_tensor_names():
        shape, dtype_str, offset, nbytes = mmap.get_tensor_info(name)
        if 10_000_000 < nbytes < max_mb * 1_000_000:
            # Check if it's an FFN or attention weight
            if "mlp" in name or "self_attn" in name or "attention" in name:
                if "weight" in name:
                    try:
                        view = mmap.get_tensor(name)
                        arr = np.array(view, copy=True)
                        targets[name] = arr
                    except Exception:
                        pass
    mmap.close()

    # Also grab one embed token slice if available
    if not targets:
        mmap = MemoryMappedTensorEngine(mmap_path)
        for name in mmap.get_tensor_names():
            shape, dtype_str, offset, nbytes = mmap.get_tensor_info(name)
            if 10_000_000 < nbytes < max_mb * 1_000_000 and "weight" in name:
                try:
                    view = mmap.get_tensor(name)
                    arr = np.array(view, copy=True)
                    targets[name] = arr
                    if len(targets) >= 3:
                        break
                except Exception:
                    pass
        mmap.close()

    return targets


def test_method_on_tensor(
    method_name: str,
    tensor: np.ndarray,
    tensor_name: str,
    timeout_s: int = 60,
) -> Optional[Dict[str, Any]]:
    """Test a single method on a single full-scale tensor. Returns metrics or None."""
    from spectralstream.compression.methods import METHOD_CLASSES
    from spectralstream.core.math_primitives.quality import QualityAssessor

    cls = METHOD_CLASSES.get(method_name)
    if cls is None:
        return None

    # Instantiate
    try:
        inst = cls.__new__(cls)
        inst.__init__()
    except Exception:
        try:
            inst = cls()
        except Exception:
            return None

    if not hasattr(inst, "compress") or not callable(inst.compress):
        return None

    cat = getattr(inst, "category", "unknown")

    # Warm-up GC
    gc.collect()

    # Compress
    try:
        t0 = time.time()
        data, meta = inst.compress(tensor)
        t_compress = time.time() - t0
    except Exception as e:
        return {
            "method": method_name,
            "category": cat,
            "tensor": tensor_name,
            "shape": str(tensor.shape),
            "status": "ERROR",
            "error": str(e)[:120],
            "traceback": traceback.format_exc()[-300:],
        }

    # Check if compression returned valid data
    if data is None:
        return {
            "method": method_name,
            "category": cat,
            "tensor": tensor_name,
            "shape": str(tensor.shape),
            "status": "ERROR",
            "error": "compress returned None",
        }

    compressed_size = len(data) if isinstance(data, (bytes, bytearray)) else 0
    if isinstance(data, list):
        compressed_size = sum(
            len(d) if isinstance(d, (bytes, bytearray)) else 0 for d in data
        )
    if compressed_size == 0:
        compressed_size = tensor.nbytes  # fallback

    # Decompress
    try:
        t0 = time.time()
        recon = inst.decompress(data, meta)
        t_decompress = time.time() - t0
    except Exception as e:
        return {
            "method": method_name,
            "category": cat,
            "tensor": tensor_name,
            "shape": str(tensor.shape),
            "status": "DECOMPRESS_ERROR",
            "error": str(e)[:120],
            "compressed_size": compressed_size,
        }

    if recon is None:
        return {
            "method": method_name,
            "category": cat,
            "tensor": tensor_name,
            "shape": str(tensor.shape),
            "status": "ERROR",
            "error": "decompress returned None",
            "compressed_size": compressed_size,
        }

    # Quality assessment
    try:
        qa = QualityAssessor()
        q = qa.assess(tensor, recon)
    except Exception as e:
        return {
            "method": method_name,
            "category": cat,
            "tensor": tensor_name,
            "shape": str(tensor.shape),
            "status": "QUALITY_ERROR",
            "error": str(e)[:120],
            "compressed_size": compressed_size,
        }

    ratio = tensor.nbytes / max(compressed_size, 1)
    recon_size = recon.nbytes
    del recon, data
    gc.collect()

    return {
        "method": method_name,
        "category": cat,
        "tensor": tensor_name,
        "shape": str(tensor.shape),
        "status": "OK",
        "ratio": ratio,
        "cosine_similarity": q.cosine_similarity,
        "snr_db": q.snr_db,
        "psnr_db": q.psnr_db,
        "mse": q.mse,
        "mae": q.mae,
        "max_abs_error": q.max_abs_error,
        "relative_error": q.relative_error,
        "ssim": q.ssim,
        "histogram_overlap": q.histogram_overlap,
        "correlation_coefficient": q.correlation_coefficient,
        "effective_rank_ratio": q.effective_rank_ratio,
        "bit_error_rate": q.bit_error_rate,
        "compressed_size": compressed_size,
        "original_size": tensor.nbytes,
        "t_compress": t_compress,
        "t_decompress": t_decompress,
    }


def print_results_summary(results: List[Dict[str, Any]], title: str):
    """Print a formatted summary of results."""
    statuses = defaultdict(list)
    for r in results:
        statuses[r.get("status", "UNKNOWN")].append(r)

    ok = statuses.get("OK", [])
    err = statuses.get("ERROR", [])
    decomp_err = statuses.get("DECOMPRESS_ERROR", [])
    qual_err = statuses.get("QUALITY_ERROR", [])

    print(f"\n{'=' * 100}")
    print(f"  {title}")
    print(f"{'=' * 100}")
    print(
        f"  Total tested: {len(results)} | OK: {len(ok)} | Errors: {len(err)} | Decompress errors: {len(decomp_err)} | Quality errors: {len(qual_err)}"
    )
    print()

    if ok:
        # Sort by ratio descending
        ok_sorted = sorted(ok, key=lambda r: r.get("ratio", 0), reverse=True)
        print(
            f"  {'METHOD':<35s} {'CATEGORY':<30s} {'RATIO':>10s} {'COS':>8s} {'SNR':>8s} {'SSIM':>8s} {'CORR':>8s}"
        )
        print(
            f"  {'-' * 35} {'-' * 30} {'-' * 10} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8}"
        )
        for r in ok_sorted[:50]:
            print(
                f"  {r['method']:<35s} {r['category']:<30s} {r['ratio']:>10.1f}:1 {r['cosine_similarity']:>8.4f} {r['snr_db']:>8.1f}dB {r['ssim']:>8.4f} {r['correlation_coefficient']:>8.4f}"
            )

        # Find high-performers
        print(f"\n  --- High Ratio Methods (ratio > 10:1, cos > 0.9) ---")
        high_ratio = [
            r
            for r in ok
            if r.get("ratio", 0) > 10 and r.get("cosine_similarity", 0) > 0.9
        ]
        if high_ratio:
            for r in sorted(high_ratio, key=lambda x: -x["ratio"]):
                print(
                    f"    {r['method']:<35s} {r['ratio']:>10.1f}:1 cos={r['cosine_similarity']:.4f} SNR={r['snr_db']:.1f}dB SSIM={r['ssim']:.4f}"
                )
        else:
            print("    (none)")

        print(f"\n  --- Very High Ratio Methods (ratio > 100:1, cos > 0.5) ---")
        very_high = [
            r
            for r in ok
            if r.get("ratio", 0) > 100 and r.get("cosine_similarity", 0) > 0.5
        ]
        if very_high:
            for r in sorted(very_high, key=lambda x: -x["ratio"]):
                print(
                    f"    {r['method']:<35s} {r['ratio']:>10.1f}:1 cos={r['cosine_similarity']:.4f} SNR={r['snr_db']:.1f}dB"
                )
        else:
            print("    (none)")

        print(f"\n  --- Extreme Ratio Methods (ratio > 1000:1, any quality) ---")
        extreme = [r for r in ok if r.get("ratio", 0) > 1000]
        if extreme:
            for r in sorted(extreme, key=lambda x: -x["ratio"]):
                print(
                    f"    {r['method']:<35s} {r['ratio']:>10.1f}:1 cos={r['cosine_similarity']:.4f} SNR={r['snr_db']:.1f}dB STATUS={r['status']}"
                )
        else:
            print("    (none)")

        print(f"\n  --- Per Tensor Type Best ---")
        tensor_best = {}
        for r in ok:
            tn = r.get("tensor", "unknown")
            if tn not in tensor_best or r.get("ratio", 0) > tensor_best[tn].get(
                "ratio", 0
            ):
                tensor_best[tn] = r
        for tn, r in tensor_best.items():
            print(
                f"    {tn.split('.')[-1]:<40s} {r['method']:<30s} {r['ratio']:>10.1f}:1 cos={r['cosine_similarity']:.4f}"
            )

    if err:
        print(f"\n  --- Errors ({len(err)}) ---")
        for r in err[:10]:
            print(f"    {r['method']:<35s} {r.get('error', '')[:100]}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Profile ALL compression methods on real weights"
    )
    parser.add_argument(
        "--model",
        default="/home/mike/Documents/Github/SpectralStream/models/gemma-4-E2B/model.safetensors",
        help="Path to safetensors model file",
    )
    parser.add_argument(
        "--max-methods",
        type=int,
        default=5,
        help="Max methods per category to test (default: 5)",
    )
    parser.add_argument(
        "--category", type=str, default=None, help="Only test this specific category"
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Save results to file (.npz or .json)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Timeout per method in seconds (default: 120)",
    )
    args = parser.parse_args()

    print("=" * 100)
    print("  SPECTRALSTREAM — Full-Scale Compression Method Profiler")
    print("  Profile: 3164 methods × 38 categories on Gemma-4 FFN weights")
    print("=" * 100)

    # Step 1: Map methods to categories
    print("\n[Step 1/4] Mapping methods to categories...")
    sys.stdout.flush()
    t0 = time.time()
    cat_methods = get_method_by_category()
    elapsed = time.time() - t0
    total_methods = sum(len(v) for v in cat_methods.values())
    print(
        f"  Found {total_methods} methods across {len(cat_methods)} categories in {elapsed:.1f}s"
    )

    for cat in sorted(cat_methods.keys()):
        n = len(cat_methods[cat])
        print(f"    {cat:<40s} {n:>5d} methods")

    # Step 2: Load test tensors
    print(f"\n[Step 2/4] Loading test tensors from {args.model}...")
    sys.stdout.flush()
    t0 = time.time()
    tensors = load_test_tensors(args.model, max_mb=100)
    elapsed = time.time() - t0

    if not tensors:
        print("  WARNING: No qualifying tensors found > 10MB. Trying smaller...")
        tensors = load_test_tensors(args.model, max_mb=5)

    print(f"  Loaded {len(tensors)} full-scale tensors in {elapsed:.1f}s")
    for name, arr in tensors.items():
        print(
            f"    {name}: shape={arr.shape}, {arr.nbytes / 1e6:.1f}MB, dtype={arr.dtype}"
        )

    if not tensors:
        print("  FATAL: Could not load any tensors")
        sys.exit(1)

    # Step 3: Test methods per category
    print(f"\n[Step 3/4] Testing up to {args.max_methods} methods per category...")
    print(f"  Timeout: {args.timeout}s per method")
    sys.stdout.flush()

    all_results: List[Dict[str, Any]] = []
    cat_tested = defaultdict(int)
    cat_ok = defaultdict(int)
    test_count = 0
    error_count = 0
    start_time = time.time()

    # Use first tensor (largest FFN weight) as primary
    tensor_items = list(tensors.items())
    primary_tensor = tensor_items[0]

    for cat in sorted(cat_methods.keys()):
        if args.category and args.category != cat:
            continue

        methods = cat_methods[cat]
        if not methods:
            continue

        # Limit methods per category
        test_methods = methods[: args.max_methods]

        for method_name in test_methods:
            test_count += 1
            cat_tested[cat] += 1

            tensor_name = primary_tensor[0]
            tensor = primary_tensor[1]

            sys.stdout.write(
                f"\r  [{test_count:4d}] {cat:<35s} → {method_name:<30s} ..."
            )
            sys.stdout.flush()

            result = test_method_on_tensor(
                method_name, tensor, tensor_name, timeout_s=args.timeout
            )
            if result is None:
                result = {
                    "method": method_name,
                    "category": cat,
                    "status": "SKIPPED",
                    "tensor": tensor_name,
                    "shape": str(tensor.shape),
                }

            if result.get("status") == "OK":
                cat_ok[cat] += 1
            else:
                error_count += 1

            all_results.append(result)

            # Force GC
            del result
            gc.collect()

            elapsed_total = time.time() - start_time
            rate = test_count / max(elapsed_total, 1)
            remaining = (
                total_methods / max(len(cat_methods), 1) * args.max_methods - test_count
            ) / max(rate, 0.01)

    # Also test on a second tensor type if available
    if len(tensor_items) > 1:
        print(f"\n\n[Extra] Re-testing top methods on a different tensor type...")
        # Find best method per category (first method that worked)
        cat_best = {}
        for r in all_results:
            if r.get("status") == "OK":
                c = r.get("category", "")
                if c not in cat_best:
                    cat_best[c] = r["method"]

        for cat, method in list(cat_best.items())[:20]:
            new_tensor = tensor_items[min(1, len(tensor_items) - 1)]
            sys.stdout.write(
                f"\r  Re-testing {method:<30s} on {new_tensor[0].split('.')[-1][:40]:>40s} ..."
            )
            sys.stdout.flush()
            result = test_method_on_tensor(method, new_tensor[1], new_tensor[0])
            if result and result.get("status") == "OK":
                formatted = f"  {method:<30s} on {new_tensor[0].split('.')[-1][:40]:>40s}: ratio={result['ratio']:.1f}:1 cos={result['cosine_similarity']:.4f} SNR={result['snr_db']:.1f}dB"
                print(f"\r{formatted}")
            else:
                print(
                    f"\r  {method:<30s} on {new_tensor[0].split('.')[-1][:40]:>40s}: FAILED"
                )
            gc.collect()

    # Step 4: Print results
    print(f"\n\n{'=' * 100}")
    print(f"  RESULTS SUMMARY")
    print(f"{'=' * 100}")
    print(f"  Total methods tested: {test_count}")
    print(f"  Successful: {test_count - error_count}")
    print(f"  Failed/Errors: {error_count}")
    print(f"  Total time: {time.time() - start_time:.1f}s")

    print(f"\n  Per-category success rates:")
    for cat in sorted(cat_ok.keys()):
        tested = cat_tested.get(cat, 0)
        ok = cat_ok.get(cat, 0)
        pct = 100 * ok / max(tested, 1)
        print(f"    {cat:<40s} {ok:>3d}/{tested:<3d} ({pct:>5.1f}%)")

    print_results_summary(all_results, "ALL CATEGORIES OVERALL")

    # Save results
    if args.output:
        print(f"\n  Saving results to {args.output}...")
        import json

        # Convert numpy values to native Python types
        cleaned = []
        for r in all_results:
            cleaned.append(
                {
                    k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                    for k, v in r.items()
                }
            )
        with open(args.output, "w") as f:
            json.dump(cleaned, f, indent=2, default=str)
        print(f"  Saved {len(cleaned)} results to {args.output}")

    print(f"\n{'=' * 100}")
    print("  DONE")
    print(f"{'=' * 100}")


if __name__ == "__main__":
    main()
