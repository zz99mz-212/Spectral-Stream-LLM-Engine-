"""Comprehensive method validation — tests every registered method on a random tensor.

Reports: method name, category, tier, ratio, error, SNR, loss_type, precision_bits.

Usage:
    from spectralstream.compression.engine.method_validation import validate_all_methods
    results = validate_all_methods()
"""

from __future__ import annotations

import gc
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spectralstream.compression.engine.method_discovery import MethodDiscovery


logger = logging.getLogger(__name__)


def snr_db(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Compute Signal-to-Noise Ratio in dB."""
    signal = np.linalg.norm(original.ravel())
    noise = np.linalg.norm(original.ravel() - reconstructed.ravel())
    if noise < 1e-30 or signal < 1e-30:
        return 100.0
    return float(20 * np.log10(signal / noise))


def validate_single_method(
    method_name: str, method_info: Dict[str, Any], tensor: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """Validate a single method on a tensor, returning rich metrics.

    Returns dict with keys:
        name, category, tier, loss_type, precision_preserved_bits,
        works, ratio, error, snr_db, compress_time_ms, decompress_time_ms
    """
    result: Dict[str, Any] = {
        "name": method_name,
        "category": method_info.get("category", "unknown"),
        "tier": method_info.get("tier", None),
        "loss_type": method_info.get("loss_type", "unknown"),
        "precision_preserved_bits": method_info.get("precision_preserved_bits", 0),
        "works": False,
        "ratio": 0.0,
        "error": 1.0,
        "snr_db": 0.0,
        "compress_time_ms": 0.0,
        "decompress_time_ms": 0.0,
    }

    if tensor is None:
        tensor = np.random.RandomState(42).randn(16, 16).astype(np.float32)

    inst = method_info.get("instance")
    method_cls = method_info.get("class")
    if inst is None and method_cls is not None:
        try:
            inst = method_cls() if isinstance(method_cls, type) else method_cls
        except Exception:
            return result

    if inst is None:
        return result

    try:
        t0 = time.perf_counter()
        data, meta = inst.compress(tensor)
        t1 = time.perf_counter()
        recon = inst.decompress(data, meta)
        t2 = time.perf_counter()

        if recon.shape != tensor.shape:
            recon = recon.reshape(tensor.shape)

        ratio = max(
            tensor.nbytes / max(len(data) if isinstance(data, bytes) else 1, 1), 1.0
        )
        err = float(
            np.linalg.norm(recon.ravel() - tensor.ravel())
            / max(np.linalg.norm(tensor.ravel()), 1e-30)
        )
        s = snr_db(tensor, recon)

        result["works"] = True
        result["ratio"] = ratio
        result["error"] = err
        result["snr_db"] = s
        result["compress_time_ms"] = (t1 - t0) * 1000.0
        result["decompress_time_ms"] = (t2 - t1) * 1000.0

        method_info["validated"] = True
        method_info["validated_ratio"] = ratio
        method_info["validated_error"] = err
        method_info["validated_snr_db"] = s

    except Exception:
        pass

    gc.collect()
    return result


def validate_all_methods(
    max_methods: Optional[int] = None,
    batch_size: int = 20,
    tensor: Optional[np.ndarray] = None,
    verbose: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """Validate ALL discovered methods, returning rich metrics for each.

    Args:
        max_methods: Limit validation to first N methods (None = all).
        batch_size: GC batch size for memory safety.
        tensor: Custom test tensor (default: 16x16 random float32).
        verbose: Print progress during validation.

    Returns:
        Dict mapping method_name -> result dict with keys:
            name, category, tier, loss_type, precision_preserved_bits,
            works, ratio, error, snr_db, compress_time_ms, decompress_time_ms
    """
    methods = MethodDiscovery.discover()
    items = list(methods.items())
    if max_methods is not None:
        items = items[:max_methods]

    results: Dict[str, Dict[str, Any]] = {}
    total = len(items)
    working = 0

    for i in range(0, total, batch_size):
        batch = items[i : i + batch_size]
        for mname, minfo in batch:
            res = validate_single_method(mname, minfo, tensor=tensor)
            results[mname] = res
            if res["works"]:
                working += 1

        if verbose:
            pct = min((i + batch_size) * 100 // total, 100)
            print(
                f"  Validated {min(i + batch_size, total)}/{total} ({pct}%) — {working} working"
            )
        gc.collect()

    if verbose:
        print(f"\n=== Summary: {working}/{total} methods working ===")

    return results


def summarize_results(results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate validation results into a summary dict."""
    total = len(results)
    working = sum(1 for r in results.values() if r["works"])
    by_loss_type: Dict[str, int] = {}
    by_category: Dict[str, Dict[str, int]] = {}
    avg_ratio = 0.0
    avg_error = 0.0
    avg_snr = 0.0
    w_count = 0

    for r in results.values():
        lt = r.get("loss_type", "unknown")
        by_loss_type[lt] = by_loss_type.get(lt, 0) + 1

        cat = r.get("category", "unknown")
        if cat not in by_category:
            by_category[cat] = {"total": 0, "working": 0}
        by_category[cat]["total"] += 1
        if r["works"]:
            by_category[cat]["working"] += 1

        if r["works"]:
            avg_ratio += r["ratio"]
            avg_error += r["error"]
            avg_snr += r["snr_db"]
            w_count += 1

    avg_ratio = avg_ratio / max(w_count, 1)
    avg_error = avg_error / max(w_count, 1)
    avg_snr = avg_snr / max(w_count, 1)

    return {
        "total": total,
        "working": working,
        "pass_rate": working / max(total, 1) * 100,
        "avg_ratio": avg_ratio,
        "avg_error": avg_error,
        "avg_snr_db": avg_snr,
        "by_loss_type": by_loss_type,
        "by_category": by_category,
    }


def print_report(results: Dict[str, Dict[str, Any]]) -> None:
    """Print a human-readable validation report."""
    summary = summarize_results(results)
    print("=" * 90)
    print(f"METHOD VALIDATION REPORT — {summary['total']} methods")
    print("=" * 90)
    print(
        f"  Working: {summary['working']}/{summary['total']} ({summary['pass_rate']:.1f}%)"
    )
    print(f"  Avg ratio: {summary['avg_ratio']:.2f}x")
    print(f"  Avg error: {summary['avg_error']:.6f}")
    print(f"  Avg SNR:   {summary['avg_snr_db']:.2f} dB")
    print()
    print("  By loss type:")
    for lt, cnt in sorted(summary["by_loss_type"].items(), key=lambda x: -x[1]):
        print(f"    {lt:20s}: {cnt:5d}")
    print()
    print("  By category (working/total):")
    for cat, info in sorted(
        summary["by_category"].items(), key=lambda x: -x[1]["total"]
    ):
        print(
            f"    {cat:30s}: {info['working']:4d}/{info['total']:4d}"
            f"  ({info['working'] / max(info['total'], 1) * 100:5.1f}%)"
        )
    print()

    print(
        f"  {'Status':8s} | {'Method':40s} | {'Cat':20s} | {'Tier':4s} | {'Loss':14s} | {'Bits':4s} | {'Ratio':8s} | {'Error':10s} | {'SNR':8s}"
    )
    print("-" * 90)
    for mname in sorted(results.keys()):
        r = results[mname]
        status = "OK" if r["works"] else "FAIL"
        cat = r["category"][:20]
        tier = (
            str(r["tier"].value if hasattr(r["tier"], "value") else r["tier"])
            if r["tier"]
            else "?"
        )
        lt = r["loss_type"][:14]
        bits = str(r["precision_preserved_bits"])
        ratio = f"{r['ratio']:.1f}x" if r["works"] else "-"
        err = f"{r['error']:.6f}" if r["works"] else "-"
        snr_val = f"{r['snr_db']:.1f}" if r["works"] else "-"
        print(
            f"  {status:8s} | {mname:40s} | {cat:20s} | {tier:4s} | {lt:14s} | {bits:4s} | {ratio:8s} | {err:10s} | {snr_val:8s}"
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    results = validate_all_methods(max_methods=200, verbose=True)
    print_report(results)
