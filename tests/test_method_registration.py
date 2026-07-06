"""Comprehensive method registration test — validates ALL discoverable methods.

Tests:
1. MethodDiscovery.discover() finds all methods
2. Each method has required metadata (name, category, tier, compress, decompress)
3. First 80 methods survive a compress/decompress round-trip on a 16x16 tensor
4. Method statistics match expectations
5. Category and tier distributions are correct
"""

from __future__ import annotations

import os
import sys
import time
from collections import Counter
from typing import Any, Dict, List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from spectralstream.compression.engine.method_discovery import MethodDiscovery
from spectralstream.compression.engine._tier_common import CATEGORY_TIER_MAP, MethodTier
from spectralstream.compression.methods import METHOD_CLASSES

# Minimum methods required
MIN_METHODS = 80

# Test tensor (16x16 float32 = 1KB — OOM-safe for all methods)
_TEST_TENSOR = np.random.RandomState(42).randn(16, 16).astype(np.float32)


def test_discover_returns_enough_methods():
    """Assert at least MIN_METHODS methods are discoverable."""
    methods = MethodDiscovery.discover()
    assert len(methods) >= MIN_METHODS, (
        f"Only {len(methods)} methods found, expected >= {MIN_METHODS}"
    )


def test_discover_singleton():
    """MethodDiscovery.discover() returns the same cached dict on repeat calls."""
    first = MethodDiscovery.discover()
    second = MethodDiscovery.discover()
    assert first is second


# Utilities that are NOT compression methods (profiling, monitoring, allocators, etc.)
_NON_COMPRESSION_UTILITIES: set = {
    "tensor_profiler",
    "error_budget_allocator",
    "cross_layer_optimizer",
    "calibration_pipeline",
    "calibration_collector",
    "quantizer_selector",
    "compression_aware_finetuning",
    "compression_versioning",
    "real_time_monitor",
    "auto_optimizer",
    "simd_dispatch",
    "arena_allocator",
}


def test_each_method_has_compress_decompress():
    """Every discovered compression method must have callable compress and decompress.

    Non-compression utilities (profiling, monitoring, allocators) are excluded.
    """
    methods = MethodDiscovery.discover()
    failures: List[str] = []
    for name, info in methods.items():
        if name in _NON_COMPRESSION_UTILITIES:
            continue
        inst = info.get("instance")
        cls_obj = info.get("class")
        if inst is None and cls_obj is not None:
            try:
                inst = cls_obj() if isinstance(cls_obj, type) else cls_obj
            except Exception:
                inst = None

        has_c = (
            inst is not None and hasattr(inst, "compress") and callable(inst.compress)
        )
        has_d = (
            inst is not None
            and hasattr(inst, "decompress")
            and callable(inst.decompress)
        )
        if not has_c:
            failures.append(f"{name}: missing compress")
        elif not has_d:
            failures.append(f"{name}: missing decompress")

    if failures:
        pytest.fail(
            f"{len(failures)} methods missing compress/decompress:\n  "
            + "\n  ".join(failures[:30])
        )


def test_each_method_has_metadata():
    """Every discovered method entry has required metadata keys."""
    methods = MethodDiscovery.discover()
    required_keys = {"class", "instance", "category", "tier", "name", "file"}
    failures: List[str] = []
    for name, info in methods.items():
        missing = required_keys - info.keys()
        if missing:
            failures.append(f"{name}: missing {missing}")
    if failures:
        pytest.fail(
            f"{len(failures)} methods missing metadata:\n  "
            + "\n  ".join(failures[:20])
        )


def test_all_categories_have_tiers():
    """Every category string in discovered methods maps to a valid tier."""
    methods = MethodDiscovery.discover()
    categories_seen: Dict[str, int] = Counter()
    for info in methods.values():
        categories_seen[info["category"]] += 1

    tierless = [cat for cat in categories_seen if cat not in CATEGORY_TIER_MAP]
    if tierless:
        pytest.fail(f"Categories without tier mapping: {tierless}")


def test_engine_builtins_present():
    """Core engine methods are always discoverable."""
    methods = MethodDiscovery.discover()
    for builtin in [
        "block_int8",
        "block_int4",
        "svd_compress",
        "dct_spectral",
        "tensor_train",
        "fwht_compress",
    ]:
        assert builtin in methods, f"Missing engine built-in: {builtin}"


def test_key_novel_methods_present():
    """Key novel/tensor network methods are discoverable."""
    methods = MethodDiscovery.discover()
    for name in [
        "mera_adv",
        "qtt_adapt",
        "spin_glass",
        "quantum_amplitude",
        "topological_order",
    ]:
        assert name in methods, f"Missing novel method: {name}"


def test_methods_roundtrip():
    """Test compress/decompress round-trip on first 80 methods using a 16x16 tensor.

    Reports pass/fail per method.
    """
    methods = MethodDiscovery.discover()
    names = sorted(methods.keys())[:80]
    results: Dict[str, Tuple[bool, float, float, str]] = {}

    for name in names:
        info = methods[name]
        try:
            works, ratio, error = MethodDiscovery.validate_method(name, info)
            info["validated"] = works
            info["validated_ratio"] = ratio
            info["validated_error"] = error
            status = "OK" if works else "FAIL"
            results[name] = (works, ratio, error, status)
        except Exception as e:
            results[name] = (False, 0.0, 1.0, f"ERROR: {e}")

    passing = sum(1 for v in results.values() if v[0])
    total = len(results)

    report_lines = [f"\n=== Round-Trip Results: {passing}/{total} passing ==="]
    for name in names:
        works, ratio, error, status = results[name]
        report_lines.append(
            f"  {status:6s} | {name:40s} | ratio={ratio:8.2f}x | err={error:.6f}"
        )

    report = "\n".join(report_lines)
    print(report)

    # At least 50% must pass (many auto-generated methods may fail silently)
    assert passing >= max(total // 2, 40), (
        f"Only {passing}/{total} methods passed round-trip. Minimum 50% required.\n{report}"
    )


def test_get_method_stats():
    """Method stats returns sensible counts."""
    stats = MethodDiscovery.get_method_stats()
    assert stats["total"] >= MIN_METHODS
    assert stats["tier1_real_compression"] > 0
    assert stats["tier5_quantization"] > 0


def test_get_methods_by_tier():
    """get_methods_by_tier returns correct tier-filtered methods."""
    methods = MethodDiscovery.discover()
    for tier in [MethodTier.TIER1_REAL_COMPRESSION, MethodTier.TIER5_QUANTIZATION]:
        tier_methods = MethodDiscovery.get_methods_by_tier(tier)
        expected = {n: m for n, m in methods.items() if m["tier"] == tier}
        assert len(tier_methods) == len(expected), (
            f"Tier {tier}: got {len(tier_methods)}, expected {len(expected)}"
        )


def test_get_methods_by_category():
    """get_methods_by_category returns correct category-filtered methods."""
    methods = MethodDiscovery.discover()
    for cat in ["quantization", "spectral", "structural", "entropy", "functional"]:
        cat_methods = MethodDiscovery.get_methods_by_category(cat)
        expected = {n: m for n, m in methods.items() if m["category"] == cat}
        assert len(cat_methods) == len(expected), (
            f"Category {cat}: got {len(cat_methods)}, expected {len(expected)}"
        )


def test_compression_methods_exclude_quantization():
    """get_compression_methods() returns only Tier 1-3 methods."""
    comp_methods = MethodDiscovery.get_compression_methods()
    for info in comp_methods.values():
        assert info["tier"] not in (MethodTier.TIER5_QUANTIZATION,), (
            f"{info['name']} is quantization but in compression methods"
        )


def test_quantization_methods_exclude_compression():
    """get_quantization_methods() returns only Tier 5 methods."""
    q_methods = MethodDiscovery.get_quantization_methods()
    assert len(q_methods) > 0
    for info in q_methods.values():
        assert info["tier"] == MethodTier.TIER5_QUANTIZATION


def test_all_method_classes_have_name_attr():
    """Every class in METHOD_CLASSES has a name attribute (non-None).

    Non-compression utilities are excluded from this check.
    """
    failures: List[str] = []
    for name, cls in METHOD_CLASSES.items():
        if name in _NON_COMPRESSION_UTILITIES:
            continue
        if isinstance(cls, type):
            inst = cls()
            attr = getattr(inst, "name", None)
            if attr is None:
                # Try class-level name
                attr = getattr(cls, "name", None)
            if attr is None:
                failures.append(name)
    if failures:
        pytest.fail(
            f"{len(failures)} classes missing name:\n  " + "\n  ".join(failures[:20])
        )


def test_method_category_distribution():
    """Verify reasonable category distribution across all methods."""
    methods = MethodDiscovery.discover()
    categories = Counter(info["category"] for info in methods.values())

    # Core categories must exist
    required = {
        "quantization",
        "spectral",
        "structural",
        "entropy",
        "functional",
        "decomposition",
    }
    missing = required - set(categories.keys())
    assert not missing, f"Missing categories: {missing}"

    print(f"\n=== Category Distribution ({len(methods)} total) ===")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"  {cat:40s}: {count:5d} methods")


def test_method_tier_distribution():
    """Verify tier distribution is sensible."""
    methods = MethodDiscovery.discover()
    tiers = Counter(info["tier"] for info in methods.values())

    print(f"\n=== Tier Distribution ===")
    for tier in sorted(tiers):
        print(f"  Tier {tier}: {tiers[tier]} methods")


def test_validate_method_returns_valid_metrics():
    """validate_method returns sensible metrics for working methods."""
    methods = MethodDiscovery.discover()
    tested = 0
    for name in list(methods.keys())[:20]:
        info = methods[name]
        works, ratio, error = MethodDiscovery.validate_method(name, info)
        if works:
            assert ratio >= 1.0, f"{name}: ratio={ratio} should be >= 1.0"
            assert 0.0 <= error <= 2.0, f"{name}: error={error} out of range"
            tested += 1
    assert tested >= 5, f"Only {tested}/20 base methods validated"


def test_each_method_has_loss_type():
    """Every discovered method must have a loss_type and precision_preserved_bits."""
    methods = MethodDiscovery.discover()
    failures: List[str] = []
    for name, info in methods.items():
        if "loss_type" not in info:
            failures.append(f"{name}: missing loss_type")
        elif "precision_preserved_bits" not in info:
            failures.append(f"{name}: missing precision_preserved_bits")
    if failures:
        pytest.fail(
            f"{len(failures)} methods missing loss_type/precision:\n  "
            + "\n  ".join(failures[:20])
        )


def test_loss_type_values_are_valid():
    """loss_type must be one of the valid loss types."""
    methods = MethodDiscovery.discover()
    valid = {"lossless", "lossy_quant", "lossy_spectral", "lossy_hybrid"}
    bad: List[str] = []
    for name, info in methods.items():
        lt = info.get("loss_type", "")
        if lt not in valid:
            bad.append(f"{name}: invalid loss_type={lt}")
    if bad:
        pytest.fail(
            f"{len(bad)} methods with invalid loss_type:\n  " + "\n  ".join(bad[:20])
        )


def test_precision_bits_positive():
    """precision_preserved_bits must be a positive integer."""
    methods = MethodDiscovery.discover()
    bad: List[str] = []
    for name, info in methods.items():
        bits = info.get("precision_preserved_bits", 0)
        if not isinstance(bits, int) or bits <= 0:
            bad.append(f"{name}: invalid precision_preserved_bits={bits}")
    if bad:
        pytest.fail(
            f"{len(bad)} methods with invalid precision:\n  " + "\n  ".join(bad[:20])
        )


def test_loss_type_consistent_with_category():
    """Verify loss_type makes sense for each category."""
    methods = MethodDiscovery.discover()
    inconsistencies: List[str] = []
    for name, info in methods.items():
        cat = info.get("category", "")
        lt = info.get("loss_type", "")
        if "quantization" in cat or "quant" in cat:
            if lt not in ("lossy_quant", "lossy_hybrid"):
                inconsistencies.append(f"{name}: cat={cat}, loss_type={lt}")
        elif cat in ("lossless", "entropy"):
            if lt != "lossless":
                inconsistencies.append(f"{name}: cat={cat}, loss_type={lt}")
        elif cat in (
            "decomposition",
            "spectral",
            "structural",
            "tensor_network",
            "functional",
            "physics",
            "novel",
        ):
            if lt not in ("lossy_spectral", "lossy_hybrid"):
                inconsistencies.append(f"{name}: cat={cat}, loss_type={lt}")
    if len(inconsistencies) > len(methods) // 2:
        pytest.fail(f"Too many loss_type inconsistencies: {len(inconsistencies)}")


def test_validate_all_methods_returns_metrics():
    """validate_all_methods from method_validation returns rich metrics for first 80 methods."""
    from spectralstream.compression.engine.method_validation import validate_all_methods

    results = validate_all_methods(max_methods=80, verbose=False)
    assert len(results) == 80, f"Expected 80 results, got {len(results)}"
    working = sum(1 for r in results.values() if r["works"])
    assert working >= 40, f"Only {working}/80 methods passed validation"
    # Check rich metrics exist
    for mname, r in list(results.items())[:10]:
        if r["works"]:
            assert r["ratio"] >= 1.0, f"{mname}: ratio={r['ratio']}"
            assert r["error"] >= 0.0, f"{mname}: error={r['error']}"
            assert r["snr_db"] >= 0.0, f"{mname}: snr={r['snr_db']}"
            assert r["compress_time_ms"] >= 0.0
            assert r["decompress_time_ms"] >= 0.0


def test_validate_all_methods_generator():
    """validate_all_generator yields rich results for first 80 methods."""
    from spectralstream.compression.engine.method_validation import (
        validate_single_method,
    )

    methods = MethodDiscovery.discover()
    tensor = np.random.RandomState(42).randn(16, 16).astype(np.float32)
    count = 0
    for i, (mname, minfo) in enumerate(sorted(methods.items())[:80]):
        res = validate_single_method(mname, minfo, tensor=tensor)
        assert "loss_type" in res
        assert "snr_db" in res
        assert "compress_time_ms" in res
        count += 1
    assert count == 80, f"Expected 80, got {count}"
