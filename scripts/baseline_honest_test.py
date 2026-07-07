#!/usr/bin/env python3
"""Baseline Honest Compression Test — ground truth before 5-stage pipeline.

Tests every engine built-in method + cascade patterns on realistic synthetic
weight distributions.  Reports honest (measured, not estimated) ratios and
error metrics using the authoritative ``honest_metrics`` module.

Usage:
    python scripts/baseline_honest_test.py
    python scripts/baseline_honest_test.py --quick
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import signal
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spectralstream.compression.honest_metrics import (
    dual_ratio,
    end_to_end_error,
    serialized_nbytes,
    ErrorMetrics,
)
from spectralstream.compression.engine._methods import METHOD_REGISTRY
from spectralstream.compression.engine.aggressive_cascades import (
    AGGRESSIVE_CASCADES,
    build_cascade_stages,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("baseline_honest")


# ── Timeout decorator ────────────────────────────────────────────────────────
class TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutError("Timed out")


def with_timeout(func, timeout_s: float = 300, *args, **kwargs):
    """Run func with a wall-clock timeout.  Returns (result_or_None, elapsed_sec)."""
    import threading

    result = [None]
    exc_info = [None]

    def runner():
        try:
            result[0] = func(*args, **kwargs)
        except Exception as e:
            exc_info[0] = e

    t = threading.Thread(target=runner, daemon=True)
    t0 = time.perf_counter()
    t.start()
    t.join(timeout_s)
    elapsed = time.perf_counter() - t0

    if t.is_alive():
        return None, elapsed
    if exc_info[0] is not None:
        raise exc_info[0]  # type: ignore
    return result[0], elapsed


# ── Methods known to be slow on large tensors (per-element Python loops) ─────
SLOW_ON_LARGE = {"hadamard_int8", "hadamard_int4"}

# Max elements for slow methods (avoid minute-long Python loops)
SLOW_MAX_ELEMENTS = 512 * 512  # 262k elements


# ── Synthetic tensor generation ─────────────────────────────────────────────
def make_synthetic_tensor(
    shape: Tuple[int, ...],
    seed: int = 42,
    low_rank_scale: float = 0.8,
    outlier_scale: float = 3.0,
    outlier_fraction: float = 0.005,
) -> np.ndarray:
    """Generate a realistic LLM-weight-like tensor with low-rank structure + outliers."""
    rng = np.random.RandomState(seed)
    if len(shape) == 1:
        base = rng.randn(*shape).astype(np.float32) * 0.5
        noise = rng.randn(*shape).astype(np.float32) * 0.1
        t = base + noise
    elif len(shape) == 2:
        m, n = shape
        rank = max(min(m, n) // 16, 4)
        U = rng.randn(m, rank).astype(np.float32)
        S = np.linspace(1.0, 0.01, rank).astype(np.float32)
        Vh = rng.randn(rank, n).astype(np.float32)
        base = (U * S).dot(Vh) * low_rank_scale
        noise = rng.randn(m, n).astype(np.float32) * 0.1 * (1.0 - low_rank_scale)
        t = base + noise
        n_outliers = max(1, int(m * n * outlier_fraction))
        outlier_idx = rng.choice(m * n, n_outliers, replace=False)
        t.ravel()[outlier_idx] *= outlier_scale
    else:
        t = rng.randn(*shape).astype(np.float32) * 0.3
    return np.ascontiguousarray(t, dtype=np.float32)


# ── Test shapes ──────────────────────────────────────────────────────────────
TEST_SHAPES: List[Tuple[str, Tuple[int, ...]]] = [
    ("4096x4096", (4096, 4096)),
    ("4096x14336", (4096, 14336)),
    ("14336x4096", (14336, 4096)),
    ("512x512", (512, 512)),
    ("128x128", (128, 128)),
    ("1d_4096", (4096,)),
]


# ── Test runner ──────────────────────────────────────────────────────────────
def test_method(
    method_name: str,
    method_inst: Any,
    tensor: np.ndarray,
    shape_label: str,
    params: Optional[Dict[str, Any]] = None,
    timeout_s: float = 300,
) -> Optional[Dict[str, Any]]:
    if params is None:
        params = {}

    elements = int(tensor.size)
    fp32_bytes = elements * 4
    bf16_bytes = elements * 2

    # Compress
    def _compress():
        return method_inst.compress(tensor, **params)

    try:
        comp_result, t_compress = with_timeout(_compress, timeout_s)
    except Exception as exc:
        logger.warning(
            "  %-20s %s: compress EXCEPTION — %s", method_name, shape_label, exc
        )
        return None

    if comp_result is None:
        logger.warning(
            "  %-20s %s: compress TIMEOUT (>{:.0f}s)".format(timeout_s),
            method_name,
            shape_label,
        )
        return None

    data, meta = comp_result

    # Decompress
    def _decompress():
        return method_inst.decompress(data, meta)

    try:
        dec_result, t_decompress = with_timeout(_decompress, timeout_s)
    except Exception as exc:
        logger.warning(
            "  %-20s %s: decompress EXCEPTION — %s", method_name, shape_label, exc
        )
        return None

    if dec_result is None:
        logger.warning(
            "  %-20s %s: decompress TIMEOUT (>{:.0f}s)".format(timeout_s),
            method_name,
            shape_label,
        )
        return None
    recon = dec_result

    if recon.shape != tensor.shape:
        try:
            recon = recon.reshape(tensor.shape)
        except Exception as exc:
            logger.warning(
                "  %-20s %s: shape mismatch %s vs %s — %s",
                method_name,
                shape_label,
                recon.shape,
                tensor.shape,
                exc,
            )
            return None

    comp_bytes = max(serialized_nbytes(data), 1)
    ratio_vs_fp32 = fp32_bytes / comp_bytes
    ratio_vs_bf16 = bf16_bytes / comp_bytes
    errors = end_to_end_error(tensor, recon)

    return {
        "method": method_name,
        "category": getattr(method_inst, "category", "unknown"),
        "shape_label": shape_label,
        "shape": list(tensor.shape),
        "elements": elements,
        "fp32_bytes": fp32_bytes,
        "bf16_bytes": bf16_bytes,
        "compressed_bytes": int(comp_bytes),
        "ratio_vs_fp32": round(ratio_vs_fp32, 4),
        "ratio_vs_bf16": round(ratio_vs_bf16, 4),
        "rel_mse": round(errors.rel_mse, 8),
        "cosine_sim": round(errors.cosine_sim, 8),
        "max_abs_error": round(errors.max_abs, 8),
        "snr_db": round(errors.snr_db, 4),
        "compress_sec": round(t_compress, 4),
        "decompress_sec": round(t_decompress, 4),
    }


def test_cascade(
    cascade_name: str,
    cascade_cfg: Dict[str, Any],
    engine: Any,
    tensor: np.ndarray,
    shape_label: str,
    timeout_s: float = 300,
) -> Optional[Dict[str, Any]]:
    elements = int(tensor.size)
    fp32_bytes = elements * 4
    bf16_bytes = elements * 2

    def _run():
        return build_cascade_stages(engine, tensor, cascade_cfg)

    try:
        stages, t_total = with_timeout(_run, timeout_s)
    except Exception as exc:
        logger.warning(
            "  cascade '%-30s %s: EXCEPTION — %s", cascade_name, shape_label, exc
        )
        return None

    if stages is None:
        logger.warning(
            "  cascade '%-30s %s: TIMEOUT (>{:.0f}s)".format(timeout_s),
            cascade_name,
            shape_label,
        )
        return None

    if not stages:
        logger.warning(
            "  cascade '%-30s %s: no stages produced", cascade_name, shape_label
        )
        return None

    original_np = np.ascontiguousarray(tensor, dtype=np.float64)
    recon = np.zeros_like(original_np, dtype=np.float64)
    total_comp_bytes = 0
    stage_details = []

    for s in stages:
        sr = s["recon"].astype(np.float64)
        if sr.shape != original_np.shape:
            sr = sr.reshape(original_np.shape)
        recon += sr
        sd = s.get("compressed_data", b"")
        stage_bytes = serialized_nbytes(sd)
        total_comp_bytes += stage_bytes
        stage_details.append(
            {
                "method": s["method"],
                "params": s.get("params", {}),
                "stage_ratio": round(
                    float(original_np.nbytes / max(stage_bytes, 1)), 4
                ),
                "stage_bytes": int(stage_bytes),
            }
        )

    total_comp_bytes = max(total_comp_bytes, 1)
    ratio_vs_fp32 = fp32_bytes / total_comp_bytes
    ratio_vs_bf16 = bf16_bytes / total_comp_bytes
    errors = end_to_end_error(tensor, recon.astype(np.float32))

    return {
        "cascade": cascade_name,
        "description": cascade_cfg.get("description", ""),
        "expected_ratio": cascade_cfg.get("expected_ratio", 0),
        "stages": [s["method"] for s in stages],
        "shape_label": shape_label,
        "shape": list(tensor.shape),
        "elements": elements,
        "fp32_bytes": fp32_bytes,
        "bf16_bytes": bf16_bytes,
        "total_compressed_bytes": int(total_comp_bytes),
        "ratio_vs_fp32": round(ratio_vs_fp32, 4),
        "ratio_vs_bf16": round(ratio_vs_bf16, 4),
        "rel_mse": round(errors.rel_mse, 8),
        "cosine_sim": round(errors.cosine_sim, 8),
        "max_abs_error": round(errors.max_abs, 8),
        "snr_db": round(errors.snr_db, 4),
        "total_sec": round(t_total, 4),
        "stage_details": stage_details,
    }


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Baseline Honest Compression Test")
    parser.add_argument(
        "--output", default="/tmp/baseline_honest_results.json", help="Output JSON path"
    )
    parser.add_argument(
        "--quick", action="store_true", help="Only small shapes (512x512 and down)"
    )
    parser.add_argument(
        "--timeout", type=int, default=120, help="Per-method timeout in seconds"
    )
    args = parser.parse_args()

    timeout_s = args.timeout
    logger.info("=" * 72)
    logger.info("BASELINE HONEST COMPRESSION TEST")
    logger.info("=" * 72)

    # ── Generate test tensors ──────────────────────────────────────────────
    logger.info("\nGenerating synthetic LLM-like weight tensors...")
    test_tensors: Dict[str, np.ndarray] = {}
    for label, shape in TEST_SHAPES:
        if args.quick and label not in ("512x512", "128x128", "1d_4096"):
            logger.info("  Skipping %s (--quick)", label)
            continue
        t = make_synthetic_tensor(shape)
        test_tensors[label] = t
        logger.info("  %s: %s, range=[%.2f, %.2f]", label, shape, t.min(), t.max())

    all_results: Dict[str, Any] = {
        "meta": {
            "description": "Baseline honest compression test",
            "test_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "quick_mode": args.quick,
            "timeout_s": timeout_s,
        },
        "methods": {},
        "cascades": {},
    }

    # ── Section 1: Individual Methods ──────────────────────────────────────
    logger.info("\n" + "=" * 72)
    logger.info("SECTION 1: INDIVIDUAL METHOD TESTING")
    logger.info("=" * 72)

    available_methods = dict(METHOD_REGISTRY)
    logger.info("Methods: %s", ", ".join(available_methods.keys()))

    default_params: Dict[str, Dict[str, Any]] = {
        "block_int8": {"block_size": 128},
        "block_int4": {"block_size": 16},
        "hadamard_int8": {"block_size": 128},
        "hadamard_int4": {"block_size": 16},
        "sparsity_int4": {"group_size": 32},
        "delta_int4": {"block_size": 32},
        "svd_compress": {"rank": 32},
        "dct_spectral": {"keep_ratio": 0.1},
        "tensor_train": {"rank": 8},
        "fwht_compress": {"keep_ratio": 0.1},
    }

    svd_ranks = [8, 16, 32, 64]

    for method_name in sorted(available_methods.keys()):
        inst = available_methods[method_name]
        is_slow = method_name in SLOW_ON_LARGE
        logger.info(
            "\n--- %s%s ---",
            method_name,
            " [SLOW on large tensors, will skip big shapes]" if is_slow else "",
        )

        results = []
        params = default_params.get(method_name, {})

        for label, tensor in test_tensors.items():
            # Skip large tensors for slow methods
            if is_slow and tensor.size > SLOW_MAX_ELEMENTS:
                logger.info(
                    "  %-20s %s: SKIPPED (slow method, %d elements > %d limit)",
                    method_name,
                    label,
                    tensor.size,
                    SLOW_MAX_ELEMENTS,
                )
                continue

            # SVD gets multiple rank tests
            if method_name == "svd_compress":
                for rank in svd_ranks:
                    effective_rank = (
                        min(rank, min(tensor.shape) // 2) if tensor.ndim >= 2 else rank
                    )
                    params_r = {"rank": max(effective_rank, 1)}
                    r = test_method(
                        method_name,
                        inst,
                        tensor,
                        f"{label}_r{rank}",
                        params_r,
                        timeout_s,
                    )
                    if r is not None:
                        r["svd_rank"] = rank
                        results.append(r)
                        logger.info(
                            "  %-20s %-16s: ratio=%.1fx vs_fp32 (%.1fx vs_bf16), "
                            "SNR=%.1f dB, cos=%.6f, err=%.2e, %.3fs",
                            method_name,
                            f"{label}_r{rank}",
                            r["ratio_vs_fp32"],
                            r["ratio_vs_bf16"],
                            r["snr_db"],
                            r["cosine_sim"],
                            r["rel_mse"],
                            r["compress_sec"],
                        )
                continue

            r = test_method(method_name, inst, tensor, label, params, timeout_s)
            if r is None:
                continue

            results.append(r)
            logger.info(
                "  %-20s %-16s: ratio=%.1fx vs_fp32 (%.1fx vs_bf16), "
                "SNR=%.1f dB, cos=%.6f, err=%.2e, max_err=%.2e, %.3fs",
                method_name,
                label,
                r["ratio_vs_fp32"],
                r["ratio_vs_bf16"],
                r["snr_db"],
                r["cosine_sim"],
                r["rel_mse"],
                r["max_abs_error"],
                r["compress_sec"],
            )

        all_results["methods"][method_name] = results

    # ── Section 2: Cascade Patterns ────────────────────────────────────────
    logger.info("\n" + "=" * 72)
    logger.info("SECTION 2: CASCADE PATTERN TESTING")
    logger.info("=" * 72)

    cascade_test_tensors = {}
    for label in ["4096x4096", "512x512"]:
        if label in test_tensors:
            cascade_test_tensors[label] = test_tensors[label]
    if not cascade_test_tensors:
        cascade_test_tensors = {
            "128x128": test_tensors.get(
                "128x128", test_tensors[list(test_tensors.keys())[0]]
            )
        }

    class _MinimalEngine:
        def __init__(self, methods):
            self._methods = methods

    engine = _MinimalEngine(available_methods)

    requested_cascades = [
        "lightning",
        "balanced",
        "aggressive",
        "extreme",
        "svd_int4_sparse_huffman",
        "tt_quant_sparse_fwht_huffman",
    ]

    simple_cascade_map = {
        "lightning": {
            "stages": ["dct_spectral"],
            "params": [{"keep_ratio": 0.1}],
            "description": "Single-stage DCT spectral compression",
            "expected_ratio": 5.0,
            "target_tensors": ["weight"],
        },
        "balanced": {
            "stages": ["svd_compress"],
            "params": [{"rank": 32}],
            "description": "SVD decomposition only",
            "expected_ratio": 50.0,
            "target_tensors": ["weight"],
        },
        "aggressive": {
            "stages": ["svd_compress", "dct_spectral"],
            "params": [{"rank": 32}, {"keep_ratio": 0.08}],
            "description": "SVD + DCT cascade",
            "expected_ratio": 100.0,
            "target_tensors": ["weight"],
        },
        "extreme": {
            "stages": ["svd_compress", "block_int4", "sparsity_int4"],
            "params": [{"rank": 32}, {"block_size": 16}, {"group_size": 32}],
            "description": "Multi-stage cascade for maximum compression",
            "expected_ratio": 200.0,
            "target_tensors": ["weight"],
        },
    }

    cascade_configs = dict(simple_cascade_map)
    for name, cfg in AGGRESSIVE_CASCADES.items():
        cascade_configs[name] = cfg

    for cascade_name in requested_cascades:
        cfg = cascade_configs.get(cascade_name)
        if cfg is None:
            logger.info("\n--- %s: NOT FOUND ---", cascade_name)
            continue

        logger.info(
            "\n--- Cascade: %s ('%s') ---", cascade_name, cfg.get("description", "")
        )
        logger.info("    Stages: %s", cfg.get("stages", []))
        logger.info(
            "    Expected ratio (fabricated?): %.0fx", cfg.get("expected_ratio", 0)
        )

        for label, tensor in cascade_test_tensors.items():
            r = test_cascade(cascade_name, cfg, engine, tensor, label, timeout_s)
            if r is None:
                continue

            # Compute ratio delta vs expected
            expected = cfg.get("expected_ratio", 0)
            honest = r["ratio_vs_fp32"]
            gap_pct = ((honest - expected) / expected * 100) if expected > 0 else 0

            logger.info(
                "  %-30s %s:\n"
                "    Stages:     %s\n"
                "    Ratio vs_fp32: %.1fx (expected: %.0fx — gap: %+.0f%%)\n"
                "    Ratio vs_bf16: %.1fx\n"
                "    rel_mse:   %.2e\n"
                "    cos_sim:   %.6f\n"
                "    SNR:       %.1f dB\n"
                "    max_err:   %.2e\n"
                "    time:      %.3fs",
                cascade_name,
                label,
                r["stages"],
                honest,
                expected,
                gap_pct,
                r["ratio_vs_bf16"],
                r["rel_mse"],
                r["cosine_sim"],
                r["snr_db"],
                r["max_abs_error"],
                r["total_sec"],
            )

            if cascade_name not in all_results["cascades"]:
                all_results["cascades"][cascade_name] = []
            all_results["cascades"][cascade_name].append(r)

    # ── Section 3: Summary ─────────────────────────────────────────────────
    logger.info("\n" + "=" * 72)
    logger.info("SUMMARY — Method Averages Across All Tested Shapes")
    logger.info("=" * 72)

    summaries = []
    for method_name in sorted(available_methods.keys()):
        results = all_results["methods"].get(method_name, [])
        if not results:
            logger.info("  %-20s: ALL FAILED / SKIPPED", method_name)
            continue

        ratios_fp32 = [r["ratio_vs_fp32"] for r in results if r["ratio_vs_fp32"] > 0]
        ratios_bf16 = [r["ratio_vs_bf16"] for r in results if r["ratio_vs_bf16"] > 0]
        snrs = [
            r["snr_db"]
            for r in results
            if isinstance(r["snr_db"], (int, float)) and r["snr_db"] != float("inf")
        ]
        cosims = [
            r["cosine_sim"]
            for r in results
            if isinstance(r["cosine_sim"], (int, float))
        ]
        errors = [
            r["rel_mse"] for r in results if isinstance(r["rel_mse"], (int, float))
        ]

        summary = {
            "method": method_name,
            "n_tests": len(results),
            "ratio_vs_fp32_mean": round(float(np.mean(ratios_fp32)), 2)
            if ratios_fp32
            else 0,
            "ratio_vs_fp32_min": round(float(np.min(ratios_fp32)), 2)
            if ratios_fp32
            else 0,
            "ratio_vs_fp32_max": round(float(np.max(ratios_fp32)), 2)
            if ratios_fp32
            else 0,
            "ratio_vs_bf16_mean": round(float(np.mean(ratios_bf16)), 2)
            if ratios_bf16
            else 0,
            "snr_db_mean": round(float(np.mean(snrs)), 1) if snrs else float("-inf"),
            "cosine_sim_mean": round(float(np.mean(cosims)), 6) if cosims else 0,
            "rel_mse_mean": float(np.mean(errors)) if errors else 0,
        }
        summaries.append(summary)

        avg_ratio_fp32 = summary["ratio_vs_fp32_mean"]
        avg_ratio_bf16 = summary["ratio_vs_bf16_mean"]
        avg_snr = summary["snr_db_mean"]
        avg_cos = summary["cosine_sim_mean"]

        logger.info(
            "  %-20s: %2d tests, ratio_fp32=%6.1fx (min=%5.1f, max=%5.1f), "
            "ratio_bf16=%6.1fx, SNR=%5.1f dB, cos=%.6f",
            method_name,
            summary["n_tests"],
            avg_ratio_fp32,
            summary["ratio_vs_fp32_min"],
            summary["ratio_vs_fp32_max"],
            avg_ratio_bf16,
            avg_snr,
            avg_cos,
        )

    # Cascade summary
    logger.info("\nCASCADE SUMMARY:")
    for cascade_name, results in all_results["cascades"].items():
        if not results:
            logger.info("  %-30s: ALL FAILED", cascade_name)
            continue
        r = results[0]
        expected = cascade_configs.get(cascade_name, {}).get("expected_ratio", 0)
        honest = r["ratio_vs_fp32"]
        gap_pct = ((honest - expected) / expected * 100) if expected > 0 else 0
        logger.info(
            "  %-30s: honest=%.1fx (expected=%.0fx, delta=%+.0f%%), "
            "SNR=%.1f dB, cos=%.6f",
            cascade_name,
            honest,
            expected,
            gap_pct,
            r["snr_db"],
            r["cosine_sim"],
        )

    all_results["summaries"] = summaries

    # ── Save ──────────────────────────────────────────────────────────────────
    output_path = args.output
    os.makedirs(
        os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
        exist_ok=True,
    )
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str, cls=_Encoder)
    logger.info("\nResults saved to %s", output_path)

    # ── Verdict ──────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 72)
    logger.info("VERDICT")
    logger.info("=" * 72)
    n_methods = len(all_results["methods"])
    n_method_working = sum(1 for v in all_results["methods"].values() if v)
    n_cascades = len(all_results["cascades"])
    n_cascade_working = sum(1 for v in all_results["cascades"].values() if v)
    logger.info("Methods working:  %d/%d", n_method_working, n_methods)
    logger.info("Cascades working: %d/%d", n_cascade_working, n_cascades)
    logger.info("Output: %s", output_path)
    return all_results


class _Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


if __name__ == "__main__":
    main()
