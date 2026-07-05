"""
Comprehensive validation of ALL 159 compression methods against real Gemma 4 E2B weights.

Tests every registered method on representative tensors from:
    - embedding (large, sensitive)
    - attention Q/K/V projections (medium, critical)
    - attention output projection (medium)
    - FFN gate/up/down projections (large)
    - layer norms (small, robust)
    - audio tower projections (medium)
    - per-layer projections (small)
    - layer scalars (tiny)

Outputs: JSON, HTML, and Markdown reports to tests/output/
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest

try:
    import torch
except ImportError:
    torch = None  # type: ignore

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Paths ────────────────────────────────────────────────────────────────
MODEL_PATH = Path("models/gemma-4-E2B/model.safetensors")
OUTPUT_DIR = Path("tests/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

JSON_REPORT = OUTPUT_DIR / "gemma4_method_validation.json"
HTML_REPORT = OUTPUT_DIR / "gemma4_method_validation.html"
MD_REPORT = OUTPUT_DIR / "gemma4_method_validation.md"


# ── PyTest markers ───────────────────────────────────────────────────────
# Register custom markers in pyproject.toml to avoid warnings
def pytest_register_marks() -> None:
    pass


pytestmark = [
    pytest.mark.gemma4,
    pytest.mark.validation,
    pytest.mark.skipif(
        not MODEL_PATH.exists(), reason=f"Gemma 4 model not found at {MODEL_PATH}"
    ),
]

# ── Sampling size for large tensors ──────────────────────────────────────
MAX_SAMPLE_ELEMENTS = 10_000

# ── Tensor category labels ───────────────────────────────────────────────
TENSOR_CATEGORIES: Dict[str, str] = {}


def _tensor_label(key: str) -> str:
    """Assign a human-readable category label to a tensor key."""
    if "embed_tokens" in key:
        return "embedding"
    if "q_proj" in key:
        return "attention_q"
    if "k_proj" in key:
        return "attention_k"
    if "v_proj" in key:
        return "attention_v"
    if "o_proj" in key:
        return "attention_o"
    if "gate_proj" in key:
        return "ffn_gate"
    if "up_proj" in key:
        return "ffn_up"
    if "down_proj" in key:
        return "ffn_down"
    if "layernorm" in key or "rms_norm" in key:
        return "norm"
    if "audio_tower" in key:
        return "audio"
    if "per_layer_projection" in key:
        return "per_layer_proj"
    if "layer_scalar" in key:
        return "layer_scalar"
    return "other"


def _sample_tensor(
    tensor: np.ndarray, max_elements: int = MAX_SAMPLE_ELEMENTS
) -> np.ndarray:
    """Sample a tensor down to max_elements if it's too large."""
    flat = tensor.ravel()
    if len(flat) <= max_elements:
        return flat
    indices = np.random.RandomState(42).choice(len(flat), max_elements, replace=False)
    indices.sort()
    return flat[indices]


# ── Select representative tensors ───────────────────────────────────────
def _select_tensors() -> List[Dict[str, Any]]:
    """Open the safetensors file and select ~20 representative tensors."""
    from safetensors import safe_open

    selected: List[Dict[str, Any]] = []

    with safe_open(str(MODEL_PATH), framework="pt") as f:
        keys = list(f.keys())

        # Language-model weight tensors only (exclude quantized metadata)
        lm_weight_keys = [
            k
            for k in keys
            if "language_model" in k
            and k.endswith(".weight")
            and not any(x in k for x in ["input_min", "input_max"])
        ]

        # Build candidate pool by category
        categories = {
            "embedding": [k for k in lm_weight_keys if "embed_tokens" in k],
            "attention_q": [k for k in lm_weight_keys if "q_proj" in k],
            "attention_k": [k for k in lm_weight_keys if "k_proj" in k],
            "attention_v": [k for k in lm_weight_keys if "v_proj" in k],
            "attention_o": [k for k in lm_weight_keys if "o_proj" in k],
            "ffn_gate": [k for k in lm_weight_keys if "gate_proj" in k],
            "ffn_up": [k for k in lm_weight_keys if "up_proj" in k],
            "ffn_down": [k for k in lm_weight_keys if "down_proj" in k],
            "norm": [k for k in lm_weight_keys if "layernorm" in k],
            "per_layer_proj": [
                k for k in lm_weight_keys if "per_layer_projection" in k
            ],
        }

        # Audio tower weights
        audio_keys = [
            k
            for k in keys
            if "audio_tower" in k and k.endswith(".weight") and "linear" in k
        ]

        # Pick 1-2 per category
        for cat, cat_keys in categories.items():
            sample_count = 2 if len(cat_keys) > 1 else 1
            for k in cat_keys[:sample_count]:
                tensor = f.get_tensor(k)
                flat = _sample_tensor(tensor.float().numpy())
                selected.append(
                    {
                        "key": k,
                        "category": cat,
                        "original_shape": list(tensor.shape),
                        "original_dtype": str(tensor.dtype),
                        "sampled": len(flat) < tensor.numel(),
                        "n_elements": len(flat),
                        "flat": flat.astype(np.float32),
                    }
                )

        # Audio towers
        for k in audio_keys[:3]:
            tensor = f.get_tensor(k)
            flat = _sample_tensor(tensor.numpy())
            selected.append(
                {
                    "key": k,
                    "category": "audio",
                    "original_shape": list(tensor.shape),
                    "original_dtype": str(tensor.dtype),
                    "sampled": len(flat) < tensor.numel(),
                    "n_elements": len(flat),
                    "flat": flat.astype(np.float32),
                }
            )

        # Some scalars / small tensors
        scalar_keys = [k for k in keys if "layer_scalar" in k][:2]
        for k in scalar_keys:
            tensor = f.get_tensor(k)
            flat = tensor.numpy().ravel()
            selected.append(
                {
                    "key": k,
                    "category": "layer_scalar",
                    "original_shape": list(tensor.shape),
                    "original_dtype": str(tensor.dtype),
                    "sampled": False,
                    "n_elements": len(flat),
                    "flat": flat.astype(np.float32),
                }
            )

    return selected


# ── Metric computation (matches engine._compute_metrics) ─────────────────
def _compute_metrics(orig: np.ndarray, recon: np.ndarray) -> Dict[str, float]:
    o = orig.astype(np.float64).ravel()
    r = recon.astype(np.float64).ravel()
    n_min = min(len(o), len(r))
    o, r = o[:n_min], r[:n_min]
    noise = o - r
    mse = float(np.mean(noise**2))
    signal_power = float(np.mean(o**2)) + 1e-30
    snr_db = 10.0 * math.log10(signal_power / (mse + 1e-30))
    max_val = float(np.max(np.abs(o)))
    psnr_db = 10.0 * math.log10(max_val**2 / (mse + 1e-30)) if max_val > 0 else snr_db
    rel_error = float(np.linalg.norm(noise) / (np.linalg.norm(o) + 1e-30))
    cos_sim = float(np.dot(o, r) / (np.linalg.norm(o) * np.linalg.norm(r) + 1e-30))
    return {
        "mse": mse,
        "snr_db": snr_db,
        "psnr_db": psnr_db,
        "relative_error": rel_error,
        "cosine_similarity": cos_sim,
    }


# ── Single test runner ───────────────────────────────────────────────────
def _test_method_on_tensor(
    method_name: str,
    method_info: Dict[str, Any],
    tensor_info: Dict[str, Any],
) -> Dict[str, Any]:
    """Run one method on one tensor, return result dict."""
    inst = method_info.get("instance")
    if inst is None:
        return {
            "method": method_name,
            "tensor": tensor_info["key"],
            "category": tensor_info["category"],
            "status": "error",
            "error": "No instance",
        }

    tensor_flat = tensor_info["flat"]
    t_start = time.perf_counter()
    try:
        data, meta = inst.compress(tensor_flat)
        t_compress = time.perf_counter() - t_start
    except Exception as e:
        return {
            "method": method_name,
            "tensor": tensor_info["key"],
            "category": tensor_info["category"],
            "status": "error",
            "error": f"compress failed: {e}",
            "traceback": traceback.format_exc(),
        }

    # Compression ratio
    raw_bytes = tensor_flat.nbytes
    compressed_bytes = len(data) if hasattr(data, "__len__") else 1
    compression_ratio = raw_bytes / max(compressed_bytes, 1)

    t_start = time.perf_counter()
    try:
        recon = inst.decompress(data, meta)
        t_decompress = time.perf_counter() - t_start
    except Exception as e:
        return {
            "method": method_name,
            "tensor": tensor_info["key"],
            "category": tensor_info["category"],
            "status": "error",
            "error": f"decompress failed: {e}",
            "traceback": traceback.format_exc(),
            "compression_ratio": compression_ratio,
            "compression_time_s": t_compress,
        }

    # Reshape if needed
    if recon.shape != tensor_flat.shape:
        try:
            recon = recon.reshape(tensor_flat.shape)
        except Exception:
            pass

    # Trim / pad to match
    o = tensor_flat.ravel()
    r = recon.ravel().astype(np.float32)
    min_len = min(len(o), len(r))
    o, r = o[:min_len], r[:min_len]

    metrics = _compute_metrics(o, r)

    return {
        "method": method_name,
        "method_category": method_info.get("category", "?"),
        "method_tier": str(method_info.get("tier", "?")),
        "tensor": tensor_info["key"],
        "tensor_category": tensor_info["category"],
        "original_shape": tensor_info["original_shape"],
        "n_elements": tensor_info["n_elements"],
        "status": "ok",
        "compression_ratio": compression_ratio,
        "compression_time_s": t_compress,
        "decompression_time_s": t_decompress,
        **metrics,
    }


# ── Global test data ─────────────────────────────────────────────────────
_ALL_RESULTS: List[Dict[str, Any]] = []
_ALL_METHODS: Optional[Dict[str, Dict[str, Any]]] = None
_ALL_TENSORS: Optional[List[Dict[str, Any]]] = None


def _ensure_data():
    global _ALL_METHODS, _ALL_TENSORS
    from spectralstream.compression.engine.method_discovery import MethodDiscovery

    md = MethodDiscovery()
    if _ALL_METHODS is None:
        _ALL_METHODS = md.discover()
    if _ALL_TENSORS is None:
        _ALL_TENSORS = _select_tensors()


# ═══════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════


def test_selection_sanity():
    """Verify we selected meaningful tensors."""
    _ensure_data()
    assert _ALL_TENSORS is not None and len(_ALL_TENSORS) >= 15, (
        f"Expected >=15 tensors, got {len(_ALL_TENSORS) if _ALL_TENSORS else 0}"
    )
    categories_seen = {t["category"] for t in _ALL_TENSORS}
    required = {
        "embedding",
        "attention_q",
        "attention_k",
        "attention_v",
        "attention_o",
        "ffn_gate",
        "ffn_up",
        "ffn_down",
        "norm",
    }
    missing = required - categories_seen
    assert not missing, f"Missing tensor categories: {missing}"


def test_all_methods_discovered():
    """Verify all 159+ methods are discovered."""
    _ensure_data()
    assert _ALL_METHODS is not None
    assert len(_ALL_METHODS) >= 150, f"Expected >=150 methods, got {len(_ALL_METHODS)}"


def test_full_method_validation():
    """Run ALL methods on ALL sampled tensors and record results."""
    _ensure_data()
    assert _ALL_METHODS is not None
    assert _ALL_TENSORS is not None

    all_tasks = []
    for mname, minfo in _ALL_METHODS.items():
        for tinfo in _ALL_TENSORS:
            all_tasks.append((mname, minfo, tinfo))

    total = len(all_tasks)
    print(
        f"\n  Running {total} method×tensor combinations "
        f"({len(_ALL_METHODS)} methods × {len(_ALL_TENSORS)} tensors)"
    )

    results: List[Dict[str, Any]] = []
    max_workers = min(16, os.cpu_count() or 4)
    done = 0

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_test_method_on_tensor, mname, minfo, tinfo): (
                mname,
                tinfo["key"],
            )
            for mname, minfo, tinfo in all_tasks
        }

        for future in as_completed(futures):
            mname, tkey = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                results.append(
                    {
                        "method": mname,
                        "tensor": tkey,
                        "status": "error",
                        "error": f"unexpected: {e}",
                        "traceback": traceback.format_exc(),
                    }
                )
            done += 1
            if done % 200 == 0 or done == total:
                ok_count = sum(1 for r in results if r.get("status") == "ok")
                err_count = sum(1 for r in results if r.get("status") == "error")
                print(f"    Progress: {done}/{total} ({ok_count} ok, {err_count} err)")

    # Store globally for report generation
    global _ALL_RESULTS
    _ALL_RESULTS = results

    # Generate reports
    _generate_json_report(results)
    _generate_markdown_report(results)
    _generate_html_report(results)

    # Summary assertions
    ok_count = sum(1 for r in results if r.get("status") == "ok")
    err_count = sum(1 for r in results if r.get("status") == "error")
    pass_count = sum(
        1
        for r in results
        if r.get("status") == "ok" and r.get("relative_error", 1.0) < 0.01
    )

    print(
        f"\n  Results: {ok_count} ok, {err_count} error, {pass_count} pass (<1% error)"
    )
    assert ok_count > 0, "No methods produced valid results"
    print(f"  Reports saved to:")
    print(f"    {JSON_REPORT}")
    print(f"    {HTML_REPORT}")
    print(f"    {MD_REPORT}")


# ═══════════════════════════════════════════════════════════════════════════
# Report generators
# ═══════════════════════════════════════════════════════════════════════════


def _generate_json_report(results: List[Dict[str, Any]]):
    report = {
        "generated": datetime.utcnow().isoformat(),
        "model": str(MODEL_PATH),
        "total_methods": len({r["method"] for r in results}),
        "total_tensors": len({r["tensor"] for r in results}),
        "total_combinations": len(results),
        "summary": _compute_summary(results),
        "results": results,
    }
    JSON_REPORT.write_text(json.dumps(report, indent=2, default=str))
    return report


def _compute_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    ok = [r for r in results if r.get("status") == "ok"]
    err = [r for r in results if r.get("status") == "error"]
    passing = [r for r in ok if r.get("relative_error", 1.0) < 0.01]

    methods = {r["method"] for r in results}
    tensors = {r["tensor"] for r in results}
    cats = {r["tensor_category"] for r in results}

    # Methods that pass on all tensor types
    pass_by_method: Dict[str, int] = {}
    total_by_method: Dict[str, int] = {}
    for r in ok:
        pass_by_method[r["method"]] = pass_by_method.get(r["method"], 0) + (
            1 if r.get("relative_error", 1.0) < 0.01 else 0
        )
        total_by_method[r["method"]] = total_by_method.get(r["method"], 0) + 1

    fully_passing = [
        m
        for m in methods
        if total_by_method.get(m, 0) > 0
        and pass_by_method.get(m, 0) == total_by_method[m]
    ]

    # Best per tensor category (highest ratio with <1% error)
    best_per_cat: Dict[str, List[Dict]] = {}
    for cat in cats:
        cat_results = [r for r in passing if r["tensor_category"] == cat]
        if not cat_results:
            continue
        # Group by method, average ratio
        from collections import defaultdict

        method_ratios: Dict[str, List[float]] = defaultdict(list)
        for r in cat_results:
            method_ratios[r["method"]].append(r["compression_ratio"])
        ranked = sorted(
            [
                {
                    "method": m,
                    "avg_ratio": float(np.mean(ratios)),
                    "avg_error": float(
                        np.mean(
                            [
                                rr["relative_error"]
                                for rr in cat_results
                                if rr["method"] == m
                            ]
                        )
                    ),
                    "avg_snr": float(
                        np.mean(
                            [rr["snr_db"] for rr in cat_results if rr["method"] == m]
                        )
                    ),
                }
                for m, ratios in method_ratios.items()
            ],
            key=lambda x: -x["avg_ratio"],
        )[:10]
        best_per_cat[cat] = ranked

    # Overall ranking
    method_stats: Dict[str, Dict] = {}
    for r in ok:
        m = r["method"]
        if m not in method_stats:
            method_stats[m] = {"ratios": [], "errors": [], "snrs": []}
        method_stats[m]["ratios"].append(r["compression_ratio"])
        method_stats[m]["errors"].append(r.get("relative_error", 1.0))
        method_stats[m]["snrs"].append(r.get("snr_db", 0))

    overall_ranking = sorted(
        [
            {
                "method": m,
                "avg_ratio": float(np.mean(s["ratios"])),
                "avg_error": float(np.mean(s["errors"])),
                "avg_snr": float(np.mean(s["snrs"])),
                "pass_rate": (
                    sum(1 for e in s["errors"] if e < 0.01) / len(s["errors"])
                    if s["errors"]
                    else 0
                ),
                "n_tensors": len(s["errors"]),
            }
            for m, s in method_stats.items()
        ],
        key=lambda x: -x["pass_rate"] * x["avg_ratio"] / max(x["avg_error"], 1e-10),
    )

    # Failures by method
    fail_counts: Dict[str, int] = {}
    for r in err:
        fail_counts[r["method"]] = fail_counts.get(r["method"], 0) + 1

    # Ratio distribution
    ratios = [r["compression_ratio"] for r in passing]
    ratio_dist = {
        "count": len(ratios),
        "min": float(np.min(ratios)) if ratios else 0,
        "max": float(np.max(ratios)) if ratios else 0,
        "mean": float(np.mean(ratios)) if ratios else 0,
        "median": float(np.median(ratios)) if ratios else 0,
        "p25": float(np.percentile(ratios, 25)) if ratios else 0,
        "p75": float(np.percentile(ratios, 75)) if ratios else 0,
    }

    return {
        "ok_combinations": len(ok),
        "error_combinations": len(err),
        "pass_combinations": len(passing),
        "methods_count": len(methods),
        "tensors_count": len(tensors),
        "categories_count": len(cats),
        "fully_passing_methods": sorted(fully_passing),
        "fully_passing_count": len(fully_passing),
        "best_per_tensor_category": best_per_cat,
        "overall_ranking": overall_ranking[:30],
        "methods_with_failures": dict(sorted(fail_counts.items(), key=lambda x: -x[1])),
        "ratio_distribution": ratio_dist,
    }


def _generate_markdown_report(results: List[Dict[str, Any]]):
    summary = _compute_summary(results)

    lines = [
        "# Gemma 4 Method Validation Report",
        "",
        f"**Generated:** {datetime.utcnow().isoformat()}",
        f"**Model:** `{MODEL_PATH}`",
        "",
        "## Summary",
        "",
        f"- Total methods: {summary['methods_count']}",
        f"- Total tensors: {summary['tensors_count']}",
        f"- Total combinations: {summary['ok_combinations'] + summary['error_combinations']}",
        f"- OK: {summary['ok_combinations']}",
        f"- Errors: {summary['error_combinations']}",
        f"- Passing (<1% error): {summary['pass_combinations']}",
        f"- Fully passing methods: {summary['fully_passing_count']} / {summary['methods_count']}",
        "",
        "## Fully Passing Methods",
        "",
    ]

    if summary["fully_passing_methods"]:
        for m in summary["fully_passing_methods"]:
            lines.append(f"- `{m}`")
    else:
        lines.append("_(none)_")

    lines.extend(
        [
            "",
            "## Best Methods Per Tensor Category (Top 5 by ratio, <1% error)",
            "",
        ]
    )

    for cat, ranked in sorted(summary["best_per_tensor_category"].items()):
        lines.append(f"### {cat}")
        lines.append("")
        lines.append("| Method | Avg Ratio | Avg Error | Avg SNR (dB) |")
        lines.append("|--------|-----------|-----------|--------------|")
        for entry in ranked[:5]:
            lines.append(
                f"| `{entry['method']}` | {entry['avg_ratio']:.2f}x | "
                f"{entry['avg_error']:.6f} | {entry['avg_snr']:.1f} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Overall Ranking (Top 20)",
            "",
            "| Rank | Method | Avg Ratio | Avg Error | Avg SNR | Pass Rate | Tensors |",
            "|------|--------|-----------|-----------|---------|-----------|---------|",
        ]
    )
    for i, entry in enumerate(summary["overall_ranking"][:20]):
        lines.append(
            f"| {i + 1} | `{entry['method']}` | {entry['avg_ratio']:.2f}x | "
            f"{entry['avg_error']:.6f} | {entry['avg_snr']:.1f} dB | "
            f"{entry['pass_rate']:.0%} | {entry['n_tensors']} |"
        )

    lines.extend(
        [
            "",
            "## Methods With Failures",
            "",
            "| Method | Failure Count |",
            "|--------|---------------|",
        ]
    )
    for m, cnt in summary.get("methods_with_failures", {}).items():
        lines.append(f"| `{m}` | {cnt} |")

    lines.extend(
        [
            "",
            "## Compression Ratio Distribution (passing only)",
            "",
            f"- Count: {summary['ratio_distribution']['count']}",
            f"- Min: {summary['ratio_distribution']['min']:.2f}x",
            f"- Max: {summary['ratio_distribution']['max']:.2f}x",
            f"- Mean: {summary['ratio_distribution']['mean']:.2f}x",
            f"- Median: {summary['ratio_distribution']['median']:.2f}x",
            f"- P25: {summary['ratio_distribution']['p25']:.2f}x",
            f"- P75: {summary['ratio_distribution']['p75']:.2f}x",
        ]
    )

    MD_REPORT.write_text("\n".join(lines))
    return lines


def _generate_html_report(results: List[Dict[str, Any]]):
    summary = _compute_summary(results)

    # Build best-per-category table rows
    best_rows = ""
    for cat, ranked in sorted(summary["best_per_tensor_category"].items()):
        top5 = ranked[:5]
        if top5:
            best_rows += f"<tr><td rowspan='{len(top5)}'><b>{cat}</b></td>"
            for j, entry in enumerate(top5):
                if j > 0:
                    best_rows += "<tr>"
                best_rows += (
                    f"<td><code>{entry['method']}</code></td>"
                    f"<td>{entry['avg_ratio']:.2f}x</td>"
                    f"<td>{entry['avg_error']:.6f}</td>"
                    f"<td>{entry['avg_snr']:.1f}</td></tr>"
                )

    # Overall ranking rows
    ranking_rows = ""
    for i, entry in enumerate(summary["overall_ranking"][:20]):
        color = (
            "#d4edda"
            if entry["pass_rate"] > 0.9
            else "#fff3cd"
            if entry["pass_rate"] > 0.5
            else "#f8d7da"
        )
        ranking_rows += (
            f"<tr style='background:{color}'>"
            f"<td>{i + 1}</td>"
            f"<td><code>{entry['method']}</code></td>"
            f"<td>{entry['avg_ratio']:.2f}x</td>"
            f"<td>{entry['avg_error']:.6f}</td>"
            f"<td>{entry['avg_snr']:.1f}</td>"
            f"<td>{entry['pass_rate']:.0%}</td>"
            f"<td>{entry['n_tensors']}</td></tr>"
        )

    # Failure rows
    fail_rows = ""
    for m, cnt in summary.get("methods_with_failures", {}).items():
        fail_rows += f"<tr><td><code>{m}</code></td><td>{cnt}</td></tr>"

    # Full passing list
    full_pass_html = (
        "<br>".join(
            f"<code>{m}</code>" for m in summary.get("fully_passing_methods", [])
        )
        or "<em>(none)</em>"
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Gemma 4 Method Validation Report</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 2em; background: #f8f9fa; color: #333; }}
h1, h2, h3 {{ color: #1a1a2e; }}
table {{ border-collapse: collapse; width: 100%; margin: 1em 0; background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #dee2e6; }}
th {{ background: #1a1a2e; color: white; }}
tr:hover {{ background: #f1f3f5; }}
.summary-card {{ display: inline-block; background: white; border-radius: 8px; padding: 1.5em; margin: 1em; box-shadow: 0 2px 4px rgba(0,0,0,0.1); min-width: 180px; text-align: center; }}
.summary-card .number {{ font-size: 2.5em; font-weight: bold; color: #1a1a2e; }}
.summary-card .label {{ font-size: 0.9em; color: #6c757d; }}
.ok {{ color: #28a745; }}
.err {{ color: #dc3545; }}
.container {{ max-width: 1200px; margin: 0 auto; }}
</style>
</head>
<body>
<div class="container">
<h1>🔬 Gemma 4 Method Validation Report</h1>
<p><strong>Generated:</strong> {datetime.utcnow().isoformat()}<br>
<strong>Model:</strong> <code>{MODEL_PATH}</code></p>

<h2>Summary</h2>
<div>
<div class="summary-card"><div class="number">{summary["methods_count"]}</div><div class="label">Methods</div></div>
<div class="summary-card"><div class="number">{summary["tensors_count"]}</div><div class="label">Tensors</div></div>
<div class="summary-card"><div class="number">{summary["ok_combinations"]}</div><div class="label">OK</div></div>
<div class="summary-card"><div class="number err">{summary["error_combinations"]}</div><div class="label">Errors</div></div>
<div class="summary-card"><div class="number ok">{summary["pass_combinations"]}</div><div class="label">Passing (<1% error)</div></div>
<div class="summary-card"><div class="number">{summary["fully_passing_count"]}</div><div class="label">Fully Passing Methods</div></div>
</div>

<h3>Fully Passing Methods</h3>
<p>{full_pass_html}</p>

<h2>Best Methods Per Tensor Category (Top 5 by ratio, &lt;1% error)</h2>
<table>
<tr><th>Category</th><th>Method</th><th>Avg Ratio</th><th>Avg Error</th><th>Avg SNR (dB)</th></tr>
{best_rows}
</table>

<h2>Overall Ranking (Top 20)</h2>
<table>
<tr><th>Rank</th><th>Method</th><th>Avg Ratio</th><th>Avg Error</th><th>Avg SNR</th><th>Pass Rate</th><th>Tensors</th></tr>
{ranking_rows}
</table>

<h2>Methods With Failures</h2>
<table>
<tr><th>Method</th><th>Failure Count</th></tr>
{fail_rows or '<tr><td colspan="2"><em>None</em></td></tr>'}
</table>

<h2>Compression Ratio Distribution (passing only)</h2>
<table>
<tr><th>Stat</th><th>Value</th></tr>
<tr><td>Count</td><td>{summary["ratio_distribution"]["count"]}</td></tr>
<tr><td>Min</td><td>{summary["ratio_distribution"]["min"]:.2f}x</td></tr>
<tr><td>Max</td><td>{summary["ratio_distribution"]["max"]:.2f}x</td></tr>
<tr><td>Mean</td><td>{summary["ratio_distribution"]["mean"]:.2f}x</td></tr>
<tr><td>Median</td><td>{summary["ratio_distribution"]["median"]:.2f}x</td></tr>
<tr><td>P25</td><td>{summary["ratio_distribution"]["p25"]:.2f}x</td></tr>
<tr><td>P75</td><td>{summary["ratio_distribution"]["p75"]:.2f}x</td></tr>
</table>
</div>
</body>
</html>"""

    HTML_REPORT.write_text(html)
    return html


# ═══════════════════════════════════════════════════════════════════════════
# Main entry point (also runnable standalone)
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "--timeout=300"])
