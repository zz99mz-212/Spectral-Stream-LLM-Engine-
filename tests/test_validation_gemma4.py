"""
Gemma 4 E2B Compression Validation Suite
=========================================
Tests every registered compression method on real Gemma 4 E2B weights.
Generates JSON, HTML, and Markdown reports with full quality metrics.
"""

from __future__ import annotations

import base64
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

HAVE_TORCH = False
try:
    import torch

    HAVE_TORCH = True
except ImportError:
    pass

HAVE_SAFETENSORS = False
try:
    from safetensors import safe_open

    HAVE_SAFETENSORS = True
except ImportError:
    pass

from spectralstream.compression.engine.method_discovery import MethodDiscovery

MODEL_PATH = "models/gemma-4-E2B/model.safetensors"
OUTPUT_DIR = Path("tests/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Representative tensor selection ─────────────────────────────────────────

TENSOR_SELECTION_SPEC = [
    ("embedding", "model.language_model.embed_tokens.weight", "embed_tokens"),
    ("embedding", "model.embed_audio.embedding_projection.weight", "embed_audio"),
    ("embedding", "model.embed_vision.embedding_projection.weight", "embed_vision"),
    (
        "attention_q",
        "model.language_model.layers.0.self_attn.q_proj.weight",
        "lm_attn_q_0",
    ),
    (
        "attention_k",
        "model.language_model.layers.0.self_attn.k_proj.weight",
        "lm_attn_k_0",
    ),
    (
        "attention_v",
        "model.language_model.layers.0.self_attn.v_proj.weight",
        "lm_attn_v_0",
    ),
    (
        "attention_out",
        "model.language_model.layers.0.self_attn.o_proj.weight",
        "lm_attn_o_0",
    ),
    (
        "attention_q",
        "model.language_model.layers.4.self_attn.q_proj.weight",
        "lm_attn_q_4_full",
    ),
    (
        "attention_out",
        "model.language_model.layers.4.self_attn.o_proj.weight",
        "lm_attn_o_4_full",
    ),
    ("ffn_gate", "model.language_model.layers.0.mlp.gate_proj.weight", "lm_ffn_gate_0"),
    ("ffn_up", "model.language_model.layers.0.mlp.up_proj.weight", "lm_ffn_up_0"),
    ("ffn_down", "model.language_model.layers.0.mlp.down_proj.weight", "lm_ffn_down_0"),
    (
        "ffn_gate",
        "model.language_model.layers.15.mlp.gate_proj.weight",
        "lm_ffn_gate_15_wide",
    ),
    (
        "ffn_up",
        "model.language_model.layers.15.mlp.up_proj.weight",
        "lm_ffn_up_15_wide",
    ),
    (
        "ffn_down",
        "model.language_model.layers.15.mlp.down_proj.weight",
        "lm_ffn_down_15_wide",
    ),
    (
        "per_layer_proj",
        "model.language_model.layers.0.per_layer_projection.weight",
        "lm_per_layer_0",
    ),
    (
        "per_layer_proj",
        "model.language_model.per_layer_model_projection.weight",
        "lm_per_layer_global",
    ),
    (
        "attention_q",
        "model.vision_tower.encoder.layers.0.self_attn.q_proj.linear.weight",
        "vis_attn_q_0",
    ),
    (
        "ffn_gate",
        "model.vision_tower.encoder.layers.0.mlp.gate_proj.linear.weight",
        "vis_ffn_gate_0",
    ),
    (
        "attention_out",
        "model.vision_tower.encoder.layers.0.self_attn.o_proj.linear.weight",
        "vis_attn_o_0",
    ),
    (
        "attention_q",
        "model.audio_tower.layers.0.self_attn.q_proj.linear.weight",
        "aud_attn_q_0",
    ),
    (
        "ffn_gate",
        "model.audio_tower.layers.0.feed_forward1.ffw_layer_1.linear.weight",
        "aud_ffn_0",
    ),
    ("norm", "model.language_model.layers.0.pre_layer_norm.weight", "lm_norm_0"),
    ("norm", "model.vision_tower.encoder.layers.0.pre_layer_norm.weight", "vis_norm_0"),
]


@dataclass
class TensorSample:
    name: str
    tensor_type: str
    key: str
    short_name: str
    original_shape: Tuple[int, ...]
    data: np.ndarray
    is_sampled: bool = False


def _load_gemma4_tensors(max_elements: int = 30000) -> List[TensorSample]:
    """Load representative tensors from Gemma 4, with sampling for large tensors."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not HAVE_SAFETENSORS:
        raise ImportError("safetensors not installed")
    if not HAVE_TORCH:
        raise ImportError("torch not installed")

    samples: List[TensorSample] = []
    with safe_open(MODEL_PATH, framework="pt") as f:
        for ttype, key, short_name in TENSOR_SELECTION_SPEC:
            if key not in f.keys():
                continue
            t = f.get_tensor(key)
            arr = t.cpu().to(torch.float32).numpy().astype(np.float32)
            orig_shape = arr.shape
            total_el = arr.size
            is_sampled = False
            if total_el > max_elements:
                is_sampled = True
                first = arr.ravel()[:10000]
                last = arr.ravel()[-10000:]
                rng = np.random.RandomState(42)
                idx = rng.choice(
                    total_el - 20000, min(10000, total_el - 20000), replace=False
                )
                middle = arr.ravel()[20000:][idx]
                arr = np.concatenate([first, last, middle]).astype(np.float32)
            samples.append(
                TensorSample(
                    name=key,
                    tensor_type=ttype,
                    key=key,
                    short_name=short_name,
                    original_shape=orig_shape,
                    data=arr,
                    is_sampled=is_sampled,
                )
            )
    return samples


# ── Metrics ─────────────────────────────────────────────────────────────────


def compute_metrics(
    original: np.ndarray, reconstructed: np.ndarray
) -> Dict[str, float]:
    orig = original.ravel().astype(np.float64)
    recon = reconstructed.ravel().astype(np.float64)
    diff = orig - recon
    mse = float(np.mean(diff**2))
    orig_norm = float(np.linalg.norm(orig))
    diff_norm = float(np.linalg.norm(diff))
    relative_error = diff_norm / max(orig_norm, 1e-30)
    snr = float(10 * np.log10(max(np.var(orig), 1e-30) / max(mse, 1e-30)))
    psnr = float(
        10
        * np.log10(
            max(float(np.max(orig) - np.min(orig)) ** 2, 1e-30) / max(mse, 1e-30)
        )
    )
    cos_sim = float(
        np.dot(orig, recon) / max(np.linalg.norm(orig) * np.linalg.norm(recon), 1e-30)
    )
    return {
        "relative_error": relative_error,
        "mse": mse,
        "snr_db": snr,
        "psnr_db": psnr,
        "cosine_similarity": cos_sim,
    }


def harmonic_mean_quality(ratio: float, error: float, alpha: float = 0.5) -> float:
    err_contrib = max(1.0 - error, 0.01)
    ratio_contrib = min(ratio / 10.0, 100.0)
    h = 2.0 * ratio_contrib * err_contrib / max(ratio_contrib + err_contrib, 1e-30)
    return float(h)


@dataclass
class MethodResult:
    method_name: str
    tensor_short_name: str
    tensor_type: str
    original_shape: Tuple[int, ...]
    compressed_bytes: int
    original_bytes: int
    ratio: float
    relative_error: float
    mse: float
    snr_db: float
    psnr_db: float
    cosine_similarity: float
    compression_time_ms: float
    decompression_time_ms: float
    success: bool
    error: str = ""
    params: Dict[str, Any] = field(default_factory=dict)


def _safe_compress_decompress(
    inst, tensor: np.ndarray, timeout: float = 300.0
) -> Tuple[bool, bytes, dict, float, float, str]:
    """Compress and decompress with timing."""
    import signal

    class TimeoutError(Exception):
        pass

    def _handler(signum, frame):
        raise TimeoutError("Operation timed out")

    signal.signal(signal.SIGALRM, _handler)
    signal.alarm(int(timeout) + 1)

    try:
        t0 = time.perf_counter()
        data, meta = inst.compress(tensor)
        t1 = time.perf_counter()
        comp_time = (t1 - t0) * 1000.0
        signal.alarm(0)
    except Exception as e:
        signal.alarm(0)
        return False, b"", {}, 0.0, 0.0, f"compress failed: {type(e).__name__}: {e}"

    try:
        signal.alarm(int(timeout) + 1)
        t2 = time.perf_counter()
        recon = inst.decompress(data, meta)
        t3 = time.perf_counter()
        decomp_time = (t3 - t2) * 1000.0
        signal.alarm(0)
    except Exception as e:
        signal.alarm(0)
        return (
            False,
            data,
            meta,
            comp_time,
            0.0,
            f"decompress failed: {type(e).__name__}: {e}",
        )

    if not isinstance(recon, np.ndarray):
        try:
            recon = np.array(recon)
        except Exception:
            return (
                False,
                data,
                meta,
                comp_time,
                decomp_time,
                f"decompress returned {type(recon).__name__}, not np.ndarray",
            )

    if recon.shape != tensor.shape:
        try:
            recon = recon.reshape(tensor.shape)
        except Exception:
            return (
                False,
                data,
                meta,
                comp_time,
                decomp_time,
                f"shape mismatch: {recon.shape} vs {tensor.shape}",
            )

    return True, data, meta, comp_time, decomp_time, ""


def _extract_params(meta: dict) -> Dict[str, Any]:
    params = {}
    for key in (
        "block_size",
        "rank",
        "threshold",
        "n_components",
        "max_rank",
        "num_blocks",
        "sparsity",
        "quant_bits",
        "codebook_size",
        "num_centroids",
    ):
        if key in meta:
            params[key] = meta[key]
    return params


# ── Report Generation ───────────────────────────────────────────────────────


def _error_class(err: float) -> str:
    if err < 0.001:
        return "green"
    if err < 0.01:
        return "yellow"
    return "red"


def _error_class_hex(err: float) -> str:
    if err < 0.001:
        return "#00cc66"
    if err < 0.01:
        return "#ffcc00"
    return "#ff4444"


def generate_json_report(all_results: List[MethodResult]) -> str:
    path = OUTPUT_DIR / "gemma4_validation_results.json"
    report = {
        "timestamp": datetime.now().isoformat(),
        "model": "Gemma 4 E2B",
        "model_path": MODEL_PATH,
        "total_tests": len(all_results),
        "successful": sum(1 for r in all_results if r.success),
        "failed": sum(1 for r in all_results if not r.success),
        "results": [
            {
                "method": r.method_name,
                "tensor": r.tensor_short_name,
                "tensor_type": r.tensor_type,
                "shape": list(r.original_shape),
                "compressed_bytes": r.compressed_bytes,
                "original_bytes": r.original_bytes,
                "ratio": r.ratio,
                "relative_error": r.relative_error,
                "mse": r.mse,
                "snr_db": r.snr_db,
                "psnr_db": r.psnr_db,
                "cosine_similarity": r.cosine_similarity,
                "compression_time_ms": r.compression_time_ms,
                "decompression_time_ms": r.decompression_time_ms,
                "success": r.success,
                "error": r.error,
                "params": r.params,
            }
            for r in all_results
        ],
    }
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    return str(path)


def generate_html_report(all_results: List[MethodResult]) -> str:
    path = OUTPUT_DIR / "gemma4_validation_results.html"

    methods = sorted(set(r.method_name for r in all_results))
    tensors = sorted(set(r.tensor_short_name for r in all_results))
    tensor_types = {r.tensor_short_name: r.tensor_type for r in all_results}

    # Build table rows
    table_rows = ""
    for m in methods:
        row = f"<tr><td class='method-cell'>{m}</td>"
        for t in tensors:
            matches = [
                r
                for r in all_results
                if r.method_name == m and r.tensor_short_name == t
            ]
            if not matches:
                row += "<td class='na'>N/A</td>"
            else:
                r = matches[0]
                if not r.success:
                    row += f"<td class='fail' title='{r.error}'>FAIL</td>"
                else:
                    cls = _error_class(r.relative_error)
                    ratio_str = f"{r.ratio:.1f}x"
                    err_str = f"{r.relative_error * 100:.2f}%"
                    row += f"<td class='{cls}' title='ratio={ratio_str}, SNR={r.snr_db:.1f}dB'>{err_str}<br><small>{ratio_str}</small></td>"
        table_rows += row + "</tr>\n"

    # Best method per tensor type
    tensor_type_results: Dict[str, List[MethodResult]] = {}
    for r in all_results:
        if r.success:
            tensor_type_results.setdefault(r.tensor_type, []).append(r)

    best_per_type = ""
    for ttype, results in sorted(tensor_type_results.items()):
        best = min(results, key=lambda x: x.relative_error)
        best_ratio = max(results, key=lambda x: x.ratio)
        best_harmonic = max(
            results, key=lambda x: harmonic_mean_quality(x.ratio, x.relative_error)
        )
        best_per_type += f"<tr><td>{ttype}</td><td>{best.method_name} ({best.relative_error * 100:.3f}%)</td><td>{best_ratio.method_name} ({best_ratio.ratio:.1f}x)</td><td>{best_harmonic.method_name}</td></tr>\n"

    # Ranking by harmonic mean
    method_avg: Dict[str, List[float]] = {}
    for r in all_results:
        if r.success:
            method_avg.setdefault(r.method_name, []).append(
                harmonic_mean_quality(r.ratio, r.relative_error)
            )
    ranking = sorted(method_avg.items(), key=lambda x: np.mean(x[1]), reverse=True)
    ranking_rows = ""
    for i, (m, scores) in enumerate(ranking[:20], 1):
        avg_h = np.mean(scores)
        ranking_rows += f"<tr><td>{i}</td><td>{m}</td><td>{avg_h:.2f}</td></tr>\n"

    # Summary stats
    total = len(all_results)
    success = sum(1 for r in all_results if r.success)
    failed = total - success
    avg_ratio = np.mean([r.ratio for r in all_results if r.success]) if success else 0
    avg_err = (
        np.mean([r.relative_error for r in all_results if r.success]) if success else 0
    )
    avg_snr = np.mean([r.snr_db for r in all_results if r.success]) if success else 0

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Gemma 4 E2B Compression Validation Report</title>
<style>
body {{ font-family: -apple-system,BlinkMacSystemFont,sans-serif; margin: 20px; background: #f5f5f5; }}
h1,h2,h3 {{ color: #333; }}
.summary {{ display: flex; gap: 20px; margin: 20px 0; }}
.card {{ background: white; border-radius: 8px; padding: 20px; flex:1; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
.card h3 {{ margin:0 0 10px 0; }}
.card .value {{ font-size: 24px; font-weight: bold;}}
table {{ border-collapse: collapse; margin: 10px 0; font-size: 12px; }}
th, td {{ border: 1px solid #ddd; padding: 4px 8px; text-align: center; }}
th {{ background: #4a90d9; color: white; position: sticky; top: 0; }}
tr:nth-child(even) {{ background: #f9f9f9; }}
.green {{ background: #d4edda; }}
.yellow {{ background: #fff3cd; }}
.red {{ background: #f8d7da; }}
.fail {{ background: #e8e8e8; color: #999; }}
.na {{ background: #f0f0f0; color: #ccc; }}
.method-cell {{ text-align: left; font-family: monospace; white-space: nowrap; }}
small {{ color: #666; }}
.section {{ background: white; border-radius: 8px; padding: 20px; margin: 20px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.1); overflow-x: auto; }}
</style>
</head>
<body>
<h1>Gemma 4 E2B — Compression Validation Report</h1>
<p>Generated: {datetime.now().isoformat()}</p>

<div class="summary">
<div class="card"><h3>Tests</h3><div class="value">{total}</div><span>{success} passed, {failed} failed</span></div>
<div class="card"><h3>Avg Ratio</h3><div class="value">{avg_ratio:.1f}x</div></div>
<div class="card"><h3>Avg Error</h3><div class="value">{avg_err * 100:.2f}%</div></div>
<div class="card"><h3>Avg SNR</h3><div class="value">{avg_snr:.1f} dB</div></div>
</div>

<div class="section">
<h2>Method Ranking (Top 20 by Harmonic Mean)</h2>
<table><tr><th>Rank</th><th>Method</th><th>Avg Harmonic Score</th></tr>
{ranking_rows}</table>
</div>

<div class="section">
<h2>Best Method Per Tensor Type</h2>
<table><tr><th>Type</th><th>Lowest Error</th><th>Highest Ratio</th><th>Best Harmonic</th></tr>
{best_per_type}</table>
</div>

<div class="section">
<h2>Error-Ratio Matrix</h2>
<p>Color: <span class="green">&nbsp;green&nbsp;</span> error &lt; 0.1%, <span class="yellow">&nbsp;yellow&nbsp;</span> &lt; 1%, <span class="red">&nbsp;red&nbsp;</span> &gt; 1%</p>
<table>
<tr><th>Method</th>{"".join(f'<th title="{tensor_types[t]}">{t}</th>' for t in tensors)}</tr>
{table_rows}</table>
</div>

<div class="section">
<h2>Failure Summary</h2>
<table><tr><th>Method</th><th>Tensor</th><th>Error</th></tr>
{"".join(f"<tr><td>{r.method_name}</td><td>{r.tensor_short_name}</td><td>{r.error}</td></tr>" for r in all_results if not r.success)}
</table>
</div>

</body>
</html>"""
    with open(path, "w") as f:
        f.write(html)
    return str(path)


def generate_markdown_report(all_results: List[MethodResult]) -> str:
    path = OUTPUT_DIR / "gemma4_validation_results.md"

    success_results = [r for r in all_results if r.success]
    failed_results = [r for r in all_results if not r.success]

    lines = [
        "# Gemma 4 E2B — Compression Validation Report",
        f"**Generated:** {datetime.now().isoformat()}",
        f"**Model:** Gemma 4 E2B",
        "",
        "## Summary",
        f"| Metric | Value |",
        "|--------|-------|",
        f"| Total tests | {len(all_results)} |",
        f"| Successful | {len(success_results)} |",
        f"| Failed | {len(failed_results)} |",
        f"| Avg compression ratio | {np.mean([r.ratio for r in success_results]):.1f}x"
        if success_results
        else "0",
        f"| Avg relative error | {np.mean([r.relative_error for r in success_results]) * 100:.2f}%"
        if success_results
        else "0",
        f"| Avg SNR | {np.mean([r.snr_db for r in success_results]):.1f} dB"
        if success_results
        else "0",
        "",
        "## Method Ranking (Top 20 by Harmonic Mean)",
        "| Rank | Method | Avg Harmonic Score |",
        "|------|--------|-------------------|",
    ]

    method_avg = {}
    for r in success_results:
        method_avg.setdefault(r.method_name, []).append(
            harmonic_mean_quality(r.ratio, r.relative_error)
        )
    ranking = sorted(method_avg.items(), key=lambda x: np.mean(x[1]), reverse=True)
    for i, (m, scores) in enumerate(ranking[:20], 1):
        lines.append(f"| {i} | {m} | {np.mean(scores):.2f} |")

    lines.extend(
        [
            "",
            "## Best Method Per Tensor Type",
            "| Type | Lowest Error | Highest Ratio | Best Harmonic |",
        ]
    )
    tensor_type_results = {}
    for r in success_results:
        tensor_type_results.setdefault(r.tensor_type, []).append(r)
    for ttype, results in sorted(tensor_type_results.items()):
        best_err = min(results, key=lambda x: x.relative_error)
        best_ratio = max(results, key=lambda x: x.ratio)
        best_harm = max(
            results, key=lambda x: harmonic_mean_quality(x.ratio, x.relative_error)
        )
        lines.append(
            f"| {ttype} | {best_err.method_name} ({best_err.relative_error * 100:.3f}%) | {best_ratio.method_name} ({best_ratio.ratio:.1f}x) | {best_harm.method_name} |"
        )

    lines.extend(["", "## All Results", ""])
    lines.append(
        "| Method | Tensor | Type | Ratio | Error(%) | SNR(dB) | PSNR(dB) | CosineSim | Time(ms) |"
    )
    lines.append(
        "|--------|--------|------|-------|----------|---------|----------|-----------|----------|"
    )
    for r in sorted(all_results, key=lambda x: (x.method_name, x.tensor_short_name)):
        if r.success:
            lines.append(
                f"| {r.method_name} | {r.tensor_short_name} | {r.tensor_type} | {r.ratio:.1f}x | {r.relative_error * 100:.2f} | {r.snr_db:.1f} | {r.psnr_db:.1f} | {r.cosine_similarity:.4f} | {r.compression_time_ms:.1f} |"
            )
        else:
            lines.append(
                f"| {r.method_name} | {r.tensor_short_name} | {r.tensor_type} | FAIL | {r.error} |"
            )

    if failed_results:
        lines.extend(["", "## Failures", "", "| Method | Tensor | Error |"])
        lines.append("|--------|--------|-------|")
        for r in failed_results:
            lines.append(f"| {r.method_name} | {r.tensor_short_name} | {r.error} |")

    with open(path, "w") as f:
        f.write("\n".join(lines))
    return str(path)


def generate_tuning_report(all_results: List[MethodResult]) -> str:
    path = OUTPUT_DIR / "gemma4_method_tuning.json"

    tuning: Dict[str, Dict[str, Any]] = {}
    for r in all_results:
        if r.success:
            key = f"{r.method_name}__{r.tensor_type}"
            if key not in tuning or r.relative_error < tuning[key].get(
                "best_error", 1.0
            ):
                tuning[key] = {
                    "method": r.method_name,
                    "tensor_type": r.tensor_type,
                    "tensor": r.tensor_short_name,
                    "best_error": r.relative_error,
                    "best_ratio": r.ratio,
                    "params": r.params,
                    "snr_db": r.snr_db,
                }

    grouped: Dict[str, Dict[str, Any]] = {}
    for key, info in tuning.items():
        method = info["method"]
        if method not in grouped:
            grouped[method] = {}
        grouped[method][info["tensor_type"]] = {
            "tensor": info["tensor"],
            "best_error": info["best_error"],
            "best_ratio": info["best_ratio"],
            "snr_db": info["snr_db"],
            "recommended_params": info["params"],
        }

    with open(path, "w") as f:
        json.dump(grouped, f, indent=2, default=str)
    return str(path)


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def gemma4_tensors():
    return _load_gemma4_tensors()


@pytest.fixture(scope="session")
def all_methods():
    return MethodDiscovery.discover()


# ── Tests ───────────────────────────────────────────────────────────────────

# We collect results globally for report generation after all tests
GLOBAL_RESULTS: List[MethodResult] = []


def pytest_sessionfinish(session, exitstatus):
    if GLOBAL_RESULTS:
        json_path = generate_json_report(GLOBAL_RESULTS)
        html_path = generate_html_report(GLOBAL_RESULTS)
        md_path = generate_markdown_report(GLOBAL_RESULTS)
        tune_path = generate_tuning_report(GLOBAL_RESULTS)
        success = sum(1 for r in GLOBAL_RESULTS if r.success)
        total = len(GLOBAL_RESULTS)
        print(f"\n\n{'=' * 60}")
        print(f"  Gemma 4 E2B Validation Complete: {success}/{total} passed")
        print(f"  JSON:   {json_path}")
        print(f"  HTML:   {html_path}")
        print(f"  MD:     {md_path}")
        print(f"  Tuning: {tune_path}")
        print(f"{'=' * 60}")


def pytest_generate_tests(metafunc):
    if (
        "method_name" in metafunc.fixturenames
        and "tensor_sample" in metafunc.fixturenames
    ):
        methods = MethodDiscovery.discover()
        try:
            tensors = _load_gemma4_tensors()
        except (FileNotFoundError, ImportError) as e:
            tensors = []
        params = [(mname, ts) for mname in methods for ts in tensors]
        ids = [f"{m}__{t.short_name}" for m, t in params]
        metafunc.parametrize("method_name,tensor_sample", params, ids=ids)


@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="Gemma 4 not available")
class TestGemma4Validation:
    @pytest.mark.timeout(300)
    def test_method_on_tensor(self, method_name: str, tensor_sample: TensorSample):
        methods = MethodDiscovery.discover()
        if method_name not in methods:
            pytest.skip(f"Method {method_name} not available")

        info = methods[method_name]
        inst = info.get("instance")
        if inst is None:
            GLOBAL_RESULTS.append(
                MethodResult(
                    method_name=method_name,
                    tensor_short_name=tensor_sample.short_name,
                    tensor_type=tensor_sample.tensor_type,
                    original_shape=tensor_sample.original_shape,
                    compressed_bytes=0,
                    original_bytes=tensor_sample.data.nbytes,
                    ratio=0.0,
                    relative_error=1.0,
                    mse=0.0,
                    snr_db=0.0,
                    psnr_db=0.0,
                    cosine_similarity=0.0,
                    compression_time_ms=0.0,
                    decompression_time_ms=0.0,
                    success=False,
                    error="No instance available",
                )
            )
            pytest.skip("No instance")

        success, data, meta, comp_time, decomp_time, err_msg = (
            _safe_compress_decompress(inst, tensor_sample.data)
        )

        if not success:
            GLOBAL_RESULTS.append(
                MethodResult(
                    method_name=method_name,
                    tensor_short_name=tensor_sample.short_name,
                    tensor_type=tensor_sample.tensor_type,
                    original_shape=tensor_sample.original_shape,
                    compressed_bytes=len(data) if data else 0,
                    original_bytes=tensor_sample.data.nbytes,
                    ratio=0.0,
                    relative_error=1.0,
                    mse=0.0,
                    snr_db=0.0,
                    psnr_db=0.0,
                    cosine_similarity=0.0,
                    compression_time_ms=comp_time,
                    decompression_time_ms=decomp_time,
                    success=False,
                    error=err_msg,
                )
            )
            pytest.fail(err_msg)

        # Verify compress/decompress signatures (some methods return dict, not bytes)
        if not isinstance(data, (bytes, dict, list, tuple, np.ndarray)):
            result = MethodResult(
                method_name=method_name,
                tensor_short_name=tensor_sample.short_name,
                tensor_type=tensor_sample.tensor_type,
                original_shape=tensor_sample.original_shape,
                compressed_bytes=0,
                original_bytes=tensor_sample.data.nbytes,
                ratio=0.0,
                relative_error=1.0,
                mse=0.0,
                snr_db=0.0,
                psnr_db=0.0,
                cosine_similarity=0.0,
                compression_time_ms=comp_time,
                decompression_time_ms=decomp_time,
                success=False,
                error=f"Unexpected compress return type: {type(data)}",
            )
            GLOBAL_RESULTS.append(result)
            pytest.fail(f"Unexpected compress return type: {type(data)}")

        if not isinstance(meta, dict):
            result = MethodResult(
                method_name=method_name,
                tensor_short_name=tensor_sample.short_name,
                tensor_type=tensor_sample.tensor_type,
                original_shape=tensor_sample.original_shape,
                compressed_bytes=0,
                original_bytes=tensor_sample.data.nbytes,
                ratio=0.0,
                relative_error=1.0,
                mse=0.0,
                snr_db=0.0,
                psnr_db=0.0,
                cosine_similarity=0.0,
                compression_time_ms=comp_time,
                decompression_time_ms=decomp_time,
                success=False,
                error=f"compress meta must be dict, got {type(meta)}",
            )
            GLOBAL_RESULTS.append(result)
            pytest.fail(f"compress meta must be dict, got {type(meta)}")

        # Calculate compressed size
        if isinstance(data, bytes):
            compressed_bytes = len(data)
        elif isinstance(data, dict):
            compressed_bytes = sum(
                v.nbytes if isinstance(v, np.ndarray) else len(str(v))
                for v in data.values()
            )
        else:
            compressed_bytes = len(str(data))

        # Decompress and get reconstructed
        try:
            recon = inst.decompress(data, meta)
        except Exception as e:
            result = MethodResult(
                method_name=method_name,
                tensor_short_name=tensor_sample.short_name,
                tensor_type=tensor_sample.tensor_type,
                original_shape=tensor_sample.original_shape,
                compressed_bytes=compressed_bytes,
                original_bytes=tensor_sample.data.nbytes,
                ratio=0.0,
                relative_error=1.0,
                mse=0.0,
                snr_db=0.0,
                psnr_db=0.0,
                cosine_similarity=0.0,
                compression_time_ms=comp_time,
                decompression_time_ms=0.0,
                success=False,
                error=f"decompress failed: {type(e).__name__}: {e}",
            )
            GLOBAL_RESULTS.append(result)
            pytest.fail(f"decompress failed: {type(e).__name__}: {e}")
        if not isinstance(recon, np.ndarray):
            recon = np.array(recon)
        if recon.shape != tensor_sample.data.shape:
            try:
                recon = recon.reshape(tensor_sample.data.shape)
            except Exception as e:
                result = MethodResult(
                    method_name=method_name,
                    tensor_short_name=tensor_sample.short_name,
                    tensor_type=tensor_sample.tensor_type,
                    original_shape=tensor_sample.original_shape,
                    compressed_bytes=compressed_bytes,
                    original_bytes=tensor_sample.data.nbytes,
                    ratio=0.0,
                    relative_error=1.0,
                    mse=0.0,
                    snr_db=0.0,
                    psnr_db=0.0,
                    cosine_similarity=0.0,
                    compression_time_ms=comp_time,
                    decompression_time_ms=decomp_time,
                    success=False,
                    error=f"shape mismatch: {recon.shape} vs {tensor_sample.data.shape}",
                )
                GLOBAL_RESULTS.append(result)
                pytest.fail(
                    f"shape mismatch: {recon.shape} vs {tensor_sample.data.shape}"
                )

        metrics = compute_metrics(tensor_sample.data, recon)
        ratio = max(tensor_sample.data.nbytes / max(len(data), 1), 1.0)
        params = _extract_params(meta)

        result = MethodResult(
            method_name=method_name,
            tensor_short_name=tensor_sample.short_name,
            tensor_type=tensor_sample.tensor_type,
            original_shape=tensor_sample.original_shape,
            compressed_bytes=len(data),
            original_bytes=tensor_sample.data.nbytes,
            ratio=ratio,
            relative_error=metrics["relative_error"],
            mse=metrics["mse"],
            snr_db=metrics["snr_db"],
            psnr_db=metrics["psnr_db"],
            cosine_similarity=metrics["cosine_similarity"],
            compression_time_ms=comp_time,
            decompression_time_ms=decomp_time,
            success=True,
            params=params,
        )
        GLOBAL_RESULTS.append(result)

        # Log summary per method+tensor
        print(
            f"  [{method_name}] {tensor_sample.short_name}: "
            f"ratio={ratio:.1f}x, err={metrics['relative_error'] * 100:.2f}%, "
            f"SNR={metrics['snr_db']:.1f}dB, "
            f"comp={comp_time:.1f}ms, decomp={decomp_time:.1f}ms"
        )


# ── Direct runner for manual execution ──────────────────────────────────────


def run_validation(limit_methods: Optional[int] = None) -> List[MethodResult]:
    """Run validation directly without pytest parametrization."""
    print("Loading Gemma 4 tensors...")
    tensors = _load_gemma4_tensors()
    print(f"Loaded {len(tensors)} representative tensors")

    methods = MethodDiscovery.discover()
    mnames = (
        sorted(methods.keys())[:limit_methods]
        if limit_methods
        else sorted(methods.keys())
    )
    print(f"Testing {len(mnames)} methods...")

    results: List[MethodResult] = []
    for midx, mname in enumerate(mnames):
        info = methods[mname]
        inst = info.get("instance")
        if inst is None:
            for ts in tensors:
                results.append(
                    MethodResult(
                        method_name=mname,
                        tensor_short_name=ts.short_name,
                        tensor_type=ts.tensor_type,
                        original_shape=ts.original_shape,
                        compressed_bytes=0,
                        original_bytes=ts.data.nbytes,
                        ratio=0.0,
                        relative_error=1.0,
                        mse=0.0,
                        snr_db=0.0,
                        psnr_db=0.0,
                        cosine_similarity=0.0,
                        compression_time_ms=0.0,
                        decompression_time_ms=0.0,
                        success=False,
                        error="No instance",
                    )
                )
            continue

        for ts in tensors:
            success, data, meta, comp_time, decomp_time, err_msg = (
                _safe_compress_decompress(inst, ts.data)
            )
            if not success:
                results.append(
                    MethodResult(
                        method_name=mname,
                        tensor_short_name=ts.short_name,
                        tensor_type=ts.tensor_type,
                        original_shape=ts.original_shape,
                        compressed_bytes=len(data) if data else 0,
                        original_bytes=ts.data.nbytes,
                        ratio=0.0,
                        relative_error=1.0,
                        mse=0.0,
                        snr_db=0.0,
                        psnr_db=0.0,
                        cosine_similarity=0.0,
                        compression_time_ms=comp_time,
                        decompression_time_ms=decomp_time,
                        success=False,
                        error=err_msg,
                    )
                )
                continue

            recon = inst.decompress(data, meta)
            if not isinstance(recon, np.ndarray):
                recon = np.array(recon)
            if recon.shape != ts.data.shape:
                recon = recon.reshape(ts.data.shape)

            metrics = compute_metrics(ts.data, recon)
            ratio = max(ts.data.nbytes / max(len(data), 1), 1.0)
            params = _extract_params(meta)

            results.append(
                MethodResult(
                    method_name=mname,
                    tensor_short_name=ts.short_name,
                    tensor_type=ts.tensor_type,
                    original_shape=ts.original_shape,
                    compressed_bytes=len(data),
                    original_bytes=ts.data.nbytes,
                    ratio=ratio,
                    relative_error=metrics["relative_error"],
                    mse=metrics["mse"],
                    snr_db=metrics["snr_db"],
                    psnr_db=metrics["psnr_db"],
                    cosine_similarity=metrics["cosine_similarity"],
                    compression_time_ms=comp_time,
                    decompression_time_ms=decomp_time,
                    success=True,
                    params=params,
                )
            )
            print(
                f"  [{midx + 1}/{len(mnames)}] {mname} on {ts.short_name}: ratio={ratio:.1f}x, err={metrics['relative_error'] * 100:.2f}%"
            )

    # Generate reports
    json_path = generate_json_report(results)
    html_path = generate_html_report(results)
    md_path = generate_markdown_report(results)
    tune_path = generate_tuning_report(results)

    success = sum(1 for r in results if r.success)
    total = len(results)
    print(f"\n{'=' * 60}")
    print(f"  Validation Complete: {success}/{total} passed")
    print(f"  JSON:   {json_path}")
    print(f"  HTML:   {html_path}")
    print(f"  MD:     {md_path}")
    print(f"  Tuning: {tune_path}")
    print(f"{'=' * 60}")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Gemma 4 E2B Validation")
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of methods"
    )
    args = parser.parse_args()
    run_validation(limit_methods=args.limit)
