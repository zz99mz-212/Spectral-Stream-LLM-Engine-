#!/usr/bin/env python3
"""Benchmark compression performance across model sizes.

Usage:
    python scripts/benchmark_compression.py --output-dir /tmp/benchmark_results
"""

import argparse
import gc
import json
import logging
import os
import struct
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spectralstream.compression.engine import (
    CompressionConfig,
    CompressionIntelligenceEngine,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("benchmark")


def make_model(
    num_layers: int,
    vocab_size: int = 2048,
    d_model: int = 512,
    ff_dim: int = 2048,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Create structured low-rank synthetic model."""
    rng = np.random.RandomState(seed)
    tensors: Dict[str, np.ndarray] = {}
    n_heads = 8
    n_kv_heads = 2
    head_dim = d_model // n_heads

    def _low_rank(rows: int, cols: int, rank: int = 4) -> np.ndarray:
        base = rng.randn(rows, rank).astype(np.float32) @ rng.randn(rank, cols).astype(
            np.float32
        )
        return base * (0.02 / max(float(np.std(base)), 1e-10))

    tensors["embed_tokens.weight"] = _low_rank(vocab_size, d_model, rank=6)
    tensors["norm.weight"] = np.ones(d_model, dtype=np.float32) * 0.1

    for layer in range(num_layers):
        prefix = f"model.layers.{layer}"
        for name, shape, rank in [
            (f"{prefix}.attn.q_proj.weight", (d_model, d_model), 4),
            (f"{prefix}.attn.k_proj.weight", (d_model, n_kv_heads * head_dim), 4),
            (f"{prefix}.attn.v_proj.weight", (d_model, n_kv_heads * head_dim), 4),
            (f"{prefix}.attn.o_proj.weight", (d_model, d_model), 4),
            (f"{prefix}.feed_forward.gate_proj.weight", (d_model, ff_dim), 6),
            (f"{prefix}.feed_forward.up_proj.weight", (d_model, ff_dim), 6),
            (f"{prefix}.feed_forward.down_proj.weight", (ff_dim, d_model), 8),
            (f"{prefix}.input_layernorm.weight", (d_model,), 0),
            (f"{prefix}.post_attention_layernorm.weight", (d_model,), 0),
        ]:
            if rank > 0:
                tensors[name] = _low_rank(*shape, rank=rank)
            else:
                tensors[name] = rng.randn(shape[0]).astype(np.float32) * 0.01

    tensors["lm_head.weight"] = _low_rank(vocab_size, d_model, rank=6)
    tensors["final_norm.weight"] = rng.randn(d_model).astype(np.float32) * 0.1
    return tensors


def _human_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def _grade_error(err: float) -> str:
    if err < 0.0002:
        return "S"
    if err < 0.001:
        return "A"
    if err < 0.005:
        return "B"
    if err < 0.01:
        return "C"
    if err < 0.05:
        return "D"
    return "F"


def benchmark_model(
    tensors: Dict[str, np.ndarray],
    model_name: str,
    target_ratio: float = 5000.0,
    max_error: float = 0.01,
) -> Dict[str, Any]:
    """Run compression benchmark on a model and return metrics."""
    total_orig = sum(t.nbytes for t in tensors.values())
    n_tensors = len(tensors)

    logger.info(
        "Benchmarking %s: %d tensors, %s",
        model_name,
        n_tensors,
        _human_size(total_orig),
    )

    config = CompressionConfig(
        target_ratio=target_ratio,
        max_error=max_error,
        num_workers=1,
    )
    engine = CompressionIntelligenceEngine(config)

    import psutil

    mem_before = psutil.Process().memory_info().rss / (1024 * 1024)

    t_start = time.perf_counter()
    compressed_list = []
    total_comp_bytes = 0
    all_errors = []
    all_ratios = []
    method_counts: Dict[str, int] = {}
    n_failures = 0
    per_tensor = []

    for name, tensor in tensors.items():
        t0 = time.perf_counter()
        try:
            ct = engine.compress_fast(tensor, name=name)
            delta = time.perf_counter() - t0
            total_comp_bytes += len(ct.data)
            all_errors.append(ct.relative_error)
            all_ratios.append(ct.compression_ratio)
            method_counts[ct.method] = method_counts.get(ct.method, 0) + 1
            per_tensor.append(
                {
                    "name": name,
                    "method": ct.method,
                    "ratio": round(ct.compression_ratio, 2),
                    "error": round(ct.relative_error, 6),
                    "grade": _grade_error(ct.relative_error),
                    "time_sec": round(delta, 3),
                }
            )
            logger.debug(
                "  %s: %s ratio=%.1f err=%.4f time=%.2fs",
                name[-40:],
                ct.method,
                ct.compression_ratio,
                ct.relative_error,
                delta,
            )
        except Exception as e:
            n_failures += 1
            logger.error("  FAILED: %s: %s", name, e)

    elapsed = time.perf_counter() - t_start
    mem_after = psutil.Process().memory_info().rss / (1024 * 1024)
    mem_peak = engine._memory_peak_mb

    overall_ratio = total_orig / max(total_comp_bytes, 1)
    avg_error = float(np.mean(all_errors)) if all_errors else 0.0
    max_err = float(np.max(all_errors)) if all_errors else 0.0
    speed_bytes_per_sec = total_orig / max(elapsed, 0.001)

    return {
        "model_name": model_name,
        "n_layers": (n_tensors - 4) // 9 if n_tensors > 4 else 0,
        "n_tensors": n_tensors,
        "n_failures": n_failures,
        "original_bytes": total_orig,
        "original_size": _human_size(total_orig),
        "compressed_bytes": total_comp_bytes,
        "compressed_size": _human_size(total_comp_bytes),
        "overall_ratio": round(overall_ratio, 2),
        "avg_error": round(avg_error, 6),
        "avg_error_pct": round(avg_error * 100, 4),
        "max_error": round(max_err, 6),
        "max_error_pct": round(max_err * 100, 4),
        "time_seconds": round(elapsed, 2),
        "speed_mb_per_sec": round(speed_bytes_per_sec / 1e6, 2),
        "memory_peak_mb": round(mem_peak, 1),
        "memory_delta_mb": round(mem_after - mem_before, 1),
        "method_counts": method_counts,
        "per_tensor": per_tensor,
    }


def generate_report(results: List[Dict[str, Any]], output_dir: str):
    """Generate benchmark reports in JSON and Markdown formats."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = Path(output_dir) / f"benchmark_report_{ts}"
    report_dir.mkdir(parents=True, exist_ok=True)

    # JSON report
    json_path = report_dir / "benchmark_report.json"
    json_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    # Markdown report
    md_lines = [
        "# SpectralStream Compression Benchmark",
        f"**Timestamp:** {ts}",
        "",
        "## Summary Table",
        "",
        "| Model | Tensors | Original | Compressed | Ratio | Avg Error | Max Error | Time | Speed | Peak Memory |",
        "|-------|---------|----------|------------|-------|-----------|-----------|------|-------|-------------|",
    ]
    for r in results:
        md_lines.append(
            f"| {r['model_name']} | {r['n_tensors']} | {r['original_size']} | "
            f"{r['compressed_size']} | **{r['overall_ratio']}x** | "
            f"{r['avg_error_pct']}% | {r['max_error_pct']}% | "
            f"{r['time_seconds']}s | {r['speed_mb_per_sec']} MB/s | "
            f"{r['memory_peak_mb']} MB |"
        )

    md_lines.extend(
        [
            "",
            "## Detailed Per-Model Results",
            "",
        ]
    )
    for r in results:
        grade_dist: Dict[str, int] = {"S": 0, "A": 0, "B": 0, "C": 0, "D": 0, "F": 0}
        for t in r.get("per_tensor", []):
            g = t.get("grade", "F")
            grade_dist[g] = grade_dist.get(g, 0) + 1
        method_str = ", ".join(
            f"{m}: {c}" for m, c in sorted(r["method_counts"].items())
        )
        grade_str = ", ".join(
            f"{g}={c}" for g, c in sorted(grade_dist.items()) if c > 0
        )

        md_lines.extend(
            [
                f"### {r['model_name']}",
                f"- **Ratio:** {r['overall_ratio']}x",
                f"- **Avg Error:** {r['avg_error_pct']}%",
                f"- **Max Error:** {r['max_error_pct']}%",
                f"- **Time:** {r['time_seconds']}s @ {r['speed_mb_per_sec']} MB/s",
                f"- **Memory Peak:** {r['memory_peak_mb']} MB",
                f"- **Methods:** {method_str}",
                f"- **Grades:** {grade_str}",
                "",
                "#### Per-Tensor Breakdown",
                "",
                "| Tensor | Method | Ratio | Error | Grade | Time (s) |",
                "|--------|--------|-------|-------|-------|----------|",
            ]
        )
        for t in r.get("per_tensor", []):
            md_lines.append(
                f"| {t['name'][:50]} | {t['method']} | {t['ratio']}x | "
                f"{t['error'] * 100:.4f}% | {t['grade']} | {t['time_sec']} |"
            )
        md_lines.append("")

    md_path = report_dir / "benchmark_report.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    logger.info("Reports saved to %s", report_dir)
    return str(report_dir)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compression benchmark across model sizes"
    )
    parser.add_argument("--output-dir", default="/tmp/benchmark_results")
    parser.add_argument("--target-ratio", type=float, default=5000.0)
    parser.add_argument("--max-error", type=float, default=0.01)
    parser.add_argument("--layers", type=int, nargs="+", default=[2, 4, 8])
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for n_layers in args.layers:
        gc.collect()
        time.sleep(1)

        logger.info("=" * 60)
        logger.info("Benchmark: %d layers", n_layers)
        logger.info("=" * 60)

        try:
            tensors = make_model(num_layers=n_layers)
            model_name = f"{n_layers}-layer (vocab=2048, d_model=512, ff_dim=2048)"
            r = benchmark_model(
                tensors,
                model_name,
                target_ratio=args.target_ratio,
                max_error=args.max_error,
            )
            results.append(r)
            logger.info(
                "Result: ratio=%.1fx, error=%.4f%%, time=%.1fs, mem=%.0fMB",
                r["overall_ratio"],
                r["avg_error_pct"],
                r["time_seconds"],
                r["memory_peak_mb"],
            )
        except Exception as e:
            logger.error("Benchmark failed for %d layers: %s", n_layers, e)
            import traceback

            traceback.print_exc()

    if results:
        report_dir = generate_report(results, str(output_dir))
        logger.info("Benchmark complete. Report: %s", report_dir)

    # Print summary
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    print(
        f"{'Model':<25} {'Tensors':<8} {'Ratio':<10} {'Error%':<10} {'Time':<8} {'Mem(MB)':<10}"
    )
    print("-" * 70)
    for r in results:
        print(
            f"{r['model_name'][:24]:<25} {r['n_tensors']:<8} {r['overall_ratio']:<10.1f}x "
            f"{r['avg_error_pct']:<10.4f} {r['time_seconds']:<8.1f}s {r['memory_peak_mb']:<10.1f}"
        )
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
